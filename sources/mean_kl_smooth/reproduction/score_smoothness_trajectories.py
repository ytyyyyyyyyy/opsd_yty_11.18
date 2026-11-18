"""Resume exact-position, full-vocabulary scoring of frozen diagnostic rollouts.

Student prompts are the exact IDs used for generation.  The privileged teacher
uses the same frozen base with adapters disabled and the frozen OPSD collator's
full reference-solution prompt.  A trajectory is eligible only when generation
completed and its strict status is correct/incorrect; every response token,
including the first token and EOS, is scored without shifting or truncation.

Only one decoder KV cache is live at a time.  A first pass writes student hidden
states to a temporary CPU memmap; the teacher pass reads them in position order.
The shared output head is applied in vocabulary blocks.  Online softmax moments
compute FP32 forward/reverse KL and entropies without [response, vocabulary]
logits.  All model operations run under inference_mode.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
METRICS = ("gap", "reverse_gap", "entropy_student", "entropy_teacher")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_ids(values, vocabulary, label):
    if (not isinstance(values, list) or not values
            or any(type(value) is not int or not 0 <= value < vocabulary for value in values)):
        raise ValueError(f"{label}: expected a nonempty list of in-vocabulary integer token IDs")
    return values


def validate_context(student_prompt, teacher_prompt, response, maximum):
    # Conservatively budget the entire response, although the final response
    # token need not be fed to the model to predict that token.
    student_total = len(student_prompt) + len(response)
    teacher_total = len(teacher_prompt) + len(response)
    if max(student_total, teacher_total) > maximum:
        raise ValueError(f"Context overflow: student={student_total}, teacher={teacher_total}, "
                         f"limit={maximum}; no truncation is permitted")
    return student_total, teacher_total


def _collator_class(source_dir):
    path = source_dir / "data_collator.py"
    if not path.is_file():
        raise ValueError(f"Frozen data_collator.py is missing: {path}")
    spec = importlib.util.spec_from_file_location("_smoothness_score_frozen_collator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SelfDistillationDataCollator


def build_teacher_prompt(tokenizer, collator, problem, solution, maximum):
    message = (
        f"Problem: {problem}\n\nHere is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{collator.transition_prompt}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    text = tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
    untruncated = tokenizer(text, truncation=False)["input_ids"]
    if len(untruncated) > maximum:
        raise ValueError(f"Teacher reference prompt alone exceeds context: {len(untruncated)} > {maximum}")
    # Independently check against the frozen implementation.  This prevents a
    # future template edit from silently changing the teacher's conditioning.
    actual = collator([{"problem": problem, "solution": solution}])["teacher_prompts"][0].tolist()
    if actual != untruncated:
        raise ValueError("Frozen collator / untruncated teacher prompt mismatch")
    return untruncated


def prepare(args):
    """CPU-only protocol, prompt, solution, token, and context validation."""
    from datasets import load_from_disk
    from transformers import AutoConfig, AutoTokenizer

    for name in ("dataset", "model", "adapter", "source_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.results = [path.resolve() for path in args.results]
    if len(set(args.results)) != len(args.results):
        raise ValueError("Repeated --results paths are not distinct rounds")
    if args.chunk_size < 1 or args.vocab_chunk_size < 1:
        raise ValueError("Chunk sizes must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if not math.isfinite(args.gpu_memory_gib) or args.gpu_memory_gib <= 0:
        raise ValueError("Allocator limit must be finite and positive")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError("This audited decoder/head path currently supports Qwen3 only")
    native_limit = int(config.max_position_embeddings)
    maximum = args.max_context or native_limit
    if maximum < 1 or maximum > native_limit:
        raise ValueError("--max-context cannot extend the model's configured context")
    vocabulary = int(config.vocab_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    collator = _collator_class(args.source_dir)(
        tokenizer, max_length=maximum, reason_first=False,
        student_thinking=False, teacher_thinking=True,
    )
    dataset = load_from_disk(str(args.dataset))
    if not {"problem", "solution", "answer"}.issubset(dataset.column_names):
        raise ValueError("Diagnostic dataset must contain problem, solution, answer; answer-only teaching is not supported")
    references = {}
    for index, row in enumerate(dataset):
        problem, solution = row["problem"], row["solution"]
        if (not isinstance(problem, str) or not problem.strip()
                or not isinstance(solution, str) or not solution.strip()):
            raise ValueError(f"Missing genuine reference solution/problem at dataset row {index}")
        if problem in references:
            raise ValueError("Duplicate exact problem in diagnostic reference dataset")
        references[problem] = {"dataset_index": index, "id": str(row.get("id", index)),
                               "problem": problem, "solution": solution, "answer": str(row["answer"])}
    adapter_weights = args.adapter / "adapter_model.safetensors"
    adapter_config_path = args.adapter / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config_path.is_file():
        raise ValueError("Both adapter weights and adapter_config.json are required")
    adapter_config = json.loads(adapter_config_path.read_text())
    targets = adapter_config.get("target_modules") or []
    targets = [targets] if isinstance(targets, str) else targets
    if (adapter_config.get("modules_to_save")
            or any("lm_head" in target for target in targets)):
        raise ValueError("Scoring requires the unchanged shared output head; lm_head adapters/modules_to_save are unsupported")
    adapter_hash = sha256(adapter_weights)
    model_files = {path.name: sha256(path) for path in sorted(args.model.iterdir())
                   if path.is_file() and (path.suffix == ".safetensors"
                       or path.name in ("config.json", "generation_config.json", "tokenizer.json",
                                        "tokenizer_config.json", "model.safetensors.index.json"))}
    step = args.checkpoint_step
    if step is None:
        match = re.search(r"checkpoint-(\d+)$", args.adapter.name)
        if match is None:
            raise ValueError("Cannot infer checkpoint step; supply --checkpoint-step")
        step = int(match.group(1))
    if step < 0:
        raise ValueError("Checkpoint step must be nonnegative")
    sources, selected, question_records, counts = [], [], {}, Counter()
    for source_index, path in enumerate(args.results):
        data = json.loads(path.read_text())
        if not data.get("complete", data.get("summary", {}).get("complete", False)):
            raise ValueError(f"Results must be a completed, stable evaluation before scoring: {path}")
        protocol = data["protocol"]
        if protocol.get("adapter_sha256") and protocol["adapter_sha256"] != adapter_hash:
            raise ValueError(f"Generated adapter hash differs from scoring adapter: {path}")
        if protocol.get("adapter"):
            recorded_adapter = Path(protocol["adapter"]).resolve()
            if recorded_adapter != args.adapter and not protocol.get("adapter_sha256"):
                raise ValueError("Different adapter path without a matching recorded weight hash")
        if protocol.get("model") and Path(protocol["model"]).resolve() != args.model:
            raise ValueError("Generation and scoring base model paths differ")
        for name, expected in protocol.get("model_config_tokenizer_sha256", {}).items():
            if model_files.get(name) != expected:
                raise ValueError(f"Generation/scoring model or tokenizer mismatch: {name}")
        source_hash = sha256(path)
        sources.append({"path": str(path), "sha256": source_hash,
                        "protocol_sha256": canonical_hash(protocol)})
        questions = {q["index"]: q for q in data["questions"]}
        if len(questions) != len(data["questions"]):
            raise ValueError("Repeated question index")
        seen_questions = set()
        for row in data["rows"]:
            qi = row["index"]
            if qi not in questions or qi in seen_questions:
                raise ValueError("Repeated/unknown completed question row")
            seen_questions.add(qi)
            question = questions[qi]
            problem = question["problem"]
            if row["problem"] != problem or problem not in references:
                raise ValueError("Question/reference exact-problem join failed")
            reference = references[problem]
            if str(question["ground_truth"]).strip() != reference["answer"].strip():
                raise ValueError("Generation/reference answer mismatch")
            sp = _validate_ids(question["prompt_token_ids"], vocabulary, "student prompt")
            if question.get("prompt") is not None:
                if tokenizer.encode(question["prompt"], add_special_tokens=False) != sp:
                    raise ValueError("Saved student prompt text/token IDs disagree")
            reference_key = canonical_hash({"problem": problem, "solution": reference["solution"]})
            if reference_key not in question_records:
                tp = build_teacher_prompt(tokenizer, collator, problem, reference["solution"], maximum)
                question_records[reference_key] = {**reference, "teacher_prompt_token_ids": tp}
            tp = question_records[reference_key]["teacher_prompt_token_ids"]
            if question.get("teacher_prompt_token_ids") is not None and question["teacher_prompt_token_ids"] != tp:
                raise ValueError("Saved teacher prompt does not equal frozen full-solution collator prompt")
            seen_samples = set()
            for sample in row["samples"]:
                si = sample["sample_index"]
                if type(si) is not int or si < 0 or si in seen_samples:
                    raise ValueError("Repeated/invalid sample index")
                seen_samples.add(si)
                grade = sample["status"]
                counts[grade] += 1
                if grade not in ("correct", "incorrect"):
                    continue
                if sample.get("finish_reason") != "stop" or sample.get("truncated", False):
                    raise ValueError("Strict correctness was assigned to an incomplete response")
                for field in ("correct", "strict_correct", "answer_match"):
                    if field in sample and sample[field] is not (grade == "correct"):
                        raise ValueError(f"Inconsistent strict grade field: {field}")
                ids = _validate_ids(sample["token_ids"], vocabulary, "response")
                if sample.get("tokens", len(ids)) != len(ids):
                    raise ValueError("Saved response token count mismatch")
                try:
                    totals = validate_context(sp, tp, ids, maximum)
                except ValueError as exc:
                    raise ValueError(f"{path.name}, question {qi}, sample {si}: {exc}") from exc
                key = f"r{source_index:03d}_q{qi:04d}_s{si:03d}"
                metadata = {
                    "trajectory_id": key, "qid": str(question.get("id", qi)),
                    "question_index": qi, "question_id": str(question.get("id", qi)),
                    "problem": problem, "sample_index": si, "source_result_index": source_index,
                    "grade": grade, "finish_reason": "stop", "length": len(ids),
                    "optimizer_step": step, "checkpoint_step": step,
                    "student_prompt_length": len(sp), "teacher_prompt_length": len(tp),
                    "student_thinking": protocol.get("student_thinking", protocol.get("thinking")),
                    "teacher_thinking": True,
                    "generation_temperature": protocol.get("temperature"),
                    "token_ids_sha256": canonical_hash(ids),
                    "student_prompt_sha256": canonical_hash(sp),
                    "teacher_prompt_sha256": canonical_hash(tp),
                    "reference_sha256": reference_key,
                    "required_student_context": totals[0], "required_teacher_context": totals[1],
                }
                selected.append({"metadata": metadata, "token_ids": ids,
                                 "student_prompt": sp, "teacher_prompt": tp})
        if seen_questions != set(questions):
            raise ValueError("Completed result is missing question rows")
    identity = {
        "schema_version": SCHEMA_VERSION, "sources": sources,
        "dataset_path": str(args.dataset), "dataset_fingerprint": dataset._fingerprint,
        "references_sha256": canonical_hash(list(references.values())),
        "base_model_path": str(args.model), "base_model_sha256": model_files,
        "adapter_path": str(args.adapter), "adapter_sha256": adapter_hash,
        "adapter_config_sha256": sha256(adapter_config_path), "checkpoint_step": step,
        "scorer_sha256": sha256(__file__),
        "collator_sha256": sha256(args.source_dir / "data_collator.py"),
        "temperature": args.temperature, "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "context_limit": maximum, "native_context_limit": native_limit,
        "selected_trajectories_sha256": canonical_hash([item["metadata"] for item in selected]),
        "selection": "All completed strict correct/incorrect; all original response positions including first/EOS",
        "teacher": "Same frozen base, adapters disabled, complete reference solution visible",
        "student": "Checkpoint adapter enabled, exact generation prompt IDs; thinking mode from each result protocol",
        "metric": "Unclipped full-vocabulary KL(T||S), KL(S||T), and entropies; FP32 online softmax moments over native-dtype logits",
        "alignment": "Response token t predicted at original prompt_length - 1 + t, zero based",
        "context_policy": "Budget complete prompt+response; reject overflow, never truncate",
    }
    max_context = max((item["metadata"]["required_teacher_context"] for item in selected), default=0)
    max_context = max(max_context, max((item["metadata"]["required_student_context"] for item in selected), default=0))
    dtype_bytes = 4 if args.dtype == "float32" else 2
    kv_bytes = (2 * config.num_hidden_layers * config.num_key_value_heads
                * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                * max_context * dtype_bytes)
    plan = {"eligible_trajectories": len(selected), "all_status_counts": dict(counts),
            "eligible_status_counts": dict(Counter(x["metadata"]["grade"] for x in selected)),
            "max_required_context": max_context, "context_limit": maximum,
            "estimated_max_single_decoder_kv_gib": kv_bytes / 1024**3,
            "max_student_hidden_memmap_gib": max((len(x["token_ids"]) for x in selected), default=0)
                * config.hidden_size * dtype_bytes / 1024**3,
            "estimated_kv_excludes_weights_activations_workspace": True,
            "identity_sha256": canonical_hash(identity)}
    return identity, plan, selected, question_records, config


def iter_response_hidden(decoder, prompt, response, *, chunk_size, device):
    """Yield aligned [start:end] predictor hidden states with a fresh KV cache."""
    import torch
    if not prompt or not response or chunk_size < 1:
        raise ValueError("Nonempty prompt/response and positive chunk size required")
    inputs = prompt + response[:-1]
    first = len(prompt) - 1
    stop = first + len(response)
    past = output = None
    covered = 0
    try:
        for begin in range(0, len(inputs), chunk_size):
            end = min(begin + chunk_size, len(inputs))
            ids = torch.tensor([inputs[begin:end]], dtype=torch.long, device=device)
            positions = torch.arange(begin, end, device=device)
            output = decoder(
                input_ids=ids, attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                position_ids=positions.unsqueeze(0), cache_position=positions,
                past_key_values=past, use_cache=True, return_dict=True,
            )
            past = output.past_key_values
            if past is None:
                raise ValueError("Decoder did not return a KV cache")
            lower, upper = max(begin, first), min(end, stop)
            if lower < upper:
                a, b = lower - first, upper - first
                if a != covered:
                    raise ValueError("Response hidden-state alignment has a gap/overlap")
                hidden = output.last_hidden_state[0, lower - begin:upper - begin]
                if hidden.shape[0] != b - a:
                    raise ValueError("Decoder returned an unexpected hidden-state length")
                yield a, b, hidden
                covered = b
        if covered != len(response):
            raise ValueError("Not all response prediction positions were produced")
    finally:
        # Closing an interrupted generator also releases its single-side cache.
        past = output = None


def write_student_hidden(decoder, prompt, response, path, *, chunk_size, device, hidden_size, dtype):
    import torch
    numpy_dtype = {torch.bfloat16: np.uint16, torch.float16: np.float16,
                   torch.float32: np.float32}[dtype]
    array = np.lib.format.open_memmap(path, mode="w+", dtype=numpy_dtype,
                                      shape=(len(response), hidden_size))
    with closing(iter_response_hidden(decoder, prompt, response, chunk_size=chunk_size, device=device)) as iterator:
        for begin, end, hidden in iterator:
            if hidden.dtype != dtype or hidden.shape[1] != hidden_size:
                raise ValueError("Unexpected hidden-state dtype/width")
            cpu = hidden.detach().to("cpu").contiguous()
            if dtype == torch.bfloat16:
                cpu = cpu.view(torch.uint16)
            array[begin:end] = cpu.numpy()
    array.flush()
    del array


def read_hidden_slice(array, begin, end, *, dtype, device):
    import torch
    # A copy avoids writable-tensor aliases to the read-only memmap.  uint16
    # preserves every bfloat16 bit; converting through float16 would not.
    tensor = torch.from_numpy(np.array(array[begin:end], copy=True))
    if dtype == torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
    if tensor.dtype != dtype:
        raise ValueError("Student hidden memmap dtype mismatch")
    return tensor.to(device)


def metrics_from_hidden(student, teacher, head, *, temperature, vocab_chunk_size):
    """Exact-vocabulary moments with FP32 accumulation and bounded workspace.

    For each distribution maintain a running maximum m, partition Z, own-logit
    moment A, and cross-logit moment B, all scaled by exp(-m).  Then
    H = m + log Z - A/Z, and KL(T||S) = (A_t-B_t)/Z_t + logZ_s-logZ_t.
    This needs only one pass over vocabulary blocks, not full logits or a
    second softmax-normalization pass.
    """
    import torch
    import torch.nn.functional as F
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("Student/teacher predictor hidden states must align")
    n = student.shape[0]
    shape = (n,)
    sm = torch.full(shape, -torch.inf, dtype=torch.float32, device=student.device)
    tm = sm.clone()
    sz = torch.zeros_like(sm); tz = torch.zeros_like(sm)
    sa = torch.zeros_like(sm); ta = torch.zeros_like(sm)
    sb = torch.zeros_like(sm); tb = torch.zeros_like(sm)
    for begin in range(0, head.weight.shape[0], vocab_chunk_size):
        end = min(begin + vocab_chunk_size, head.weight.shape[0])
        weight = head.weight[begin:end]
        bias = head.bias[begin:end] if head.bias is not None else None
        s = F.linear(student, weight, bias).float() / temperature
        t = F.linear(teacher, weight, bias).float() / temperature
        new_sm = torch.maximum(sm, s.max(-1).values)
        new_tm = torch.maximum(tm, t.max(-1).values)
        sr, tr = (sm - new_sm).exp(), (tm - new_tm).exp()
        sw, tw = (s - new_sm[:, None]).exp(), (t - new_tm[:, None]).exp()
        sz, tz = sz * sr + sw.sum(-1), tz * tr + tw.sum(-1)
        sa, ta = sa * sr + (sw * s).sum(-1), ta * tr + (tw * t).sum(-1)
        sb, tb = sb * sr + (sw * t).sum(-1), tb * tr + (tw * s).sum(-1)
        sm, tm = new_sm, new_tm
    slz, tlz = sm + sz.log(), tm + tz.log()
    values = {
        "gap": (ta - tb) / tz + slz - tlz,
        "reverse_gap": (sa - sb) / sz + tlz - slz,
        "entropy_student": slz - sa / sz,
        "entropy_teacher": tlz - ta / tz,
    }
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite {name} in full-vocabulary scoring")
        if bool((value < -5e-4).any()):
            raise FloatingPointError(f"Negative {name} exceeds FP32 roundoff tolerance")
    # This is roundoff protection at zero, not upper clipping of divergences.
    return {name: value.clamp_min(0).detach().cpu().numpy() for name, value in values.items()}


def _paths(output, item):
    key = item["metadata"]["trajectory_id"]
    return output / "tokens" / (key + ".npz"), output / "tokens" / (key + ".json")


def completed_record(output, item, identity_hash):
    array_path, metadata_path = _paths(output, item)
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if (metadata["identity_sha256"] != identity_hash
            or metadata["trajectory_metadata"] != item["metadata"]):
        raise ValueError("Completed trajectory belongs to a different scoring identity")
    if not array_path.is_file() or sha256(array_path) != metadata["npz_sha256"]:
        raise ValueError(f"Completed trajectory file is missing/corrupt: {array_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["token_ids"], item["token_ids"]):
            raise ValueError("Completed trajectory token IDs differ from generation")
        for name in METRICS:
            if arrays[name].shape != (len(item["token_ids"]),) or not np.isfinite(arrays[name]).all():
                raise ValueError("Completed trajectory metric alignment/values are invalid")
    return metadata


def export_completed(output, selected, identity_hash, plan):
    rows, chunks = [], {name: [] for name in METRICS}
    offsets = [0]
    for item in selected:
        metadata = completed_record(output, item, identity_hash)
        if metadata is None:
            continue
        path, _ = _paths(output, item)
        with np.load(path, allow_pickle=False) as arrays:
            for name in METRICS:
                chunks[name].append(np.array(arrays[name], dtype=np.float32))
        offsets.append(offsets[-1] + len(item["token_ids"]))
        rows.append({**item["metadata"], "token_file": str(path),
                     "identity_sha256": identity_hash})
    packed = {name: np.concatenate(values) if values else np.empty(0, dtype=np.float32)
              for name, values in chunks.items()}
    packed["gaps"] = packed.pop("gap")
    now = datetime.now(timezone.utc).isoformat()
    atomic_npz(output / "flattened_gap.npz", **packed,
               offsets=np.array(offsets, dtype=np.int64),
               steps=np.array([row["checkpoint_step"] for row in rows], dtype=np.int64),
               grades=np.array([row["grade"] for row in rows], dtype="U16"),
               snapshot_time=np.array(now))
    atomic_json(output / "rows.json", rows)
    summary = {**plan, "scored_trajectories": len(rows), "scored_tokens": offsets[-1],
               "complete": len(rows) == len(selected), "snapshot_time": now,
               "scored_grade_counts": dict(Counter(row["grade"] for row in rows)),
               "distinct_questions_by_grade": {
                   grade: len({row["problem"] for row in rows if row["grade"] == grade})
                   for grade in ("correct", "incorrect")},
               "first_response_token_included": True, "eos_included_when_present": True,
               "steps_semantics": "Frozen checkpoint index, not a sequence of training updates",
               "zero_padding": False}
    atomic_json(output / "summary.json", summary)
    return summary


def score(args, identity, plan, selected, config):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    output = args.output_dir
    identity_hash = plan["identity_sha256"]
    pending = [item for item in selected if completed_record(output, item, identity_hash) is None]
    if not pending:
        return export_completed(output, selected, identity_hash, plan)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only a single CPU or CUDA device is supported")
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and len(visible.split(",")) != 1:
            raise ValueError("Expose at most one CUDA device to the scorer")
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        limit = int(args.gpu_memory_gib * 1024**3)
        if limit >= total or free < limit + 1024**3:
            raise RuntimeError("Insufficient free GPU memory for allocator limit plus 1 GiB reserve")
        # This limits PyTorch allocator reservations; CUDA/context/third-party
        # memory is outside it and must be reserved by the external scheduler.
        torch.cuda.set_per_process_memory_fraction(limit / total, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype,
        attn_implementation=args.attn_implementation, device_map={"": str(device)},
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    model.requires_grad_(False)
    causal = model.get_base_model()
    decoder, head = causal.model, causal.get_output_embeddings()
    if type(head) is not torch.nn.Linear or head.weight.shape != (config.vocab_size, config.hidden_size):
        raise ValueError("Output head is not the unchanged Qwen3 linear head")
    cache_dir = output / "hidden_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    phase, current = "loaded", None
    try:
        with torch.inference_mode():
            for item in pending:
                current = item["metadata"]["trajectory_id"]
                started = time.monotonic()
                hidden_path = cache_dir / (current + ".student.npy")
                phase = "student_hidden"
                atomic_json(output / "progress.json", {"stage": phase, "trajectory_id": current,
                            "eligible_trajectories": len(selected), "identity_sha256": identity_hash})
                write_student_hidden(
                    decoder, item["student_prompt"], item["token_ids"], hidden_path,
                    chunk_size=args.chunk_size, device=device,
                    hidden_size=config.hidden_size, dtype=dtype,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                stored = np.load(hidden_path, mmap_mode="r", allow_pickle=False)
                arrays = {name: np.empty(len(item["token_ids"]), dtype=np.float32) for name in METRICS}
                phase = "teacher_and_metrics"
                atomic_json(output / "progress.json", {"stage": phase, "trajectory_id": current,
                            "identity_sha256": identity_hash})
                try:
                    with model.disable_adapter():
                        with closing(iter_response_hidden(
                            decoder, item["teacher_prompt"], item["token_ids"],
                            chunk_size=args.chunk_size, device=device,
                        )) as iterator:
                            for begin, end, teacher_hidden in iterator:
                                student_hidden = read_hidden_slice(stored, begin, end, dtype=dtype, device=device)
                                metrics = metrics_from_hidden(
                                    student_hidden, teacher_hidden, head,
                                    temperature=args.temperature, vocab_chunk_size=args.vocab_chunk_size,
                                )
                                for name in METRICS:
                                    arrays[name][begin:end] = metrics[name]
                                del student_hidden, teacher_hidden, metrics
                finally:
                    model.requires_grad_(False)
                    del stored
                array_path, metadata_path = _paths(output, item)
                atomic_npz(array_path, token_ids=np.array(item["token_ids"], dtype=np.int32), **arrays)
                atomic_json(metadata_path, {
                    "schema_version": SCHEMA_VERSION, "identity_sha256": identity_hash,
                    "trajectory_metadata": item["metadata"], "npz_sha256": sha256(array_path),
                    "chunk_size": args.chunk_size, "vocab_chunk_size": args.vocab_chunk_size,
                    "temperature": args.temperature, "dtype": args.dtype,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                hidden_path.unlink()
                atomic_json(output / "progress.json", {"stage": "trajectory_complete", "trajectory_id": current,
                            "elapsed_seconds": time.monotonic() - started,
                            "identity_sha256": identity_hash})
                print("TRAJECTORY_SCORED", current, item["metadata"]["grade"], len(item["token_ids"]),
                      round(time.monotonic() - started, 3), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    except Exception as exc:
        oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        atomic_json(output / "last_error.json", {
            "trajectory_id": current, "phase": phase, "oom": oom,
            "error": f"{type(exc).__name__}: {exc}",
            "completed_trajectories_preserved": True,
            "retry": "Same identity may resume with smaller chunk sizes; no context/response truncation",
        })
        export_completed(output, selected, identity_hash, plan)
        raise
    finally:
        del decoder, head, causal, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return export_completed(output, selected, identity_hash, plan)


def check_cpu():
    """Tiny causal attention model: full-prefix vs cached/chunked predictions."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(13011)

    class ToyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(29, 12)
            self.position = torch.nn.Embedding(128, 12)
            self.q = torch.nn.Linear(12, 12, bias=False)
            self.k = torch.nn.Linear(12, 12, bias=False)
            self.v = torch.nn.Linear(12, 12, bias=False)
            self.norm = torch.nn.LayerNorm(12)
            self.adapter = .2

        def forward(self, input_ids, attention_mask=None, position_ids=None,
                    cache_position=None, past_key_values=None, use_cache=False, return_dict=True):
            old = 0 if past_key_values is None else past_key_values[0].shape[1]
            n = input_ids.shape[1]
            positions = torch.arange(old, old + n)
            if position_ids is not None:
                assert torch.equal(position_ids, positions[None])
            if cache_position is not None:
                assert torch.equal(cache_position, positions)
            if attention_mask is not None:
                assert attention_mask.shape == (1, old + n)
            x = self.embedding(input_ids) + self.position(positions)[None]
            x = x + self.adapter * torch.tanh(x)
            q, k, v = self.q(x), self.k(x), self.v(x)
            if past_key_values is not None:
                k = torch.cat((past_key_values[0], k), dim=1)
                v = torch.cat((past_key_values[1], v), dim=1)
            attention = q @ k.transpose(-1, -2) / math.sqrt(12)
            future = torch.arange(old + n)[None, :] > positions[:, None]
            attention = attention.masked_fill(future[None], -torch.inf).softmax(-1)
            hidden = self.norm(x + attention @ v)
            return SimpleNamespace(last_hidden_state=hidden,
                                   past_key_values=(k, v) if use_cache else None)

    decoder = ToyDecoder().eval()
    head = torch.nn.Linear(12, 29, bias=True).eval()
    sp, tp = [3, 5, 7, 9, 4], [2, 3, 1, 4, 5, 6, 8, 9]
    response = [8, 12, 4, 10, 3, 6, 14, 2, 7, 11, 2]
    records = []
    with torch.inference_mode(), tempfile.TemporaryDirectory(prefix="score_alignment_") as folder:
        def full(prompt, adapter):
            decoder.adapter = adapter
            hidden = decoder(torch.tensor([prompt + response[:-1]])).last_hidden_state[0]
            return hidden[len(prompt) - 1:len(prompt) - 1 + len(response)]

        full_s, full_t = full(sp, .2), full(tp, 0.)
        slog = F.log_softmax(head(full_s).float() / 1.1, dim=-1)
        tlog = F.log_softmax(head(full_t).float() / 1.1, dim=-1)
        reference = {
            "gap": (tlog.exp() * (tlog - slog)).sum(-1).numpy(),
            "reverse_gap": (slog.exp() * (slog - tlog)).sum(-1).numpy(),
            "entropy_student": -(slog.exp() * slog).sum(-1).numpy(),
            "entropy_teacher": -(tlog.exp() * tlog).sum(-1).numpy(),
        }
        for chunk in (1, 2, 7, 128):
            decoder.adapter = .2
            path = Path(folder) / "student.npy"
            write_student_hidden(decoder, sp, response, path, chunk_size=chunk,
                                 device="cpu", hidden_size=12, dtype=torch.float32)
            array = np.load(path, mmap_mode="r")
            student = read_hidden_slice(array, 0, len(response), dtype=torch.float32, device="cpu")
            torch.testing.assert_close(student, full_s, atol=1e-6, rtol=1e-6)
            decoder.adapter = 0.
            parts = list(iter_response_hidden(decoder, tp, response, chunk_size=chunk, device="cpu"))
            teacher = torch.cat([hidden for _, _, hidden in parts])
            assert parts[0][0] == 0 and parts[-1][1] == len(response)
            torch.testing.assert_close(teacher, full_t, atol=1e-6, rtol=1e-6)
            for vocabulary_chunk in (1, 5, 29, 100):
                actual = metrics_from_hidden(student, teacher, head, temperature=1.1,
                                             vocab_chunk_size=vocabulary_chunk)
                errors = {name: float(np.max(np.abs(actual[name] - reference[name]))) for name in METRICS}
                assert max(errors.values()) < 2e-6, errors
                records.append({"chunk_size": chunk, "vocab_chunk_size": vocabulary_chunk,
                                "max_abs_error": max(errors.values())})
            del array
        bf = torch.randn(4, 12).bfloat16()
        raw_path = Path(folder) / "bits.npy"
        np.save(raw_path, bf.view(torch.uint16).numpy())
        raw = np.load(raw_path, mmap_mode="r")
        assert torch.equal(bf, read_hidden_slice(raw, 0, 4, dtype=torch.bfloat16, device="cpu"))
        del raw
        try:
            validate_context(sp, tp, response, len(tp) + len(response) - 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Context overflow was not rejected")
        validate_context(sp, tp, response, len(tp) + len(response))
    return {"passed": True, "device": "cpu", "first_token_and_last_token_checked": True,
            "separate_student_teacher_cache_checked": True, "bfloat16_memmap_bits_exact": True,
            "context_overflow_rejected": True, "comparisons": records,
            "maximum_metric_error": max(row["max_abs_error"] for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", type=Path)
    for name in ("dataset", "model", "adapter", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--gpu-memory-gib", "--allocator-limit-gib", dest="gpu_memory_gib", type=float, default=16.)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Accepted explicitly; identity-checked resume is always enabled")
    parser.add_argument("--prepare-only", action="store_true", help="CPU identity/prompt/context checks only")
    parser.add_argument("--export-only", action="store_true", help="CPU export of already completed trajectory files")
    parser.add_argument("--check-cpu", action="store_true", help="Standalone toy alignment/metric test; no model or GPU")
    parser.add_argument("--check-report", type=Path)
    args = parser.parse_args()
    if args.check_cpu:
        report = check_cpu()
        if args.check_report:
            atomic_json(args.check_report, report)
        print(json.dumps(report, indent=2), flush=True)
        return
    for name in ("results", "dataset", "model", "adapter", "output_dir"):
        if not getattr(args, name):
            parser.error("--" + name.replace("_", "-") + " is required outside --check-cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "score.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity, plan, selected, questions, config = prepare(args)
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity"] != identity:
                raise ValueError("Existing scoring manifest differs; use a separate output directory")
        else:
            manifest = {"identity": identity, "plan": plan, "runtime_sessions": []}
            atomic_json(manifest_path, manifest)
            atomic_json(args.output_dir / "questions.json", list(questions.values()))
        print("SCORING_PREPARED", json.dumps(plan), flush=True)
        if args.prepare_only:
            return
        if args.export_only:
            print(json.dumps(export_completed(args.output_dir, selected, plan["identity_sha256"], plan)), flush=True)
            return
        manifest["runtime_sessions"].append({
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "chunk_size": args.chunk_size, "vocab_chunk_size": args.vocab_chunk_size,
            "pytorch_allocator_limit_gib": args.gpu_memory_gib,
            "allocator_limit_excludes_cuda_context_and_other_processes": True,
        })
        atomic_json(manifest_path, manifest)
        summary = score(args, identity, plan, selected, config)
        print("SCORING_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
