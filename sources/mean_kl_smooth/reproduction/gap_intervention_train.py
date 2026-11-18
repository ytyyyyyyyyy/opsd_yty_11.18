"""Install trajectory intervention only; retain the local OPSD all-position loss.

This entry point is called by run_gap_intervention.py only after explicit launch.
"""
import ast
import copy
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install(base_class=None):
    import torch
    from math_verify import parse, verify
    if base_class is None:
        import opsd_trainer
        original = opsd_trainer.OPSDTrainer
    else:
        # Exercise the exact intervention methods on CPU without importing the
        # optional CUDA-only vLLM stack used by the real base trainer.
        opsd_trainer = None
        original = base_class
    from transformers import TrainerCallback
    from training_metrics import completion_info
    from reproduction.gap_intervention_core import (
        candidate_positions, choose_positions, gap_window, parse_arm,
        prediction_positions, stable_seed, token_category, atomic_json, summarize_events,
    )
    from reproduction.gap_intervention_math import full_vocabulary_gap

    grader_functions = [node for node in ast.parse((ROOT / "eval/evaluate_math.py").read_text()).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name in ("extract_boxed_answer", "grade_answer")]
    namespace = {"parse": parse, "verify": verify}
    exec(compile(ast.Module(body=grader_functions, type_ignores=[]), "official_grader", "exec"), namespace)
    extract, grade = namespace["extract_boxed_answer"], namespace["grade_answer"]
    run = Path(os.environ["GAP_INTERVENTION_RUN"])
    config = json.loads((run / "config.json").read_text())
    arm = os.environ["GAP_INTERVENTION_ARM"]
    strategy, k = parse_arm(arm)

    class MetadataCollator:
        def __init__(self, original):
            self.original = original

        def __call__(self, features):
            batch = self.original(features)
            batch["gap_problem"] = [f["problem"] for f in features]
            batch["gap_reference"] = [f["solution"] for f in features]
            return batch

    class ManualStop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if (run / "STOP").exists():
                control.should_training_stop = True
                control.should_save = True
                print("MANUAL_STOP_AT_UPDATE", state.global_step, flush=True)
            return control

    class PerStepLog(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step < 1 or not logs or "loss" not in logs:
                return control
            directory = Path(args.output_dir) / "step_logs"
            record = {"arm": arm, "optimizer_step": state.global_step,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "framework_metrics": dict(logs),
                      "trajectory_diagnostics": summarize_events(
                          self.trainer.gap_events, state.global_step - 1, state.global_step)}
            path = directory / f"step-{state.global_step:06d}.json"
            if path.exists():
                previous = directory / "previous" / f"step-{state.global_step:06d}-{time.time_ns()}.json"
                previous.parent.mkdir(parents=True, exist_ok=True)
                previous.write_bytes(path.read_bytes())
            atomic_json(path, record)
            with (directory / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            print("STEP_LOG", state.global_step, str(path), flush=True)
            return control

    class GapInterventionTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.use_vllm or self.reason_first or not self.fixed_teacher:
                raise ValueError("Protocol requires Transformers rollouts and the fixed privileged OPSD teacher")
            if (self.beta != 0 or self.use_thinking_machines_loss or self.top_k_loss
                    or self.position_supervision_mode != "all"):
                raise ValueError("Protocol requires full-vocabulary, all-position OPSD forward KL")
            self.data_collator = MetadataCollator(self.data_collator)
            self.gap_events = Path(self.args.output_dir) / "intervention_events.jsonl"
            self.gap_events.parent.mkdir(parents=True, exist_ok=True)
            self.add_callback(ManualStop())
            self.add_callback(PerStepLog(self))

        @torch.no_grad()
        def _gap_generate(self, model, prompt, generation_config, budget, seed=None, teacher=False):
            if budget < 1:
                return []
            settings = copy.deepcopy(generation_config)
            settings.max_new_tokens = budget
            settings.use_cache = True
            settings.return_dict_in_generate = False
            settings.output_scores = False
            old_cache = model.config.use_cache
            model.config.use_cache = True
            devices = [prompt.device.index] if prompt.is_cuda else []
            rng = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
            adapter = model.disable_adapter() if teacher else nullcontext()
            try:
                with rng, adapter:
                    if seed is not None:
                        torch.manual_seed(seed)
                    output = model.generate(input_ids=prompt.unsqueeze(0),
                                            attention_mask=torch.ones_like(prompt).unsqueeze(0),
                                            generation_config=settings, use_cache=True)
                return output[0, prompt.numel():].detach().cpu().tolist()
            finally:
                model.config.use_cache = old_cache

        @torch.no_grad()
        def _score_gap(self, model, student_prompt, teacher_prompt, response):
            if not response:
                return [], []
            def forward(prompt):
                tail = torch.tensor(response, device=prompt.device, dtype=prompt.dtype)
                sequence = torch.cat([prompt, tail]).unsqueeze(0)
                positions = torch.tensor(prediction_positions(prompt.numel(), len(response)),
                                         device=prompt.device, dtype=torch.long)
                output = model(input_ids=sequence, attention_mask=torch.ones_like(sequence),
                               logits_to_keep=positions, use_cache=False)
                if output.logits.shape[1] != len(response):
                    raise RuntimeError("Response/logit alignment failed")
                return output.logits[0]
            student_logits = forward(student_prompt)
            with model.disable_adapter():
                teacher_logits = forward(teacher_prompt)
            return full_vocabulary_gap(student_logits, teacher_logits,
                                       temperature=self.temperature,
                                       chunk_tokens=config["gap"]["chunk_tokens"])

        @torch.no_grad()
        def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
            if inputs["student_prompts"].shape[0] != 1:
                raise ValueError("Audited token-prefix semantics require micro-batch 1")
            started = time.monotonic()
            was_training = model.training
            model.eval()
            try:
                prompt = inputs["student_prompts"][0]
                teacher_prompt = inputs["teacher_prompts"][0]
                budget = generation_config.max_new_tokens
                initial = self._gap_generate(model, prompt, generation_config, budget)
                old_gap, old_entropy = self._score_gap(model, prompt, teacher_prompt, initial)
                candidates = candidate_positions(initial, self.processing_class.all_special_ids,
                                                 max_k=max(config["ks"]))
                key = stable_seed(self.args.seed, inputs["gap_problem"][0],
                                  self.state.global_step, initial)
                chosen = choose_positions(old_gap, candidates, strategy, key) if strategy != "opsd" else None
                final, patch = list(initial), []
                new_gap, new_entropy = old_gap, old_entropy
                record = {"event_key": str(key), "arm": arm,
                          "optimizer_step": self.state.global_step + 1,
                          "problem": inputs["gap_problem"][0], "initial_ids": initial,
                          "eligible_positions": len(candidates), "requested_k": k,
                          "applied": False,
                          "fallback": "opsd_control" if strategy == "opsd" else "no_common_eligible_position"}
                if chosen is not None:
                    start, peak = chosen["replacement_start"], chosen["peak_response_index"]
                    prefix = initial[:start]
                    prefix_tensor = torch.tensor(prefix, device=prompt.device, dtype=prompt.dtype)
                    patch = self._gap_generate(model, torch.cat([teacher_prompt, prefix_tensor]),
                                               generation_config, k,
                                               stable_seed(key, "teacher"), teacher=True)
                    if not patch:
                        raise RuntimeError("Teacher returned an empty replacement")
                    new_prefix = prefix + patch
                    eos = generation_config.eos_token_id
                    eos = {eos} if isinstance(eos, int) else set(eos or [])
                    if patch[-1] not in eos:
                        tail = self._gap_generate(
                            model, torch.cat([prompt, torch.tensor(new_prefix, device=prompt.device,
                                                                  dtype=prompt.dtype)]),
                            generation_config, budget - len(new_prefix), stable_seed(key, "continuation"))
                    else:
                        tail = []
                    final = new_prefix + tail
                    assert final[:start] == initial[:start]
                    assert len(final) <= budget and len(patch) <= k
                    new_gap, new_entropy = self._score_gap(model, prompt, teacher_prompt, final)
                    horizon = config["gap"]["horizon"]
                    # The common window is a max-vs-pre-max diagnostic only.
                    # Random starts use their own paired post-intervention window.
                    common = (gap_window(old_gap, new_gap, peak + k, horizon)
                              if strategy != "random" else None)
                    start_text = self.processing_class.decode([initial[start]], skip_special_tokens=False)
                    original_slice = initial[start:start + len(patch)]
                    record.update(chosen, applied=True, fallback=None,
                                  actual_k=len(patch), teacher_ended_early=len(patch) < k,
                                  teacher_ended_with_eos=patch[-1] in eos,
                                  retained_prefix_exact=True, inserted_ids=patch,
                                  unchanged_span=patch == original_slice,
                                  changed_token_count=sum(a != b for a, b in zip(patch, original_slice)),
                                  peak_gap_original=old_gap[peak],
                                  peak_gap_modified=new_gap[peak] if peak < len(new_gap) else None,
                                  peak_entropy_original=old_entropy[peak],
                                  peak_token_text=self.processing_class.decode([initial[peak]]),
                                  start_token_text=start_text, start_token_category=token_category(start_text),
                                  replacement_relative_position=start / max(1, len(initial)),
                                  boxed_answer_already_in_prefix="\\boxed" in self.processing_class.decode(prefix),
                                  after_patch_window=gap_window(old_gap, new_gap, start + k, horizon),
                                  common_window=common,
                                  original_gaps=old_gap, modified_gaps=new_gap,
                                  student_raw_logits_prediction_index=int(prompt.numel()) - 1 + start,
                                  teacher_raw_logits_prediction_index=int(teacher_prompt.numel()) - 1 + start)
                initial_text = self.processing_class.decode(initial, skip_special_tokens=True)
                final_text = self.processing_class.decode(final, skip_special_tokens=True)
                target = extract(inputs["gap_reference"][0])
                record.update(final_ids=final,
                              model_step=self.state.global_step,
                              session_id=getattr(self.training_metrics, 'session_id', None),
                              sequence_index=getattr(self.training_metrics, 'trace_index', None),
                              diagnostic_temperature=self.temperature,
                              diagnostic_mode="eval_before_update_full_vocabulary",
                              original_gaps=old_gap, modified_gaps=new_gap,
                              original_student_entropies=old_entropy,
                              modified_student_entropies=new_entropy,
                              reference_answer_extractable=target is not None,
                              initial_correct=bool(grade(extract(initial_text), target)) if target is not None else None,
                              final_correct=bool(grade(extract(final_text), target)) if target is not None else None,
                              actual_student_generated_tokens=len(initial) + (
                                  len(final) - record["replacement_start"] - len(patch) if patch else 0),
                              actual_teacher_generated_tokens=len(patch),
                              elapsed_seconds=time.monotonic() - started,
                              objective="all-position original OPSD on final trajectory only")
                with self.gap_events.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                sequence = torch.cat([prompt, torch.tensor(final, device=prompt.device,
                                                          dtype=prompt.dtype)]).unsqueeze(0)
                attention = torch.ones_like(sequence)
                labels = sequence.clone()
                if pad_token_id is not None:
                    attention[sequence == pad_token_id] = 0
                    labels[sequence == pad_token_id] = -100
                self._last_rollout_info = [completion_info(
                    final, generation_config.eos_token_id, pad_token_id, budget,
                    source="gap_intervention_" + arm)]
                if self.training_metrics is not None:
                    self.training_metrics.observe_rollout(self._last_rollout_info,
                                                         time.monotonic() - started)
                return sequence, attention, labels
            finally:
                model.train(was_training)

    if opsd_trainer is not None:
        opsd_trainer.OPSDTrainer = GapInterventionTrainer
    return GapInterventionTrainer


if __name__ == "__main__":
    install()
    runpy.run_path(str(ROOT / "opsd_train.py"), run_name="__main__")
