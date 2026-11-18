"""Prepare without GPU work; explicit `run` launches the manually stopped queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.gap_intervention_core import (
    ARMS, KS, atomic_json, canonical_problem, complete_checkpoint,
    latest_checkpoint, record_validation, sha256, summarize_events,
)

DEFAULT_RUN = ROOT / "runs/gap_intervention_20261104"
PYTHON = ROOT / ".env-vllm/bin/python"
SOURCES = (
    "opsd_train.py", "opsd_trainer.py", "data_collator.py", "training_metrics.py",
    "reproduction/position_supervision.py", "reproduction/formal_callbacks.py",
    "reproduction/evaluate_full.py", "reproduction/evaluation_metrics.py",
    "reproduction/with_acceleration.sh", "eval/evaluate_math.py",
    "reproduction/gap_intervention_core.py", "reproduction/gap_intervention_math.py",
    "reproduction/gap_intervention_train.py", "reproduction/run_gap_intervention.py",
    "reproduction/check_gap_intervention.py", "reproduction/GAP_INTERVENTION.md",
    "reproduction/gap_intervention_plot_requirements.lock.txt",
)


def defaults(gpu):
    return {
        "schema_version": 1, "project_root": str(ROOT), "gpu": gpu,
        "model": str(ROOT / "models/Qwen3-1.7B"),
        "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "teacher": "Frozen same base model, reference solution visible, LoRA disabled",
        "ks": list(KS), "arms": list(ARMS),
        "training": {"seed": 42, "data_seed": 42, "micro_batch": 1,
                     "gradient_accumulation": 32, "learning_rate": 5e-6,
                     "planned_epochs": 30, "max_completion_length": 1024,
                     "max_prompt_length": 20000, "temperature": 1.1,
                     "top_p": .95, "top_k": 20, "max_grad_norm": .1,
                     "lora_rank": 64, "lora_alpha": 128,
                     "student_thinking": False, "teacher_thinking": True,
                     "pytorch_allocator_limit_gib": 20,
                     "loss": "FP32 all-position full-vocabulary forward KL; vocabulary-term upper clip .05",
                     "beta": 0, "token_clip": .05,
                     "train_on": "Final revised trajectory only; no additional original-trajectory loss",
                     "checkpoint_retention": "Keep all checkpoints; no automatic deletion"},
        "gap": {"divergence": "Unclipped full-vocabulary forward KL at temperature 1.1",
                "chunk_tokens": 16, "horizon": 64, "candidate_max_k": 10,
                "index": "Zero-based RESPONSE TOKEN index: peak C=2; max starts C=2; pre starts B=1",
                "candidates": "Response index >=1; all 10 target tokens and preceding token must be non-special",
                "max_pre_common_window": "peak_response_index + K, H<=64, paired common valid length",
                "random_window": "random_start + K, paired with original branch",
                "short_trajectory": "Keep original trace, record fallback; do not drop the training example",
                "teacher_eos": "Honor EOS, record actual K; no fabricated padding or continuation after EOS",
                "selection_rng": "Private deterministic RNG, does not consume student rollout RNG"},
        "validation": {"dataset": "aime25", "questions": 30, "n": 12,
                       "metric": "Avg@12 = correct generations / 360",
                       "every_optimizer_updates": 5, "temperature": 1.,
                       "max_new_tokens": 38912, "thinking": True,
                       "independent_student": True, "automatic_early_stopping": False,
                       "gpu_memory_gib": 26., "max_num_seqs": 12},
        "schedule": "Base validation once; round-robin each arm for 5 updates, save, release training process, validate, log and plot",
        "stop": "User stops manually using STOP file or SIGINT; finish current optimizer update and save",
        "datasets": {"training_source": str(ROOT / "data/formal_train"),
                     "training_repository": "siyanzhao/Openthoughts_math_30k_opsd",
                     "training_revision": "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62",
                     "validation_source": str(ROOT / "data/full_eval/aime25"),
                     "validation_revision": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
                     "overlap_policy": "Remove normalized exact AIME25 question matches; preserve order; no outcome/length filtering",
                     "test_dataset": None},
    }


def status(run, stage, **fields):
    row = {"stage": stage, "time_utc": datetime.now(timezone.utc).isoformat(), **fields}
    atomic_json(run / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def file_inventory(directory):
    return {str(p.resolve()): sha256(p) for p in sorted(Path(directory).rglob("*"))
            if p.is_file() and p.suffix in (".arrow", ".json", ".safetensors")}


def prepare(run, gpu, refresh=False):
    from datasets import load_from_disk
    config = defaults(gpu)
    if (run / "protocol.json").is_file():
        if not refresh:
            print("ALREADY_PREPARED", run)
            return
        if (run / "validation_points.json").exists() or any((run / "training").glob("*/checkpoint-*")):
            raise RuntimeError("Cannot refresh frozen sources/configuration after experiment execution")
    run.mkdir(parents=True, exist_ok=True)
    training = load_from_disk(config["datasets"]["training_source"])
    validation = load_from_disk(config["datasets"]["validation_source"])
    if len(training) != 29434 or len(validation) != 30:
        raise ValueError("Cached dataset sizes differ from the pinned source manifests")
    heldout = {canonical_problem(row["problem"]) for row in validation}
    excluded = [i for i, row in enumerate(training) if canonical_problem(row["problem"]) in heldout]
    excluded_set = set(excluded)
    indices = [i for i in range(len(training)) if i not in excluded_set]
    if not (run / "train_dataset").exists():
        training.select(indices).save_to_disk(str(run / "train_dataset"))
    if not (run / "validation_dataset").exists():
        validation.save_to_disk(str(run / "validation_dataset"))
    split = {"source_training_rows": len(training), "training_rows": len(indices),
             "validation_rows": len(validation), "excluded_training_indices": excluded,
             "training_indices": indices,
             "matching": "NFKC, lowercase, remove leading Problem:, remove whitespace; exact matching only"}
    atomic_json(run / "data_split.json", split)
    atomic_json(run / "config.json", config)
    hashes = {}
    for name in SOURCES:
        source, target = ROOT / name, run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = sha256(target)
    model_link = run / "source/models/Qwen3-1.7B"
    model_link.parent.mkdir(parents=True, exist_ok=True)
    if not model_link.exists():
        model_link.symlink_to(config["model"], target_is_directory=True)
    for name in (".env-vllm", ".env", "cache"):
        link = run / "source" / name
        if not link.exists():
            link.symlink_to(ROOT / name, target_is_directory=True)
    manifest = json.loads((ROOT / "reproduction/full_eval_manifest.json").read_text())
    manifest = {"aime25": {**manifest["aime25"], "path": str(run / "validation_dataset")}}
    atomic_json(run / "source/reproduction/full_eval_manifest.json", manifest)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "trl", "peft", "datasets", "vllm", "matplotlib", "math-verify")}
    protocol = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": hashes, "config_sha256": sha256(run / "config.json"),
                "split_sha256": sha256(run / "data_split.json"),
                "frozen_eval_manifest_sha256": sha256(run / "source/reproduction/full_eval_manifest.json"),
                "model_sha256": file_inventory(config["model"]),
                "dataset_sha256": {**file_inventory(run / "train_dataset"),
                                   **file_inventory(run / "validation_dataset")},
                "environment_versions": versions,
                "validation_role": "AIME25 is validation; no test split and no claim of unseen-test performance"}
    atomic_json(run / "protocol.json", protocol)
    (run / "preflight.json").unlink(missing_ok=True)
    status(run, "prepared_not_started", training_rows=len(indices), validation_rows=len(validation),
           training_configurations=len(ARMS), gpu=gpu)
    print("PREPARED_ONLY: no model loaded, no training or validation generation launched.", flush=True)


def verify_prepared(run):
    config = json.loads((run / "config.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    for path, expected in ((run / "config.json", protocol["config_sha256"]),
                           (run / "data_split.json", protocol["split_sha256"]),
                           (run / "source/reproduction/full_eval_manifest.json", protocol["frozen_eval_manifest_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Frozen preparation changed: {path}")
    for name, expected in protocol["source_sha256"].items():
        if sha256(run / "source" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Active preparation/runner source changed; refresh before the first launch: {name}")
    for group in ("model_sha256", "dataset_sha256"):
        for name, expected in protocol[group].items():
            if sha256(name) != expected:
                raise ValueError(f"Pinned input changed: {name}")
    current = {name: importlib.metadata.version(name) for name in protocol["environment_versions"]}
    if current != protocol["environment_versions"]:
        raise ValueError("Python environment versions changed since preparation")
    return config


def train_command(run, arm, target, previous=None):
    cfg = json.loads((run / "config.json").read_text())
    t = cfg["training"]
    command = [str(PYTHON), "-u", str(run / "source/reproduction/gap_intervention_train.py"),
               "--model_name_or_path", cfg["model"], "--dataset_path", str(run / "train_dataset"),
               "--output_dir", str(run / "training" / arm), "--run_config", arm,
               "--num_train_epochs", str(t["planned_epochs"]), "--stop_after_steps", str(target),
               "--save_steps", "5", "--logging_steps", "1", "--save_only_model", "False",
               "--seed", str(t["seed"]), "--data_seed", str(t["data_seed"]),
               "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "32",
               "--learning_rate", str(t["learning_rate"]), "--max_grad_norm", str(t["max_grad_norm"]),
               "--gradient_checkpointing", "--max_completion_length", str(t["max_completion_length"]),
               "--max_length", str(t["max_prompt_length"]), "--attn_implementation", "flash_attention_2",
               "--dtype", "bfloat16", "--bf16", "True", "--beta", "0", "--use_vllm", "False",
               "--task_gpu_allocator_limit_gib", str(t["pytorch_allocator_limit_gib"]),
               "--use_peft", "--fixed_teacher", "--lora_r", "64", "--lora_alpha", "128",
               "--lora_target_modules", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
               "--temperature", str(t["temperature"]), "--top_p", str(t["top_p"]), "--top_k", str(t["top_k"]),
               "--lmbda", "1", "--jsd_token_clip", str(t["token_clip"]), "--metrics_chunk_tokens", "16",
               "--student_thinking", "False", "--teacher_thinking", "True",
               "--position_supervision_mode", "all", "--report_to", "none"]
    if previous is not None:
        if not complete_checkpoint(previous):
            raise ValueError("Refuse incomplete resume checkpoint")
        command.extend(["--resume_checkpoint", str(previous)])
    return command


def evaluation_command(run, arm, step):
    cfg = json.loads((run / "config.json").read_text())
    output = run / "validation" / f"{arm}_step{step}.json"
    command = [str(PYTHON), "-u", str(run / "source/reproduction/evaluate_full.py"),
               "--dataset", "aime25", "--output", str(output),
               "--gpu-memory-gib", str(cfg["validation"]["gpu_memory_gib"]),
               "--max-num-seqs", str(cfg["validation"]["max_num_seqs"])]
    if arm != "base":
        checkpoint = run / "training" / arm / f"checkpoint-{step}"
        if not complete_checkpoint(checkpoint, step):
            raise ValueError(f"Cannot validate incomplete checkpoint: {checkpoint}")
        command.extend(["--adapter", str(checkpoint)])
    return command, output


def idle_gpu(gpu):
    details = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    used, utilization = map(int, details.strip().split(","))
    pids = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip()
    return used < 500 and utilization < 5 and not pids, details.strip()


def run_child(run, config, label, command, arm="opsd"):
    while not (run / "STOP").exists():
        idle, details = idle_gpu(config["gpu"])
        if idle:
            break
        status(run, "waiting_for_idle_gpu", gpu=config["gpu"], details=details, next_job=label)
        time.sleep(15)
    if (run / "STOP").exists():
        return None
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(config["gpu"]), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
               WANDB_MODE="disabled", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               GAP_INTERVENTION_RUN=str(run), GAP_INTERVENTION_ARM=arm,
               PYTHONPATH=str(run / "source") + os.pathsep + str(ROOT))
    log_path = run / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = run / "source/reproduction/with_acceleration.sh"
    env["OPSD_PROJECT_ROOT"] = str(ROOT)
    wrapper_command = ["bash", str(wrapper), *command]
    started = time.monotonic()
    with log_path.open("a") as stream:
        proc = subprocess.Popen(wrapper_command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        status(run, "running", job=label, arm=arm, pid=proc.pid, log=str(log_path), command=command)
        announced_stop = False
        last_heartbeat = started
        while proc.poll() is None:
            if (run / "STOP").exists() and not announced_stop:
                print("MANUAL_STOP_REQUESTED: saving at the next optimizer boundary; an active validation finishes durably.", flush=True)
                announced_stop = True
            if time.monotonic() - last_heartbeat >= 60:
                print("JOB_HEARTBEAT", label, "elapsed_seconds", int(time.monotonic() - started), flush=True)
                last_heartbeat = time.monotonic()
            time.sleep(2)
    with (run / "runtime.jsonl").open("a") as stream:
        stream.write(json.dumps({"job": label, "gpu": config["gpu"], "command": command,
                                 "exit_code": proc.returncode,
                                 "elapsed_seconds": time.monotonic() - started}) + "\n")
    if proc.returncode:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}; inspect {log_path}")
    return proc.returncode


def validate(run, config, arm, step):
    command, output = evaluation_command(run, arm, step)
    if not output.is_file() or not json.loads(output.read_text()).get("summary", {}).get("complete"):
        if run_child(run, config, f"validate_{arm}_step{step}", command, arm) is None:
            return
    previous_points = json.loads((run / "validation_points.json").read_text()) if (run / "validation_points.json").exists() else []
    previous = max((p["step"] for p in previous_points if p["arm"] == arm and p["step"] < step), default=0)
    block = summarize_events(run / "training" / arm / "intervention_events.jsonl", previous, step) if arm != "base" else {}
    point = record_validation(run, arm, step, output, block)
    line = (f"VALIDATION arm={arm} step={step} source={point['source']} AIME25_Avg@12={100 * point['accuracy']:.4f}% "
            f"correct={point['correct']}/360 gap_delta={point.get('gap_delta_common')} "
            f"gap_delta_after_patch={point.get('gap_delta_after_patch')} "
            f"wrong_to_right={point.get('wrong_to_right')} right_to_wrong={point.get('right_to_wrong')}")
    print(line, flush=True)
    with (run / "validation.log").open("a") as stream:
        stream.write(line + "\n")


def execute(run, resume=False, selected=None):
    config = verify_prepared(run)
    preflight = json.loads((run / "preflight.json").read_text()) if (run / "preflight.json").is_file() else {}
    if not preflight.get("passed") or preflight.get("protocol_sha256") != sha256(run / "protocol.json"):
        raise RuntimeError("CPU preflight has not passed")
    lock = (run / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This experiment queue is already running")
    if (run / "STOP").exists():
        if not resume:
            raise RuntimeError("Manual STOP exists; an explicit --resume is required")
        (run / "STOP").unlink()
    arms = selected or list(ARMS)
    if any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError("Invalid or duplicate selected arms")
    selection_path = run / "active_arms.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != arms:
        raise ValueError("Changing the active arm list mid-run requires a separate run directory")
    atomic_json(selection_path, arms)

    def request_stop(signum, frame):
        (run / "STOP").write_text(f"signal {signum}\n")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        validate(run, config, "base", 0)
        while not (run / "STOP").exists():
            points = json.loads((run / "validation_points.json").read_text())
            progress = {arm: max((p["step"] for p in points if p["arm"] == arm), default=0) for arm in arms}
            lowest = min(progress.values())
            # Catch up arms first; maintain equal update counts at each complete round.
            for arm in arms:
                if (run / "STOP").exists():
                    break
                if progress[arm] > lowest:
                    continue
                target = lowest + config["validation"]["every_optimizer_updates"]
                step, previous = latest_checkpoint(run / "training" / arm)
                if step > target:
                    raise RuntimeError("Unvalidated checkpoints ahead of the next scheduled validation")
                if step < target:
                    command = train_command(run, arm, target, previous)
                    if run_child(run, config, f"train_{arm}_to{target}", command, arm) is None:
                        break
                if (run / "STOP").exists():
                    break
                latest, _ = latest_checkpoint(run / "training" / arm)
                if latest != target:
                    raise RuntimeError(f"Training reached {latest}, expected {target}; inspect the saved state before continuing")
                validate(run, config, arm, target)
        status(run, "manually_stopped", active_arms=arms)
    except Exception as exc:
        status(run, "failed", error=str(exc))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "plan", "run", "stop", "status"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.action == "prepare":
        prepare(run, args.gpu, args.refresh)
    elif args.action == "check":
        verify_prepared(run)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"}
        subprocess.run([str(PYTHON), str(run / "source/reproduction/check_gap_intervention.py"),
                        "--run-dir", str(run)], cwd=ROOT, env=env, check=True)
    elif args.action == "plan":
        cfg = json.loads((run / "config.json").read_text())
        print(json.dumps({"config": cfg, "first_training_command": train_command(run, "max_gap_k1", 5),
                          "base_validation_command": evaluation_command(run, "base", 0)[0]}, indent=2))
    elif args.action == "run":
        execute(run, args.resume, args.arms)
    elif args.action == "stop":
        (run / "STOP").write_text("manual user request\n")
        print("STOP_REQUESTED", run)
    else:
        print((run / "status.json").read_text())


if __name__ == "__main__":
    main()
