"""CPU-only protocol, token-index semantics, and reports for gap interventions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from pathlib import Path

KS = (1, 3, 5, 10)
STRATEGIES = ("max_gap", "pre_max_gap", "random")
ARMS = ("opsd",) + tuple(f"{strategy}_k{k}" for k in KS for strategy in STRATEGIES)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


def parse_arm(arm):
    if arm == "opsd":
        return "opsd", 0
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    strategy, k = arm.rsplit("_k", 1)
    return strategy, int(k)


def canonical_problem(text):
    """Exact normalized matching, not a claim of semantic decontamination."""
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = re.sub(r"^problem\s*:\s*", "", text)
    return re.sub(r"\s+", "", text)


def candidate_positions(ids, special_ids=(), max_k=max(KS)):
    """Response indices, zero based. C=2 in [A,B,C,D,...].

    All K and both peak arms share candidates. Index 0 is excluded because
    the pre-peak arm must have a response token to replace. Both full spans
    must avoid special tokens; an EOS is never selected as a repair target.
    """
    special = set(special_ids)
    return [i for i in range(1, len(ids) - max_k + 1)
            if not any(token in special for token in ids[i - 1:i + max_k])]


def choose_positions(gaps, candidates, strategy, seed):
    if not candidates:
        return None
    if any(not math.isfinite(gaps[i]) for i in candidates):
        raise ValueError("Non-finite gap at an eligible position")
    peak = max(candidates, key=lambda i: gaps[i])  # earliest index breaks ties
    if strategy == "max_gap":
        start = peak
    elif strategy == "pre_max_gap":
        start = peak - 1
    elif strategy == "random":
        start = random.Random(seed).choice(candidates)
    else:
        raise ValueError(strategy)
    return {"peak_response_index": peak, "replacement_start": start}


def prediction_positions(prompt_length, response_length):
    """Raw logits indices that predict response tokens 0..response_length-1."""
    if prompt_length < 1 or response_length < 0:
        raise ValueError("Invalid prompt/response length")
    return list(range(prompt_length - 1, prompt_length + response_length - 1))


def gap_window(original, modified, start, horizon):
    if start < 0 or horizon < 1:
        raise ValueError("Invalid comparison window")
    count = max(0, min(horizon, len(original) - start, len(modified) - start))
    if not count:
        return {"start": start, "count": 0, "original_mean": None,
                "modified_mean": None, "delta": None}
    before = sum(original[start:start + count]) / count
    after = sum(modified[start:start + count]) / count
    return {"start": start, "count": count, "original_mean": before,
            "modified_mean": after, "delta": before - after}


def token_category(text):
    if not text or text.isspace():
        return "whitespace"
    if re.search(r"\d", text):
        return "contains_digit"
    if re.search(r"[=+*/<>^{}$\\−×÷]", text):
        return "math_or_latex_symbol"
    if re.search(r"[A-Za-z]", text):
        return "alphabetic"
    return "other"


def complete_checkpoint(directory, expected_step=None):
    directory = Path(directory)
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "adapter_model.safetensors", "adapter_config.json")
    if not all((directory / name).is_file() for name in required):
        return False
    try:
        state = json.loads((directory / "trainer_state.json").read_text())
        return expected_step is None or state["global_step"] == expected_step
    except (ValueError, KeyError):
        return False


def latest_checkpoint(directory):
    candidates = []
    for path in Path(directory).glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if complete_checkpoint(path, step):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda pair: pair[0])


def summarize_events(path, after_step, through_step):
    selected = []
    if Path(path).is_file():
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                if after_step < row["optimizer_step"] <= through_step:
                    selected.append(row)
    # A failed partial block may be retried from its durable checkpoint.
    rows = {r["event_key"]: r for r in selected}.values()
    rows = list(rows)
    applied = [r for r in rows if r.get("applied")]
    common = [r["common_window"]["delta"] for r in applied
              if (r.get("common_window") or {}).get("delta") is not None]
    own = [r["after_patch_window"]["delta"] for r in applied
           if r.get("after_patch_window", {}).get("delta") is not None]
    graded = [r for r in applied if r.get("initial_correct") is not None]
    return {
        "training_examples": len(rows), "interventions": len(applied),
        "gap_delta_common": sum(common) / len(common) if common else None,
        "gap_delta_common_examples": len(common),
        "gap_delta_after_patch": sum(own) / len(own) if own else None,
        "wrong_to_right": sum(not r["initial_correct"] and r["final_correct"] for r in graded),
        "right_to_wrong": sum(r["initial_correct"] and not r["final_correct"] for r in graded),
        "gradable_interventions": len(graded),
        "unchanged_spans": sum(r.get("unchanged_span", False) for r in applied),
        "actual_student_generated_tokens": sum(r["actual_student_generated_tokens"] for r in rows),
        "actual_teacher_generated_tokens": sum(r["actual_teacher_generated_tokens"] for r in rows),
    }


def record_validation(run, arm, step, result_path, block_summary=None):
    run, result_path = Path(run), Path(result_path)
    result = json.loads(result_path.read_text())
    summary, protocol = result["summary"], result["protocol"]
    if (not summary["complete"] or summary.get("probe") or summary["questions"] != 30
            or summary["samples"] != 360 or protocol["n"] != 12):
        raise ValueError("Only complete 30-question Avg@12 evaluations can enter the accuracy curve")
    rows = result["rows"]
    if (len({r["index"] for r in rows}) != 30 or any(len(r["samples"]) != 12 for r in rows)):
        raise ValueError("Invalid validation question/sample counts")
    correct = sum(bool(s["correct"]) for row in rows for s in row["samples"])
    if summary["correct"] != correct:
        raise ValueError("Validation summary disagrees with sample-level answers")
    if (arm == "base") != (protocol.get("adapter") is None):
        raise ValueError("Base/trained validation adapter mismatch")
    point = {"arm": arm, "step": step, "accuracy": correct / 360,
             "correct": correct, "samples": 360,
             "source": result.get("provenance", {}).get("source", "current_run"),
             "historical_source_file": result.get("provenance", {}).get("historical_source_file"),
             "truncated_rate": summary["truncated_rate"],
             "generated_tokens": summary["generated_tokens"],
             "result_file": str(result_path.resolve()), **(block_summary or {})}
    ledger = run / "validation_points.json"
    points = json.loads(ledger.read_text()) if ledger.is_file() else []
    matching = [p for p in points if (p["arm"], p["step"]) == (arm, step)]
    if matching and (matching[0]["correct"] != point["correct"]
                     or matching[0]["result_file"] != point["result_file"]):
        raise ValueError("Conflicting validation point; never overwrite a different evaluation")
    points = [p for p in points if (p["arm"], p["step"]) != (arm, step)] + [point]
    points.sort(key=lambda p: (p["step"], p["arm"]))
    atomic_json(ledger, points)
    keys = list(dict.fromkeys(key for p in points for key in p))
    with (run / "validation_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(points)
    render_curves(run, points)
    return point


def render_curves(run, points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    base_point = next((p for p in points if p["arm"] == "base"), None)
    base = base_point["accuracy"] if base_point else None
    active_path = run / "active_arms.json"
    active = json.loads(active_path.read_text()) if active_path.exists() else list(ARMS)
    ks = [k for k in KS if any(arm.endswith(f"_k{k}") for arm in active)] or list(KS)
    colors = {"opsd": "#777777", "random": "#e69f00", "max_gap": "#0072b2",
              "pre_max_gap": "#009e73"}
    if len(ks) == 1:
        fig, axis = plt.subplots(figsize=(10, 6))
        axes = [axis]
    else:
        fig, grid = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        axes = list(grid.flat)
    for k, ax in zip(ks, axes):
        if base is not None:
            label = "Base (historical)" if base_point.get("source") == "historical" else "Base (step 0)"
            ax.axhline(100 * base, color="#333333", linestyle="--", label=label)
        for strategy in ("opsd", "random", "max_gap", "pre_max_gap"):
            arm = strategy if strategy == "opsd" else f"{strategy}_k{k}"
            series = sorted((p for p in points if p["arm"] == arm), key=lambda p: p["step"])
            if not series:
                continue
            x, y = [p["step"] for p in series], [100 * p["accuracy"] for p in series]
            if base is not None and (not x or x[0] != 0):
                x, y = [0] + x, [100 * base] + y
            label = strategy + (" (historical)" if all(p.get("source") == "historical" for p in series) else "")
            ax.plot(x, y, ".-", color=colors[strategy], label=label)
        ax.set(title=f"K = {k}", xlabel="Optimizer updates", ylabel="AIME25 Avg@12 (%)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("AIME25 validation accuracy — raw observations, manual stopping")
    fig.tight_layout()
    temporary = run / "validation_accuracy.tmp.png"
    fig.savefig(temporary, dpi=160)
    temporary.replace(run / "validation_accuracy.png")
    plt.close(fig)
