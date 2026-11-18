# Gap-position teacher intervention experiment

This directory prepares the agreed experiment. Training and validation generation
start only with the explicit `run` command below. Preparation and checks use CPU.

## User update after launch, 2026-11-04

Only `max_gap_k5` is now active: replace five tokens starting at C, the peak.
Do not start the other twelve training configurations. Preserve a metrics record
for every optimizer step (`metrics/training_metrics.jsonl` and CSV), raw per-example
original/revised token traces and gaps, and optimizer witnesses. Validate AIME25
every five optimizer updates as before. See `active_arms.json` for execution scope.

Reuse the completed original OPSD experiment `runs/main_20261014_184922` as a
historical reference: checkpoints 25/50/75/100, plus the base model at step 0.
All imported points retain original paths and hashes, and the plot labels them
historical. Do not invent missing five-step OPSD measurements. The historical
training used vLLM rollouts and micro-batch 4/accumulation 8; the current training
uses Transformers and micro-batch 1/accumulation 32, with the chunked FP32
implementation. Effective batch, model/data revisions and main hyperparameters
match, but this is a historical comparison, not an identical rerun. Validation
sampling, prompts and grader match; sampled generations need not match exactly.
The interrupted duplicate base evaluation is archived with its partial samples.
`history_reuse.json` records checks, differences and result provenance.

## Frozen protocol

- Model: cached Qwen3-1.7B, fresh base/LoRA for every arm. Teacher is the frozen
  same base model with the reference solution, not the separate Qwen3-8B critic
  used by older experiments in this repository.
- Training: the original 29,434-row OpenThoughts OPSD math cache, excluding any
  normalized exact AIME25 question matches. Keep source order; no outcome or
  short-context filtering. Actual row counts/exclusions are in `data_split.json`.
- Only validation: AIME25, all 30 questions, no additional test split.
- 13 training arms: original OPSD; random, maximum-gap, and pre-maximum-gap
  starts, each with K=1,3,5,10. Base is evaluated once, with no training.
- Original local OPSD settings: seed/data_seed 42, micro-batch 1, accumulation 32,
  learning rate 5e-6, 30-epoch LR schedule, maximum response 1024, prompt cap
  20,000, LoRA rank 64/alpha 128, gradient clip .1, temperature 1.1, top-p .95,
  top-k 20, student non-thinking, privileged teacher thinking.
- Loss: existing all-position, full-vocabulary FP32 forward KL, vocabulary-entry
  upper clipping .05, average over valid response tokens. Train only the final
  mixed trajectory, including the teacher segment. The original branch is for
  selection/comparison, not an extra training example. The original OPSD arm
  uses the same implementation/settings and trains its unmodified trajectory.
- Gap: full-vocabulary forward KL at T=1.1, before clipping and sampling filters.

## Response token indices

Indices are zero based: `[A,B,C,D,E,...]`, peak token C has index 2.
Maximum-gap starts at index 2 (retain A,B); pre-maximum-gap starts at index 1
(retain A). Both replace exactly K tokens unless the teacher emits EOS.
The peak is located once on the original rollout; it is not located again after
the prefix changes. Raw logits index `prompt_length-1+j` predicts response j.
Teacher and student have different prompt lengths, so their logits are aligned
by response-token index, not raw input index.

All K use the same candidate set: response index >=1, with ten available
non-special target tokens and a non-special preceding token. If no candidate is
available, retain the original response and log a fallback; keep the example in
training. Teacher resamples autoregressively under the new prefix. Old suffix
and KV cache are discarded. EOS is honored and actual K is recorded.

## Paired diagnostics

Both branches are scored under one frozen model snapshot per training example.
At each branch prefix, the teacher and student score the same next-token state.
For max versus pre-max, compare at common absolute response positions beginning
at `peak+K`, for at most 64 positions available in both the original and revised
branches. This excludes teacher replacements in both arms. Random intervention
has its own paired original-versus-revised window beginning at `random_start+K`.
An additional own-boundary window is logged for each arm. These two window
definitions are explicitly separate; do not compare random's window as if it
were the max/pre common window.

Log positive `gap_delta = original_mean - revised_mean`, effective window length,
EOS/truncation, unchanged replacements, start-token lexical category, normalized
position, whether boxed-answer text already occurs in the prefix, all original/
revised token IDs and gaps. Lexical categories are descriptive heuristics, not
proof of a reasoning error. K=1 additionally exposes the original peak gap
after changing only its preceding token. Record wrong-to-right/right-to-wrong
where the reference has an extractable boxed answer; report grading coverage.
Record actual teacher/student generated tokens and runtime; equal optimizer
updates do not imply equal compute.

## Validation, scheduling, and manual stop

Base step 0 once; then round-robin all 13 arms in blocks of five optimizer
updates. Save complete adapter/optimizer/scheduler/RNG checkpoints, release the
training process, then run AIME25 on the independent student. Every completed
block prints one `VALIDATION` line and updates `validation.log`, JSON, CSV and
`validation_accuracy.png` (four K panels). The plot contains raw cumulative
Avg@12 observations and a base-model reference line, not a smoothing-based
stopping rule. All checkpoints are retained.

Validation matches the existing cached OPSD evaluator: 12 generations per
question, temperature 1, top-p 1, top-k disabled, thinking enabled, up to 38,912
new tokens, pinned question seeds, official boxed-answer/math-verify grading.
Avg@12 is correct generations /360, not pass@12. The entire 30-question
evaluation completes before its point enters the curve. No privileged
reference or teacher replacement is provided to the validation student.
Training is non-thinking and validation thinking, matching the existing local
OPSD reproduction. These settings are identical for all arms.

GPU 1 is the default. Before each child, require an idle card; do not terminate
other processes. Training and evaluation are sequential so their memory does
not overlap. Training PyTorch allocator cap: 20 GiB. Validation engine budget:
26 GiB. No process is launched during preparation. A GPU smoke/first real
training batch has deliberately not been run before launch authorization.

Stop manually; no plateau-based early stopping. A stop request saves at the
next optimizer boundary, and a validation already running finishes durably.
The original 30-epoch training horizon is retained as the LR plan; it is not a
new short automatic stop. Compare arms over matched updates when their manual
stop times differ. Final results are validation results, with no heldout-test
claim at this exploratory stage.

## Commands

From `/home/mafeiya/projects/OPSD`:

```bash
# Safe preparation / CPU checks; no training or validation generation.
bash reproduction/with_acceleration.sh .env-vllm/bin/python reproduction/run_gap_intervention.py prepare
bash reproduction/with_acceleration.sh .env-vllm/bin/python reproduction/run_gap_intervention.py check
bash reproduction/with_acceleration.sh .env-vllm/bin/python reproduction/run_gap_intervention.py plan

# RUN ONLY AFTER THE USER'S FINAL LAUNCH COMMAND.
bash reproduction/with_acceleration.sh .env-vllm/bin/python -u reproduction/run_gap_intervention.py run

# Request a durable manual stop from another shell.
.env-vllm/bin/python reproduction/run_gap_intervention.py stop

# Explicitly resume after a manual stop, preserving model/optimizer/RNG state.
bash reproduction/with_acceleration.sh .env-vllm/bin/python -u reproduction/run_gap_intervention.py run --resume
```

Default output: `runs/gap_intervention_20261104/`. Frozen config, source snapshots,
model/data hashes and Python package versions are recorded before launch.
`plan` only prints commands. `check` uses a CPU toy model through the actual
intervention path, verifies C/B alignment, common windows, training masks,
teacher stop-gradient, EOS/boundaries and accuracy/plot export. Synthetic plot
tests live in temporary directories and never enter actual validation results.
