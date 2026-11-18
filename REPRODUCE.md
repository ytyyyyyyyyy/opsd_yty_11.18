# Reproduce the experiments

The two source directories are separate run roots. Commands below assume Linux and run from the indicated root. Their Python files match the recorded runtime hashes. The AIME25 manifests use a relative path for the staged data, and the runners rewrite this entry when they freeze a new run. The original manifest hashes are retained under [provenance](provenance/source_sha256_as_run.json).

## Environment and inputs

The recorded runtime used Python 3.10, Torch 2.8.0, Transformers 4.57.1, TRL 0.26.0, PEFT 0.17.1, datasets 3.6.0, vLLM 0.11.0, and math-verify 0.8.0. See [the version inventory](provenance/max_gap_k5_environment.json) and [recorded package lists](environment). The runners use `.env-vllm/bin/python`, CUDA, FlashAttention, FlashInfer, and vLLM. Set up a compatible environment in each source root, or link the same environment under both. The archived acceleration wrapper was written for the original SM 12.0 host. On other hardware, use the CUDA toolkit, kernels and architecture supported by that GPU.

Place the Qwen/Qwen3-1.7B snapshot at `models/Qwen3-1.7B` in each source root. The recorded model revision was `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

`prepare_inputs.py` stages the fixed-revision mathematical training and AIME25 data in Hugging Face datasets format. You can reuse staged directories in both source roots. From the package root, with the Python data dependencies installed, run:

```bash
python prepare_inputs.py sources/max_gap_k5
python prepare_inputs.py sources/mean_kl_smooth
```

Each experiment's preparation applies the original exact normalized AIME25 match exclusion. The recorded training data retained all 29,434 rows. This procedure only checks exact normalized question matches.

## Maximum gap and 5-token replacement

From `sources/max_gap_k5/`:

```bash
# CPU checks on synthetic fixtures
CUDA_VISIBLE_DEVICES="" .env-vllm/bin/python reproduction/check_gap_intervention.py

# Freeze local inputs and validate the frozen run, then train/evaluate on GPU 0
.env-vllm/bin/python reproduction/run_gap_intervention.py prepare --run-dir runs/gap --gpu 0
.env-vllm/bin/python reproduction/run_gap_intervention.py check --run-dir runs/gap
.env-vllm/bin/python reproduction/run_gap_intervention.py run --run-dir runs/gap --arms max_gap_k5
```

The run command uses a manual stopping policy. From another terminal, run `.env-vllm/bin/python reproduction/run_gap_intervention.py stop --run-dir runs/gap`. The original run stopped after the complete step-65 validation and three further optimizer updates that were logged without being checkpointed. A new run evaluates its own base model. The separate historical reference points in the checked-in CSV came from an earlier OPSD run with the same evaluation sampling settings but different training implementation and batching.

## Mean KL and smoothness

From `sources/mean_kl_smooth/`:

```bash
CUDA_VISIBLE_DEVICES="" .env-vllm/bin/python reproduction/check_gap_smoothness.py

# Set OPSD_ORIGINAL_RUN to the original OPSD run directory before proceeding.
.env-vllm/bin/python reproduction/run_smoothness_pilot.py prepare \
  --run-dir runs/smooth_pilot --origin "${OPSD_ORIGINAL_RUN:?set the original OPSD run path}" --gpu 0
.env-vllm/bin/python reproduction/run_smoothness_pilot.py run --run-dir runs/smooth_pilot

.env-vllm/bin/python reproduction/run_smoothness_followup.py prepare \
  --run-dir runs/smooth_followup --parent-run runs/smooth_pilot
.env-vllm/bin/python reproduction/run_smoothness_followup.py run \
  --run-dir runs/smooth_followup --parent-run runs/smooth_pilot
```

The preparation expects the original run layout: `config.json`, `protocol.json`, `source/`, `train_dataset/`, `validation_dataset/` and `training/opsd/checkpoint-35/`. The follow-up preparation also reads the original `training/opsd/intervention_events.jsonl` to reproduce the exclusion of questions already seen in recorded rollouts. Supply the historical artifacts to rerun the exact split and continuation. The [checkpoint manifest](provenance/checkpoint_35_sha256.json) and [fixed diagnostic question indices](provenance/diagnostic_pool.json) identify the archived inputs. For a new run, train an OPSD-only arm with these sources and the [recorded baseline settings](provenance/original_opsd_config.json) to produce a fresh step-35 checkpoint. Different hardware, library builds or rollouts can produce different samples.

## Read the results without a GPU

The CSV tables contain measured data. Accuracies and rates in the CSV files are fractions in [0, 1]. The position CSV files give the contributor count `n`, mean gap, arm and answer grade at each absolute response-token position. The endpoint CSV is compressed with gzip. Empty fixed-window cells indicate no complete trajectories for that window. The [full report](report.pdf) includes original plots, configuration context and the limits of the evidence.
