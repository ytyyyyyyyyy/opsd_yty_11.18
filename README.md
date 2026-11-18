# OPSD experiments on Qwen3-1.7B

Two exploratory experiments evaluate teacher intervention at the maximum teacher–student KL gap and an auxiliary loss for mean KL and local smoothness. This directory contains the historical source snapshots, aggregate results, figure data, and [the full Chinese report](report.pdf).

| Experiment | AIME25 observation | Comparison |
| --- | --- | --- |
| Maximum-gap teacher replacement, 5 tokens | Step 50: **162/360 (45.00%)**. Step 65: 152/360 (42.22%). | Historical OPSD at step 50: 148/360 (41.11%), a difference of **+3.89 percentage points**. Rollout backends and batch layouts differ. |
| Mean KL + SmoothL1 | Step 40: **26/60 (43.33%)**. | Original OPSD continuation: 25/60 (41.67%). Mean KL alone: 23/60 (38.33%). |

AIME25 used 12 generations per question in the intervention experiment and 2 per question in the smoothness experiment. Both experiments used one training seed and reused AIME25 for validation. Their scores do not estimate held-out test performance.

## Maximum-gap teacher replacement

The student first samples a response. At each eligible position, compute the full-vocabulary, unclipped forward KL from the teacher to the student, at temperature 1.1. The teacher sees the reference solution. Keep the prefix before the largest eligible gap, let the teacher generate up to five tokens starting at the peak, and let the student continue. Train on the completed mixed trajectory using the original clipped OPSD loss.

- [Complete validation curve data](results/max_gap_k5/aime25_avg_at_12.csv), [training metrics](results/max_gap_k5/training_metrics.csv), [original figure](figures/max_gap_k5.pdf).
- [Paired rollout diagnostics](results/max_gap_k5/paired_intervention.csv). Through step 65, the mean gap in the common post-intervention window rose from 0.13523 to 0.16225 nat. The window gap fell in 39.80% of 2,078 pairs. There were 107 wrong-to-right and 116 right-to-wrong changes.
- [Retrospective token replacement](results/max_gap_k5/token_counterfactuals.csv) and [same-prefix teacher support](results/max_gap_k5/teacher_token_support.csv). These small, selected samples use a separate frozen model snapshot. See the report for the manual error review and the distinction between the main training run and these diagnostic probes.

## Mean KL and local smoothness

All three arms resume from the same original OPSD step-35 checkpoint. Each arm runs five optimizer updates, with 160 student trajectories. Arm A uses original OPSD. Arm B adds 0.05 times the mean unclipped KL across content tokens. Arm C adds the same mean KL and an adjacent-gap SmoothL1 term (beta 0.1, inner weight 0.25) normalized by the number of content tokens. The auxiliary term excludes the first response token, EOS, padding and pairs across mask gaps.

- [Short AIME25 evaluation](results/mean_kl_smooth/aime25_short.csv) and [long AIME25 evaluation](results/mean_kl_smooth/aime25_long.csv). The short evaluation uses an 8,192-token cap and has 51–54 truncations out of 60 generations per checkpoint. Use the 38,912-token long evaluation for the main result.
- [Diagnostic counts](results/mean_kl_smooth/diagnostic_counts.csv), [fixed-window statistics](results/mean_kl_smooth/gap_windows.csv), [aligned per-position training data](results/mean_kl_smooth/training_position_means.csv), [endpoint per-position data](results/mean_kl_smooth/endpoint_position_means.csv.gz), and [original figures](figures/smoothness_diagnostics.pdf).
- The fixed-pool endpoint diagnosis uses 64 questions sampled from the training corpus after excluding recorded training questions. On token positions 2–500, the smoothness arm reduced mean gap by about 9.0% and absolute mean-curve roughness by about 5.7% compared with the baseline. Relative roughness increased by about 3.6%. The three arms share only one question that has one correct and one incorrect sampled trajectory in every arm.

## Code and reproduction

| Directory | Contents |
| --- | --- |
| [sources/max_gap_k5](sources/max_gap_k5) | Original gap-run trainer, rollout logic, evaluator, runner, and CPU preflight |
| [sources/mean_kl_smooth](sources/mean_kl_smooth) | Smoothness pilot, long evaluation, fixed-pool scorer, analysis, and the OPSD code inherited by the pilot |
| [provenance](provenance) | Actual configurations, source hashes, checkpoint-35 hashes, data selection and environment versions |
| [environment](environment) | Environment reference and archived package lists |

See [REPRODUCE.md](REPRODUCE.md) for scripts, input layout, and the distinction between CPU checks and a full GPU run. The publication bundle includes aggregate results and plots. To reproduce the exact continuation, supply the full original checkpoint-35, including optimizer, scheduler, RNG and trainer state. Model weights, training data, checkpoints and raw generations are external inputs.

The underlying OPSD implementation comes from [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD), upstream commit `ae7d251`. File headers are retained.
