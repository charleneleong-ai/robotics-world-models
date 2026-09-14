# Supplementary Materials: Whose Reliability? When Self-Loss Weighting Reduces Forgetting in Embodied Continual Learning, and World-Model Prediction Error Does Not

## Overview

This directory maps every table and figure in `main.tex` to the script that produced it and the data file it was read from. All scripts live in `experiments/causal_trust_world_model_learning/` of the `robotics-world-models` repository (branch `feat/cl4fmagents-provenance`, PR #32, stacked on PR #31); paths below are relative to that directory. Every experiment ran on a single shared NVIDIA A100 80GB; long runs were launched as detached daemons (`setsid nohup python3 -u ...`) and their logs are in `logs/`.

Statistics are computed from the result files: paired t-tests at the unit stated in the paper (seed, or ordering with seeds averaged within ordering), Holm-Bonferroni correction for the Result 1 grid, Monte Carlo null and power checks (`power_mc.py`), and matched-count controls where a schedule or detector is evaluated.

## Evaluation protocols

- **LIBERO task-order protocol** (Results 2-4, 10, the FLOW experiments): trains on and scores each task's demonstrations (5 per task), so error measures retention of trained behaviour. `interference_signals.py` with `HELDOUT=1` repeats the key arms with 10 demonstrations per task, 8 for training and 2 held out (Appendix, `tab:heldout`).
- **ManiSkill classification harness** (Result 6, `tab:cl_baselines`, `tab:flow_si`): `cl_baselines_heldout.py` evaluates on a held-out 20% of each task's samples; `cl_baselines_full_rerun.py` / `flow_si.py` evaluate on the training samples (reported alongside).
- **Triggered consolidation, boundary detection, world-model self-loss**: held-out splits inside the scripts (`EVAL_FRAC`, `HOLDOUT`).

## Table-by-table provenance

| Paper table / figure | Script | Data file |
|---|---|---|
| `tab:libero_sweep`, Result 1 (LIBERO 6x4x3 grid, 5 seeds) | `full_backbone_sweep.py`; statistics `libero_significance.py`, `gen_libero_table.py` | `sweep_5seeds_all.json`, `sweep_5seeds_spatial.json`, `sweep_5seeds_goal.json` |
| Result 1, n=9 replication | `extend_seeds_all18.py` | `extended_seeds_5to8_all18.json` |
| `tab:maniskill_sweep`, Result 1 (ManiSkill 6x4, 5 seeds) | `maniskill_backbone_sweep.py`; `gen_maniskill_table.py` | `results_maniskill_backbone_sweep/aggregated.json` |
| `sec:power` (minimum detectable effects) | `power_mc.py` | derived from the sweep files above |
| `tab:task_order`, Result 2 (10 orderings x 3 seeds, 4 backbones) | `task_order_sensitivity.py` | `task_order_{mlp,rssm,jepa,dreamerv3}_ema.json`, `task_order_mlp_none.json` |
| Result 2, JEPA fresh-ordering replication | `jepa_replication.py` | `results_per_param_consolidation/jepa_replication.json` |
| `tab:signal_variants` (current-obs / true-next-obs / warmed-state scoring) | `warmed_trust.py` | `results_per_param_consolidation/warmed_trust_{mlp_rssm,jepa_dreamerv3}.json` |
| `sec:cross_suite` (Object -> Spatial -> Goal, trust weighting) | `cross_suite_shift.py` | `results_cross_suite_shift/results.json` |
| `tab:per_param`, Result 4 (policy A and world-model C consolidation) | `per_param_consolidation.py` | `results_per_param_consolidation/results_A.json`, `results_C.json` |
| `tab:gdumb_curation`, Result 5 (replay; trust-guided eviction n=5, n=9) | `selective_replay_5seeds.py`; `trust_guided_gdumb.py`, `trust_guided_gdumb_n9.py` | `selective_replay_5seeds.json`; `trust_guided_gdumb_results.json`, `trust_guided_gdumb_n9_results.json` |
| `tab:cl_baselines`, Result 6 (held-out 20%) | `cl_baselines_heldout.py` | `results_per_param_consolidation/cl_baselines_heldout.json` |
| Result 6, training-sample accuracies; constant-trust ablation | `cl_baselines_full_rerun.py`; `trust_weighting_ablation.py` | `results_cl_baselines_full/`; `trust_weighting_ablation_results.json` |
| `tab:threshold_sweep`, Result 7 (calibrated gating, corruption test) | `trust_rejection_real.py` | `results_trust_rejection/results.json` |
| `tab:relative_trust_b`, Analysis (z-scored scorer) | `relative_trust.py` | `results_relative_trust/results.json` |
| `fig:trust_hist` | `trust_histogram_dump.py` | `trust_histogram_stats.json`, `fig_trust_histogram.pdf` |
| `tab:failure_recovery`, Result 7 | `experiment_4_failure_recovery.py` | `results_failure_recovery_real/failure_recovery_real_results.json` |
| `tab:jepa_decoder`, Result 8 (decoder-aware trust; on-task world-model error) | `archive/jepa_decoder_trust.py`; sequential logs | `jepa_decoder_trust.json`, `jepa_decoder_log.txt`; `log_spatial_rssm.txt`, `log_spatial_dreamerv3.txt` |
| `tab:boundary`, `fig:boundary_trace`, Result 9 (CUSUM detection, chance baseline) | `boundary_cusum.py` | `results_boundary_cusum/results.json`, `chance_baseline.json`, `fig_boundary_trace.pdf` |
| `tab:triggered`, Result 9 (triggered consolidation vs oracle / periodic / random) | `triggered_consolidation.py` | `results_triggered_consolidation/results.json` |
| `tab:triggered_flow` (CUSUM-triggered FLOW; run by a parallel session) | `triggered_flow.py` | `results_triggered_flow/results.json` |
| `tab:flow`, `tab:flow_main`, Result 3 (FLOW vs world-model-error form vs inverse trust) | `flow_weighting.py` | `results_per_param_consolidation/flow_weighting_{mlp,rssm,jepa,dreamerv3}.json` |
| `tab:flow_controls` (constant / shuffled / reverse, fresh orderings) | `flow_controls.py` | `results_per_param_consolidation/flow_controls_{mlp,rssm}.json` |
| `tab:flow_ewc` (FLOW + EWC, raw and mean-normalised weights) | `flow_ewc.py` | `results_per_param_consolidation/flow_ewc_{mlp,rssm}.json` |
| `tab:flow_robust` (temperature, suites, lambda 10-3000, 10 orderings, cross-suite, 20 demos) | `flow_robustness.py {temp,suites,lambda,lambda_fine:30,300,lambda_fine:3000,scale,cross_suite,demos}` | `results_per_param_consolidation/flow_robustness_*.json` |
| `tab:flow_mechanism` (per-sample probe; rank-based FLOW) | `flow_suite_mechanism.py`; `flow_robustness.py rank`; `flow_weight_diagnostic.py` | `flow_suite_mechanism.json`, `flow_robustness_rank.json`, `flow_weight_diagnostic.json`, `flow_weight_diagnostic_suites.json` (all under `results_per_param_consolidation/`) |
| `tab:flow_si` (FLOW on FT / EWC / SI in the ManiSkill harness) | `cl_baselines_heldout.py` (held-out); `flow_si.py` (training samples) | `cl_baselines_heldout.json`; `flow_si_maniskill.json` |
| `tab:flow_second` (ManiSkill expert-demo BC; world-model self-loss on ManiSkill and KinDER) | `flow_maniskill_bc.py`; `flow_world_model.py {maniskill,kinder}` with and without `NORMALISE=1` | `flow_maniskill_bc_ord{0_1_2,3_4_5}.json`; `flow_world_model_{maniskill,kinder}{,_norm}.json`, `wm_stream_{maniskill,kinder}.npz` (cached random-action streams) |
| `tab:heldout`, Result 10 (ER / MIR / A-GEM / conflict / lookahead / ensemble; both protocols) | `interference_signals.py spatial <orderings>`, `interference_signals.py cross`, with and without `HELDOUT=1` | `interference_spatial{,_heldout}_ord{0_1,2_3,4_5}.json`, `interference_cross{,_heldout}.json` |
| `sec:meta_learner` | `meta_learner_loo.py` | derived from the Result 1 grid |
| `tab:compute` | wall-clock from `logs/` | -- |

## Reproducing

```bash
cd experiments/causal_trust_world_model_learning
python3 flow_weighting.py mlp                      # Result 3, one backbone
python3 flow_ewc.py mlp                            # FLOW + EWC
python3 flow_robustness.py lambda                  # one robustness sub-experiment
HELDOUT=1 python3 interference_signals.py spatial 0,1   # Result 10, held-out, two orderings
python3 cl_baselines_heldout.py 5                  # Result 6, held-out
NORMALISE=1 python3 flow_world_model.py maniskill 6 3   # world-model self-loss
```

LIBERO demonstrations are read from the suite directories in `task_order_sensitivity.SUITE_DIRS`; ManiSkill demonstrations for `flow_maniskill_bc.py` are the official motion-planning sets (`python3 -m mani_skill.utils.download_demo <task>`); KinDER environments are instantiated by their registered ids and the benchmark class's synthetic-data fallback is disabled.

## Contents of this local bundle

`README.md` (this file), `scripts/` (a copy of every script named above) and `results/` (every result file named above except the two `.npz` stream caches). It is an inspection copy; full reproduction needs the repository, the demonstration data, and the GPU host.
