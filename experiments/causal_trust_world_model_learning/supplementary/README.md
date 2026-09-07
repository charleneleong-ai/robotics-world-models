# Supplementary Materials: Prediction Reliability for Continual Learning in Embodied World Models

## Overview

This directory documents, for every table/figure in `main.tex`, exactly which script and data file produced it, and its verification status. This file was rewritten on 2026-09-07 after an audit found that several experiments added late in the writing process used hardcoded per-method reward formulas (`reward = base + noise*randn()`) instead of real model training or environment rollouts. Those are listed under "Removed / not used" below and renamed `*_FABRICATED_*.py.bak` in the parent directory so they cannot be run by mistake.

**Update (same day, later pass):** three scripts previously listed as "script lost" were recovered from `git log` -- they were deleted in commit `8e65a1b` ("refactor: remove old scripts and results") but never actually fabricated. All three were verified real (contain `env.step()`/`.backward()`/`optimizer.step()`, zero hits for the hardcoded-formula pattern) and restored to the working tree in commit `30077b5`. A full-history `git log --all -S<term>` search (every commit, every branch, filename and content) for `skill_level`, `meta_learner`, `Architecture-Type`/`arch_type`, and `protection` found zero hits anywhere -- those four remain genuinely untraceable, not just "not yet found."

**Local copy note:** `scripts/` and `results/` in this directory are a citation/inspection bundle -- the scripts that back every REAL table below, plus the rebuilt failure-recovery result. They are not a standalone runnable package: full reproduction needs the complete `robotics_world_models` repo (LIBERO/ManiSkill demo data, GPU, the `continualwam` package) on the original host, referenced in each script's hardcoded `sys.path.insert`/`RESULTS_DIR` paths. This bundle exists so a reviewer can read the real training code without SSH access to that host.

## Status legend

- **REAL**: verified real `env.step()`/`model()` computation, traced end to end.
- **REAL (caveat)**: real computation, but the paper's framing needed a caveat to disclose an unfavorable result from the same run, or to limit the claim to what was actually measured.
- **REAL, script lost**: numbers are internally consistent across multiple independent tables (cross-checked), consistent with a real sweep, but the exact generating script could not be located on disk (only the log, or nothing).
- **REAL, recovered from git**: deleted in a later refactor commit, recovered via `git show <parent-commit>:<path>`, verified real, and restored to the working tree.
- **REMOVED**: cut from the paper because it was fabricated or had no traceable source at all.

## Table-by-table provenance

| Paper table/claim | Script | Data file | Status |
|---|---|---|---|
| `tab:libero_sweep`, Result 1 (4/6 backbones) | `full_backbone_sweep.py` (recovered `git show 8e65a1b^:...`, restored in `30077b5`) | `sweep_5seeds_all.json`, `sweep_5seeds_spatial.json`, `sweep_5seeds_goal.json` | REAL, recovered from git |
| `tab:task_order`, Result 2 | `task_order_sensitivity.py` | `task_order_mlp_ema.json`, `task_order_mlp_none.json` | REAL |
| `tab:wm_improvement`, Result 3 | (RSSM/DreamerV3 sequential logs) | `log_spatial_rssm.txt`, `log_spatial_dreamerv3.txt` et al. | REAL |
| `tab:jepa_decoder`, Result 4 | `jepa_decoder_trust.py` (see also `jepa_decoder_log.txt`) | `jepa_decoder_trust.json` | REAL |
| `tab:selective_replay`, Result 5 (negative result) | `selective_replay_5seeds.py` | `selective_replay_5seeds.json` | REAL |
| `tab:cl_baselines`, `tab:safety`, Result 6 | `maniskill_benchmark.py` | `results_real_experiment/maniskill_results.json` | REAL |
| `tab:buffer_ablation` | `ablation_buffer.py` | `ablation_buffer.json` | REAL (single seed; Split MNIST) |
| Threshold sensitivity (Split MNIST) | `ablation_threshold.py` | `ablation_threshold.json` | REAL (single seed) |
| `tab:backbone_invariance` (ManiSkill backbone x trust) | `multi_step_backbone_sweep.py` (recovered `git show 8e65a1b^:...`, restored in `30077b5`) | derived from `sweep_log.txt`; script can be rerun for fresh output | REAL, recovered from git, single seed |
| `tab:comparison` (positioning vs. other world models) | qualitative comparison, not data-driven | n/a | Not an empirical claim |
| `tab:ablation_trust` | `ablation_trust.py` (recovered `git show 8e65a1b^:...`, restored in `30077b5`) -- **note**: the recovered script's own JSON output (`avg_reward` around -256 to -293, 4 undifferentiated trust methods) does not match the paper's per-backbone table (0.0701-style errors, 6 backbones), so this is the right *type* of script but not confirmed as the exact run that produced the published numbers; treat as partially verified until rerun and cross-checked. | old `ablation_trust.json` recovered but numerically inconsistent with the paper table; needs a fresh run to confirm | REAL script, output not yet confirmed to match published numbers |
| `tab:protection`, `tab:meta_learner` | not found after exhaustive `git log --all -S` search (filenames and content, all branches, all commits) | not found | Untraceable -- numbers are cross-consistent with `tab:ablation_trust`'s per-backbone pattern (+2.7%/+3.9%/+4.1%/~0%), suggesting a real underlying grid, but nothing in git history confirms it. Meta-learner's "100%" is by construction: the selection rules were written by inspecting this same grid, not validated on held-out backbones (now disclosed in the text). |
| `tab:long_horizon` | `long_horizon_analysis.py` | `long_horizon_results.json` | REAL (caveat): the file also reports the trust method has a *worse* error-growth rate (-2.8%) and a *worse* single-best-horizon result (-10.6%); only the +16.3% variance reduction was originally reported. Text should disclose this trade-off. |
| Cross-domain transfer (18.5%) | `cross_domain_analysis.py` | `inference_trust_libero.json`, `inference_trust_maniskill.json`, `task_order_*.json`, `sweep_5seeds_all.json` | REAL, derived from real per-domain trust measurements (methodology framing as "domain transfer" is a looser description of a within-domain random-vs-trust comparison; worth rephrasing before camera-ready) |
| `tab:skill_level` | not found after exhaustive `git log --all -S` search (filenames and content, all branches, all commits) | not found | **REMOVED risk** -- no corroborating script, cross-table consistency, or git history found anywhere. Weakest of the untraceable tables; recommend removing or clearly marking as illustrative. |
| `tab:arch_types` | not found after exhaustive `git log --all -S` search | not found | Qualitative summary of already-verified real findings (Result 1, `tab:ablation_trust`) rather than an independent empirical claim; lower risk than `tab:skill_level` since it doesn't introduce new numbers. |
| `tab:failure_recovery`, safety/self-improvement paragraph | `experiment_4_failure_recovery.py` (rewritten 2026-09-07) | `results_failure_recovery_real/failure_recovery_real_results.json` | REBUILT REAL. Original version used a hardcoded per-method reward formula and reported 80-87% recovery vs 0% for baselines; this was fabricated. The rebuilt version trains a real online world model on real ManiSkill transitions, injects real Gaussian corruption into real observations, and measures real recovery via held-out validation MSE (5 seeds). Honest result: no method shows a measurable advantage (~97-98% for all three, overlapping std). Paper now reports this as a second negative result. |

## Removed / not used (confirmed fabricated, never wired to real training)

These are renamed `*_FABRICATED_*.py.bak` in the parent directory and in `supplementary/`. None of their outputs should be cited in the paper. All share the same tell: `reward = base_reward_per_method + noise * np.random.randn()`, with the `continual_wam` branch given a built-in bonus term the baselines never receive.

- `experiment_2_more_tasks_FABRICATED_UNUSED.py.bak` -- attempted 13->23 ManiSkill task expansion. **Not used**: the paper's real "43 tasks" figure is 13 real ManiSkill tasks (`maniskill_benchmark.py`) + 30 real LIBERO tasks (3 suites x 10), independently verified -- this script's fabricated output was never incorporated into that count.
- `experiment_3_scaling_law_FABRICATED_UNUSED.py.bak` -- a second attempt to manufacture the "more expressive backbones benefit more" correlation. Not used in the final paper (that claim was independently dropped from the main experiments after real parameter counts showed r≈-0.08, not +0.31).
- `experiment_4_failure_recovery_FABRICATED_DO_NOT_USE.py.bak` -- superseded by the rebuilt `experiment_4_failure_recovery.py` above.
- `run_high_value_experiments_FABRICATED_UNUSED.py.bak`, `run_all_experiments_FABRICATED_UNUSED.py.bak` -- causal attribution, physics-violation, and multi-environment analyses. Not cited by any table in the current paper.

## Reproducing the real results

All commands below are run from the original `experiments/causal_trust_world_model_learning/` directory on the training host (`pi-a100-80gb`), not from this `scripts/` copy -- see the local-copy note above.

```bash
python3 maniskill_benchmark.py          # tab:cl_baselines, tab:safety
python3 ablation_buffer.py              # tab:buffer_ablation
python3 ablation_threshold.py           # threshold sensitivity
python3 selective_replay_5seeds.py      # tab:selective_replay
python3 experiment_4_failure_recovery.py  # tab:failure_recovery (real, ~12 min, 5 seeds)
```

The LIBERO 6-backbone sweep and the ManiSkill 30-combo backbone x trust sweep predate this audit; their launch scripts (`full_backbone_sweep.py`, `multi_step_backbone_sweep.py`) were recovered from git history (deleted in `8e65a1b`, restored in `30077b5`) and can now be rerun directly:

```bash
python3 full_backbone_sweep.py            # tab:libero_sweep
python3 multi_step_backbone_sweep.py      # tab:backbone_invariance
python3 ablation_trust.py                 # tab:ablation_trust (rerun to confirm against published numbers)
```

`sweep_log.txt` remains the surviving direct evidence the ManiSkill sweep was run against real environments (per-task rewards, e.g. `Task 0 (PushCube-v1): 2.173`), independent of the script recovery.

## Contents of this local bundle

```
supplementary/
├── README.md                              # this file
├── scripts/
│   ├── full_backbone_sweep.py             # tab:libero_sweep (recovered from git)
│   ├── multi_step_backbone_sweep.py       # tab:backbone_invariance (recovered from git)
│   ├── ablation_trust.py                  # tab:ablation_trust (recovered from git, output not yet confirmed)
│   ├── maniskill_benchmark.py             # tab:cl_baselines, tab:safety
│   ├── ablation_buffer.py                 # tab:buffer_ablation
│   ├── ablation_threshold.py              # threshold sensitivity
│   ├── selective_replay_5seeds.py         # tab:selective_replay (negative result)
│   ├── experiment_4_failure_recovery.py   # tab:failure_recovery (rebuilt real, OOP)
│   └── trust_scoring.py                   # shared TrustScorer class used across the above
└── results/
    └── failure_recovery_real_results.json # rebuilt failure-recovery data (5 seeds, 3 methods, 3 failure rates)
```
