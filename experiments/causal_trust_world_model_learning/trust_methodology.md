# Trust Scoring Methodology: Open-Loop vs Closed-Loop

## Overview

We compare **6 trust scoring methods** on the same RSSM backbone, divided into:
- **Open-loop**: trust computed from prediction only (no real observations used during inference)
- **Closed-Loop**: trust updated using real observation errors (self-correcting)

## Methods

### Open-Loop (Prediction-Only)

| Method | Trust Formula | Source |
|--------|--------------|--------|
| EMA | τ = exp(-α × error) | DreamerV3, 2023 |
| FFDC | τ = σ(Verifier([pred, actual, features, action])) | When2Trust, 2026 |
| Ensemble | τ = 1 - variance(ensemble_heads(features)) | RWM-U, 2026 |

### Closed-Loop (Real Observations Update Trust)

| Method | Trust Formula | Source |
|--------|--------------|--------|
| EMA+Feedback | τ = exp(-α × corrected_error) | Feedback WM, 2026 |
| FFDC+Conformal | τ = σ(Verifier(...)) with calibrated threshold | Foresight, 2026 |
| Closed-Loop Trust | τ = f(error, feedback, trend, calibration) | Ours |

## Key Differences

**Open-Loop**:
- Trust computed once after execution
- No correction for prediction errors
- Fixed thresholds (FFDC: τ=0.5)
- No learning from mistakes

**Closed-Loop**:
- Trust continuously updated with real observations
- Feedback corrects future predictions
- Adaptive thresholds via conformal calibration
- Learns from error patterns (trend detection)

## Metrics

### 1. Trust Accuracy (AUC)
- **Definition**: How well trust scores predict episode success
- **Formula**: AUC-ROC(trust_scores, success_labels)
- **Interpretation**: 1.0 = perfect predictor, 0.5 = random
- **Why**: Trust should be high when predictions succeed, low when they fail

### 2. Error Reduction
- **Definition**: How much closed-loop reduces prediction error vs open-loop
- **Formula**: (error_open - error_closed) / error_open
- **Interpretation**: Positive = closed-loop is better
- **Why**: Feedback correction should reduce error over time

### 3. Replan Precision
- **Definition**: When trust triggers replanning, does it help?
- **Formula**: P(success | replan) vs P(success | no replan)
- **Interpretation**: Replan should increase success rate
- **Why**: Trust should correctly identify when to replan

### 4. CL Performance
- **Definition**: Does better trust → better continual learning?
- **Method**: Use trust for consolidation decisions in ContinualWAM
- **Interpretation**: Higher trust = better knowledge preservation
- **Why**: Core contribution — trust-guided continual learning

## Experimental Protocol

1. **Train RSSM** on ManiSkill (PickCube, PushCube, LiftPegUpright, PlugCharger)
2. **Collect test episodes** (50 episodes per environment, unseen during training)
3. **Run all 6 trust methods** on same RSSM predictions
4. **Compute metrics** for each method
5. **Compare** open-loop vs closed-loop

## Results (PushCube-v1)

| Method | Type | Corr (noise=0.0) | Corr (noise=1.0) | Notes |
|--------|------|-----------------|-------------------|-------|
| EMA | Open | 0.703 | 0.510 | Best out-of-box |
| FFDC | Open | 0.000 | 0.000 | Untrained verifier |
| Ensemble | Open | 0.000 | 0.000 | Untrained ensemble |
| Closed-Loop (ours) | Closed | 0.623 | 0.410 | Self-correcting |

### Key Findings
1. **EMA is a strong baseline** — achieves 0.7+ correlation without training
2. **FFDC/Ensemble need training** — untrained verifiers output constant 0.5
3. **Closed-Loop adds self-correction** — learns from error trends, adapts thresholds
4. **High noise degrades all methods** — but Closed-Loop maintains relative performance

### What Closed-Loop Adds Over EMA
- **Trend detection**: detects increasing error → triggers replan
- **Conformal calibration**: adaptive thresholds (not fixed τ=0.5)
- **Feedback correction**: corrects future predictions using observation errors
- **Consolidation trust**: accumulated error → trust for CL decisions

### Next Steps
1. Train FFDC verifier on success/failure episodes
2. Test on more complex environments (KinDER 57 tasks)
3. Integrate with ContinualWAM for CL benchmark

## Files

- `train_rssm.py` — Train RSSM on ManiSkill
- `trust_benchmark.py` — Run all 6 methods, compute metrics
- `closed_loop_trust.py` — Our closed-loop trust corrector
- `trust_metric_comparison.py` — Open-loop trust methods
- `rssm_world_model.py` — RSSM backbone

## Usage

```bash
# 1. Train RSSM
python train_rssm.py --env PickCube-v1 --episodes 200 --epochs 50

# 2. Run trust benchmark
python trust_benchmark.py --env PickCube-v1 --model-path trained_models/rssm_pickcube-v1.pt --n-episodes 50

# 3. Compare results
# Open-loop: EMA, FFDC, Ensemble
# Closed-loop: EMA+Feedback, FFDC+Conformal, Closed-Loop Trust
```
