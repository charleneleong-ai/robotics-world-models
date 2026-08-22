# Trust Scoring for WAMs: Metrics and Enhancement Opportunities

## Best Trust Metrics (Ranked by Empirical Evidence)

### 1. Action-State Consistency (Best Overall)
**Source:** "Is the Future Compatible?" (2026)
**Metric:** Compare predicted future observations vs real observations after executing predicted actions
**Performance:** AUC 0.77-0.88 for predicting success/failure
**Formula:** c_t = similarity(o_{t+Δ}, ô_{t+Δ}) where ô is predicted, o is actual

### 2. Forward-Inverse Cycle Consistency (Best for Self-Improvement)
**Source:** World Action Verifier (WAV, 2026)
**Metric:** Decompose into state plausibility + action reachability
**Performance:** 2x sample efficiency, +22% policy performance
**Key insight:** Verifying via sparse inverse process is easier than dense forward generation

### 3. Ensemble Disagreement (Best for Epistemic Uncertainty)
**Source:** RWM-U (2026)
**Metric:** Variance across bootstrap ensemble predictions
**Performance:** Strong correlation with prediction error, enables uncertainty-penalized RL
**Key insight:** Epistemic uncertainty tracks compounding error in autoregressive rollouts

### 4. Feedback Correction (Best for OOD Robustness)
**Source:** Feedback World Model (2026)
**Metric:** Compare predicted vs observed next state, maintain lightweight feedback state
**Performance:** 76.4% reduction in prediction error, +30% OOD success rate
**Key insight:** Execution itself provides natural trust signal

### 5. Adaptive Execution Verification (Best for Deployment)
**Source:** When2Trust / FFDC (2026)
**Metric:** Joint reasoning over predicted actions, visual dynamics, real observations, language
**Performance:** -69% forward passes, -34% execution time, +2.5% success rate
**Key insight:** Trust determines action chunk size adaptively

### 6. Conformal Prediction Calibration (Best for Safety)
**Source:** Uncertainty-aware Latent Safety Filters (2025)
**Metric:** Calibrated uncertainty thresholds via conformal prediction
**Performance:** Reliable OOD detection for safety-critical control
**Key insight:** Formal coverage guarantees for trust thresholds

### 7. Action Dependence (Best for Diagnostics)
**Source:** WAMProbe (2026)
**Metric:** Do predicted futures separate across candidate actions?
**Performance:** Catches "action-agnostic" and "wrong-direction" failures
**Key insight:** Visual quality != action quality

---

## How Trust Scoring Could Enhance Each WAM Architecture

### For Video-Based WAMs (DreamZero, GE, Cosmos 3, Flex-pi)
| Enhancement | Method | Expected Gain |
|-------------|--------|---------------|
| Adaptive execution | FFDC verifier | -69% compute, +2.5% success |
| OOD detection | Feedback correction | +30% OOD success |
| Test-time selection | Action-state consistency | +1-3% success |
| Training signal | Forward-inverse cycle | +22% policy performance |

### For Latent WAMs (V-JEPA, LaWAM, CoWVLA)
| Enhancement | Method | Expected Gain |
|-------------|--------|---------------|
| Planning reliability | ACPC diagnostic | Better plan selection |
| Representation quality | Bisimulation metrics | More robust features |
| Trust-weighted MPC | Consistency scoring | Safer planning |

### For MoT-Based WAMs (Motus, OpenWAM, Flex-pi)
| Enhancement | Method | Expected Gain |
|-------------|--------|---------------|
| Stream selection | Per-stream trust | Dynamic modality weighting |
| Expert routing | Trust-gated MoE | Better specialization |
| Multi-modal fusion | Cross-modal consistency | More coherent predictions |

### For All WAMs: Continual Learning via Trust
| Enhancement | Method | Expected Gain |
|-------------|--------|---------------|
| Trust-weighted EWC | Our ContinualWAM | Modest CL improvement |
| Trust-prioritized replay | Prediction error sampling | Better data efficiency |
| Trust-based consolidation | Adaptive regularization | Reduced forgetting |

---

## The Missing Piece: Trust for CL in Existing WAMs

**None of the 24 WAM architectures use trust scoring for continual learning.** They use it for:
- Test-time action selection (FFDC, WAV)
- OOD detection (RWM-U, conformal prediction)
- Planning reliability (ACPC, WAMProbe)
- Self-improvement (WAV, feedback correction)

**But NOT for:**
- Deciding when to consolidate knowledge (our contribution)
- Weighting regularization during sequential task learning
- Prioritizing replay based on trust signals
- Protecting high-trust parameters from forgetting

This is ContinualWAM's unique position: using trust scoring for **continual learning consolidation**, not just test-time reliability.

---

## Recommended Trust Metric Stack for Production WAMs

1. **Primary:** Action-State Consistency (simple, effective, value-free)
2. **Secondary:** Ensemble Disagreement (epistemic uncertainty)
3. **Safety:** Conformal Prediction Calibration (formal guarantees)
4. **Adaptive:** FFDC Verifier (dynamic action chunk sizing)
5. **CL:** Trust-Weighted EWC (our contribution for continual learning)

---

## Key Insight: Trust is Underutilized

Current WAMs use trust for **reactive** decisions (select actions, detect OOD, trigger replan).
ContinualWAM uses trust for **proactive** decisions (consolidate knowledge, protect parameters, prioritize replay).

The combination of both reactive and proactive trust would be powerful:
- **Reactive:** FFDC + Feedback Correction at deployment time
- **Proactive:** Trust-Weighted EWC + Trust-Prioritized Replay during training

This suggests a natural extension: ContinualWAM + FFDC = Trust-Aware WAM that both learns continually AND deploys reliably.
