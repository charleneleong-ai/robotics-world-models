# Robustness stress test — privileged-model brittleness vs world-model robustness

**Status:** specced (2026-06-23). World-model arm **gated on a TD-MPC2 checkpoint** (none exists yet). **PR:** robustness-stress-test.

## Why — the most compelling version of the argument

The nominal 3-way (TD-MPC2 0.84 · classical 0.75 · PPO 0.00) shows the world model *matches* a privileged classical planner. But it only *hints* at the real point. The sharp version: **degrade the privileged information the classical planner relies on (ground-truth peg/hole geometry + pose) and watch it collapse, while the reactive world-model policy holds up.** That turns "they're comparable" into "the world model is *robust* where the planner is *brittle*" — the defensible thesis of model-based learning.

## Design

Sweep a perturbation magnitude σ and measure success rate for each method:

- **Geometry/pose noise:** inject Gaussian noise (σ ∈ {0, 5, 10, 15, 20 mm} on position; small on orientation) into the **peg/hole pose** the planner reads (`env.peg.pose`, the OBB centre, the goal pose). The open-loop plan then targets the *wrong* pose → misses the hole.
- **(Variant) observation noise:** add the same noise to the *state observation* both methods consume, so it's a fair like-for-like input-noise test.

**Expected shape:** classical (open-loop, plans once to the noisy pose) **collapses** as σ grows — it can't correct mid-execution. A trained TD-MPC2 policy (reactive, re-plans every step via MPC, learned contact recovery) should **degrade far more gracefully** — the success-vs-σ curves diverge. PPO is already 0.00, so it's out.

## Metrics

`success_rate` vs σ, per method (100 episodes each σ). Headline = the **divergence point** + the area between the classical and world-model curves. Plus a side-by-side rollout video at a mid σ (planner missing vs policy recovering).

## ⚠️ The gate (honest scoping)

| arm | feasible now? | why |
|---|---|---|
| **Classical under noise** | ✅ yes | clean pose-injection point in the solution; runs in the isolated `mp` env (numpy<2), ~minutes |
| **World model under noise** | ❌ **gated** | needs a trained **TD-MPC2 PegInsertion checkpoint** to eval (via `tdmpc2/evaluate.py`). The killed runs saved **none** (only a PickCube smoke `final.pt`). |

So the **world-model arm requires first getting a clean TD-MPC2 PegInsertion run to completion** (which saves `final.pt`) — i.e. reviving the clean-2M sequential run we cancelled, when the GPU is free. The stress test is therefore **chained behind a clean TD-MPC2 run**. Without it, only the classical-collapse half is measurable (half the argument — informative, but not the contrast).

## Implementation

- `experiments/stress/classical_noise_sweep.py` — copies the `peg_insertion_side` solve logic, injects pose noise at level σ, runs N episodes per σ, writes `results.json` (σ → success). Runnable now (mp env).
- World-model arm — `tdmpc2/evaluate.py checkpoint=<final.pt>` wrapped to add obs noise; **pending the checkpoint**.

## Queue plan (options for the operator)

1. **Classical-only now:** queue the classical-noise sweep (GPU-free auto-launch, mp env) → the collapse curve lands in minutes. World-model arm added later.
2. **Full, chained:** when the GPU frees → clean TD-MPC2 run to completion (~24h, also yields the converged number + checkpoint) → then both noise sweeps → the full divergence plot. One hands-off pipeline, but it re-introduces the multi-day TD-MPC2 run we cancelled.

## Out of scope / sub-options

- **cuRobo** (GPU-native planner) as a *stronger* classical point — needs a CUDA build (not installed); orthogonal to the robustness story (it's "a better planner," not "a more robust paradigm").
