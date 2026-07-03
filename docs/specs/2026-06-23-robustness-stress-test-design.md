# Robustness stress test — privileged-model brittleness vs world-model robustness

**Status:** harness built + deployed + smoke-validated (2026-06-24). The clean TD-MPC2 PegInsertion run is **in-flight and ~complete** on the A100; an armed watcher → orchestrator runs both noise sweeps + the divergence plot the moment `final.pt` lands, then hands off to PlugCharger. **PR:** robustness-stress-test.

## Why — the most compelling version of the argument

The nominal 3-way (TD-MPC2 0.84 · classical 0.75 · PPO 0.00) shows the world model *matches* a privileged classical planner. But it only *hints* at the real point. The sharp version: **degrade the privileged information the classical planner relies on (ground-truth peg/hole geometry + pose) and watch it collapse, while the reactive world-model policy holds up.** That turns "they're comparable" into "the world model is *robust* where the planner is *brittle*" — the defensible thesis of model-based learning.

## Design

Sweep a perturbation magnitude σ and measure success rate for each method:

- **Geometry/pose noise:** inject Gaussian noise (σ ∈ {0, 5, 10, 15, 20 mm} on position; small on orientation) into the **peg/hole pose** the planner reads (`env.peg.pose`, the OBB centre, the goal pose). The open-loop plan then targets the *wrong* pose → misses the hole.
- **(Variant) observation noise:** add the same noise to the *state observation* both methods consume, so it's a fair like-for-like input-noise test.

**Expected shape:** classical (open-loop, plans once to the noisy pose) **collapses** as σ grows — it can't correct mid-execution. A trained TD-MPC2 policy (reactive, re-plans every step via MPC, learned contact recovery) should **degrade far more gracefully** — the success-vs-σ curves diverge. PPO is already 0.00, so it's out.

## Metrics

`success_rate` vs σ, per method (100 episodes each σ). Headline = the **divergence point** + the area between the classical and world-model curves. Plus a side-by-side rollout video at a mid σ (planner missing vs policy recovering).

## The gate (resolved)

| arm | status | how |
|---|---|---|
| **Classical under noise** | ✅ built + smoke-validated | `classical_noise_sweep.py` perturbs only the *perceived* peg/goal pose the planner aligns to (grasp stays on the true peg); physics object is unmoved → open-loop miss. Runs under `/workspace/mplib_venv` (numpy 1.26 — the box's numpy 2.x SIGSEGVs mplib). Smoke (n=5): σ=0 → 0.40, σ=20 mm → **0.00**. |
| **World model under noise** | ✅ built, pending `final.pt` | `wm_noise_eval.py` replicates the `evaluate.py` loop + injects `obs += (σ/1000)·randn` before each `agent.act`. Pipeline validated end-to-end against the smoke checkpoint. The cancelled clean-2M run is **revived and ~complete**, so the real checkpoint is hours away, not gone. |

The two arms are no longer sequenced by hand: a watcher waits for `final.pt`, then the orchestrator runs the full sweep before yielding the box.

## Implementation (built — `experiments/stress_test/`)

- `classical_noise_sweep.py` · `wm_noise_eval.py` — the two arms above (typer CLIs, `--sigmas-mm 0,5,10,15,20`).
- `plot_divergence.py` — success-vs-σ curves for both methods → `divergence.png`.
- `stress_then_plugcharger.sh` — orchestrator (deployed `/workspace/`): WM eval → classical sweep → plot, each best-effort, then launches PlugCharger TD-MPC2 (W&B on). Invoked by the stress-first watcher on `final.pt`.

## Queue plan — active

**Full, chained, autonomous:** the in-flight clean TD-MPC2 run completes (~done) → `final.pt` saved → watcher fires the orchestrator → both noise sweeps + divergence plot → PlugCharger training. No manual steps; the watcher *holds* PlugCharger behind the stress test so it can't jump the queue.

## Out of scope / sub-options

- **cuRobo** (GPU-native planner) as a *stronger* classical point — needs a CUDA build (not installed); orthogonal to the robustness story (it's "a better planner," not "a more robust paradigm").
