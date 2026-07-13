# Planning-augmented action selection on the DreamerV3 world model

**Status:** 📋 SPEC — queued. **Date:** 2026-07-13.

## Why

The [PlugCharger paradigm study](../experiments/plugchargerdense.md) landed a sharp negative: DreamerV3's **reactive imagination actor** never completes a multi-stage manipulation sequence. On trivial PickCube it gets 0 success; a demo warm-start lifts grasp 0.02 → 0.81 and a *persistent* non-evicting demo buffer holds grasp 0.84 across **6.2M steps** — yet `success` stays **0.00**. The world model learns cleanly (losses converge, imagined rollouts are coherent); the *policy* is the failure.

The study attributes this to **how the action is chosen**: TD-MPC2 solves the identical env + reward because it does short-horizon **MPC planning**, whereas Dreamer's amortized actor settles at each dense-reward local optimum. That contrast is currently *cross-model* (different world models). This spec makes it **same-model**: hold DreamerV3's exact learned world model fixed and swap only the action-selection mechanism — reactive actor → planner. It's the one experiment that isolates the causal variable.

## Hypothesis

Planning over the *same* trained DreamerV3 RSSM will complete the grasp→lift→place sequence on PickCube (`success` > 0.5) where the reactive actor stalls at 0 — because search over action *sequences* escapes the per-stage local optima a reactive policy cannot. If true, it confirms "the world model is fine; the reactive actor is the bottleneck" and gives the generative-WM ladder its real numbers. If planning *also* fails, the bottleneck is deeper (the learned reward/dynamics near the success manifold), which is itself a stronger claim.

## Method

Reuse the trained world model; replace deployment-time action selection with a latent-space planner.

- **World model:** the existing JAX DreamerV3 RSSM (`~/dreamerv3_jax`, env `jax_dreamer`) — retrain PickCube state-obs (~hours at 255 steps/s; the old checkpoints were reclaimed for disk), or checkpoint-and-freeze a fresh run.
- **Planner:** a **CEM or MPPI** loop at each env step — sample `N` action sequences over horizon `H`, roll each through the RSSM latent dynamics + reward head (no simulator calls — pure imagination), score by predicted `λ`-return + the learned value at the leaf, execute the first action of the best sequence, replan next step. Optionally seed the sampling distribution from the trained actor (TD-MPC2-style prior) to cut planning cost.
- **Integration surface:** danijar's RSSM exposes the latent step + reward/continue/value heads the planner needs; the build is a CEM/MPPI loop calling `imagine`-style rollouts, plus a policy shim that returns the planned action instead of the actor's. danijar's code is **not** set up for MPC, so this is a real implementation, not a config flag.

## Gate — PickCube first (same discipline as the study)

The blocker showed up on trivial PickCube, so that's the gate. Instrument (already in the adapter) `epstats/log/is_grasped/{avg,max}` and `epstats/log/success/{avg,max}`.
- **GO:** planning gets `success/avg` > 0.5 with grasp holding → the reactive actor *was* the bottleneck → run the PegInsertion / StackCube / PlugCharger ladder with the planner, completing the generative-WM column.
- **NO-GO:** planning also stays ~0 → the wall is in the learned model near the success manifold, not the actor → bank that (deeper) finding.

Cost axis: planning multiplies per-step compute by `N×H` model rollouts — report planning wall-clock vs the reactive actor's, since that's the practical tradeoff of the approach.

## Honest framing

This is **confirmation, not discovery** — the study's finding already stands on the TD-MPC2 contrast plus the exhausted demo levers. The value here is making it *same-world-model airtight* and (if GO) unlocking real ladder numbers for the generative-WM column. It is a **multi-day build** (a planner over danijar's RSSM), so it earns its slot only if the airtight version or the ladder numbers are wanted — otherwise the study is complete as-is.

## Deliverable

- A `plan_action` module (CEM/MPPI over the DreamerV3 RSSM) + a deployment shim, in the `jax_dreamer` setup on `pi-a100-80gb`.
- PickCube gate result: grasp + success under planning vs the reactive baseline (grasp 0.84 / success 0.00), with planning-vs-reactive wall-clock.
- If GO: the generative-WM ladder row (PegInsertion / StackCube / PlugCharger) folded into [`plugchargerdense.md`](../experiments/plugchargerdense.md).

## Reusable assets (already on `pi-a100-80gb`)

`~/dreamerv3_jax/embodied/envs/maniskill.py` (adapter with control-mode arg + grasp/placed/static logging), `~/setup_jax_dreamer.sh`, `~/launch_*.sh`; env `jax_dreamer` (py3.11, jax[cuda12] 0.4.33, gymnasium 0.29.1, ManiSkill 3.0.1 — no shim). W&B project `chaleong/wm-manip`.
