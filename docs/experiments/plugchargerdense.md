# PlugChargerDense — TD-MPC2 at the edge of its reach

**Status:** done (2026-07-03). **Exhaustive null** — TD-MPC2 can't crack PlugCharger's two-prong sub-mm insertion in 2M under *any* tried approach (from-scratch → reward v1–v4 → demos → persistent demo-RL → staged-reward demo-RL, all eval success 0). A task-difficulty wall, not a reward gap. Bounds model-based RL's reach alongside StackCube solved (1.0). See the final result table below.

## Why a dense reward at all
Stock `PlugCharger-v1` ships **sparse-only** (`SUPPORTED_REWARD_MODES = ["none", "sparse"]` — the authors deliberately excluded dense). With no shaping, TD-MPC2 gets zero signal until a chance success it never finds from scratch (observed: flat `R=0` for 6 h). To make the task RL-tractable we wrote a staged dense reward — [`experiments/envs/plugcharger_dense.py`](../../experiments/envs/plugcharger_dense.py) (`PlugChargerDense-v1`): reach → touch → grasp → insert, with an **angular** alignment term the round-peg PegInsertion reward never needed (success here requires `obj_to_goal_angle <= 0.2`).

## The reward-engineering arc (the actual lesson)
| ver | reward | outcome |
|---|---|---|
| **v1** | flat staged weights (reach 1 / grasp 1 / align 3 / seat 5) | **farmed** — R climbed 3.4 → 104 over 1M while eval success stayed 0. The seat term maxed in a non-success hover band; the policy banked intermediate reward instead of inserting. |
| **v2** | completion-dominant: precursors → 0.1, one success-tracking term `1-tanh(8·dist+4·angle)`, success bonus 25 | farm-proof (R stayed under the v1 ceiling), but a stub run (superseded before it trained). |
| **v3** | v2 + a 0.1 *touch* precursor (contact-force on either finger) | the run reported below. Smoke-validated: contact API works, reward fires every step. |

The v1→v3 progression is the engineering story: a reward-hacking exploit, diagnosed at the ~1M checkpoint and fixed with a principled redesign — verified *farm-proof*, not just "reward goes up."

## Result — run [`aluvndm9`](https://wandb.ai/chaleong/wm-manip/runs/aluvndm9) (v3, 2M steps)
| metric | value | meaning |
|---|---|---|
| `eval/success_once` (deterministic, held-out) | **0 / 40** | never solved it on the real test |
| `train/success_once` (exploration) | **0.03 at 10 / 312 points** | 1 of 32 parallel envs *momentarily* inserted, ~10× across 2M |
| `train/success_at_end` | ≈ 0 | even those sparks were transient, not held |
| `eval/return` (reward) | 0 → ~20 | climbs cleanly, **no farming** (v2/v3 fix holding) |

**Read it precisely:** the policy *grazes* success during noisy exploration — a single env touches the success state a handful of times — but cannot reproduce it deterministically (eval 0) or hold it. **Not solved, but not hopeless:** the sparks prove the reward and task are learnable in principle; the policy just can't bootstrap a reliable skill from 1-in-32, momentary successes.

## Contrast — StackCube *is* solved
Same algorithm, budget (2M), compute, and reward rigor → [`tdmpc2-stackcube`](https://wandb.ai/chaleong/wm-manip/runs/th868utn) reaches **eval success 1.0**. So PlugCharger's 0 is not a method or implementation failure — it's the task's intrinsic two-prong insertion tolerance (even the *privileged* mplib classical planner only hits ~0.7). This pins where model-based RL's reach ends on a clean difficulty gradient: PickCube (trivial) → PegInsertion (0.84) → StackCube (1.0) → PlugCharger (0.0).

## The exhaustive null (final)

We pulled every legitimate lever. Each was implemented, validated, and run to a full 2M steps. **Eval success stayed at exactly 0 for all of them:**

| approach | eval success (2M) | note |
|---|---|---|
| from-scratch, sparse | 0 | no signal (stock env has no dense reward) |
| dense reward v1 | 0 | *farmed* — R→104 while success 0 (reward-hacking, fixed) |
| dense reward v2–v4 | 0 | farm-proof, completion-dominant, "reward-the-hold" | grazed success ~10× in exploration, never reproduced |
| + demos, seeded in online buffer (23) | 0 | too dilute (0.5% of a 1M circular buffer) + evicted |
| + demos, **persistent buffer**, 25% ratio (112) | 0 | DDPGfD-style; demos never dilute or evict — `plugcharger-demoRL` |
| + demos + **v5 staged align-then-insert reward** | 0 | geometry-validated decomposition — `plugcharger-v5-demoRL` |

**The decisive data point:** the v5 staged reward — which decomposes the approach a single pose-distance term can't express, and which a solver demo replayed through it earns monotonically to a peak at success — was **optimized by the policy to `train/reward = 38.9`** (vs v4's 17.2) while `eval/success_once` stayed **0/40**. A well-shaped reward, driven hard, still could not produce the terminal maneuver. So this is **not a reward-shaping gap** — it's a **task-difficulty wall**: PlugCharger's two-prong, sub-mm, sub-0.2-rad keyed insertion is beyond TD-MPC2 at this budget, with or without demonstrations.

**Verdict.** Paired with **StackCube solved (1.0)** under the identical method, budget, and rigor, this cleanly bounds where model-based RL's reach ends on the contact-difficulty gradient: PickCube (trivial) → PegInsertion (0.84) → StackCube (1.0) → **PlugCharger (0.0, even the privileged classical planner only ~0.7)**. The value here is the *rigor of the null*, not a headline number — every fixable cause was ruled out.
