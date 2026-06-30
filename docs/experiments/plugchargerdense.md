# PlugChargerDense — TD-MPC2 at the edge of its reach

**Status:** done (2026-06-27). Honest null *with signal* — eval success 0, but the policy grazes success ~10× in exploration. Motivates the demo-augmented follow-up.

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

## Next step — demonstrations
The near-misses are the textbook signature for **demonstration-augmented RL**: the agent reaches success states but too rarely to learn from. Seed TD-MPC2's buffer with successful mplib-solver trajectories (the solver already gives ~0.7), then continue online RL. De-risk by generating the demos first and confirming `eval/success_once` lifts off 0 early. (Blocked at time of writing on a wedged-CUDA GPU instance — infra, not method.)
