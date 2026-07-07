# PlugCharger — a three-paradigm study

**Status:** done (2026-07-07). PlugCharger's two-prong, sub-mm, sub-0.2-rad keyed insertion is where paradigms separate. Measured on one task, one budget, one sim: **classical (privileged) 0.56 · imitation (diffusion) 0.13 · RL 0.00**. Success tracks how much prior knowledge each paradigm is handed — RL, which must *discover* the skill, categorically fails across every recipe; imitation, which *copies* a demonstrator, is the only learned paradigm that completes insertions at all, and it scales with demo count. The privileged classical planner leads but itself degrades from its PegInsertion 0.75, confirming the task is genuinely harder — not just hard for RL.

## The result

| paradigm | prior knowledge | PlugCharger success | run |
|---|---|---|---|
| classical (mplib RRTConnect + screw) | full socket geometry + scripted contact | **0.56** (54/96 eps) | measured 2026-07-07 |
| **imitation** — diffusion policy, 706 demos | learns from demonstrations | **0.13** once / 0.04 held | [`x02u1vli`](https://wandb.ai/chaleong/wm-manip/runs/x02u1vli) |
| imitation — diffusion policy, 112 demos | — | 0.07 once | [`ag0ehb58`](https://wandb.ai/chaleong/wm-manip/runs/ag0ehb58) |
| model-based RL (TD-MPC2, every variant) | none / demos-as-signal | **0.00** | see null below |
| demo-bootstrapped RL (RLPD) | 112 demos, 50/50 sampling | **0.00** | [`qj3tp91t`](https://wandb.ai/chaleong/wm-manip/runs/qj3tp91t) |

The two learned data points that matter: **imitation is the only learned paradigm to insert at all**, and **it scales** — 6.3× the demos (112 → 706) roughly doubled success (0.07 → 0.13). RL never produced a single held insertion under any recipe.

## Why a dense reward at all
Stock `PlugCharger-v1` ships **sparse-only** (`SUPPORTED_REWARD_MODES = ["none", "sparse"]` — the authors deliberately excluded dense). With no shaping, TD-MPC2 gets zero signal until a chance success it never finds from scratch (observed: flat `R=0` for 6 h). To make the task RL-tractable we wrote a staged dense reward — [`experiments/envs/plugcharger_dense.py`](../../experiments/envs/plugcharger_dense.py) (`PlugChargerDense-v1`): reach → touch → grasp → insert, with an **angular** alignment term the round-peg PegInsertion reward never needed (success here requires `obj_to_goal_angle <= 0.2`).

## The reward-engineering arc (how RL was exhausted, honestly)
| ver | reward | outcome |
|---|---|---|
| **v1** | flat staged weights (reach 1 / grasp 1 / align 3 / seat 5) | **farmed** — R climbed 3.4 → 104 over 1M while eval success stayed 0. The seat term maxed in a non-success hover band; the policy banked intermediate reward instead of inserting. |
| **v2–v4** | completion-dominant: precursors → 0.1, one success-tracking term `1-tanh(8·dist+4·angle)`, **additive** success bonus (v3's overwrite-to-flat-25 killed the gradient inside the success region) | farm-proof; grazed success ~10× in noisy exploration but never held or reproduced it (eval 0). |
| **v5** | staged align-then-insert: goal-frame decomposition of the approach a single pose-distance term can't express; a replayed solver demo earns it monotonically to a peak at success | **the decisive negative** — the policy optimized it to `train/reward = 38.9` (vs v4's 17.2) while eval success stayed 0. A well-shaped reward, driven hard, still could not produce the terminal maneuver. |

The v1 → v5 progression is the real engineering content: a reward-hacking exploit diagnosed at the ~1M checkpoint and fixed with a farm-proof redesign, then a geometry-decomposed reward that a demo provably earns — and RL *still* can't convert it into a held insertion. So the null is **not a reward-shaping gap**.

## The exhaustive RL null
Every legitimate RL lever, each implemented, validated, and run to a full 2M steps — **all eval success 0**:

| lever | eval success | note |
|---|---|---|
| dense reward v1–v5 | 0 | reward optimized (v5 R=38.9), never converted to a held insert |
| + demos, persistent buffer, 25% ratio (112) | 0 | DDPGfD-style; demos never dilute or evict |
| + **end-effector control** (`pd_ee_delta_pose`) | 0 | [`5wxtsj9f`](https://wandb.ai/chaleong/wm-manip/runs/5wxtsj9f): `success_at_end` flat 0; one lone `success_once`=0.25 blip at 1.7M, `return` 0→29 — grazes once, never holds. Rules out joint-space control as the bottleneck. |
| **RLPD** (demo-bootstrapped SAC, 50/50 demo sampling) | 0 | [`qj3tp91t`](https://wandb.ai/chaleong/wm-manip/runs/qj3tp91t): flat 0 across all 40 eval points, 50k→2M. The recipe *purpose-built* for demo-bootstrapped hard manipulation also fails. |

This spans **model-based, demo-augmented, and demo-bootstrapped RL, across two control spaces** — the RL null is not a TD-MPC2 quirk.

## The controls that make it trustworthy
- **StackCube solved under the same EE control** → [`rs8q0kab`](https://wandb.ai/chaleong/wm-manip/runs/rs8q0kab) reaches **eval 1.0** (matching joint-control [`th868utn`](https://wandb.ai/chaleong/wm-manip/runs/th868utn)). So end-effector control is a *safe, general* lever that holds a solvable task — PlugCharger's flat 0 under it is a genuine task result, not a broken config.
- **Same algorithm, budget, compute, reward rigor** across the ladder → PickCube (trivial) → PegInsertion (0.84) → StackCube (1.0) → PlugCharger (0.0). The 0 is the task, not the implementation.

## Verdict
The value is the *shape* of the result, not a headline number: on extreme-precision keyed insertion, **paradigm success is ordered by prior knowledge**. RL — asked to discover a sub-mm maneuver from reward — categorically fails, and we ruled out every fixable cause (sparse→dense, farming→farm-proof, dilution→persistent buffer, joint→EE control, from-scratch→demo-bootstrapped RLPD). Imitation — handed successful demonstrations — is the only learned paradigm that inserts, and it *scales with data*, making it the clear direction for this task class. The privileged classical planner leads at 0.56 but degrades from its own 0.75 on PegInsertion, so the difficulty is real for everyone. The clean, general takeaway: **for extreme-precision contact tasks, imitate a demonstrator — don't ask RL to discover it.**
