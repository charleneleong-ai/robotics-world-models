# PlugCharger-v1 — the harder-contact rung (Tier 1 task ladder)

**Status:** 📋 SPEC — queued. **Date:** 2026-06-24.

## Why

The PegInsertionSide 3-way is done: **TD-MPC2 0.84 · classical (mplib) 0.75 · PPO 0.00**. That establishes the crossover on *one* contact-rich task. The obvious next question a reviewer asks: **does the world-model advantage hold — or widen — as contact gets harder?** A single task is an anecdote; a *task ladder* is an argument.

`PlugCharger-v1` is the cleanest next rung — same sim, same baselines, no new infra:

- **Tighter tolerance.** Two-prong charger into a wall socket; sub-mm lateral/angular misalignment jams instead of seating. PegInsertion's single round peg is far more forgiving.
- **Alignment-dominated, not insertion-dominated.** The hard part is co-aligning *two* prongs simultaneously — a scripted screw-motion is brittle to it, a reactive learned policy is not.

**Hypothesis:** classical success **drops** relative to PegInsertion's 0.75 (the privileged scripted plan degrades on tighter contact), while the world model holds higher → the **TD-MPC2-vs-classical gap widens**. If true, that's the headline: *"as contact difficulty rises, the privileged classical planner degrades faster than the learned world model"* — a trend, not a point.

## Method (drop-in — confirmed, not hand-rolled)

ManiSkill ships an **mplib** (RRTConnect + screw-motion) solver for this exact task — verified present at
`mani_skill/examples/motionplanning/panda/solutions/plug_charger.py`, driven by the same `run.py` used for PegInsertion. **Zero new planner code.**

Three legs, each reusing the PegInsertion pipeline verbatim with `-e PlugCharger-v1`:

| leg | command / driver | privileged model | compute |
|---|---|---|---|
| **Classical** | `python -m mani_skill.examples.motionplanning.panda.run -e PlugCharger-v1 -n 100 -b gpu --save-video` (isolated `mp` env, numpy 1.26) | **yes** (geometry + scripted contact) | light, ~minutes |
| **TD-MPC2** | `autoresearch` driver, `PlugCharger-v1` Hydra config, ~2M steps, single A100 | no | ~1 day (queues after the PegInsertion clean run) |
| **PPO** | tyro `ppo.py`, `PlugCharger-v1`, state obs (floor reference) | no | optional — likely 0.00, run only if cheap |

## Metrics

- **Success rate** over 100 held-out episodes — directly comparable across the three legs and against the PegInsertion row.
- **Δ-from-PegInsertion** per method — the actual deliverable. A table where classical drops more than TD-MPC2 *is* the result.
- **Planning + execution time** (classical) / **env steps to threshold** (TD-MPC2) — the cost axis.
- **Qualitative:** rollout GIF of a clean planned insertion vs a learned reactive one (mirrors the PegInsertion demo).

## The honest framing (unchanged from the classical baseline)

The classical planner stays **privileged** — handed exact socket geometry + a scripted insertion sub-motion; TD-MPC2 and PPO get none of that. So this is still a *tradeoff illustration*, not a fair fight. The new information is the **slope**: how each paradigm's success moves as the task's contact tolerance tightens. The clean takeaway only holds if classical genuinely drops — if it stays at ~0.75, the honest reporting is "PlugCharger wasn't actually harder for a privileged planner," and the ladder needs a Tier-2 (deformable) rung instead.

## Deliverable

- `experiments/plugcharger_classical/results.json` — success rate, n, mean planning time.
- `experiments/plugcharger_floor/` — TD-MPC2 curve + verdict (mirrors `peginsertion_floor/`).
- A **2-task ladder table** folded into the Project #1 result: PegInsertion vs PlugCharger × {TD-MPC2, classical, PPO}, with the Δ column as the punchline.

## Queue / compute

- **Classical** — `experiments/plugcharger_launch.sh`: cron, GPU-free gate (≥40 GB, never contends `laguna`), mirrors `classical_launch.sh` with `-e PlugCharger-v1`. Completes in one short burst when a card is free.
- **TD-MPC2** — must **queue after** the in-flight PegInsertion clean run (TD-MPC2 is CPU-bound on MPC planning; two on one box contend and blow the wall-clock cap). A chained watcher waits for the PegInsertion `final.pt`, then launches PlugCharger on the freed A100.

## Task ladder — scope of *this* rung vs later

| rung | task | shipped mplib solver? | status |
|---|---|---|---|
| current | `PegInsertionSide-v1` | yes (`peg_insertion_side.py`) | ✅ done (3-way) |
| **this spec** | `PlugCharger-v1` | **yes (`plug_charger.py`)** | 📋 queued |
| next (cheap) | `StackCube-v1` (multi-stage) | yes (`stack_cube.py`) | drop-in follow-on |
| deferred | `TwoRobotPickCube` (bimanual), `AssemblingKits` | **no** → hand-written planner | out of scope |

The deferred rungs need a hand-rolled classical baseline (no shipped solver), which breaks the "runnable, not hand-rolled" discipline — they belong to a later, dedicated effort, not this drop-in extension.
