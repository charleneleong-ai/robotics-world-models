# Classical planning baseline — completing the PegInsertionSide 3-way comparison

**Status:** specced + queued (auto-runs when the shared GPU frees). **Date:** 2026-06-22.

## Why

Project #1 currently shows a **2-way** comparison on contact-rich `PegInsertionSide-v1`:

| method | paradigm | success_once | env steps |
|---|---|---|---|
| TD-MPC2 | world model (learned dynamics + MPC) | **0.84** | ~2M |
| PPO | model-free RL | **0.00** | 50M |

The project plan calls the **world-model-vs-classical** crossover "the real contribution" — the angle most candidates can't produce. This baseline adds the **third** leg: a **classical sampling-based motion planner**, turning the story into *learned-dynamics vs model-free vs classical planning*.

## Method (runnable, not hand-rolled)

ManiSkill ships an **mplib** (RRTConnect + screw-motion) solution for this exact task:
`mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py`, driven by `run.py`. mplib 0.1.1 is installed in the `wm` env.

**Run:** `python -m mani_skill.examples.motionplanning.panda.run -e PegInsertionSide-v1 -n 100 -b gpu --save-video` → 100 planned trajectories on the *same* task/scene, reporting the success rate.

## Metrics

- **Success rate** over 100 episodes (the headline, directly comparable to the 0.84 / 0.00).
- **Planning + execution time** per episode (classical's cost is wall-clock planning, not training).
- **Qualitative:** a rollout video (clean planned insertions vs the learned policy's behaviour).

## The honest framing — this is NOT apples-to-apples, and that's the point

The classical planner is **privileged**: it's handed the **known peg/hole geometry + a scripted insertion sub-motion + collision model**, and plans in that known model. TD-MPC2 and PPO get **none** of that — they learn the skill from state + reward. So the comparison illustrates a **tradeoff, not a winner**:

- **Classical** — high success *when* you have an accurate model + can script the contact phase; **zero training**, but **non-reactive**, brittle to model error, and needs per-task engineering.
- **World model (TD-MPC2)** — **no privileged model**; learns dynamics + a reactive policy from interaction; generalizes/adapts where the scripted plan would break (perturbed geometry, sensor noise, novel contact).
- **Model-free (PPO)** — learns reactively too but is **wildly sample-inefficient** here (0.00 at 50M).

Expected outcome: classical likely scores **high** (it's a scripted solution with ground-truth geometry). The takeaway is *why* — "classical wins clean, modelled, free-space-ish problems; the world model earns its keep when you DON'T have the model or the contact is reactive." That's the nuanced, defensible point.

## Deliverable

- `experiments/peginsertion_classical/results.json` — success rate, n, mean planning time.
- A 3-way table + short writeup folded into the Project #1 result (`experiments/peginsertion_floor/verdict.md` or a sibling), + the rollout video link.

## Queue / compute

Needs the GPU (ManiSkill Vulkan sim) but is **light + fast** (1 env, ~minutes for 100 episodes) — far cheaper than the RL runs. Auto-launches via `experiments/classical_launch.sh` (cron, GPU-free gate ≥40 GB so it never contends `laguna`). No training, so no multi-day wait — it completes in one short burst the moment the card is free.

## Out of scope (future)

- cuRobo (GPU-native planner) as a faster/stronger classical point.
- A perturbed-geometry / noisy-state stress test to *show* the classical brittleness vs world-model robustness (the most compelling version of the argument).
