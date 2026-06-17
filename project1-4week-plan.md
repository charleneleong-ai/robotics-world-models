# Project #1 — 4-Week Plan: World Model for Manipulation vs Classical Planning

> Compressed 1-month version of the build described in `project1-world-models-manipulation-SOTA.md`.
> **Assumption:** ~full-time effort + one A100-class GPU. If it's evenings/weekends, this is a 6–8 week plan — see *If part-time* at the end.

## The deliverable (what "done" looks like in 4 weeks)

A public repo + W&B report + short write-up showing, on **2 contact-rich ManiSkill3 tasks**:
1. A learned world model (**TD-MPC2** anchor + **DreamerV3** same-family baseline) that is **more sample-efficient than model-free SAC/PPO** — success-rate-vs-env-steps curves, ≥5 seeds, IQM+CI via `rliable`.
2. A **classical-vs-learned crossover table** (MoveIt 2 / OMPL on the *same* scenes) — the differentiator only you can credibly produce, framed bidirectionally (classical wins clean free-space; world model wins contact-rich + reactive).

That single repo flips your CV from "CV+robotics+MLOps generalist" to "builds world-model systems." It's enough.

## Scope discipline — what's IN vs CUT

**IN:** TD-MPC2 + DreamerV3 + SAC/PPO floor on PickCube + PegInsertion (or StackCube); OMPL/MoveIt classical baseline; rliable eval; W&B report; write-up.
**CUT (name these as "future work" in the write-up — it shows judgment):** V-JEPA 2-AC latent MPC, iVideoGPT/Cosmos fine-tune, Meta-World secondary suite, >2 tasks, real-robot/LeRobot, 4D anything. Do **not** try to *beat* TD-MPC2 — contextualize against it.

---

## Week 1 — Bring-up + first learning curve (the risky week)

Goal by Friday: **TD-MPC2 learning PickCube on ManiSkill3, beating a PPO floor, logged to W&B.** Install is the #1 time sink — front-load it.

- **Day 1–2 — Environment.** Provision the A100 box. Install **ManiSkill3** (`mani-skill/ManiSkill`, needs NVIDIA + **Vulkan** for render — verify `vulkaninfo` works headless, this is the classic footgun). Run a bundled PPO example on `PickCube-v1` to confirm sim+render+GPU end-to-end *before* touching world models. Set up W&B project.
- **Day 3–4 — TD-MPC2 anchor.** Clone [`nicklashansen/tdmpc2`](https://github.com/nicklashansen/tdmpc2) (stale since May 2025 but stable; it already speaks ManiSkill2 tasks — lowest friction). Get single-task online training running on PickCube (~8 GB GPU). Confirm a success-rate curve climbs. Wire W&B logging.
- **Day 5 — Floors.** Launch **SAC + PPO** on identical PickCube (ManiSkill ships PPO). Kick off 3 seeds each as background runs over the weekend.

**Week 1 exit check:** TD-MPC2 success curve trending up + SAC/PPO baselines running. If Vulkan/install ate the week, that's *normal* — push DreamerV3 to Week 2 and protect the classical baseline (Week 2) at all costs, it's the differentiator.

> De-risk lever: if ManiSkill render install fights you on day 1, prototype the loop on **MuJoCo Playground** `PandaPickCube` (`pip install playground`, A100-fine, JIT 1–3 min) to keep momentum, then port to ManiSkill3.

## Week 2 — Same-family baseline + the classical differentiator

Goal: **DreamerV3 running, and the OMPL/MoveIt classical baseline producing a crossover table.**

- **Day 1–2 — DreamerV3.** Use [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch) (PyTorch, maintained Mar 2026 — avoids the JAX↔CUDA matching footgun of `danijar/dreamerv3`). Point it at PickCube. 3 seeds running.
- **Day 3–5 — Classical baseline (your edge).** Stand up **MoveIt 2 / OMPL** **RRT-Connect** on the *same* PickCube scene geometry (URDF + planning scene from the sim). Measure success rate (fixed planning-time budget), planning time (median + tail — these are heavy-tailed), path quality (length/smoothness/clearance). Use **MotionBenchMaker** scene logs rather than hand-rolling if time allows.

**Week 2 exit check:** three methods (TD-MPC2, DreamerV3, model-free) + a classical planner all producing numbers on PickCube. First draft crossover table exists.

## Week 3 — Second task, seeds, rigor

Goal: **2 tasks, ≥5 seeds, publication-grade plots.**

- **Day 1–2 — Second task.** Add **PegInsertion-v1** (or StackCube) — more contact-rich, where the world model *should* beat the geometric planner. Re-run TD-MPC2 + DreamerV3 + SAC/PPO + OMPL on it.
- **Day 3–4 — Seeds + rliable.** Scale every learned method to **≥5 seeds**. Plot **IQM + stratified bootstrap CI** with [`rliable`](https://github.com/google-research/rliable) (not mean±std over 3 seeds — it's the current reviewer expectation). Report **steps-to-threshold** (env steps to 50%/90% success) as the headline MBRL metric.
- **Day 5 — Robustness probe.** Add a perturbation/dynamic-obstacle test where closed-loop world-model MPC should beat plan-then-execute classical — the cleanest demonstration of *why* a world model.

**Week 3 exit check:** all curves at ≥5 seeds with CIs; crossover table covers 2 tasks + robustness; W&B report assembled.

## Week 4 — Write-up, polish, ship

Goal: **a repo and write-up a hiring manager at Reka/Odyssey/Wayve reads in 5 minutes and gets it.**

- **Day 1–2 — Write-up** (`README` + short blog/W&B report). Lead with the crossover finding. State the honest caveat up front (classical planners assume known geometry/kinematics and are near-optimal there — your model isn't meant to beat RRT-Connect at clean free-space reaching; it wins on contact-rich + reactive). Cite Neural MP's asymmetry as supporting evidence. Report **both** env-step efficiency *and* wall-clock + planning cost.
- **Day 3 — Repo hygiene.** Reproducible: pinned deps, one-command train scripts, seeds, W&B links, a results table in the README. This is where your MLOps background is a genuine edge — most research repos are unreproducible; yours won't be.
- **Day 4 — Framing for the role.** Add a "Why this matters for world models" paragraph using the cluster keywords (learning dynamics from interaction, closed-loop reactive control, sample efficiency). Add the résumé line (see `cv-positioning-rewrite.md` §6).
- **Day 5 — Buffer / stretch.** Use slack for whatever slipped. *Only if everything's done:* a half-day V-JEPA 2-AC inference demo (CEM-MPC on released MIT weights, ~16s/action on one GPU) as a "modern world model" cherry — but don't let it threaten the core deliverable.

---

## Compute budget (single A100)

| Item | Per-seed wall-clock | Notes |
|---|---|---|
| TD-MPC2 single-task | hours → ~1–2 days | ~8 GB GPU; a few ×10⁵–10⁶ env steps |
| DreamerV3 single-task | hours → ~1–2 days | PyTorch port avoids JAX/CUDA matching |
| SAC / PPO floor | hours | PPO native + fast on GPU-parallel ManiSkill |
| 2 tasks × 4 methods × 5 seeds | run in parallel/batched | GPU time is **not** the bottleneck — install + OMPL harness are |

## Top 3 risks → mitigations

1. **Vulkan / ManiSkill render install** eats Week 1 → prototype on MuJoCo Playground in parallel; keep a clean conda env; budget 2 full days, don't panic if it's the whole week.
2. **OMPL/MoveIt-on-same-scene harness** is fiddly → it's the differentiator, so protect its time; use MotionBenchMaker logs; if truly stuck, a standalone OMPL (no full MoveIt 2 stack) on the exported URDF/planning-scene is an acceptable v0.
3. **Scope creep** (V-JEPA, more tasks, 4D) → all explicitly CUT. 2 tasks done > 5 tasks half-done.

## If part-time (evenings/weekends)

Stretch to ~6–8 weeks: Weeks 1–2 here become Weeks 1–3 (install always costs more part-time), and ship **one task** fully (PickCube + classical crossover + 5 seeds) as a complete v0 before adding the second task. A single-task result with a rigorous classical crossover already proves the skill.
