# Related Work (Project #1 README draft)

> Drop-in prose for the project README's "Related Work" / "Positioning" section. Written to claim
> exactly the defensible contribution and pre-empt the overclaim traps. Links use arxiv abs pages;
> swap to your citation style if you add a `.bib`. Update any number you quote against the live
> ManiSkill3 wandb dashboard before publishing.

---

## Related Work

### Learned vs. classical motion planning

Comparing learned approaches against classical sampling-based and optimization planners is a
well-established line of work, not a new question. [MPNet](https://arxiv.org/abs/1907.06013)
(Qureshi et al., ICRA 2019) first showed a neural planner could approximate sampling-based planners
(Informed-RRT\*, BIT\*) at a fraction of the planning time while falling back to a classical planner
for completeness guarantees. [Motion Policy Networks](https://arxiv.org/abs/2210.12209) (Fishman et
al., CoRL 2022) pushed this to end-to-end reactive policies that outperform global planners on
partial point clouds, and [Neural MP](https://arxiv.org/abs/2409.05864) (Dalal et al., IROS 2025)
demonstrated a generalist learned planner beating sampling, optimization, and prior learning
baselines by large margins on cluttered real-world scenes. Most directly,
[Fidalgo Astorquia et al. (*Sensors* 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431259/)
benchmark a SAC policy against OMPL planners on an industrial arm and report essentially the
crossover this project studies — the learned method wins on speed, success, and trajectory
compactness, while classical planning retains zero-shot generality and completeness guarantees.

Crucially, **this entire literature compares learned *policies or planners* against classical
methods, and almost exclusively on free-space (collision-avoidance) motion.** The reasons classical
planners struggle with contact are themselves well-documented — sampling-based planners treat
contact as an obstacle to avoid rather than a resource to exploit, which recent work such as
[HapticRRT](https://arxiv.org/abs/2506.00351) (2025) addresses by recasting RRT onto a contact
equilibrium manifold. What is absent is a comparison where the *learned* side is a **model-based-RL
world model** used for planning, evaluated on **contact-rich** tasks.

### World models for manipulation

Model-based RL with learned world models is now a mature family.
[Dreamer](https://arxiv.org/abs/1912.01603) and [DreamerV3](https://arxiv.org/abs/2301.04104)
(Hafner et al., Nature 2025) learn a latent dynamics model and optimize a policy inside imagined
rollouts; [TD-MPC2](https://arxiv.org/abs/2310.16828) (Hansen et al., ICLR 2024) learns an implicit,
decoder-free latent model and plans with MPPI, reporting state-of-the-art continuous control across
104 tasks including Meta-World and ManiSkill2 manipulation;
[DayDreamer](https://arxiv.org/abs/2206.14176) (Wu et al., CoRL 2022) trains such models directly on
physical robots. A 2025–26 wave of generative and self-supervised world models —
[Ctrl-World](https://arxiv.org/abs/2510.10125), [WMPO](https://arxiv.org/abs/2511.09515),
[iVideoGPT](https://arxiv.org/abs/2405.15223), [V-JEPA 2](https://arxiv.org/abs/2506.09985),
[AdaWorld](https://arxiv.org/abs/2503.18938) — extends the family toward video prediction and
imagination-based policy improvement, surveyed in
[Wang et al. (2026)](https://arxiv.org/abs/2606.00113).

Across this literature, world models are benchmarked **against each other and against model-free RL
(SAC, PPO)** — never against classical motion planners. The two questions ("does a world model beat
model-free RL?" and "does a learned planner beat a classical one?") are studied in separate
communities with disjoint baselines.

### Where this project sits

This project occupies the intersection those two literatures leave open: it places a **model-based-RL
world model** (DreamerV3 and TD-MPC2) and a **classical planner** (OMPL / MoveIt 2, with a strong
modern GPU-trajectory-optimization baseline) on **identical contact-rich ManiSkill3 scenes**
(PegInsertionSide as the contact-rich task, PickCube as a sanity baseline) and characterizes the
crossover under a single evaluation protocol. The contribution is **reproduction, controlled
comparison, and honest characterization**, not a new method:

1. A **DreamerV3 integration on ManiSkill3** (TD-MPC2 ships as a baseline; DreamerV3 does not).
2. A **like-for-like DreamerV3 vs. TD-MPC2 vs. SAC/PPO vs. classical** comparison on identical
   tasks, seeds, observation modalities, and step budgets — published TD-MPC2 manipulation numbers
   are on ManiSkill2 with the authors' own pipeline, so a clean ManiSkill3 head-to-head against
   both a same-family world-model baseline and classical planners is, to our knowledge, unpublished.
3. **rliable-grade statistics** on the world-model-vs-classical crossover: where the world model's
   sample efficiency and closed-loop reactivity win (contact-rich, perturbed), and where classical
   planning remains superior (clean free-space motion — faster, complete, no training).

### What this project does *not* claim

To be explicit, and to avoid claims the literature above falsifies:

- It does **not** claim to be the first to compare learned and classical planning — that lineage
  begins with MPNet (2019).
- It does **not** claim to *discover* that learned methods beat classical ones on contact-rich
  tasks — Neural MP, MπNets, and the contact-planning literature already establish that classical
  planners forfeit contact-rich regimes. This project *quantifies the crossover cleanly and
  reproducibly* rather than discovering it.
- It does **not** introduce a new world model, planner, or training method; DreamerV3, TD-MPC2, and
  OMPL are used as published. The contribution is their controlled composition and evaluation.
- It does **not** claim a higher performance *ceiling* for world models; consistent with
  [MBPO](https://arxiv.org/abs/1906.08253), the supported claim is asymptotic parity with model-free
  RL plus a sample-efficiency gain.

### Evaluation methodology

Results follow the accepted standard for both communities. Reinforcement-learning curves report
success rate versus environment steps, separating **sample efficiency** (steps-to-threshold) from
**asymptotic success**, with [rliable](https://arxiv.org/abs/2108.13264) (Agarwal et al., NeurIPS
2021) interquartile means and stratified-bootstrap 95% confidence intervals over ≥5 seeds rather
than bare mean±std, and all compared methods share an observation modality. Classical-planning
results follow [OMPL Planner Arena](https://arxiv.org/abs/1412.6673) conventions — success within a
fixed time budget, planning time (median and tail), path length, smoothness, and clearance reported
as distributions — on scenes drawn from [MotionBenchMaker](https://arxiv.org/abs/2112.06402).

### Known limitations

Learned world models face **discontinuous contact dynamics**
([Khader et al., RA-L 2020](https://arxiv.org/abs/1909.04915)) and **compounding rollout error** over
the imagination horizon ([MBPO](https://arxiv.org/abs/1906.08253);
[Dreamer](https://arxiv.org/abs/1912.01603)). Sample efficiency is reported in environment steps,
not wall-clock — world models trade environment interaction for per-step compute, and both axes are
disclosed. All results are in simulation; consistent with
[DayDreamer](https://arxiv.org/abs/2206.14176), no sim-to-real transfer is claimed.
