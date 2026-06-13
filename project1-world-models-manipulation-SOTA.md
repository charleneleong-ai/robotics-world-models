# Project 1 — Learned World Model for Robotic Manipulation: 2026 SOTA Research

> **Goal:** Train a learned world model (Dreamer-style latent dynamics) for a robotic manipulation task in simulation, use it for model-based planning, and benchmark against classical motion planning (MoveIt/OMPL).
>
> **Target role:** World Models / Embodied AI research engineer at a robotics + world-models lab (Reka).
>
> **Author background relevant to this project:** production CV (segmentation), distributed A100 training, eval pipelines, and 2018–19 ROS + MoveIt! / OMPL industrial-arm motion planning + Bayesian optimization. The classical-planning baseline is a genuine differentiator — most ML candidates cannot run a fair one.
>
> **Verification note:** All repo URLs, star counts, releases, arxiv IDs, and licenses below were verified against live sources on **2026-06-12**. Star counts are approximate (GitHub rounds and they move daily).

---

## TL;DR — Top Recommendations

| Decision | Recommendation | Why |
|---|---|---|
| **Simulator** | **ManiSkill3** (primary) + **MuJoCo Playground / MJX** (fast-iteration sidecar) | A100-ready (no RT-core requirement), manipulation-native, GPU-parallel sim+render, ships a TD-MPC2 baseline. Skip Isaac Lab as a *starting* choice — its renderer needs RT Cores that datacenter A100/H100 lack. |
| **Primary method** | **TD-MPC2** (anchor) + **DreamerV3** (same-family world-model baseline) | Both single-GPU, both reproducible, both target real manipulation benchmarks. TD-MPC2 ships with ManiSkill2 tasks and 300+ checkpoints. |
| **Stretch method** | **V-JEPA 2-AC** (modern action-conditioned latent world model, MIT weights, single-GPU latent MPC on a real Franka) — or fine-tune **iVideoGPT / Cosmos-Predict2.5-2B** as a generative manipulation world model | Frontier "I understand 2025–26 world models" signal without the (impossible) cost of pretraining a video world model from scratch. |
| **Canonical repo** | `nicklashansen/tdmpc2` (PyTorch) or `danijar/dreamerv3` (JAX) / `NM512/dreamerv3-torch` (PyTorch port) on **ManiSkill3** envs | Lowest-friction, maintained, manipulation-relevant. |
| **First milestone** | DreamerV3 **or** TD-MPC2 on a single ManiSkill3 / MuJoCo Playground manipulation task (PickCube / PegInsertion), beating an SAC/PPO floor, vs an OMPL classical baseline on the same scenes | A complete end-to-end MBRL-vs-classical result on one A100 in days. |

---

## 1. Simulators

A Dreamer-style world model is **off-policy and model-based** — you train latent dynamics from replayed experience, so you do **not** need the extreme on-policy throughput PPO-on-16k-envs demands. What you need: clean manipulation tasks, image (RGBD) observations, moderate-to-high parallel rollout to fill a replay buffer, and an install that actually works on an A100.

### Comparison table

| Simulator | Repo / Stars | GPU-parallel rollout | Manipulation coverage | Install / HW | License | Maintained 2026 |
|---|---|---|---|---|---|---|
| **ManiSkill3** | [mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill) · ~3.0k | **Yes** — 1000s envs, parallel sim+render via SAPIEN, ~30k+ FPS, heterogeneous envs | **Widest** — ~12 domains: rigid, articulated, dexterous, mobile, humanoid, real2sim | pip, Linux+NVIDIA, needs Vulkan for render; **runs on A100** | Apache-2.0 | Yes — active (Jun 2026), still 3.0.0bXX beta |
| **MuJoCo Playground + MJX** | [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) · ~2.0k | **Yes** — 1000s envs via MJX (JAX/XLA) + MuJoCo Warp backend; GPU **or TPU** | LEAP hand, Panda pick-cube, ALOHA peg-insert; ~50 envs | `pip install playground`, very light; **A100/H100/TPU fine** | Apache-2.0 | Yes — active (May 2026) |
| **NVIDIA Isaac Lab** | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) · ~7.4k | **Best-in-class** — ~4k envs/GPU, 16k on 8 GPU (>1.6M FPS Franka); PhysX + tiled RTX render | Franka lift/cabinet, Allegro in-hand, NIST factory assembly; 30+ envs | Heavy: Isaac Sim/Omniverse, ~50 GB, needs **RTX GPU (RT Cores) — NOT A100-friendly** | BSD-3 + Apache-2.0 | Yes — daily commits; v3.0.0-beta |
| **Genesis** | [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) · ~29.3k | Yes, but simple contact solver; weak on contact-rich | **None curated** — physics substrate only | pip; cross-platform | Apache-2.0 | Yes — v1.1.1 (Jun 2026) |
| **RoboSuite** | [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) · ~2.5k | **No** — CPU-bound MuJoCo; parallelism = many CPU procs | Lift/Stack/Door/NutAssembly/PickPlace + bimanual | Very easy, pip, laptop/CPU | MIT | Yes — v1.5.2 (Dec 2025) |
| **mjlab** | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) · ~2.5k | **Yes** — 1000s envs via MJWarp | Reference manipulation tasks (not a full suite yet) | pip, light; **no Omniverse**; A100 fine | Apache-2.0 | Yes — v1.4.0 (May 2026) |
| **NVIDIA Newton** | [newton-physics/newton](https://github.com/newton-physics/newton) · ~5.1k | **Yes** — Warp multi-solver, differentiable | Engine, not a task suite | NVIDIA-GPU-centric | Apache-2.0 | **1.0 GA Apr 2026**, v1.3.0 (Jun 2026) |
| **RoboCasa** | [robocasa/robocasa](https://github.com/robocasa/robocasa) · ~1.5k | **No** — RoboSuite/MuJoCo, CPU-bound | 365 tasks, 2500+ kitchen scenes, 2200+ hrs demos | RoboSuite stack | MIT + CC BY 4.0 | Yes — v1.0 (Feb 2026) |

### Per-simulator notes

**NVIDIA Isaac Lab — the heavyweight standard.** Officially replaces Isaac Gym Preview, IsaacGymEnvs, **OmniIsaacGymEnvs** (OIGE 4.0.0 was the last), and Orbit. Isaac Gym is deprecated/legacy. Best at GPU-parallel physics **+ photorealistic RTX rendering** (vision-in-the-loop, sim-to-real, contact-rich industrial assembly). **The A100 problem (critical):** Isaac Sim needs **RT Cores for raytraced rendering** — datacenter A100/H100 are explicitly *not recommended* (current docs cite a min RTX 4080, ≥16 GB VRAM). Headless physics-only RL may run, but anything touching the renderer wants RTX 4090 / RTX 6000 Ada / L40S. Plus ~50 GB disk and an Omniverse/USD learning curve. **Right tool later, wrong first choice on an A100.**

**MuJoCo + Playground + MJX — the accessible powerhouse.** Frictionless, fully open, JAX-native, trains on **GPU or TPU**. MJX runs thousands of envs; recent releases default to the **MuJoCo Warp** backend (fixes much of MJX's contact bottleneck). Honest weakness: classic MJX contact cost scales with *possible* contacts, not active ones (JAX static shapes), so dense multi-object contact-rich scenes scale poorly — the Warp backend is the fix. First-run JIT compile is 1–3 min. Runs perfectly on an A100.

**ManiSkill3 — the purpose-built manipulation benchmark.** Literally a GPU-parallel manipulation benchmark + training framework on SAPIEN/PhysX (UCSD Hao Su lab / Hillbot). True GPU sim AND render via SAPIEN's parallel renderer — thousands of envs, **30,000+ FPS** with RGBD/segmentation, 2–4× less GPU memory than competitors. Standout: **heterogeneous parallel envs** (every parallel env can have a different scene/object) — great for generalization. Bundles PPO/SAC/**TD-MPC2**, BC/diffusion-policy, and VLA baselines (Octo/RDT-1B/RT-X) + sim2real examples. Caveats: full functionality is Linux + NVIDIA (Windows/macOS GPU-sim limited); still carries a beta label but is stable and widely used. Runs cleanly on A100. **Note:** repo moved to org `mani-skill/ManiSkill` (`haosulab/ManiSkill` redirects).

**Genesis — viral, but be skeptical.** ~29.3k stars and genuinely broad multi-physics (rigid/FEM/MPM/SPH/IPC/SAP, deformables/fluids/soft bodies), but the headline **"generative" data-generation layer was never open-sourced** (~18 months later, still unreleased). The "430,000× faster than realtime" claim is misleading (measured with 1 substep, robot idle >90%, self-collisions off, static objects hibernated). On a realistic cube-pick it ran ~150× slower than the headline and 3–10× slower than ManiSkill/SAPIEN on contact-rich tasks, with accuracy issues. No curated manipulation suite, no bundled RL/IL baselines. **Not production-ready for manipulation RL / world-model research in mid-2026.**

**RoboSuite — easy IL workhorse, CPU-bound.** De-facto academic standard for imitation learning and the substrate for robomimic. Crucial limit: **CPU-bound MuJoCo** — no native thousands-of-GPU-envs; parallelism is many CPU processes. Lowest barrier to entry on the list; great if you pivot to IL from demos.

**Newer 2025–26 entrants worth knowing:** **NVIDIA Newton** (differentiable multi-solver engine on Warp + OpenUSD, NVIDIA + DeepMind + Disney, 1.0 GA Apr 2026; a physics engine, not a task suite); **MuJoCo Warp** (the GPU MuJoCo physics layer, a Newton solver); **mjlab** ("Isaac Lab API, powered by MuJoCo-Warp" — thousands of GPU envs, single pip install, no Omniverse, A100-friendly — the most promising lightweight Isaac-Lab alternative, but young with a thin task library); **RoboCasa** (365 tasks / 2500+ scenes / 2200+ hrs demos for generalist IL, CPU-bound); **RoboVerse** (meta-platform wrapping ~18 sims).

### Recommendation

**Primary: ManiSkill3.** Runs on a bare A100 with no RT-core problem (the single biggest reason to skip Isaac Lab first); bundled manipulation tasks with RGBD observations (world models need pixels); GPU-parallel sim+render fills replay buffers fast without 50 CPU workers; already ships **TD-MPC2** (a model-based code path to learn from). Apache-2.0, maintained.

**Fast-iteration sidecar: MuJoCo Playground / MJX.** Trivial install, runs on A100 *and* TPU, JAX-native — ideal for prototyping the world-model loop on simple tasks before scaling, and composes cleanly with `danijar/dreamerv3` (also JAX).

**Honest caveats:**
- **Skip Isaac Lab as the starting choice** — A100/RT-core mismatch + ~50 GB Omniverse + steep USD curve. Right later for photoreal vision-in-the-loop or 8-GPU on-policy RL on RTX/L40 hardware.
- **Skip Genesis for now** — manipulation RL ecosystem and the "generative" capability aren't there; speed advantage evaporates on contact-rich tasks.
- **RoboSuite/RoboCasa** if you pivot to imitation learning from demos — but CPU physics caps rollout throughput.
- **Watch mjlab** — the most promising 2026 on-ramp for Isaac-Lab-style task authoring without Omniverse and with A100 compatibility; revisit in a quarter.
- **Contact-rich reality check:** dexterous in-hand and tight peg-insertion stress every GPU sim's contact solver. ManiSkill/SAPIEN (PhysX) and the Warp/Newton stack handle this best; classic MJX is weakest on dense contacts (mitigated by its Warp backend).

---

## 2. World Model Methods

The honest theme: **frontier video world models are datacenter-only to train.** The solo game is either (a) train a small simulator-style model **from scratch** on one GPU, or (b) **fine-tune / plan** with released weights on a frozen backbone.

### The Dreamer line

| Item | Detail |
|---|---|
| **DreamerV3** | "Mastering Diverse Domains through World Models," Hafner et al. — published in **Nature 2025**, arxiv [2301.04104](https://arxiv.org/abs/2301.04104). RSSM latent world model, learns a policy in imagination. **Single GPU, JAX**, maintained: [danijar/dreamerv3](https://github.com/danijar/dreamerv3). Atari/Crafter/DMLab/Minecraft. **No native robot-manipulation suite**, but the architecture ports to any gym-style env. The reference "stable online world-model RL" baseline. |
| **Dreamer 4 (current latest)** | "Training Agents Inside of Scalable World Models," **Hafner, Yan, Lillicrap, 2025**, arxiv [2509.24527](https://arxiv.org/abs/2509.24527), [project page](https://danijar.com/project/dreamer4/). A pivot from V3: the world model is now a **conditional latent video-diffusion transformer** with a "shortcut forcing" objective, trained **offline** on ~2500h of action-labeled Minecraft video. First agent to mine diamonds in Minecraft purely offline. **Real-time interactive inference on a single GPU** — but that's inference; full training is large-scale. Explicitly demonstrates a **robotics dataset** with counterfactual rollouts. **No official danijar code release yet**; community ports exist (see §3). |

**Solo-dev verdict:** **DreamerV3 is fully tractable and the safest portfolio anchor** — single GPU, maintained, easy to point at a manipulation env. **Dreamer 4 inference may be runnable, but training/reproducing it is not a clean solo project yet** — position it as the *direction*, not the deliverable. Note the line's center of gravity has moved toward **offline + video-diffusion world models**.

### TD-MPC2

**"TD-MPC2: Scalable, Robust World Models for Continuous Control,"** Hansen, Su, Wang, ICLR 2024, arxiv [2310.16828](https://arxiv.org/abs/2310.16828). Learns an implicit (**decoder-free**) latent world model and plans with **MPPI** in latent space. PyTorch, single GPU (8 GB single-task), [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) ships **300+ checkpoints**. Covers **104 continuous-control tasks including Meta-World (50) and ManiSkill2 (5) manipulation** plus DMControl/MyoSuite. **No official TD-MPC3** as of June 2026 — only 2025–26 follow-ups improving it (value-overestimation fix, discrete-codebook variants [2503.00653](https://arxiv.org/html/2503.00653v1), bisimulation-metric MPC, latent gradient-MPC). **The single most reproducible, manipulation-relevant model-based RL method available.** (Caveat: repo has had no commits since May 2025 — stable but not actively developed.)

### Video / generative world models for robotics

The open/closed split is stark. **Fully closed (no local path):** all DeepMind **Genie 1/2/3** (latest Genie 3, Aug 2025, productized "Project Genie" Jan 2026), **UniSim**, all Google **Gemini Robotics** flagships, Wayve GAIA-1/2. **Open and genuinely usable:**

| Method | Latest / year | arxiv / URL | Solo tractable | Compute notes |
|---|---|---|---|---|
| **NVIDIA Cosmos** (Predict2.5 / Transfer2.5 / Reason2; **Cosmos 3** Jun 2026) | platform [2501.03575](https://arxiv.org/abs/2501.03575); Predict2.5 [2511.00062](https://arxiv.org/abs/2511.00062); Cosmos 3 [2606.02800](https://arxiv.org/abs/2606.02800) | [github.com/nvidia-cosmos](https://github.com/nvidia-cosmos) | **✓ inference / LoRA at 2B–14B** | 2B inference ~26–33 GB (1×40 GB A100); 14B ~49–57 GB (1×80 GB). 2B fine-tune on 1–2×80 GB. **Pretrain (100M+ clips) datacenter-only.** Best frontier-quality open option. |
| **1X World Model Challenge** (`1xgpt`) | tech report Oct 2025 [2510.07092](https://arxiv.org/abs/2510.07092) | [github.com/1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt) | **✓ strong** | Pre-tokenized ~100h humanoid data; benchmarked on a single RTX 4090. Best *humanoid* world-model entry point. |
| **iVideoGPT** | NeurIPS 2024 [2405.15223](https://arxiv.org/abs/2405.15223) | [github.com/thuml/iVideoGPT](https://github.com/thuml/iVideoGPT) | **✓ full pipeline** | Token-based interactive WM, pretrained on Open-X, plugs into MBRL on **manipulation**. Inference <8 GB; fine-tune 1–3 GPU-days. Most robotics-relevant fully-open WM. |
| **DIAMOND** | NeurIPS 2024 [2405.12399](https://arxiv.org/abs/2405.12399) | [github.com/eloialonso/diamond](https://github.com/eloialonso/diamond) | **✓ train from scratch** | Cleanest diffusion-WM-for-RL recipe. Atari ~3 GPU-days/game; CS:GO ~12 days on one 4090. Sim/games, not real-robot. |
| **Vidar** | 2025 [2507.12898](https://arxiv.org/abs/2507.12898) | [github.com/thu-ml/vidar](https://github.com/thu-ml/vidar) | inference + ~20-min real-robot adapt | Video-diffusion prior + masked inverse dynamics; adapts to unseen robot with ~20 min of demos. |
| **Ctrl-World / Vid2World / IRASim** | 2024–25 ([2510.10125](https://arxiv.org/abs/2510.10125) / [2505.14357](https://arxiv.org/abs/2505.14357) / [2406.14540](https://arxiv.org/abs/2406.14540)) | thuml / bytedance / Robert-gyj | inference / fine-tune 1–2 A100 | Controllable manipulation video WMs; policy-in-the-loop imagination. |

**Cosmos3-Nano — the frontier reference rung (not a floor baseline).** Within the Cosmos 3 family, [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) is the smallest SKU: a **16B omnimodal** world foundation model (text/image/video/audio/action-trajectory in and out), **OpenMDW license**, BF16, and — critically — **Ampere-supported, so it runs on a single A100 80GB for inference** (16B BF16 ≈ ~32 GB weights). It ships a **`Cosmos3-Nano-Policy-DROID`** manipulation-policy variant, which puts it directly on this project's lane. **Position it as the frontier reference / stretch dynamics arm — NOT a floor baseline:** you cannot fairly chart a web-pretrained 16B model against from-scratch TD-MPC2/SAC on sample efficiency (it has seen millions of clips; they have seen only this task). Its value is *adaptation* (zero-shot DROID policy, or LoRA), giving a **frontier-vs-reproducible-vs-classical** comparison on identical scenes. **Caveat: verify exact LoRA/fine-tune VRAM before committing** — the HF page is summarised by a small model; 16B is BF16-only and a serious fine-tune is likely datacenter territory. (`Cosmos-Reason2` is the sibling spatial-VLM frontier reference for the grounding arm in Projects #3-spatial / #4.)

**Honest compute reality:** training any frontier video world model from scratch (Genie 11B, UniSim, Cosmos pretrain, GAIA) is **off the table solo**. Realistic paths: **inference + LoRA on Cosmos (incl. Cosmos3-Nano)**, **train-from-scratch on small academic WMs (DIAMOND / 1xgpt)**, or **fine-tune a small head/adapter (Vidar, iVideoGPT)**.

### JEPA / V-JEPA for robotics

**Latest: V-JEPA 2 + V-JEPA 2-AC** (Meta, June 2025), arxiv [2506.09985](https://arxiv.org/abs/2506.09985), [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), **MIT-licensed weights**. **No V-JEPA 3 exists** (2026 follow-ups are V-JEPA 2.1 dense-feature recipe and VL-JEPA [2512.10942](https://arxiv.org/abs/2512.10942), neither robotics-action-focused). **It IS action-conditioned and used for planning:** V-JEPA 2-AC is a ~300M-param predictor post-trained on a **frozen ViT-g encoder** on <62h of unlabeled DROID robot video. It plans by **energy minimization via Cross-Entropy-Method MPC in latent space**. Zero-shot Franka results across two labs: Reach 100%, Pick-and-place 65–80%, Grasp 25–65%, deployed with no data from the deployment labs.

**Solo-dev verdict:** Pretraining (22M videos, >1M hours, 1B-param ViT-g) is **datacenter-only — do not attempt.** But **CEM-MPC planning inference runs on ONE consumer GPU (~16s/action on RTX 4090)**, and **fine-tuning the ~300M AC predictor on a frozen encoder with your own robot video is the realistic solo entry point** (few A100s, days). The strongest *self-supervised, real-manipulation, planning-capable* open release.

### Diffusion / transformer dynamics models for model-based control

**Critical distinction:** "world model" conflates (1) **simulators rolled out in an MPC/RL loop** (true model-based control), (2) **video predictors → actions via inverse dynamics** (data engines), and (3) **diffusion/transformer action *policies*** (NOT world models — Diffusion Policy, Decision Transformer, GR00T). Keep focus on (1)/(2).

- **Transformer dynamics (from scratch, single GPU):** **STORM** [2310.09615](https://arxiv.org/abs/2310.09615) (most tractable: 4.3h/game on one RTX 3090; note: no LICENSE file — start from OC-STORM [2501.16443](https://arxiv.org/abs/2501.16443)); **IRIS** [2209.00588](https://arxiv.org/abs/2209.00588) (canonical VQ-VAE+GPT WM); **TWM** [2303.07109](https://arxiv.org/abs/2303.07109); **Trajectory Transformer** [2106.02039](https://arxiv.org/abs/2106.02039).
- **Diffusion dynamics:** **DIAMOND** (best from-scratch diffusion-WM-for-RL recipe); **Diffusion Forcing** [2407.01392](https://arxiv.org/abs/2407.01392) + **DFoT** [2502.06764](https://arxiv.org/abs/2502.06764); **AVDC** [2310.08576](https://arxiv.org/abs/2310.08576) (the runnable open UniPi successor); **Diffusion World Model (DWM)** [2402.03570](https://arxiv.org/abs/2402.03570) (cheapest idea but no official code).
- **Universal caveats:** iterative diffusion sampling makes video-rollout planners slow at inference; the small transformer/diffusion WMs are Atari/maze/sim benchmarks, not real-robot manipulation — they demonstrate MBRL mechanics, not a manipulation result.

### Recommendations

**PRIMARY (tractable, reproducible, strong portfolio signal): TD-MPC2 anchor + DreamerV3 same-family baseline, with V-JEPA 2-AC as the modern headline.**
- **TD-MPC2** as the model-based RL backbone: single GPU, PyTorch, real manipulation benchmarks (Meta-World, ManiSkill2), 300+ checkpoints, ICLR 2024 pedigree. A complete model-based-RL-for-manipulation result on one A100 in days.
- **DreamerV3** as the same-family world-model comparison (per your own "always establish a baseline" rule).
- **V-JEPA 2-AC** for the "I understand 2025–26 world models" signal: SOTA self-supervised, action-conditioned, planning-capable, MIT weights, single-GPU latent MPC on a real Franka. Fine-tuning the ~300M AC head is genuinely solo-feasible. If forced to pick one, **TD-MPC2** is the lowest-risk anchor; **V-JEPA 2-AC** is the higher-signal modern piece.

**STRETCH (ambitious video/generative angle): fine-tune iVideoGPT or NVIDIA Cosmos-Predict2.5-2B as a manipulation world model.**
- **iVideoGPT** — fully open MIT, pretrained on Open-X, explicitly supports MBRL on manipulation, full pipeline fine-tunable on a single GPU. Highest "I implemented the whole MBRL loop" credibility.
- **Cosmos-Predict2.5-2B** — the only frontier-quality *open* world foundation model a solo dev can run (inference ~32 GB on one 40 GB A100; LoRA on 1–2×80 GB). Strong "physical AI" framing for a lab.
- **Cosmos3-Nano (16B) + `Cosmos3-Nano-Policy-DROID`** — the current-frontier omnimodal rung that still fits a single A100 80GB for *inference* (~32 GB weights, Ampere-supported). Use it as a **frontier dynamics arm** (zero-shot DROID policy or LoRA) for a frontier-vs-reproducible-vs-classical comparison — **not** a sample-efficiency floor baseline. Verify fine-tune VRAM first.
- Cite **Dreamer 4** as the exciting frontier reference/motivation, not the deliverable.

**One-line compute summary:** TD-MPC2, DreamerV3, STORM/IRIS/DIAMOND = train-from-scratch on one GPU. V-JEPA 2-AC, Vidar, iVideoGPT, Cosmos-2B = fine-tune/plan on released weights, single-to-few A100. Everything frontier-video (Genie, UniSim, Cosmos pretrain, GAIA, Dreamer 4 training) = datacenter-only, don't attempt from scratch.

---

## 3. Canonical Repos

All star counts, last-push dates, and frameworks verified live on 2026-06-12.

| Repo | URL | Stars | Framework | Maintained? (last push) | Install / hardware notes |
|---|---|---|---|---|---|
| **DreamerV3 (official)** | https://github.com/danijar/dreamerv3 | ~3,400 | **JAX** | Yes — May 25, 2026 | Python 3.11+. Install JAX first (matching your CUDA), then `pip install -r requirements.txt`. Dockerfile provided. **JAX↔CUDA version mismatch is the #1 install footgun** — no hard pins, you match JAX to your CUDA. GPU strongly recommended. |
| **dreamerv3-torch** (NM512) | https://github.com/NM512/dreamerv3-torch | ~860 | **PyTorch** | Yes — Mar 8, 2026 | Best-regarded PyTorch reimplementation of DreamerV3. Easy to hack, standard PyTorch+CUDA, no exotic deps. The pragmatic solo-dev choice if you want to avoid JAX. |
| **TD-MPC2 (official)** | https://github.com/nicklashansen/tdmpc2 | ~860 | **PyTorch** | **Stale — May 21, 2025** | Conda or Docker. Single-task online: 8 GB+ GPU, 12 GB+ RAM. Multi-task offline: 24 GB+ GPU, 128 GB+ RAM. 300+ checkpoints. DMControl/Meta-World/ManiSkill2/MyoSuite (104 tasks). Stable/usable but not actively developed. |
| **Isaac Lab** | https://github.com/isaac-sim/IsaacLab | ~7,400 | PyTorch (on Isaac Sim) | Yes — daily commits | Heavyweight. Needs Isaac Sim + Omniverse, an **RTX GPU (min RTX 4080, ≥16 GB VRAM rec.)**, CUDA ≥12.4. Ubuntu 22.04 best. Steep solo setup. |
| **ManiSkill3** | https://github.com/mani-skill/ManiSkill | ~3,000 | PyTorch (SAPIEN) | Yes — Jun 11, 2026 | Repo moved to `mani-skill/ManiSkill`. Still beta (3.0.0bXX). Linux best, no macOS GPU sim. Needs NVIDIA GPU + **Vulkan** for rendering. Python ≥3.10. GPU-parallelized. |
| **MuJoCo Playground** | https://github.com/google-deepmind/mujoco_playground | ~2,000 | **JAX** (MJX) | Yes — May 27, 2026 | `pip install playground`. Built on MJX + optional MuJoCo Warp backend. Needs JAX+CUDA 12 GPU. On Ampere (RTX 30/40) set `JAX_DEFAULT_MATMUL_PRECISION=highest`. Has dexterous + non-prehensile manipulation envs + a batch renderer. |
| **LeRobot** | https://github.com/huggingface/lerobot | ~24,900 | **PyTorch** | Yes — daily commits | `pip install lerobot`. See relevance note. |
| **Dreamer 4 (community)** | [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4) (~190★, active) · [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) (~320★, Mar 2026) · edwhu/dreamer4-jax | PyTorch / JAX | Community only — no official danijar release | Video-diffusion scale, compute-hungry. Aspirational, not a first build. |

**LeRobot — relevance, specifically:** Primarily a **real-robot + imitation-learning / VLA** library (ACT, Diffusion Policy, VQ-BeT, Pi0, GR00T, SmolVLA) with a hardware abstraction layer for physical arms and a big datasets/checkpoints hub. **Not** a model-based / world-model RL framework at its core. In 2026 it has grown adjacent pieces (a TDMPC policy, a VLA-JEPA model-based entry, HIL-SERL). For a *world-model manipulation* project it's tangential — useful if you later want real-robot deployment, standardized datasets, or imitation/VLA baselines to compare against. **Don't centre the project on it; treat it as the real-robot/data on-ramp.**

---

## 4. Benchmarks, Metrics & Baselines

### Standard manipulation benchmarks

| Benchmark | Sim engine | Tasks | What makes it standard | 2026 status |
|---|---|---|---|---|
| **Meta-World** (v2 / MT10 / MT50 / ML45) | MuJoCo | 50 tabletop tasks, Sawyer arm; MT50 = learn all 50, ML45 = meta-train 45→test 5 | Default **multi-task / meta-RL** suite; dense reward + binary `success` flag | **Still standard** — use **Meta-World+** ([2505.11289](https://arxiv.org/pdf/2505.11289), May 2025) which fixes reward/reproducibility/space drift |
| **ManiSkill3** | SAPIEN (GPU) | 12 domains; thousands of objects | **GPU-parallel sim+render, 30k+ FPS** → visual RL in minutes; the modern go-to for fast MBRL iteration | **Rising standard**, ICLR 2025; best fit for a solo compute budget |
| **RoboMimic** | MuJoCo (robosuite) | 8 tasks, Franka, 6000+ demos | Reference for **learning-from-demonstration / offline** | Still used (imitation/offline) |
| **RLBench** | CoppeliaSim/PyRep | 100 tasks, multi-cam RGB-D | Long-horizon, vision-rich; **VLA / multi-task** eval | Still widely used (VLA) |
| **CALVIN** | PyBullet | Long-horizon **language-conditioned** chains, Franka | Standard for **language-conditioned long-horizon** | Still used (language/VLA) |
| **LIBERO** | robosuite/MuJoCo | 130 tasks, 4 suites | Standard for **lifelong / continual** + VLA generalization | Very active 2025–26 |
| **SIMPLER / SimplerEnv** | SAPIEN | Tabletop mirroring real Google-Robot/Bridge | Built to **minimize sim-to-real gap** | Active (validates VLA sim scores vs real) |

**Key 2025–26 shift:** the field bifurcated. VLA / language-conditioned work → RLBench, CALVIN, LIBERO, RoboCasa, SIMPLER. **MBRL / world-model / continuous-control work → Meta-World(+), ManiSkill, DMControl, MyoSuite** — exactly your lane. The canonical TD-MPC2 suite (104 tasks) is DMControl + Meta-World + ManiSkill2 + MyoSuite, so matching that set buys instant comparability. ([Meta-World](https://meta-world.github.io/) · [ManiSkill3 2410.00425](https://arxiv.org/abs/2410.00425) · [RLBench](https://arxiv.org/pdf/1909.12271))

### Metrics a credible world-model / MBRL result must show

**Primary (must-have):**
- **Success rate** — fraction of episodes achieving the goal (per-task *and* averaged; don't hide a bimodal distribution behind a mean).
- **Sample efficiency** — success rate **as a function of real env steps**; report **steps-to-threshold** (env steps to reach 50% / 90% success). The single most important MBRL metric and the whole point of a world model.
- **Asymptotic performance** — final converged success / return at the step budget.

**Secondary (expected for a world-model result):**
- **Wall-clock time** — train time to threshold *and* total. World models trade env steps for compute; env-step efficiency can look great while wall-clock is worse — **disclose both axes**.
- **Planning / inference cost** — for MPC (TD-MPC2): planning cost per decision (CEM/MPPI iterations, candidate trajectories, control freq / latency). For Dreamer imagination: horizon and rollout cost.
- **Aggregate return** alongside success.

**Rigor (separates "credible" from "toy"):**
- **≥5–10 seeds**, plot **mean ± stratified bootstrap CI** or use **rliable** (IQM / performance profiles) rather than mean±std over 3 seeds. An explicit reviewer expectation now.
- State the **exact benchmark version** (Meta-World vs Meta-World+, ManiSkill2 vs 3), success-threshold definition, and eval-time stochasticity. Meta-World+ exists *because* non-comparable setups inflated claims.
- State **state vs visual** observations clearly — sample-efficiency numbers aren't comparable across them.

### Classical-planning baseline (your differentiator)

Your 2018–19 ROS + MoveIt! / OMPL background makes this the strongest, most distinctive part of the portfolio.

**Stack to benchmark against:** **MoveIt 2** (orchestration; default planner = OMPL) → **OMPL** sampling-based planners (**RRT-Connect/BiRRT**, **RRT\***, **PRM/PRM\***, **BIT\*** / Informed-RRT\*) → trajectory optimization (**CHOMP**, **TrajOpt**, MoveIt **Pilz** for deterministic LIN/PTP). Use **MotionBenchMaker** (5 robots × 8 envs, OMPL-compatible logs) → **OMPL Planner Arena** for apples-to-apples plots rather than hand-rolling. ([OMPL](https://ompl.kavrakilab.org/) · [MotionBenchMaker](https://carlosquinterop.github.io/project/motionbenchmaker/))

**Fair-comparison metrics (use the established motion-planning ones verbatim):** success rate (within a fixed planning-time budget); planning time (median + tail — classical planners are heavy-tailed); path quality (length, smoothness/jerk, clearance); generalization (across randomized scenes / obstacle poses / start-goal pairs).

**Honest caveats — state these explicitly; they make the comparison credible rather than a strawman:**
- Classical planners **assume a known kinematic model and known collision geometry** (URDF + planning scene). They are near-optimal in that regime — your learned model should **not** be expected to beat RRT-Connect at free-space collision-free reaching with a clean model. Claiming otherwise is a red flag.
- The learned world model's value is in **contact-rich / under-modeled / unknown-dynamics** regimes (pushing, insertion, deformables, friction, slippage) where a geometric planner has no contact model.
- Classical = **plan-then-execute**; world-model MPC = **closed-loop reactive** — also compare on **robustness to perturbation / dynamic obstacles**, where the reactive policy should win.
- Cite real 2024–25 evidence honestly: **Neural MP** (learned generalist planner) reports +23% / +17% / +79% success over sampling / optimization / prior-learning planners on 64 real tasks — **but** its failure modes are noisy point clouds and tight confined-space cases, exactly where clean-geometry classical planners are reliable. **That asymmetry is the story.** ([Neural MP](https://mihdalal.github.io/neuralmotionplanner/))

**Framing:** "Classical planners win on clean-geometry free-space motion (fast, complete, optimal); the learned world model wins on contact-rich / under-modeled dynamics and on closed-loop reactivity. I quantify the crossover." A nuanced, bidirectional result is far more impressive than a one-sided "ML beats classical."

### Model-based RL baseline ladder

| Baseline | Type | Why include |
|---|---|---|
| **SAC** | Model-free, off-policy | Data-efficiency floor for continuous control; mandatory |
| **PPO** | Model-free, on-policy | Standard reference; native strong baseline on GPU-parallel ManiSkill |
| **DreamerV3** | Model-based, latent imagination | Reference world-model baseline; same family → apples-to-apples |
| **TD-MPC2** | Model-based, latent + MPC | Current **SOTA MBRL** on the Meta-World/ManiSkill/DMControl/MyoSuite suite; the bar to clear or contextualize |

Verified positioning: **TD-MPC2 outperforms SAC and DreamerV3** in data efficiency and final performance across 104 online tasks (>60% on the hardest manipulation task, Pick-YCB, where others fail in budget). You don't need to *beat* TD-MPC2 — include it as the honest SOTA reference and position your contribution (the classical-baseline crossover, or a specific contact-rich regime) relative to it.

### Recommendation for this project

- **Primary benchmark: ManiSkill3** — GPU-parallel, trains in minutes-to-hours on one GPU; genuine contact-rich tasks (PegInsertion, PickCube, StackCube) that justify a world model over a geometric planner; gives a clean point-cloud/state planning scene to drive MoveIt/OMPL on the *same* scenes.
- **Secondary: Meta-World+ (MT10 or a curated subset)** — cheap, ubiquitous, instant comparability with the TD-MPC2/DreamerV3 literature. Use Meta-World+, not the original, and say so.
- Skip CALVIN/RLBench/LIBERO/RoboCasa/SIMPLER — language/VLA-oriented, off the world-model thesis.
- **Baseline ladder:** SAC + PPO (floor) → DreamerV3 (same-family) → TD-MPC2 (SOTA anchor) → MoveIt 2 / OMPL classical (your differentiator).
- **Headline deliverable:** success-rate-vs-env-steps curves (≥5 seeds, IQM + CI via rliable) on a handful of ManiSkill3 contact-rich tasks, **plus a classical-vs-learned crossover table** (success rate / planning time / path quality / robustness-to-perturbation) showing classical dominating clean free-space motion and the world model dominating contact-rich/reactive regimes — with the known-model/known-geometry caveat stated. Report wall-clock + planning cost alongside env-step efficiency.

---

## 5. Key Papers — Prioritized Reading / Reproduction List

Ordered must-read → stretch. **[REPRODUCE]** = public code, tractable to run/re-implement solo. **[LANDMARK]** = read for ideas/framing; not realistically reproducible solo (closed weights or industrial scale). *(Genie 2/3 have no formal arxiv paper — blog only.)*

### Tier 0 — Core canon. Read + reproduce first; they ARE the job.

1. **World Models** — Ha & Schmidhuber, 2018 — **[REPRODUCE]** — [1803.10122](https://arxiv.org/abs/1803.10122). Origin of VAE→MDN-RNN→controller and "train the policy inside the dream." Every Dreamer descendant refines this.
2. **PlaNet — Learning Latent Dynamics for Planning from Pixels** — Hafner et al., 2018/19 — **[REPRODUCE]** — [1811.04551](https://arxiv.org/abs/1811.04551). Introduces the **RSSM** — the literal backbone you'll build.
3. **Dream to Control (Dreamer V1)** — Hafner et al., 2019 — **[REPRODUCE]** — [1912.01603](https://arxiv.org/abs/1912.01603). Actor-critic learning in imagination via analytic value gradients — the "Dreamer-style" core.
4. **Mastering Atari with Discrete World Models (DreamerV2)** — Hafner et al., 2020 — **[REPRODUCE]** — [2010.02193](https://arxiv.org/abs/2010.02193). Discrete categorical latents + straight-through gradients — the representation choice that made world models scale.
5. **Mastering Diverse Domains through World Models (DreamerV3)** — Hafner et al., 2023 (Nature 2025) — **[REPRODUCE — primary baseline]** — [2301.04104](https://arxiv.org/abs/2301.04104). **Your baseline.** One config across 150+ tasks via symlog, two-hot returns, free-bits, percentile normalization — robustness tricks to port to manipulation.
6. **TD-MPC2 — Scalable, Robust World Models for Continuous Control** — Hansen, Su, Wang, 2023 — **[REPRODUCE — primary baseline]** — [2310.16828](https://arxiv.org/abs/2310.16828). Strongest **decoder-free** alternative; the "reconstruct pixels or not?" design fork for manipulation.

### Tier 1 — World models on real robots / manipulation. Reproduce where you have data/hardware.

7. **DayDreamer — World Models for Physical Robot Learning** — Wu, Escontrela, Hafner, Goldberg, Abbeel, 2022 — **[REPRODUCE]** — [2206.14176](https://arxiv.org/abs/2206.14176). Dreamer trained directly on physical robots (arm pick-and-place, quadruped walking in 1h). The most direct precedent for your project.
8. **V-JEPA 2 — Self-Supervised Video Models Enable Understanding, Prediction and Planning** — Assran et al. (Meta), 2025 — **[REPRODUCE — V-JEPA 2-AC is tractable]** — [2506.09985](https://arxiv.org/abs/2506.09985). Frontier of the action-conditioned, non-reconstructing latent world model; zero-shot Franka pick-and-place via MPC; weights public.
9. **Ctrl-World — A Controllable Generative World Model for Robot Manipulation** — Guo, Shi, Chen, Finn (Stanford/Tsinghua), 2025 (ICLR 2026) — **[REPRODUCE]** — [2510.10125](https://arxiv.org/abs/2510.10125). Closest 2025–26 paper to exactly what you're building; public code + weights. Strong interview talking point.
10. **WMPO — World-Model-based Policy Optimization for VLA Models** — 2025 — **[REPRODUCE]** — [2511.09515](https://arxiv.org/abs/2511.09515). On-policy RL for VLA policies *entirely inside a pixel-based world model*; the integration pattern labs ship.

### Tier 2 — Diffusion / generative planning. Reproduce the small ones.

11. **Diffuser — Planning with Diffusion for Flexible Behavior Synthesis** — Janner, Du, Tenenbaum, Levine, 2022 — **[REPRODUCE]** — [2205.09991](https://arxiv.org/abs/2205.09991). Planning as denoising whole trajectories. Small enough to reproduce.
12. **Decision Diffuser — Is Conditional Generative Modeling All You Need?** — Ajay, Du et al., 2022 — **[REPRODUCE]** — [2211.15657](https://arxiv.org/abs/2211.15657). Classifier-free guided diffusion over states; skill/constraint composition at test time; includes Kuka block-stacking.
13. **UniPi — Learning Universal Policies via Text-Guided Video Generation** — Du, Yang et al., 2023 — **[LANDMARK / partial]** — [2302.00111](https://arxiv.org/abs/2302.00111). The cleanest statement of video-generation-as-policy that Genie/Cosmos/V-JEPA build on.

### Tier 3 — Industrial foundation world models. **[LANDMARK]** — read for framing; these define what the target lab cares about.

14. **UniSim — Learning Interactive Real-World Simulators** — Yang, Du, Abbeel et al. (DeepMind), 2023 — **[LANDMARK]** — [2310.06114](https://arxiv.org/abs/2310.06114). "One learned simulator of real-world interaction"; intellectual ancestor of Genie/Cosmos.
15. **Genie — Generative Interactive Environments** — Bruce et al. (DeepMind), 2024 — **[LANDMARK]** — [2402.15391](https://arxiv.org/abs/2402.15391). 11B foundation world model with an unsupervised **latent action model** from unlabeled video. The one Genie paper with a full technical writeup.
16. **Genie 2 / Genie 3** — DeepMind, 2024 / 2025 — **[LANDMARK — blog only, no arxiv]** — [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) · [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/). Real-time (24 fps, 720p), minutes-long-consistent promptable worlds. Know for the interview; cannot reproduce.
17. **Cosmos World Foundation Model Platform for Physical AI** — NVIDIA, 2025 — **[LANDMARK — open weights, partial]** — [2501.03575](https://arxiv.org/abs/2501.03575). Open-weight WFM platform for robotics → you can post-train/fine-tune even if pretraining is out of reach.
18. **Cosmos 3 — Omnimodal World Models for Physical AI** — NVIDIA, June 2026 — **[LANDMARK — current frontier]** — [2606.02800](https://arxiv.org/abs/2606.02800). Single mixture-of-transformers unifying VLM + video generation + world simulator + world-action model; "where the field is right now."
19. **1X World Model Challenge Technical Report** — 2025 — **[LANDMARK / benchmark — REPRODUCE the benchmark]** — [2510.07092](https://arxiv.org/abs/2510.07092). Open humanoid WM benchmark; its **eval methodology** (PSNR future-frame prediction, latent-token cross-entropy) is exactly the kind of world-model evaluation a lab will ask you to design.

*Orientation survey (taxonomy, not a primary read):* **World Models for Robotic Manipulation: A Survey** (2026) — [2606.00113](https://arxiv.org/abs/2606.00113).

---

## 6. Concrete Minimal First Milestone

**Smallest end-to-end version that proves the skill:** Train **one** Dreamer-style / model-based agent on **one** contact-rich manipulation task, beat a model-free floor, and benchmark it against a classical OMPL planner on the same scene.

**Concrete v0:**
1. **Env:** ManiSkill3 `PickCube-v1` (or MuJoCo Playground `PandaPickCube` for the lightest possible bring-up) with RGBD + state observations.
2. **Method:** DreamerV3 (`danijar/dreamerv3` JAX, or `NM512/dreamerv3-torch` if you prefer PyTorch) **or** TD-MPC2 (`nicklashansen/tdmpc2`, which already speaks ManiSkill2 tasks). TD-MPC2 is the lowest-friction starting point because the ManiSkill integration exists.
3. **Baseline floor:** SAC and/or PPO (ManiSkill ships PPO) on the identical task.
4. **Classical baseline:** MoveIt 2 / OMPL RRT-Connect on the same PickCube scene geometry — report success rate + planning time + path quality.
5. **Deliverable:** a success-rate-vs-env-steps curve (start with 3 seeds for v0, scale to ≥5 for the final write-up; report IQM + CI via rliable) showing your world model is more sample-efficient than SAC; plus a small crossover table vs the classical planner. Track everything in W&B.

**Rough compute / time budget (single A100):**
- Single-task DreamerV3 / TD-MPC2 on one manipulation task: typically **hours to ~1–2 days** of wall-clock per seed (a few × 10^5–10^6 env steps). TD-MPC2 single-task needs only ~8 GB GPU.
- Prototype the loop first on MuJoCo Playground (JIT compile 1–3 min, then very fast) before scaling on ManiSkill3.
- Full v0 (one task, ~3 seeds, all baselines, classical comparison, write-up): realistically **~1–2 weeks of part-time work**, dominated by environment/install bring-up and the OMPL harness, not GPU time.

**Then iterate toward the portfolio piece:** add a second contact-rich task (PegInsertion/StackCube), expand to ≥5 seeds with rliable, add DreamerV3-vs-TD-MPC2 as a head-to-head, and write the **classical-vs-learned crossover analysis** as the headline differentiator. **Stretch:** swap in / add V-JEPA 2-AC latent MPC for the "modern world model" signal, or fine-tune iVideoGPT as a generative manipulation world model.

**Honest difficulty notes:**
- Environment/sim install is the most common time sink (Vulkan for ManiSkill render; JAX↔CUDA matching for DreamerV3/MJX). Budget for it.
- The OMPL/MoveIt 2 harness on the *same* scene geometry as the sim is fiddly but is your differentiator — worth the effort.
- Don't try to *beat* TD-MPC2; contextualize against it. Don't try to train any video world model from scratch.
