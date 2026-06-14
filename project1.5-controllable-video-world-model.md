# Project #1.5 — Controllable Video World Model

**Solo · single A100 80 GB · ~3–4 weeks**
**Target role:** Runway — *Member of Technical Staff, Research Engineer* (world models). Double-counts for **Odyssey** (interactive video world models).

> **Verification stance.** Every model ID, arXiv number, license, and VRAM figure below was live-verified by direct fetch on **2026-06-14** (GitHub / HuggingFace / arXiv). Figures I could not confirm from a primary source are marked **[inferred]** or **[unverified]**. The one-line rule: *don't pretrain a video world model (datacenter-only) — adapt, post-train, and evaluate one.* That is precisely the slice of Runway's full-stack research-engineering role this project demonstrates.

---

## 0. The gap this closes

The portfolio already covers post-training (RLVR), eval-driven development, data pipelines, distributed A100 training, and prototype→prod. The single missing axis for Runway/Odyssey is **large-scale controllable *video-generation* world models**. Runway is now explicitly a world-models company: it shipped **GWM-1** (General World Model, autoregressive, built on Gen-4.5, real-time interactive via *camera pose, events, robot pose, speech*), with post-trained **GWM Worlds / Avatars / Robotics** variants, and opened a London world-models HQ. ([GWM-1](https://runwayml.com/research/introducing-runway-gwm-1) · [General World Models](https://runwayml.com/research/introducing-general-world-models) · [London HQ](https://runwayml.com/news/runway-opens-london-hq))

This project = **take an open, action-conditioned video world model and teach it a *new* behavior** (a robot-manipulation action space it has never seen), with a real **post-training method** and a **purpose-built eval harness** — the exact loop in Runway's "Datasets" JD.

---

## 1. Verified base — LingBot-World

| Item | Verified value | Source |
|---|---|---|
| Repo | `github.com/robbyant/lingbot-world` | [GitHub](https://github.com/robbyant/lingbot-world) |
| Tech report | **arXiv 2601.20540** — "Advancing Open-source World Models" (Robbyant, released Jan 29 2026) | [abs](https://arxiv.org/abs/2601.20540) · [html](https://arxiv.org/html/2601.20540v1) |
| License | **Apache-2.0** ✅ (genuinely permissive — no geo/MAU/use restrictions) | [README](https://github.com/robbyant/lingbot-world/blob/main/README.md) |
| Base | **Wan2.2 14B image-to-video DiT**, extended to **2-expert MoE = 28B total**, **1 expert (~14B) active per denoising step** (high-noise / low-noise expert split) | [arXiv html](https://arxiv.org/html/2601.20540v1) |
| Variants | **Base (Act)** action-conditioned · **Base (Cam)** camera-pose · **Fast** (KV-cache + DMD distillation, 16 fps @ 480p) · community **4-bit NF4** quant | README |
| Model IDs | `robbyant/lingbot-world-base-act` · `robbyant/lingbot-world-base-cam` · `robbyant/lingbot-world-fast` · `cahlen/lingbot-world-base-cam-nf4` (community) | [act](https://huggingface.co/robbyant/lingbot-world-base-act) · [cam](https://huggingface.co/robbyant/lingbot-world-base-cam) · [fast](https://huggingface.co/robbyant/lingbot-world-fast) · [nf4](https://huggingface.co/cahlen/lingbot-world-base-cam-nf4) |
| Resolutions / horizon | 480p (480×832) & 720p (720×1280); 121–961 frames (≈ up to 1 min @ 16 fps) | README |

### 1.1 Action-conditioning interface (verified — this is load-bearing)

Conditioning is **hybrid**, injected via **adaptive layer-norm (AdaLN)** inside the DiT blocks:
- **Camera / continuous motion** → **Plücker embeddings** (via a Plücker encoder) — geometric, suited to continuous 3D transforms.
- **Discrete actions** → **multi-hot vectors over keys (W/A/S/D-style)**.

The README's inference interface exposes both: camera as `intrinsics.npy [num_frames,4]=(fx,fy,cx,cy)` + `poses.npy [num_frames,4,4]` (OpenCV convention), and discrete actions as strings like `w-10,a-10,d-10,iw-15,none-10,j-10,l-10,s-15` (key-duration tokens) **or** pre-extracted action vectors. Camera params can be auto-extracted with **ViPE / MegaSAM**.

> **Why this matters for the build:** the report states post-training **freezes the main DiT and fine-tunes only the action-adapter** (action-embedding projections + AdaLN params). That is the cheap, single-A100-feasible recipe — and it's the model authors' *own* method, so it's the right thing to reproduce and extend. ([arXiv html, §post-training](https://arxiv.org/html/2601.20540v1))

### 1.2 VRAM — the critical feasibility trap ⚠️

| Mode | VRAM | Single A100 80 GB? | Source |
|---|---|---|---|
| Unquantized inference (full 28B MoE: T5 + VAE + 2 DiT experts) | **~85 GB** | ❌ **does not fit** | [nf4 card](https://huggingface.co/cahlen/lingbot-world-base-cam-nf4) |
| **NF4 4-bit inference** | **~32 GB** (≈3.9× compression) | ✅ (4090/5090/A100) — **inference-only** | nf4 card |
| Repo reference inference | 8-GPU `torchrun` examples; `--t5_cpu` offload flag | multi-GPU assumed by default | README |
| LoRA / adapter fine-tune | **Not published** by Robbyant | **[inferred feasible — see below]** | — |

**Honest read:** the *unquantized* 28B model is ~85 GB → it **will not even do inference on one 80 GB A100** without offloading. Two genuine paths exist on a single card:
1. **Inference** via the NF4 quant (~32 GB), and/or DiT-expert + T5-CPU offload to squeeze the unquantized model onto 80 GB with offloading.
2. **Adapter post-training:** because only the action-adapter (AdaLN + projection layers, a tiny fraction of params) is trained while the 14B-active backbone is **frozen** (and can be held in 4-bit / bf16 with gradient checkpointing + DiT offload), this is **[inferred]** feasible on one A100 80 GB for **480p, short clips (≈49–81 frames), small batch, gradient checkpointing**. No official single-A100 fine-tune recipe is published — **this is a real risk to de-risk in Milestone 0** (see §5). If LingBot adapter-tuning won't fit, fall back to **Cosmos-Predict2.5-2B** (LoRA verified ~20 GB) or **Wan2.2-TI2V-5B** (LoRA ~31 GB) — see §2.

---

## 2. Base comparison (single A100 80 GB)

All verified 2026-06-14. "Control OOTB" = controllability available out of the box.

| Model | Params | License | Control OOTB | LoRA/adapter on 1×A100 80 GB | Robotics relevance | Verdict |
|---|---|---|---|---|---|---|
| **LingBot-World** | 28B MoE (14B active) | **Apache-2.0** | **Action (WASD multi-hot) + Camera (Plücker/pose)** built-in | **Adapter-only [inferred yes]**; unquant inf ❌ (85 GB), NF4 inf ✅ (32 GB) | High (embodied/sim sandbox) | **★ Primary** — only open model with *both* Act + Cam baked in + Apache |
| **Cosmos-Predict2.5-2B** | 2B | **OpenMDW-1.1** (permissive, commercial OK) | Text/Image/Video→World (no native action axis) | **✅ LoRA ~20 GB; full FT ~50 GB** (official toolkit, "min one 80 GB GPU") | Very high (Physical AI, GR1/DROID FT blog) | **★ Frontier-reference + fallback** |
| Cosmos3-Nano | **16B** omnimodal (8B AR + 8B diff, MoT) | OpenMDW-1.1 | Omnimodal incl. **action** I/O; fwd+inverse dynamics | ❌ tested GB200/H100; SFT used 8×H100 | Highest (DROID policy variant) | Cite as **SOTA frontier**, not a build target |
| Cosmos3-Nano-Policy-DROID | 16B | OpenMDW-1.1 | lang+obs→**robot action trajectories** (not video) | ❌ multi-H100 | Highest | Reference only (outputs actions, not video) |
| **Wan2.2-TI2V-5B** | 5B dense | **Apache-2.0** | text+image→video (no action) | **✅ LoRA ~31 GB** (DiffSynth-Studio) | Medium | Strong fallback if LingBot adapter won't fit |
| Wan2.2-A14B (MoE) | 27B/14B active | Apache-2.0 | I2V + VACE-Fun camera/pose | LoRA tight ~75 GB; full FT multi-GPU | Medium | The base under LingBot; richer control but tight |
| CogVideoX-2B / 5B | 2B / 5B | 2B Apache-2.0; 5B custom | I2V; Fun pose/depth/canny/camera (separate) | ✅ 2B LoRA ~47 GB; 5B ~63 GB | Low–Med | Easy but no native action |
| LTX-Video (2B/13B) | 2B / 13B | LTXV Open Weights (free <$10M ARR) | I2V + **Depth/Pose/Canny IC-LoRAs**; real-time | ✅ official LTX-Video-Trainer | Low–Med | Best *generic* solo-FT control story; no robot-action |
| HunyuanVideo | 13B+ | Tencent — **excludes EU/UK/SK** ⚠️ | I2V only | LoRA via musubi (24 GB+); full FT multi-GPU | Low | **Avoid** (geo-license risk for UK/EU employers) |
| Mochi-1 | 10B | Apache-2.0 | **T2V only — none** | ✅ LoRA "exactly one A100 80 GB" (tight) | Low | No control axis |
| Open-Sora 2.0 / 1.2 | 11B / 1.1B | Apache-2.0 | I2V + motion-score | 1.2 ✅; 2.0 borderline | Low | 1.1B is solo-FT-able but no diffusers / no action |

**Sources:** [Cosmos-Predict2.5 LoRA doc](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_cosmos_nemo_assets_lora.md) · [Cosmos robot-video FT blog](https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation) · [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) · [OpenMDW-1.1 adoption](https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1.1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families) · [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) · [CogVideoX VRAM](https://github.com/huggingface/diffusers/blob/main/docs/source/en/training/cogvideox.md) · [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) · [HunyuanVideo license](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/LICENSE.txt)

**Recommendation:** Build on **LingBot-World Base(Act)** — the only open model with action *and* camera conditioning baked in, Apache-2.0, and an explicit freeze-backbone/train-adapter recipe. Keep **Cosmos-Predict2.5-2B** as both the *frontier reference* (the 16B Cosmos3-Nano / Policy-DROID is the SOTA you're approximating at small scale) **and the de-risked fallback** if LingBot adapter-tuning blows the 80 GB budget in Milestone 0.

---

## 3. The build (Runway-shaped)

**Thesis:** *Teach LingBot-World a new behavior — a robot-manipulation action space — by post-training its action adapter, then prove with eval that the generated video actually obeys the action.*

### 3.1 Data pipeline (the candidate's strength)

Two sources, picked for single-GPU feasibility:

- **Option A — ManiSkill3 (primary, fully solo-controlled).** Render scripted manipulation rollouts: each `env.step(action)` returns the RGB frame *and* the action that produced it, giving exact `(frame, action)` pairs for free, GPU-side (~30k+ FPS, hundreds of envs at single-digit GB). Pick an explicit control mode (`pd_ee_delta_pose` or `pd_joint_pos`) so the conditioning action is well-defined; map it into LingBot's action-token format. ([ManiSkill](https://github.com/haosulab/ManiSkill) · [paper 2410.00425](https://arxiv.org/pdf/2410.00425)) — *leverages the existing ManiSkill setup.*
- **Option B — DROID (real-robot stretch).** 76k trajectories / 350 hrs, 3 synced RGB + robot state + control commands in multiple action spaces, **CC-BY-4.0**, 100-episode 2 GB sample for prototyping. ([droid-dataset.github.io](https://droid-dataset.github.io/)) Open-X-Embodiment (Apache-2.0, 1M+ traj) is the scale option but needs action-schema normalization. ([OXE](https://robotics-transformer-x.github.io/))

Pipeline sketch (Hydra-configured, the standard project layout):
1. **Rollout/ingest** → ManiSkill scripted policies (or DROID RLDS shards).
2. **Action encoding** → map continuous EE/joint deltas → Plücker embeddings (camera-frame motion) + discretize gripper/mode → multi-hot tokens matching LingBot's adapter input.
3. **Caption/condition** → static-scene + dense-temporal captions (LingBot's data engine separates layout from motion); auto-caption with a VLM.
4. **Filter/QC** → drop failed episodes, dedup, balance action distribution (Runway JD: "filtering and quality control").
5. **Pack** → 480p clips of 49–81 frames, WebDataset shards.

### 3.2 Post-training method

**Primary (verified-cheap): freeze-backbone adapter fine-tune.** Reproduce LingBot's own recipe — freeze the 14B-active DiT, train only the action-embedding projections + AdaLN params on the new ManiSkill action space. Backbone held in 4-bit/bf16 + gradient checkpointing + DiT/T5 offload. This is the minimal viable "teach a new behavior" result.

**Stretch (the Runway differentiator — RL/DPO post-training):** once an eval reward exists (§4), add **RL-from-eval-reward** or **DPO-style** preference post-training on the adapter: sample pairs of rollouts under the *same* action, score with the action-following metric (§4.2), and prefer the higher-fidelity one. This directly hits the JD's "SFT/RL post-training" bullet and reuses the candidate's RLVR background. Keep it adapter-scoped to stay in budget.

---

## 4. Eval design (core strength + explicit Runway bullet)

A small custom harness (`evals/`) wrapping real, named metrics across four axes + one holistic leaderboard. The discriminating axis most repos skip is **action-following** — lead with it.

### 4.1 Temporal consistency / generation quality
- **cd-fvd** (`pip install cd-fvd`, VideoMAE-v2 features — fixes classic FVD content-bias) and **FVMD** (`pip install fvmd`, motion-specific) for distribution + motion quality. ([cd-fvd 2404.12391](https://arxiv.org/abs/2404.12391) · [FVMD 2407.16124](https://arxiv.org/abs/2407.16124))
- **VBench-2.0** sub-metrics (subject/background consistency, motion smoothness, dynamic degree, temporal flickering) as the holistic quality layer. ([VBench-2.0 2503.21755](https://arxiv.org/abs/2503.21755) · [leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard))

### 4.2 Action-following fidelity ★ (the headline metric)
- **Δt PSNR / controllability score** (Genie): `PSNR(GT-action) − PSNR(random-action)`; larger ⇒ more controllable — the cleanest "does it obey" number. ([Genie 2402.15391](https://arxiv.org/abs/2402.15391))
- **IDM action-error / Action Error Ratio** (AVID): run an inverse-dynamics model on the generated video, recover the action, compare to the conditioning action; report gen-error ÷ real-error. ([AVID 2410.12822](https://arxiv.org/abs/2410.12822) · Vista "Trajectory Difference" [2405.17398](https://arxiv.org/abs/2405.17398))
- For ManiSkill data the IDM is cheap to train (we have ground-truth actions), making this axis the project's signature contribution.

### 4.3 Camera-control accuracy
- **RotErr** (geodesic SO(3)) + **TransErr** (translation L2) from CameraCtrl, with trajectories Sim(3)/Umeyama-aligned; estimate poses from generated video with **ViPE** (NVIDIA, pose+intrinsics+metric depth in one pass — the 2026 SOTA backend, also LingBot's own annotator). ([CameraCtrl 2404.02101](https://arxiv.org/abs/2404.02101) · [ViPE](https://github.com/nv-tlabs/vipe))

### 4.4 Physical plausibility
- **VideoPhy-2-AutoEval** (HF auto-scorer) and/or **Physics-IQ** (DeepMind, scriptable scorer). ([VideoPhy](https://github.com/Hritikbansal/videophy) · [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark))

### 4.5 Holistic world-model leaderboard
- **WorldScore** (Controllability + Quality + Dynamics, HF leaderboard) and/or **WorldModelBench** (instruction-following + common-sense + physical-adherence, 2B auto-judge). ([WorldScore 2504.00983](https://arxiv.org/abs/2504.00983) · [WorldModelBench](https://github.com/WorldModelBench-Team/WorldModelBench))

---

## 5. First milestone + compute budget (1×A100 80 GB)

**Milestone 0 — de-risk VRAM (days 1–2, do this FIRST).** Stand up NF4 inference (~32 GB ✅) and confirm the action interface end-to-end on stock examples. Then probe **adapter fine-tune** memory: 480p, 49 frames, batch 1, gradient checkpointing, frozen backbone in 4-bit + DiT/T5 offload. **If it fits → proceed on LingBot. If not → pivot to Cosmos-Predict2.5-2B (LoRA ~20 GB, verified) or Wan2.2-TI2V-5B (~31 GB).** This gate is non-negotiable given the ~85 GB unquantized footprint.

**Milestone 1 — minimal controllable result (weeks 1–2):**
1. ManiSkill pipeline → ~5–20k `(clip, action)` pairs, one task family (e.g. PickCube), `pd_ee_delta_pose`.
2. Adapter fine-tune on the new action space; generate rollouts under held-out action sequences.
3. Eval harness MVP: **Δt PSNR + IDM action-error** (action-following) + **cd-fvd** (quality). Baseline = frozen pretrained vs adapter-tuned; show the action-following metric *moves*.

**Realistic vs not:**
- ✅ Feasible solo on 1×A100 80 GB: NF4 inference; freeze-backbone adapter post-training at 480p/short clips; the full eval harness; ManiSkill data gen.
- ⚠️ Tight/uncertain: unquantized inference (needs offload); 720p or long-horizon adapter training; DPO/RL stretch (do adapter-scoped, small).
- ❌ Not solo-feasible (state this up front): full fine-tune of the 28B MoE; pretraining any video WM; fine-tuning Cosmos3-Nano-16B / HunyuanVideo-13B (multi-H100).

**Compute envelope [inferred]:** Milestone 0–1 ≈ a few GPU-days (adapter steps are cheap with a frozen backbone; the cost is data gen + eval inference passes, not gradient steps).

---

## 6. JD mapping + portfolio composition

### Runway JD → demonstrated

| Runway bullet | What #1.5 demonstrates |
|---|---|
| "Teach world models new behaviors — action following, scene manipulation, camera control" | Post-train LingBot's action adapter on a *new* ManiSkill/DROID action space; camera axis via Plücker/pose conditioning |
| "SFT / RL post-training" | Freeze-backbone adapter SFT (primary) + RL/DPO-from-eval-reward (stretch) — reuses RLVR background |
| "Design evaluations that measure model capabilities" | Custom harness: Δt PSNR + IDM action-error + RotErr/TransErr + cd-fvd/FVMD + VideoPhy/WorldScore |
| "Robust data pipelines (synthetic gen, filtering, QC)" | ManiSkill synthetic-rollout pipeline + DROID ingest, action-encoding, captioning, QC, WebDataset packing |
| "Prototype → production" | Standard sweep-orchestration layout (Hydra configs, `results.jsonl`, W&B tracking, detached daemons) |
| "Multimodal generative models (video/image)" | Video diffusion world model end-to-end |

### Composition with the rest of the portfolio
- **Project #1 (compact world model + classical crossover):** #1.5 is the *generative-video* counterpart — #1 argues compact/efficient WMs, #1.5 shows fluency with the large-scale video-diffusion stack the field actually ships. Together: "I understand WMs from compact-latent to full-video."
- **Project #4 (semantic-predictive 4D fusion / simulation):** #1.5's action-conditioned rollouts + ViPE pose/depth feed directly into #4's 4D fusion story; shared eval primitives (ViPE, FVMD).
- **Odyssey double-count:** Odyssey builds interactive video world models (next-frame from scene-state + user input, ~40 ms, browser-served). LingBot's **Fast** variant (KV-cache + DMD distillation, 16 fps, <1 s latency) is the same interactive-streaming regime — #1.5 demonstrates exactly Odyssey's tech direction with no extra work.

---

## 7. Honesty ledger (traps & unverified)

- **VRAM trap:** unquantized LingBot ≈ 85 GB → no single-A100 unquantized inference without offload; NF4 (32 GB) is inference-only. Adapter-tune feasibility on 80 GB is **[inferred]**, gated by Milestone 0.
- **No official LingBot fine-tune/LoRA script** is published — only the report's description of freeze-backbone adapter training. Expect to write the training loop.
- **Cosmos license:** OpenMDW-1.1 is permissive (commercial OK) but the exact acceptable-use clause text was **[unverified]** — confirm before any commercial claim. Cosmos3-Nano fine-tune VRAM is **unpublished**.
- **HunyuanVideo geo-license** excludes EU/UK/SK — avoid given UK/EU target employers.
- A few 2026 eval-benchmark arXiv IDs sit at the edge of the knowledge window — re-fetch before citing specific *numbers* from them; the metric *taxonomy* (Δt PSNR, RotErr/TransErr, cd-fvd, FVMD, ViPE) is solid.
