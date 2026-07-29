# Action-conditioned diffusion dynamics model — from-scratch generative world model

**Status:** 📋 PLANNED (2026-07-29)
**Execution window:** 2026-07-29 → 2026-08-02 (Milestone 1)
**Hardware:** 1×A100 80 GB (`pi-a100-80gb`)

> **One-line:** Build and train a small, from-scratch action-conditioned diffusion dynamics model on ManiSkill3 state transitions — proving generative-model implementation skill, not just black-box benchmarking — then add RGB pixel-observation training as Milestone 2 and fidelity/calibration metrics as Milestone 3.

## Why

The just-completed project arc ([#1](../project1-world-models-manipulation-SOTA.md) → [`plugchargerdense.md`](../experiments/plugchargerdense.md)) used world models *as black boxes* — TD-MPC2 and DreamerV3 from ManiSkill/danijar repos. That demonstrated benchmarking rigor but not generative-model implementation, which is the core skill for the World Models Research Engineer role. The SOTA doc itself flags this next step at line 59: *"Swap Dreamer's RSSM for a video/diffusion WM"* — this spec executes that.

This expansion also addresses the **RSSM representation wall** diagnosed in the PlugCharger study (`plugchargerdense.md`): the RSSM's Gaussian latent can only represent a single future outcome, so the planner cannot reason about multimodal contact outcomes (grasp succeeds vs fails). A diffusion model represents the full future-state *distribution* — the planner can sample diverse outcomes and plan for robustness.

| Job requirement | What this builds | Status quo |
|---|---|---|
| Diffusion / transformer world models | Conditional diffusion dynamics model (MLP→DiT) | ❌ None — TD-MPC2 uses deterministic/Gaussian latents |
| Action-conditioned prediction | `(s_t, a_t) → diffusion denoising → s_{t+1}` | ❌ Delegated to black-box TD-MPC2 |
| Custom training loop + data pipeline | Streaming dataloader, W&B logging, checkpointing | ❌ All delegated to subprocess calls |
| Large-scale distributed training (Milestone 4) | FSDP/DDP-ready trainer skeleton | ❌ Single GPU only |
| Video prediction (Milestone 2) | RGB-pixel latent diffusion | ❌ State-obs only |
| Fidelity / calibration (Milestone 3) | Prediction intervals, divergence detection | ❌ Not built |

## Hypothesis / goal

A small (≤50M param) action-conditioned diffusion model trained on ManiSkill PegInsertionSide rollouts can predict next-state distributions at least as accurately as TD-MPC2's deterministic/Gaussian dynamics head (measured by MSE, negative log-likelihood, and rollout divergence over 5, 10, 20-step horizons), despite having ≥10× fewer parameters and zero task-specific inductive bias. If true, the diffusion formulation's ability to represent multimodal futures (contact outcomes, grasp success/failure) gives it an advantage the deterministic baseline cannot capture — measurable as lower NLL on held-out transitions.

**Milestone 1 gates (state-based):**
- **GO:** diffusion model matches or beats TD-MPC2 dynamics on 1-step MSE and 20-step rollout divergence on PegInsertionSide. Code is clean, tested, and reproducible.
- **PARTIAL:** diffusion model matches TD-MPC2 on 1-step MSE but diverges faster over horizons. The calibration/fidelity story (Milestone 3) becomes the differentiator.
- **NO-GO:** diffusion model underperforms TD-MPC2 by >2× on 1-step MSE, indicating a fundamental architecture or training bug.

**Milestone 2 gate (pixel-based):**
- **GO:** trained on RGB frames from ManiSkill, the model produces visually plausible action-conditioned next-frame predictions (FVD < reference threshold, action-following ΔPSNR > 0).
- **NO-GO:** training is unstable or the model learns a static-scene prior that ignores actions.

**Milestone 3 gate (fidelity/calibration):**
- **GO:** calibrated prediction intervals that correlate with empirical error — a trustworthiness score deployed on the M1 and M2 models.
- **PARTIAL:** prediction intervals exist but are miscalibrated (CDF error > 0.1).

## Method

### Milestone 1 — State-based diffusion dynamics

```
(s_t, a_t) → MLP/DiT denoiser → s_{t+1}
```

**Data pipeline:**
1. Roll out trained TD-MPC2 / PPO policies on `PickCube-v1` and `PegInsertionSide-v1` at high FPS (GPU envs).
2. Save `(s_t, a_t, s_{t+1}, reward, done)` transitions as flat `.npy` shards.
3. Target: ~1–5M transitions (hours of sim time, cheap via parallel envs).
4. Streaming dataset with `memmap` + prefetch, no loading all into RAM.

**Architecture:**
- **Primary:** Conditional MLP diffusion — 6–8 layer MLP with SiLU, LayerNorm, residual connections. Condition on `(s_t, a_t)` via concatenation + FiLM modulation from noise timestep embedding.
- **Stretch:** Small DiT (4 blocks, 4 heads, 256 dim) — shows transformer skill directly.
- **Diffusion:** DDPM (Ho et al., 2020), cosine noise schedule, 1000 timesteps (reduced to 100 at inference). Predict epsilon (standard), with MSE loss.

**Training:**
- Batch size 1024–4096 (state vectors are small — ~50 dim)
- AdamW, lr 3e-4, cosine decay, 500k gradient steps
- Gradient clipping at 1.0
- Validation on held-out episodes every 10k steps
- W&B: train/val MSE, NLL, rollout divergence curves

**Evaluation:**
- **1-step:** MSE, MAE, NLL on held-out transitions
- **Multi-step:** rollout `(s_t, a_t) → s' → (s', a_{t+1}) → s'' ...` for 5/10/20 steps, measuring position drift, contact-mode accuracy, divergence from ground-truth trajectory
- **Baseline:** TD-MPC2's dynamics head evaluated on same data

### Milestone 2 — RGB pixel-observation latent diffusion

1. Switch ManiSkill env from `obs=state` to `obs=rgbd` (64×64 or 128×128).
2. Encode frames with a frozen pretrained VAE (e.g., Stable Diffusion VAE or a small from-scratch VAE).
3. Train latent diffusion model conditioned on `(latent_t, a_t)` to predict `latent_{t+1}`.
4. Decode latents back to pixels for qualitative eval.
5. Metrics: FVD (VideoMAE-v2 features), LPIPS, ΔPSNR action-following (Genie-style), PSNR on held-out frames.

### Milestone 3 — Fidelity / calibration

1. **Ensemble uncertainty:** Train 5 diffusion models with different seeds → prediction variance as uncertainty proxy.
2. **Conformal prediction:** Calibrate prediction intervals on a held-out set. Measure empirical coverage vs nominal coverage.
3. **Divergence detector:** Train a binary classifier to predict when the model's rollout diverges from reality (trained on early timesteps of divergent vs convergent rollouts).
4. **Deployment:** The divergence detector runs alongside any policy evaluation, flagging untrustworthy predictions.

### Milestone 4 — Distributed training infrastructure (stretch)

Refactor the training loop to support:
- **FSDP** for scaling beyond single GPU
- **WebDataset / streaming** from cloud storage
- **Async checkpointing** (save while training continues)
- **Gradient accumulation** for large effective batch sizes

This is infrastructure code that demonstrates scalability without needing a multi-GPU cluster.

## Dataset & compute budget

| Stage | Data needed | GPU-hours |
|---|---|---|
| M1: collect transitions from trained policies | ~1–5M (s,a,s') from 4× Policy × Task | 2–4 h (sim at 30k+ FPS) |
| M1: train state diffusion | 1–5M transitions | 2–6 h (state vectors are tiny) |
| M2: collect RGB rollouts | ~500k–1M RGB 64×64 frames | 4–8 h (render is slower) |
| M2: train latent diffusion | 500k–1M latent frames | 12–24 h (bigger model) |
| M3: ensemble (5× M1) | Same as M1 | 10–30 h total |

**Total M1 envelope:** ~8–10 GPU-hours, easily within a single overnight run.

## Deliverables

1. **Code:**
   - `experiments/diffusion_wm/collect.py` — rollout collector
   - `experiments/diffusion_wm/dataset.py` — streaming dataset
   - `experiments/diffusion_wm/model.py` — conditional diffusion dynamics (MLP + DiT variants)
   - `experiments/diffusion_wm/train.py` — training loop with W&B
   - `experiments/diffusion_wm/eval.py` — eval harness (1-step, multi-step, vs TD-MPC2)
    - `experiments/diffusion_wm/fidelity.py` — calibration + divergence detector (M3)

2. **Results:**
   - PegInsertionSide: diffusion vs TD-MPC2 dynamics comparison (table + learning curve)
   - PickCube: same comparison (easier task, should be near-perfect for both)

3. **Design docs:** This spec + per-phase gate decisions
4. **Experiment writeup:** `docs/experiments/diffusion-wm-phase1.md`

## Honest framing

This is a **competence-builder**, not a research contribution. Every building block (DDPM, conditioned on observations, MLP/DiT backbone) is standard. The point is to have *shipped code* that demonstrates the full stack — data pipeline, model implementation, distributed-ready training loop, and a rigorous evaluation against a credible baseline (TD-MPC2 dynamics).

The hardest risk is **Milestone 2 VRAM**: RGB training on 1×A100 80 GB requires a small VAE, gradient checkpointing, and likely gradient accumulation. If it doesn't fit, defer to a smaller image size (32×32) or a shallower VAE.

## Relation to existing projects

- **Project #1 (compact world model baseline):** Uses TD-MPC2/DreamerV3 as black boxes. This spec builds a *transparent, from-scratch* alternative to their dynamics heads — directly comparable on MSE, NLL, and rollout divergence. Critically, diffusion's ability to represent *multimodal* future distributions (grasp succeeds/fails) addresses the RSSM coverage wall diagnosed in `plugchargerdense.md`, which limits TD-MPC2's planner to a single imagined outcome.
- **Project #1.5 (controllable video world model):** Milestone 2 is the *small-scale executable version* of the same skill — action-conditioned video prediction at the latent level, deferring the full video-DiT scale to LingBot/Cosmos.
- **Project #4 (capacity-sensitive world model, capstone scenario):** If used to replace TD-MPC2's dynamics, the diffusion WM becomes the bottleneck for counterfactual rollouts during MBRL planning. The capstone can then measure how sampling cost and representation capacity trade off against plan quality — directly connecting generative-model fidelity to downstream task success.
- **scene_rep package:** Milestone 2 reuses LPIPS from scene_rep; Milestone 3's calibration methodology extends the measurement philosophy.

## Env & assets

- **Simulator:** ManiSkill3 (already working on A100)
- **Checkpoints:** TD-MPC2 checkpoints from existing PegInsertionSide + PickCube runs (or re-run if needed)
- **Compute:** `pi-a100-80gb`, same env as previous experiments
- **Tracking:** W&B `chaleong/wm-manip`
- **Python:** 3.11+, PyTorch 2.4+, `torchvision`, `wandb`, `einops`
