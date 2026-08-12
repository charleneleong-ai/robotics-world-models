# Dynamic 4D Gaussian Splatting — deformation-field + monocular-motion pipeline, temporal eval (Milestone 2)

**Status:** 📋 PLANNED (spec, not yet executed) — kicks off after the [Milestone 1 GO](../experiments/semantic-3dgs-m1.md). Two reproduction bars against published numbers (same move M1 made when ScanNet++ was access-gated): synthetic **D-NeRF → 4DGS-Wu**, then real monocular **DyCheck-iPhone → Shape-of-Motion**. Extends the M1 harness with a temporal-consistency metric. **Date:** 2026-07-28.

> **One-line:** take one static-3DGS substrate to **4D**: a synthetic D-NeRF clip through a deformation-field splatter (4DGS-Wu) to stand up the temporal pipeline, then one real monocular clip through Shape-of-Motion's explicit SE(3) motion bases — reporting per-frame PSNR/SSIM/LPIPS **plus a temporal metric (tOF / tLPIPS)** on held-out frames, and visualizing the motion bases. The smallest slice that proves *time-consistent* scene representation, and the direct substrate for the phase-3 Gaussian world model.

## Why

This is the temporal step of the [[project_robotics_world_models_pivot]] representation stack: [Milestone 1](2026-07-21-semantic-3dgs-scene-representation.md) proved a static 3D Gaussian field plus a held-out-view eval harness ([`scene_rep/metrics.py`](../../scene_rep/metrics.py)) work end-to-end. A world model predicts a scene's *future*, so the representation has to carry **time** before anything predicts on it — a static field is the wrong input to phase 3. M2 adds motion two ways that bracket the difficulty range: a **deformation field** over a canonical set of Gaussians (easy, near-multiview supervision) and **explicit per-Gaussian trajectories** from a single real camera (hard, ill-posed). Both extend the M1 substrate rather than replace it; the reusable asset that carries forward is the **temporal metric** bolted onto the existing harness.

Full 4D landscape (deformation-field vs explicit-motion vs feed-forward, datasets, metrics) in [`project2-3d4d-scene-representation-SOTA.md`](../../project2-3d4d-scene-representation-SOTA.md); this spec is its executable Milestone 2, not a re-derivation.

## Hypothesis / goal

A self-run 4DGS reconstruction that **reproduces 4DGS-Wu's published D-NeRF per-frame fidelity** (~33–34 PSNR on the synthetic test split) **plus** a Shape-of-Motion run that **reproduces its DyCheck-iPhone number** (~16–17 mPSNR on covisibility-masked held-out views), each carrying a **stable temporal metric**, is a credible 4D-competence signal — and a more honest one than a single hero clip. If the temporal metric on the *real* clip degrades far below the synthetic while per-frame PSNR looks fine, that is the exact failure MonoDyGauBench documents (per-frame-sharp but temporally-brittle) — and the metric must catch it, not the eye.

## Method

Five steps, bracketing easy→hard. First and last reuse M1 discipline (reproduce a published number, score held-out only); the middle three are the new temporal skill, bounded to "extend the field over time, don't invent a method":

1. **Synthetic deformation field.** One [D-NeRF](https://arxiv.org/abs/2011.13961) scene (`jumpingjacks` or `bouncingballs` — ships camera poses + timestamps, zero pose wrangling) through **4DGS-Wu** ([arXiv 2310.08528](https://arxiv.org/abs/2310.08528), HexPlane deformation over canonical Gaussians). Output: per-frame PSNR/SSIM/LPIPS on the held-out test split vs the paper's number, train time, render FPS. Proves the temporal substrate runs and reproduces. **Deformable-3DGS** ([arXiv 2309.13101](https://arxiv.org/abs/2309.13101), MLP deformation) is the pinned fallback if the HexPlane repo fights the toolchain.
2. **Temporal metric → harness.** Extend [`scene_rep/metrics.py`](../../scene_rep/metrics.py) with **tOF** (flow-warp error between consecutive frames, Chu et al. TecoGAN [arXiv 1811.09393](https://arxiv.org/abs/1811.09393): `‖OF(gt_{t-1},gt_t) − OF(pred_{t-1},pred_t)‖₁`) and **tLPIPS** (consecutive-frame perceptual-distance gap, `|LPIPS(pred_{t-1},pred_t) − LPIPS(gt_{t-1},gt_t)|`). Add a `TemporalMetrics` row alongside `ViewMetrics` and fold it into `combined_table`. Pure-numpy tOF (reuse a light flow backend); tLPIPS reuses the existing lazy-torch `lpips`. Known-answer tests (static clip → tOF≈0; frame-shuffle → tOF spikes) mirror the 20-test M1 validation.
3. **Real monocular pipeline.** One [DyCheck iPhone](https://arxiv.org/abs/2210.13445) clip through **Shape-of-Motion** ([arXiv 2407.13764](https://arxiv.org/abs/2407.13764), ICCV'25 — compact set of SE(3) motion bases, each Gaussian a linear combination → persistent explicit trajectories). The time sink is **preprocessing, not training**: monocular depth ([Depth-Anything-V2](https://arxiv.org/abs/2406.09414) / [UniDepth](https://arxiv.org/abs/2403.18913)) + long-range 2D tracks ([CoTracker](https://arxiv.org/abs/2307.07635) / [TAPIR](https://arxiv.org/abs/2306.14435)) + foreground masks (the M1 SAM stack). Report mPSNR/mSSIM/mLPIPS + tOF/tLPIPS on the covisibility-masked held-out views Shape-of-Motion evaluates on.
4. **Motion-basis visualization.** Cluster the learned SE(3) bases, color Gaussians by motion group, and render a short orbit that shows persistent trajectories (the interpretability payoff of *explicit* motion over a black-box deformation MLP — the same "every primitive is inspectable" thesis that motivated 3DGS over an RSSM in M1).
5. **Package + honesty check.** README numbers table (synthetic vs real, per-frame + temporal side by side), motion-basis render, and an explicit read against **MonoDyGauBench** ([arXiv 2412.04457](https://arxiv.org/abs/2412.04457)): where the real clip lands, and whether the temporal metric exposes the fast-but-brittle gap the benchmark predicts.

## Gate — Milestone 2

Two measurements, held-out frames only, in order:

1. **Synthetic temporal substrate:** the D-NeRF scene reproduces **4DGS-Wu's published per-frame number** (~33–34 PSNR / high SSIM on the test split) with train-time and render FPS reported, **and** tOF/tLPIPS compute and stay low on this near-multiview setting. Proves the pipeline + the new metric reproduce a credibility bar.
2. **Real monocular reproduction:** Shape-of-Motion on one DyCheck-iPhone clip reproduces its **~16–17 mPSNR** on covisibility-masked held-out views, the motion bases cluster into coherent groups, and tOF/tLPIPS are reported next to the synthetic ones for contrast.

- **GO:** both bars met → time-consistent representation is proven on synthetic *and* real, with a temporal metric that discriminates the two → phase 3 (Gaussian world model).
- **PARTIAL:** synthetic reproduces but the real clip's temporal metric collapses while per-frame PSNR looks acceptable → the reconstruction is per-frame-sharp but temporally-brittle (the MonoDyGauBench failure) → a clean, publishable finding, not "it didn't work"; try smoother-motion priors or a longer-track preproc before calling it.
- **NO-GO:** cannot reproduce the synthetic 4DGS-Wu substrate → toolchain problem (CUDA / HexPlane build / dataset split), not a research result → fix the env (fall back to Deformable-3DGS) before touching the real path.

## After the gate

- **Phase 3 — Gaussian world model (the on-thesis tilt):** an action-conditioned model that predicts `(scene_t, a_t) → scene_{t+1}` directly on the Gaussian field (**GWM** [2508.17600](https://arxiv.org/abs/2508.17600), **GAF** [2506.14135](https://arxiv.org/abs/2506.14135)) — the explicit-3D counterpart to the latent RSSM that hit a representational wall in the manipulation thread. M2's explicit trajectories (step 3) are the natural conditioning signal: a world model that predicts *where the motion bases go next* is a smaller leap than predicting a raw future field. Research-grade-hard and code-immature per the survey; M2's temporal metric becomes its rollout-quality gate.

## Honest framing

This is a **two-reproduction + metric-extension** milestone, not a new 4D method — the defensible signal is reproducing a synthetic *and* a real published number in a modality one step past M1, with a temporal metric rigorous enough to tell them apart. Two unknowns are named up front. (1) **Preproc is the risk, not training** — depth/tracks/masks are noisy and Shape-of-Motion consolidates them; a bad track set caps the real bar regardless of the splatter. (2) **The synthetic bar is easier than it looks** — MonoDyGauBench's core finding is that D-NeRF's teleporting camera is effectively multi-view, so a strong synthetic number does *not* imply real-monocular competence; that is exactly why the real DyCheck bar and the temporal metric exist, and why a hero synthetic clip alone would be an overclaim. Effort ~1–1.5 GPU-weeks, front-loaded on preprocessing.

## Deliverable

- Reproduced 4DGS-Wu reconstruction of one D-NeRF scene (per-frame + temporal table, render).
- Shape-of-Motion reconstruction of one DyCheck-iPhone clip, with the motion-basis visualization.
- The **temporal-metric extension** to [`scene_rep/metrics.py`](../../scene_rep/metrics.py) (tOF + tLPIPS, `TemporalMetrics`, folded into `combined_table`), unit-tested like the M1 harness — the asset phase 3 reports rollout quality against.
- A writeup at [`docs/experiments/semantic-4dgs-m2.md`](../experiments/semantic-4dgs-m2.md) following the study-writeup convention (hypothesis / method / results table / verdict / next move).

## Env & assets (to provision — not yet stood up)

- **Stack:** 4DGS-Wu (HexPlane, INRIA rasterizer lineage) and Shape-of-Motion each in their **own isolated env** — the M1 bar-1/bar-2 env split proved two 3DGS repos won't share one (gsplat vs `diff-gaussian-rasterization`). Preproc env: Depth-Anything-V2 / UniDepth + CoTracker / TAPIR + the M1 SAM stack.
- **Data:** D-NeRF synthetic (ships poses+times, ~1 scene) + one DyCheck-iPhone clip (has covisibility-masked GT held-out views — the reproduction target). Self-captured phone clip is *optional*, qualitative-only (no novel-camera GT → motion-basis demo, not a scored bar) and must be flagged as such.
- **Compute:** `pi-a100-80gb`, isolated env mirroring the M1 setup (other projects' HF weights out of scope to touch). 4DGS-Wu ~20–40 min/scene; Shape-of-Motion training modest, preproc (depth+tracks over all frames) is the wall-clock cost.
- **Tracking:** W&B `chaleong/scene-rep`.
