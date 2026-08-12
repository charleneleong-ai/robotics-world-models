# Semantic 3D Gaussian Splatting — static scene representation + eval harness (Milestone 1)

**Status:** ✅ EXECUTED (2026-07-24) — **GO on both gate bars.** Bar 1: `garden` reconstruction hit PSNR 27.15 / SSIM 0.853 / LPIPS 0.103 (held-out). Bar 2: SAM→3DGS lift reproduced OpenSplat3D's published LERF-Mask `figurines` number (mIoU 92.15 vs 92.3) within 0.15 pt. Dataset swapped ScanNet++→LERF-Mask (access gate) so the bar became a reproduction target; full result + honest deviations in [`semantic-3dgs-m1.md`](../experiments/semantic-3dgs-m1.md). **Date:** 2026-07-21 → executed 2026-07-24.

> **One-line:** reconstruct one real scene as a 3D Gaussian field, lift SAM2 masks into a per-Gaussian instance field, and stand up the reusable `mIoU + PSNR/SSIM/LPIPS`-on-held-out-views eval harness — the smallest end-to-end slice that proves the differentiator (2D-segmentation expertise → 3D lift) with credible numbers. Substrate for the 4D (Milestone 2) and Gaussian-world-model (phase 3) extensions.

## Why

This is the second project in the [[project_robotics_world_models_pivot]] portfolio and the **representation** end of the world-model stack — where a static 3D scene gets an explicit, editable, semantically-grounded encoding before anything predicts its future. The just-closed manipulation thread ([`plugchargerdense.md`](../experiments/plugchargerdense.md)) exhausted *latent* (RSSM) world models and hit a representational wall: the RSSM couldn't allocate capacity to a decision-critical state its policy never visited. An **explicit 3D Gaussian field** is the opposite design point — every primitive is inspectable and maskable — which is exactly why it composes cleanly with the author's production-segmentation strength (panoptic/semantic at scale, no prior neural rendering). Milestone 1 builds that substrate and, critically, the **eval harness** — the reusable rigor asset that every later milestone reports against.

Full survey (representation / framework / semantic / 4D / datasets / metrics, cross-checked 2026-06) in [`project2-3d4d-scene-representation-SOTA.md`](../../project2-3d4d-scene-representation-SOTA.md); this spec is its executable Milestone 1, not a re-derivation.

## Hypothesis / goal

A self-trained 3DGS reconstruction at **~27 PSNR / ~0.81 SSIM / ~0.21 LPIPS on one Mip-NeRF360 scene** (with train-time + render-FPS), **plus** a SAM2→3DGS instance lift scoring **~70% mIoU on held-out views** of a ScanNet++ scene, is a clear 2026 competence signal — and a stronger one than chasing SOTA fidelity. If the semantic lift reproduces on held-out (not training) views, the differentiator holds; if it only works on training views, the lift is memorizing 2D masks rather than building a view-consistent 3D field (the failure the harness must catch).

## Method

The SOTA doc's Milestone-1 pipeline — five steps, first and last are pure 3DGS + eval rigor (existing strength in a new modality), middle three are the genuinely new neural-rendering skill bounded to "lift masks I already know how to produce":

1. **Reconstruct** one Mip-NeRF360 scene (`garden` or `room`) with **`gsplat`/Splatfacto** via Nerfstudio `ns-train splatfacto`. Output: PSNR/SSIM/LPIPS vs the ~27 baseline, render FPS, train time. This alone proves the substrate runs.
2. **2D masks** on training views with **SAM2** (automatic + tracked mode for view-consistency; Grounded-SAM only if text-prompted classes are wanted). SAM2 over Mask2Former deliberately — Mask2Former instance labels flicker across viewpoints and break cross-view association.
3. **Lift masks into the Gaussian field** with **OpenSplat3D** (primary — cleanest live repo, `just setup`, ScanNet++/LERF eval built in). Fallbacks pinned and ready: **SAGA** (most-reproduced promptable) / **LangSplat** (canonical language field).
4. **Eval harness** (the reusable rigor asset): render segmentation from *held-out* views, compare to GT → **mIoU** (+ PQ if panoptic); PSNR/SSIM/LPIPS for NVS; one combined table. Held-out-view discipline is the whole point — training-view mIoU is not reported.
5. **Package**: Nerfstudio web viewer / short render (click-to-select-object), README numbers table.

Datasets: **Mip-NeRF360** (NVS bar) + **one ScanNet++ scene** (semantic GT, native OpenSplat3D eval). **Replica** or **3D-OVS** as a controlled sanity check if ScanNet++ preproc fights.

## Gate — Milestone 1

Two measurements, held-out views only, in order:

1. **NVS substrate:** reconstructed scene reaches **≥ ~27 PSNR / ~0.81 SSIM / ~0.21 LPIPS** on Mip-NeRF360 held-out views, with train-time (tens of min) and render FPS (100+) reported. Proves the pipeline reproduces the credibility bar.
2. **Semantic lift:** rendered segmentation from held-out ScanNet++ views scores **≥ ~70% mIoU** (the "working 3D-seg pipeline" bar; ScanNet++'s long-tail 1000+ classes make modest mIoU credible).

- **GO:** both bars met → the differentiator is proven and the eval harness exists → Milestone 2 (4D).
- **PARTIAL:** NVS reproduces but the lift underperforms on held-out (works on training views only) → the lift is 2D-memorizing, not 3D-consistent → swap OpenSplat3D → SAGA/LangSplat, or debug SAM2 cross-view association; a cleaner-separated finding than "it didn't work."
- **NO-GO:** cannot reproduce the ~27 PSNR substrate → environment/pipeline problem (almost certainly CUDA/COLMAP), not a research result → fix the env before anything else.

## After the gate

- **Milestone 2 — 4D / temporal consistency:** one D-NeRF synthetic clip through **Deformable-3DGS / 4DGS-Wu** (fast, minimal data wrangling) to stand up a deformation-field pipeline, then one real monocular phone clip through **Shape-of-Motion** (ICCV'25 Highlight, monocular, explicit persistent trajectories). Report per-frame PSNR/SSIM/LPIPS **+ a temporal metric (tOF or tLPIPS)** and visualize the motion bases; cite **MonoDyGauBench** as the honesty check. Preprocessing (depth/masks/2D-tracks) is the time sink, not training.
- **Phase 3 — Gaussian world model (the on-thesis tilt):** an action-conditioned 3D world model that predicts `(scene_t, a_t) → scene_{t+1}` on the Gaussian field (**GWM** [2508.17600](https://arxiv.org/abs/2508.17600), **GAF** [2506.14135](https://arxiv.org/abs/2506.14135)) — the explicit-3D counterpart to the latent RSSM that hit the wall in the manipulation thread. Research-grade-hard and code-immature per the survey; deliberately *not* the first milestone (baseline before modifying).

## Honest framing

This is a **reproduction + eval-harness** milestone, not a new method — the defensible signal is a reproduced ~27 PSNR substrate plus a held-out-view mIoU harness, in a modality the author hasn't worked in, built on their segmentation strength. The one real unknown is **OpenSplat3D's preproc env** (CUDA 11.8 / COLMAP / glomap — the ecosystem's #1 install grief), which is why two maintained fallbacks are pinned before starting and `gsplat` prebuilt wheels (keyed to the Python/PyTorch/CUDA triple) are used to skip on-the-fly `nvcc` compilation. Effort ~1 GPU-week including the harness.

## Deliverable

- Reproduced 3DGS reconstruction of one Mip-NeRF360 scene (numbers table + viewer render).
- SAM2 → OpenSplat3D instance lift on one ScanNet++ scene.
- The reusable **held-out-view eval harness** (mIoU/PQ + PSNR/SSIM/LPIPS, one combined table) — the asset every later milestone reports against.
- A writeup at [`docs/experiments/semantic-3dgs-m1.md`](../experiments/) following the study-writeup convention (hypothesis / method / results table / verdict / next move).

## Env & assets (to provision — not yet stood up)

- **Stack:** Nerfstudio (`ns-train splatfacto`, Apache-2.0) → `gsplat` backend (prebuilt wheels) → **SAM2** masks → **OpenSplat3D** lift; SAGA/LangSplat fallbacks pinned. CUDA **11.8**, COLMAP/glomap for preproc.
- **Compute:** one A100 80GB or 24 GB consumer GPU (3DGS/scene ~20–40 min; OpenSplat3D lift ~1–3 h incl. COLMAP; harness ~1–2 days engineering).
- **Box note:** `pi-a100-80gb` is shared — other projects' HF weights (~90G, Laguna/materialhack) are out of scope to touch; provision an isolated env + a scoped ScanNet++ subset (one scene, not the full 1000+), mirroring the `jax_dreamer` isolation used for the manipulation thread.
- **Tracking:** W&B `chaleong/wm-manip` (or a new `scene-rep` project).
