# Semantic 3DGS — Milestone 1: static semantic scene representation + eval harness

**Status:** done (2026-07-24) — **GO on both gate bars.** Reconstruction reproduces the Mip-NeRF360 credibility bar, and the SAM→3DGS instance lift reproduces OpenSplat3D's published LERF-Mask number essentially exactly. The differentiator — 2D segmentation lifted into a per-Gaussian 3D field, scored on held-out views — works end-to-end. Executes [spec `2026-07-21`](../specs/2026-07-21-semantic-3dgs-scene-representation.md).

## The result

| Gate | Task | Metric | This run | Bar | |
|---|---|---|---|---|---|
| **Bar 1 — NVS substrate** | Mip-NeRF360 `garden`, `ns-train splatfacto` 30k | PSNR | **27.15** | ~27 | ✅ |
| | | SSIM | **0.853** | ~0.81 | ✅ beats |
| | | LPIPS | **0.103** | ~0.21 | ✅ beats |
| | | render FPS | 47.5 | real-time | ✅ |
| **Bar 2 — semantic lift** | LERF-Mask `figurines`, SAM→3DGS instance field | mIoU | **92.15** | 92.3 (pub.) | ✅ |
| | (993k Gaussians, 160 clusters, 299 train / 4 held-out) | mBIoU | **89.24** | 89.4 (pub.) | ✅ |

Bar 2 reproduces OpenSplat3D's published `figurines` number within **0.15 points** (per-class mIoU 88–94: red/green apple 94, rubber duck / toy chairs 93, camera 89, porcelain hand 88).

Per-class diagnostic (mIoU vs mBIoU, both vs published aggregate):

![figurines per-class mIoU/mBIoU](https://github.com/charleneleong-ai/robotics-world-models/blob/docs/semantic-3dgs-m1-results/docs/experiments/semantic-3dgs-m1-figurines.png?raw=true)

The weak tails are the diagnosis: **camera** (mIoU 89.0, but mBIoU drops to **81.7**) and **porcelain hand** (88.0/87.9) drag the aggregate down, while the two toy chairs are mBIoU-*strong* (93.1/92.0 vs their mIoU 93.2/93.2) — small-object/occlusion classes lose mask overlap the most on held-out views.

## Method

The spec's Milestone-1 pipeline, both halves on the A100 (`pi-a100-80gb`), each in its own isolated env:

- **Bar 1 (reconstruction):** `garden` reconstructed with **`gsplat`/Splatfacto** via Nerfstudio (`ns-train splatfacto`, 30k, downscale-4, COLMAP-seeded from the dataset's `sparse/0`), ~21 min. Held-out PSNR/SSIM/LPIPS via `ns-eval` on the interval-8 split.
- **Bar 2 (semantic lift):** [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d) run on `figurines` — **SAM ViT-H** masks on the training views → lifted into a per-Gaussian instance/feature field (feature_dim 8, 160 clusters) → text-prompted eval (`eval_lerf_mask.py`, GroundingDINO) → mIoU/mBIoU on the 4 held-out frames.

The reusable **eval harness** ([`scene_rep/metrics.py`](../../scene_rep/metrics.py), merged in [#17](../../pull/17)) is validated by 20 known-answer tests; see the cross-check note under follow-ups.

## Honest deviations from the spec

Recorded because each changed the *bar*, not just the incidentals:

- **Dataset — LERF-Mask/`figurines`, not ScanNet++.** ScanNet++ is access-gated (academic approval), so the bar 2 target became **OpenSplat3D's own published `figurines` number** (a reproduction bar), not the spec's "~70% mIoU on ScanNet++." Reproducing a peer-reviewed number to 0.15 pt is a *stronger* signal than an absolute-threshold pass on a self-chosen split.
- **SAM v1 (ViT-H), not SAM2.** The OpenSplat3D repo lifts SAM v1 masks; the lift is identical in spirit (masks → per-Gaussian identities). Not worth forking the repo to swap the segmenter.
- **gsplat only in bar 1.** OpenSplat3D uses INRIA's `diff-gaussian-rasterization`, so bar 2 was a *fresh* isolated env (python 3.12 / torch 2.5.1+cu124 / cuml 25.04 / compiled `diff_gaussian_rasterization`+`simple_knn`+`fused_ssim`+`groundingdino`), not the bar-1 nerfstudio recipe (python 3.10 / gsplat 1.4.0 — gsplat ships cp310 wheels only).
- **`garden`, not `room`.** Either Mip-NeRF360 scene satisfies the substrate bar; `garden` is the field's standard headline scene.

Newer-toolchain build fixes the bar-2 env needed (all upstreamable): `setuptools<81` for CLIP's `pkg_resources` build, `#include <cstdint>/<cstddef>` for nvcc 12.9 / gcc 12.3, manual `glm`+`simple-knn` submodule fetch, `TORCH_CUDA_ARCH_LIST=8.0` for the A100 (sm_80).

## Verdict

**GO — the differentiator is proven end-to-end.** A self-trained 3DGS reconstruction reproduces the ~27 PSNR credibility bar, and a SAM→3DGS instance lift reproduces a peer-reviewed LERF-Mask number to within 0.15 mIoU, both scored on held-out views. The reusable held-out-view eval harness exists and is unit-tested. This is the substrate the 4D (M2) and Gaussian-world-model (phase 3) extensions build on.

## Next move

- **Milestone 2 — 4D / temporal consistency:** D-NeRF synthetic → Deformable-3DGS/4DGS-Wu, then one real monocular clip → Shape-of-Motion, reporting per-frame + a temporal metric (tOF/tLPIPS). Its own spec when kicked off.
- **Harness cross-check (small follow-up):** `eval_lerf_mask.py` computes predicted masks in memory and persists only aggregate metrics; GT masks are on disk (`~/scene_rep_data/lerf_mask/figurines/test_mask/`) but rendered predictions are not. Validating [`scene_rep.metrics.mean_iou`](../../scene_rep/metrics.py) against OpenSplat3D on real predictions needs a ~10-line eval tweak to dump per-frame masks — deferred, not hacked into the vendored repo.

## Assets (on `pi-a100-80gb`)

Bar 1: `nerfstudio` env; garden checkpoint `~/scene_rep_logs/garden/splatfacto/2026-07-21_220843/` (raw images reclaimed — re-downloadable). Bar 2: `opensplat3d` build env + repo `~/opensplat3d` (uv-managed `.venv`); model output `~/scene_rep_data/output/figurines/…`; eval `…/eval_results/grounded/…/results.json`.
