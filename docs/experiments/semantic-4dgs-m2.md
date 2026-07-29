# Semantic 4DGS — Milestone 2: dynamic scene representation + temporal eval

**Status:** 📋 not started — stub awaiting M2 kickoff (after the [M1 GO](semantic-3dgs-m1.md)). Executes [spec `2026-07-28`](../specs/2026-07-28-semantic-4dgs-dynamic-scene.md). Two reproduction bars — synthetic **D-NeRF → 4DGS-Wu**, real monocular **DyCheck-iPhone → Shape-of-Motion** — each carrying a temporal metric bolted onto the M1 harness. Tables below are placeholders; the numbers land on execution.

## Hypothesis

A self-run 4DGS reconstruction that reproduces 4DGS-Wu's published D-NeRF per-frame fidelity (~33–34 PSNR) **plus** a Shape-of-Motion run that reproduces its DyCheck-iPhone number (~16–17 mPSNR on covisibility-masked held-out views), each with a stable temporal metric, is a credible — and honest — 4D-competence signal. The temporal metric must catch the MonoDyGauBench failure: per-frame-sharp but temporally-brittle on the real clip while synthetic looks clean. Full rationale in the [spec](../specs/2026-07-28-semantic-4dgs-dynamic-scene.md#hypothesis--goal).

## The result

*Pending execution. Bars and targets from the [gate](../specs/2026-07-28-semantic-4dgs-dynamic-scene.md#gate--milestone-2).*

| Gate | Task | Metric | This run | Bar | |
|---|---|---|---|---|---|
| **Bar 1 — synthetic substrate** | D-NeRF `jumpingjacks`/`bouncingballs`, 4DGS-Wu | PSNR | TBD | ~33–34 | |
| | | SSIM | TBD | high | |
| | | tOF / tLPIPS | TBD | low (near-multiview) | |
| | | train time / render FPS | TBD | reported | |
| **Bar 2 — real monocular** | DyCheck-iPhone clip, Shape-of-Motion | mPSNR | TBD | ~16–17 (pub.) | |
| | (covisibility-masked held-out views) | mSSIM / mLPIPS | TBD | reported | |
| | | tOF / tLPIPS | TBD | reported vs synthetic | |

## Method

The spec's five-step pipeline (easy→hard); [full detail there](../specs/2026-07-28-semantic-4dgs-dynamic-scene.md#method). On execution, record here the same way M1 did — what actually ran, in which isolated env, and every deviation that moved a *bar* (not incidentals):

- **Bar 1 (synthetic):** which D-NeRF scene, 4DGS-Wu (HexPlane) vs the Deformable-3DGS MLP fallback if the repo fights the toolchain, train recipe, held-out split.
- **Temporal metric:** the tOF/tLPIPS extension to [`scene_rep/metrics.py`](../../scene_rep/metrics.py) (`TemporalMetrics` row, folded into `combined_table`), and its known-answer tests (static clip → tOF≈0; frame-shuffle → tOF spikes).
- **Bar 2 (real):** DyCheck-iPhone clip, the preproc stack that dominates wall-clock (depth + long-range tracks + M1 SAM masks), Shape-of-Motion SE(3) bases, motion-basis clustering + orbit render.

## Honest deviations from the spec

*To fill on execution — record any that changed a bar (dataset swap, fallback splatter, preproc substitution), M1-style.*

## Verdict

*Pending.* GO / PARTIAL / NO-GO per the [gate's three outcomes](../specs/2026-07-28-semantic-4dgs-dynamic-scene.md#gate--milestone-2) — notably PARTIAL is a clean finding (temporal metric exposes the brittle real clip), not a failure.

## Next move

- **Phase 3 — Gaussian world model:** action-conditioned `(scene_t, a_t) → scene_{t+1}` on the Gaussian field (GWM [2508.17600](https://arxiv.org/abs/2508.17600), GAF [2506.14135](https://arxiv.org/abs/2506.14135)); M2's explicit motion trajectories are the conditioning signal, its temporal metric the rollout-quality gate. Its own spec when kicked off.

## Assets (on `pi-a100-80gb`)

*To fill on execution — env split (4DGS-Wu / Shape-of-Motion / preproc each isolated, per the M1 two-repo lesson), checkpoints, data paths. W&B `chaleong/scene-rep`.*
