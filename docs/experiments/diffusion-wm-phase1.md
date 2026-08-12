# Diffusion world model — Phase 1 (state-based) gate

**Status:** PARTIAL (2026-08-09). The from-scratch action-conditioned DDPM dynamics model (`p(s_{t+1} | s_t, a_t)`) trains cleanly and its denoising media is fully logged, but it **cannot be scored against the TD-MPC2 baseline in the same space**: the baseline checkpoint has no `decode()`, so TD-MPC2 errors live in its 512-dim latent while ours live in 43-dim raw state. On the metrics both can produce, the diffusion model is stable and monotone-improving but numerically far from the baseline — which the space mismatch prevents us from interpreting as either win or loss. **The honest gate outcome is PARTIAL pending a comparable-space eval, not GO and not NO-GO.**

## The result

| Metric | Diffusion (ours, 43-dim obs) | TD-MPC2 (512-dim latent) |
|---|---|---|
| 1-step MSE / MAE | 0.3946 / 0.4761 | 0.0005 / 0.0066 |
| 5-step rollout MSE | 0.5947 | 0.0195 |
| 10-step rollout MSE | 0.7057 | 0.0241 |
| 20-step rollout MSE | 0.7883 | 0.0270 |

Diffusion loss: 1.32 → 0.0035 over 50k steps (~25 steps/s); best val_loss **0.006578** at step ~43k. Both runs (v2, v3) independently converged to the same best val_loss — the 0.006578 is reproducible.

**Not comparable.** TD-MPC2's 1-step error of 0.0005 in a 512-dim latent is not the same quantity as 0.3946 in 43-dim raw state — a normalized latent can look arbitrarily small against a raw obs space with contact dynamics. The honest comparison would need TD-MPC2's obs-space decoder (absent from the shipped checkpoint) or a latent-space eval of our model (not built).

## Training quality (the part that IS solid)

- Clean convergence: 1.32 → 0.0035, monotone with warmup-cosine LR.
- **Both v2 and v3 reached the identical best val_loss 0.006578** — the run is reproducible, not a lucky seed.
- Media pipeline works end-to-end and lands in one W&B run: train/val denoising grids with GT overlay every 5k steps + per-split rollout trajectories at eval. Single-run merge (train → eval resume) verified.
- Tests: 62 pass (diffusion model, noise schedule, denoise monotonicity, media-pool train/val caching, viz shapes), ruff clean.

## Media

All logged to the merged run [`q4y9dbcm`](https://wandb.ai/chaleong/wm-manip/runs/q4y9dbcm): 18 training denoising grids (9 train + 9 val, steps 5000–45000) + 4 per-split eval charts (denoising grid, rollout trajectories) + train/val loss curves + eval metrics.

## Verdict

- **GO:** NO — cannot claim to match/beat TD-MPC2 without a comparable-space metric.
- **PARTIAL:** YES — pipeline, training stability, reproducibility, and media infrastructure all pass; only the cross-model comparison is blocked.
- **NO-GO:** NO — training is stable and converged; there is no architecture/training bug.

## Next move

1. **Comparable-space eval** (unblocks the gate): add a latent encoder + obs decoder to our model (or train a small obs-decoder for TD-MPC2 latents) so both models are scored on the same predicted-state error.
2. Fidelity/calibration (spec M3): prediction intervals + divergence detector — this is the differentiator for a diffusion WM vs a deterministic/Gaussian latent head.
3. PickCube comparison (spec M1): easier task, should be near-perfect for both — validates the harness independently of contact complexity.
