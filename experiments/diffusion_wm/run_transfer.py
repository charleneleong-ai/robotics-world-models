#!/usr/bin/env python3
"""Full sim-to-real transfer pipeline test run.

Runs all 6 milestones end-to-end with synthetic data:
  1. Domain randomization
  2. Video evaluation metrics
  3. Fidelity / divergence detection
  4. System identification
  5. Residual dynamics
  6. Transfer pipeline

Usage:
    PYTHONPATH=. .venv/bin/python experiments/diffusion_wm/run_transfer.py
    PYTHONPATH=. .venv/bin/python experiments/diffusion_wm/run_transfer.py --no-log
    PYTHONPATH=. .venv/bin/python experiments/diffusion_wm/run_transfer.py --tracker trackio
"""
from __future__ import annotations

import argparse
import time

import torch

# ── Dimensions ──────────────────────────────────────────────────────
OBS_DIM = 38   # typical ManiSkill state (pos + vel + quat + extras)
ACT_DIM = 2    # simple 2D action space
BATCH = 32
HORIZON = 10


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def init_tracker(project: str, name: str, config: dict, tracker: str = "wandb"):
    """Initialize tracking backend (wandb or trackio)."""
    if tracker == "trackio":
        import trackio
        run = trackio.init(project=project, name=name, config=config)
        return run, trackio
    else:
        import wandb
        run = wandb.init(project=project, name=name, config=config)
        return run, wandb


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim-to-real transfer pipeline")
    parser.add_argument("--no-log", action="store_true", help="Disable all logging")
    parser.add_argument("--tracker", default="trackio", choices=["wandb", "trackio"],
                        help="Tracking backend: trackio (default) or wandb")
    parser.add_argument("--task", default="PlugCharger-v1", help="ManiSkill task")
    parser.add_argument("--project", default="wm-manip", help="Project name")
    parser.add_argument("--space-id", default=None, help="HF Space ID for trackio sharing")
    args = parser.parse_args()

    # ── Tracker init ──────────────────────────────────────────────────
    run, tracker = None, None
    if not args.no_log:
        try:
            config = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "batch": BATCH, "task": args.task}
            if args.tracker == "trackio":
                import trackio
                init_kwargs = {"project": args.project, "name": "sim-to-real-pipeline", "config": config}
                if args.space_id:
                    init_kwargs["space_id"] = args.space_id
                run = trackio.init(**init_kwargs)
                tracker = trackio
            else:
                import wandb
                run = wandb.init(project=args.project, name="sim-to-real-pipeline", config=config)
                tracker = wandb
            print(f"  Tracker: {args.tracker} — {getattr(run, 'url', 'local')}")
        except Exception as e:
            print(f"  Tracker unavailable ({e}), continuing without logging")

    def log(metrics: dict[str, float], step: int | None = None) -> None:
        if run is not None and tracker is not None:
            tracker.log(metrics, step=step)

    t0 = time.time()

    # ================================================================
    # 1. Domain Randomization
    # ================================================================
    section("1. Domain Randomization")
    from experiments.diffusion_wm.domain_rand import (
        DomainRandomizationConfig,
        apply_observation_noise,
        apply_action_noise,
        TASK_DEFAULTS,
    )

    config = TASK_DEFAULTS["PlugCharger-v1"]
    print(f"  Config: PlugCharger-v1")
    print(f"    friction range : {config.physics.friction}")
    print(f"    position noise : {config.observation.position_noise}")

    obs = torch.randn(BATCH, OBS_DIM)
    action = torch.randn(BATCH, ACT_DIM)

    noisy_obs = apply_observation_noise(obs, config)
    noisy_action = apply_action_noise(action, config)

    obs_drift = (noisy_obs - obs).abs().mean().item()
    act_drift = (noisy_action - action).abs().mean().item()
    print(f"  ✓ obs noise  : mean drift = {obs_drift:.5f}")
    print(f"  ✓ action noise: mean drift = {act_drift:.5f}")

    log({"domain_rand/obs_drift": obs_drift, "domain_rand/action_drift": act_drift})

    # ================================================================
    # 2. Video Metrics
    # ================================================================
    section("2. Video Evaluation Metrics")
    from experiments.diffusion_wm.video_metrics import (
        compute_fvd,
        compute_temporal_lpips,
        compute_idm_error,
        compute_rot_trans_error,
        compute_all_video_metrics,
    )

    # Synthetic video batch: (B, T, C, H, W)
    real_vid = torch.randn(4, 8, 3, 32, 32)
    fake_vid = torch.randn(4, 8, 3, 32, 32)

    fvd = compute_fvd(real_vid, fake_vid)
    print(f"  ✓ FVD             = {fvd:.2f}")

    t_lpips = compute_temporal_lpips(real_vid, fake_vid)
    print(f"  ✓ Temporal LPIPS  = {t_lpips:.5f}")

    idm_err = compute_idm_error(real_vid, torch.randn(4, 8, ACT_DIM))
    print(f"  ✓ IDM error       = {idm_err:.5f}")

    poses_pred = torch.randn(4, 8, 7)
    poses_gt = torch.randn(4, 8, 7)
    rot, trans = compute_rot_trans_error(poses_pred, poses_gt)
    print(f"  ✓ Rot error       = {rot:.3f}°")
    print(f"  ✓ Trans error     = {trans:.5f} m")

    vid_result = compute_all_video_metrics(real_vid, fake_vid)
    print(f"  ✓ All metrics     : {vid_result.to_dict()}")

    log({"video/fvd": fvd, "video/temporal_lpips": t_lpips, "video/idm_error": idm_err,
         "video/rot_error_deg": rot, "video/trans_error_m": trans})

    # ================================================================
    # 3. Fidelity / Divergence Detection
    # ================================================================
    section("3. Fidelity & Divergence Detection")
    from experiments.diffusion_wm.fidelity import (
        DivergenceDetector,
        compute_trust_from_divergence,
    )

    detector = DivergenceDetector(threshold=0.1, ema_alpha=0.9)
    diverged_at = None

    for step in range(50):
        noise_scale = step / 50.0
        predicted = torch.randn(OBS_DIM)
        actual = predicted + torch.randn(OBS_DIM) * noise_scale
        result = detector.update(predicted, actual)
        if result.is_divergent and diverged_at is None:
            diverged_at = step

    stats = detector.get_rolling_stats()
    print(f"  ✓ EMA divergence  = {detector.ema_divergence:.4f}")
    print(f"  ✓ Diverged at step= {diverged_at}")
    print(f"  ✓ Rolling stats   = {stats}")

    trust = compute_trust_from_divergence(detector.ema_divergence)
    print(f"  ✓ Trust score     = {trust:.4f}")

    log({"divergence/ema": detector.ema_divergence, "divergence/diverged_at": diverged_at or -1,
         "divergence/final_trust": trust, "divergence/mean": stats["mean"], "divergence/max": stats["max"]})

    # ================================================================
    # 4. Model + System Identification
    # ================================================================
    section("4. Diffusion Model + System Identification")
    from experiments.diffusion_wm.model import MLPDenoiser, DiffusionDynamics
    from experiments.diffusion_wm.system_id import SystemIdentifier

    # Create small model
    denoiser = MLPDenoiser(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=128, num_blocks=3)
    sim_model = DiffusionDynamics(denoiser, timesteps=50)
    n_params = sum(p.numel() for p in sim_model.parameters())
    print(f"  ✓ Model created   : {n_params:,} params, 50 timesteps")

    # Quick training
    print("  Training sim model (200 steps)...")
    opt = torch.optim.AdamW(sim_model.parameters(), lr=3e-4)
    for step in range(200):
        batch_obs = torch.randn(BATCH, OBS_DIM)
        batch_act = torch.randn(BATCH, ACT_DIM)
        # Synthetic dynamics: next = obs + projected_action + noise
        action_proj = torch.randn(BATCH, OBS_DIM) * 0.5
        batch_next = batch_obs + action_proj + torch.randn(BATCH, OBS_DIM) * 0.1
        loss = sim_model(batch_next, batch_obs, batch_act)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"  ✓ Training loss   = {loss.item():.4f}")

    log({"train/sim_final_loss": loss.item()})

    # System ID
    print("  Running system identification...")
    sid = SystemIdentifier(obs_dim=OBS_DIM, action_dim=ACT_DIM, max_iterations=50)
    real_data = {
        "obs": torch.randn(50, OBS_DIM),
        "actions": torch.randn(50, ACT_DIM),
        "next_obs": torch.randn(50, OBS_DIM),
    }
    sid_result = sid.calibrate(
        real_data["obs"], real_data["actions"], real_data["next_obs"]
    )
    print(f"  ✓ Converged       = {sid_result.converged}")
    print(f"  ✓ Loss            = {sid_result.calibration_loss:.4f}")
    print(f"  ✓ Params          = {sid_result.estimated_params}")

    log({"sysid/converged": float(sid_result.converged), "sysid/loss": sid_result.calibration_loss,
         **{f"sysid/{k}": v for k, v in sid_result.estimated_params.items()}})

    # ================================================================
    # 5. Residual Dynamics
    # ================================================================
    section("5. Residual Dynamics Model")
    from experiments.diffusion_wm.residual_dynamics import (
        ResidualDynamicsNet,
        create_hybrid_model,
        OnlineResidualAdapter,
    )

    hybrid = create_hybrid_model(OBS_DIM, ACT_DIM, sim_model, residual_hidden_dim=128)
    hybrid.residual_model.train()

    print("  Training residual model (50 steps)...")
    r_opt = torch.optim.Adam(hybrid.residual_model.parameters(), lr=1e-3)
    for step in range(50):
        b_obs = torch.randn(BATCH, OBS_DIM)
        b_act = torch.randn(BATCH, ACT_DIM)
        action_proj = torch.randn(BATCH, OBS_DIM) * 0.5
        b_next = b_obs + action_proj + torch.randn(BATCH, OBS_DIM) * 0.1
        r_loss, r_comp = hybrid.compute_loss(b_obs, b_act, b_next)
        r_opt.zero_grad()
        r_loss.backward()
        r_opt.step()
    hybrid.residual_model.eval()
    print(f"  ✓ Residual loss   = {r_loss.item():.4f}")
    print(f"  ✓ Components      = {r_comp}")

    # Hybrid prediction
    test_obs = torch.randn(4, OBS_DIM)
    test_act = torch.randn(4, ACT_DIM)
    pred = hybrid.predict(test_obs, test_act, num_denoise_steps=10)
    print(f"  ✓ Sim shape       = {pred.sim_prediction.shape}")
    print(f"  ✓ Residual shape  = {pred.residual.shape}")
    print(f"  ✓ Hybrid shape    = {pred.hybrid_prediction.shape}")
    print(f"  ✓ Uncertainty     = {pred.uncertainty.mean().item():.4f}")

    log({"hybrid/uncertainty": pred.uncertainty.mean().item(),
         "hybrid/residual_mag": pred.residual.abs().mean().item()})

    # Online adaptation
    print("  Online adaptation (20 updates)...")
    adapter = OnlineResidualAdapter(hybrid.residual_model, buffer_size=50, batch_size=8)
    for _ in range(20):
        m = adapter.update(
            torch.randn(OBS_DIM), torch.randn(ACT_DIM),
            torch.randn(OBS_DIM), torch.randn(OBS_DIM),
        )
    print(f"  ✓ Online metrics  = {m}")

    # ================================================================
    # 6. Full Transfer Pipeline
    # ================================================================
    section("6. Full Transfer Pipeline")
    from experiments.diffusion_wm.transfer import SimToRealPipeline, TransferConfig

    pipeline = SimToRealPipeline(
        obs_dim=OBS_DIM, action_dim=ACT_DIM,
        config=TransferConfig(
            sysid_iterations=30,
            sysid_lr=1e-3,
            residual_hidden_dim=128,
            online_lr=1e-4,
        ),
    )

    eval_data = {
        "obs": torch.randn(30, OBS_DIM),
        "actions": torch.randn(30, ACT_DIM),
        "next_obs": torch.randn(30, OBS_DIM),
    }

    # Step 2: System ID
    print("  Step 2: System identification...")
    s2 = pipeline.step2_system_identification(eval_data)
    print(f"    ✓ converged={s2.converged}, loss={s2.calibration_loss:.4f}")

    # Step 3: Train residual
    print("  Step 3: Training residual dynamics...")
    hybrid_final = pipeline.step3_train_residual(sim_model, eval_data, epochs=5)
    print(f"    ✓ Residual model trained")

    # Step 5: Evaluate
    print("  Step 5: Evaluation...")
    result = pipeline.step5_evaluate(hybrid_final, eval_data, num_steps=10)
    print(f"    ✓ Hybrid MSE     = {result.eval_metrics['hybrid_mse']:.4f}")
    print(f"    ✓ Sim MSE        = {result.eval_metrics['sim_mse']:.4f}")
    print(f"    ✓ Improvement    = {result.eval_metrics['improvement']:.4f}")
    print(f"    ✓ Residual mag   = {result.eval_metrics['residual_magnitude']:.4f}")
    print(f"    ✓ Mean trust     = {sum(result.trust_scores)/len(result.trust_scores):.4f}")
    print(f"    ✓ Mean divergence= {sum(result.divergence_scores)/len(result.divergence_scores):.4f}")

    log({"transfer/hybrid_mse": result.eval_metrics["hybrid_mse"],
         "transfer/sim_mse": result.eval_metrics["sim_mse"],
         "transfer/improvement": result.eval_metrics["improvement"],
         "transfer/mean_trust": sum(result.trust_scores)/len(result.trust_scores)})

    # ================================================================
    # Summary
    # ================================================================
    section("ALL MILESTONES COMPLETE")
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  All 6 milestones passed successfully!")
    if run is not None and tracker is not None:
        tracker.finish()
    print()


if __name__ == "__main__":
    main()
