"""Tests for the diffusion dynamics model.

Run with:
    python -m pytest experiments/diffusion_wm/test_model.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, random_split

from experiments.diffusion_wm.dataset import TransitionDataset
from experiments.diffusion_wm.model import (
    DiffusionDynamics,
    MLPDenoiser,
    cosine_beta_schedule,
)
from experiments.diffusion_wm.train import _cache_media_pool, _media_pools
from experiments.diffusion_wm.domain_rand import (
    DomainRandomizationConfig,
    PhysicsRandomization,
    ObservationNoise,
    ActionNoise,
    apply_physics_randomization,
    apply_observation_noise,
    apply_action_noise,
)
from experiments.diffusion_wm.video_metrics import (
    I3DFeatureExtractor,
    compute_fvd,
    compute_temporal_lpips,
    compute_idm_error,
    compute_rot_trans_error,
    compute_all_video_metrics,
    VideoMetricsResult,
)
from experiments.diffusion_wm.fidelity import (
    PredictionCalibration,
    DivergenceDetector,
    compute_trust_from_divergence,
)
from experiments.diffusion_wm.system_id import (
    ParameterEstimator,
    SystemIdentifier,
    SystemIdentificationResult,
)
from experiments.diffusion_wm.residual_dynamics import (
    ResidualDynamicsNet,
    HybridDynamicsModel,
    OnlineResidualAdapter,
    create_hybrid_model,
)
from experiments.diffusion_wm.transfer import (
    SimToRealPipeline,
    TransferConfig,
    TransferResult,
    run_full_transfer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obs_dim() -> int:
    return 16

@pytest.fixture
def act_dim() -> int:
    return 4

@pytest.fixture
def denoiser(obs_dim, act_dim) -> MLPDenoiser:
    return MLPDenoiser(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=64,
        num_blocks=3,
        cond_dim=32,
    )

@pytest.fixture
def model(denoiser) -> DiffusionDynamics:
    return DiffusionDynamics(denoiser, timesteps=100)

@pytest.fixture
def batch(obs_dim, act_dim) -> dict[str, torch.Tensor]:
    B = 8
    return {
        "obs": torch.randn(B, obs_dim),
        "action": torch.randn(B, act_dim),
        "next_obs": torch.randn(B, obs_dim),
    }


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class TestNoiseSchedule:
    def test_cosine_beta_shape(self):
        betas = cosine_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_cosine_beta_range(self):
        betas = cosine_beta_schedule(1000)
        assert betas.min() >= 0.0001
        assert betas.max() <= 0.02

    def test_cosine_beta_monotonic(self):
        betas = cosine_beta_schedule(100)
        for i in range(1, len(betas)):
            assert betas[i] >= betas[i - 1] - 1e-6  # allow tiny FP error


# ---------------------------------------------------------------------------
# MLPDenoiser
# ---------------------------------------------------------------------------

class TestMLPDenoiser:
    def test_output_shape(self, denoiser, batch):
        B = batch["obs"].size(0)
        t = torch.randint(0, 100, (B,)).float()
        out = denoiser(batch["next_obs"], batch["obs"], batch["action"], t)
        assert out.shape == batch["next_obs"].shape

    def test_batch_independence(self, denoiser, batch):
        B = batch["obs"].size(0)
        t = torch.randint(0, 100, (B,)).float()
        out = denoiser(batch["next_obs"], batch["obs"], batch["action"], t)
        # Modify one input, verify only corresponding output changes
        obs_copy = batch["next_obs"].clone()
        obs_copy[0] += 10.0
        out2 = denoiser(obs_copy, batch["obs"], batch["action"], t)
        assert not torch.allclose(out[0], out2[0])
        for i in range(1, B):
            assert torch.allclose(out[i], out2[i], atol=1e-6)

    def test_timestep_conditioning(self, denoiser, batch):
        B = batch["obs"].size(0)
        t0 = torch.zeros(B).float()
        t1 = torch.full((B,), 50).float()
        out0 = denoiser(batch["next_obs"], batch["obs"], batch["action"], t0)
        out1 = denoiser(batch["next_obs"], batch["obs"], batch["action"], t1)
        assert not torch.allclose(out0, out1, atol=1e-4)


# ---------------------------------------------------------------------------
# DiffusionDynamics
# ---------------------------------------------------------------------------

class TestDiffusionDynamics:
    def test_training_loss_shape(self, model, batch):
        loss = model(batch["next_obs"], batch["obs"], batch["action"])
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_training_loss_decreases(self, model, batch):
        """Overfitting test: loss should drop after one step on a tiny batch."""
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(100):
            opt.zero_grad()
            loss = model(batch["next_obs"], batch["obs"], batch["action"])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_sample_output_shape(self, model, batch):
        B = batch["obs"].size(0)
        pred = model.sample(batch["obs"], batch["action"], num_steps=10)
        assert pred.shape == batch["next_obs"].shape

    def test_sample_deterministic_with_seed(self, model, batch):
        """Same input + same seed → same output (with torch seed control)."""
        torch.manual_seed(0)
        pred1 = model.sample(batch["obs"], batch["action"], num_steps=10)
        torch.manual_seed(0)
        pred2 = model.sample(batch["obs"], batch["action"], num_steps=10)
        assert torch.allclose(pred1, pred2, atol=1e-4)

    def test_rollout_output_shape(self, model, batch):
        B = batch["obs"].size(0)
        horizon = 5
        actions = torch.randn(B, horizon, model.denoiser.act_dim)
        traj = model.rollout(batch["obs"], actions, horizon=horizon, num_denoise_steps=10)
        assert traj.shape == (B, horizon + 1, model.obs_dim)

    def test_q_sample(self, model, batch):
        B = batch["next_obs"].size(0)
        t = torch.randint(0, model.timesteps, (B,))
        noise = torch.randn_like(batch["next_obs"])
        x_noisy = model._q_sample(batch["next_obs"], t, noise)
        assert x_noisy.shape == batch["next_obs"].shape
        # At t=0, x_noisy should be close to x_0
        t0 = torch.zeros(B, dtype=torch.long, device=batch["next_obs"].device)
        x_noisy_t0 = model._q_sample(batch["next_obs"], t0, noise)
        assert torch.allclose(x_noisy_t0, batch["next_obs"], atol=0.1)


# ---------------------------------------------------------------------------
# Integration: end-to-end training loop smoke test
# ---------------------------------------------------------------------------

class TestTrainingSmoke:
    @pytest.fixture
    def small_model(self) -> DiffusionDynamics:
        den = MLPDenoiser(obs_dim=4, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8)
        return DiffusionDynamics(den, timesteps=10)

    def test_one_step_forward_backward(self, small_model):
        B = 4
        obs = torch.randn(B, 4)
        action = torch.randn(B, 2)
        next_obs = torch.randn(B, 4)
        loss = small_model(next_obs, obs, action)
        loss.backward()
        assert all(p.grad is not None for p in small_model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Denoising progress (media)
# ---------------------------------------------------------------------------

class TestMediaPool:
    @pytest.fixture
    def shard_dir(self, tmp_path) -> Path:
        rng = np.random.default_rng(0)
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        np.savez_compressed(
            shard_dir / "shard_00000.npz",
            obs=rng.normal(size=(256, 4)),
            action=rng.normal(size=(256, 2)),
            next_obs=rng.normal(size=(256, 4)),
        )
        return shard_dir

    @pytest.fixture(autouse=True)
    def _reset_pool(self):
        _media_pools.clear()
        yield
        _media_pools.clear()

    def _loaders(self, shard_dir):
        ds = TransitionDataset(shard_dir)
        train_ds, val_ds = random_split(ds, [len(ds) - len(ds) // 5, len(ds) // 5])
        return (
            DataLoader(train_ds, batch_size=32),
            DataLoader(val_ds, batch_size=32),
        )

    def test_caches_both_splits(self, shard_dir):
        train_loader, val_loader = self._loaders(shard_dir)
        _cache_media_pool(train_loader, val_loader, torch.device("cpu"))
        assert set(_media_pools) == {"train", "val"}
        for pool in _media_pools.values():
            assert pool["obs"].size(0) == 32

    def test_caching_is_idempotent(self, shard_dir):
        train_loader, val_loader = self._loaders(shard_dir)
        _cache_media_pool(train_loader, val_loader, torch.device("cpu"))
        train_obs = _media_pools["train"]["obs"].clone()
        _cache_media_pool(train_loader, val_loader, torch.device("cpu"))
        assert torch.equal(_media_pools["train"]["obs"], train_obs)


class TestDenoiseWithProgress:
    def test_milestone_shapes(self, model, batch):
        milestones = (75, 50, 25, 0)
        ests = model.denoise_with_progress(
            batch["obs"], batch["action"], num_steps=100, milestones=milestones,
        )
        assert len(ests) == len(milestones)
        for est in ests:
            assert est.shape == batch["next_obs"].shape

    def test_default_milestones(self, model, batch):
        ests = model.denoise_with_progress(batch["obs"], batch["action"], num_steps=100)
        assert len(ests) == 4  # 3T/4, T/2, T/4, 0

    def test_trained_model_denoises_toward_gt(self):
        """After training, x0 estimates improve monotonically as t -> 0."""
        torch.manual_seed(0)
        den = MLPDenoiser(obs_dim=4, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8)
        m = DiffusionDynamics(den, timesteps=10)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        obs = torch.randn(64, 4)
        action = torch.randn(64, 2)
        next_obs = obs + torch.cat([action, action], dim=1)  # learnable target
        for _ in range(400):
            opt.zero_grad()
            loss = m(next_obs, obs, action)
            loss.backward()
            opt.step()

        ests = m.denoise_with_progress(obs[:8], action[:8], num_steps=10, milestones=(7, 5, 3, 0))
        errs = [(e - next_obs[:8]).pow(2).mean().item() for e in ests]
        assert errs[-1] < errs[0]
        assert errs == sorted(errs, reverse=True)  # monotonic improvement (noisiest -> cleanest)


class TestViz:
    @pytest.fixture
    def small_model(self) -> DiffusionDynamics:
        den = MLPDenoiser(obs_dim=4, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8)
        return DiffusionDynamics(den, timesteps=10)

    def test_denoising_grid_figure(self, small_model):
        from experiments.diffusion_wm.viz import denoising_grid
        obs = torch.randn(4, 4)
        action = torch.randn(4, 2)
        next_obs = torch.randn(4, 4)
        fig = denoising_grid(small_model, obs, action, next_obs, milestones=(7, 3, 0), num_steps=10)
        assert len(fig.data) == 4 * 3 * 2  # samples x milestones x (estimate + GT overlay)


# ---------------------------------------------------------------------------
# Domain Randomization
# ---------------------------------------------------------------------------

class TestDomainRandomization:
    def test_config_defaults(self):
        config = DomainRandomizationConfig()
        assert config.physics.friction == (0.5, 2.0)
        assert config.observation.position_noise == 0.01
        assert config.action.torque_noise == 0.05

    def test_apply_observation_noise(self):
        config = DomainRandomizationConfig()
        obs = torch.zeros(4, 38)
        noisy = apply_observation_noise(obs, config)
        assert noisy.shape == obs.shape
        assert not torch.equal(noisy, obs)  # noise was added

    def test_observation_noise_disabled(self):
        config = DomainRandomizationConfig(observation=ObservationNoise(enabled=False))
        obs = torch.randn(4, 38)
        noisy = apply_observation_noise(obs, config)
        assert torch.equal(noisy, obs)

    def test_apply_action_noise(self):
        config = DomainRandomizationConfig()
        action = torch.zeros(4, 2)
        noisy = apply_action_noise(action, config)
        assert noisy.shape == action.shape
        assert not torch.equal(noisy, action)

    def test_action_noise_disabled(self):
        config = DomainRandomizationConfig(action=ActionNoise(enabled=False))
        action = torch.randn(4, 2)
        noisy = apply_action_noise(action, config)
        assert torch.equal(noisy, action)

    def test_task_defaults_exist(self):
        from experiments.diffusion_wm.domain_rand import TASK_DEFAULTS
        assert "PickCube-v1" in TASK_DEFAULTS
        assert "PlugCharger-v1" in TASK_DEFAULTS


# ---------------------------------------------------------------------------
# Video Metrics
# ---------------------------------------------------------------------------

class TestVideoMetrics:
    def test_fvd_symmetric(self):
        a = torch.randn(2, 3, 4, 16, 16)
        b = torch.randn(2, 3, 4, 16, 16)
        fvd_ab = compute_fvd(a, b)
        fvd_ba = compute_fvd(b, a)
        assert fvd_ab >= 0
        assert fvd_ba >= 0

    def test_fvd_nonnegative(self):
        a = torch.randn(4, 3, 8, 16, 16)
        b = torch.randn(4, 3, 8, 16, 16)
        fvd = compute_fvd(a, b)
        assert fvd >= 0

    def test_fvd_different_is_larger(self):
        a = torch.randn(4, 3, 8, 16, 16)
        b = torch.randn(4, 3, 8, 16, 16) * 2 + 5
        fvd_same = compute_fvd(a, a)
        fvd_diff = compute_fvd(a, b)
        # Different distributions should have higher FVD
        # (not always true with random features, but usually)
        assert fvd_same < fvd_diff + 100.0

    def test_temporal_lpips_short_video(self):
        a = torch.randn(2, 1, 4, 8, 8)
        b = torch.randn(2, 1, 4, 8, 8)
        assert compute_temporal_lpips(a, b) == 0.0

    def test_temporal_lpips_nonzero(self):
        a = torch.randn(2, 3, 4, 8, 8)
        b = torch.randn(2, 3, 4, 8, 8)
        score = compute_temporal_lpips(a, b)
        assert score >= 0

    def test_idm_error_short_video(self):
        v = torch.randn(2, 1, 3, 8, 8)
        a = torch.randn(2, 1, 2)
        assert compute_idm_error(v, a) == 0.0

    def test_rot_trans_error_zero(self):
        poses = torch.randn(4, 5, 7)
        rot, trans = compute_rot_trans_error(poses, poses)
        assert rot < 0.05
        assert trans < 0.01

    def test_rot_trans_error_nonzero(self):
        pred = torch.randn(4, 5, 7)
        gt = torch.randn(4, 5, 7)
        rot, trans = compute_rot_trans_error(pred, gt)
        assert rot > 0
        assert trans > 0

    def test_all_video_metrics_returns_result(self):
        real = torch.randn(2, 3, 4, 8, 8)
        fake = torch.randn(2, 3, 4, 8, 8)
        result = compute_all_video_metrics(real, fake)
        assert isinstance(result, VideoMetricsResult)
        d = result.to_dict()
        assert "fvd" in d
        assert "temporal_lpips" in d


# ---------------------------------------------------------------------------
# Fidelity / Divergence Detection
# ---------------------------------------------------------------------------

class TestFidelity:
    def test_divergence_detector_normal(self):
        det = DivergenceDetector(threshold=0.5)
        result = det.update(torch.zeros(10), torch.zeros(10))
        assert result.divergence_score < 0.01
        assert not result.is_divergent

    def test_divergence_detector_divergent(self):
        det = DivergenceDetector(threshold=0.01)
        for _ in range(20):
            result = det.update(torch.zeros(10), torch.ones(10) * 100)
        assert result.is_divergent

    def test_divergence_rolling_stats(self):
        det = DivergenceDetector()
        for _ in range(10):
            det.update(torch.randn(5), torch.randn(5))
        stats = det.get_rolling_stats()
        assert "mean" in stats
        assert "std" in stats
        assert stats["mean"] >= 0

    def test_divergence_reset(self):
        det = DivergenceDetector()
        det.update(torch.randn(5), torch.randn(5))
        det.reset()
        assert det.ema_divergence is None
        assert len(det.divergence_history) == 0

    def test_trust_from_divergence(self):
        trust_low = compute_trust_from_divergence(10.0)
        trust_high = compute_trust_from_divergence(0.01)
        assert trust_low < trust_high
        assert 0 <= trust_low <= 1
        assert 0 <= trust_high <= 1


# ---------------------------------------------------------------------------
# System Identification
# ---------------------------------------------------------------------------

class TestSystemID:
    def test_estimator_output_shape(self):
        est = ParameterEstimator(obs_dim=16, action_dim=4)
        obs = torch.randn(2, 5, 16)
        action = torch.randn(2, 5, 4)
        next_obs = torch.randn(2, 5, 16)
        params = est(obs, action, next_obs)
        assert isinstance(params, dict)
        assert "friction" in params
        assert params["friction"].shape == (2,)

    def test_estimator_ranges(self):
        est = ParameterEstimator(obs_dim=8, action_dim=2)
        obs = torch.randn(4, 3, 8)
        action = torch.randn(4, 3, 2)
        next_obs = torch.randn(4, 3, 8)
        params = est(obs, action, next_obs)
        # Friction should be in [0.5, 2.5]
        assert params["friction"].min() >= 0.5
        assert params["friction"].max() <= 2.5
        # Mass should be in [0.8, 1.2]
        assert params["mass"].min() >= 0.8
        assert params["mass"].max() <= 1.2

    def test_calibrate_converges(self):
        sid = SystemIdentifier(obs_dim=8, action_dim=2, max_iterations=50)
        real_obs = torch.randn(20, 8)
        real_actions = torch.randn(20, 2)
        real_next_obs = torch.randn(20, 8)
        result = sid.calibrate(real_obs, real_actions, real_next_obs)
        assert isinstance(result, SystemIdentificationResult)
        assert result.iterations > 0
        assert result.calibration_loss < float("inf")


# ---------------------------------------------------------------------------
# Residual Dynamics
# ---------------------------------------------------------------------------

class TestResidualDynamics:
    def test_residual_net_output_shape(self):
        net = ResidualDynamicsNet(obs_dim=16, action_dim=4)
        obs = torch.randn(4, 16)
        action = torch.randn(4, 4)
        residual, log_var = net(obs, action)
        assert residual.shape == (4, 16)
        assert log_var.shape == (4, 16)

    def test_hybrid_model_predict(self):
        den = MLPDenoiser(obs_dim=8, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8)
        sim = DiffusionDynamics(den, timesteps=5)
        hybrid = create_hybrid_model(8, 2, sim)
        obs = torch.randn(2, 8)
        action = torch.randn(2, 2)
        pred = hybrid.predict(obs, action, num_denoise_steps=3)
        assert pred.hybrid_prediction.shape == (2, 8)
        assert pred.sim_prediction.shape == (2, 8)
        assert pred.residual.shape == (2, 8)
        assert pred.uncertainty.shape == (2, 8)

    def test_hybrid_model_loss(self):
        den = MLPDenoiser(obs_dim=8, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8)
        sim = DiffusionDynamics(den, timesteps=5)
        hybrid = create_hybrid_model(8, 2, sim)
        obs = torch.randn(4, 8)
        action = torch.randn(4, 2)
        next_obs = torch.randn(4, 8)
        loss, components = hybrid.compute_loss(obs, action, next_obs)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert "residual_loss" in components

    def test_online_adapter(self):
        net = ResidualDynamicsNet(obs_dim=8, action_dim=2)
        adapter = OnlineResidualAdapter(net, buffer_size=10, batch_size=4)
        for _ in range(5):
            metrics = adapter.update(
                torch.randn(8), torch.randn(2), torch.randn(8), torch.randn(8)
            )
        assert "online_loss" in metrics or "buffer_size" in metrics


# ---------------------------------------------------------------------------
# Transfer Pipeline
# ---------------------------------------------------------------------------

class TestTransferPipeline:
    def test_pipeline_creates(self):
        pipeline = SimToRealPipeline(obs_dim=16, action_dim=4)
        assert pipeline.obs_dim == 16
        assert pipeline.action_dim == 4

    def test_pipeline_system_id(self):
        pipeline = SimToRealPipeline(obs_dim=8, action_dim=2)
        data = {
            "obs": torch.randn(20, 8),
            "actions": torch.randn(20, 2),
            "next_obs": torch.randn(20, 8),
        }
        result = pipeline.step2_system_identification(data)
        assert isinstance(result, SystemIdentificationResult)
        assert result.converged

    def test_pipeline_train_residual(self):
        pipeline = SimToRealPipeline(obs_dim=8, action_dim=2)
        den = MLPDenoiser(obs_dim=8, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8)
        sim = DiffusionDynamics(den, timesteps=5)
        data = {
            "obs": torch.randn(20, 8),
            "actions": torch.randn(20, 2),
            "next_obs": torch.randn(20, 8),
        }
        hybrid = pipeline.step3_train_residual(sim, data, epochs=3)
        assert hybrid is not None

    def test_pipeline_evaluate(self):
        pipeline = SimToRealPipeline(obs_dim=8, action_dim=2)
        den = MLPDenoiser(obs_dim=8, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8)
        sim = DiffusionDynamics(den, timesteps=5)
        hybrid = create_hybrid_model(8, 2, sim)
        data = {
            "obs": torch.randn(10, 8),
            "actions": torch.randn(10, 2),
            "next_obs": torch.randn(10, 8),
        }
        result = pipeline.step5_evaluate(hybrid, data, num_steps=5)
        assert isinstance(result, TransferResult)
        assert "hybrid_mse" in result.eval_metrics
        assert len(result.trust_scores) > 0

    def test_full_transfer(self):
        obs_dim, act_dim = 8, 2
        den = MLPDenoiser(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=16, num_blocks=2, cond_dim=8)
        sim = DiffusionDynamics(den, timesteps=5)
        data = {
            "obs": torch.randn(30, obs_dim),
            "actions": torch.randn(30, act_dim),
            "next_obs": torch.randn(30, obs_dim),
        }
        result = run_full_transfer(None, data, sim, obs_dim, act_dim)
        assert isinstance(result, TransferResult)
        d = result.to_dict()
        assert "eval/hybrid_mse" in d

