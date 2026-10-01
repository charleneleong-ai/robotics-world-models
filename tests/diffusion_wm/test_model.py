"""Tests for the diffusion dynamics model, sim-to-real components, and the joint DiffusionWAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, random_split

from experiments.diffusion_wm.dataset import TransitionDataset
from experiments.diffusion_wm.domain_rand import (
    ActionNoise,
    DomainRandomizationConfig,
    ObservationNoise,
    apply_action_noise,
    apply_observation_noise,
)
from experiments.diffusion_wm.fidelity import DivergenceDetector, compute_trust_from_divergence
from experiments.diffusion_wm.model import DiffusionDynamics, MLPDenoiser, cosine_beta_schedule
from experiments.diffusion_wm.residual_dynamics import (
    OnlineResidualAdapter,
    ResidualDynamicsNet,
    create_hybrid_model,
)
from experiments.diffusion_wm.system_id import ParameterEstimator, SystemIdentificationResult
from experiments.diffusion_wm.train import _cache_media_pool, _media_pools
from experiments.diffusion_wm.transfer import SimToRealPipeline, TransferResult, run_full_transfer
from experiments.diffusion_wm.video_metrics import (
    I3DFeatureExtractor,
    VideoMetricsResult,
    _frechet_distance,
    compute_all_video_metrics,
    compute_fvd,
    compute_idm_error,
    compute_rot_trans_error,
    compute_temporal_lpips,
)
from experiments.diffusion_wm.viz import denoising_grid
from experiments.diffusion_wm.world_action_model import DiffusionWAM, WAMDenoiser

OBS_DIM, ACT_DIM = 16, 4


def tiny_dynamics(obs_dim: int = 4, act_dim: int = 2, timesteps: int = 10) -> DiffusionDynamics:
    den = MLPDenoiser(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=16, num_blocks=2, cond_dim=8)
    return DiffusionDynamics(den, timesteps=timesteps)


def transitions(n: int, obs_dim: int = 8, act_dim: int = 2) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(n, obs_dim),
        "actions": torch.randn(n, act_dim),
        "next_obs": torch.randn(n, obs_dim),
    }


@pytest.fixture
def denoiser() -> MLPDenoiser:
    return MLPDenoiser(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=64, num_blocks=3, cond_dim=32)


@pytest.fixture
def model(denoiser: MLPDenoiser) -> DiffusionDynamics:
    return DiffusionDynamics(denoiser, timesteps=100)


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(8, OBS_DIM),
        "action": torch.randn(8, ACT_DIM),
        "next_obs": torch.randn(8, OBS_DIM),
    }


@pytest.fixture
def wam() -> DiffusionWAM:
    return DiffusionWAM(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=64, num_blocks=3, cond_dim=32, timesteps=100, action_horizon=1,
    )


class TestNoiseSchedule:
    def test_cosine_beta_range(self) -> None:
        betas = cosine_beta_schedule(1000)
        assert betas.shape == (1000,)
        assert betas.min() >= 0.0001
        assert betas.max() <= 0.02

    def test_cosine_beta_monotonic(self) -> None:
        betas = cosine_beta_schedule(100)
        assert torch.all(betas[1:] >= betas[:-1] - 1e-6)


class TestMLPDenoiser:
    def test_batch_independence(self, denoiser: MLPDenoiser, batch: dict[str, torch.Tensor]) -> None:
        t = torch.randint(0, 100, (8,)).float()
        out = denoiser(batch["next_obs"], batch["obs"], batch["action"], t)
        assert out.shape == batch["next_obs"].shape
        perturbed = batch["next_obs"].clone()
        perturbed[0] += 10.0
        out2 = denoiser(perturbed, batch["obs"], batch["action"], t)
        assert not torch.allclose(out[0], out2[0])
        assert torch.allclose(out[1:], out2[1:], atol=1e-6)

    def test_timestep_conditioning(self, denoiser: MLPDenoiser, batch: dict[str, torch.Tensor]) -> None:
        out0 = denoiser(batch["next_obs"], batch["obs"], batch["action"], torch.zeros(8))
        out1 = denoiser(batch["next_obs"], batch["obs"], batch["action"], torch.full((8,), 50.0))
        assert not torch.allclose(out0, out1, atol=1e-4)


class TestDiffusionDynamics:
    def test_training_loss_decreases(self, model: DiffusionDynamics, batch: dict[str, torch.Tensor]) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(100):
            opt.zero_grad()
            loss = model(batch["next_obs"], batch["obs"], batch["action"])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_sample_deterministic_with_seed(self, model: DiffusionDynamics, batch: dict[str, torch.Tensor]) -> None:
        torch.manual_seed(0)
        pred1 = model.sample(batch["obs"], batch["action"], num_steps=10)
        torch.manual_seed(0)
        pred2 = model.sample(batch["obs"], batch["action"], num_steps=10)
        assert pred1.shape == batch["next_obs"].shape
        assert torch.allclose(pred1, pred2, atol=1e-4)

    def test_rollout_output_shape(self, model: DiffusionDynamics, batch: dict[str, torch.Tensor]) -> None:
        actions = torch.randn(8, 5, ACT_DIM)
        traj = model.rollout(batch["obs"], actions, horizon=5, num_denoise_steps=10)
        assert traj.shape == (8, 6, OBS_DIM)

    def test_q_sample_is_clean_at_t0(self, model: DiffusionDynamics, batch: dict[str, torch.Tensor]) -> None:
        noise = torch.randn_like(batch["next_obs"])
        x_t0 = model._q_sample(batch["next_obs"], torch.zeros(8, dtype=torch.long), noise)
        assert torch.allclose(x_t0, batch["next_obs"], atol=0.1)

    def test_backward_reaches_every_parameter(self) -> None:
        small = tiny_dynamics()
        small(torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 2)).backward()
        assert all(p.grad is not None for p in small.parameters() if p.requires_grad)


class TestMediaPool:
    @pytest.fixture
    def loaders(self, tmp_path: Path) -> tuple[DataLoader, DataLoader]:
        rng = np.random.default_rng(0)
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        np.savez_compressed(
            shard_dir / "shard_00000.npz",
            obs=rng.normal(size=(256, 4)),
            action=rng.normal(size=(256, 2)),
            next_obs=rng.normal(size=(256, 4)),
        )
        ds = TransitionDataset(shard_dir)
        train_ds, val_ds = random_split(ds, [len(ds) - len(ds) // 5, len(ds) // 5])
        return DataLoader(train_ds, batch_size=32), DataLoader(val_ds, batch_size=32)

    @pytest.fixture(autouse=True)
    def reset_pool(self) -> None:
        _media_pools.clear()
        yield
        _media_pools.clear()

    def test_caches_both_splits(self, loaders: tuple[DataLoader, DataLoader]) -> None:
        _cache_media_pool(*loaders, torch.device("cpu"))
        assert set(_media_pools) == {"train", "val"}
        assert all(pool["obs"].size(0) == 32 for pool in _media_pools.values())

    def test_caching_is_idempotent(self, loaders: tuple[DataLoader, DataLoader]) -> None:
        _cache_media_pool(*loaders, torch.device("cpu"))
        train_obs = _media_pools["train"]["obs"].clone()
        _cache_media_pool(*loaders, torch.device("cpu"))
        assert torch.equal(_media_pools["train"]["obs"], train_obs)


class TestDenoiseWithProgress:
    def test_milestones(self, model: DiffusionDynamics, batch: dict[str, torch.Tensor]) -> None:
        ests = model.denoise_with_progress(batch["obs"], batch["action"], num_steps=100, milestones=(75, 50, 25, 0))
        assert [e.shape for e in ests] == [batch["next_obs"].shape] * 4
        assert len(model.denoise_with_progress(batch["obs"], batch["action"], num_steps=100)) == 4

    def test_trained_model_denoises_toward_gt(self) -> None:
        torch.manual_seed(0)
        m = tiny_dynamics()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        obs, action = torch.randn(64, 4), torch.randn(64, 2)
        next_obs = obs + torch.cat([action, action], dim=1)
        for _ in range(400):
            opt.zero_grad()
            m(next_obs, obs, action).backward()
            opt.step()
        ests = m.denoise_with_progress(obs[:8], action[:8], num_steps=10, milestones=(7, 5, 3, 0))
        errs = [(e - next_obs[:8]).pow(2).mean().item() for e in ests]
        assert errs == sorted(errs, reverse=True)
        assert errs[-1] < errs[0]


def test_denoising_grid_figure() -> None:
    fig = denoising_grid(
        tiny_dynamics(), torch.randn(4, 4), torch.randn(4, 2), torch.randn(4, 4), milestones=(7, 3, 0), num_steps=10,
    )
    assert len(fig.data) == 4 * 3 * 2


NOISE_CASES = [
    ("observation", apply_observation_noise, (4, 38), ObservationNoise(enabled=False)),
    ("action", apply_action_noise, (4, 2), ActionNoise(enabled=False)),
]


class TestDomainRandomization:
    @pytest.mark.parametrize(("field", "apply", "shape", "disabled"), NOISE_CASES, ids=[c[0] for c in NOISE_CASES])
    def test_noise_applied_unless_disabled(self, field: str, apply, shape: tuple[int, int], disabled) -> None:
        x = torch.randn(*shape)
        noisy = apply(x, DomainRandomizationConfig())
        assert noisy.shape == x.shape
        assert not torch.equal(noisy, x)
        assert torch.equal(apply(x, DomainRandomizationConfig(**{field: disabled})), x)

    def test_action_noise_is_reproducible_with_generator(self) -> None:
        action = torch.zeros(4, 2)
        draws = [
            apply_action_noise(action, DomainRandomizationConfig(), rng=torch.Generator().manual_seed(7))
            for _ in range(2)
        ]
        assert torch.equal(*draws)


@pytest.fixture(scope="module")
def extractor() -> I3DFeatureExtractor:
    torch.manual_seed(0)
    return I3DFeatureExtractor(torch.device("cpu"))


class TestVideoMetrics:
    def test_fvd_is_zero_for_identical_and_symmetric(self, extractor: I3DFeatureExtractor) -> None:
        a = torch.randn(16, 3, 8, 16, 16)
        b = torch.randn(16, 3, 8, 16, 16) * 2 + 5
        fvd_ab = compute_fvd(a, b, extractor=extractor)
        assert compute_fvd(a, a, extractor=extractor) == pytest.approx(0.0, abs=1e-3 * fvd_ab)
        assert fvd_ab > 0
        assert fvd_ab == pytest.approx(compute_fvd(b, a, extractor=extractor), rel=1e-3)

    @pytest.mark.parametrize("dim", [1, 4])
    def test_frechet_distance_closed_form(self, dim: int) -> None:
        mu1, mu2 = torch.zeros(dim), torch.ones(dim)
        fd = _frechet_distance(mu1, torch.eye(dim), mu2, 4 * torch.eye(dim))
        assert fd == pytest.approx(2 * dim, rel=1e-5)

    @pytest.mark.parametrize(
        ("metric", "args"),
        [
            (compute_temporal_lpips, (torch.randn(2, 1, 4, 8, 8), torch.randn(2, 1, 4, 8, 8))),
            (compute_idm_error, (torch.randn(2, 1, 3, 8, 8), torch.randn(2, 1, 2))),
        ],
        ids=["temporal_lpips", "idm"],
    )
    def test_short_or_single_channel_video_scores_zero(self, metric, args: tuple[torch.Tensor, ...]) -> None:
        assert metric(*args) == 0.0

    def test_rot_trans_error(self) -> None:
        poses = torch.randn(4, 5, 7)
        rot, trans = compute_rot_trans_error(poses, poses)
        assert rot < 0.05 and trans < 0.01
        rot, trans = compute_rot_trans_error(poses, torch.randn(4, 5, 7))
        assert rot > 0 and trans > 0

    def test_all_video_metrics(self) -> None:
        result = compute_all_video_metrics(torch.randn(2, 3, 4, 8, 8), torch.randn(2, 3, 4, 8, 8))
        assert isinstance(result, VideoMetricsResult)
        assert {"fvd", "temporal_lpips"} <= set(result.to_dict())


class TestFidelity:
    def test_divergence_detector(self) -> None:
        assert not DivergenceDetector(threshold=0.5).update(torch.zeros(10), torch.zeros(10)).is_divergent
        det = DivergenceDetector(threshold=0.01)
        for _ in range(20):
            result = det.update(torch.zeros(10), torch.ones(10) * 100)
        assert result.is_divergent

    def test_divergence_rolling_stats_and_reset(self) -> None:
        det = DivergenceDetector()
        for _ in range(10):
            det.update(torch.randn(5), torch.randn(5))
        assert det.get_rolling_stats()["mean"] >= 0
        det.reset()
        assert det.ema_divergence is None
        assert len(det.divergence_history) == 0

    def test_trust_decreases_with_divergence(self) -> None:
        low, high = compute_trust_from_divergence(10.0), compute_trust_from_divergence(0.01)
        assert 0 <= low < high <= 1


class TestSystemID:
    def test_estimator_shapes_and_ranges(self) -> None:
        estimator = ParameterEstimator(obs_dim=8, action_dim=2)
        params = estimator(torch.randn(4, 3, 8), torch.randn(4, 3, 2), torch.randn(4, 3, 8))
        assert params["friction"].shape == (4,)
        assert 0.5 <= params["friction"].min() <= params["friction"].max() <= 2.5
        assert 0.8 <= params["mass"].min() <= params["mass"].max() <= 1.2


class TestResidualDynamics:
    def test_residual_net_output_shape(self) -> None:
        residual, log_var = ResidualDynamicsNet(obs_dim=16, action_dim=4)(torch.randn(4, 16), torch.randn(4, 4))
        assert residual.shape == log_var.shape == (4, 16)

    def test_hybrid_model_predict_and_loss(self) -> None:
        hybrid = create_hybrid_model(8, 2, tiny_dynamics(8, 2, timesteps=5))
        pred = hybrid.predict(torch.randn(2, 8), torch.randn(2, 2), num_denoise_steps=3)
        for field in ("hybrid_prediction", "sim_prediction", "residual", "uncertainty"):
            assert getattr(pred, field).shape == (2, 8)
        loss, components = hybrid.compute_loss(torch.randn(4, 8), torch.randn(4, 2), torch.randn(4, 8))
        assert loss.ndim == 0 and loss.item() > 0
        assert "residual_loss" in components

    def test_online_adapter_trains_once_buffer_fills(self) -> None:
        adapter = OnlineResidualAdapter(ResidualDynamicsNet(obs_dim=8, action_dim=2), buffer_size=10, batch_size=4)
        metrics = [adapter.update(torch.randn(8), torch.randn(2), torch.randn(8), torch.randn(8)) for _ in range(5)]
        assert "online_loss" not in metrics[0]
        assert "online_loss" in metrics[-1]


class TestTransferPipeline:
    def test_system_identification_converges(self) -> None:
        result = SimToRealPipeline(obs_dim=8, action_dim=2).step2_system_identification(transitions(20))
        assert isinstance(result, SystemIdentificationResult)
        assert result.converged

    def test_evaluate(self) -> None:
        hybrid = create_hybrid_model(8, 2, tiny_dynamics(8, 2, timesteps=5))
        result = SimToRealPipeline(obs_dim=8, action_dim=2).step5_evaluate(hybrid, transitions(10), num_steps=5)
        assert isinstance(result, TransferResult)
        assert "hybrid_mse" in result.eval_metrics
        assert len(result.trust_scores) > 0

    def test_full_transfer(self) -> None:
        result = run_full_transfer(None, transitions(30), tiny_dynamics(8, 2, timesteps=5), 8, 2)
        assert "eval/hybrid_mse" in result.to_dict()


class TestDiffusionWAM:
    @pytest.mark.parametrize(("head", "dim"), [("state", OBS_DIM), ("action", ACT_DIM)])
    def test_denoiser_head_shapes(self, head: str, dim: int) -> None:
        den = WAMDenoiser(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=64, num_blocks=3, cond_dim=32)
        out = den(torch.randn(8, dim), torch.randn(8, OBS_DIM), head, torch.randint(0, 100, (8,)).float())
        assert out.shape == (8, dim)

    def test_training_loss_components(self, wam: DiffusionWAM, batch: dict[str, torch.Tensor]) -> None:
        total, losses = wam.training_loss(batch["obs"], batch["next_obs"], batch["action"])
        assert total.ndim == 0
        assert losses["state_loss"] > 0 and losses["action_loss"] > 0
        assert losses["total_loss"] == pytest.approx(losses["state_loss"] + losses["action_loss"], rel=1e-5)

    def test_prediction_shapes(self, wam: DiffusionWAM, batch: dict[str, torch.Tensor]) -> None:
        assert wam.predict_action(batch["obs"], num_steps=10).shape == (8, ACT_DIM)
        assert wam.predict_next_state(batch["obs"], batch["action"], num_steps=10).shape == (8, OBS_DIM)
        assert wam.predict_action_chunk(batch["obs"], horizon=5, num_steps=10).shape == (8, 5, ACT_DIM)
        traj = wam.rollout(batch["obs"], torch.randn(8, 5, ACT_DIM), horizon=5, num_denoise_steps=10)
        assert traj.shape == (8, 6, OBS_DIM)

    def test_state_dict_roundtrip(self, wam: DiffusionWAM) -> None:
        clone = DiffusionWAM(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=64, num_blocks=3, cond_dim=32, timesteps=wam.timesteps,
        )
        clone.load_state_dict(wam.state_dict())
        for (name, p1), (_, p2) in zip(wam.named_parameters(), clone.named_parameters()):
            assert torch.equal(p1, p2), name

    def test_backward_reaches_every_parameter(self) -> None:
        small = DiffusionWAM(obs_dim=4, act_dim=2, hidden_dim=16, num_blocks=2, cond_dim=8, timesteps=10)
        loss, _ = small.training_loss(torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 2))
        loss.backward()
        assert all(p.grad is not None for p in small.parameters() if p.requires_grad)

    def test_overfits_tiny_batch(self) -> None:
        torch.manual_seed(0)
        small = DiffusionWAM(obs_dim=4, act_dim=2, hidden_dim=32, num_blocks=2, cond_dim=16, timesteps=10)
        opt = torch.optim.AdamW(small.parameters(), lr=1e-2)
        obs, action = torch.randn(8, 4), torch.randn(8, 2)
        next_obs = 1.5 * obs
        for _ in range(200):
            opt.zero_grad()
            small.training_loss(obs, next_obs, action)[0].backward()
            opt.step()
        pred = small.predict_next_state(obs, action, num_steps=10)
        assert (pred - next_obs).pow(2).mean().item() < 1.0
