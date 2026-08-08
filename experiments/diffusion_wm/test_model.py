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
        assert len(fig.data) == 4 * (3 + 1)  # samples x (milestones + GT)

