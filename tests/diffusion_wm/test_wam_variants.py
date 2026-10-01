"""Tests for the shared WAM base, the scaled and cascaded variants, and checkpoint loading."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.diffusion_wm.cascaded_wam import CascadedWAM
from experiments.diffusion_wm.scaled_wam import ScaledDiffusionWAM
from experiments.diffusion_wm.wam_base import BaseDiffusionWAM, pad_to
from experiments.diffusion_wm.wam_factory import load_wam
from experiments.diffusion_wm.world_action_model import DiffusionWAM

OBS_DIM, ACT_DIM, BATCH, TIMESTEPS = 6, 3, 8, 50
CHECKPOINT_BUFFERS = {"betas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"}


def scaled() -> ScaledDiffusionWAM:
    return ScaledDiffusionWAM(
        OBS_DIM, ACT_DIM, hidden_dim=32, num_blocks=2, num_heads=4, cond_dim=16, timesteps=TIMESTEPS,
    )


def cascaded() -> CascadedWAM:
    return CascadedWAM(OBS_DIM, ACT_DIM, hidden_dim=32, wm_blocks=2, ad_blocks=2, cond_dim=16, timesteps=TIMESTEPS)


VARIANTS = [scaled, cascaded]


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    obs = torch.randn(BATCH, OBS_DIM)
    return obs, torch.randn(BATCH, OBS_DIM), torch.tanh(torch.randn(BATCH, ACT_DIM))


@pytest.mark.parametrize(("width", "kept"), [(5, 3), (2, 2)])
def test_pad_to(width: int, kept: int) -> None:
    x = torch.arange(3.0).expand(2, 3)
    out = pad_to(x, width)
    assert out.shape == (2, width)
    assert torch.equal(out[:, :kept], x[:, :kept])
    assert not out[:, kept:].any()


class TestSampler:
    def test_visits_every_sampled_step_from_noisiest_to_clean(self) -> None:
        model, visited = scaled(), []

        def spy(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            visited.append(int(t[0]))
            return torch.zeros_like(x)

        model.sample(spy, (BATCH, ACT_DIM), num_steps=10)
        assert len(visited) == 10
        assert visited[0] == TIMESTEPS - 1 and visited[-1] == 0
        assert visited == sorted(visited, reverse=True)

    @pytest.mark.parametrize("clip", [None, 0.5])
    def test_exact_noise_recovers_x0(self, clip: float | None) -> None:
        model = scaled()
        x0 = torch.rand(BATCH, ACT_DIM) * 1.6 - 0.8

        def oracle(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            alpha = model.alphas_cumprod[t][:, None]
            return (x - alpha.sqrt() * x0) / (1 - alpha).sqrt()

        out = model.sample(oracle, (BATCH, ACT_DIM), num_steps=10, clip=clip)
        expected = x0 if clip is None else x0.clamp(-clip, clip)
        assert torch.allclose(out, expected, atol=1e-4)


@pytest.mark.parametrize("make", VARIANTS, ids=lambda f: f.__name__)
class TestVariants:
    def test_loss_is_sum_of_terms_and_reaches_every_parameter(self, make, batch) -> None:
        model: BaseDiffusionWAM = make()
        total, logged = model.training_loss(*batch)
        total.backward()
        terms = {k: v for k, v in logged.items() if k != "total_loss"}
        assert len(terms) == 2
        assert logged["total_loss"] == pytest.approx(sum(terms.values()), rel=1e-5)
        assert all(p.grad is not None for p in model.parameters())

    def test_action_chunk_shape(self, make, batch) -> None:
        model: BaseDiffusionWAM = make().eval()
        assert model.predict_action_chunk(batch[0], horizon=3, num_steps=5).shape == (BATCH, 3, ACT_DIM)

    def test_learns_a_fixed_action(self, make, batch) -> None:
        torch.manual_seed(0)
        model: BaseDiffusionWAM = make()
        obs, next_state, _ = batch
        target = torch.full((BATCH, ACT_DIM), 0.6)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(400):
            opt.zero_grad()
            model.training_loss(obs, next_state, target)[0].backward()
            opt.step()
        pred = model.eval().predict_action(obs, num_steps=20)
        assert (pred - target).pow(2).mean().item() < 0.05


def test_chunk_feeds_predicted_state_forward(batch) -> None:
    model = cascaded().eval()

    def chunk() -> torch.Tensor:
        torch.manual_seed(0)
        return model.predict_action_chunk(batch[0], horizon=2, num_steps=5)

    before = chunk()
    with torch.no_grad():
        model.world_model.output_head[1].bias.add_(1.0)
    after = chunk()
    assert torch.equal(before[:, 0], after[:, 0])
    assert not torch.equal(before[:, 1], after[:, 1])


def test_cascaded_next_state_depends_on_action(batch) -> None:
    model = cascaded().eval()
    obs = batch[0]
    assert not torch.allclose(
        model.predict_next_state(obs, torch.zeros(BATCH, ACT_DIM)),
        model.predict_next_state(obs, torch.ones(BATCH, ACT_DIM)),
    )


def test_scaled_state_dict_matches_existing_checkpoints() -> None:
    keys = set(scaled().state_dict())
    assert CHECKPOINT_BUFFERS <= keys
    assert {k.split(".")[0] for k in keys} == CHECKPOINT_BUFFERS | {"denoiser"}


@pytest.mark.parametrize(
    ("hidden_dim", "expected"),
    [(32, DiffusionWAM), (520, ScaledDiffusionWAM)],
    ids=["mlp", "scaled"],
)
def test_load_wam_roundtrip(tmp_path: Path, hidden_dim: int, expected: type) -> None:
    config = {"hidden_dim": hidden_dim, "num_blocks": 1, "cond_dim": 16, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
              "diffusion_timesteps": TIMESTEPS, "action_horizon": 1}
    kwargs = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden_dim": hidden_dim, "num_blocks": 1, "cond_dim": 16,
              "timesteps": TIMESTEPS}
    original = expected(**kwargs)
    path = tmp_path / "best.pt"
    torch.save({"model": original.state_dict(), "config": config}, path)

    loaded = load_wam(path, "cpu")
    assert type(loaded) is expected
    assert not loaded.training
    for (name, p1), (_, p2) in zip(original.denoiser.named_parameters(), loaded.denoiser.named_parameters()):
        assert torch.equal(p1, p2), name
