"""Tests for the opt-in action-conditioned state head and the planner's disagreement bonus."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from experiments.diffusion_wm.scaled_wam import ScaledDiffusionWAM
from experiments.diffusion_wm.wam_factory import load_wam
from experiments.diffusion_wm.wm_planner import CEMConfig, WMPlanner
from experiments.diffusion_wm.world_action_model import DiffusionWAM

OBS_DIM, ACT_DIM, BATCH, TIMESTEPS = 6, 3, 4, 20


def mlp_wam(condition_on_action: bool) -> DiffusionWAM:
    return DiffusionWAM(OBS_DIM, ACT_DIM, hidden_dim=32, num_blocks=2, cond_dim=16, timesteps=TIMESTEPS,
                        condition_on_action=condition_on_action)


def scaled_wam(condition_on_action: bool) -> ScaledDiffusionWAM:
    return ScaledDiffusionWAM(OBS_DIM, ACT_DIM, hidden_dim=32, num_blocks=2, num_heads=4, cond_dim=16,
                              timesteps=TIMESTEPS, condition_on_action=condition_on_action)


def next_state(model: DiffusionWAM | ScaledDiffusionWAM, obs: Tensor, action: Tensor) -> Tensor:
    torch.manual_seed(0)
    return model.predict_next_state(obs, action, num_steps=5)


@pytest.mark.parametrize("make", [mlp_wam, scaled_wam], ids=["mlp", "scaled"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_next_state_depends_on_action_only_when_conditioned(make, conditioned: bool) -> None:
    torch.manual_seed(0)
    model = make(conditioned).eval()
    obs = torch.randn(BATCH, OBS_DIM)
    same = torch.equal(next_state(model, obs, torch.zeros(BATCH, ACT_DIM)),
                       next_state(model, obs, torch.ones(BATCH, ACT_DIM)))
    assert same is not conditioned


@pytest.mark.parametrize("make", [mlp_wam, scaled_wam], ids=["mlp", "scaled"])
def test_conditioned_state_loss_trains_the_action_projection(make) -> None:
    model = make(True)
    loss, _ = model.training_loss(torch.randn(BATCH, OBS_DIM), torch.randn(BATCH, OBS_DIM), torch.randn(BATCH, ACT_DIM))
    loss.backward()
    assert model.denoiser.state_action_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(("hidden_dim", "cls"), [(32, DiffusionWAM), (520, ScaledDiffusionWAM)], ids=["mlp", "scaled"])
def test_load_wam_restores_conditioning(tmp_path: Path, hidden_dim: int, cls: type) -> None:
    kwargs = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden_dim": hidden_dim, "num_blocks": 1, "cond_dim": 16,
              "timesteps": TIMESTEPS, "condition_on_action": True}
    original = cls(**kwargs)
    config = {"hidden_dim": hidden_dim, "num_blocks": 1, "cond_dim": 16, "diffusion_timesteps": TIMESTEPS,
              "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "condition_on_action": True}
    path = tmp_path / "best.pt"
    torch.save({"model": original.state_dict(), "config": config}, path)
    loaded = load_wam(path, "cpu")
    assert torch.equal(loaded.denoiser.state_action_proj.weight, original.denoiser.state_action_proj.weight)


class NoisyWhereActionIsLarge:
    """Next-state samples whose spread grows with |action|, so disagreement should favour large actions."""

    act_dim = ACT_DIM

    def __init__(self) -> None:
        self.anchor = torch.nn.Linear(1, 1)

    def parameters(self):
        return self.anchor.parameters()

    def predict_next_state(self, state: Tensor, action: Tensor, num_steps: int | None = None) -> Tensor:
        return state + action.abs().sum(dim=1, keepdim=True) * torch.randn_like(state)


def test_disagreement_bonus_scores_uncertain_actions_higher() -> None:
    torch.manual_seed(0)
    planner = WMPlanner(NoisyWhereActionIsLarge(), CEMConfig(horizon=2, exploration_weight=1.0, uncertainty_samples=8))
    candidates = torch.stack([torch.zeros(2, ACT_DIM), torch.full((2, ACT_DIM), 0.9)])
    calm, uncertain = planner._evaluate_sequences(torch.zeros(1, OBS_DIM), candidates)
    assert calm == 0
    assert uncertain > 0
