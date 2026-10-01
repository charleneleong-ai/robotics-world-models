"""Cascaded World Action Model: a deterministic world model plus a separate diffusion action decoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import SinusoidalEmbedding
from .wam_base import BaseDiffusionWAM, NoiseFn, apply_film_blocks, film_blocks


class WorldModel(nn.Module):
    """Predicts the next state from (obs, action), with FiLM conditioning on the action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512, num_blocks: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.film = nn.Linear(act_dim, num_blocks * hidden_dim * 2)
        self.blocks = film_blocks(hidden_dim, num_blocks)
        self.output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, obs_dim))

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        h = self.input_proj(torch.cat([obs, action], dim=-1))
        return self.output_head(apply_film_blocks(self.blocks, h, self.film(action)))


class ActionDecoder(nn.Module):
    """Predicts action noise from (noisy action, obs, timestep), with FiLM conditioning on timestep and obs."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film = nn.Linear(cond_dim + obs_dim, num_blocks * hidden_dim * 2)
        self.blocks = film_blocks(hidden_dim, num_blocks)
        self.output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, act_dim))

    def forward(self, noisy_action: Tensor, obs: Tensor, timestep: Tensor) -> Tensor:
        h = self.input_proj(torch.cat([obs, noisy_action], dim=-1))
        film = self.film(torch.cat([self.time_embed(timestep), obs], dim=-1))
        return self.output_head(apply_film_blocks(self.blocks, h, film))


class CascadedWAM(BaseDiffusionWAM):
    """Separates dynamics from control: the world model is trained by regression, actions by diffusion."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        wm_blocks: int = 4,
        ad_blocks: int = 4,
        cond_dim: int = 256,
        timesteps: int = 1000,
        action_horizon: int = 1,
    ) -> None:
        super().__init__(obs_dim, act_dim, timesteps, action_horizon)
        self.world_model = WorldModel(obs_dim, act_dim, hidden_dim, wm_blocks)
        self.action_decoder = ActionDecoder(obs_dim, act_dim, hidden_dim, ad_blocks, cond_dim)

    def decoder(self, obs: Tensor) -> NoiseFn:
        return lambda x, t: self.action_decoder(x, obs, t)

    def loss_terms(self, obs: Tensor, next_state: Tensor, action: Tensor) -> dict[str, Tensor]:
        return {
            "wm_loss": F.mse_loss(self.world_model(obs, action), next_state),
            "ad_loss": self.noise_loss(action, self.decoder(obs), self.random_timesteps(obs)),
        }

    def predict_action(self, obs: Tensor, num_steps: int | None = None) -> Tensor:
        return self.sample(self.decoder(obs), (obs.size(0), self.act_dim), num_steps, clip=1.0)

    @torch.no_grad()
    def predict_next_state(self, state: Tensor, action: Tensor, num_steps: int | None = None) -> Tensor:
        return self.world_model(state, action)
