"""Scaled World Action Model: transformer denoiser with FiLM timestep conditioning (100M+ params)."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import SinusoidalEmbedding
from .wam_base import BaseDiffusionWAM, NoiseFn, pad_to


class TransformerBlock(nn.Module):
    """Pre-norm self-attention and GELU feed-forward block."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        return x + self.ff(self.norm2(x))


class ScaledWAMDenoiser(nn.Module):
    """Shared transformer trunk over [obs, noisy target] with separate state and action noise heads."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 1024,
        num_blocks: int = 12,
        num_heads: int = 8,
        cond_dim: int = 512,
        condition_on_action: bool = False,
    ) -> None:
        super().__init__()
        self.target_dim = max(obs_dim, act_dim)
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.input_proj = nn.Linear(obs_dim + self.target_dim, hidden_dim)
        self.state_action_proj = nn.Linear(act_dim, hidden_dim) if condition_on_action else None
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film = nn.Linear(cond_dim, num_blocks * hidden_dim * 2)
        self.blocks = nn.ModuleList(TransformerBlock(hidden_dim, num_heads) for _ in range(num_blocks))
        self.state_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, obs_dim))
        self.action_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, act_dim))

    def forward(
        self, x_noisy: Tensor, state: Tensor, target_type: str, timestep: Tensor, action: Tensor | None = None,
    ) -> Tensor:
        head = {"state": self.state_head, "action": self.action_head}[target_type]
        h = self.input_proj(torch.cat([state, pad_to(x_noisy, self.target_dim)], dim=-1))
        if target_type == "state" and self.state_action_proj is not None:
            h = h + self.state_action_proj(action)
        film = self.film(self.time_embed(timestep)).view(-1, self.num_blocks, self.hidden_dim * 2)
        scales, shifts = film[..., : self.hidden_dim], film[..., self.hidden_dim :]
        for i, block in enumerate(self.blocks):
            h = block(h * (1 + scales[:, i]) + shifts[:, i])
        return head(h)


class ScaledDiffusionWAM(BaseDiffusionWAM):
    """Jointly denoises next state and action from the observation with a transformer denoiser.

    With `condition_on_action` the state head also sees the action; otherwise `predict_next_state` ignores it.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 1024,
        num_blocks: int = 12,
        num_heads: int = 8,
        cond_dim: int = 512,
        timesteps: int = 1000,
        action_horizon: int = 1,
        condition_on_action: bool = False,
    ) -> None:
        super().__init__(obs_dim, act_dim, timesteps, action_horizon)
        self.denoiser = ScaledWAMDenoiser(
            obs_dim, act_dim, hidden_dim, num_blocks, num_heads, cond_dim, condition_on_action,
        )

    def head(self, obs: Tensor, target_type: str, action: Tensor | None = None) -> NoiseFn:
        return lambda x, t: self.denoiser(x, obs, target_type, t, action)

    def loss_terms(self, obs: Tensor, next_state: Tensor, action: Tensor) -> dict[str, Tensor]:
        t = self.random_timesteps(obs)
        return {
            "state_loss": self.noise_loss(next_state, self.head(obs, "state", action), t),
            "action_loss": self.noise_loss(action, self.head(obs, "action"), t),
        }

    def predict_action(self, obs: Tensor, num_steps: int | None = None) -> Tensor:
        return self.sample(self.head(obs, "action"), (obs.size(0), self.act_dim), num_steps, clip=1.0)

    def predict_next_state(self, state: Tensor, action: Tensor, num_steps: int | None = None) -> Tensor:
        return self.sample(self.head(state, "state", action), (state.size(0), self.obs_dim), num_steps)
