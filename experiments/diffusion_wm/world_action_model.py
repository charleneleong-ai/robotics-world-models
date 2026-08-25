"""World Action Model (WAM): jointly denoises next-state AND actions.

Extends DiffusionDynamics with parallel action + state denoising heads.
The WAM IS the policy — during inference it generates action chunks directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    MLPDenoiser,
    SinusoidalEmbedding,
    cosine_beta_schedule,
)


# ---------------------------------------------------------------------------
# Parallel denoiser with action + state heads
# ---------------------------------------------------------------------------

class WAMDenoiser(nn.Module):
    """Parallel denoising heads for actions and next-states.

    Shared backbone processes (obs, noisy_target, timestep).
    State head denoises next-state noise (same as MLPDenoiser).
    Action head denoises action noise (new).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection: [s_t, noisy_target] -> hidden
        # noisy_target is either noisy_s_{t+1} (state head) or noisy_a_t (action head)
        self.input_proj = nn.Linear(obs_dim + max(obs_dim, act_dim), hidden_dim)

        # Timestep embedding -> FiLM scale/bias
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film = nn.Linear(cond_dim, num_blocks * hidden_dim * 2)

        # Shared residual blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "linear1": nn.Linear(hidden_dim, hidden_dim * 4),
                "linear2": nn.Linear(hidden_dim * 4, hidden_dim),
            })
            for _ in range(num_blocks)
        ])

        # State head: predicts next-state noise
        self.state_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Action head: predicts action noise
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        state: torch.Tensor,
        target_type: str,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through shared backbone + selected head.

        Args:
            x_noisy: noisy target [B, dim] (obs_dim for state, act_dim for action)
            state: current observation [B, obs_dim]
            target_type: "state" or "action"
            timestep: diffusion timestep [B]
        """
        # Pad noisy target to max(obs_dim, act_dim) for input projection
        if x_noisy.size(-1) < self.input_proj.in_features - self.obs_dim:
            pad_size = self.input_proj.in_features - self.obs_dim - x_noisy.size(-1)
            x_padded = F.pad(x_noisy, (0, pad_size))
        else:
            x_padded = x_noisy[:, :self.input_proj.in_features - self.obs_dim]

        h = self.input_proj(torch.cat([state, x_padded], dim=-1))

        # FiLM modulation from timestep
        t_emb = self.time_embed(timestep)
        film_params = self.film(t_emb).view(-1, self.num_blocks * 2, self.hidden_dim)
        scales = film_params[:, 0::2]
        biases = film_params[:, 1::2]

        for i, block in enumerate(self.blocks):
            h_norm = block["norm"](h)
            h_mod = h_norm * (1 + scales[:, i]) + biases[:, i]
            gate = F.gelu(block["linear1"](h_mod))
            h = h + block["linear2"](gate)

        if target_type == "state":
            return self.state_head(h)
        elif target_type == "action":
            return self.action_head(h)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


# ---------------------------------------------------------------------------
# World Action Model
# ---------------------------------------------------------------------------

class DiffusionWAM(nn.Module):
    """World Action Model: jointly denoises (next_state, action) from obs.

    Uses parallel state + action denoising heads sharing a backbone.
    Training: joint MSE loss on both heads.
    Inference: generate action chunks directly from observations.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        cond_dim: int = 256,
        timesteps: int = 1000,
        action_horizon: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.timesteps = timesteps
        self.action_horizon = action_horizon

        self.denoiser = WAMDenoiser(obs_dim, act_dim, hidden_dim, num_blocks, cond_dim)

        # Diffusion constants (shared schedule for both state and action)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def _q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def training_loss(
        self,
        obs: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute joint training loss on state + action denoising."""
        B = obs.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=obs.device)

        # State loss
        state_noise = torch.randn_like(next_state)
        x_noisy_state = self._q_sample(next_state, t, state_noise)
        pred_state_noise = self.denoiser(x_noisy_state, obs, "state", t.float())
        state_loss = F.mse_loss(pred_state_noise, state_noise)

        # Action loss
        action_noise = torch.randn_like(action)
        x_noisy_action = self._q_sample(action, t, action_noise)
        pred_action_noise = self.denoiser(x_noisy_action, obs, "action", t.float())
        action_loss = F.mse_loss(pred_action_noise, action_noise)

        total_loss = state_loss + action_loss
        return total_loss, {
            "state_loss": state_loss.item(),
            "action_loss": action_loss.item(),
            "total_loss": total_loss.item(),
        }

    @torch.no_grad()
    def _denoise_target(
        self,
        state: torch.Tensor,
        target_type: str,
        target_dim: int,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Denoise a single target (state or action) conditioned on obs."""
        n = state.size(0)
        T = num_steps or self.timesteps
        device = state.device

        x = torch.randn(n, target_dim, device=device)

        for t_idx in reversed(range(T)):
            t_batch = torch.full((n,), t_idx, device=device, dtype=torch.float)
            pred_noise = self.denoiser(x, state, target_type, t_batch)
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]
            coef1 = 1 / alpha.sqrt()
            coef2 = (1 - alpha) / (1 - alpha_cumprod).sqrt()
            x = coef1 * (x - coef2 * pred_noise)
            if t_idx > 0:
                x += beta.sqrt() * torch.randn_like(x)

        return x

    @torch.no_grad()
    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Predict next state given current state and action (world model use)."""
        return self._denoise_target(state, "state", self.obs_dim, num_steps)

    @torch.no_grad()
    def predict_action(
        self,
        state: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate action from observation (policy use)."""
        return self._denoise_target(state, "action", self.act_dim, num_steps)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        state: torch.Tensor,
        horizon: int | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate a chunk of actions autoregressively.

        Args:
            state: current observation [B, obs_dim]
            horizon: number of future actions to generate
            num_steps: denoising steps per action
        Returns:
            action_chunk [B, horizon, act_dim]
        """
        h = horizon or self.action_horizon
        B = state.size(0)
        actions = []
        s = state

        for _ in range(h):
            a = self.predict_action(s, num_steps)
            actions.append(a)
            # Use predicted next state for next action (autoregressive)
            s = self.predict_next_state(s, a, num_steps)

        return torch.stack(actions, dim=1)  # [B, horizon, act_dim]

    @torch.no_grad()
    def rollout(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        horizon: int = 20,
        num_denoise_steps: int = 100,
    ) -> torch.Tensor:
        """Rollout: predict future states given state sequence and actions.

        Args:
            states: initial states [B, obs_dim]
            actions: sequence of actions [B, T, act_dim] (T >= horizon)
            horizon: number of steps to rollout
        Returns:
            predicted_states [B, horizon+1, obs_dim]
        """
        B = states.size(0)
        preds = [states]
        s = states
        for h in range(horizon):
            a = actions[:, h]
            s_pred = self.predict_next_state(s, a, num_steps=num_denoise_steps)
            preds.append(s_pred)
            s = s_pred
        return torch.stack(preds, dim=1)

    def state_dict(self) -> dict:
        return {
            "denoiser": self.denoiser.state_dict(),
            "timesteps": self.timesteps,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "action_horizon": self.action_horizon,
        }

    def load_state_dict(self, d: dict) -> None:
        self.denoiser.load_state_dict(d["denoiser"])
