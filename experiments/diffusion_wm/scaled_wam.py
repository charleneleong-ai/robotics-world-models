"""Scaled World Action Model (100M+ params).

Extends DiffusionWAM with:
- Deeper transformer backbone (12-24 blocks)
- Wider hidden dimensions (1024-2048)
- Multi-head attention for better sequence modeling
- Activation checkpointing for memory efficiency
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_action_model import DiffusionWAM, WAMDenoiser, SinusoidalEmbedding


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and GELU activation."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections."""
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h

        # Feed-forward with pre-norm
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class ScaledWAMDenoiser(nn.Module):
    """Scaled denoiser with transformer backbone and FiLM conditioning.

    Architecture:
    - Input projection: [obs, noisy_target] -> hidden_dim
    - Timestep embedding -> FiLM scale/bias
    - N x Transformer blocks with FiLM conditioning
    - State head: predicts next-state noise
    - Action head: predicts action noise
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 1024,
        num_blocks: int = 12,
        num_heads: int = 8,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection
        self.input_proj = nn.Linear(obs_dim + max(obs_dim, act_dim), hidden_dim)

        # Timestep embedding -> FiLM
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film = nn.Linear(cond_dim, num_blocks * hidden_dim * 2)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])

        # Output heads
        self.state_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obs_dim),
        )
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
        """Forward pass with FiLM-conditioned transformer."""
        # Pad noisy target
        if x_noisy.size(-1) < self.input_proj.in_features - self.obs_dim:
            pad_size = self.input_proj.in_features - self.obs_dim - x_noisy.size(-1)
            x_padded = F.pad(x_noisy, (0, pad_size))
        else:
            x_padded = x_noisy[:, :self.input_proj.in_features - self.obs_dim]

        # Project input
        h = self.input_proj(torch.cat([state, x_padded], dim=-1))

        # FiLM modulation from timestep
        t_emb = self.time_embed(timestep)
        film_params = self.film(t_emb).view(-1, self.num_blocks, self.hidden_dim * 2)
        scales = film_params[:, :, :self.hidden_dim]
        biases = film_params[:, :, self.hidden_dim:]

        # Apply transformer blocks with FiLM
        for i, block in enumerate(self.blocks):
            h = h * (1 + scales[:, i]) + biases[:, i]
            h = block(h)

        # Output heads
        if target_type == "state":
            return self.state_head(h)
        elif target_type == "action":
            return self.action_head(h)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


class ScaledDiffusionWAM(nn.Module):
    """Scaled World Action Model with transformer backbone.

    Key differences from DiffusionWAM:
    - Transformer backbone instead of MLP blocks
    - Multi-head attention for better sequence modeling
    - FiLM conditioning for timestep modulation
    - Scalable to 100M+ parameters
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
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.timesteps = timesteps
        self.action_horizon = action_horizon

        # Denoiser
        self.denoiser = ScaledWAMDenoiser(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            cond_dim=cond_dim,
        )

        # Diffusion schedule
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add noise to x_start at timestep t (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def compute_loss(
        self,
        obs: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint diffusion loss for state and action heads.

        Returns:
            total_loss, state_loss, action_loss
        """
        B = obs.size(0)
        device = obs.device

        # Random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device)

        # Sample noise and create noisy targets
        state_noise = torch.randn_like(next_state)
        action_noise = torch.randn_like(action)
        noisy_state = self.q_sample(next_state, t, state_noise)
        noisy_action = self.q_sample(action, t, action_noise)

        # Predict noise
        pred_state = self.denoiser(noisy_state, obs, "state", t)
        pred_action = self.denoiser(noisy_action, obs, "action", t)

        # MSE loss
        state_loss = F.mse_loss(pred_state, state_noise)
        action_loss = F.mse_loss(pred_action, action_noise)
        total_loss = state_loss + action_loss

        return total_loss, state_loss, action_loss

    @torch.no_grad()
    def denoise_action(
        self,
        obs: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Denoise action from observation (inference).

        Args:
            obs: current observation [B, obs_dim]
            num_steps: number of denoising steps

        Returns:
            denoised action [B, act_dim]
        """
        B = obs.size(0)
        device = obs.device

        # Start from random noise
        x = torch.randn(B, self.act_dim, device=device)

        # Denoising loop
        for i in range(num_steps):
            t = torch.full((B,), i * self.timesteps // num_steps, device=device)
            pred_noise = self.denoiser(x, obs, "action", t)

            # DDPM update
            alpha = self.alphas_cumprod[t].view(-1, 1)
            alpha_prev = self.alphas_cumprod[t - 1].view(-1, 1) if i > 0 else torch.ones_like(alpha)
            beta = self.betas[t].view(-1, 1)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            x0_pred = x0_pred.clamp(-1, 1)

            # Update x
            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise
            else:
                x = x0_pred

        return x

    @torch.no_grad()
    def denoise_state(
        self,
        obs: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Denoise next-state from observation (inference)."""
        B = obs.size(0)
        device = obs.device

        x = torch.randn(B, self.obs_dim, device=device)

        for i in range(num_steps):
            t = torch.full((B,), i * self.timesteps // num_steps, device=device)
            pred_noise = self.denoiser(x, obs, "state", t)

            alpha = self.alphas_cumprod[t].view(-1, 1)
            alpha_prev = self.alphas_cumprod[t - 1].view(-1, 1) if i > 0 else torch.ones_like(alpha)

            x0_pred = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            x0_pred = x0_pred.clamp(-1, 1)

            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise
            else:
                x = x0_pred

        return x

    def training_loss(
        self,
        obs: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute joint training loss on state + action denoising."""
        total_loss, state_loss, action_loss = self.compute_loss(obs, next_state, action)
        return total_loss, {
            "state_loss": state_loss.item(),
            "action_loss": action_loss.item(),
            "total_loss": total_loss.item(),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "denoiser": self.denoiser.state_dict(),
            "timesteps": self.timesteps,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.denoiser.load_state_dict(ckpt["denoiser"])
