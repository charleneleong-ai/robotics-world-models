"""Cascaded World Action Model.

Architecture:
1. World Model: predicts next_state from (obs, action)
2. Action Decoder: predicts action from (obs, noisy_action, timestep)

This separates world modeling from action generation, allowing:
- Independent training of world model
- Simpler action decoder (no diffusion needed)
- Better interpretability
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_action_model import SinusoidalEmbedding


class WorldModel(nn.Module):
    """MLP-based world model: predicts next_state from (obs, action).

    Uses FiLM conditioning for action conditioning.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Input: [obs, action]
        self.input_proj = nn.Linear(obs_dim + act_dim, hidden_dim)

        # FiLM conditioning from action
        self.film = nn.Linear(act_dim, num_blocks * hidden_dim * 2)

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "linear1": nn.Linear(hidden_dim, hidden_dim * 4),
                "linear2": nn.Linear(hidden_dim * 4, hidden_dim),
            })
            for _ in range(num_blocks)
        ])

        # Output: predicted next_state
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next_state from (obs, action)."""
        h = self.input_proj(torch.cat([obs, action], dim=-1))

        # FiLM conditioning
        film_params = self.film(action).view(-1, len(self.blocks) * 2, self.hidden_dim)
        scales = film_params[:, 0::2]
        biases = film_params[:, 1::2]

        for i, block in enumerate(self.blocks):
            h_norm = block["norm"](h)
            h_mod = h_norm * (1 + scales[:, i]) + biases[:, i]
            gate = F.gelu(block["linear1"](h_mod))
            h = h + block["linear2"](gate)

        return self.output_head(h)


class ActionDecoder(nn.Module):
    """Denoising action decoder: predicts clean action from noisy action.

    Uses FiLM conditioning for observation and timestep.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        cond_dim: int = 256,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps

        # Input: [obs, noisy_action]
        self.input_proj = nn.Linear(obs_dim + act_dim, hidden_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FiLM from timestep + observation
        self.film = nn.Linear(cond_dim + obs_dim, num_blocks * hidden_dim * 2)

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "linear1": nn.Linear(hidden_dim, hidden_dim * 4),
                "linear2": nn.Linear(hidden_dim * 4, hidden_dim),
            })
            for _ in range(num_blocks)
        ])

        # Output: predicted action noise
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(
        self,
        noisy_action: torch.Tensor,
        obs: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict action noise from (noisy_action, obs, timestep)."""
        h = self.input_proj(torch.cat([obs, noisy_action], dim=-1))

        # FiLM conditioning
        t_emb = self.time_embed(timestep)
        film_input = torch.cat([t_emb, obs], dim=-1)
        film_params = self.film(film_input).view(-1, len(self.blocks) * 2, self.hidden_dim)
        scales = film_params[:, 0::2]
        biases = film_params[:, 1::2]

        for i, block in enumerate(self.blocks):
            h_norm = block["norm"](h)
            h_mod = h_norm * (1 + scales[:, i]) + biases[:, i]
            gate = F.gelu(block["linear1"](h_mod))
            h = h + block["linear2"](gate)

        return self.output_head(h)


class CascadedWAM(nn.Module):
    """Cascaded World Action Model.

    Architecture:
    1. World Model: predicts next_state from (obs, action)
    2. Action Decoder: denoises action from (obs, noisy_action, timestep)

    Training:
    - World Model: MSE loss on predicted next_state
    - Action Decoder: MSE loss on predicted action noise

    Inference:
    1. Sample action from action decoder
    2. Predict next_state from world model
    3. Use predicted state for planning
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        wm_blocks: int = 4,
        ad_blocks: int = 4,
        cond_dim: int = 256,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.timesteps = timesteps

        # World Model
        self.world_model = WorldModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            num_blocks=wm_blocks,
        )

        # Action Decoder (with diffusion)
        self.action_decoder = ActionDecoder(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            num_blocks=ad_blocks,
            cond_dim=cond_dim,
            timesteps=timesteps,
        )

        # Diffusion schedule for action decoder
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
        """Add noise to x_start at timestep t."""
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
        """Compute joint loss for world model and action decoder.

        Returns:
            total_loss, wm_loss, ad_loss
        """
        B = obs.size(0)
        device = obs.device

        # World Model loss: predict next_state from (obs, action)
        pred_next_state = self.world_model(obs, action)
        wm_loss = F.mse_loss(pred_next_state, next_state)

        # Action Decoder loss: denoise action
        t = torch.randint(0, self.timesteps, (B,), device=device)
        action_noise = torch.randn_like(action)
        noisy_action = self.q_sample(action, t, action_noise)
        pred_action_noise = self.action_decoder(noisy_action, obs, t)
        ad_loss = F.mse_loss(pred_action_noise, action_noise)

        total_loss = wm_loss + ad_loss
        return total_loss, wm_loss, ad_loss

    @torch.no_grad()
    def denoise_action(
        self,
        obs: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Denoise action from observation (inference)."""
        B = obs.size(0)
        device = obs.device

        x = torch.randn(B, self.act_dim, device=device)

        for i in range(num_steps):
            t = torch.full((B,), i * self.timesteps // num_steps, device=device)
            pred_noise = self.action_decoder(x, obs, t)

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

    @torch.no_grad()
    def predict_next_state(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next_state using world model."""
        return self.world_model(obs, action)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "world_model": self.world_model.state_dict(),
            "action_decoder": self.action_decoder.state_dict(),
            "timesteps": self.timesteps,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.action_decoder.load_state_dict(ckpt["action_decoder"])
