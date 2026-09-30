"""Base classes for World Action Models (WAM).

Shared functionality between DiffusionWAM and ScaledDiffusionWAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import cosine_beta_schedule


class WAMDenoiserBase(nn.Module):
    """Base class for WAM denoisers with shared functionality."""
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        num_blocks: int,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Diffusion constants
        betas = cosine_beta_schedule(1000)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance", 
                           betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def _q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise


class BaseDiffusionWAM(nn.Module):
    """Base class for World Action Models with shared inference logic."""
    
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
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.action_horizon = action_horizon
        
        # Diffusion constants (subclasses must register buffers)
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
        self.register_buffer("posterior_variance", 
                           betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def _q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def _denoise_target(
        self,
        state: torch.Tensor,
        target_type: str,
        target_dim: int,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Denoise a target (state or action) from observation."""
        B = state.size(0)
        T = num_steps or self.timesteps
        device = state.device
        
        x = torch.randn(B, target_dim, device=device)
        
        for t_idx in range(T):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.float)
            pred_noise = self.denoiser(x, state, target_type, t_batch)
            alpha = self.alphas_cumprod[t_idx]
            alpha_prev = self.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.ones_like(alpha)
            
            x0_pred = (x - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise
            else:
                x = x0_pred
        
        return x

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
        """Generate a chunk of actions autoregressively."""
        h = horizon or self.action_horizon
        B = state.size(0)
        actions = []
        s = state
        for _ in range(h):
            a = self.predict_action(s, num_steps)
            actions.append(a)
            s = self.predict_next_state(s, a, num_steps=num_steps)
        return torch.stack(actions, dim=1)

    @torch.no_grad()
    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Predict next state given current state and action (world model use)."""
        # Subclasses must implement this to use the action
        raise NotImplementedError("Subclasses must implement predict_next_state")


# Shared utilities
class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-torch.arange(half) * torch.log(torch.tensor(10000.0)) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] -> [B, dim]"""
        half = self.dim // 2
        emb = t[:, None] * self.freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# Export for backward compatibility
__all__ = [
    "WAMDenoiserBase",
    "BaseDiffusionWAM",
    "SinusoidalEmbedding",
]
