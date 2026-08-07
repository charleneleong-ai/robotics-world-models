"""Conditional diffusion dynamics models for state-based world modelling.

Architectures:
    - MLPDenoiser: MLP-based conditional denoising network (primary)
    - DiTDenoiser: Small diffusion transformer (stretch)

Both predict the noise epsilon given (noisy_next_state, state, action, timestep).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------------------------------------------------------------------------
# Noise schedule (cosine — Nichol & Dhariwal 2021)
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0.0001, 0.02)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (same as DDPM / DiT)
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * torch.log(torch.tensor(10000.0)) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# MLP-based conditional denoiser
# ---------------------------------------------------------------------------

class MLPDenoiser(nn.Module):
    """Conditional MLP that predicts epsilon from (s_t, a_t, noisy_s_{t+1}, t).

    Conditioning (s_t, a_t) is concatenated with the noisy state and fed
    through residual MLP blocks with FiLM modulation from the timestep embedding.
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

        # Input projection: [s_t, a_t, noisy_s_{t+1}] -> hidden
        input_dim = obs_dim + act_dim + obs_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Timestep embedding -> FiLM scale/bias
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FiLM modulation: per-block scale + bias
        self.film = nn.Linear(cond_dim, num_blocks * hidden_dim * 2)

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "linear1": nn.Linear(hidden_dim, hidden_dim * 4),
                "linear2": nn.Linear(hidden_dim * 4, hidden_dim),
            })
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.num_blocks = num_blocks

    def forward(
        self,
        x_noisy: torch.Tensor,       # noisy s_{t+1}  [B, obs_dim]
        state: torch.Tensor,          # s_t            [B, obs_dim]
        action: torch.Tensor,         # a_t            [B, act_dim]
        timestep: torch.Tensor,       # t              [B]
    ) -> torch.Tensor:
        h = self.input_proj(torch.cat([state, action, x_noisy], dim=-1))

        t_emb = self.time_embed(timestep)
        film_params = self.film(t_emb)  # [B, num_blocks * hidden_dim * 2]
        film_params = film_params.view(-1, self.num_blocks * 2, self.hidden_dim)
        scales = film_params[:, 0::2]
        biases = film_params[:, 1::2]

        for i, block in enumerate(self.blocks):
            h_norm = block["norm"](h)
            scale = scales[:, i]
            bias = biases[:, i]
            h_mod = h_norm * (1 + scale) + bias
            gate = F.gelu(block["linear1"](h_mod))
            h = h + block["linear2"](gate)

        return self.output_proj(h)


# ---------------------------------------------------------------------------
# DDPM diffusion process
# ---------------------------------------------------------------------------

class DiffusionDynamics(nn.Module):
    """Action-conditioned diffusion dynamics model.

    Learns p(s_{t+1} | s_t, a_t) via DDPM denoising.
    """
    def __init__(self, denoiser: nn.Module, timesteps: int = 1000):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.obs_dim = denoiser.obs_dim

        # Precompute diffusion constants
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

    def forward(
        self,
        next_state: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss (simple MSE on epsilon)."""
        t = torch.randint(0, self.timesteps, (next_state.size(0),), device=next_state.device)
        noise = torch.randn_like(next_state)
        x_noisy = self._q_sample(next_state, t, noise)
        pred_noise = self.denoiser(x_noisy, state, action, t.float())
        return F.mse_loss(pred_noise, noise)

    def _q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Denoise from pure noise to predict s_{t+1} conditioned on (state, action)."""
        n = state.size(0)
        T = num_steps or self.timesteps
        device = state.device

        x = torch.randn(n, self.obs_dim, device=device)

        for t in reversed(range(T)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.float)
            pred_noise = self.denoiser(x, state, action, t_batch)
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            coef1 = 1 / alpha.sqrt()
            coef2 = (1 - alpha) / (1 - alpha_cumprod).sqrt()
            x = coef1 * (x - coef2 * pred_noise)
            if t > 0:
                noise = torch.randn_like(x)
                x += beta.sqrt() * noise

        return x

    @torch.no_grad()
    def denoise_with_progress(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_steps: int | None = None,
        milestones: tuple[int, ...] | None = None,
    ) -> list[torch.Tensor]:
        """Denoise capturing clean-state estimates at milestone timesteps.

        Returns x_0 estimates (via the model's epsilon prediction) at each
        milestone, in reverse-process order (noisiest first, final last).
        """
        n = state.size(0)
        T = num_steps or self.timesteps
        milestones = milestones or (3 * T // 4, T // 2, T // 4, 0)
        device = state.device

        x = torch.randn(n, self.obs_dim, device=device)
        estimates: dict[int, torch.Tensor] = {}

        for t in reversed(range(T)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.float)
            pred_noise = self.denoiser(x, state, action, t_batch)
            if t in milestones:
                sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt()
                x0_hat = (x - sqrt_one_minus * pred_noise) / self.sqrt_alphas_cumprod[t]
                estimates[t] = x0_hat
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            coef1 = 1 / alpha.sqrt()
            coef2 = (1 - alpha) / (1 - alpha_cumprod).sqrt()
            x = coef1 * (x - coef2 * pred_noise)
            if t > 0:
                x += beta.sqrt() * torch.randn_like(x)

        return [estimates[t] for t in milestones]

    @torch.no_grad()
    def rollout(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        horizon: int = 20,
        num_denoise_steps: int = 100,
    ) -> torch.Tensor:
        """Rollout: predict future states autoregressively.

        Args:
            states: initial states [B, obs_dim]
            actions: sequence of actions [B, T, act_dim] (T >= horizon)
            horizon: number of steps to rollout
        Returns:
            predicted_states [B, horizon, obs_dim]
        """
        B = states.size(0)
        preds = [states]
        s = states
        for h in range(horizon):
            a = actions[:, h]
            s_pred = self.sample(s, a, num_steps=num_denoise_steps)
            preds.append(s_pred)
            s = s_pred
        return torch.stack(preds, dim=1)  # [B, horizon+1, obs_dim]
