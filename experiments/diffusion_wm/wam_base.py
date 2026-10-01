"""Diffusion machinery shared by the World Action Model variants."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NoiseFn = Callable[[Tensor, Tensor], Tensor]


def pad_to(x: Tensor, width: int) -> Tensor:
    """Right-pad or truncate the last dim to `width`, so one input layer serves targets of either size."""
    if x.size(-1) >= width:
        return x[..., :width]
    return F.pad(x, (0, width - x.size(-1)))


def film_blocks(hidden_dim: int, num_blocks: int) -> nn.ModuleList:
    """Pre-norm residual MLP blocks; module names match existing WAM checkpoints."""
    return nn.ModuleList(
        nn.ModuleDict({
            "norm": nn.LayerNorm(hidden_dim),
            "linear1": nn.Linear(hidden_dim, hidden_dim * 4),
            "linear2": nn.Linear(hidden_dim * 4, hidden_dim),
        })
        for _ in range(num_blocks)
    )


def apply_film_blocks(blocks: nn.ModuleList, h: Tensor, film: Tensor) -> Tensor:
    """Run `film_blocks`, interleaved scale/shift per block as produced by a `num_blocks * hidden * 2` linear."""
    params = film.view(-1, len(blocks) * 2, h.size(-1))
    scales, shifts = params[:, 0::2], params[:, 1::2]
    for i, block in enumerate(blocks):
        modulated = block["norm"](h) * (1 + scales[:, i]) + shifts[:, i]
        h = h + block["linear2"](F.gelu(block["linear1"](modulated)))
    return h


class BaseDiffusionWAM(nn.Module, ABC):
    """Linear-schedule noise-prediction training and strided x0-renoising sampling shared by the WAM variants.

    Buffer names match the checkpoints written before this base class existed.
    """

    def __init__(self, obs_dim: int, act_dim: int, timesteps: int = 1000, action_horizon: int = 1) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.timesteps = timesteps
        self.action_horizon = action_horizon
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return self.sqrt_alphas_cumprod[t, None] * x0 + self.sqrt_one_minus_alphas_cumprod[t, None] * noise

    def random_timesteps(self, like: Tensor) -> Tensor:
        return torch.randint(0, self.timesteps, (like.size(0),), device=like.device)

    def noise_loss(self, x0: Tensor, predict_noise: NoiseFn, t: Tensor) -> Tensor:
        noise = torch.randn_like(x0)
        return F.mse_loss(predict_noise(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(
        self,
        predict_noise: NoiseFn,
        shape: tuple[int, int],
        num_steps: int | None = None,
        clip: float | None = None,
    ) -> Tensor:
        """Strided sampling from t = T-1 down to 0: estimate x0, then re-noise it to the next sampled step.

        `clip` bounds each x0 estimate; use it only for targets that live in [-clip, clip].
        """
        n = min(num_steps or 100, self.timesteps)
        steps = torch.linspace(self.timesteps - 1, 0, n, device=self.betas.device).round().long()
        x = torch.randn(shape, device=self.betas.device)
        for i, t in enumerate(steps):
            alpha = self.alphas_cumprod[t]
            x0 = (x - (1 - alpha).sqrt() * predict_noise(x, t.expand(shape[0]))) / alpha.sqrt()
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            if i + 1 == n:
                return x0
            alpha_next = self.alphas_cumprod[steps[i + 1]]
            x = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * torch.randn_like(x)
        return x

    @abstractmethod
    def loss_terms(self, obs: Tensor, next_state: Tensor, action: Tensor) -> dict[str, Tensor]: ...

    @abstractmethod
    def predict_action(self, obs: Tensor, num_steps: int | None = None) -> Tensor: ...

    @abstractmethod
    def predict_next_state(self, state: Tensor, action: Tensor, num_steps: int | None = None) -> Tensor: ...

    def training_loss(self, obs: Tensor, next_state: Tensor, action: Tensor) -> tuple[Tensor, dict[str, float]]:
        terms = self.loss_terms(obs, next_state, action)
        total = torch.stack(list(terms.values())).sum()
        return total, {**{k: v.item() for k, v in terms.items()}, "total_loss": total.item()}

    @torch.no_grad()
    def predict_action_chunk(self, state: Tensor, horizon: int | None = None, num_steps: int | None = None) -> Tensor:
        actions = []
        for _ in range(horizon or self.action_horizon):
            action = self.predict_action(state, num_steps)
            actions.append(action)
            state = self.predict_next_state(state, action, num_steps)
        return torch.stack(actions, dim=1)
