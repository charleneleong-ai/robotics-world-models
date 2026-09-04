"""Trust scoring methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TrustScorer(ABC):
    """Base class for trust scorers."""

    @abstractmethod
    def compute_trust(self, pe: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute trust scores from prediction errors.

        Args:
            pe: Prediction errors (B,) or (B, T).
        Returns:
            Trust scores (B,) in [0, 1]. Higher = more trustworthy.
        """
        ...


class NoTrust(TrustScorer):
    """No trust scoring (control)."""

    def compute_trust(self, pe: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.ones(pe.shape[0], device=pe.device)


class EMATrust(TrustScorer):
    """Exponential moving average trust."""

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self._ema: float | None = None

    def compute_trust(self, pe: torch.Tensor, **kwargs) -> torch.Tensor:
        mean_pe = pe.mean().item()
        if self._ema is None:
            self._ema = mean_pe
        else:
            self._ema = self.alpha * self._ema + (1 - self.alpha) * mean_pe
        trust = torch.exp(-pe / (self._ema + 1e-8))
        return trust.clamp(0, 1)

    def reset(self) -> None:
        self._ema = None


class MultiStepTrust(TrustScorer):
    """Adaptive multi-step rollout verification."""

    def __init__(self, max_k: int = 5, threshold: float = 0.1):
        self.max_k = max_k
        self.threshold = threshold

    def compute_trust(self, pe: torch.Tensor, **kwargs) -> torch.Tensor:
        B = pe.shape[0]
        trust = torch.ones(B, device=pe.device)
        for k in range(1, self.max_k + 1):
            mask = pe < self.threshold * k
            trust[mask] = trust[mask] * 0.9
        return trust


class EnsembleTrust(nn.Module, TrustScorer):
    """Ensemble disagreement trust."""

    def __init__(self, obs_dim: int, act_dim: int, n_models: int = 3, hidden: int = 64):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, obs_dim),
            )
            for _ in range(n_models)
        ])

    def compute_trust(self, pe: torch.Tensor, **kwargs) -> torch.Tensor:
        obs = kwargs.get("obs")
        act = kwargs.get("act")
        if obs is None or act is None:
            return torch.ones(pe.shape[0], device=pe.device)
        inp = torch.cat([obs, act], dim=-1)
        preds = torch.stack([m(inp) for m in self.models], dim=0)
        disagreement = preds.var(dim=0).mean(dim=-1)
        trust = torch.exp(-disagreement)
        return trust.clamp(0, 1)


class FFDCScorer(nn.Module, TrustScorer):
    """Future Forward Dynamics Causal attention."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.verifier = nn.Sequential(
            nn.Linear(obs_dim * 2 + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def compute_trust(self, pe: torch.Tensor, **kwargs) -> torch.Tensor:
        obs = kwargs.get("obs")
        act = kwargs.get("act")
        next_obs = kwargs.get("next_obs")
        if obs is None or act is None or next_obs is None:
            return torch.ones(pe.shape[0], device=pe.device)
        inp = torch.cat([obs, act, next_obs], dim=-1)
        return self.verifier(inp).squeeze(-1)


TRUST_METHODS: dict[str, type[TrustScorer]] = {
    "none": NoTrust,
    "ema": EMATrust,
    "multi_step": MultiStepTrust,
    "ensemble": EnsembleTrust,
    "ffdc": FFDCScorer,
}


def make_trust(name: str, obs_dim: int = 0, act_dim: int = 0, **kwargs) -> TrustScorer:
    if name not in TRUST_METHODS:
        raise ValueError(f"Unknown trust method: {name}. Available: {list(TRUST_METHODS.keys())}")
    cls = TRUST_METHODS[name]
    if name in ("ensemble", "ffdc"):
        return cls(obs_dim=obs_dim, act_dim=act_dim, **kwargs)
    return cls(**kwargs)
