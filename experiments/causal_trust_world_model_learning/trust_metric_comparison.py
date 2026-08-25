"""Trust Metric Comparison for WAMs.

Implements 5 trust scoring methods on the same RSSM backbone:
1. EMA Prediction Error (our current method)
2. Action-State Consistency (Future Compatible 2026)
3. Ensemble Disagreement (RWM-U 2026)
4. Feedback Correction (Feedback WM 2026)
5. Forward-Inverse Cycle (WAV 2026)

Compares how each trust metric performs for continual learning consolidation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
import numpy as np


class EMAPredictionTrust:
    """Trust = exp(-alpha * prediction_error). Our current method."""

    def __init__(self, alpha: float = 1.0, ema_alpha: float = 0.95):
        self.alpha = alpha
        self.ema_alpha = ema_alpha
        self.ema_error: dict[int, float] = {}

    def compute_trust(
        self, pred_obs: torch.Tensor, actual_obs: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        error = F.mse_loss(pred_obs, actual_obs, reduction="none").mean(dim=-1)
        error_np = error.detach().cpu().numpy()
        if task_id not in self.ema_error:
            self.ema_error[task_id] = float(error_np.mean())
        else:
            self.ema_error[task_id] = (
                self.ema_alpha * self.ema_error[task_id]
                + (1 - self.ema_alpha) * float(error_np.mean())
            )
        trust = torch.exp(-self.alpha * error / (self.ema_error[task_id] + 1e-8))
        return trust.clamp(0, 1)


class ActionStateConsistencyTrust:
    """Trust = cosine similarity between predicted and actual state transitions.
    Source: 'Is the Future Compatible?' (2026)
    """

    def __init__(self):
        pass

    def compute_trust(
        self, pred_obs: torch.Tensor, actual_obs: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        pred_delta = pred_obs
        actual_delta = actual_obs
        cos_sim = F.cosine_similarity(pred_delta, actual_delta, dim=-1)
        trust = (cos_sim + 1) / 2
        return trust.clamp(0, 1)


class EnsembleDisagreementTrust:
    """Trust = 1 - ensemble_variance.
    Source: RWM-U (2026)
    """

    def __init__(self, n_ensemble: int = 5, obs_dim: int = 64):
        self.n_ensemble = n_ensemble
        self.ensemble_heads = nn.ModuleList(
            [nn.Linear(obs_dim, obs_dim) for _ in range(n_ensemble)]
        )

    def compute_trust(
        self, features: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        predictions = torch.stack(
            [head(features) for head in self.ensemble_heads], dim=0
        )
        variance = predictions.var(dim=0).mean(dim=-1)
        trust = torch.exp(-variance)
        return trust.clamp(0, 1)


class FFDCTrust:
    """Trust via Future Forward Dynamics Causal Attention.
    Source: When to Trust Imagination (FFDC, 2026, arxiv 2605.06222)
    Jointly reasons over predicted actions, visual dynamics, real observations, language.
    """

    def __init__(self, obs_dim: int = 64, action_dim: int = 64, feature_dim: int = 0):
        if feature_dim <= 0:
            feature_dim = obs_dim
        input_dim = obs_dim * 2 + feature_dim + action_dim
        self.verifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def compute_trust(
        self,
        pred_obs: torch.Tensor,
        actual_obs: torch.Tensor,
        action: torch.Tensor,
        features: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        verifier_input = torch.cat([pred_obs, actual_obs, features, action], dim=-1)
        trust = self.verifier(verifier_input).squeeze(-1)
        return trust.clamp(0, 1)


class ForwardInverseCycleTrust:
    """Trust = cycle consistency between forward and inverse predictions.
    Source: World Action Verifier (2026)
    """

    def __init__(self, obs_dim: int = 64, action_dim: int = 2):
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )

    def compute_trust(
        self,
        pred_obs: torch.Tensor,
        actual_obs: torch.Tensor,
        action: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        pred_action = self.inverse_model(
            torch.cat([pred_obs, actual_obs], dim=-1)
        )
        action_error = F.mse_loss(pred_action, action, reduction="none").mean(dim=-1)
        obs_error = F.mse_loss(pred_obs, actual_obs, reduction="none").mean(dim=-1)
        cycle_error = obs_error + action_error
        trust = torch.exp(-cycle_error)
        return trust.clamp(0, 1)


class TrustMetricComparator:
    """Compares 5 trust metrics on the same RSSM backbone."""

    def __init__(self, obs_dim: int = 64, action_dim: int = 2):
        self.ema_trust = EMAPredictionTrust()
        self.asc_trust = ActionStateConsistencyTrust()
        self.ensemble_trust = EnsembleDisagreementTrust(obs_dim=obs_dim)
        self.ffdc_trust = FFDCTrust(obs_dim=obs_dim, action_dim=action_dim)
        self.cycle_trust = ForwardInverseCycleTrust(obs_dim=obs_dim, action_dim=action_dim)

    def compute_all_trusts(
        self,
        pred_obs: torch.Tensor,
        actual_obs: torch.Tensor,
        action: torch.Tensor,
        features: torch.Tensor,
        prev_pred: Optional[torch.Tensor],
        task_id: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "ema_pred_error": self.ema_trust.compute_trust(pred_obs, actual_obs, task_id),
            "action_state_consistency": self.asc_trust.compute_trust(pred_obs, actual_obs, task_id),
            "ensemble_disagreement": self.ensemble_trust.compute_trust(features, task_id),
            "ffdc_verifier": self.ffdc_trust.compute_trust(
                pred_obs, actual_obs, action, features, task_id
            ),
            "forward_inverse_cycle": self.cycle_trust.compute_trust(
                pred_obs, actual_obs, action, task_id
            ),
        }

    def get_parameters(self) -> list[nn.Parameter]:
        params = []
        params.extend(self.ensemble_trust.ensemble_heads.parameters())
        params.extend(self.ffdc_trust.verifier.parameters())
        params.extend(self.cycle_trust.inverse_model.parameters())
        return params
