"""Residual dynamics model for learning the sim-to-real gap.

Learns a residual correction to the simulation dynamics,
compensating for unmodeled effects like friction, backlash,
and contact dynamics.

Based on Aljalbout et al. 2026 — residual dynamics models capture
what the simulator gets wrong, enabling more accurate prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import DiffusionDynamics


# ---------------------------------------------------------------------------
# Residual Dynamics Network
# ---------------------------------------------------------------------------


class ResidualDynamicsNet(nn.Module):
    """Neural network that predicts the residual between sim and real.

    The residual model learns: delta = real_next_obs - sim_next_obs
    given (obs, action) as input.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize residual dynamics network.

        Args:
            obs_dim: Observation dimensionality.
            action_dim: Action dimensionality.
            hidden_dim: Hidden layer dimensionality.
            num_layers: Number of hidden layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        input_dim = obs_dim + action_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])

        self.backbone = nn.Sequential(*layers)
        self.residual_head = nn.Linear(hidden_dim, obs_dim)

        # Uncertainty estimation head (predicts log-variance)
        self.uncertainty_head = nn.Linear(hidden_dim, obs_dim)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict residual dynamics.

        Args:
            obs: Current observation (B, obs_dim).
            action: Action taken (B, action_dim).

        Returns:
            Tuple of (residual_mean, residual_log_var).
        """
        x = torch.cat([obs, action], dim=-1)
        h = self.backbone(x)

        residual = self.residual_head(h)
        log_var = self.uncertainty_head(h)

        return residual, log_var


# ---------------------------------------------------------------------------
# Hybrid Dynamics Model
# ---------------------------------------------------------------------------


@dataclass
class HybridPrediction:
    """Prediction from hybrid (sim + residual) dynamics."""

    sim_prediction: torch.Tensor
    residual: torch.Tensor
    hybrid_prediction: torch.Tensor
    uncertainty: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "sim_prediction": self.sim_prediction,
            "residual": self.residual,
            "hybrid_prediction": self.hybrid_prediction,
            "uncertainty": self.uncertainty,
        }


class HybridDynamicsModel:
    """Combines simulation dynamics with learned residual correction.

    The hybrid model predicts: real_next_obs = sim_next_obs + residual
    where residual is learned from (sim_obs, action, real_obs) pairs.
    """

    def __init__(
        self,
        sim_model: DiffusionDynamics,
        residual_model: ResidualDynamicsNet,
        residual_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
    ) -> None:
        """Initialize hybrid dynamics model.

        Args:
            sim_model: Pre-trained diffusion dynamics model (sim).
            residual_model: Learned residual correction model.
            residual_weight: Weight for residual correction.
            uncertainty_weight: Weight for uncertainty penalty.
        """
        self.sim_model = sim_model
        self.residual_model = residual_model
        self.residual_weight = residual_weight
        self.uncertainty_weight = uncertainty_weight

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_denoise_steps: int = 100,
    ) -> HybridPrediction:
        """Predict next state using hybrid dynamics.

        Args:
            obs: Current observation (B, obs_dim).
            action: Action taken (B, action_dim).
            num_denoise_steps: Denoising steps for sim model.

        Returns:
            HybridPrediction with sim, residual, and hybrid predictions.
        """
        # Get simulation prediction
        num_steps = min(num_denoise_steps, self.sim_model.timesteps)
        sim_pred = self.sim_model.sample(obs, action, num_steps=num_steps)

        # Get residual correction
        residual, log_var = self.residual_model(obs, action)
        uncertainty = torch.exp(log_var)

        # Combine
        hybrid_pred = sim_pred + self.residual_weight * residual

        return HybridPrediction(
            sim_prediction=sim_pred,
            residual=residual,
            hybrid_prediction=hybrid_pred,
            uncertainty=uncertainty,
        )

    def compute_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        real_next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss for residual model.

        Args:
            obs: Current observations (B, obs_dim).
            action: Actions taken (B, action_dim).
            real_next_obs: Actual next observations (B, state_dim).

        Returns:
            Tuple of (total_loss, loss_components).
        """
        # Get simulation prediction (detached — we don't train the sim model)
        with torch.no_grad():
            num_steps = min(50, self.sim_model.timesteps)
            sim_pred = self.sim_model.sample(obs, action, num_steps=num_steps)

        # Compute residual
        residual, log_var = self.residual_model(obs, action)

        # Target residual
        target_residual = real_next_obs - sim_pred

        # NLL loss with uncertainty
        precision = torch.exp(-log_var)
        nll_loss = 0.5 * (precision * (target_residual - residual).pow(2) + log_var)

        # Weighted loss
        residual_loss = nll_loss.mean()
        uncertainty_penalty = self.uncertainty_weight * log_var.mean()

        total_loss = residual_loss + uncertainty_penalty

        loss_components = {
            "residual_loss": residual_loss.item(),
            "uncertainty_penalty": uncertainty_penalty.item(),
            "mean_uncertainty": torch.exp(log_var).mean().item(),
        }

        return total_loss, loss_components


# ---------------------------------------------------------------------------
# Online Adaptation
# ---------------------------------------------------------------------------


class OnlineResidualAdapter:
    """Online adaptation of residual model using real-world feedback.

    As the robot operates in the real world, this module continuously
    updates the residual model to improve accuracy.
    """

    def __init__(
        self,
        residual_model: ResidualDynamicsNet,
        learning_rate: float = 1e-4,
        buffer_size: int = 1000,
        batch_size: int = 32,
    ) -> None:
        """Initialize online adapter.

        Args:
            residual_model: Pre-trained residual model.
            learning_rate: Online learning rate.
            buffer_size: Size of experience buffer.
            batch_size: Batch size for online updates.
        """
        self.residual_model = residual_model
        self.optimizer = torch.optim.Adam(
            self.residual_model.parameters(), lr=learning_rate
        )
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        real_next_obs: torch.Tensor,
        sim_next_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Update residual model with new real-world transition.

        Args:
            obs: Current observation.
            action: Action taken.
            real_next_obs: Actual next observation.
            sim_next_obs: Sim's predicted next observation.

        Returns:
            Dictionary of training metrics.
        """
        # Add to buffer
        self.buffer.append({
            "obs": obs,
            "action": action,
            "real_next_obs": real_next_obs,
            "sim_next_obs": sim_next_obs,
        })

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Not enough data for update
        if len(self.buffer) < self.batch_size:
            return {"buffer_size": len(self.buffer)}

        # Sample batch
        import random
        batch = random.sample(self.buffer, self.batch_size)
        obs_batch = torch.stack([t["obs"] for t in batch])
        action_batch = torch.stack([t["action"] for t in batch])
        real_next_batch = torch.stack([t["real_next_obs"] for t in batch])
        sim_next_batch = torch.stack([t["sim_next_obs"] for t in batch])

        # Compute loss
        self.residual_model.train()
        self.optimizer.zero_grad()

        # Residual from sim to real
        target_residual = real_next_batch - sim_next_batch
        pred_residual, log_var = self.residual_model(obs_batch, action_batch)

        precision = torch.exp(-log_var)
        loss = 0.5 * (precision * (target_residual - pred_residual).pow(2) + log_var).mean()

        loss.backward()
        self.optimizer.step()

        self.residual_model.eval()

        return {
            "online_loss": loss.item(),
            "mean_abs_residual": target_residual.abs().mean().item(),
            "buffer_size": len(self.buffer),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_hybrid_model(
    obs_dim: int,
    action_dim: int,
    sim_model: DiffusionDynamics,
    residual_hidden_dim: int = 256,
    residual_weight: float = 1.0,
) -> HybridDynamicsModel:
    """Create a hybrid dynamics model.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        sim_model: Pre-trained simulation dynamics model.
        residual_hidden_dim: Hidden dim for residual model.
        residual_weight: Weight for residual correction.

    Returns:
        HybridDynamicsModel ready for training.
    """
    residual_model = ResidualDynamicsNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=residual_hidden_dim,
    )

    return HybridDynamicsModel(
        sim_model=sim_model,
        residual_model=residual_model,
        residual_weight=residual_weight,
    )
