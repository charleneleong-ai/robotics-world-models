"""System identification for calibrating simulation parameters.

Uses real-world data to estimate and calibrate simulation parameters,
bridging the reality gap by aligning sim with real dynamics.

Based on Aljalbout et al. 2026 — system identification reduces the
reality gap by tuning physics parameters to match real-world behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Parameter Estimation Network
# ---------------------------------------------------------------------------


class ParameterEstimator(nn.Module):
    """Neural network that estimates physics parameters from trajectory data.

    Given a short trajectory of (obs, action, next_obs) transitions,
    this network predicts the physics parameters (friction, mass, etc.)
    that best explain the observed dynamics.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_params: int = 6,
    ) -> None:
        """Initialize parameter estimator.

        Args:
            obs_dim: Observation dimensionality.
            action_dim: Action dimensionality.
            hidden_dim: Hidden layer dimensionality.
            num_params: Number of physics parameters to estimate.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Encode trajectory transitions
        transition_dim = obs_dim + action_dim + obs_dim  # obs + action + next_obs
        self.encoder = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Aggregate across time steps (attention-based)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Parameter prediction heads
        self.friction_head = nn.Linear(hidden_dim, 1)
        self.mass_head = nn.Linear(hidden_dim, 1)
        self.damping_head = nn.Linear(hidden_dim, 1)
        self.stiffness_head = nn.Linear(hidden_dim, 1)
        self.contact_stiffness_head = nn.Linear(hidden_dim, 1)
        self.contact_damping_head = nn.Linear(hidden_dim, 1)

        self.num_params = num_params

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        next_obs_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Estimate physics parameters from a trajectory.

        Args:
            obs_seq: Observation sequence (B, T, obs_dim).
            action_seq: Action sequence (B, T, action_dim).
            next_obs_seq: Next observation sequence (B, T, obs_dim).

        Returns:
            Dictionary of estimated parameter scales (multipliers).
        """
        B, T = obs_seq.shape[:2]

        # Encode each transition
        transitions = torch.cat([obs_seq, action_seq, next_obs_seq], dim=-1)
        h = self.encoder(transitions)  # (B, T, hidden_dim)

        # Self-attention to aggregate temporal information
        h_att, _ = self.attention(h, h, h)
        h = self.norm(h + h_att)

        # Global pooling
        h = h.mean(dim=1)  # (B, hidden_dim)

        # Predict parameter scales (multipliers around 1.0)
        friction = torch.sigmoid(self.friction_head(h)) * 2.0 + 0.5  # [0.5, 2.5]
        mass = torch.sigmoid(self.mass_head(h)) * 0.4 + 0.8  # [0.8, 1.2]
        damping = torch.sigmoid(self.damping_head(h)) * 1.0 + 0.5  # [0.5, 1.5]
        stiffness = torch.sigmoid(self.stiffness_head(h)) * 0.4 + 0.8  # [0.8, 1.2]
        contact_stiffness = torch.sigmoid(self.contact_stiffness_head(h)) * 2.0 + 0.5
        contact_damping = torch.sigmoid(self.contact_damping_head(h)) * 2.0 + 0.5

        return {
            "friction": friction.squeeze(-1),
            "mass": mass.squeeze(-1),
            "damping": damping.squeeze(-1),
            "stiffness": stiffness.squeeze(-1),
            "contact_stiffness": contact_stiffness.squeeze(-1),
            "contact_damping": contact_damping.squeeze(-1),
        }


# ---------------------------------------------------------------------------
# System Identification Loop
# ---------------------------------------------------------------------------


@dataclass
class SystemIdentificationResult:
    """Result of system identification."""

    estimated_params: dict[str, float]
    calibration_loss: float
    iterations: int
    converged: bool

    def to_dict(self) -> dict[str, float]:
        result = {f"sysid/{k}": v for k, v in self.estimated_params.items()}
        result["sysid/calibration_loss"] = self.calibration_loss
        result["sysid/iterations"] = float(self.iterations)
        result["sysid/converged"] = float(self.converged)
        return result


class SystemIdentifier:
    """System identification: calibrate sim parameters to match real data.

    Uses real-world trajectory data to estimate physics parameters
    that minimize the discrepancy between simulated and real dynamics.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
    ) -> None:
        """Initialize system identifier.

        Args:
            obs_dim: Observation dimensionality.
            action_dim: Action dimensionality.
            learning_rate: Learning rate for parameter optimization.
            max_iterations: Maximum optimization iterations.
            convergence_threshold: Loss change threshold for convergence.
        """
        self.estimator = ParameterEstimator(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr=learning_rate)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def calibrate(
        self,
        real_obs: torch.Tensor,
        real_actions: torch.Tensor,
        real_next_obs: torch.Tensor,
        sim_env: Any | None = None,
        seq_len: int = 10,
    ) -> SystemIdentificationResult:
        """Calibrate simulation parameters using real-world data.

        Args:
            real_obs: Real observations (B, obs_dim) or (B, T, obs_dim).
            real_actions: Real actions (B, action_dim) or (B, T, action_dim).
            real_next_obs: Real next observations (B, obs_dim) or (B, T, obs_dim).
            sim_env: Optional simulation environment to apply parameters.
            seq_len: Sequence length if input is flat (B, D).

        Returns:
            SystemIdentificationResult with estimated parameters.
        """
        # Reshape flat (B, D) to (B, T, D) if needed
        if real_obs.dim() == 2:
            B = real_obs.shape[0]
            T = min(seq_len, B)
            real_obs = real_obs[:B - B % T].reshape(-1, T, real_obs.shape[-1]) if B >= T else real_obs.unsqueeze(1)
            real_actions = real_actions[:B - B % T].reshape(-1, T, real_actions.shape[-1]) if B >= T else real_actions.unsqueeze(1)
            real_next_obs = real_next_obs[:B - B % T].reshape(-1, T, real_next_obs.shape[-1]) if B >= T else real_next_obs.unsqueeze(1)

        self.estimator.train()
        best_loss = float("inf")
        best_params = {}
        converged = False
        prev_loss = float("inf")

        for iteration in range(self.max_iterations):
            self.optimizer.zero_grad()

            # Estimate parameters
            params = self.estimator(real_obs, real_actions, real_next_obs)

            # Compute dynamics prediction error
            # (In practice, this would run the sim with estimated params
            # and compare to real data)
            loss = self._compute_dynamics_loss(
                params, real_obs, real_actions, real_next_obs
            )

            loss.backward()
            self.optimizer.step()

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.mean().item() for k, v in params.items()}

            # Check convergence
            if iteration > 0 and abs(prev_loss - loss.item()) < self.convergence_threshold:
                converged = True
                break

            prev_loss = loss.item()

        self.estimator.eval()

        return SystemIdentificationResult(
            estimated_params=best_params,
            calibration_loss=best_loss,
            iterations=iteration + 1,
            converged=converged,
        )

    def _compute_dynamics_loss(
        self,
        params: dict[str, torch.Tensor],
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamics prediction loss.

        This is a placeholder — in practice, this would run the
        simulation with estimated parameters and compare predictions
        to real data.
        """
        # Simplified loss: encourage parameters to be reasonable
        # and minimize prediction error
        # Flatten to (B, D) for loss computation
        if obs.dim() == 3:
            obs_flat = obs.reshape(-1, obs.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
            next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1])
        else:
            obs_flat = obs
            actions_flat = actions
            next_obs_flat = next_obs

        # Prediction error (placeholder — would use actual sim)
        # Scale prediction by estimated friction as a simple dynamics model
        friction = params.get("friction", torch.ones(1)).mean()
        pred_next_obs = obs_flat * friction.unsqueeze(0) if friction.dim() > 0 else obs_flat * friction
        dynamics_loss = F.mse_loss(pred_next_obs, next_obs_flat)

        # Regularization to keep parameters near 1.0
        reg_loss = sum((v - 1.0).pow(2).mean() for v in params.values())

        return dynamics_loss + 0.1 * reg_loss

    def apply_to_sim(
        self,
        sim_env: Any,
        params: dict[str, float],
    ) -> None:
        """Apply estimated parameters to a simulation environment.

        Args:
            sim_env: ManiSkill3 environment.
            params: Dictionary of parameter scales.
        """
        try:
            unwrapped = sim_env.unwrapped if hasattr(sim_env, "unwrapped") else sim_env

            if "friction" in params and hasattr(unwrapped, "friction"):
                unwrapped.friction *= params["friction"]

            if "mass" in params and hasattr(unwrapped, "link_mass"):
                unwrapped.link_mass *= params["mass"]

            if "damping" in params and hasattr(unwrapped, "damping"):
                unwrapped.damping *= params["damping"]

            if "stiffness" in params and hasattr(unwrapped, "stiffness"):
                unwrapped.stiffness *= params["stiffness"]

        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------


def collect_real_trajectory(
    env: Any,
    policy: Any,
    num_steps: int = 100,
) -> dict[str, torch.Tensor]:
    """Collect a trajectory from the real world (or a reference policy).

    Args:
        env: Environment (real or high-fidelity sim).
        policy: Policy to collect data with.
        num_steps: Number of steps to collect.

    Returns:
        Dictionary with obs, action, next_obs tensors.
    """
    obs, _ = env.reset()
    obs_list = [obs]
    action_list = []
    next_obs_list = []

    for _ in range(num_steps):
        action = policy(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)

        obs_list.append(next_obs)
        action_list.append(action)
        next_obs_list.append(next_obs)

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "obs": torch.tensor(obs_list[:-1], dtype=torch.float32),
        "actions": torch.tensor(action_list, dtype=torch.float32),
        "next_obs": torch.tensor(next_obs_list, dtype=torch.float32),
    }


def collect_sim_trajectory(
    env: Any,
    policy: Any,
    num_steps: int = 100,
    apply_randomization: bool = False,
) -> dict[str, torch.Tensor]:
    """Collect a trajectory from simulation.

    Args:
        env: Simulation environment.
        policy: Policy to collect data with.
        num_steps: Number of steps to collect.
        apply_randomization: Whether to apply domain randomization.

    Returns:
        Dictionary with obs, action, next_obs tensors.
    """
    from .domain_rand import apply_observation_noise, apply_action_noise, DomainRandomizationConfig

    config = DomainRandomizationConfig()
    obs, _ = env.reset()
    obs_list = [obs]
    action_list = []
    next_obs_list = []

    for _ in range(num_steps):
        action = policy(obs)

        if apply_randomization:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            obs_tensor = apply_observation_noise(obs_tensor, config)
            action_tensor = apply_action_noise(action_tensor, config)
            action = action_tensor.numpy()

        next_obs, _, terminated, truncated, _ = env.step(action)

        obs_list.append(next_obs)
        action_list.append(action)
        next_obs_list.append(next_obs)

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "obs": torch.tensor(obs_list[:-1], dtype=torch.float32),
        "actions": torch.tensor(action_list, dtype=torch.float32),
        "next_obs": torch.tensor(next_obs_list, dtype=torch.float32),
    }
