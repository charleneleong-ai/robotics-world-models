"""Sim-to-real transfer pipeline.

Integrates domain randomization, system identification, residual dynamics,
and evaluation into a complete transfer workflow.

Based on Aljalbout et al. 2026 — the key techniques for sim-to-real transfer
are domain randomization, system identification, and residual dynamics.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any

from .domain_rand import (
    DomainRandomizationConfig,
    apply_physics_randomization,
    apply_observation_noise,
    apply_action_noise,
)
from .model import DiffusionDynamics
from .system_id import SystemIdentifier, SystemIdentificationResult
from .residual_dynamics import (
    HybridDynamicsModel,
    OnlineResidualAdapter,
    ResidualDynamicsNet,
    create_hybrid_model,
)
from .fidelity import DivergenceDetector, PredictionCalibration


# ---------------------------------------------------------------------------
# Transfer Pipeline
# ---------------------------------------------------------------------------


@dataclass
class TransferConfig:
    """Configuration for sim-to-real transfer pipeline."""

    # Domain randomization
    domain_rand: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )

    # System identification
    sysid_iterations: int = 100
    sysid_lr: float = 1e-3

    # Residual dynamics
    residual_hidden_dim: int = 256
    residual_weight: float = 1.0

    # Online adaptation
    online_lr: float = 1e-4
    online_buffer_size: int = 1000

    # Evaluation
    num_eval_episodes: int = 10
    divergence_threshold: float = 0.1


@dataclass
class TransferResult:
    """Result of sim-to-real transfer pipeline."""

    sysid_result: SystemIdentificationResult
    residual_metrics: dict[str, float]
    eval_metrics: dict[str, float]
    trust_scores: list[float]
    divergence_scores: list[float]

    def to_dict(self) -> dict[str, float]:
        result = {}
        result.update(self.sysid_result.to_dict())
        result.update({f"residual/{k}": v for k, v in self.residual_metrics.items()})
        result.update({f"eval/{k}": v for k, v in self.eval_metrics.items()})
        if self.trust_scores:
            result["transfer/mean_trust"] = sum(self.trust_scores) / len(self.trust_scores)
        if self.divergence_scores:
            result["transfer/mean_divergence"] = (
                sum(self.divergence_scores) / len(self.divergence_scores)
            )
        return result


class SimToRealPipeline:
    """Complete sim-to-real transfer pipeline.

    Steps:
        1. Domain randomization during sim data collection
        2. System identification to calibrate sim parameters
        3. Train residual dynamics model
        4. Online adaptation during real-world deployment
        5. Continuous evaluation and trust scoring
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: TransferConfig | None = None,
    ) -> None:
        """Initialize transfer pipeline.

        Args:
            obs_dim: Observation dimensionality.
            action_dim: Action dimensionality.
            config: Transfer configuration.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or TransferConfig()

        # Components
        self.sys_identifier = SystemIdentifier(
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=self.config.sysid_lr,
            max_iterations=self.config.sysid_iterations,
        )

        self.divergence_detector = DivergenceDetector(
            threshold=self.config.divergence_threshold,
        )

        self.calibration: PredictionCalibration | None = None
        self.residual_adapter: OnlineResidualAdapter | None = None

    def step1_domain_randomization(
        self,
        sim_env: Any,
        num_episodes: int = 100,
    ) -> dict[str, Any]:
        """Step 1: Collect training data with domain randomization.

        Args:
            sim_env: Simulation environment.
            num_episodes: Number of episodes to collect.

        Returns:
            Dictionary with collected trajectories and parameters.
        """
        from .domain_rand import randomize_and_collect

        all_transitions = []
        all_params = []

        for _ in range(num_episodes):
            result = randomize_and_collect(
                sim_env,
                self.config.domain_rand,
                num_steps=100,
            )
            all_transitions.extend(result["transitions"])
            all_params.append(result["sampled_params"])

        return {
            "transitions": all_transitions,
            "params": all_params,
            "num_episodes": num_episodes,
        }

    def step2_system_identification(
        self,
        real_data: dict[str, torch.Tensor],
    ) -> SystemIdentificationResult:
        """Step 2: Calibrate simulation parameters using real data.

        Args:
            real_data: Dictionary with obs, actions, next_obs tensors.

        Returns:
            SystemIdentificationResult with estimated parameters.
        """
        result = self.sys_identifier.calibrate(
            real_obs=real_data["obs"],
            real_actions=real_data["actions"],
            real_next_obs=real_data["next_obs"],
        )

        return result

    def step3_train_residual(
        self,
        sim_model: DiffusionDynamics,
        real_data: dict[str, torch.Tensor],
        epochs: int = 50,
    ) -> HybridDynamicsModel:
        """Step 3: Train residual dynamics model.

        Args:
            sim_model: Pre-trained simulation diffusion model.
            real_data: Real-world trajectory data.
            epochs: Training epochs.

        Returns:
            Trained HybridDynamicsModel.
        """
        hybrid_model = create_hybrid_model(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            sim_model=sim_model,
            residual_hidden_dim=self.config.residual_hidden_dim,
            residual_weight=self.config.residual_weight,
        )

        # Train residual model
        optimizer = torch.optim.Adam(
            hybrid_model.residual_model.parameters(), lr=1e-3
        )

        obs = real_data["obs"]
        actions = real_data["actions"]
        next_obs = real_data["next_obs"]

        hybrid_model.residual_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, components = hybrid_model.compute_loss(obs, actions, next_obs)
            loss.backward()
            optimizer.step()

        hybrid_model.residual_model.eval()

        # Setup online adapter
        self.residual_adapter = OnlineResidualAdapter(
            residual_model=hybrid_model.residual_model,
            learning_rate=self.config.online_lr,
            buffer_size=self.config.online_buffer_size,
        )

        return hybrid_model

    def step4_online_adaptation(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        real_next_obs: torch.Tensor,
        sim_next_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Step 4: Online adaptation with real-world feedback.

        Args:
            obs: Current observation.
            action: Action taken.
            real_next_obs: Actual next observation.
            sim_next_obs: Sim's predicted next observation.

        Returns:
            Adaptation metrics.
        """
        if self.residual_adapter is None:
            return {"status": "no_adapter"}

        return self.residual_adapter.update(obs, action, real_next_obs, sim_next_obs)

    def step5_evaluate(
        self,
        hybrid_model: HybridDynamicsModel,
        real_data: dict[str, torch.Tensor],
        num_steps: int = 100,
    ) -> TransferResult:
        """Step 5: Evaluate transfer quality.

        Args:
            hybrid_model: Trained hybrid dynamics model.
            real_data: Real-world evaluation data.
            num_steps: Number of steps to evaluate.

        Returns:
            TransferResult with comprehensive evaluation metrics.
        """
        obs = real_data["obs"]
        actions = real_data["actions"]
        next_obs = real_data["next_obs"]

        # Evaluate predictions
        with torch.no_grad():
            predictions = hybrid_model.predict(obs, actions)

        # Compute metrics
        pred_error = (predictions.hybrid_prediction - next_obs).pow(2).mean().item()
        sim_error = (predictions.sim_prediction - next_obs).pow(2).mean().item()
        residual_magnitude = predictions.residual.abs().mean().item()

        # Divergence detection
        divergence_results = []
        trust_scores = []
        for t in range(min(obs.shape[0], num_steps)):
            div_result = self.divergence_detector.update(
                predictions.hybrid_prediction[t], next_obs[t]
            )
            divergence_results.append(div_result.divergence_score)
            trust_scores.append(1.0 / (1.0 + div_result.divergence_score))

        eval_metrics = {
            "hybrid_mse": pred_error,
            "sim_mse": sim_error,
            "improvement": sim_error - pred_error,
            "residual_magnitude": residual_magnitude,
            "mean_uncertainty": predictions.uncertainty.mean().item(),
        }

        return TransferResult(
            sysid_result=SystemIdentificationResult(
                estimated_params={},
                calibration_loss=0.0,
                iterations=0,
                converged=True,
            ),
            residual_metrics={"residual_magnitude": residual_magnitude},
            eval_metrics=eval_metrics,
            trust_scores=trust_scores,
            divergence_scores=divergence_results,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def run_full_transfer(
    sim_env: Any,
    real_data: dict[str, torch.Tensor],
    sim_model: DiffusionDynamics,
    obs_dim: int,
    action_dim: int,
    config: TransferConfig | None = None,
) -> TransferResult:
    """Run the complete sim-to-real transfer pipeline.

    Args:
        sim_env: Simulation environment.
        real_data: Real-world trajectory data.
        sim_model: Pre-trained simulation diffusion model.
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        config: Transfer configuration.

    Returns:
        TransferResult with all evaluation metrics.
    """
    pipeline = SimToRealPipeline(obs_dim, action_dim, config)

    # Step 2: System identification
    sysid_result = pipeline.step2_system_identification(real_data)

    # Step 3: Train residual
    hybrid_model = pipeline.step3_train_residual(sim_model, real_data)

    # Step 5: Evaluate
    result = pipeline.step5_evaluate(hybrid_model, real_data)
    result.sysid_result = sysid_result

    return result
