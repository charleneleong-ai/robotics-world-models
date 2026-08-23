"""Prediction fidelity and divergence detection for diffusion world models.

Implements calibration of prediction intervals and detection of
divergence between predicted and real-world states.

Based on Aljalbout et al. 2026 — evaluation metrics for sim-to-real
transfer quality, including calibration and divergence detection.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import DiffusionDynamics


# ---------------------------------------------------------------------------
# Prediction Calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Result of prediction interval calibration."""

    coverage: float  # Fraction of observations within prediction intervals
    mean_interval_width: float  # Average width of prediction intervals
    calibration_error: float  # Gap between nominal and actual coverage

    def to_dict(self) -> dict[str, float]:
        return {
            "calibration/coverage": self.coverage,
            "calibration/mean_interval_width": self.mean_interval_width,
            "calibration/calibration_error": self.calibration_error,
        }


class PredictionCalibration:
    """Compute and evaluate prediction intervals for diffusion world model.

    Prediction intervals quantify uncertainty in predicted states.
    A well-calibrated model should have prediction intervals that
    contain the true next state with the expected frequency.
    """

    def __init__(
        self,
        model: DiffusionDynamics,
        num_samples: int = 100,
        alphas: list[float] | None = None,
    ) -> None:
        """Initialize calibration evaluator.

        Args:
            model: Diffusion world model.
            num_samples: Number of samples for uncertainty estimation.
            alphas: Confidence levels to evaluate (e.g., [0.1, 0.5, 0.9]).
        """
        self.model = model
        self.num_samples = num_samples
        self.alphas = alphas or [0.1, 0.3, 0.5, 0.7, 0.9]

    @torch.no_grad()
    def compute_prediction_intervals(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        alpha: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prediction intervals via Monte Carlo sampling.

        Args:
            obs: Current observation (B, obs_dim).
            action: Action taken (B, action_dim).
            alpha: Significance level (1 - alpha = confidence level).

        Returns:
            Tuple of (lower_bound, mean_prediction, upper_bound).
        """
        self.model.eval()
        device = obs.device

        # Sample multiple predictions from diffusion process
        predictions = []
        for _ in range(self.num_samples):
            # Use different noise for each sample
            noise = torch.randn_like(obs)
            pred = self.model(obs, action, noise=noise)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (N, B, state_dim)

        # Compute quantiles for prediction intervals
        lower = torch.quantile(predictions, alpha / 2, dim=0)
        upper = torch.quantile(predictions, 1 - alpha / 2, dim=0)
        mean_pred = predictions.mean(dim=0)

        return lower, mean_pred, upper

    def compute_calibration_error(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        alpha: float = 0.1,
    ) -> float:
        """Compute calibration error for a given confidence level.

        Calibration error measures how well the predicted intervals
        match the actual coverage.

        Args:
            obs: Current observations (B, obs_dim).
            action: Actions taken (B, action_dim).
            next_obs: Actual next observations (B, state_dim).
            alpha: Significance level.

        Returns:
            Calibration error (lower is better).
        """
        lower, mean_pred, upper = self.compute_prediction_intervals(
            obs, action, alpha
        )

        # Check if actual next_obs falls within interval
        within_interval = (next_obs >= lower) & (next_obs <= upper)
        coverage = within_interval.float().mean().item()

        # Calibration error = |actual coverage - nominal coverage|
        nominal_coverage = 1.0 - alpha
        return abs(coverage - nominal_coverage)

    def evaluate_all_confidence_levels(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> list[CalibrationResult]:
        """Evaluate calibration across all confidence levels.

        Args:
            obs: Current observations (B, obs_dim).
            action: Actions taken (B, action_dim).
            next_obs: Actual next observations (B, state_dim).

        Returns:
            List of CalibrationResult for each alpha.
        """
        results = []
        for alpha in self.alphas:
            lower, mean_pred, upper = self.compute_prediction_intervals(
                obs, action, alpha
            )

            # Coverage
            within = (next_obs >= lower) & (next_obs <= upper)
            coverage = within.float().mean().item()

            # Mean interval width
            interval_width = (upper - lower).abs().mean().item()

            # Calibration error
            nominal = 1.0 - alpha
            cal_error = abs(coverage - nominal)

            results.append(
                CalibrationResult(
                    coverage=coverage,
                    mean_interval_width=interval_width,
                    calibration_error=cal_error,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Divergence Detection
# ---------------------------------------------------------------------------


@dataclass
class DivergenceResult:
    """Result of divergence detection."""

    divergence_score: float  # Overall divergence score
    is_divergent: bool  # Whether divergence exceeds threshold
    per_dim_scores: torch.Tensor | None = None  # Per-dimension divergence

    def to_dict(self) -> dict[str, float]:
        result = {
            "divergence/score": self.divergence_score,
            "divergence/is_divergent": float(self.is_divergent),
        }
        return result


class DivergenceDetector:
    """Detect divergence between predicted and real-world states.

    When the world model's predictions diverge from reality, the
    trust score drops and the system should trigger recovery strategies
    (e.g., switching to classical control, requesting human intervention).
    """

    def __init__(
        self,
        ema_alpha: float = 0.95,
        threshold: float = 0.1,
        window_size: int = 100,
    ) -> None:
        """Initialize divergence detector.

        Args:
            ema_alpha: EMA decay for tracking divergence history.
            threshold: Divergence threshold for triggering alerts.
            window_size: Number of steps for rolling divergence window.
        """
        self.ema_alpha = ema_alpha
        self.threshold = threshold
        self.window_size = window_size
        self.ema_divergence: float | None = None
        self.divergence_history: list[float] = []

    def update(
        self,
        predicted_state: torch.Tensor,
        actual_state: torch.Tensor,
    ) -> DivergenceResult:
        """Update divergence detector with new prediction/observation pair.

        Args:
            predicted_state: Model's predicted next state.
            actual_state: Actual observed next state.

        Returns:
            DivergenceResult with current divergence status.
        """
        # Compute per-dimension divergence
        diff = (predicted_state - actual_state).abs()
        per_dim = diff / (actual_state.abs() + 1e-8)  # Normalized

        # Overall divergence score (mean normalized error)
        divergence_score = per_dim.mean().item()

        # Update EMA
        if self.ema_divergence is None:
            self.ema_divergence = divergence_score
        else:
            self.ema_divergence = (
                self.ema_alpha * self.ema_divergence
                + (1 - self.ema_alpha) * divergence_score
            )

        # Track history
        self.divergence_history.append(divergence_score)
        if len(self.divergence_history) > self.window_size:
            self.divergence_history.pop(0)

        # Check divergence
        is_divergent = self.ema_divergence is not None and self.ema_divergence > self.threshold

        return DivergenceResult(
            divergence_score=self.ema_divergence or divergence_score,
            is_divergent=is_divergent,
            per_dim_scores=per_dim,
        )

    def get_rolling_stats(self) -> dict[str, float]:
        """Get rolling statistics of divergence history."""
        if not self.divergence_history:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

        history = torch.tensor(self.divergence_history)
        return {
            "mean": history.mean().item(),
            "std": history.std().item(),
            "max": history.max().item(),
            "min": history.min().item(),
        }

    def reset(self) -> None:
        """Reset divergence detector state."""
        self.ema_divergence = None
        self.divergence_history.clear()


# ---------------------------------------------------------------------------
# Integration with ctwm trust scoring
# ---------------------------------------------------------------------------


def compute_trust_from_divergence(
    divergence_score: float,
    ema_alpha: float = 0.95,
    min_trust: float = 0.0,
    max_trust: float = 1.0,
) -> float:
    """Convert divergence score to trust score.

    Trust = 1 / (1 + divergence) mapped to [min_trust, max_trust].

    This aligns with the ctwm trust_scoring.py EMA-based trust
    computation but uses divergence as the input signal.

    Args:
        divergence_score: Normalized divergence score.
        ema_alpha: EMA decay (unused, kept for API compatibility).
        min_trust: Minimum trust value.
        max_trust: Maximum trust value.

    Returns:
        Trust score in [min_trust, max_trust].
    """
    raw_trust = 1.0 / (1.0 + divergence_score)
    # Normalize to [min_trust, max_trust]
    trust = min_trust + (max_trust - min_trust) * raw_trust
    return trust
