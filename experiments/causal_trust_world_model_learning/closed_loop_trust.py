"""Closed-Loop Trust Correction for WAMs.

Combines FFDC + Feedback Correction + Conformal Calibration:
1. After each action, observe true next state
2. Compute actual prediction error
3. Update trust estimate with real error signal
4. Correct future predictions using feedback
5. Calibrate thresholds via conformal prediction

This is a closed-loop system that continuously self-corrects
using real observation errors, not just open-loop prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
import numpy as np


class ClosedLoopTrustCorrector:
    """Trust scoring using real observation errors in closed-loop.

    After each action execution:
    1. Observe true next state o_{t+1}
    2. Compare with predicted next state ô_{t+1}
    3. Compute error e_t = ||o_{t+1} - ô_{t+1}||
    4. Update trust: τ_t = f(e_t, history, calibration)
    5. Use trust to decide: continue / replan / consolidate
    """

    def __init__(
        self,
        obs_dim: int = 64,
        ema_alpha: float = 0.9,
        history_len: int = 50,
        conformal_alpha: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.ema_alpha = ema_alpha
        self.history_len = history_len
        self.conformal_alpha = conformal_alpha

        # Error history per task
        self.error_history: dict[int, deque] = {}
        self.ema_error: dict[int, float] = {}
        self.ema_squared_error: dict[int, float] = {}

        # Conformal calibration
        self.calibration_scores: dict[int, list[float]] = {}
        self.calibrated_thresholds: dict[int, float] = {}

        # Feedback state
        self.feedback_state: dict[int, Optional[torch.Tensor]] = {}
        self.prediction_correction: dict[int, nn.Module] = {}

    def observe_and_update(
        self,
        predicted_obs: torch.Tensor,
        actual_obs: torch.Tensor,
        action: torch.Tensor,
        task_id: int,
    ) -> dict[str, torch.Tensor]:
        """Process one closed-loop step.

        Args:
            predicted_obs: what the world model predicted
            actual_obs: what actually happened
            action: action that was executed
            task_id: current task

        Returns:
            Dictionary with trust signals
        """
        # 1. Compute raw prediction error
        raw_error = F.mse_loss(predicted_obs, actual_obs, reduction="none").mean(dim=-1)

        # 2. Update EMA error statistics
        error_val = float(raw_error.mean())
        if task_id not in self.ema_error:
            self.ema_error[task_id] = error_val
            self.ema_squared_error[task_id] = error_val ** 2
        else:
            self.ema_error[task_id] = (
                self.ema_alpha * self.ema_error[task_id]
                + (1 - self.ema_alpha) * error_val
            )
            self.ema_squared_error[task_id] = (
                self.ema_alpha * self.ema_squared_error[task_id]
                + (1 - self.ema_alpha) * error_val ** 2
            )

        # 3. Store error history
        if task_id not in self.error_history:
            self.error_history[task_id] = deque(maxlen=self.history_len)
        self.error_history[task_id].append(error_val)

        # 4. Compute normalized error (how bad relative to this task's average)
        normalized_error = raw_error / (self.ema_error[task_id] + 1e-8)

        # 5. Compute trust from normalized error
        trust_from_error = torch.exp(-normalized_error)

        # 6. Compute error trend (is error increasing?)
        if len(self.error_history[task_id]) >= 5:
            recent = list(self.error_history[task_id])[-5:]
            older = list(self.error_history[task_id])[-10:-5] if len(self.error_history[task_id]) >= 10 else recent
            trend = np.mean(recent) - np.mean(older)
            trend_signal = torch.tensor(1.0 if trend < 0 else 0.5 if abs(trend) < 0.01 else 0.2)
        else:
            trend_signal = torch.tensor(0.5)

        # 7. Compute feedback-corrected prediction
        if task_id in self.feedback_state and self.feedback_state[task_id] is not None:
            feedback_weight = torch.sigmoid(self.feedback_state[task_id])
            corrected_pred = predicted_obs * (1 + feedback_weight)
            corrected_error = F.mse_loss(corrected_pred, actual_obs, reduction="none").mean(dim=-1)
            trust_from_feedback = torch.exp(-corrected_error / (self.ema_error[task_id] + 1e-8))
        else:
            corrected_pred = predicted_obs
            trust_from_feedback = trust_from_error

        # 8. Update feedback state
        feedback_signal = (actual_obs - predicted_obs).mean(dim=-1, keepdim=True)
        if task_id not in self.feedback_state:
            self.feedback_state[task_id] = feedback_signal.mean(dim=0)
        else:
            self.feedback_state[task_id] = (
                self.ema_alpha * self.feedback_state[task_id]
                + (1 - self.ema_alpha) * feedback_signal.mean(dim=0)
            )

        # 9. Conformal calibration
        if task_id not in self.calibration_scores:
            self.calibration_scores[task_id] = []
        self.calibration_scores[task_id].append(float(raw_error.mean()))

        if len(self.calibration_scores[task_id]) >= 20:
            scores = np.array(self.calibration_scores[task_id])
            threshold = np.percentile(scores, 100 * (1 - self.conformal_alpha))
            self.calibrated_thresholds[task_id] = threshold

        # 10. Final trust = combination of all signals
        final_trust = (
            0.3 * trust_from_error
            + 0.3 * trust_from_feedback
            + 0.2 * trend_signal
            + 0.2 * (1.0 if raw_error.mean() < self.calibrated_thresholds.get(task_id, float('inf')) else 0.0)
        )

        return {
            "trust": final_trust.clamp(0, 1),
            "raw_error": raw_error,
            "normalized_error": normalized_error,
            "trust_from_error": trust_from_error,
            "trust_from_feedback": trust_from_feedback,
            "trend_signal": trend_signal.unsqueeze(0),
            "corrected_pred": corrected_pred,
            "should_replan": (final_trust < 0.5).float(),
        }

    def get_trust_for_consolidation(self, task_id: int) -> float:
        """Get accumulated trust for CL consolidation decisions."""
        if task_id not in self.ema_error:
            return 0.5
        # Low accumulated error = high trust = protect this knowledge
        trust = np.exp(-self.ema_error[task_id])
        return float(trust)

    def get_error_statistics(self, task_id: int) -> dict[str, float]:
        """Get error statistics for analysis."""
        if task_id not in self.error_history:
            return {"mean": 0, "std": 0, "max": 0, "trend": 0}

        errors = list(self.error_history[task_id])
        return {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "max": float(np.max(errors)),
            "trend": float(np.mean(errors[-5:]) - np.mean(errors[:5])) if len(errors) >= 10 else 0,
        }


class MultiStepClosedLoopTrust:
    """Closed-loop trust over multiple prediction horizons."""

    def __init__(self, obs_dim: int = 64, horizons: list[int] = [1, 4, 8, 16]):
        self.horizons = horizons
        self.trust_correctors = {
            h: ClosedLoopTrustCorrector(obs_dim=obs_dim) for h in horizons
        }

    def observe_and_update(
        self,
        predictions: dict[int, torch.Tensor],
        actual_obs: torch.Tensor,
        action: torch.Tensor,
        task_id: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Process closed-loop step at multiple horizons."""
        results = {}
        for horizon, corrector in self.trust_correctors.items():
            if horizon in predictions:
                results[f"h={horizon}"] = corrector.observe_and_update(
                    predictions[horizon], actual_obs, action, task_id
                )
        return results


if __name__ == "__main__":
    # Demo
    corrector = ClosedLoopTrustCorrector(obs_dim=64)

    print("Closed-Loop Trust Correction Demo")
    print("=" * 50)

    for step in range(20):
        predicted = torch.randn(1, 64)
        actual = predicted + torch.randn(1, 64) * (0.1 + step * 0.02)
        action = torch.randn(1, 2)

        result = corrector.observe_and_update(predicted, actual, action, task_id=0)

        print(
            f"Step {step:2d}: "
            f"trust={result['trust'].item():.3f} "
            f"error={result['raw_error'].mean().item():.4f} "
            f"replan={result['should_replan'].item():.0f}"
        )

    stats = corrector.get_error_statistics(task_id=0)
    print(f"\nFinal stats: {stats}")
    print(f"Consolidation trust: {corrector.get_trust_for_consolidation(0):.3f}")
