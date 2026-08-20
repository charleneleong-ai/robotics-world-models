"""World Model Trust Scoring for Continual Learning.

Trust = low prediction error + high model confidence.
When trust is high → consolidate (protect knowledge).
When trust is low → allow more plasticity (learn new patterns).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
import numpy as np


class TrustScorer:
    """Computes trust scores based on world model prediction errors.

    Trust is computed as:
    1. Prediction error: MSE between predicted and actual next observation
    2. Confidence: world model's own confidence estimate
    3. Combined: trust = (1 - error_norm) * confidence

    Uses exponential moving average for stability.
    """

    def __init__(
        self,
        ema_alpha: float = 0.95,
        error_window: int = 100,
        trust_threshold: float = 0.5,
    ):
        self.ema_alpha = ema_alpha
        self.error_window = error_window
        self.trust_threshold = trust_threshold

        # Per-task error tracking
        self.task_errors: dict[int, deque] = {}
        self.task_ema_error: dict[int, float] = {}
        self.task_ema_confidence: dict[int, float] = {}

    def compute_trust(
        self,
        prediction_errors: torch.Tensor,
        confidences: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """Compute trust scores for a batch.

        Args:
            prediction_errors: (B,) per-sample prediction errors
            confidences: (B,) per-sample confidence scores [0, 1]
            task_id: current task identifier

        Returns:
            trust_scores: (B,) in [0, 1]
        """
        errors = prediction_errors.detach().cpu().numpy()
        confs = confidences.detach().cpu().numpy()

        # Update EMA
        if task_id not in self.task_ema_error:
            self.task_ema_error[task_id] = float(errors.mean())
            self.task_ema_confidence[task_id] = float(confs.mean())
        else:
            self.task_ema_error[task_id] = (
                self.ema_alpha * self.task_ema_error[task_id]
                + (1 - self.ema_alpha) * float(errors.mean())
            )
            self.task_ema_confidence[task_id] = (
                self.ema_alpha * self.task_ema_confidence[task_id]
                + (1 - self.ema_alpha) * float(confs.mean())
            )

        # Normalize errors relative to EMA
        ema_error = self.task_ema_error[task_id]
        if ema_error > 0:
            error_norm = np.clip(errors / (ema_error + 1e-8), 0, 2) / 2
        else:
            error_norm = np.zeros_like(errors)

        # Trust = (1 - normalized_error) * confidence
        trust_scores = (1 - error_norm) * confs

        return torch.tensor(trust_scores, dtype=torch.float32)

    def should_consolidate(self, trust_score: float) -> bool:
        """Decide whether to consolidate based on trust score.

        High trust → consolidate (protect this knowledge).
        Low trust → allow plasticity (learn new patterns).
        """
        return trust_score > self.trust_threshold

    def get_task_stats(self) -> dict:
        """Return per-task trust statistics."""
        stats = {}
        for task_id in self.task_ema_error:
            stats[task_id] = {
                "ema_error": self.task_ema_error[task_id],
                "ema_confidence": self.task_ema_confidence[task_id],
                "trust": (1 - self.task_ema_error[task_id]) * self.task_ema_confidence[task_id],
            }
        return stats


class TrustWeightedConsolidation:
    """Consolidate network parameters based on trust-weighted importance.

    When trust is high for a task → protect parameters important for that task.
    When trust is low → allow parameters to be modified freely.

    Uses EWC-style Fisher information, weighted by trust score.
    """

    def __init__(
        self,
        model: nn.Module,
        trust_threshold: float = 0.5,
        ewc_lambda: float = 5000.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.trust_threshold = trust_threshold
        self.ewc_lambda = ewc_lambda
        self.device = device

        # Per-task Fisher information and optimal parameters
        self.fisher_info: dict[int, dict[str, torch.Tensor]] = {}
        self.optimal_params: dict[int, dict[str, torch.Tensor]] = {}
        self.task_trust: dict[int, float] = {}

    @torch.no_grad()
    def compute_fisher(
        self,
        task_id: int,
        dataloader,
        loss_fn,
        num_samples: int = 1000,
    ):
        """Compute Fisher information for a task.

        Args:
            task_id: task identifier
            dataloader: data from the current task
            loss_fn: loss function
            num_samples: number of samples to estimate Fisher
        """
        self.model.eval()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            self.model.zero_grad()
            loss = loss_fn(batch)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.data.pow(2)
            count += len(batch[0]) if isinstance(batch, (list, tuple)) else 1

        # Normalize
        for n in fisher:
            fisher[n] /= max(count, 1)

        self.fisher_info[task_id] = fisher
        self.optimal_params[task_id] = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def set_trust(self, task_id: int, trust_score: float):
        """Set trust score for a task."""
        self.task_trust[task_id] = trust_score

    def compute_penalty(self) -> torch.Tensor:
        """Compute EWC penalty, weighted by trust scores.

        High trust → protect parameters (large penalty).
        Low trust → allow modification (small penalty).
        """
        penalty = torch.tensor(0.0, device=self.device)

        for task_id, fisher in self.fisher_info.items():
            trust = self.task_trust.get(task_id, 0.5)
            trust_weight = trust if trust > self.trust_threshold else 0.1

            for n, p in self.model.named_parameters():
                if n in fisher and n in self.optimal_params[task_id]:
                    optimal = self.optimal_params[task_id][n].to(self.device)
                    penalty += (trust_weight * self.ewc_lambda * fisher[n] * (p - optimal).pow(2)).sum()

        return penalty

    def consolidation_strength(self, task_id: int) -> float:
        """Return consolidation strength for a task (0-1).

        1.0 = fully consolidated (frozen).
        0.0 = fully plastic.
        """
        trust = self.task_trust.get(task_id, 0.5)
        if trust > self.trust_threshold:
            return min(1.0, trust)
        return 0.0
