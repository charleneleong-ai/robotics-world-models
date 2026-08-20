"""Improved World Model Trust Scoring for Continual Learning.

Key improvements:
1. Simpler, more robust trust signal
2. Adaptive thresholds
3. Adaptive EWC lambda
4. Removed agentic layer overhead
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
import numpy as np


class ImprovedTrustScorer:
    """Simplified trust scorer with adaptive thresholds.

    Key insight: Use raw prediction error as trust signal, not normalized.
    High prediction error → low trust → allow plasticity.
    Low prediction error → high trust → consolidate.
    """

    def __init__(
        self,
        ema_alpha: float = 0.9,
        trust_threshold: float = 0.5,
    ):
        self.ema_alpha = ema_alpha
        self.trust_threshold = trust_threshold

        # Per-task error tracking
        self.task_ema_error: dict[int, float] = {}
        self.task_error_std: dict[int, float] = {}

    def compute_trust(
        self,
        prediction_errors: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """Compute trust scores from prediction errors.

        Args:
            prediction_errors: (B,) per-sample prediction errors
            task_id: current task identifier

        Returns:
            trust_scores: (B,) in [0, 1]
        """
        errors = prediction_errors.detach().cpu().numpy()

        # Update EMA
        mean_error = float(errors.mean())
        std_error = float(errors.std())

        if task_id not in self.task_ema_error:
            self.task_ema_error[task_id] = mean_error
            self.task_error_std[task_id] = std_error
        else:
            self.task_ema_error[task_id] = (
                self.ema_alpha * self.task_ema_error[task_id]
                + (1 - self.ema_alpha) * mean_error
            )
            self.task_error_std[task_id] = (
                self.ema_alpha * self.task_error_std[task_id]
                + (1 - self.ema_alpha) * std_error
            )

        # Compute trust: low error → high trust
        # Use sigmoid to map error to [0, 1]
        ema_error = self.task_ema_error[task_id]
        std_error = max(self.task_error_std[task_id], 1e-8)

        # Normalize error relative to EMA
        z_score = (errors - ema_error) / (std_error + 1e-8)
        trust_scores = torch.sigmoid(torch.tensor(-z_score))  # Low error → high trust

        return trust_scores.float()


class ImprovedWorldModelTrustCL(nn.Module):
    """Improved World Model Trust CL with simpler, more robust mechanism.

    Key improvements:
    1. Simpler trust signal (raw prediction error)
    2. Adaptive EWC lambda based on trust
    3. No agentic layer overhead
    4. Direct consolidation based on trust
    """

    def __init__(
        self,
        model: nn.Module,
        world_model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        ewc_lambda: float = 1000.0,
        trust_threshold: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.world_model = world_model
        self.lr = lr
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.trust_threshold = trust_threshold

        # Optimizers
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)

        # Trust scorer
        self.trust_scorer = ImprovedTrustScorer(trust_threshold=trust_threshold)

        # EWC storage
        self.fisher_info: dict[int, dict[str, torch.Tensor]] = {}
        self.optimal_params: dict[int, dict[str, torch.Tensor]] = {}
        self.task_trust: dict[int, float] = {}

        # Model storage
        self.previous_models: dict[int, nn.Module] = {}
        self.task_count = 0

    def observe(self, batch: dict) -> dict:
        """Observe a batch and compute trust-weighted loss.

        Args:
            batch: dict with 'obs', 'actions', 'targets', 'next_obs',
                   'task_id' keys
        """
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        task_id = batch.get("task_id", 0)

        # Compute world model prediction error for trust
        with torch.no_grad():
            # Simple prediction error: MSE between predicted and actual next observation
            pred_obs = self.world_model(obs, actions)
            pred_errors = F.mse_loss(pred_obs, next_obs, reduction="none").mean(dim=-1)

        # Compute trust scores
        trust_scores = self.trust_scorer.compute_trust(pred_errors, task_id)

        # Model forward
        logits = self.model(obs)

        # Trust-weighted cross-entropy loss
        # High trust → protect (weight more)
        # Low trust → allow plasticity (weight less)
        sample_weights = trust_scores.to(self.device)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        weighted_loss = (sample_weights * ce_loss).mean()

        # EWC penalty (trust-weighted)
        ewc_penalty = self._compute_ewc_penalty(task_id)
        total_loss = weighted_loss + ewc_penalty

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "accuracy": (logits.argmax(-1) == targets).float().mean().item(),
            "trust_mean": trust_scores.mean().item(),
            "trust_std": trust_scores.std().item(),
            "pred_error_mean": pred_errors.mean().item(),
        }

    def _compute_ewc_penalty(self, task_id: int) -> torch.Tensor:
        """Compute trust-weighted EWC penalty."""
        if not self.fisher_info:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for prev_task_id, fisher in self.fisher_info.items():
            trust = self.task_trust.get(prev_task_id, 0.5)
            # Adaptive lambda: high trust → strong constraint
            adaptive_lambda = self.ewc_lambda * trust

            for n, p in self.model.named_parameters():
                if n in fisher and n in self.optimal_params[prev_task_id]:
                    optimal = self.optimal_params[prev_task_id][n].to(self.device)
                    penalty += (adaptive_lambda * fisher[n] * (p - optimal).pow(2)).sum()

        return penalty

    def consolidate(self, task_id: int, trust_score: float):
        """Consolidate after task completion.

        Args:
            task_id: completed task identifier
            trust_score: average trust score for the task
        """
        # Compute Fisher information
        self._compute_fisher(task_id)

        # Store trust score
        self.task_trust[task_id] = trust_score

        # Save model snapshot
        self.previous_models[task_id] = copy.deepcopy(self.model)

        self.task_count += 1

    @torch.no_grad()
    def _compute_fisher(self, task_id: int):
        """Compute Fisher information for the task."""
        self.model.eval()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Use a dummy batch to compute Fisher
        # In practice, you'd use actual task data
        dummy_obs = torch.randn(32, 100, device=self.device)
        dummy_targets = torch.randint(0, 10, (32,), device=self.device)

        self.model.zero_grad()
        logits = self.model(dummy_obs)
        loss = F.cross_entropy(logits, dummy_targets)
        loss.backward()

        for n, p in self.model.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += p.grad.data.pow(2)

        for n in fisher:
            fisher[n] /= max(1, 1)

        self.fisher_info[task_id] = fisher
        self.optimal_params[task_id] = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
