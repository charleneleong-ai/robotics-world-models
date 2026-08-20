"""Continual World Action Model (WAM) with Trust-Weighted Consolidation.

Full pipeline:
1. Observation → [World Model] → Trust Score
2. Trust Score → [Calibration] → [Feedback Correction] → [Verification]
3. Verified Trust → [Agentic Layer] → Decision (Execute/Explore/Help)
4. Execute → [VLA Policy] → Action
5. Update → Trust-weighted gradient + Trust-weighted consolidation

Mitigations:
- Conformal prediction calibration
- Feedback correction using real-time observations
- Forward-inverse cycle consistency verification
- Ensemble of world models (optional)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import copy
import numpy as np
from collections import deque

from wam_world_model import WorldActionModel, AgenticDecisionLayer


class VLPolicy(nn.Module):
    """Vision-Language-Action Policy network.

    Simple MLP that maps observations to actions.
    In real VLA models (π₀, GEN-1), this would be a large transformer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate action from observation."""
        return self.net(obs)


class ContinualWAM:
    """Continual World Action Model with trust-weighted consolidation.

    Full pipeline:
    1. Compute trust scores from world model predictions
    2. Calibrate trust scores using conformal prediction
    3. Correct trust scores using real-time feedback
    4. Verify trust scores using forward-inverse cycle consistency
    5. Make agentic decisions (execute/explore/help)
    6. Update VLA policy with trust-weighted gradients
    7. Consolidate using trust-weighted EWC
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        confidence_threshold: float = 0.7,
        exploration_threshold: float = 0.3,
        ewc_lambda: float = 5000.0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.exploration_threshold = exploration_threshold
        self.ewc_lambda = ewc_lambda

        # World Action Model (with trust scoring, calibration, feedback, verification)
        self.wam = WorldActionModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # VLA Policy
        self.vla = VLPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # Optimizers
        self.wam_optimizer = torch.optim.Adam(self.wam.parameters(), lr=lr)
        self.vla_optimizer = torch.optim.Adam(self.vla.parameters(), lr=lr)

        # Trust-weighted consolidation
        self.trust_consolidation = TrustWeightedConsolidation(
            model=self.vla,
            ewc_lambda=ewc_lambda,
            device=device,
        )

        # Task history
        self.task_count = 0
        self.previous_models: dict[int, nn.Module] = {}
        self.task_trust_scores: dict[int, list[float]] = {}

        # Experience buffer for feedback correction
        self.experience_buffer: deque = deque(maxlen=1000)

    def compute_trust_scores(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> dict:
        """Compute trust scores with all mitigations.

        Args:
            obs: (B, obs_dim)
            action: (B, action_dim)
            next_obs: (B, obs_dim)

        Returns:
            Dictionary with trust scores and diagnostics
        """
        # Get trust scores from WAM (includes calibration, feedback, verification)
        trust_results = self.wam.compute_trust_scores(obs, action, next_obs)

        # Store experience for feedback correction
        for i in range(len(obs)):
            self.experience_buffer.append({
                "obs": obs[i].cpu(),
                "action": action[i].cpu(),
                "next_obs": next_obs[i].cpu(),
                "trust_score": trust_results["final_trust"][i].item(),
            })

        return trust_results

    def get_agentic_decision(self, trust_score: torch.Tensor) -> dict:
        """Make agentic decision based on trust score.

        Args:
            trust_score: (B,) trust scores

        Returns:
            Dictionary with decisions
        """
        return self.wam.get_agentic_decision(
            trust_score,
            self.confidence_threshold,
            self.exploration_threshold,
        )

    def update_vla_with_trust(
        self,
        obs: torch.Tensor,
        target_actions: torch.Tensor,
        trust_scores: torch.Tensor,
    ) -> dict:
        """Update VLA policy with trust-weighted gradients.

        Args:
            obs: (B, obs_dim)
            target_actions: (B, action_dim)
            trust_scores: (B,) trust scores

        Returns:
            Dictionary with loss metrics
        """
        self.vla.train()
        self.vla_optimizer.zero_grad()

        # Forward pass
        pred_actions = self.vla(obs)

        # Trust-weighted MSE loss
        action_loss = F.mse_loss(pred_actions, target_actions, reduction="none").mean(dim=-1)
        weighted_loss = (trust_scores * action_loss).mean()

        # KD loss from previous models
        kd_loss = torch.tensor(0.0, device=self.device)
        if self.previous_models:
            for prev_model in self.previous_models.values():
                prev_model.eval()
                with torch.no_grad():
                    prev_actions = prev_model(obs)
                kd_loss += F.mse_loss(pred_actions, prev_actions)

        # EWC penalty
        ewc_penalty = self.trust_consolidation.compute_penalty()

        # Total loss
        total_loss = weighted_loss + 0.1 * kd_loss + ewc_penalty

        total_loss.backward()
        self.vla_optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "action_loss": weighted_loss.item(),
            "kd_loss": kd_loss.item(),
            "ewc_penalty": ewc_penalty.item(),
        }

    def update_wam(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> dict:
        """Update world action model.

        Args:
            obs_seq: (B, T, obs_dim)
            action_seq: (B, T, action_dim)

        Returns:
            Dictionary with loss metrics
        """
        self.wam.train()
        self.wam_optimizer.zero_grad()

        # Forward pass
        predictions = self.wam(obs_seq, action_seq)

        # Reconstruction loss
        obs_loss = F.mse_loss(predictions["obs_preds"], obs_seq)

        # Reward loss
        reward_loss = F.mse_loss(predictions["reward_preds"].squeeze(-1), torch.zeros_like(predictions["reward_preds"].squeeze(-1)))

        # KL divergence between posterior and prior
        kl_loss = torch.tensor(0.0, device=self.device)
        for posterior_logits, prior_logits in zip(
            predictions["posterior_logits"], predictions["prior_logits"]
        ):
            kl_loss += F.kl_div(
                F.log_softmax(posterior_logits, dim=-1),
                F.softmax(prior_logits, dim=-1),
                reduction="batchmean",
            )
        kl_loss /= len(predictions["posterior_logits"])

        # Total loss
        total_loss = obs_loss + 0.1 * reward_loss + 0.01 * kl_loss

        total_loss.backward()
        self.wam_optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def observe_and_act(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        target_action: Optional[torch.Tensor] = None,
        task_id: int = 0,
    ) -> dict:
        """Full pipeline: observe, compute trust, decide, act, update.

        Args:
            obs: (B, obs_dim) current observation
            next_obs: (B, obs_dim) next observation
            target_action: (B, action_dim) target action (if available)
            task_id: current task identifier

        Returns:
            Dictionary with all metrics
        """
        # Generate action from VLA
        with torch.no_grad():
            generated_action = self.vla(obs)

        # Compute trust scores
        trust_results = self.compute_trust_scores(obs, generated_action, next_obs)
        trust_scores = trust_results["final_trust"]

        # Make agentic decision
        decisions = self.get_agentic_decision(trust_scores)

        # Store trust scores
        if task_id not in self.task_trust_scores:
            self.task_trust_scores[task_id] = []
        self.task_trust_scores[task_id].extend(trust_scores.tolist())

        # Update VLA with trust-weighted gradients
        if target_action is not None:
            vla_metrics = self.update_vla_with_trust(obs, target_action, trust_scores)
        else:
            vla_metrics = {"total_loss": 0, "action_loss": 0, "kd_loss": 0, "ewc_penalty": 0}

        return {
            "trust_scores": trust_scores,
            "decisions": decisions,
            "vla_metrics": vla_metrics,
            "trust_diagnostics": trust_results,
        }

    def consolidate(self, task_id: int):
        """Consolidate after task completion.

        Args:
            task_id: completed task identifier
        """
        # Compute average trust for the task
        avg_trust = np.mean(self.task_trust_scores.get(task_id, [0.5]))

        # Set trust for consolidation
        self.trust_consolidation.set_trust(task_id, avg_trust)

        # Save model snapshot
        self.previous_models[task_id] = copy.deepcopy(self.vla)

        self.task_count += 1


class TrustWeightedConsolidation:
    """Trust-weighted EWC consolidation.

    Uses trust scores to modulate the strength of the EWC penalty:
    - High trust: strong constraint (protect knowledge)
    - Low trust: weak constraint (allow plasticity)
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 5000.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device
        self.fisher_info: dict[int, dict[str, torch.Tensor]] = {}
        self.optimal_params: dict[int, dict[str, torch.Tensor]] = {}
        self.task_trust: dict[int, float] = {}

    def set_trust(self, task_id: int, trust_score: float):
        """Set trust score for a task."""
        self.task_trust[task_id] = trust_score

    def compute_fisher(self, task_id: int, dataloader, num_samples: int = 1000):
        """Compute Fisher information for the task."""
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
            obs = batch["obs"].to(self.device)
            actions = self.model(obs)
            loss = F.mse_loss(actions, obs[:, :actions.shape[-1]])  # Dummy loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.data.pow(2)
            count += 1

        for n in fisher:
            fisher[n] /= max(count, 1)

        self.fisher_info[task_id] = fisher
        self.optimal_params[task_id] = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def compute_penalty(self) -> torch.Tensor:
        """Compute trust-weighted EWC penalty."""
        if not self.fisher_info:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for task_id, fisher in self.fisher_info.items():
            trust = self.task_trust.get(task_id, 0.5)
            for n, p in self.model.named_parameters():
                if n in fisher and n in self.optimal_params[task_id]:
                    optimal = self.optimal_params[task_id][n].to(self.device)
                    # Trust-weighted penalty: high trust → strong constraint
                    penalty += trust * self.ewc_lambda * (fisher[n] * (p - optimal).pow(2)).sum()

        return penalty
