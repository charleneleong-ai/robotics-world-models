"""Continual Learning Agent with World Model Trust Scoring.

Integrates RSSM world model with CL strategies:
1. Trust-Aware CL (accuracy-based trust) — baseline
2. World-Model Trust CL (prediction error trust) — our method
3. EWC, LwF, Experience Replay — standard baselines
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
import copy
import numpy as np

from rssm_world_model import WorldModel
from trust_scoring import TrustScorer, TrustWeightedConsolidation


class ContinualLearner:
    """Base continual learning agent."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.task_count = 0

    def observe(self, batch: dict) -> dict:
        """Process a batch of data. Override in subclasses."""
        raise NotImplementedError

    def consolidate(self, task_id: int = None):
        """Consolidate after a task is complete. Override in subclasses."""
        self.task_count += 1


class FineTuningCL(ContinualLearner):
    """Simple fine-tuning (no CL strategy)."""

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)

        logits = self.model(obs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "accuracy": (logits.argmax(-1) == targets).float().mean().item()}


class AccuracyTrustCL(ContinualLearner):
    """Trust-Aware CL using accuracy-based trust (baseline)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        trust_threshold: float = 0.7,
        kd_weight: float = 1.0,
    ):
        super().__init__(model, lr, device)
        self.trust_threshold = trust_threshold
        self.kd_weight = kd_weight
        self.previous_models: dict[int, nn.Module] = {}
        self.task_accuracies: dict[int, float] = {}

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)

        logits = self.model(obs)
        loss = F.cross_entropy(logits, targets)

        # KD loss from previous models
        if self.previous_models:
            kd_loss = torch.tensor(0.0, device=self.device)
            for prev_model in self.previous_models.values():
                prev_model.eval()
                with torch.no_grad():
                    prev_logits = prev_model(obs)
                kd_loss += F.kl_div(
                    F.log_softmax(logits / 2, dim=-1),
                    F.softmax(prev_logits / 2, dim=-1),
                    reduction="batchmean",
                )
            loss = loss + self.kd_weight * kd_loss

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (logits.argmax(-1) == targets).float().mean().item(),
        }

    def consolidate(self, task_id: int = None):
        """Save model snapshot and compute accuracy-based trust."""
        # Compute accuracy on current task data
        # (caller should provide this)
        super().consolidate()
        self.previous_models[self.task_count - 1] = copy.deepcopy(self.model)


class WorldModelTrustCL(ContinualLearner):
    """Continual Learning with World Model Trust Scoring.

    Our method: uses RSSM prediction error as trust signal.
    High trust (low prediction error) → consolidate.
    Low trust (high prediction error) → allow plasticity.
    """

    def __init__(
        self,
        model: nn.Module,
        world_model: WorldModel,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        ewc_lambda: float = 5000.0,
        trust_threshold: float = 0.5,
        kd_weight: float = 1.0,
    ):
        super().__init__(model, lr, device)
        self.world_model = world_model
        self.world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
        self.trust_scorer = TrustScorer(trust_threshold=trust_threshold)
        self.consolidation = TrustWeightedConsolidation(
            model=model,
            trust_threshold=trust_threshold,
            ewc_lambda=ewc_lambda,
            device=device,
        )
        self.kd_weight = kd_weight
        self.previous_models: dict[int, nn.Module] = {}
        self.task_trust_scores: dict[int, list[float]] = {}

    def train_world_model(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        rewards_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        epochs: int = 5,
    ) -> dict:
        """Train the world model on a sequence of experience.

        Args:
            obs_seq: (B, T, obs_dim)
            actions_seq: (B, T, action_dim)
            rewards_seq: (B, T)
            dones_seq: (B, T)
            epochs: training epochs

        Returns:
            Training metrics
        """
        self.world_model.train()
        all_metrics = {}

        for epoch in range(epochs):
            self.world_model_optimizer.zero_grad()

            result = self.world_model.training_step(
                obs_seq, actions_seq, rewards_seq, dones_seq
            )

            # Backprop total loss
            if isinstance(result["total_loss"], torch.Tensor):
                result["total_loss"].backward()
                self.world_model_optimizer.step()

            for k, v in result.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = v

        return all_metrics

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
            pred_errors = self.world_model.compute_prediction_error(
                obs, actions, next_obs
            )
            trust_confidences = self.world_model.compute_trust(obs, actions)

        # Compute trust scores
        trust_scores = self.trust_scorer.compute_trust(
            pred_errors, trust_confidences, task_id
        )

        # Record trust scores
        if task_id not in self.task_trust_scores:
            self.task_trust_scores[task_id] = []
        self.task_trust_scores[task_id].extend(trust_scores.tolist())

        # Model forward
        logits = self.model(obs)

        # Trust-weighted cross-entropy loss
        sample_weights = trust_scores.to(self.device)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        weighted_loss = (sample_weights * ce_loss).mean()

        # KD loss from previous models, weighted by trust
        if self.previous_models:
            kd_loss = torch.tensor(0.0, device=self.device)
            for prev_task_id, prev_model in self.previous_models.items():
                prev_model.eval()
                with torch.no_grad():
                    prev_logits = prev_model(obs)
                kl = F.kl_div(
                    F.log_softmax(logits / 2, dim=-1),
                    F.softmax(prev_logits / 2, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                # Weight KD by trust: high trust → protect more
                prev_trust = np.mean(
                    self.task_trust_scores.get(prev_task_id, [0.5])
                )
                kd_loss = kd_loss + prev_trust * self.kd_weight * kl.mean()
            weighted_loss = weighted_loss + kd_loss

        # EWC penalty (trust-weighted)
        ewc_penalty = self.consolidation.compute_penalty()
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

    def consolidate(self, task_id: int, trust_score: float):
        """Consolidate after task completion.

        Args:
            task_id: completed task identifier
            trust_score: average trust score for the task
        """
        # Compute Fisher information
        self.consolidation.set_trust(task_id, trust_score)

        # Save model snapshot
        self.previous_models[task_id] = copy.deepcopy(self.model)

        super().consolidate()


class ExperienceReplayCL(ContinualLearner):
    """Experience Replay baseline."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        buffer_size: int = 1000,
        replay_ratio: float = 0.5,
    ):
        super().__init__(model, lr, device)
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.buffer: deque = deque(maxlen=buffer_size)

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Add to buffer
        for i in range(len(obs)):
            self.buffer.append({
                "obs": obs[i].cpu(),
                "targets": targets[i].cpu(),
            })

        # Sample from buffer
        if len(self.buffer) > 0:
            buffer_batch = list(self.buffer)
            buffer_obs = torch.stack([b["obs"] for b in buffer_batch]).to(self.device)
            buffer_targets = torch.stack([b["targets"] for b in buffer_batch]).to(self.device)

            # Sample a subset from buffer to match current batch size
            sample_size = min(len(obs), len(buffer_batch))
            indices = torch.randperm(len(buffer_batch))[:sample_size]
            replay_obs = buffer_obs[indices]
            replay_targets = buffer_targets[indices]

            # Combine current + replay
            all_obs = torch.cat([obs, replay_obs], dim=0)
            all_targets = torch.cat([targets, replay_targets], dim=0)
        else:
            all_obs = obs
            all_targets = targets

        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_targets)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (logits[:len(obs)].argmax(-1) == targets).float().mean().item(),
        }


class EWCCL(ContinualLearner):
    """Elastic Weight Consolidation baseline."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        ewc_lambda: float = 5000.0,
    ):
        super().__init__(model, lr, device)
        self.ewc_lambda = ewc_lambda
        self.fisher_info: dict[int, dict[str, torch.Tensor]] = {}
        self.optimal_params: dict[int, dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def compute_fisher(
        self,
        task_id: int,
        dataloader,
        loss_fn,
        num_samples: int = 1000,
    ):
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
            count += 1

        for n in fisher:
            fisher[n] /= max(count, 1)

        self.fisher_info[task_id] = fisher
        self.optimal_params[task_id] = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        targets = batch["targets"].to(self.device)

        logits = self.model(obs)
        loss = F.cross_entropy(logits, targets)

        # EWC penalty
        if self.fisher_info:
            penalty = torch.tensor(0.0, device=self.device)
            for task_id, fisher in self.fisher_info.items():
                for n, p in self.model.named_parameters():
                    if n in fisher and n in self.optimal_params[task_id]:
                        optimal = self.optimal_params[task_id][n].to(self.device)
                        penalty += (fisher[n] * (p - optimal).pow(2)).sum()
            loss = loss + self.ewc_lambda * penalty

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (logits.argmax(-1) == targets).float().mean().item(),
        }

    def consolidate(self, task_id: int = None):
        """After task completion, we've already computed Fisher during training."""
        super().consolidate()


class PrioritizedReplayCL(ContinualLearner):
    """Prioritized Experience Replay (PER) baseline.

    Prioritizes transitions by TD error (Schaul et al., 2016).
    Uses importance sampling to correct bias.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        buffer_size: int = 2000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 10000,
    ):
        super().__init__(model, lr, device)
        self.buffer_size = buffer_size
        self.alpha = alpha  # prioritization exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        # Sum-tree for efficient prioritized sampling
        self.buffer: list[dict] = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def _beta(self) -> float:
        """Anneal beta from beta_start to 1."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def _sample_batch(self, batch_size: int) -> tuple[list[int], np.ndarray]:
        """Sample a batch using proportional prioritization."""
        if len(self.buffer) < batch_size:
            indices = list(range(len(self.buffer)))
        else:
            # Proportional prioritization
            probs = self.priorities[:len(self.buffer)] ** self.alpha
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Importance sampling weights
        beta = self._beta()
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return indices.tolist(), weights

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)
        next_obs = batch.get("next_obs", obs).to(self.device)

        # Compute TD errors for current batch
        with torch.no_grad():
            logits = self.model(obs)
            td_errors = (logits.argmax(-1) != targets).float().cpu().numpy()

        # Store in buffer
        for i in range(len(obs)):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append({
                    "obs": obs[i].cpu(),
                    "actions": actions[i].cpu(),
                    "targets": targets[i].cpu(),
                    "next_obs": next_obs[i].cpu(),
                })
                self.priorities[self.pos] = self.max_priority
            else:
                self.buffer[self.pos] = {
                    "obs": obs[i].cpu(),
                    "actions": actions[i].cpu(),
                    "targets": targets[i].cpu(),
                    "next_obs": next_obs[i].cpu(),
                }
            self.pos = (self.pos + 1) % self.buffer_size

        # Sample from buffer with prioritization
        if len(self.buffer) > 100:
            indices, weights = self._sample_batch(len(obs))
            buffer_obs = torch.stack([self.buffer[i]["obs"] for i in indices]).to(self.device)
            buffer_targets = torch.stack([self.buffer[i]["targets"] for i in indices]).to(self.device)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

            # Combine current + replay
            all_obs = torch.cat([obs, buffer_obs], dim=0)
            all_targets = torch.cat([targets, buffer_targets], dim=0)
            all_weights = torch.cat([torch.ones(len(obs), device=self.device), weights], dim=0)

            # Update priorities
            self._update_priority(indices, td_errors)
        else:
            all_obs = obs
            all_targets = targets
            all_weights = torch.ones(len(obs), device=self.device)

        logits = self.model(all_obs)
        ce_loss = F.cross_entropy(logits, all_targets, reduction="none")
        loss = (all_weights * ce_loss).mean()
        loss.backward()
        self.optimizer.step()

        self.frame += 1

        return {
            "loss": loss.item(),
            "accuracy": (logits[:len(obs)].argmax(-1) == targets).float().mean().item(),
        }

    def _update_priority(self, indices: list[int], td_errors: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6
        self.max_priority = max(self.max_priority, self.priorities[indices].max())


class CuriousReplayCL(ContinualLearner):
    """Curious Replay baseline (De Bruin et al., 2020).

    Prioritizes transitions by prediction error (world model).
    Most directly comparable to our work — uses prediction error for replay, not consolidation.
    """

    def __init__(
        self,
        model: nn.Module,
        world_model: WorldModel,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        buffer_size: int = 2000,
        alpha: float = 0.6,
    ):
        super().__init__(model, lr, device)
        self.world_model = world_model
        self.world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
        self.buffer_size = buffer_size
        self.alpha = alpha

        # Buffer with prediction errors
        self.buffer: list[dict] = []
        self.pred_errors: np.ndarray = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0

    def _sample_batch(self, batch_size: int) -> list[int]:
        """Sample by prediction error (higher = more likely to be sampled)."""
        if len(self.buffer) < batch_size:
            return list(range(len(self.buffer)))

        probs = self.pred_errors[:len(self.buffer)] ** self.alpha
        probs = probs / probs.sum()
        return np.random.choice(len(self.buffer), batch_size, p=probs, replace=False).tolist()

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)
        next_obs = batch.get("next_obs", obs).to(self.device)

        # Compute prediction errors for current batch
        with torch.no_grad():
            pred_errors = self.world_model.compute_prediction_error(obs, actions, next_obs)
            pred_errors_np = pred_errors.cpu().numpy()

        # Store in buffer
        for i in range(len(obs)):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append({
                    "obs": obs[i].cpu(),
                    "actions": actions[i].cpu(),
                    "targets": targets[i].cpu(),
                    "next_obs": next_obs[i].cpu(),
                })
                self.pred_errors[self.pos] = pred_errors_np[i]
            else:
                self.buffer[self.pos] = {
                    "obs": obs[i].cpu(),
                    "actions": actions[i].cpu(),
                    "targets": targets[i].cpu(),
                    "next_obs": next_obs[i].cpu(),
                }
                self.pred_errors[self.pos] = pred_errors_np[i]
            self.pos = (self.pos + 1) % self.buffer_size

        # Sample from buffer with prediction error prioritization
        if len(self.buffer) > 100:
            indices = self._sample_batch(len(obs))
            buffer_obs = torch.stack([self.buffer[i]["obs"] for i in indices]).to(self.device)
            buffer_targets = torch.stack([self.buffer[i]["targets"] for i in indices]).to(self.device)

            # Combine current + replay
            all_obs = torch.cat([obs, buffer_obs], dim=0)
            all_targets = torch.cat([targets, buffer_targets], dim=0)
        else:
            all_obs = obs
            all_targets = targets

        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_targets)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (logits[:len(obs)].argmax(-1) == targets).float().mean().item(),
            "pred_error_mean": pred_errors.mean().item(),
        }


class PackNetCL(ContinualLearner):
    """PackNet baseline (Mallya & Lazebnik, 2018).

    Allocates a fixed subset of parameters per task.
    After training on a task, freezes the parameters used for that task.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        prune_ratio: float = 0.5,
    ):
        super().__init__(model, lr, device)
        self.prune_ratio = prune_ratio
        self.frozen_masks: dict[int, dict[str, torch.Tensor]] = {}
        self.task_param_usage: dict[int, set[str]] = {}

    def _compute_importance(self, task_id: int) -> dict[str, torch.Tensor]:
        """Compute parameter importance for the current task."""
        importance = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                importance[name] = param.grad.data.abs()
            else:
                importance[name] = torch.zeros_like(param.data)
        return importance

    def _prune_and_freeze(self, task_id: int):
        """Prune least important parameters and freeze them for this task."""
        importance = self._compute_importance(task_id)
        frozen = {}

        for name, param in self.model.named_parameters():
            if name in importance:
                # Compute threshold for pruning
                imp = importance[name].flatten()
                threshold = torch.quantile(imp, self.prune_ratio)

                # Create mask: 1 = keep, 0 = prune
                mask = (importance[name] > threshold).float()

                # Freeze parameters for this task
                frozen[name] = mask
                self.task_param_usage[task_id] = set(importance.keys())

        self.frozen_masks[task_id] = frozen

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        targets = batch["targets"].to(self.device)

        logits = self.model(obs)
        loss = F.cross_entropy(logits, targets)

        # Mask gradients for previously frozen parameters
        for prev_task_id, masks in self.frozen_masks.items():
            for name, param in self.model.named_parameters():
                if name in masks and param.grad is not None:
                    param.grad.data *= masks[name]

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (logits.argmax(-1) == targets).float().mean().item(),
        }

    def consolidate(self, task_id: int = None):
        """Freeze parameters used for this task."""
        self._prune_and_freeze(self.task_count)
        super().consolidate()


class LwFCL(ContinualLearner):
    """Learning without Forgetting (Li & Hoiem, 2017).

    Uses knowledge distillation from previous model to preserve old knowledge.
    No task-specific parameters — uses the same network for all tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        temperature: float = 2.0,
        kd_weight: float = 1.0,
    ):
        super().__init__(model, lr, device)
        self.temperature = temperature
        self.kd_weight = kd_weight
        self.previous_model: Optional[nn.Module] = None
        self.task_classes: dict[int, int] = {}

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        targets = batch["targets"].to(self.device)
        task_id = batch.get("task_id", 0)

        logits = self.model(obs)
        loss = F.cross_entropy(logits, targets)

        # Knowledge distillation from previous model
        if self.previous_model is not None:
            self.previous_model.eval()
            with torch.no_grad():
                prev_logits = self.previous_model(obs)

            # Soft targets from previous model
            soft_targets = F.softmax(prev_logits / self.temperature, dim=-1)
            soft_logits = F.log_softmax(logits / self.temperature, dim=-1)

            # KD loss
            kd_loss = F.kl_div(soft_logits, soft_targets, reduction="batchmean")
            loss = loss + self.kd_weight * kd_loss * (self.temperature ** 2)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (logits.argmax(-1) == targets).float().mean().item(),
        }

    def consolidate(self, task_id: int = None):
        """Save current model as previous model for KD."""
        self.previous_model = copy.deepcopy(self.model)
        super().consolidate()
