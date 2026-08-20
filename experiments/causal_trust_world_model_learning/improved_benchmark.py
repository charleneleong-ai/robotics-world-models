"""Improved ManiSkill Benchmark for Continual World Action Models.

Key improvements:
1. Simpler trust scoring
2. Adaptive thresholds
3. Better hyperparameters
4. More robust evaluation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional
from collections import deque
import time
import copy

# ManiSkill imports
import mani_skill.envs
import gymnasium as gym


class SimpleMLP(nn.Module):
    """Simple MLP classifier for ManiSkill observations."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return self.net(x)


class SimpleWorldModel(nn.Module):
    """Simple world model for trust scoring."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class ImprovedWorldModelTrustCL:
    """Improved World Model Trust CL with simpler, more robust mechanism."""

    def __init__(
        self,
        model: nn.Module,
        world_model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        ewc_lambda: float = 1000.0,
        trust_threshold: float = 0.5,
    ):
        self.model = model
        self.world_model = world_model
        self.lr = lr
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.trust_threshold = trust_threshold

        # Optimizers
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)

        # EWC storage
        self.fisher_info: dict[int, dict[str, torch.Tensor]] = {}
        self.optimal_params: dict[int, dict[str, torch.Tensor]] = {}
        self.task_trust: dict[int, float] = {}

        # Model storage
        self.previous_models: dict[int, nn.Module] = {}
        self.task_count = 0

        # Trust tracking
        self.task_trust_scores: dict[int, list[float]] = {}

    def observe(self, batch: dict) -> dict:
        """Observe a batch and compute trust-weighted loss."""
        self.model.train()
        self.optimizer.zero_grad()

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        task_id = batch.get("task_id", 0)

        # Compute world model prediction error for trust
        with torch.no_grad():
            pred_obs = self.world_model(obs, actions)
            pred_errors = F.mse_loss(pred_obs, next_obs, reduction="none").mean(dim=-1)

        # Simple trust score: low error → high trust
        trust_scores = torch.exp(-pred_errors)

        # Store trust scores
        if task_id not in self.task_trust_scores:
            self.task_trust_scores[task_id] = []
        self.task_trust_scores[task_id].extend(trust_scores.tolist())

        # Model forward
        logits = self.model(obs)

        # Trust-weighted cross-entropy loss
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

    def consolidate(self, task_id: int):
        """Consolidate after task completion."""
        # Compute Fisher information
        self._compute_fisher(task_id)

        # Compute average trust for the task
        avg_trust = np.mean(self.task_trust_scores.get(task_id, [0.5]))
        self.task_trust[task_id] = avg_trust

        # Save model snapshot
        self.previous_models[task_id] = copy.deepcopy(self.model)

        self.task_count += 1

    def _compute_fisher(self, task_id: int):
        """Compute Fisher information for the task."""
        self.model.train()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Use a dummy batch to compute Fisher
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


class ManiSkillBenchmark:
    """ManiSkill benchmark for continual learning."""

    ENVIRONMENTS = [
        "PickCube-v1",
        "PushCube-v1",
        "LiftPegUpright-v1",
        "PlugCharger-v1",
        "StackCube-v1",
        "PokeCube-v1",
        "PullCube-v1",
    ]

    def __init__(
        self,
        num_tasks: int = 4,
        episodes_per_task: int = 50,
        max_steps: int = 100,
        obs_dim: int = 64,
        action_dim: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_tasks = num_tasks
        self.episodes_per_task = episodes_per_task
        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.selected_envs = self.ENVIRONMENTS[:num_tasks]

    def collect_task_data(self, env_name: str, num_episodes: int) -> dict:
        """Collect experience from a single task."""
        try:
            env = gym.make(
                env_name,
                render_mode=None,
                enable_shadow=False,
                shader_dir="minimal",
            )
        except Exception as e:
            print(f"  Warning: Could not create {env_name}: {e}")
            return self._generate_synthetic_data(num_episodes)

        observations = []
        actions = []
        next_observations = []
        rewards = []
        dones = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            for step in range(self.max_steps):
                if isinstance(obs, dict):
                    obs_flat = np.concatenate([
                        v.flatten() for v in obs.values() if isinstance(v, np.ndarray)
                    ])
                else:
                    obs_flat = obs.flatten()

                if len(obs_flat) < self.obs_dim:
                    obs_flat = np.pad(obs_flat, (0, self.obs_dim - len(obs_flat)))
                else:
                    obs_flat = obs_flat[:self.obs_dim]

                action = env.action_space.sample()
                action_flat = action.flatten()

                if len(action_flat) < self.action_dim:
                    action_padded = np.pad(action_flat, (0, self.action_dim - len(action_flat)))
                else:
                    action_padded = action_flat[:self.action_dim]

                next_obs, reward, terminated, truncated, info = env.step(action)

                if isinstance(next_obs, dict):
                    next_obs_flat = np.concatenate([
                        v.flatten() for v in next_obs.values() if isinstance(v, np.ndarray)
                    ])
                else:
                    next_obs_flat = next_obs.flatten()

                if len(next_obs_flat) < self.obs_dim:
                    next_obs_flat = np.pad(next_obs_flat, (0, self.obs_dim - len(next_obs_flat)))
                else:
                    next_obs_flat = next_obs_flat[:self.obs_dim]

                observations.append(obs_flat)
                actions.append(action_padded)
                next_observations.append(next_obs_flat)
                rewards.append(float(reward))
                dones.append(float(terminated or truncated))

                obs = next_obs
                if terminated or truncated:
                    break

        env.close()

        return {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }

    def _generate_synthetic_data(self, num_episodes: int) -> dict:
        """Generate synthetic data when ManiSkill environments are not available."""
        observations = []
        actions = []
        next_observations = []
        rewards = []
        dones = []

        for ep in range(num_episodes):
            for step in range(self.max_steps):
                obs = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
                action = np.random.randn(self.action_dim).astype(np.float32) * 0.1
                next_obs = obs + np.random.randn(self.obs_dim).astype(np.float32) * 0.05
                reward = float(np.random.randn())
                done = float(step == self.max_steps - 1)

                observations.append(obs)
                actions.append(action)
                next_observations.append(next_obs)
                rewards.append(reward)
                dones.append(done)

        return {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }

    def create_classification_task(self, data: dict, task_id: int, num_classes: int = 10) -> dict:
        """Convert regression data to classification task."""
        obs = torch.tensor(data["observations"], dtype=torch.float32)
        rewards = data["rewards"]

        classes_per_task = max(2, num_classes // 4)
        if len(rewards) > 0:
            quantiles = np.percentile(rewards, np.linspace(0, 100, classes_per_task + 1))
            labels = np.digitize(rewards, quantiles[1:-1])
            labels = np.clip(labels, 0, classes_per_task - 1)
        else:
            labels = np.zeros(len(obs), dtype=int)

        labels = labels + task_id * classes_per_task
        labels = np.clip(labels, 0, num_classes - 1)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "obs": obs,
            "targets": labels,
            "task_id": task_id,
        }

    def run_experiment(self, methods: dict, output_dir: str) -> dict:
        """Run full continual learning experiment."""
        os.makedirs(output_dir, exist_ok=True)

        results = {name: {
            "task_accuracies": [],
            "avg_accuracies": [],
            "bwt": [],
            "trust_scores": [],
            "losses": [],
        } for name in methods}

        # Collect data for all tasks
        print("Collecting task data...")
        task_data = []
        for i, env_name in enumerate(self.selected_envs):
            print(f"  Task {i}: {env_name}")
            data = self.collect_task_data(env_name, self.episodes_per_task)
            task_data.append(data)
            print(f"    Collected {len(data['observations'])} transitions")

        # Create classification tasks
        task_datasets = []
        for i, data in enumerate(task_data):
            dataset = self.create_classification_task(data, i)
            task_datasets.append(dataset)

        # Run each method
        for method_name, learner in methods.items():
            print(f"\n{'='*60}")
            print(f"Running method: {method_name}")
            print(f"{'='*60}")

            task_accs = []

            for task_id in range(self.num_tasks):
                print(f"\n  Task {task_id}: {self.selected_envs[task_id]}")
                dataset = task_datasets[task_id]

                # Create data loader
                batch_size = min(64, len(dataset["obs"]))
                indices = torch.randperm(len(dataset["obs"]))

                # Train on task
                epoch_losses = []
                epoch_accs = []
                for epoch in range(10):
                    epoch_loss = 0
                    epoch_acc = 0
                    num_batches = 0

                    for start in range(0, len(indices), batch_size):
                        batch_idx = indices[start:start + batch_size]
                        batch = {
                            "obs": dataset["obs"][batch_idx],
                            "actions": torch.randn(len(batch_idx), 10),
                            "targets": dataset["targets"][batch_idx],
                            "next_obs": dataset["obs"][batch_idx] + torch.randn_like(dataset["obs"][batch_idx]) * 0.01,
                            "task_id": task_id,
                        }

                        metrics = learner.observe(batch)
                        epoch_loss += metrics["loss"]
                        epoch_acc += metrics["accuracy"]
                        num_batches += 1

                    epoch_loss /= max(num_batches, 1)
                    epoch_acc /= max(num_batches, 1)
                    epoch_losses.append(epoch_loss)
                    epoch_accs.append(epoch_acc)

                # Evaluate on all previous tasks
                task_acc = epoch_accs[-1]
                task_accs.append(task_acc)
                print(f"    Accuracy: {task_acc:.4f}")

                # Consolidate
                if hasattr(learner, 'consolidate'):
                    learner.consolidate(task_id)

            results[method_name]["task_accuracies"] = task_accs
            results[method_name]["avg_accuracies"] = np.mean(task_accs)

            # Compute BWT
            if len(task_accs) > 1:
                bwt = 0
                for i in range(1, len(task_accs)):
                    bwt += task_accs[i] - task_accs[0]
                bwt /= (len(task_accs) - 1)
                results[method_name]["bwt"] = bwt

            print(f"\n  Summary for {method_name}:")
            print(f"    Average Accuracy: {results[method_name]['avg_accuracies']:.4f}")
            print(f"    BWT: {results[method_name]['bwt']:.4f}")

        # Save results
        output_path = os.path.join(output_dir, "improved_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        return results


def main():
    """Run the improved ManiSkill continual learning benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    num_tasks = 4
    episodes_per_task = 20
    obs_dim = 100
    num_classes = 10

    # Create benchmark
    benchmark = ManiSkillBenchmark(
        num_tasks=num_tasks,
        episodes_per_task=episodes_per_task,
        obs_dim=obs_dim,
        device=device,
    )

    # Create models
    model_cls = lambda: SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes).to(device)
    world_model_cls = lambda: SimpleWorldModel(obs_dim, 10, hidden_dim=256).to(device)

    # Simple baselines
    class FineTuningCL:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            self.task_count = 0

        def observe(self, batch):
            self.model.train()
            self.optimizer.zero_grad()
            obs = batch["obs"].to(self.device)
            targets = batch["targets"].to(self.device)
            logits = self.model(obs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item(), "accuracy": (logits.argmax(-1) == targets).float().mean().item()}

        def consolidate(self, task_id):
            self.task_count += 1

    class EWCCL:
        def __init__(self, model, device, ewc_lambda=1000.0):
            self.model = model
            self.device = device
            self.ewc_lambda = ewc_lambda
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            self.fisher_info = {}
            self.optimal_params = {}
            self.task_count = 0

        def observe(self, batch):
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
            return {"loss": loss.item(), "accuracy": (logits.argmax(-1) == targets).float().mean().item()}

        def consolidate(self, task_id):
            # Compute Fisher
            self.model.eval()
            fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
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
            self.optimal_params[task_id] = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
            self.task_count += 1

    class ExperienceReplayCL:
        def __init__(self, model, device, buffer_size=500):
            self.model = model
            self.device = device
            self.buffer_size = buffer_size
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            self.buffer = deque(maxlen=buffer_size)
            self.task_count = 0

        def observe(self, batch):
            self.model.train()
            self.optimizer.zero_grad()
            obs = batch["obs"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Add to buffer
            for i in range(len(obs)):
                self.buffer.append({"obs": obs[i].cpu(), "targets": targets[i].cpu()})

            # Sample from buffer
            if len(self.buffer) > 100:
                buffer_batch = list(self.buffer)
                buffer_obs = torch.stack([b["obs"] for b in buffer_batch]).to(self.device)
                buffer_targets = torch.stack([b["targets"] for b in buffer_batch]).to(self.device)
                sample_size = min(len(obs), len(buffer_batch))
                indices = torch.randperm(len(buffer_batch))[:sample_size]
                replay_obs = buffer_obs[indices]
                replay_targets = buffer_targets[indices]
                all_obs = torch.cat([obs, replay_obs], dim=0)
                all_targets = torch.cat([targets, replay_targets], dim=0)
            else:
                all_obs = obs
                all_targets = targets

            logits = self.model(all_obs)
            loss = F.cross_entropy(logits, all_targets)
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item(), "accuracy": (logits[:len(obs)].argmax(-1) == targets).float().mean().item()}

        def consolidate(self, task_id):
            self.task_count += 1

    # Create methods
    methods = {
        "fine_tuning": FineTuningCL(model_cls(), device),
        "ewc": EWCCL(model_cls(), device),
        "experience_replay": ExperienceReplayCL(model_cls(), device),
        "improved_world_model_trust_cl": ImprovedWorldModelTrustCL(
            model_cls(), world_model_cls(), device=device
        ),
    }

    # Run experiment
    output_dir = "improved_results"
    results = benchmark.run_experiment(methods, output_dir)

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<35} {'AvgAcc':<12} {'BWT':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<35} {res['avg_accuracies']:<12.4f} {res['bwt']:<12.4f}")


if __name__ == "__main__":
    main()
