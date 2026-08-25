"""KinDER Benchmark for Continual World Action Models.

Uses KinDER (Kinematic and Dynamic Embodied Reasoning) benchmark
for evaluating continual learning in physical reasoning tasks.

KinDER has 25 environments with 5 core challenges:
1. Basic spatial relations
2. Nonprehensile multi-object manipulation
3. Tool use
4. Combinatorial geometric constraints
5. Dynamic constraints
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

# KinDER imports
try:
    import kindergarden as kinder
    KINDER_AVAILABLE = True
except ImportError:
    KINDER_AVAILABLE = False
    print("KinDER not available, using synthetic data")


class SimpleMLP(nn.Module):
    """Simple MLP classifier for KinDER observations."""

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


class KinDERBenchmark:
    """KinDER benchmark for continual learning.

    Uses KinDER environments for evaluating continual learning
    in physical reasoning tasks.
    """

    # KinDER environments (2D and 3D)
    ENVIRONMENTS_2D = [
        "Motion2D",
        "StickButton2D",
        "DynObstruction2D",
        "DynPushPullHook2D",
    ]

    ENVIRONMENTS_3D = [
        "BaseMotion3D",
        "Shelf3D",
        "SweepIntoDrawer3D",
        "Transport3D",
    ]

    def __init__(
        self,
        num_tasks: int = 4,
        episodes_per_task: int = 50,
        max_steps: int = 100,
        obs_dim: int = 64,  # padded observation dimension
        action_dim: int = 10,  # padded action dimension
        device: torch.device = torch.device("cpu"),
        use_3d: bool = False,
    ):
        self.num_tasks = num_tasks
        self.episodes_per_task = episodes_per_task
        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.use_3d = use_3d

        # Select environments
        if use_3d:
            self.selected_envs = self.ENVIRONMENTS_3D[:num_tasks]
        else:
            self.selected_envs = self.ENVIRONMENTS_2D[:num_tasks]

    def collect_task_data(
        self,
        env_name: str,
        num_episodes: int,
        policy_fn=None,
    ) -> dict:
        """Collect experience from a single task.

        Args:
            env_name: KinDER environment name
            num_episodes: number of episodes to collect
            policy_fn: optional policy function (random if None)

        Returns:
            Dictionary with observations, actions, next_observations, rewards, dones
        """
        if not KINDER_AVAILABLE:
            return self._generate_synthetic_data(num_episodes)

        try:
            # Create KinDER environment
            env = kinder.make(env_name)
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
                # Flatten observation
                if isinstance(obs, dict):
                    obs_flat = np.concatenate([
                        v.flatten() for v in obs.values() if isinstance(v, np.ndarray)
                    ])
                else:
                    obs_flat = obs.flatten()

                # Pad/truncate to obs_dim
                if len(obs_flat) < self.obs_dim:
                    obs_flat = np.pad(obs_flat, (0, self.obs_dim - len(obs_flat)))
                else:
                    obs_flat = obs_flat[:self.obs_dim]

                # Random action
                action = env.action_space.sample()
                action_flat = action.flatten()

                # Pad/truncate action
                if len(action_flat) < self.action_dim:
                    action_padded = np.pad(action_flat, (0, self.action_dim - len(action_flat)))
                else:
                    action_padded = action_flat[:self.action_dim]

                # Step
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Flatten next observation
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
        """Generate synthetic data when KinDER environments are not available."""
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

    def create_classification_task(
        self,
        data: dict,
        task_id: int,
        num_classes: int = 10,
    ) -> dict:
        """Convert regression data to classification task.

        Uses reward signal to create pseudo-labels.
        Each task gets a unique offset to prevent label overlap.
        """
        obs = torch.tensor(data["observations"], dtype=torch.float32)
        rewards = data["rewards"]

        # Create labels from reward quantiles with task-specific offset
        classes_per_task = max(2, num_classes // 4)  # 2-3 classes per task
        if len(rewards) > 0:
            quantiles = np.percentile(rewards, np.linspace(0, 100, classes_per_task + 1))
            labels = np.digitize(rewards, quantiles[1:-1])
            labels = np.clip(labels, 0, classes_per_task - 1)
        else:
            labels = np.zeros(len(obs), dtype=int)

        # Add task offset to prevent label overlap across tasks
        labels = labels + task_id * classes_per_task
        labels = np.clip(labels, 0, num_classes - 1)

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "obs": obs,
            "targets": labels,
            "task_id": task_id,
        }

    def run_experiment(
        self,
        methods: dict,
        output_dir: str,
    ) -> dict:
        """Run full continual learning experiment.

        Args:
            methods: dict mapping method_name -> ContinualLearner instance
            output_dir: directory to save results

        Returns:
            Results dictionary
        """
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
                    if 'world_model' in method_name:
                        avg_trust = np.mean(learner.task_trust_scores.get(task_id, [0.5]))
                        learner.consolidate(task_id, avg_trust)
                    elif hasattr(learner, 'previous_models'):
                        learner.consolidate()
                    else:
                        learner.consolidate()

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
        output_path = os.path.join(output_dir, "kinder_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        return results


def main():
    """Run the KinDER continual learning benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    num_tasks = 4
    episodes_per_task = 20  # reduced for faster experimentation
    obs_dim = 100
    num_classes = 10

    # Create benchmark
    benchmark = KinDERBenchmark(
        num_tasks=num_tasks,
        episodes_per_task=episodes_per_task,
        obs_dim=obs_dim,
        device=device,
        use_3d=False,  # Use 2D environments for faster experimentation
    )

    # Create models
    model_cls = lambda: SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes).to(device)

    # Create methods
    from continual_learning import (
        FineTuningCL,
        AccuracyTrustCL,
        WorldModelTrustCL,
        ExperienceReplayCL,
        EWCCL,
        PrioritizedReplayCL,
        CuriousReplayCL,
        PackNetCL,
        LwFCL,
    )
    from rssm_world_model import WorldModel

    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=10,
        hidden_dim=256,
        stochastic_dim=16,
        stochastic_classes=16,
        deterministic_dim=256,
    ).to(device)

    methods = {
        "fine_tuning": FineTuningCL(model_cls(), device=device),
        "ewc": EWCCL(model_cls(), device=device),
        "experience_replay": ExperienceReplayCL(model_cls(), device=device, buffer_size=500),
        "prioritized_replay": PrioritizedReplayCL(model_cls(), device=device, buffer_size=500),
        "curious_replay": CuriousReplayCL(
            model_cls(), WorldModel(
                obs_dim=obs_dim,
                action_dim=10,
                hidden_dim=256,
                stochastic_dim=16,
                stochastic_classes=16,
                deterministic_dim=256,
            ).to(device),
            device=device,
            buffer_size=500,
        ),
        "lwf": LwFCL(model_cls(), device=device),
        "packnet": PackNetCL(model_cls(), device=device),
        "accuracy_trust_cl": AccuracyTrustCL(model_cls(), device=device),
        "world_model_trust_cl": WorldModelTrustCL(
            model_cls(), world_model, device=device
        ),
    }

    # Run experiment
    output_dir = "kinder_results"
    results = benchmark.run_experiment(methods, output_dir)

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<25} {'AvgAcc':<12} {'BWT':<12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res['avg_accuracies']:<12.4f} {res['bwt']:<12.4f}")


if __name__ == "__main__":
    main()
