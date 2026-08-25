#!/usr/bin/env python3
"""Runner script for ManiSkill continual learning experiments.

Usage:
    python run_maniskill.py [--num-tasks 4] [--episodes 50] [--output-dir results]
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Add experiment directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rssm_world_model import WorldModel
from continual_learning import (
    FineTuningCL,
    AccuracyTrustCL,
    WorldModelTrustCL,
    ExperienceReplayCL,
    EWCCL,
)
from maniskill_benchmark import ManiSkillBenchmark, SimpleMLP


def test_maniskill_envs():
    """Test which ManiSkill environments work."""
    import mani_skill.envs
    import gymnasium as gym

    test_envs = [
        "PickCube-v1",
        "PushCube-v1",
        "LiftPegUpright-v1",
        "PlugCharger-v1",
        "StackCube-v1",
        "PokeCube-v1",
        "PullCube-v1",
        "RotateCube-v1",
    ]

    working_envs = []
    for env_name in test_envs:
        try:
            env = gym.make(
                env_name,
                render_mode=None,
                enable_shadow=False,
                shader_dir="minimal",
            )
            obs, info = env.reset()
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            working_envs.append(env_name)
            print(f"  ✓ {env_name} works (obs shape: {np.array(obs).shape if not isinstance(obs, dict) else 'dict'})")
        except Exception as e:
            print(f"  ✗ {env_name} failed: {e}")

    return working_envs


def run_experiment(
    num_tasks: int = 7,
    episodes: int = 30,
    output_dir: str = "maniskill_results",
):
    """Run the full experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # First test environments
    print("\nTesting ManiSkill environments...")
    working_envs = test_maniskill_envs()

    if len(working_envs) < num_tasks:
        print(f"\nWarning: Only {len(working_envs)} environments work, reducing num_tasks")
        num_tasks = len(working_envs)

    if num_tasks == 0:
        print("No environments work, using synthetic data")
        num_tasks = 4

    # Create benchmark
    obs_dim = 64  # padded to max env obs size
    action_dim = 10  # padded to max action size
    num_classes = 10

    benchmark = ManiSkillBenchmark(
        num_tasks=num_tasks,
        episodes_per_task=episodes,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    # Override selected envs if we have working ones
    if working_envs:
        benchmark.selected_envs = working_envs[:num_tasks]

    # Create models
    model_cls = lambda: SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes).to(device)

    # Create world model
    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        stochastic_dim=16,
        stochastic_classes=16,
        deterministic_dim=256,
    ).to(device)

    # Create methods
    methods = {
        "fine_tuning": FineTuningCL(model_cls(), device=device),
        "ewc": EWCCL(model_cls(), device=device, ewc_lambda=5000.0),
        "experience_replay": ExperienceReplayCL(
            model_cls(), device=device, buffer_size=500, replay_ratio=0.5
        ),
        "accuracy_trust_cl": AccuracyTrustCL(
            model_cls(), device=device, trust_threshold=0.7
        ),
        "world_model_trust_cl": WorldModelTrustCL(
            model_cls(), world_model, device=device,
            ewc_lambda=5000.0, trust_threshold=0.5,
        ),
    }

    # Run experiment
    os.makedirs(output_dir, exist_ok=True)
    results = benchmark.run_experiment(methods, output_dir)

    # Print comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"{'Method':<25} {'AvgAcc':<12} {'BWT':<12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res['avg_accuracies']:<12.4f} {res['bwt']:<12.4f}")

    # Save summary
    summary = {
        "num_tasks": num_tasks,
        "episodes_per_task": episodes,
        "environments": benchmark.selected_envs,
        "results": {
            name: {
                "avg_accuracy": res["avg_accuracies"],
                "bwt": res["bwt"],
                "task_accuracies": res["task_accuracies"],
            }
            for name, res in results.items()
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="ManiSkill CL Benchmark")
    parser.add_argument("--num-tasks", type=int, default=4, help="Number of tasks")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task")
    parser.add_argument("--output-dir", type=str, default="maniskill_results", help="Output directory")
    parser.add_argument("--test-envs", action="store_true", help="Test environments only")
    args = parser.parse_args()

    if args.test_envs:
        print("Testing ManiSkill environments...")
        working = test_maniskill_envs()
        print(f"\nWorking environments: {working}")
        return

    run_experiment(
        num_tasks=args.num_tasks,
        episodes=args.episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
