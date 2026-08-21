"""Goal-Direction Task Design for CL Benchmarks.

Creates binary classification tasks from environment transitions:
- Label 1: agent moved toward goal (reward > 0)
- Label 0: agent moved away from goal (reward <= 0)

This creates a meaningful, shared task across different environments
while preserving genuine cross-task interference.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Protocol, Any


class EnvCollector(Protocol):
    """Protocol for environment data collection."""
    def reset(self) -> tuple[Any, dict]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]: ...
    @property
    def action_space(self) -> Any: ...


def collect_transitions(
    env: EnvCollector,
    n_episodes: int,
    max_steps: int,
    obs_dim: int,
    action_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """Collect transitions from an environment.
    
    Returns dict with keys:
        observations: (N, obs_dim)
        actions: (N, act_dim) 
        next_observations: (N, obs_dim)
        rewards: (N,)
        dones: (N,)
    """
    obs_list, act_list, next_list, rew_list, done_list = [], [], [], [], []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            obs_flat = _flatten(obs, obs_dim)
            action = env.action_space.sample()
            act_flat = _flatten(action, action_dim or len(np.atleast_1d(action)))
            
            next_obs, reward, term, trunc, info = env.step(action)
            next_flat = _flatten(next_obs, obs_dim)
            
            obs_list.append(obs_flat)
            act_list.append(act_flat)
            next_list.append(next_flat)
            rew_list.append(float(reward))
            done_list.append(float(term or trunc))
            
            obs = next_obs
            if term or trunc:
                break
    
    return {
        "observations": np.array(obs_list, np.float32),
        "actions": np.array(act_list, np.float32),
        "next_observations": np.array(next_list, np.float32),
        "rewards": np.array(rew_list, np.float32),
        "dones": np.array(done_list, np.float32),
    }


def compute_goal_direction_labels(rewards: np.ndarray) -> np.ndarray:
    """Compute binary labels from rewards.
    
    Label 1: reward > 0 (moving toward goal)
    Label 0: reward <= 0 (not moving toward goal)
    """
    return (rewards > 0).astype(np.int64)


def make_goal_direction_dataset(
    transitions: dict[str, np.ndarray],
    obs_dim: int,
) -> dict[str, torch.Tensor]:
    """Convert transitions to goal-direction classification dataset.
    
    Returns dict with:
        obs: (N, obs_dim) observations
        labels: (N,) binary labels (1=toward goal, 0=away)
    """
    labels = compute_goal_direction_labels(transitions["rewards"])
    
    return {
        "obs": torch.tensor(transitions["observations"], dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _flatten(x: Any, target_dim: int) -> np.ndarray:
    """Flatten and pad/truncate to target dimension."""
    if isinstance(x, dict):
        flat = np.concatenate([v.flatten() for v in x.values() if isinstance(v, np.ndarray)])
    else:
        flat = np.asarray(x, np.float32).flat[:]
    
    if len(flat) < target_dim:
        flat = np.pad(flat, (0, target_dim - len(flat)))
    
    return flat[:target_dim].astype(np.float32)


def collect_kinder_env(
    env_name: str,
    n_episodes: int = 30,
    max_steps: int = 100,
    obs_dim: int = 100,
) -> dict[str, np.ndarray] | None:
    """Collect data from a KinDER environment."""
    try:
        import kinder
        env = kinder.make(env_name)
    except Exception as e:
        print(f"  WARN: {env_name} failed: {e}")
        return None
    
    data = collect_transitions(env, n_episodes, max_steps, obs_dim)
    env.close()
    return data


def collect_maniskill_env(
    env_name: str,
    n_episodes: int = 30,
    max_steps: int = 100,
    obs_dim: int = 100,
) -> dict[str, np.ndarray] | None:
    """Collect data from a ManiSkill environment."""
    try:
        import mani_skill.envs
        env = mani_skill.envs.make(env_name)
    except Exception as e:
        print(f"  WARN: {env_name} failed: {e}")
        return None
    
    data = collect_transitions(env, n_episodes, max_steps, obs_dim)
    env.close()
    return data
