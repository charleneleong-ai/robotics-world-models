"""ManiSkill3 data collector: collects rollout episodes as sharded .npz files.

Supports random policies (initial data) and learned policy functions.
Outputs the same format as collect.py for compatibility with TransitionDataset.

Usage:
    PYTHONPATH=. .venv/bin/python -m experiments.diffusion_wm.collector \\
        --task PegInsertionSide-v1 \\
        --num-episodes 100 \\
        --out data/peginsertion_round1
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import typer


def _get_env_dims(task: str) -> tuple[int, int]:
    """Get obs/action dims from ManiSkill3 task (requires mani_skill installed)."""
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        env = gym.make(task, num_envs=1, render_mode=None)
        obs, _ = env.reset()
        obs_dim = obs["agent"]["qpos"].shape[-1] + obs["agent"]["qvel"].shape[-1]
        act_dim = env.action_space.shape[-1]
        env.close()
        return obs_dim, act_dim
    except ImportError:
        raise ImportError("mani_skill not installed. Install with: pip install mani_skill")


class ManiSkillCollector:
    """Collect rollout data from ManiSkill3 environments."""

    def __init__(
        self,
        task: str,
        num_envs: int = 1,
        max_steps: int = 200,
        seed: int = 42,
    ):
        self.task = task
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.seed = seed
        self._env = None

    def _make_env(self):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        return gym.make(
            self.task,
            num_envs=self.num_envs,
            render_mode=None,
            shader_dir="minimal",
        )

    def _ensure_env(self):
        if self._env is None:
            self._env = self._make_env()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def collect_episode(self, policy_fn: Callable | None = None) -> dict:
        """Run one episode, return trajectory dict.

        Args:
            policy_fn: callable(obs) -> action. If None, uses random actions.

        Returns:
            dict with keys: obs, action, next_obs, reward, done, success
        """
        self._ensure_env()
        obs, _ = self._env.reset(seed=self.seed)
        obs_list, action_list, next_obs_list, reward_list = [], [], [], []

        for _ in range(self.max_steps):
            if policy_fn is not None:
                action = policy_fn(obs)
            else:
                action = self._env.action_space.sample()

            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated | truncated

            obs_list.append(_extract_state(obs))
            action_list.append(action[0] if self.num_envs > 1 else action)
            next_obs_list.append(_extract_state(next_obs))
            reward_list.append(float(reward[0] if self.num_envs > 1 else reward))

            if done.any() if self.num_envs > 1 else done:
                break

            obs = next_obs

        return {
            "obs": np.stack(obs_list),
            "action": np.stack(action_list),
            "next_obs": np.stack(next_obs_list),
            "reward": np.array(reward_list, dtype=np.float32),
            "success": float(info.get("success", 0)),
        }

    def collect_dataset(
        self,
        num_episodes: int,
        policy_fn: Callable | None = None,
        out: Path = Path("data/diffusion_wm"),
        shard_size: int = 50_000,
    ) -> Path:
        """Collect multiple episodes, save as sharded .npz files.

        Returns path to dataset directory.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(exist_ok=True)

        all_obs, all_action, all_next_obs, all_reward = [], [], [], []
        episodes_collected = 0
        shard_idx = 0
        start_time = time.monotonic()

        print(f"Collecting {num_episodes} episodes from {self.task}...")
        while episodes_collected < num_episodes:
            ep = self.collect_episode(policy_fn)
            all_obs.append(ep["obs"])
            all_action.append(ep["action"])
            all_next_obs.append(ep["next_obs"])
            all_reward.append(ep["reward"])
            episodes_collected += 1

            total_transitions = sum(len(o) for o in all_obs)
            if total_transitions >= shard_size:
                self._write_shard(all_obs, all_action, all_next_obs, all_reward, out, shard_idx)
                shard_idx += 1
                all_obs, all_action, all_next_obs, all_reward = [], [], [], []

            if episodes_collected % 10 == 0:
                elapsed = time.monotonic() - start_time
                fps = total_transitions / max(1, elapsed)
                print(f"  {episodes_collected}/{num_episodes} episodes, {total_transitions} steps, {fps:.0f} steps/s")

        if all_obs:
            self._write_shard(all_obs, all_action, all_next_obs, all_reward, out, shard_idx)
            shard_idx += 1

        elapsed = time.monotonic() - start_time
        total_transitions = sum(
            len(np.load(out / f"shard_{i:05d}.npz")["obs"])
            for i in range(shard_idx)
        )
        obs_dim, act_dim = self._get_dims(out)

        meta = {
            "task": self.task,
            "policy_type": "random" if policy_fn is None else "learned",
            "num_episodes": episodes_collected,
            "num_transitions": total_transitions,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "num_shards": shard_idx,
            "shard_size": shard_size,
        }
        (out / "meta/collection.json").write_text(json.dumps(meta, indent=2))
        print(f"Done: {episodes_collected} episodes, {total_transitions} transitions in {elapsed:.0f}s")
        return out

    def _write_shard(
        self,
        obs_list: list[np.ndarray],
        action_list: list[np.ndarray],
        next_obs_list: list[np.ndarray],
        reward_list: list[np.ndarray],
        out_dir: Path,
        idx: int,
    ) -> None:
        obs = np.concatenate(obs_list)
        action = np.concatenate(action_list)
        next_obs = np.concatenate(next_obs_list)
        reward = np.concatenate(reward_list)
        path = out_dir / f"shard_{idx:05d}.npz"
        np.savez_compressed(path, obs=obs, action=action, next_obs=next_obs, reward=reward)

    def _get_dims(self, out_dir: Path) -> tuple[int, int]:
        shards = sorted(out_dir.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"No shards in {out_dir}")
        with np.load(shards[-1]) as data:
            return data["obs"].shape[1], data["action"].shape[1]


def _extract_state(obs: dict) -> np.ndarray:
    """Extract flat state vector from ManiSkill3 observation dict."""
    parts = []
    if "agent" in obs:
        for key in ("qpos", "qvel"):
            if key in obs["agent"]:
                parts.append(obs["agent"][key].flatten())
    if "extra" in obs:
        for key in ("tcp_pose", "goal"):
            if key in obs["extra"]:
                parts.append(obs["extra"][key].flatten())
    if parts:
        return np.concatenate(parts).astype(np.float32)
    # Fallback: flatten everything
    return _flatten_dict(obs).astype(np.float32)


def _flatten_dict(d: dict, prefix: str = "") -> np.ndarray:
    parts = []
    for k, v in sorted(d.items()):
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            parts.append(_flatten_dict(v, full_key))
        elif hasattr(v, "numpy"):
            parts.append(v.numpy().flatten())
        elif hasattr(v, "flatten"):
            parts.append(v.flatten())
    return np.concatenate(parts) if parts else np.array([])


def main(
    task: str = typer.Option("PegInsertionSide-v1", help="ManiSkill3 task"),
    num_episodes: int = typer.Option(100, help="Episodes to collect"),
    max_steps: int = typer.Option(200, help="Max steps per episode"),
    out: Path = typer.Option(Path("data/diffusion_wm"), help="Output directory"),
    num_envs: int = typer.Option(1, help="Parallel envs"),
    shard_size: int = typer.Option(50_000),
    seed: int = typer.Option(42),
) -> None:
    collector = ManiSkillCollector(task, num_envs=num_envs, max_steps=max_steps, seed=seed)
    collector.collect_dataset(num_episodes, policy_fn=None, out=out, shard_size=shard_size)
    collector.close()


if __name__ == "__main__":
    typer.run(main)
