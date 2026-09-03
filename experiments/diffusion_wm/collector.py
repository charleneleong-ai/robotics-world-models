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
import torch
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

            # Convert tensors to numpy for stepping
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = obs
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = action

            next_obs, reward, terminated, truncated, info = self._env.step(action_np)
            done = terminated | truncated

            # Handle scalar vs batched
            if self.num_envs > 1:
                is_done = bool(done.any()) if isinstance(done, torch.Tensor) else bool(done)
                r = float(reward[0]) if isinstance(reward, torch.Tensor) else float(reward)
            else:
                is_done = bool(done) if not isinstance(done, torch.Tensor) else bool(done.item())
                r = float(reward) if not isinstance(reward, torch.Tensor) else float(reward.item())

            obs_list.append(_extract_state(obs_np))
            action_list.append(action_np.flatten() if isinstance(action_np, np.ndarray) else np.array(action_np).flatten())
            next_obs_list.append(_extract_state(next_obs))
            reward_list.append(r)

            if is_done:
                break

            obs = next_obs

        success = 0.0
        if isinstance(info, dict) and "success" in info:
            s = info["success"]
            success = float(s[0] if isinstance(s, torch.Tensor) else s)

        return {
            "obs": np.stack(obs_list),
            "action": np.stack(action_list),
            "next_obs": np.stack(next_obs_list),
            "reward": np.array(reward_list, dtype=np.float32),
            "success": success,
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


def _extract_state(obs) -> np.ndarray:
    """Extract flat state vector from ManiSkill3 observation.

    ManiSkill3 v3.0+ returns a flat torch.Tensor directly.
    Older versions return a nested dict with 'agent', 'extra' keys.
    """
    if isinstance(obs, torch.Tensor):
        return obs.cpu().numpy().flatten().astype(np.float32)
    if isinstance(obs, np.ndarray):
        return obs.flatten().astype(np.float32)
    # Legacy dict format
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


def load_demonstration_data(
    task: str,
    demo_dir: Path,
    max_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 42,
) -> Path:
    """Load ManiSkill demonstration data by replaying expert actions.

    Downloads demos if not present, replays actions in the environment,
    and saves transitions in the standard format for TransitionDataset.

    Args:
        task: ManiSkill3 task ID (e.g., 'PickCube-v1')
        demo_dir: Directory containing downloaded demo H5 files
        max_episodes: Maximum number of episodes to load
        max_steps: Maximum steps per episode (truncates longer demos)
        seed: Random seed for environment reset

    Returns:
        Path to the saved dataset directory
    """
    import h5py
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    # Find the demo H5 file
    demo_path = demo_dir / task
    h5_files = list(demo_path.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No demo H5 files found in {demo_path}")

    # Prefer RL demos with pd_joint_delta_pos action space (8-dim, matches ManiSkill default)
    rl_files = [f for f in h5_files if "rl" in str(f) and "pd_joint_delta_pos" in str(f)]
    if not rl_files:
        rl_files = [f for f in h5_files if "rl" in str(f)]
    h5_file = rl_files[0] if rl_files else h5_files[0]
    print(f"Loading demos from: {h5_file}")

    # Open H5 file and get trajectory keys
    h5 = h5py.File(h5_file, "r")
    traj_keys = sorted(
        [k for k in h5.keys() if k.startswith("traj_")],
        key=lambda x: int(x.split("_")[1]),
    )
    traj_keys = traj_keys[:max_episodes]
    print(f"Found {len(traj_keys)} trajectories")

    # Create environment
    env = gym.make(task, num_envs=1, obs_mode="state", render_mode=None)

    # Create output directory
    out_dir = demo_path / "replayed"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_count = 0
    total_transitions = 0

    for traj_key in traj_keys:
        traj = h5[traj_key]
        actions = np.array(traj["actions"])  # (T, act_dim)
        success = np.array(traj["success"])

        # Skip failed demonstrations (check if any step succeeded)
        if not success.any():
            continue

        # Truncate to max_steps
        actions = actions[:max_steps]

        # Reset environment with different seed per episode
        obs, _ = env.reset(seed=seed + episode_count)
        obs = _extract_state(obs)

        # Replay actions and record transitions
        states, actss, next_states, rewards, dones = [], [], [], [], []

        for i in range(len(actions)):
            action = actions[i : i + 1]  # Keep batch dim
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_np = _extract_state(next_obs)

            states.append(obs)
            actss.append(action.flatten())
            next_states.append(next_obs_np)
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))

            obs = next_obs_np
            if terminated or truncated:
                break

        # Save as NPZ
        n = len(states)
        if n > 0:
            np.savez(
                out_dir / f"episode_{episode_count:05d}.npz",
                states=np.array(states),
                actions=np.array(actss),
                next_states=np.array(next_states),
                rewards=np.array(rewards),
                dones=np.array(dones),
            )
            total_transitions += n
            episode_count += 1

        if episode_count >= max_episodes:
            break

    h5.close()
    env.close()

    print(f"Loaded {episode_count} episodes, {total_transitions} transitions -> {out_dir}")
    return out_dir


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
