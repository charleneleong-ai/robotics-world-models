"""Sim evaluation worker: runs WAM policy in ManiSkill3 environment.

Runs as a subprocess, queries the Ray Serve policy server over HTTP.
Saves results JSON + trajectory pickles for the close-the-loop filter.

Usage:
    PYTHONPATH=. .venv/bin/python -m experiments.diffusion_wm.eval_worker \\
        --task PegInsertionSide-v1 \\
        --policy-url http://localhost:8000 \\
        --num-episodes 20 \\
        --out eval_results/peginsertion_round1
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import requests
import torch
import typer


def query_policy(policy_url: str, obs: np.ndarray, timeout: float = 30.0) -> np.ndarray:
    """HTTP POST to policy server, return action."""
    resp = requests.post(
        f"{policy_url}/predict",
        json={"obs": obs.tolist()},
        timeout=timeout,
    )
    resp.raise_for_status()
    return np.array(resp.json()["action"], dtype=np.float32)


def query_health(policy_url: str) -> bool:
    """Check if policy server is healthy."""
    try:
        resp = requests.get(f"{policy_url}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _extract_state(obs) -> np.ndarray:
    """Extract flat state vector from ManiSkill3 observation."""
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


def run_eval(
    task: str,
    policy_url: str = "http://localhost:8000",
    num_episodes: int = 20,
    max_steps: int = 200,
    out: Path = Path("eval_results"),
    seed: int = 42,
) -> dict:
    """Run evaluation episodes, return results dict."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if not query_health(policy_url):
        raise ConnectionError(f"Policy server not healthy at {policy_url}")

    env = gym.make(task, num_envs=1, render_mode=None, shader_dir="minimal")
    episodes = []
    total_reward = 0.0
    total_steps = 0
    successes = 0

    print(f"Evaluating {task} on {policy_url} ({num_episodes} episodes)...")
    start_time = time.monotonic()

    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        trajectory = {
            "obs": [],
            "action": [],
            "next_obs": [],
            "reward": [],
            "done": [],
        }
        ep_reward = 0.0

        for step in range(max_steps):
            state = _extract_state(obs)
            action = query_policy(policy_url, state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            trajectory["obs"].append(state)
            trajectory["action"].append(action.flatten() if isinstance(action, np.ndarray) else np.array(action).flatten())
            trajectory["next_obs"].append(_extract_state(next_obs))
            trajectory["reward"].append(float(reward) if not isinstance(reward, torch.Tensor) else float(reward.item()))
            trajectory["done"].append(bool(done) if not isinstance(done, torch.Tensor) else bool(done.item()))

            ep_reward += trajectory["reward"][-1]
            obs = next_obs

            is_done = bool(done) if not isinstance(done, torch.Tensor) else bool(done.item())
            if is_done:
                break

        total_reward += ep_reward
        total_steps += len(trajectory["obs"])
        success = 0.0
        if isinstance(info, dict) and "success" in info:
            s = info["success"]
            success = float(s[0] if isinstance(s, torch.Tensor) else s)
        successes += success

        episodes.append({
            "episode": ep_idx,
            "reward": ep_reward,
            "steps": len(trajectory["obs"]),
            "success": success,
        })

        # Save trajectory pickle
        traj_path = out / f"episode_{ep_idx:04d}.pkl"
        with open(traj_path, "wb") as f:
            pickle.dump(trajectory, f)

    elapsed = time.monotonic() - start_time
    mean_reward = total_reward / num_episodes
    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes

    results = {
        "task": task,
        "policy_url": policy_url,
        "num_episodes": num_episodes,
        "mean_reward": mean_reward,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "total_time": elapsed,
        "fps": total_steps / max(1, elapsed),
        "episodes": episodes,
    }

    results_path = out / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Done: success_rate={success_rate:.3f}, mean_reward={mean_reward:.3f}, {elapsed:.0f}s")

    env.close()
    return results


def main(
    task: str = typer.Option("PegInsertionSide-v1"),
    policy_url: str = typer.Option("http://localhost:8000"),
    num_episodes: int = typer.Option(20),
    max_steps: int = typer.Option(200),
    out: Path = typer.Option(Path("eval_results")),
    seed: int = typer.Option(42),
) -> None:
    run_eval(task, policy_url, num_episodes, max_steps, out, seed)


if __name__ == "__main__":
    typer.run(main)
