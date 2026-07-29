"""Rollout collector: run trained policies and save (obs, action, next_obs) shards.

Usage:
    python -m experiments.diffusion_wm.collect \\
        --checkpoint /path/to/tdmpc2.pt \\
        --env_id PegInsertionSide-v1 \\
        --num_episodes 2000 \\
        --out data/peginsertion

Collects from TD-MPC2 / PPO / SAC checkpoints by default (via ManiSkill's eval
entry point). Saves shards as ``{out}/shard_{i:05d}.npz`` with fields:
    obs, action, next_obs, reward, done
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra.utils
import numpy as np
import torch
import typer
from omegaconf import OmegaConf

NPPZ = np.lib.format.open_memmap  # for large shards

DEFAULT_TDMPC2_DIR = Path("/workspace/ManiSkill/examples/baselines/tdmpc2")


def _resolve_tdmpc2_dir() -> Path:
    for cand in (Path.cwd(), DEFAULT_TDMPC2_DIR):
        if (cand / "config.yaml").exists() and (cand / "common").is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not locate tdmpc2 baseline dir. Looked in {Path.cwd()} and {DEFAULT_TDMPC2_DIR}."
    )


def _build_cfg(base_dir: Path, overrides: dict) -> OmegaConf:
    cfg = OmegaConf.load(base_dir / "config.yaml")
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    hydra.utils.get_original_cwd = lambda: str(base_dir)
    from common.parser import parse_cfg
    return parse_cfg(cfg)


def main(
    checkpoint: Path = typer.Option(..., help="Trained policy checkpoint (.pt)."),
    out: Path = typer.Option(Path("data/diffusion_wm"), help="Output directory for shards."),
    env_id: str = typer.Option("PegInsertionSide-v1"),
    num_episodes: int = typer.Option(2000, help="Total episodes to collect."),
    max_steps: int = typer.Option(200, help="Max steps per episode."),
    policy_type: str = typer.Option("tdmpc2", help="Policy type: tdmpc2, ppo, sac."),
    model_size: int = typer.Option(5),
    obs: str = typer.Option("state"),
    control_mode: str = typer.Option("pd_joint_delta_pos"),
    num_envs: int = typer.Option(32, help="Parallel envs for fast collection."),
    seed: int = typer.Option(42),
    shard_size: int = typer.Option(50_000, help="Transitions per shard file."),
):
    """Collect (obs, action, next_obs) transitions from a trained policy."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    assert torch.cuda.is_available(), "CUDA required for fast sim."

    # --- Build cfg & env ---
    base_dir = _resolve_tdmpc2_dir()
    sys.path.insert(0, str(base_dir))

    overrides = {
        "env_id": env_id,
        "model_size": model_size,
        "obs": obs,
        "control_mode": control_mode,
        "num_envs": num_envs,
        "num_eval_envs": num_envs,
        "eval_episodes_per_env": 1,
        "env_type": "gpu",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "save_video_local": False,
    }
    cfg = _build_cfg(base_dir, overrides)

    from common.seed import set_seed
    from envs import make_envs
    from tdmpc2 import TDMPC2

    set_seed(cfg.seed)
    env = make_envs(cfg, cfg.num_envs, is_eval=True)
    agent = TDMPC2(cfg)
    assert checkpoint.exists(), f"Checkpoint {checkpoint} not found."
    agent.load(str(checkpoint))
    device = "cuda" if cfg.env_type == "gpu" else "cpu"

    # --- Collection loop ---
    obs_buf, _ = env.reset()
    episodes_collected = 0
    transitions = []
    step_count = 0
    shard_idx = 0
    timers = {"reset": 0.0, "act": 0.0, "step": 0.0}
    episode_steps = np.zeros(cfg.num_envs, dtype=np.int32)
    episode_reward = np.zeros(cfg.num_envs, dtype=np.float32)

    print(f"Collecting {num_episodes} episodes from {env_id} ({policy_type})...")
    start_time = time.monotonic()
    pbar_step = 0
    log_interval = 1000

    while episodes_collected < num_episodes:
        # Act
        t0 = time.monotonic()
        action = agent.act(obs_buf, t0=False, eval_mode=True)
        timers["act"] += time.monotonic() - t0

        # Step
        t0 = time.monotonic()
        next_obs, reward, terminated, truncated, info = env.step(action)
        timers["step"] += time.monotonic() - t0
        done = terminated | truncated

        # Store transitions
        for i in range(cfg.num_envs):
            transitions.append({
                "obs": obs_buf[i].cpu().numpy().astype(np.float32),
                "action": action[i].cpu().numpy().astype(np.float32),
                "next_obs": next_obs[i].cpu().numpy().astype(np.float32),
                "reward": float(reward[i].cpu().item()),
                "done": bool(done[i].cpu().item()),
            })
            episode_steps[i] += 1
            episode_reward[i] += float(reward[i].cpu().item())

        step_count += 1
        pbar_step += cfg.num_envs

        # Handle done episodes (reset + count)
        if done.any():
            t0 = time.monotonic()
            for i in np.where(done.cpu().numpy())[0]:
                episodes_collected += 1
                episode_steps[i] = 0
                episode_reward[i] = 0.0
            obs_buf, _ = env.reset()
            timers["reset"] += time.monotonic() - t0
        else:
            obs_buf = next_obs

        # Write shard when buffer is full
        if len(transitions) >= shard_size:
            _write_shard(transitions, out, shard_idx)
            shard_idx += 1
            transitions = []

        # Log
        if pbar_step >= log_interval * cfg.num_envs:
            elapsed = time.monotonic() - start_time
            fps = pbar_step / elapsed if elapsed > 0 else 0
            print(
                f"  collected {episodes_collected}/{num_episodes} eps, "
                f"{pbar_step} steps, {fps:.0f} fps, "
                f"act={timers['act']/elapsed*100:.1f}% step={timers['step']/elapsed*100:.1f}%"
            )
            pbar_step = 0

    # Flush remaining
    if transitions:
        _write_shard(transitions, out, shard_idx)
        shard_idx += 1

    elapsed = time.monotonic() - start_time
    total_steps = step_count * cfg.num_envs
    print(
        f"\nDone: {episodes_collected} episodes, {total_steps} transitions, "
        f"{elapsed:.0f}s ({total_steps/elapsed:.0f} fps)"
    )

    # Write meta
    obs_dim = len(transitions[0]["obs"]) if transitions else 0
    act_dim = len(transitions[0]["action"]) if transitions else 0
    meta = {
        "env_id": env_id,
        "policy_type": policy_type,
        "num_episodes": episodes_collected,
        "num_transitions": total_steps,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "obs_mode": obs,
        "control_mode": control_mode,
        "num_shards": shard_idx,
        "shard_size": shard_size,
        "fps": total_steps / elapsed,
    }
    (out / "meta/collection.json").write_text(json.dumps(meta, indent=2))
    print(f"Meta: {out / 'meta/collection.json'}")

    env.close()


def _write_shard(transitions: list[dict], out_dir: Path, idx: int) -> None:
    """Write a shard of transitions to disk as compressed npz."""
    obs = np.stack([t["obs"] for t in transitions])
    action = np.stack([t["action"] for t in transitions])
    next_obs = np.stack([t["next_obs"] for t in transitions])
    reward = np.array([t["reward"] for t in transitions], dtype=np.float32)
    done = np.array([t["done"] for t in transitions], dtype=bool)
    path = out_dir / f"shard_{idx:05d}.npz"
    np.savez_compressed(path, obs=obs, action=action, next_obs=next_obs, reward=reward, done=done)
    print(f"  wrote {path} ({len(transitions)} transitions)")


if __name__ == "__main__":
    typer.run(main)
