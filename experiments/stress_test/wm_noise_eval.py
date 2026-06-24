"""World-model (TD-MPC2) robustness sweep under observation noise.

Replicates the TD-MPC2 `evaluate.py` eval loop, but injects Gaussian noise on
the observation right before `agent.act(...)`:

    obs = obs + sigma_m * torch.randn_like(obs)      # sigma_m = sigma_mm / 1000

Because TD-MPC2 replans reactively each step via MPC, it should degrade
gracefully relative to the open-loop classical planner.

Standalone: loads config.yaml via OmegaConf + applies overrides + calls
parse_cfg, *without* @hydra.main, so it is importable and runnable with
explicit args. Must run from the tdmpc2 baseline dir (imports `common`,
`envs`, `tdmpc2` from there).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra.utils
import torch
import typer
from omegaconf import OmegaConf

TDMPC2_DIR = Path(__file__).resolve().parent
DEFAULT_TDMPC2_DIR = Path("/workspace/ManiSkill/examples/baselines/tdmpc2")


def _resolve_tdmpc2_dir() -> Path:
    """Find the tdmpc2 baseline dir (has config.yaml + common/ + envs/)."""
    for cand in (TDMPC2_DIR, DEFAULT_TDMPC2_DIR):
        if (cand / "config.yaml").exists() and (cand / "common").is_dir():
            return cand
    raise FileNotFoundError(
        "Could not locate tdmpc2 baseline dir (config.yaml + common/). "
        f"Looked in {TDMPC2_DIR} and {DEFAULT_TDMPC2_DIR}."
    )


def _build_cfg(base_dir: Path, overrides: dict) -> OmegaConf:
    cfg = OmegaConf.load(base_dir / "config.yaml")
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    # parse_cfg builds work_dir via hydra.utils.get_original_cwd(); we run
    # outside @hydra.main, so pin it to base_dir.
    hydra.utils.get_original_cwd = lambda: str(base_dir)  # type: ignore[assignment]
    from common.parser import parse_cfg

    return parse_cfg(cfg)


def _eval_sigma(agent, env, cfg, sigma_mm: float) -> float:
    sigma_m = sigma_mm / 1000.0
    device = "cuda" if cfg.env_type == "gpu" else "cpu"
    successes = []
    for _ in range(cfg.eval_episodes_per_env):
        obs, _ = env.reset()
        done = torch.full((cfg.num_eval_envs,), False, device=device)
        t = 0
        info = {}
        while not done[0]:
            noisy_obs = obs + sigma_m * torch.randn_like(obs) if sigma_m > 0 else obs
            action = agent.act(noisy_obs, t0=t == 0, eval_mode=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            t += 1
        if "success" in info:
            successes.append(info["final_info"]["success"].float().mean().item())
    return float(sum(successes) / len(successes)) if successes else 0.0


def main(
    checkpoint: Path = typer.Option(..., help="Path to TD-MPC2 .pt checkpoint."),
    sigmas_mm: str = typer.Option("0,5,10,15,20", help="Comma-separated noise sigmas (mm)."),
    eval_episodes: int = typer.Option(25, help="Eval episodes per env, per sigma."),
    out: Path = typer.Option(Path("wm_results.json"), help="Output JSON path."),
    env_id: str = typer.Option("PegInsertionSide-v1"),
    model_size: int = typer.Option(5),
    obs: str = typer.Option("state"),
    control_mode: str = typer.Option("pd_joint_delta_pos"),
    num_eval_envs: int = typer.Option(4),  # ManiSkillVectorEnv action_space is 1-D at num_envs=1
    env_type: str = typer.Option("gpu"),
    seed: int = typer.Option(1),
) -> None:
    base_dir = _resolve_tdmpc2_dir()
    sys.path.insert(0, str(base_dir))

    overrides = {
        "env_id": env_id,
        "model_size": model_size,
        "obs": obs,
        "control_mode": control_mode,
        "num_eval_envs": num_eval_envs,
        "eval_episodes_per_env": eval_episodes,
        "env_type": env_type,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "save_video_local": False,
    }
    cfg = _build_cfg(base_dir, overrides)

    from common.seed import set_seed
    from envs import make_envs
    from tdmpc2 import TDMPC2

    assert torch.cuda.is_available(), "CUDA required for TD-MPC2 eval."
    set_seed(cfg.seed)

    env = make_envs(cfg, cfg.num_eval_envs, is_eval=True)
    agent = TDMPC2(cfg)
    assert checkpoint.exists(), f"Checkpoint {checkpoint} not found."
    agent.load(str(checkpoint))

    sigmas = [float(s) for s in sigmas_mm.split(",")]
    results = []
    for sigma_mm in sigmas:
        rate = _eval_sigma(agent, env, cfg, sigma_mm)
        n = cfg.eval_episodes_per_env * cfg.num_eval_envs
        print(f"=== sigma={sigma_mm}mm success_rate={rate:.3f} (n={n}) ===", flush=True)
        results.append({"sigma_mm": sigma_mm, "n": n, "success_rate": rate})

    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    typer.run(main)
