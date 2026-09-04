"""Config-driven experiment runner."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from continualwam import (
    BACKBONES,
    TRUST_METHODS,
    eval_bc,
    get_backbone,
    load_demos,
    make_trust,
    train_bc,
    train_bc_trust,
    train_wm,
)
from continualwam.configs import SweepConfig


def run_sweep(cfg: SweepConfig) -> dict:
    """Run backbone × trust sweep."""
    print(f"\n{'='*60}")
    print(f"  Sweep: {len(cfg.backbones)} backbones × {len(cfg.trust_methods)} trusts")
    print(f"  Suite: {cfg.suite_dir}")
    print(f"  Seeds: {cfg.n_seeds}")
    print(f"{'='*60}")

    demos = load_demos(cfg.suite_dir, cfg.n_tasks, cfg.max_demos)
    print(f"Loaded {len(demos)} tasks")

    obs_dim, act_dim = demos[0][0]["obs"].shape[-1], demos[0][0]["acts"].shape[-1]
    results = {b: {t: [] for t in ["none", *cfg.trust_methods]} for b in cfg.backbones}

    for bone_name in cfg.backbones:
        for seed in range(cfg.n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()

            # Train world model
            wm = get_backbone(bone_name, obs_dim, act_dim).to(cfg.device)
            train_wm(wm, demos[0], cfg.wm_epochs, device=cfg.device)

            # No-trust BC (control)
            std_policy = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, act_dim),
            ).to(cfg.device)
            train_bc(std_policy, demos[0], cfg.bc_epochs, cfg.batch_size, device=cfg.device)
            test_demos = demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]
            results[bone_name]["none"].append(eval_bc(std_policy, test_demos, cfg.device))

            # Trust-weighted BC
            for tm_name in cfg.trust_methods:
                trust = make_trust(tm_name, obs_dim, act_dim)
                tru_policy = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                    torch.nn.Linear(128, 128), torch.nn.ReLU(),
                    torch.nn.Linear(128, act_dim),
                ).to(cfg.device)
                train_bc_trust(tru_policy, wm, trust, demos[0], cfg.bc_epochs, cfg.batch_size, device=cfg.device)
                results[bone_name][tm_name].append(eval_bc(tru_policy, test_demos, cfg.device))

            elapsed = time.time() - t0
            vals = " ".join(f"{m}={results[bone_name][m][-1]:.4f}" for m in results[bone_name])
            print(f"  {bone_name} s{seed}: {vals} ({elapsed:.0f}s)")

    # Compute stats
    stats = {}
    for bone in results:
        stats[bone] = {}
        for method in results[bone]:
            arr = np.array(results[bone][method])
            stats[bone][method] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci95": float(1.96 * arr.std() / np.sqrt(cfg.n_seeds)),
                "seeds": results[bone][method],
            }

    # Save
    out_path = Path(cfg.output_dir) / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ContinualWAM experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--suite", choices=["spatial", "object", "goal"], help="Override suite")
    parser.add_argument("--backbones", nargs="+", help="Override backbones")
    parser.add_argument("--trusts", nargs="+", help="Override trust methods")
    parser.add_argument("--seeds", type=int, help="Override n_seeds")
    args = parser.parse_args()

    cfg = SweepConfig.from_yaml(args.config)
    if args.suite:
        suite_map = {
            "spatial": "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial",
            "object": "/home/ubuntu/robotics_world_models/LIBERO/libero_object",
            "goal": "/home/ubuntu/robotics_world_models/LIBERO/libero_goal",
        }
        cfg.suite_dir = suite_map[args.suite]
    if args.backbones:
        cfg.backbones = args.backbones
    if args.trusts:
        cfg.trust_methods = args.trusts
    if args.seeds:
        cfg.n_seeds = args.seeds

    run_sweep(cfg)


if __name__ == "__main__":
    main()
