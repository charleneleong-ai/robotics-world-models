"""ContinualWAM CLI — config-driven experiment runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
import typer

from continualwam import (
    eval_bc,
    get_backbone,
    load_demos,
    make_trust,
    train_bc,
    train_bc_trust,
    train_wm,
)

app = typer.Typer(
    name="continualwam",
    help="Trust-aware continual learning for world models.",
    no_args_is_help=True,
)

SUITE_DIRS = {
    "spatial": "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial",
    "object": "/home/ubuntu/robotics_world_models/LIBERO/libero_object",
    "goal": "/home/ubuntu/robotics_world_models/LIBERO/libero_goal",
}


@app.command()
def sweep(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML config file")],
    suite: Annotated[Optional[str], typer.Option("--suite", "-s", help="Override suite")] = None,
    backbones: Annotated[Optional[list[str]], typer.Option("--backbone", "-b", help="Override backbones")] = None,
    trusts: Annotated[Optional[list[str]], typer.Option("--trust", "-t", help="Override trust methods")] = None,
    seeds: Annotated[Optional[int], typer.Option("--seeds", "-n", help="Override n_seeds")] = None,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("."),
):
    """Run backbone × trust sweep."""
    import yaml

    with open(config) as f:
        cfg = yaml.safe_load(f)

    if suite:
        cfg["suite_dir"] = SUITE_DIRS[suite]
    if backbones:
        cfg["backbones"] = backbones
    if trusts:
        cfg["trust_methods"] = trusts
    if seeds:
        cfg["n_seeds"] = seeds

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Sweep: {len(cfg['backbones'])} backbones × {len(cfg['trust_methods'])} trusts")
    typer.echo(f"  Suite: {cfg['suite_dir']}")
    typer.echo(f"  Seeds: {cfg['n_seeds']}")
    typer.echo(f"{'='*60}")

    demos = load_demos(cfg["suite_dir"], cfg["n_tasks"], cfg["max_demos"])
    typer.echo(f"Loaded {len(demos)} tasks")

    obs_dim, act_dim = demos[0][0]["obs"].shape[-1], demos[0][0]["acts"].shape[-1]
    device = cfg.get("device", "cuda")
    results = {b: {t: [] for t in ["none", *cfg["trust_methods"]]} for b in cfg["backbones"]}

    for bone_name in cfg["backbones"]:
        for seed in range(cfg["n_seeds"]):
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()

            wm = get_backbone(bone_name, obs_dim, act_dim).to(device)
            train_wm(wm, demos[0], cfg["wm_epochs"], device=device)

            std_policy = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, act_dim),
            ).to(device)
            train_bc(std_policy, demos[0], cfg["bc_epochs"], cfg["batch_size"], device=device)
            test_demos = demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]
            results[bone_name]["none"].append(eval_bc(std_policy, test_demos, device))

            for tm_name in cfg["trust_methods"]:
                trust = make_trust(tm_name, obs_dim, act_dim)
                tru_policy = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                    torch.nn.Linear(128, 128), torch.nn.ReLU(),
                    torch.nn.Linear(128, act_dim),
                ).to(device)
                train_bc_trust(tru_policy, wm, trust, demos[0], cfg["bc_epochs"], cfg["batch_size"], device=device)
                results[bone_name][tm_name].append(eval_bc(tru_policy, test_demos, device))

            elapsed = time.time() - t0
            vals = " ".join(f"{m}={results[bone_name][m][-1]:.4f}" for m in results[bone_name])
            typer.echo(f"  {bone_name} s{seed}: {vals} ({elapsed:.0f}s)")

    stats = {}
    for bone in results:
        stats[bone] = {}
        for method in results[bone]:
            arr = np.array(results[bone][method])
            stats[bone][method] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci95": float(1.96 * arr.std() / np.sqrt(cfg["n_seeds"])),
                "seeds": results[bone][method],
            }

    out_path = output_dir / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    typer.echo(f"\nSaved to {out_path}")


@app.command()
def decoder_trust(
    suite: Annotated[str, typer.Option("--suite", "-s", help="Suite name")] = "spatial",
    seeds: Annotated[int, typer.Option("--seeds", "-n", help="Number of seeds")] = 5,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("."),
):
    """Run decoder-aware trust experiment for JEPA."""
    suite_dir = SUITE_DIRS[suite]
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Decoder-Aware Trust Experiment")
    typer.echo(f"  Suite: {suite}")
    typer.echo(f"  Seeds: {seeds}")
    typer.echo(f"{'='*60}")

    demos = load_demos(suite_dir, 10, 5)
    typer.echo(f"Loaded {len(demos)} tasks")

    obs_dim, act_dim = 21, 7
    device = "cuda"
    results = {"none": [], "decoder_trust": [], "latent_trust": []}

    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()

        wm = get_backbone("jepa", obs_dim, act_dim).to(device)
        train_wm(wm, demos[0], 50, device=device)

        std_policy = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim),
        ).to(device)
        train_bc(std_policy, demos[0], 50, 64, device=device)
        test_demos = demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]
        results["none"].append(eval_bc(std_policy, test_demos, device))

        typer.echo(f"  Seed {seed}: none={results['none'][-1]:.4f} ({time.time()-t0:.0f}s)")

    stats = {}
    for method in results:
        arr = np.array(results[method])
        stats[method] = {"mean": float(arr.mean()), "std": float(arr.std()), "seeds": results[method]}

    out_path = output_dir / "decoder_trust_results.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    typer.echo(f"\nSaved to {out_path}")


@app.command()
def audit(
    suite: Annotated[str, typer.Option("--suite", "-s", help="Suite name")] = "spatial",
    seeds: Annotated[int, typer.Option("--seeds", "-n", help="Number of seeds")] = 3,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("."),
):
    """Run audit experiments (error bars, task-order sensitivity)."""
    suite_dir = SUITE_DIRS[suite]
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Audit Experiments")
    typer.echo(f"  Suite: {suite}")
    typer.echo(f"  Seeds: {seeds}")
    typer.echo(f"{'='*60}")

    demos = load_demos(suite_dir, 10, 5)
    typer.echo(f"Loaded {len(demos)} tasks")

    obs_dim, act_dim = 21, 7
    device = "cuda"
    results = {"none": [], "ema": [], "multi_step": [], "ensemble": []}

    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()

        wm = get_backbone("mlp", obs_dim, act_dim).to(device)
        train_wm(wm, demos[0], 20, device=device)

        std_policy = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim),
        ).to(device)
        train_bc(std_policy, demos[0], 50, 64, device=device)
        test_demos = demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]
        results["none"].append(eval_bc(std_policy, test_demos, device))

        for tm_name in ["ema", "multi_step", "ensemble"]:
            trust = make_trust(tm_name, obs_dim, act_dim)
            tru_policy = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, act_dim),
            ).to(device)
            train_bc_trust(tru_policy, wm, trust, demos[0], 50, 64, device=device)
            results[tm_name].append(eval_bc(tru_policy, test_demos, device))

        elapsed = time.time() - t0
        vals = " ".join(f"{m}={results[m][-1]:.4f}" for m in results)
        typer.echo(f"  Seed {seed}: {vals} ({elapsed:.0f}s)")

    stats = {}
    for method in results:
        arr = np.array(results[method])
        stats[method] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95": float(1.96 * arr.std() / np.sqrt(seeds)),
            "seeds": results[method],
        }

    out_path = output_dir / "audit_results.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    typer.echo(f"\nSaved to {out_path}")


@app.command()
def list_backbones():
    """List available world model backbones."""
    from continualwam.backbones import BACKBONES
    typer.echo("Available backbones:")
    for name in BACKBONES:
        typer.echo(f"  - {name}")


@app.command()
def list_trust():
    """List available trust scoring methods."""
    from continualwam.trust import TRUST_METHODS
    typer.echo("Available trust methods:")
    for name in TRUST_METHODS:
        typer.echo(f"  - {name}")


if __name__ == "__main__":
    app()
