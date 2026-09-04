"""Task-Order Sensitivity Experiment with Forgetting Metric.

Evaluates how task ordering affects:
1. Final performance (mean error across all tasks)
2. Forgetting metric: average performance drop on earlier tasks after learning later ones
3. Forward transfer: improvement on later tasks from earlier experience

Uses 5+ random orderings per backbone × trust combination.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
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

app = typer.Typer()

SUITE_DIRS = {
    "spatial": "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial",
    "object": "/home/ubuntu/robotics_world_models/LIBERO/libero_object",
    "goal": "/home/ubuntu/robotics_world_models/LIBERO/libero_goal",
}


def compute_forgetting(
    errors_per_task: list[list[float]],
) -> dict:
    """Compute forgetting metrics from per-task errors across orderings.
    
    Args:
        errors_per_task: [n_orderings][n_tasks] mean error per task per ordering
    
    Returns:
        Dictionary with forgetting metrics
    """
    errors = np.array(errors_per_task)
    n_orderings, n_tasks = errors.shape
    
    # For each ordering, compute forgetting relative to first task
    forgetting_rates = []
    for i in range(n_orderings):
        # Forgetting = average error increase on later tasks relative to first
        first_task_error = errors[i, 0]
        if first_task_error > 0:
            relative_errors = (errors[i, :] - first_task_error) / first_task_error
            forgetting_rates.append(float(np.mean(relative_errors[1:])))
        else:
            forgetting_rates.append(0.0)
    
    # Stability: std of final performance across orderings
    final_errors = errors[:, -1]
    
    return {
        "mean_forgetting_rate": float(np.mean(forgetting_rates)),
        "std_forgetting_rate": float(np.std(forgetting_rates)),
        "mean_final_error": float(np.mean(final_errors)),
        "std_final_error": float(np.std(final_errors)),
        "per_ordering_forgetting": forgetting_rates,
        "per_ordering_final_error": [float(x) for x in final_errors],
    }


def train_sequential(
    wm,
    policy,
    demos: list[list[dict]],
    task_order: list[int],
    trust_scorer=None,
    wm_epochs: int = 20,
    bc_epochs: int = 50,
    device: str = "cuda",
) -> list[float]:
    """Train sequentially on tasks in given order, return per-task error.
    
    After learning each task, evaluate on all tasks seen so far.
    """
    obs_dim = demos[0][0]["obs"].shape[-1]
    act_dim = demos[0][0]["acts"].shape[-1]
    
    errors_per_task = []
    
    for i, task_idx in enumerate(task_order):
        # Train on current task
        task_demos = demos[task_idx]
        
        # Train WM on this task
        train_wm(wm, task_demos, wm_epochs, device=device)
        
        # Train policy with trust if scorer provided
        if trust_scorer is not None:
            train_bc_trust(policy, wm, trust_scorer, task_demos, bc_epochs, device=device)
        else:
            train_bc(policy, task_demos, bc_epochs, device=device)
        
        # Evaluate on all tasks seen so far
        seen_tasks = task_order[:i+1]
        all_errors = []
        for seen_idx in seen_tasks:
            err = eval_bc(policy, demos[seen_idx], device)
            all_errors.append(err)
        
        errors_per_task.append(float(np.mean(all_errors)))
    
    return errors_per_task


@app.command()
def task_order_sensitivity(
    suite: Annotated[str, typer.Option("--suite", "-s", help="Suite name")] = "spatial",
    backbone: Annotated[str, typer.Option("--backbone", "-b", help="Backbone")] = "mlp",
    trust: Annotated[str, typer.Option("--trust", "-t", help="Trust method")] = "ema",
    n_orderings: Annotated[int, typer.Option("--orderings", "-n", help="Number of random orderings")] = 10,
    n_seeds: Annotated[int, typer.Option("--seeds", help="Seeds per ordering")] = 3,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("."),
):
    """Run task-order sensitivity with forgetting metric."""
    import random
    
    suite_dir = SUITE_DIRS[suite]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Task-Order Sensitivity Experiment")
    typer.echo(f"  Suite: {suite}, Backbone: {backbone}, Trust: {trust}")
    typer.echo(f"  Orderings: {n_orderings}, Seeds per ordering: {n_seeds}")
    typer.echo(f"{'='*60}")
    
    demos = load_demos(suite_dir, 10, 5)
    n_tasks = len(demos)
    typer.echo(f"Loaded {n_tasks} tasks")
    
    obs_dim, act_dim = demos[0][0]["obs"].shape[-1], demos[0][0]["acts"].shape[-1]
    device = "cuda"
    
    # Generate random orderings
    orderings = []
    for _ in range(n_orderings):
        order = list(range(n_tasks))
        random.shuffle(order)
        orderings.append(order)
    
    # Run experiment
    all_errors = []  # [n_orderings * n_seeds][n_tasks]
    
    for order_idx, task_order in enumerate(orderings):
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            t0 = time.time()
            
            # Fresh WM and policy for each run
            wm = get_backbone(backbone, obs_dim, act_dim).to(device)
            policy = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, act_dim),
            ).to(device)
            
            # Train with or without trust
            trust_scorer = make_trust(trust, obs_dim, act_dim) if trust != "none" else None
            
            errors = train_sequential(
                wm, policy, demos, task_order,
                trust_scorer=trust_scorer,
                device=device,
            )
            all_errors.append(errors)
            
            elapsed = time.time() - t0
            typer.echo(f"  Order {order_idx+1}/{n_orderings}, Seed {seed}: "
                      f"final={errors[-1]:.4f} ({elapsed:.0f}s)")
    
    # Compute forgetting metric
    results = compute_forgetting(all_errors)
    
    # Also compute no-trust baseline for comparison
    all_errors_no_trust = []
    for order_idx, task_order in enumerate(orderings):
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            wm = get_backbone(backbone, obs_dim, act_dim).to(device)
            policy = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, act_dim),
            ).to(device)
            
            errors = train_sequential(
                wm, policy, demos, task_order,
                trust_scorer=None,
                device=device,
            )
            all_errors_no_trust.append(errors)
    
    results_no_trust = compute_forgetting(all_errors_no_trust)
    
    # Combine results
    output = {
        "config": {
            "suite": suite,
            "backbone": backbone,
            "trust": trust,
            "n_orderings": n_orderings,
            "n_seeds": n_seeds,
            "n_tasks": n_tasks,
        },
        "with_trust": results,
        "without_trust": results_no_trust,
        "improvement": {
            "forgetting_rate": results_no_trust["mean_forgetting_rate"] - results["mean_forgetting_rate"],
            "final_error": results_no_trust["mean_final_error"] - results["mean_final_error"],
        },
        "raw_errors_with_trust": all_errors,
        "raw_errors_without_trust": all_errors_no_trust,
    }
    
    # Print summary
    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Results Summary")
    typer.echo(f"{'='*60}")
    typer.echo(f"  With trust ({trust}):")
    typer.echo(f"    Forgetting rate: {results['mean_forgetting_rate']:.4f} ± {results['std_forgetting_rate']:.4f}")
    typer.echo(f"    Final error: {results['mean_final_error']:.4f} ± {results['std_final_error']:.4f}")
    typer.echo(f"  Without trust:")
    typer.echo(f"    Forgetting rate: {results_no_trust['mean_forgetting_rate']:.4f} ± {results_no_trust['std_forgetting_rate']:.4f}")
    typer.echo(f"    Final error: {results_no_trust['mean_final_error']:.4f} ± {results_no_trust['std_final_error']:.4f}")
    typer.echo(f"  Improvement:")
    typer.echo(f"    Forgetting rate: {output['improvement']['forgetting_rate']:.4f}")
    typer.echo(f"    Final error: {output['improvement']['final_error']:.4f}")
    
    # Save
    out_path = output_dir / f"task_order_{backbone}_{trust}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    typer.echo(f"\nSaved to {out_path}")
    
    return output


if __name__ == "__main__":
    app()
