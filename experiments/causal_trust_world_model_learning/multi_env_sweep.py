"""Multi-environment sweep: RSSM → Trust Methods → WAM CL Benchmark.

Runs the full pipeline on multiple ManiSkill environments.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

app = typer.Typer(help="Multi-environment sweep for trust benchmark.")

MANISKILL_ENVS = ["PushCube-v1", "LiftPegUpright-v1", "PlugCharger-v1", "StackCube-v1"]

@app.command()
def run(
    envs: list[str] = typer.Option(MANISKILL_ENVS, help="Environments to run"),
    episodes: int = typer.Option(200, help="RSSM training episodes"),
    epochs: int = typer.Option(50, help="RSSM training epochs"),
    trust_episodes: int = typer.Option(100, help="Trust method training episodes"),
    trust_epochs: int = typer.Option(30, help="Trust method training epochs"),
    cl_tasks: int = typer.Option(10, help="CL tasks per environment"),
    cl_episodes: int = typer.Option(20, help="CL training episodes per task"),
    device: str = typer.Option("cuda", help="Device"),
    save_dir: str = typer.Option("trained_models", help="Save directory"),
) -> None:
    """Run full pipeline on multiple environments."""
    base_dir = Path(__file__).parent
    results = {}

    for env in envs:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"ENVIRONMENT: {env}")
        typer.echo(f"{'='*60}")

        env_save = Path(save_dir) / env.lower().replace("-", "_").replace("/", "_")
        env_save.mkdir(parents=True, exist_ok=True)

        # Step 1: Train RSSM
        typer.echo(f"\n[1/3] Training RSSM on {env}...")
        subprocess.run([
            "python3", str(base_dir / "train_rssm.py"),
            "--env", env,
            "--episodes", str(episodes),
            "--epochs", str(epochs),
            "--device", device,
            "--save-dir", str(env_save),
        ], check=True)

        rssm_path = env_save / f"rssm_{env.lower().replace('-', '_').replace('/', '_')}.pt"

        # Step 2: Train trust methods
        typer.echo(f"\n[2/3] Training trust methods on {env}...")
        subprocess.run([
            "python3", str(base_dir / "train_trust_methods.py"),
            "--env", env,
            "--rssm-path", str(rssm_path),
            "--episodes", str(trust_episodes),
            "--epochs", str(trust_epochs),
            "--device", device,
            "--save-dir", str(env_save),
        ], check=True)

        # Step 3: Run WAM CL benchmark
        typer.echo(f"\n[3/3] Running WAM CL benchmark on {env}...")
        subprocess.run([
            "python3", str(base_dir / "wam_trust_benchmark.py"),
            "--env", env,
            "--n-tasks", str(cl_tasks),
            "--n-episodes", str(cl_episodes),
            "--n-eval", "10",
            "--device", device,
            "--rssm-path", str(rssm_path),
            "--trained-dir", str(env_save),
        ], check=True)

        # Load results
        result_path = Path("benchmark_results") / "wam_trust_comparison.json"
        if result_path.exists():
            with open(result_path) as f:
                results[env] = json.load(f)

    # Print summary
    typer.echo("\n" + "=" * 80)
    typer.echo("MULTI-ENVIRONMENT SUMMARY")
    typer.echo("=" * 80)
    typer.echo(f"{'Environment':<25} {'EMA':>8} {'FFDC':>8} {'Ensemble':>8} {'Closed-Loop':>8}")
    typer.echo("-" * 80)

    for env, data in results.items():
        row = f"{env:<25}"
        for method in ["EMA", "FFDC", "Ensemble", "Closed-Loop"]:
            if method in data:
                row += f" {data[method]['avg_acc']:>7.3f}"
            else:
                row += f" {'N/A':>7}"
        typer.echo(row)

    typer.echo("=" * 80)

    # Save summary
    summary_path = Path("benchmark_results") / "multi_env_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    typer.echo(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    app()
