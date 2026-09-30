"""ContinualWAM v2 decisive test: trust arms vs online EWC and adaptive-λ on LIBERO.

Protocol (fixed before running): λ is selected per arm on SELECT_SEEDS by final average
held-out error, then every arm is compared at its selected λ on EVAL_SEEDS.
Kill criterion: a trust arm must beat both `online_ewc` and `adaptive_lambda` on BWT by
more than one pooled std across EVAL_SEEDS, with mean trust spread > 0.05.
Forgetting gate (cross bench): plain fine-tuning must show BWT > GATE_FRACTION × plasticity
on every gate seed, otherwise there is no forgetting for consolidation to prevent.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import h5py
import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from full_backbone_sweep import BACKBONES
from trust_ewc_v2 import (
    ARMS,
    BATCH_SIZE,
    Arm,
    Demo,
    Phases,
    eval_error,
    run_sequence,
    transition_batches,
)

LIBERO_ROOT = Path.home() / "robotics_world_models/LIBERO"
OBS_KEYS = ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]
SELECT_SEEDS = [0, 1]
EVAL_SEEDS = [2, 3, 4, 5, 6]
GATE_SEEDS = [0, 1]
GATE_FRACTION = 0.1
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 1e5, 1e6]
MIN_SPREAD = 0.05
LEGACY_BENCH = "spatial10"

app = typer.Typer()
console = Console()
Row = dict[str, Any]


@dataclass(frozen=True)
class Bench:
    suites: tuple[str, ...]
    tasks_per_suite: int
    n_train: int
    n_eval: int
    phase_per_suite: bool
    epochs: int


BENCHES = {
    "spatial10": Bench(("spatial",), 10, n_train=5, n_eval=3, phase_per_suite=False, epochs=50),
    "cross": Bench(("spatial", "object", "goal"), 5, n_train=40, n_eval=10, phase_per_suite=True, epochs=20),
}


def load_task(path: Path, n_train: int, n_eval: int) -> tuple[list[Demo], list[Demo]]:
    with h5py.File(path, "r") as hf:
        keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[1]))[: n_train + n_eval]
        demos = [
            {
                "obs": np.concatenate([hf["data"][k]["obs"][o][:] for o in OBS_KEYS], -1),
                "actions": hf["data"][k]["actions"][:],
            }
            for k in keys
        ]
    return demos[:n_train], demos[n_train:]


def load_phases(bench: Bench) -> tuple[Phases, Phases]:
    train: Phases = []
    held_out: Phases = []
    for suite in bench.suites:
        files = sorted((LIBERO_ROOT / f"libero_{suite}").glob("*.hdf5"))[: bench.tasks_per_suite]
        tasks = [load_task(f, bench.n_train, bench.n_eval) for f in files]
        if bench.phase_per_suite:
            train.append([d for tr, _ in tasks for d in tr])
            held_out.append([d for _, ev in tasks for d in ev])
        else:
            train += [tr for tr, _ in tasks]
            held_out += [ev for _, ev in tasks]
    return normalise(train, held_out)


def normalise(train: Phases, held_out: Phases) -> tuple[Phases, Phases]:
    stack = {k: np.concatenate([d[k] for t in train for d in t]) for k in ("obs", "actions")}
    stats = {k: (v.mean(0), v.std(0) + 1e-6) for k, v in stack.items()}

    def apply(phases: Phases) -> Phases:
        return [
            [{k: ((d[k] - stats[k][0]) / stats[k][1]).astype(np.float32) for k in stats} for d in t]
            for t in phases
        ]

    return apply(train), apply(held_out)


def read_rows(path: Path, backbone: str, bench: str) -> list[Row]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["backbone"] == backbone and r.get("bench", LEGACY_BENCH) == bench]


def values(rows: list[Row], arm: str, lam: float, seeds: list[int], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows if r["arm"] == arm and r["lam"] == lam and r["seed"] in seeds])


def selected_lambda(rows: list[Row], arm: str) -> float:
    scores = {lam: values(rows, arm, lam, SELECT_SEEDS, "final_err") for lam in LAMBDAS}
    means = {lam: float(v.mean()) for lam, v in scores.items() if v.size and np.isfinite(v).all()}
    return min(means, key=means.get)


def fmt_stat(a: np.ndarray) -> str:
    return f"{a.mean():+.4f} ± {a.std(ddof=1):.4f} (n={a.size})" if a.size > 1 else "—"


def pick_arms(names: str) -> list[Arm]:
    wanted = names.split(",")
    return [a for a in ARMS if a.name in wanted]


class Sweep:
    def __init__(self, out: Path, backbone: str, bench_name: str, device: str) -> None:
        self.out, self.backbone, self.bench_name, self.device = out, backbone, bench_name, device
        self.bench = BENCHES[bench_name]
        self.train, self.held_out = load_phases(self.bench)
        self.obs_dim = self.train[0][0]["obs"].shape[-1]
        self.act_dim = self.train[0][0]["actions"].shape[-1]
        sizes = [sum(len(d["obs"]) for d in p) for p in self.train]
        console.print(f"{bench_name}: {len(self.train)} phases, transitions/phase={sizes}, "
                      f"obs_dim={self.obs_dim} act_dim={self.act_dim}")

    def model(self, seed: int) -> torch.nn.Module:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return BACKBONES[self.backbone](self.obs_dim, self.act_dim).to(self.device)

    def rows(self) -> list[Row]:
        return read_rows(self.out, self.backbone, self.bench_name)

    def record(self, fh: TextIO, row: Row) -> None:
        fh.write(json.dumps({"backbone": self.backbone, "bench": self.bench_name, **row}) + "\n")
        fh.flush()

    def sequential(self, fh: TextIO, arm: Arm, lam: float, seed: int) -> Row:
        res = run_sequence(self.model(seed), self.train, self.held_out, arm, lam, epochs=self.bench.epochs)
        row = {"arm": arm.name, "lam": lam, "seed": seed, "bwt": res.backward_transfer,
               "final_err": res.final_avg_error, "plasticity": res.plasticity,
               "trust_spread": res.trust_spread, "errors": res.errors.tolist()}
        self.record(fh, row)
        console.print(f"{arm.name} λ={lam:g} seed={seed} bwt={res.backward_transfer:+.4f} "
                      f"plast={res.plasticity:.4f} final={res.final_avg_error:.4f} "
                      f"spread={res.trust_spread:.3f}")
        return row

    def joint(self, fh: TextIO, arm: Arm, seed: int) -> Row:
        wm = self.model(seed)
        pooled = [[d for p in self.train for d in p]]
        res = run_sequence(wm, pooled, pooled, arm, lam=0.0, epochs=self.bench.epochs)
        per_phase = [eval_error(wm, transition_batches(p, BATCH_SIZE, self.device)) for p in self.held_out]
        row = {"arm": f"joint_{arm.name}", "lam": 0.0, "seed": seed,
               "final_err": float(np.mean(per_phase)), "trust_spread": res.trust_spread}
        self.record(fh, row)
        console.print(f"joint {arm.name} seed={seed} final={row['final_err']:.4f}")
        return row

    def execute(self, jobs: list[tuple[Arm, float, int]], label: str) -> None:
        done = {(r["arm"], r["lam"], r["seed"]) for r in self.rows()}
        jobs = [j for j in jobs if (j[0].name, j[1], j[2]) not in done]
        console.print(f"[{label}] {len(jobs)} runs pending")
        with Progress(console=console) as progress, self.out.open("a") as fh:
            bar = progress.add_task(label, total=len(jobs))
            for arm, lam, seed in jobs:
                self.sequential(fh, arm, lam, seed)
                progress.advance(bar)


@app.command()
def gate(out: Path, backbone: str = "mlp", bench: str = "cross", device: str = "cuda") -> None:
    """Check the benchmark has real forgetting, and measure the joint-training upper bound."""
    sweep = Sweep(out, backbone, bench, device)
    unweighted = pick_arms("online_ewc")[0]
    with out.open("a") as fh:
        rows = [sweep.sequential(fh, unweighted, 0.0, s) for s in GATE_SEEDS]
        joint = [sweep.joint(fh, unweighted, s)["final_err"] for s in GATE_SEEDS]
    passed = all(r["bwt"] > GATE_FRACTION * r["plasticity"] for r in rows)
    finetune_final = np.mean([r["final_err"] for r in rows])
    console.print(f"fine-tune BWT={[round(r['bwt'], 4) for r in rows]} "
                  f"plasticity={[round(r['plasticity'], 4) for r in rows]}")
    console.print(f"fine-tune final={finetune_final:.4f}  joint final={np.mean(joint):.4f}")
    console.print(f"FORGETTING GATE: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise typer.Exit(code=3)


@app.command()
def run(
    out: Path,
    backbone: str = "mlp",
    bench: str = "spatial10",
    arms: str = ",".join(a.name for a in ARMS),
    device: str = "cuda",
) -> None:
    sweep = Sweep(out, backbone, bench, device)
    chosen_arms = pick_arms(arms)
    sweep.execute([(a, lam, s) for s in SELECT_SEEDS for lam in LAMBDAS for a in chosen_arms], "select")
    rows = sweep.rows()
    chosen = {a.name: selected_lambda(rows, a.name) for a in chosen_arms}
    console.print(f"selected λ: {chosen}")
    sweep.execute([(a, chosen[a.name], s) for s in EVAL_SEEDS for a in chosen_arms], "eval")


@app.command()
def joint(out: Path, backbone: str = "mlp", bench: str = "spatial10", device: str = "cuda") -> None:
    """Robust-loss control: pooled (non-sequential) training, weighting on vs off."""
    sweep = Sweep(out, backbone, bench, device)
    with out.open("a") as fh:
        for seed in EVAL_SEEDS:
            for arm in pick_arms("online_ewc,trust_sample"):
                sweep.joint(fh, arm, seed)


def beats(arm_bwt: np.ndarray, baseline_bwt: np.ndarray) -> bool:
    pooled_std = np.sqrt((baseline_bwt.var(ddof=1) + arm_bwt.var(ddof=1)) / 2)
    return baseline_bwt.mean() - arm_bwt.mean() > pooled_std


@app.command()
def summarize(results: Path, backbone: str = "mlp", bench: str = LEGACY_BENCH) -> None:
    rows = read_rows(results, backbone, bench)
    table = Table("arm", "λ*", "BWT (eval)", "final err", "plasticity", "trust spread")
    picked: dict[str, np.ndarray] = {}
    spread: dict[str, float] = {}
    for arm in ARMS:
        if not any(r["arm"] == arm.name and r["seed"] in SELECT_SEEDS for r in rows):
            continue
        lam = selected_lambda(rows, arm.name)
        keys = ("bwt", "final_err", "plasticity", "trust_spread")
        stat = {k: values(rows, arm.name, lam, EVAL_SEEDS, k) for k in keys}
        picked[arm.name] = stat["bwt"]
        spread[arm.name] = float(stat["trust_spread"].mean())
        table.add_row(arm.name, f"{lam:g}", *(fmt_stat(v) for v in stat.values()))
    console.print(table)
    baselines = [picked[b] for b in ("online_ewc", "adaptive_lambda") if b in picked]
    for arm in ("trust_sample", "trust_fisher", "trust_full"):
        if arm not in picked or len(baselines) < 2 or picked[arm].size < 2:
            continue
        wins = all(beats(picked[arm], b) for b in baselines)
        verdict = "PASS" if wins and spread[arm] > MIN_SPREAD else "FAIL"
        console.print(f"{arm}: {verdict} (beats baselines={wins}, trust spread={spread[arm]:.3f})")


if __name__ == "__main__":
    app()
