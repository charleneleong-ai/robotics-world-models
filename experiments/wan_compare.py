#!/usr/bin/env python3
"""Which representation is best for LIBERO behaviour cloning?

Probes identical MLP action heads on every cached representation, paired across the same
tasks, seeds and held-out split (2 of 12 demonstrations per task, split by whole
trajectory). Reports each against three references so the comparisons that matter are
explicit: proprioception (is video worth anything?), the per-frame DiT (does temporal
context buy anything, i.e. is the dynamics prior doing work?), and the VAE (does the
world model beat its own tokenizer?).

  state          21-dim proprioception, the representation the earlier papers used
  vae            frozen Wan VAE latent, per frame
  dit_ctx1       Wan DiT mid-block features, one frame -- temporal attention inert
  dit_ctx8       Wan DiT mid-block features, 8-frame window
  dit_ctx8_state dit_ctx8 concatenated with proprioception

Caches that are absent are skipped, along with every comparison that needs them.
"""
from __future__ import annotations

import numpy as np
import typer
import wandb
from scipy import stats

from probe_common import CACHE_ROOT, TaskArrays, fit_probe, held_out_mask, load_task

CTXS = [1, 4, 8, 16]
ALL_SOURCES = {"vae": CACHE_ROOT / "spatial_agentview_rgb",
               **{f"dit_ctx{c}": CACHE_ROOT / f"spatial_dit_ctx{c}" for c in CTXS},
               **{name: CACHE_ROOT / f"spatial_{name}"
                  for name in ("siglip2", "siglip2_ctx8", "openvla", "openvla_ctx8")}}
# 1-frame group and 2-frame group are each internally matched on temporal context
ALL_REPS = (["state", "vae", "dit_ctx1", "siglip2", "openvla"]
            + ["dit_ctx4", "dit_ctx8", "dit_ctx16", "siglip2_ctx8", "openvla_ctx8"]
            + ["openvla_ctx8_state"])
PAIRS = [("siglip2", "vae"), ("openvla", "vae"), ("openvla", "dit_ctx1"), ("siglip2", "dit_ctx1"),
         ("openvla", "siglip2"), ("openvla_ctx8", "siglip2_ctx8"),
         ("openvla_ctx8", "dit_ctx8"), ("siglip2_ctx8", "dit_ctx8"),
         ("openvla_ctx8", "openvla"), ("siglip2_ctx8", "siglip2")]
N_TASKS, N_SEEDS, N_EVAL_DEMOS = 10, 3, 2
STATE_SUFFIX = "_state"


def base_rep(rep: str) -> str:
    return rep.removesuffix(STATE_SUFFIX) if rep != "state" else rep


def features(task: TaskArrays, rep: str) -> np.ndarray:
    if rep == "state":
        return task.state
    if rep.endswith(STATE_SUFFIX):
        return np.concatenate([task.features[base_rep(rep)], task.state], 1)
    return task.features[rep]


class Comparison:
    """Every available representation probed on the same tasks, seeds and held-out split."""

    def __init__(self) -> None:
        self.sources = {k: v for k, v in ALL_SOURCES.items() if (v / "task00.npz").exists()}
        self.reps = [r for r in ALL_REPS if r == "state" or base_rep(r) in self.sources]
        self.dims = {r: features(load_task(self.sources, 0), r).shape[1] for r in self.reps}

    def run(self) -> dict[str, np.ndarray]:
        err = {r: np.zeros((N_TASKS, N_SEEDS)) for r in self.reps}
        for ti in range(N_TASKS):
            task = load_task(self.sources, ti)
            ev = held_out_mask(task.demo, N_EVAL_DEMOS)
            tr = ~ev
            for rep in self.reps:
                x = features(task, rep)
                for s in range(N_SEEDS):
                    err[rep][ti, s] = fit_probe(x[tr], task.action[tr], x[ev], task.action[ev], s)
                wandb.log({"task": ti, f"heldout_mse/{rep}": err[rep][ti].mean()})
            print("task %d  " % ti + "  ".join("%s=%.4f" % (r, err[r][ti].mean()) for r in self.reps), flush=True)
        return err

    def summarise(self, pt: dict[str, np.ndarray], refs: list[str]) -> wandb.Table:
        tbl = wandb.Table(columns=["rep", "dims", "mse"] + [c for r in refs for c in (f"vs_{r}_%", f"p_{r}")])
        for r in self.reps:
            row = [r, self.dims[r], round(float(pt[r].mean()), 5)]
            for ref in refs:
                row += ([None, None] if r == ref else
                        [round(100 * (pt[r].mean() / pt[ref].mean() - 1), 2),
                         round(float(stats.ttest_rel(pt[ref], pt[r])[1]), 5)])
            tbl.add_data(*row)
            wandb.summary[f"mse/{r}"] = float(pt[r].mean())
            print("== %-19s dims=%5d  MSE %.4f   vs state %+7.1f%%" % (
                r, self.dims[r], pt[r].mean(), 100 * (pt[r].mean() / pt["state"].mean() - 1)))
        return tbl

    def report_pairs(self, pt: dict[str, np.ndarray]) -> None:
        candidates = [(r, "state") for r in self.reps if r != "state"] + PAIRS
        for a, b in [(a, b) for a, b in candidates if a in self.reps and b in self.reps]:
            p_val = float(stats.ttest_rel(pt[b], pt[a])[1])
            wandb.summary[f"p/{a}_vs_{b}"] = p_val
            print("   %-19s vs %-12s %+7.1f%%  p=%.4f  %d/%d tasks" % (
                a, b, 100 * (pt[a].mean() / pt[b].mean() - 1), p_val, int((pt[a] < pt[b]).sum()), N_TASKS))


def main() -> None:
    comparison = Comparison()
    run = wandb.init(project="video-wam", job_type="compare", name="compare-backbone-audit",
                     config=dict(suite="spatial", n_tasks=N_TASKS, n_seeds=N_SEEDS, reps=comparison.reps,
                                 eval_demos_per_task=N_EVAL_DEMOS, epochs=30, renders_frames=False))
    err = comparison.run()
    pt = {r: err[r].mean(1) for r in comparison.reps}
    print()
    tbl = comparison.summarise(pt, [x for x in ("state", "dit_ctx1", "vae") if x in comparison.reps])
    comparison.report_pairs(pt)
    wandb.log({"results": tbl})
    np.savez(CACHE_ROOT / "compare_results.npz", **err)
    run.finish()


if __name__ == "__main__":
    typer.run(main)
