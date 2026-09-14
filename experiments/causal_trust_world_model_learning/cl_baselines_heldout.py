#!/usr/bin/env python3
"""Held-out rerun of the ManiSkill CL baselines (Result 6) and the FLOW-on-{FT,EWC,SI} arms.

cl_baselines_full_rerun.py evaluates each task on the same samples it trained on. Here
each task's classification set is split 80/20 (fixed per task); learners train on the
80% and the accuracy matrix, BWT and FWT are computed on the 20%. Everything else
(benchmark construction, methods, epochs, seeds) is reused unchanged.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import cl_baselines_full_rerun as base  # noqa: E402
import flow_si  # noqa: E402

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = Path(__file__).parent / "results_per_param_consolidation" / "cl_baselines_heldout.json"
EVAL_FRAC = 0.2


class HeldoutExperiment(base.ManiSkillCLExperiment):
    """Splits every task set once; training uses the 80% part, evaluation the 20% part."""

    def collect(self) -> None:
        super().collect()
        self.eval_map: dict[int, dict] = {}
        for i, d in enumerate(self.task_datasets):
            n = len(d["obs"]); perm = torch.randperm(n, generator=torch.Generator().manual_seed(1000 + i)); k = int(n * (1 - EVAL_FRAC))
            pick = lambda idx: {key: (v[idx] if torch.is_tensor(v) and len(v) == n else v) for key, v in d.items()}
            train, ev = pick(perm[:k]), pick(perm[k:])
            self.task_datasets[i] = train; self.eval_map[id(train)] = ev


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_evaluate = base.evaluate
    per_seed = []
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        bench = base.ManiSkillBenchmark(num_tasks=base.NUM_TASKS, episodes_per_task=50, max_steps=100, obs_dim=64, action_dim=10)
        exp = HeldoutExperiment(bench, device); exp.collect()
        base.evaluate = lambda model, dataset, dev, _m=exp.eval_map: original_evaluate(model, _m.get(id(dataset), dataset), dev)
        obs_dim = exp.task_datasets[0]["obs"].shape[1]
        learners = base.build_methods(obs_dim, 10, device)
        learners.update({k: v for k, v in flow_si.build(obs_dim, 10, device).items() if k.endswith("_flow")})
        row = {}
        for name, learner in learners.items():
            torch.manual_seed(seed); t0 = time.time(); m = exp.run_method(name, learner); row[name] = m
            print(f"seed={seed} {name:18s} acc={m['avg_accuracy']:.4f} bwt={m['bwt']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        per_seed.append(row); json.dump({"per_seed": per_seed}, open(OUT, "w"), indent=2)
    base.evaluate = original_evaluate
    methods = list(per_seed[0].keys())
    table = {m: {k: {"mean": float(np.mean([r[m][k] for r in per_seed])), "std": float(np.std([r[m][k] for r in per_seed]))} for k in ("avg_accuracy", "bwt", "fwt")} for m in methods}
    pairs = {}
    for basem, flow in (("fine_tuning", "ft_flow"), ("ewc", "ewc_flow"), ("si", "si_flow")):
        for k in ("avg_accuracy", "bwt"):
            a = np.array([r[basem][k] for r in per_seed]); b = np.array([r[flow][k] for r in per_seed])
            pairs[f"{flow}_vs_{basem}_{k}"] = {"base": float(a.mean()), "flow": float(b.mean()), "p_paired": float(stats.ttest_rel(a, b)[1]), "better": int((b > a).sum())}
    print("== heldout table: " + json.dumps({m: {k: round(v["mean"], 4) for k, v in t.items()} for m, t in table.items()}), flush=True)
    print("== heldout flow pairs: " + json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in pairs.items()}), flush=True)
    json.dump({"per_seed": per_seed, "table": table, "flow_pairs": pairs}, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
