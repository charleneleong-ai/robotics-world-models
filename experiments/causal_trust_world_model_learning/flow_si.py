#!/usr/bin/env python3
"""FLOW self-loss weighting in the ManiSkill CL harness, on top of fine-tuning, EWC and SI.

Same benchmark, accuracy matrix and metrics as the 10-method comparison
(cl_baselines_full_rerun.py). FLOW weights are exp(-l_i / median l) with l_i the
per-sample cross-entropy under the model frozen at the end of the previous task
(median taken over the minibatch); the first task is trained unweighted.
Arms: fine_tuning / ft_flow / ewc / ewc_flow / si / si_flow, NUM_SEEDS seeds,
paired by seed.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cl_baselines_full_rerun import EPOCHS_PER_TASK, NUM_TASKS, ManiSkillCLExperiment, SynapticIntelligenceCL  # noqa: F401
from continual_learning import EWCCL, FineTuningCL
from maniskill_benchmark import ManiSkillBenchmark, SimpleMLP

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
OUT = Path(__file__).parent / "results_per_param_consolidation" / "flow_si_maniskill.json"


class FlowMixin:
    """Adds FLOW weights from a frozen previous-task snapshot; subclasses supply the penalty."""

    theta_star: torch.nn.Module | None = None

    def flow_weights(self, obs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.theta_star is None:
            return torch.ones(len(targets), device=self.device)
        with torch.no_grad():
            loss = F.cross_entropy(self.theta_star(obs), targets, reduction="none")
            return torch.exp(-loss / loss.median().clamp_min(1e-8))

    def penalty(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

    def after_step(self, grads: dict[str, torch.Tensor]) -> None:
        pass

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        obs, targets = batch["obs"].to(self.device), batch["targets"].to(self.device)
        logits = self.model(obs)
        loss = (self.flow_weights(obs, targets) * F.cross_entropy(logits, targets, reduction="none")).mean() + self.penalty()
        loss.backward()
        grads = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.optimizer.step()
        self.after_step(grads)
        return {"loss": loss.item(), "accuracy": (logits.argmax(-1) == targets).float().mean().item()}

    def snapshot(self) -> None:
        self.theta_star = copy.deepcopy(self.model).eval()
        for p in self.theta_star.parameters():
            p.requires_grad_(False)


class FlowFineTuningCL(FlowMixin, FineTuningCL):
    def consolidate(self, task_id: int = None):
        super().consolidate(task_id)
        self.snapshot()


class FlowEWCCL(FlowMixin, EWCCL):
    def penalty(self) -> torch.Tensor:
        pen = torch.tensor(0.0, device=self.device)
        for task_id, fisher in self.fisher_info.items():
            for n, p in self.model.named_parameters():
                if n in self.optimal_params[task_id]:
                    pen = pen + (fisher[n] * (p - self.optimal_params[task_id][n].to(self.device)).pow(2)).sum()
        return self.ewc_lambda * pen

    def observe(self, batch: dict) -> dict:
        out = super().observe(batch)
        self._task_samples.append({"obs": batch["obs"].detach().cpu(), "targets": batch["targets"].detach().cpu()})
        return out

    def consolidate(self, task_id: int = None):
        super().consolidate(task_id)
        self.snapshot()


class FlowSICL(FlowMixin, SynapticIntelligenceCL):
    def penalty(self) -> torch.Tensor:
        pen = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            pen = pen + (self.omega[n] * (p - self.theta_anchor[n]).pow(2)).sum()
        return self.si_c * pen

    def after_step(self, grads: dict[str, torch.Tensor]) -> None:
        for n, p in self.model.named_parameters():
            self.w[n] += -grads[n] * (p.detach() - self.theta_prev[n])
            self.theta_prev[n] = p.detach().clone()

    def consolidate(self, task_id: int = None):
        super().consolidate(task_id)
        self.snapshot()


def build(obs_dim: int, num_classes: int, device: torch.device) -> dict:
    mk = lambda: SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes).to(device)
    return {"fine_tuning": FineTuningCL(mk(), device=device), "ft_flow": FlowFineTuningCL(mk(), device=device),
            "ewc": EWCCL(mk(), device=device), "ewc_flow": FlowEWCCL(mk(), device=device),
            "si": SynapticIntelligenceCL(mk(), device=device), "si_flow": FlowSICL(mk(), device=device)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed: list[dict[str, dict]] = []
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        bench = ManiSkillBenchmark(num_tasks=NUM_TASKS, episodes_per_task=50, max_steps=100, obs_dim=64, action_dim=10)
        exp = ManiSkillCLExperiment(bench, device); exp.collect()
        obs_dim = exp.task_datasets[0]["obs"].shape[1]
        num_classes = 10  # as in build_methods of cl_baselines_full_rerun.py
        row = {}
        for name, learner in build(obs_dim, num_classes, device).items():
            torch.manual_seed(seed)
            t0 = time.time(); m = exp.run_method(name, learner); row[name] = m
            print(f"seed={seed} {name:12s} acc={m['avg_accuracy']:.4f} bwt={m['bwt']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        per_seed.append(row)
        json.dump({"per_seed": per_seed}, open(OUT, "w"), indent=2)
    summary = {}
    for base, flow in (("fine_tuning", "ft_flow"), ("ewc", "ewc_flow"), ("si", "si_flow")):
        for k in ("avg_accuracy", "bwt"):
            a = np.array([r[base][k] for r in per_seed]); b = np.array([r[flow][k] for r in per_seed])
            summary[f"{flow}_vs_{base}_{k}"] = {"base": float(a.mean()), "flow": float(b.mean()), "delta": float(b.mean() - a.mean()),
                                                 "p_paired": float(stats.ttest_rel(a, b)[1]), "better": int((b > a).sum()), "n": len(a)}
    print("== " + json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in summary.items()}), flush=True)
    json.dump({"per_seed": per_seed, "summary": summary}, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
