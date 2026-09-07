#!/usr/bin/env python3
"""Real 10-method CL-baselines comparison on ManiSkill, with real FWT/BWT.

Adds two previously-missing real implementations (SynapticIntelligenceCL,
GDumbCL) to the existing real continual_learning.py classes, and replaces
the single-pass accuracy tracking in ManiSkillBenchmark.run_experiment with
a full accuracy matrix (task i evaluated after every subsequent task),
which is what real forward/backward transfer require. Everything else
(env rollouts, classification-task construction, the other 8 methods) is
reused unmodified from maniskill_benchmark.py / continual_learning.py.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maniskill_benchmark import ManiSkillBenchmark, SimpleMLP
from continual_learning import (
    ContinualLearner,
    FineTuningCL,
    EWCCL,
    ExperienceReplayCL,
    PrioritizedReplayCL,
    CuriousReplayCL,
    LwFCL,
    PackNetCL,
    WorldModelTrustCL,
)
from rssm_world_model import WorldModel

RESULTS_DIR = Path(__file__).parent / "results_cl_baselines_full"
RESULTS_DIR.mkdir(exist_ok=True)
NUM_SEEDS = 5
NUM_TASKS = 4
EPOCHS_PER_TASK = 10


class SynapticIntelligenceCL(ContinualLearner):
    """Synaptic Intelligence (Zenke et al., 2017): online per-parameter importance
    from the path integral of (-grad * parameter update), penalizing drift from
    each task's anchor weighted by accumulated importance.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3,
                 device: torch.device = torch.device("cpu"), si_c: float = 1.0, xi: float = 1e-3):
        super().__init__(model, lr, device)
        self.si_c = si_c
        self.xi = xi
        self.omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        self.w = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        self.theta_prev = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.theta_anchor = {n: p.detach().clone() for n, p in model.named_parameters()}

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        obs = batch["obs"].to(self.device)
        targets = batch["targets"].to(self.device)

        logits = self.model(obs)
        task_loss = F.cross_entropy(logits, targets)
        si_penalty = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            si_penalty = si_penalty + (self.omega[n] * (p - self.theta_anchor[n]).pow(2)).sum()
        loss = task_loss + self.si_c * si_penalty
        loss.backward()

        grads = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                 for n, p in self.model.named_parameters()}
        self.optimizer.step()

        for n, p in self.model.named_parameters():
            delta = p.detach() - self.theta_prev[n]
            self.w[n] += -grads[n] * delta
            self.theta_prev[n] = p.detach().clone()

        return {"loss": loss.item(), "accuracy": (logits.argmax(-1) == targets).float().mean().item()}

    def consolidate(self, task_id: int = None):
        for n, p in self.model.named_parameters():
            delta_total = p.detach() - self.theta_anchor[n]
            self.omega[n] += self.w[n] / (delta_total.pow(2) + self.xi)
            self.w[n] = torch.zeros_like(p)
            self.theta_anchor[n] = p.detach().clone()
        self.task_count += 1


class GDumbCL(ContinualLearner):
    """GDumb (Prabhu et al., 2020): greedy class-balanced buffer, no online
    learning -- a fresh model is retrained from scratch on the buffer after
    each task. `observe()` only fills the buffer; `consolidate()` does the
    actual (re-)training.
    """

    def __init__(self, model_cls, lr: float = 1e-3,
                 device: torch.device = torch.device("cpu"), buffer_size: int = 500):
        self.model_cls = model_cls
        self.model = model_cls()
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.task_count = 0
        self.buffer_size = buffer_size
        self.buffer_obs: list[torch.Tensor] = []
        self.buffer_targets: list[torch.Tensor] = []

    def observe(self, batch: dict) -> dict:
        self._greedy_balanced_add(batch["obs"], batch["targets"])
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch["obs"].to(self.device))
            acc = (logits.argmax(-1) == batch["targets"].to(self.device)).float().mean().item()
        return {"loss": 0.0, "accuracy": acc}

    def _greedy_balanced_add(self, obs: torch.Tensor, targets: torch.Tensor) -> None:
        for o, t in zip(obs, targets):
            if len(self.buffer_obs) < self.buffer_size:
                self.buffer_obs.append(o.clone())
                self.buffer_targets.append(t.clone())
            else:
                counts = torch.bincount(torch.stack(self.buffer_targets))
                largest_class = counts.argmax().item()
                candidates = [i for i, bt in enumerate(self.buffer_targets) if bt.item() == largest_class]
                evict = candidates[np.random.randint(len(candidates))]
                self.buffer_obs[evict] = o.clone()
                self.buffer_targets[evict] = t.clone()

    def consolidate(self, task_id: int = None):
        self.model = self.model_cls()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        obs = torch.stack(self.buffer_obs).to(self.device)
        targets = torch.stack(self.buffer_targets).to(self.device)
        self.model.train()
        for _ in range(20):
            perm = torch.randperm(len(obs))
            for start in range(0, len(obs), 64):
                idx = perm[start:start + 64]
                self.optimizer.zero_grad()
                loss = F.cross_entropy(self.model(obs[idx]), targets[idx])
                loss.backward()
                self.optimizer.step()
        self.task_count += 1


def evaluate(model: nn.Module, dataset: dict, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(dataset["obs"].to(device))
        return (logits.argmax(-1) == dataset["targets"].to(device)).float().mean().item()


def compute_cl_metrics(acc_matrix: list[list[float]]) -> dict:
    """acc_matrix[i][j] = accuracy on task i after training through task j (j >= i)."""
    n = len(acc_matrix)
    final_accs = [acc_matrix[i][n - 1] for i in range(n)]
    avg_accuracy = float(np.mean(final_accs))

    bwt_terms = [acc_matrix[i][n - 1] - acc_matrix[i][i] for i in range(n - 1)]
    bwt = float(np.mean(bwt_terms)) if bwt_terms else 0.0

    fwt_terms = [acc_matrix[i][i - 1] - acc_matrix[0][0] for i in range(1, n)]
    fwt = float(np.mean(fwt_terms)) if fwt_terms else 0.0

    return {"avg_accuracy": avg_accuracy, "bwt": bwt, "fwt": fwt}


class ManiSkillCLExperiment:
    """Owns one seed's run across all 10 methods, tracking the full accuracy matrix."""

    def __init__(self, bench: ManiSkillBenchmark, device: torch.device = torch.device("cpu")):
        self.bench = bench
        self.device = device
        self.task_datasets: list[dict] = []

    def collect(self) -> None:
        for i, env_name in enumerate(self.bench.selected_envs):
            data = self.bench.collect_task_data(env_name, self.bench.episodes_per_task)
            self.task_datasets.append(self.bench.create_classification_task(data, i))

    def run_method(self, method_name: str, learner) -> dict:
        n = len(self.task_datasets)
        acc_matrix = [[0.0] * n for _ in range(n)]

        for task_id in range(n):
            dataset = self.task_datasets[task_id]
            if task_id > 0:
                # Zero-shot accuracy on this task using only knowledge from tasks 0..task_id-1
                # (forward transfer signal); undefined/unused for task 0.
                acc_matrix[task_id][task_id - 1] = evaluate(learner.model, dataset, self.device)

            batch_size = min(64, len(dataset["obs"]))
            for _ in range(EPOCHS_PER_TASK):
                indices = torch.randperm(len(dataset["obs"]))
                for start in range(0, len(indices), batch_size):
                    idx = indices[start:start + batch_size]
                    batch = {
                        "obs": dataset["obs"][idx],
                        "actions": torch.randn(len(idx), 10),
                        "targets": dataset["targets"][idx],
                        "next_obs": dataset["obs"][idx] + torch.randn_like(dataset["obs"][idx]) * 0.01,
                        "task_id": task_id,
                    }
                    learner.observe(batch)

            if hasattr(learner, "consolidate"):
                if method_name == "world_model_trust_cl":
                    avg_trust = np.mean(learner.task_trust_scores.get(task_id, [0.5]))
                    learner.consolidate(task_id, avg_trust)
                else:
                    learner.consolidate(task_id)

            for j in range(n):
                acc_matrix[j][task_id] = evaluate(learner.model, self.task_datasets[j], self.device)

        metrics = compute_cl_metrics(acc_matrix)
        metrics["acc_matrix"] = acc_matrix
        return metrics


def build_methods(obs_dim: int, num_classes: int, device: torch.device) -> dict:
    model_cls = lambda: SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes).to(device)
    world_model = lambda: WorldModel(obs_dim=obs_dim, action_dim=10, hidden_dim=256,
                                      stochastic_dim=16, stochastic_classes=16, deterministic_dim=256).to(device)
    return {
        "fine_tuning": FineTuningCL(model_cls(), device=device),
        "ewc": EWCCL(model_cls(), device=device),
        "si": SynapticIntelligenceCL(model_cls(), device=device),
        "lwf": LwFCL(model_cls(), device=device),
        "packnet": PackNetCL(model_cls(), device=device),
        "gdumb": GDumbCL(model_cls, device=device, buffer_size=500),
        "experience_replay": ExperienceReplayCL(model_cls(), device=device, buffer_size=2000),
        "prioritized_replay": PrioritizedReplayCL(model_cls(), device=device, buffer_size=2000),
        "curious_replay": CuriousReplayCL(model_cls(), world_model(), device=device, buffer_size=2000),
        "world_model_trust_cl": WorldModelTrustCL(model_cls(), world_model(), device=device),
    }


def run_sweep() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_seed_results: dict[str, list[dict]] = {}
    t0 = time.time()

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        bench = ManiSkillBenchmark(num_tasks=NUM_TASKS, episodes_per_task=50, max_steps=100,
                                    obs_dim=64, action_dim=10)
        exp = ManiSkillCLExperiment(bench, device=device)
        exp.collect()
        obs_dim = exp.task_datasets[0]["obs"].shape[1]
        num_classes = 10
        methods = build_methods(obs_dim, num_classes, device)

        for name, learner in methods.items():
            metrics = exp.run_method(name, learner)
            all_seed_results.setdefault(name, []).append(metrics)
            print(f"seed={seed} {name:20s} avg_acc={metrics['avg_accuracy']:.4f} "
                  f"bwt={metrics['bwt']:.4f} fwt={metrics['fwt']:.4f} [{time.time()-t0:.0f}s]", flush=True)

    aggregated = {}
    for name, seed_results in all_seed_results.items():
        aggregated[name] = {
            "avg_accuracy_mean": float(np.mean([r["avg_accuracy"] for r in seed_results])),
            "avg_accuracy_std": float(np.std([r["avg_accuracy"] for r in seed_results])),
            "bwt_mean": float(np.mean([r["bwt"] for r in seed_results])),
            "bwt_std": float(np.std([r["bwt"] for r in seed_results])),
            "fwt_mean": float(np.mean([r["fwt"] for r in seed_results])),
            "fwt_std": float(np.std([r["fwt"] for r in seed_results])),
            "seeds": {
                "avg_accuracy": [r["avg_accuracy"] for r in seed_results],
                "bwt": [r["bwt"] for r in seed_results],
                "fwt": [r["fwt"] for r in seed_results],
            },
        }
    return aggregated


def main() -> None:
    agg = run_sweep()
    with open(RESULTS_DIR / "aggregated.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
