#!/usr/bin/env python3
"""Real Failure-Recovery Experiment (ManiSkill).

Trains a small online world model (MLP predicting next_obs from obs+action)
on REAL transitions collected from real ManiSkill env rollouts (via
ManiSkillBenchmark.collect_task_data, which calls gym.make(...) and
env.step(...) directly -- no synthetic np.random.randn() stand-in for
observations or rewards).

A "failure" is injected by corrupting a fraction of REAL observations in a
training batch with heavy Gaussian noise before the batch is used for
training (a real distribution-shift / sensor-failure event, not a scripted
outcome).

Three methods, all training on the SAME real batches:
  - fine_tuning: always takes the gradient step, corrupted batch or not.
  - ewc: fine_tuning + an EWC penalty toward the previous task's optimum.
  - continual_wam: uses TrustScorer (the same class used elsewhere in this
    codebase) on the model's own prediction error; if trust falls below
    `trust_threshold`, the gradient update for that batch is skipped
    (rather than being permanently protected -- it is retried on the next,
    hopefully-clean batch).

"Recovery" is measured on a held-out set of CLEAN validation transitions:
for every injected failure, we record whether validation MSE returns to
within `recovery_tol` (relative) of its pre-failure value within the next
`recovery_window` training steps. recovery_rate = recovered / total_failures.
"""
from __future__ import annotations

import json
import sys
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maniskill_benchmark import ManiSkillBenchmark
from trust_scoring import TrustScorer

RESULTS_DIR = Path(__file__).parent / "results_failure_recovery_real"
RESULTS_DIR.mkdir(exist_ok=True)

TASKS = ["PushCube-v1", "LiftPegUpright-v1", "PlugCharger-v1", "StackCube-v1", "PokeCube-v1"]
FAILURE_RATES = [0.1, 0.3, 0.5]
METHODS = ["fine_tuning", "ewc", "continual_wam"]
NUM_SEEDS = 5
BATCH_SIZE = 16


class WorldModelMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


@dataclass
class _PendingFailure:
    step: int
    pre_mse: float


class FailureRecoveryExperiment:
    """One (method, failure_rate, seed) run of the online failure-recovery test.

    Owns all mutable training state (model, optimizer, trust scorer, EWC
    anchors, recovery bookkeeping) as instance attributes so `run()` reads as
    a plain sequential procedure instead of threading a dozen locals through
    free functions.
    """

    def __init__(
        self,
        bench: ManiSkillBenchmark,
        method: str,
        failure_rate: float,
        batch_size: int = BATCH_SIZE,
        recovery_window: int = 5,
        recovery_tol: float = 0.20,
        ewc_lambda: float = 1000.0,
        trust_threshold: float = 0.5,
        corruption_scale: float = 3.0,
    ):
        self.bench = bench
        self.method = method
        self.failure_rate = failure_rate
        self.batch_size = batch_size
        self.recovery_window = recovery_window
        self.recovery_tol = recovery_tol
        self.ewc_lambda = ewc_lambda
        self.corruption_scale = corruption_scale

        self.model = WorldModelMLP(bench.obs_dim, bench.action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.trust_scorer = TrustScorer(trust_threshold=trust_threshold) if method == "continual_wam" else None

        self.ewc_anchors: list[tuple[dict, dict]] = []  # (param_star, fisher) per completed task
        self.pending_failures: list[_PendingFailure] = []
        self.val_mse_history: list[float] = []
        self.total_failures = 0
        self.recovered_failures = 0
        self.global_step = 0

    def run(self, seed: int) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for task_id, env_name in enumerate(TASKS):
            self._run_task(task_id, env_name)
        self._resolve_remaining_failures()
        return self._summary()

    def _run_task(self, task_id: int, env_name: str) -> None:
        data = self.bench.collect_task_data(env_name, num_episodes=self.bench.episodes_per_task, policy_fn=None)
        obs = torch.tensor(np.asarray(data["observations"]), dtype=torch.float32)
        act = torch.tensor(np.asarray(data["actions"]), dtype=torch.float32)
        nobs = torch.tensor(np.asarray(data["next_observations"]), dtype=torch.float32)

        n_val = max(self.batch_size, len(obs) // 5)
        perm = torch.randperm(len(obs))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        obs_tr, act_tr, nobs_tr = obs[train_idx], act[train_idx], nobs[train_idx]
        self.obs_val, self.act_val, self.nobs_val = obs[val_idx], act[val_idx], nobs[val_idx]

        for obs_b, act_b, nobs_b in self._batches(obs_tr, act_tr, nobs_tr):
            self._train_step(task_id, obs_b, act_b, nobs_b)

        if self.method == "ewc":
            self.ewc_anchors.append(self._compute_fisher(obs_tr, act_tr, nobs_tr))

    def _batches(self, obs: torch.Tensor, act: torch.Tensor, nobs: torch.Tensor):
        idx = torch.randperm(len(obs))
        for start in range(0, len(obs) - self.batch_size + 1, self.batch_size):
            b = idx[start:start + self.batch_size]
            yield obs[b], act[b], nobs[b]

    def _train_step(self, task_id: int, obs_b: torch.Tensor, act_b: torch.Tensor, nobs_b: torch.Tensor) -> None:
        is_failure = np.random.random() < self.failure_rate
        if is_failure:
            obs_b = obs_b + torch.randn_like(obs_b) * self.corruption_scale
            self.total_failures += 1

        pred_error = self._eval_loss(obs_b, act_b, nobs_b)
        if self._should_skip_update(task_id, pred_error):
            pass
        else:
            self._gradient_step(obs_b, act_b, nobs_b)

        val_mse = self._eval_loss(self.obs_val, self.act_val, self.nobs_val)
        self.val_mse_history.append(val_mse)

        if is_failure:
            pre_mse = self.val_mse_history[-2] if len(self.val_mse_history) > 1 else val_mse
            self.pending_failures.append(_PendingFailure(step=self.global_step, pre_mse=pre_mse))
        self._check_recoveries()
        self.global_step += 1

    def _should_skip_update(self, task_id: int, pred_error: float) -> bool:
        if self.trust_scorer is None:
            return False
        trust = self.trust_scorer.compute_trust(
            torch.tensor([pred_error]), torch.tensor([1.0]), task_id
        ).item()
        return trust < self.trust_scorer.trust_threshold

    def _gradient_step(self, obs_b: torch.Tensor, act_b: torch.Tensor, nobs_b: torch.Tensor) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(self.model(obs_b, act_b), nobs_b)
        if self.method == "ewc":
            loss = loss + self._ewc_penalty()
        loss.backward()
        self.optimizer.step()

    def _ewc_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0)
        for p_star, fisher in self.ewc_anchors:
            for name, p in self.model.named_parameters():
                if name in p_star:
                    penalty = penalty + (self.ewc_lambda / 2) * (fisher[name] * (p - p_star[name]).pow(2)).sum()
        return penalty

    def _compute_fisher(self, obs_tr: torch.Tensor, act_tr: torch.Tensor, nobs_tr: torch.Tensor,
                         num_samples: int = 50) -> tuple[dict, dict]:
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        n = min(num_samples, len(obs_tr))
        self.model.eval()
        for i in range(n):
            self.model.zero_grad()
            loss = torch.nn.functional.mse_loss(
                self.model(obs_tr[i:i + 1], act_tr[i:i + 1]), nobs_tr[i:i + 1]
            )
            loss.backward()
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.data.pow(2) / n
        p_star = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        return p_star, fisher

    def _eval_loss(self, obs: torch.Tensor, act: torch.Tensor, nobs: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            return torch.nn.functional.mse_loss(self.model(obs, act), nobs).item()

    def _check_recoveries(self) -> None:
        resolved = []
        for pf in self.pending_failures:
            if self.global_step - pf.step >= self.recovery_window:
                best_since = min(self.val_mse_history[pf.step:self.global_step + 1])
                if best_since <= pf.pre_mse * (1 + self.recovery_tol):
                    self.recovered_failures += 1
                resolved.append(pf)
        for pf in resolved:
            self.pending_failures.remove(pf)

    def _resolve_remaining_failures(self) -> None:
        for pf in self.pending_failures:
            end = min(pf.step + self.recovery_window, len(self.val_mse_history) - 1)
            best_since = min(self.val_mse_history[pf.step:end + 1])
            if best_since <= pf.pre_mse * (1 + self.recovery_tol):
                self.recovered_failures += 1
        self.pending_failures.clear()

    def _summary(self) -> dict:
        rate = self.recovered_failures / self.total_failures if self.total_failures > 0 else 0.0
        return {
            "recovery_rate": rate,
            "total_failures": self.total_failures,
            "recovered_failures": self.recovered_failures,
            "avg_val_mse": float(np.mean(self.val_mse_history)),
        }


def run_sweep() -> dict:
    bench = ManiSkillBenchmark(num_tasks=len(TASKS), episodes_per_task=15, max_steps=60,
                                obs_dim=64, action_dim=10)
    results: dict = {m: {} for m in METHODS}
    t0 = time.time()
    for method in METHODS:
        for fr in FAILURE_RATES:
            seed_results = [FailureRecoveryExperiment(bench, method, fr).run(seed) for seed in range(NUM_SEEDS)]
            for seed, r in enumerate(seed_results):
                print(f"{method:14s} FR={fr} seed={seed} recovery={r['recovery_rate']:.3f} "
                      f"({r['recovered_failures']}/{r['total_failures']}) [{time.time() - t0:.0f}s]", flush=True)
            rates = [r["recovery_rate"] for r in seed_results]
            results[method][str(fr)] = {
                "avg_recovery_rate": float(np.mean(rates)),
                "std_recovery_rate": float(np.std(rates)),
                "seeds": rates,
                "total_failures_per_seed": [r["total_failures"] for r in seed_results],
            }
    return results


def main() -> None:
    results = run_sweep()
    with open(RESULTS_DIR / "failure_recovery_real_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
