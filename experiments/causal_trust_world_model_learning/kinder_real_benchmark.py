"""KinDER Benchmark v3 — Permuted-MNIST style CL evaluation.

Each task = real KinDER environment observations + a different random
feature permutation. All tasks share the same 10-class label space.
Labels come from the raw (unpermuted) features via a fixed random projection.

This creates genuine forgetting: learning Task 2's permutation mapping
overwrites Task 0's mapping because they use the same network weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from collections import deque
import copy

import kinder
kinder.register_all_environments()


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return self.net(x)


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


def _flatten(obs, dim):
    if isinstance(obs, dict):
        flat = np.concatenate([v.flatten() for v in obs.values() if isinstance(v, np.ndarray)])
    else:
        flat = np.asarray(obs, np.float32).flat[:]
    if len(flat) < dim:
        flat = np.pad(flat, (0, dim - len(flat)))
    return flat[:dim].astype(np.float32)


def collect_env(env_name, n_episodes, max_steps, obs_dim):
    try:
        env = kinder.make(env_name)
    except Exception as e:
        print(f"  WARN {env_name}: {e}")
        return np.random.randn(n_episodes * max_steps, obs_dim).astype(np.float32) * 0.1

    obs_list = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            obs_list.append(_flatten(obs, obs_dim))
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    env.close()
    return np.array(obs_list, np.float32)


def make_tasks(raw_obs_list, num_tasks, num_classes, obs_dim, seed=0):
    """Create permuted tasks from real observations.

    - W_label: fixed random projection for labeling (shared across all tasks)
    - perms[i]: random permutation of features for task i
    - Task i input = raw_obs[:, perms[i]]
    - Task i label = (raw_obs @ W_label).argmax  (same label space for all)
    """
    rng = np.random.RandomState(seed)

    # Fixed label projection (shared)
    W_label = rng.randn(obs_dim, num_classes).astype(np.float32) * 0.3

    # Per-task permutations
    perms = [rng.permutation(obs_dim) for _ in range(num_tasks)]

    tasks = []
    for tid, obs_arr in enumerate(raw_obs_list):
        # Labels from raw observations (same for all tasks)
        labels = (obs_arr @ W_label).argmax(axis=1)
        # Input is permuted version
        perm_obs = obs_arr[:, perms[tid]]

        tasks.append({
            "obs": torch.tensor(perm_obs, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    return tasks


def train_epoch(learner, dataset, batch_size, device):
    indices = torch.randperm(len(dataset["obs"]))
    total_loss, total_acc, n = 0, 0, 0
    for start in range(0, len(indices), batch_size):
        bi = indices[start:start + batch_size]
        batch = {
            "obs": dataset["obs"][bi].to(device),
            "labels": dataset["labels"][bi].to(device),
        }
        m = learner.observe(batch)
        total_loss += m["loss"]
        total_acc += m["acc"]
        n += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


def evaluate(learner, dataset, device):
    learner.model.eval()
    with torch.no_grad():
        obs = dataset["obs"].to(device)
        labels = dataset["labels"].to(device)
        logits = learner.model(obs)
        return (logits.argmax(-1) == labels).float().mean().item()


# ======================== CL METHODS ========================

class FineTuningCL:
    name = "Fine-tuning"
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        logits = self.model(obs)
        loss = F.cross_entropy(logits, lbl)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid): pass


class EWCCL:
    name = "EWC"
    def __init__(self, model, device, lam=5000.0):
        self.model, self.device, self.lam = model, device, lam
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.fisher, self.star = {}, {}

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        logits = self.model(obs)
        loss = F.cross_entropy(logits, lbl)
        if self.fisher:
            pen = torch.tensor(0.0, device=self.device)
            for tid, f in self.fisher.items():
                for n, p in self.model.named_parameters():
                    if n in f:
                        pen += (f[n] * (p - self.star[tid][n]).pow(2)).sum()
            loss += self.lam * pen
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        in_dim = self.model.net[0].in_features
        fisher = {}
        self.model.zero_grad()
        d = torch.randn(128, in_dim, device=self.device)
        loss = F.cross_entropy(self.model(d), torch.randint(0, 10, (128,), device=self.device))
        loss.backward()
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                fisher[n] = p.grad.data.pow(2).clone()
        self.fisher[tid] = fisher
        self.star[tid] = {n: p.data.clone() for n, p in self.model.named_parameters()}


class LwFCL:
    name = "LwF"
    def __init__(self, model, device, alpha=0.5, T=2.0):
        self.model, self.device = model, device
        self.alpha, self.T = alpha, T
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.prev = None

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        logits = self.model(obs)
        loss_new = F.cross_entropy(logits, lbl)
        if self.prev is not None:
            self.prev.eval()
            with torch.no_grad():
                prev_logits = self.prev(obs)
            loss_kd = F.kl_div(
                F.log_softmax(logits / self.T, -1),
                F.softmax(prev_logits / self.T, -1),
                reduction="batchmean",
            ) * (self.T ** 2)
            loss = (1 - self.alpha) * loss_new + self.alpha * loss_kd
        else:
            loss = loss_new
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        self.prev = copy.deepcopy(self.model)


class ExperienceReplayCL:
    name = "Experience Replay"
    def __init__(self, model, device, buf_size=500):
        self.model, self.device = model, device
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.buf = deque(maxlen=buf_size)

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        for i in range(len(obs)):
            self.buf.append({"obs": obs[i].cpu(), "lbl": lbl[i].cpu()})
        if len(self.buf) > 128:
            s = list(self.buf)
            idx = np.random.choice(len(s), min(len(obs) * 2, len(s)), replace=False)
            ro = torch.stack([s[i]["obs"] for i in idx]).to(self.device)
            rl = torch.stack([s[i]["lbl"] for i in idx]).to(self.device)
            all_obs, all_lbl = torch.cat([obs, ro]), torch.cat([lbl, rl])
        else:
            all_obs, all_lbl = obs, lbl
        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_lbl)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits[:len(obs)].argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid): pass


class PrioritizedReplayCL:
    name = "Prioritized Replay"
    def __init__(self, model, device, buf_size=500, alpha=0.6):
        self.model, self.device, self.alpha = model, device, alpha
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.buf = deque(maxlen=buf_size)
        self.pris = deque(maxlen=buf_size)

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        with torch.no_grad():
            sl = F.cross_entropy(self.model(obs), lbl, reduction="none")
        for i in range(len(obs)):
            self.buf.append({"obs": obs[i].cpu(), "lbl": lbl[i].cpu()})
            self.pris.append(sl[i].item() + 1e-6)
        if len(self.buf) > 128:
            s = list(self.buf)
            p = np.array(list(self.pris)) ** self.alpha
            p /= p.sum()
            idx = np.random.choice(len(s), min(len(obs) * 2, len(s)), p=p, replace=True)
            ro = torch.stack([s[i]["obs"] for i in idx]).to(self.device)
            rl = torch.stack([s[i]["lbl"] for i in idx]).to(self.device)
            all_obs, all_lbl = torch.cat([obs, ro]), torch.cat([lbl, rl])
        else:
            all_obs, all_lbl = obs, lbl
        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_lbl)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits[:len(obs)].argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid): pass


class WMTrustCL:
    name = "WM Trust CL"
    def __init__(self, model, wm, device, lam=5000.0):
        self.model, self.wm, self.device, self.lam = model, wm, device, lam
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        self.fisher, self.star, self.trust_map = {}, {}, {}

    def observe(self, batch):
        self.model.train()
        self.wm.train()
        obs = batch["obs"].to(self.device)
        lbl = batch["labels"].to(self.device)
        act_dim = self.wm.net[0].in_features - obs.shape[1]
        act = torch.randn(len(obs), act_dim, device=self.device)

        self.wm_opt.zero_grad()
        wm_loss = F.mse_loss(self.wm(obs, act), obs)
        wm_loss.backward()
        self.wm_opt.step()

        self.opt.zero_grad()
        logits = self.model(obs)
        ce = F.cross_entropy(logits, lbl, reduction="none")
        with torch.no_grad():
            err = F.mse_loss(self.wm(obs, act), obs, reduction="none").mean(dim=-1)
            trust = torch.exp(-err)
        loss = (trust * ce).mean()

        if self.fisher:
            pen = torch.tensor(0.0, device=self.device)
            for tid, f in self.fisher.items():
                t = self.trust_map.get(tid, 0.5)
                for n, p in self.model.named_parameters():
                    if n in f:
                        pen += self.lam * t * (f[n] * (p - self.star[tid][n]).pow(2)).sum()
            loss += pen
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        in_dim = self.model.net[0].in_features
        fisher = {}
        self.model.zero_grad()
        d = torch.randn(128, in_dim, device=self.device)
        loss = F.cross_entropy(self.model(d), torch.randint(0, 10, (128,), device=self.device))
        loss.backward()
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                fisher[n] = p.grad.data.pow(2).clone()
        self.fisher[tid] = fisher
        self.star[tid] = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.trust_map[tid] = 0.5


class ContinualWAM:
    name = "ContinualWAM (ours)"
    def __init__(self, model, wm, device, buf_size=500, lam=5000.0):
        self.model, self.wm, self.device = model, wm, device
        self.buf = deque(maxlen=buf_size)
        self.lam = lam
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        self.fisher, self.star, self.trust_map = {}, {}, {}
        self.ema_trust = 0.5

    def observe(self, batch):
        self.model.train()
        self.wm.train()
        obs = batch["obs"].to(self.device)
        lbl = batch["labels"].to(self.device)
        act_dim = self.wm.net[0].in_features - obs.shape[1]
        act = torch.randn(len(obs), act_dim, device=self.device)

        self.wm_opt.zero_grad()
        wm_loss = F.mse_loss(self.wm(obs, act), obs)
        wm_loss.backward()
        self.wm_opt.step()

        self.opt.zero_grad()
        with torch.no_grad():
            err = F.mse_loss(self.wm(obs, act), obs, reduction="none").mean(dim=-1)
            trust = torch.exp(-err)
            self.ema_trust = 0.9 * self.ema_trust + 0.1 * trust.mean().item()

        for i in range(len(obs)):
            self.buf.append({"obs": obs[i].cpu(), "lbl": lbl[i].cpu(), "trust": trust[i].item()})

        if len(self.buf) > 128:
            s = list(self.buf)
            t = np.clip(np.array([x["trust"] for x in s]), 1e-6, None)
            p = t / t.sum()
            idx = np.random.choice(len(s), min(len(obs) * 2, len(s)), p=p, replace=True)
            ro = torch.stack([s[i]["obs"] for i in idx]).to(self.device)
            rl = torch.stack([s[i]["lbl"] for i in idx]).to(self.device)
            all_obs, all_lbl = torch.cat([obs, ro]), torch.cat([lbl, rl])
        else:
            all_obs, all_lbl = obs, lbl

        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_lbl)

        if self.fisher:
            pen = torch.tensor(0.0, device=self.device)
            for tid, f in self.fisher.items():
                t = self.trust_map.get(tid, 0.5)
                adaptive_lam = self.lam * (1 - t)
                for n, p in self.model.named_parameters():
                    if n in f:
                        pen += adaptive_lam * (f[n] * (p - self.star[tid][n]).pow(2)).sum()
            loss += pen
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits[:len(obs)].argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        in_dim = self.model.net[0].in_features
        fisher = {}
        self.model.zero_grad()
        d = torch.randn(128, in_dim, device=self.device)
        loss = F.cross_entropy(self.model(d), torch.randint(0, 10, (128,), device=self.device))
        loss.backward()
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                fisher[n] = p.grad.data.pow(2).clone()
        self.fisher[tid] = fisher
        self.star[tid] = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.trust_map[tid] = self.ema_trust


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    obs_dim = 100
    num_tasks = 3
    num_classes = 10
    n_episodes = 30
    max_steps = 100
    epochs_per_task = 15

    envs = [
        "kinder/Motion2D-p0-v0",
        "kinder/StickButton2D-b1-v0",
        "kinder/Obstruction2D-o1-v0",
    ]

    print("Collecting real KinDER observations...")
    raw_obs_list = []
    for name in envs:
        print(f"  {name}", end="...", flush=True)
        data = collect_env(name, n_episodes, max_steps, obs_dim)
        print(f" {len(data)} obs")
        raw_obs_list.append(data)

    print(f"\nCreating permuted tasks ({num_classes} classes, {num_tasks} tasks)...")
    tasks = make_tasks(raw_obs_list, num_tasks, num_classes, obs_dim)
    for i, t in enumerate(tasks):
        dist = torch.bincount(t["labels"], minlength=num_classes).tolist()
        print(f"  Task {i}: {len(t['obs'])} samples, classes={dist}")

    def mk_model():
        return MLP(obs_dim, 256, num_classes).to(device)
    def mk_wm():
        return WorldModel(obs_dim, 10, 256).to(device)

    methods = [
        FineTuningCL(mk_model(), device),
        EWCCL(mk_model(), device, lam=5000),
        LwFCL(mk_model(), device),
        ExperienceReplayCL(mk_model(), device, buf_size=500),
        PrioritizedReplayCL(mk_model(), device, buf_size=500),
        WMTrustCL(mk_model(), mk_wm(), device, lam=5000),
        ContinualWAM(mk_model(), mk_wm(), device, buf_size=500, lam=5000),
    ]

    all_results = {}
    for learner in methods:
        print(f"\n{'='*60}\n{learner.name}\n{'='*60}")
        task_accs = []

        for tid in range(num_tasks):
            ds = {"obs": tasks[tid]["obs"], "labels": tasks[tid]["labels"]}

            for ep in range(epochs_per_task):
                train_epoch(learner, ds, 64, device)

            acc = evaluate(learner, ds, device)
            task_accs.append(acc)

            prev_accs = []
            for ptid in range(tid):
                prev_ds = {"obs": tasks[ptid]["obs"], "labels": tasks[ptid]["labels"]}
                prev_accs.append(evaluate(learner, prev_ds, device))

            prev_str = " ".join(f"T{p}={a:.3f}" for p, a in enumerate(prev_accs)) if prev_accs else "---"
            print(f"  T{tid}: train={acc:.4f}  prev=[{prev_str}]")

            learner.on_task_done(tid)

        avg = np.mean(task_accs)
        bwt = sum(task_accs[i] - task_accs[0] for i in range(1, len(task_accs))) / max(len(task_accs) - 1, 1)

        all_results[learner.name] = {
            "task_accuracies": [float(a) for a in task_accs],
            "avg_accuracy": float(avg),
            "bwt": float(bwt),
        }

    out_dir = "kinder_real_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kinder_real_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}\nKinDER Real Data Results (Permuted Tasks)\n{'='*80}")
    print(f"{'Method':<30} {'AvgAcc':>8} {'BWT':>8}  Tasks")
    print("-" * 80)
    for name, res in all_results.items():
        tacc = " ".join(f"{a:.3f}" for a in res["task_accuracies"])
        print(f"{name:<30} {res['avg_accuracy']:>8.4f} {res['bwt']:>8.4f}  [{tacc}]")


if __name__ == "__main__":
    main()
