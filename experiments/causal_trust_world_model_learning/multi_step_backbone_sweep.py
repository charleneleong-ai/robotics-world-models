"""Multi-Step Adaptive Trust + Backbone Sweep.

Tests trust scoring across3 backbone architectures:
1. RSSM (DreamerV3-style recurrent state-space model)
2. JEPA (Joint-Embedding Predictive Architecture - no decoder)
3. MLP (simple feedforward world model)

And 4 trust methods:
1. EMA (single-step prediction error)
2. Multi-Step Adaptive (k-step rollout, adaptive horizon)
3. FFDC (Future Forward Dynamics Causal Attention)
4. Ensemble Disagreement

Runs all combinations on ManiSkill Phase2 tasks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

app = typer.Typer(help="Multi-step trust + backbone sweep.")


# ---------------------------------------------------------------------------
# BACKBONE 1: RSSM (existing, cleaned up)
# ---------------------------------------------------------------------------

class RSSMBackbone(nn.Module):
    """Recurrent State-Space Model backbone."""

    def __init__(self, obs_dim: int, act_dim: int, h_dim: int = 256,
                 z_dim: int = 8, z_classes: int = 16, det_dim: int = 256):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.z_dim, self.z_classes, self.det_dim = z_dim, z_classes, det_dim
        self.z_flat = z_dim * z_classes

        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, h_dim), nn.SiLU())
        self.act_enc = nn.Sequential(nn.Linear(act_dim, h_dim), nn.SiLU())

        self.dyn_input = nn.Linear(det_dim + self.z_flat + h_dim, h_dim)
        self.gru = nn.GRUCell(h_dim, det_dim)

        self.prior_net = nn.Sequential(nn.Linear(det_dim, h_dim), nn.SiLU(),
                                       nn.Linear(h_dim, self.z_flat))
        self.post_net = nn.Sequential(nn.Linear(det_dim + h_dim, h_dim), nn.SiLU(),
                                      nn.Linear(h_dim, self.z_flat))

        self.obs_dec = nn.Sequential(nn.Linear(det_dim + self.z_flat, h_dim), nn.SiLU(),
                                     nn.Linear(h_dim, obs_dim))
        self.reward_head = nn.Sequential(nn.Linear(det_dim + self.z_flat, h_dim), nn.SiLU(),
                                         nn.Linear(h_dim, 1))

    def initial_state(self, B: int, device: torch.device):
        h = torch.zeros(B, self.det_dim, device=device)
        z = torch.zeros(B, self.z_flat, device=device)
        return h, z

    def _step(self, h, z, act, obs=None):
        a = self.act_enc(act)
        d = self.dyn_input(torch.cat([h, z, a], dim=-1))
        h_new = self.gru(d, h)

        prior_logits = self.prior_net(h_new).view(-1, self.z_dim, self.z_classes)
        if obs is not None:
            o = self.obs_enc(obs)
            post_logits = self.post_net(torch.cat([h_new, o], dim=-1)).view(-1, self.z_dim, self.z_classes)
            logits = post_logits
        else:
            logits = prior_logits

        z_dist = torch.distributions.Categorical(logits=logits)
        z_s = z_dist.sample()
        z_oh = F.one_hot(z_s, self.z_classes).float().view(-1, self.z_flat)

        state = torch.cat([h_new, z_oh], dim=-1)
        obs_pred = self.obs_dec(state)
        rew_pred = self.reward_head(state).squeeze(-1)
        return h_new, z_oh, obs_pred, rew_pred, prior_logits, logits

    def train_loss(self, obs_seq, act_seq):
        B, T = obs_seq.shape[:2]
        device = obs_seq.device
        h, z = self.initial_state(B, device)
        kl_loss = torch.tensor(0.0, device=device)
        obs_loss = torch.tensor(0.0, device=device)

        for t in range(T):
            h, z, obs_pred, _, prior_logits, post_logits = self._step(h, z, act_seq[:, t], obs_seq[:, t])
            prior = torch.distributions.Categorical(logits=prior_logits)
            post = torch.distributions.Categorical(logits=post_logits)
            kl = torch.distributions.kl_divergence(post, prior).sum(-1).mean()
            kl_loss = kl_loss + kl
            obs_loss = obs_loss + F.mse_loss(obs_pred, obs_seq[:, t])

        return kl_loss / T + 0.1 * obs_loss / T

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        h, z = self.initial_state(obs.shape[0], obs.device)
        _, _, obs_pred, _, _, _ = self._step(h, z, act, obs)
        return F.mse_loss(obs_pred, next_obs, reduction="none").mean(dim=-1)


# ---------------------------------------------------------------------------
# BACKBONE 2: JEPA (Joint-Embedding Predictive Architecture)
# ---------------------------------------------------------------------------

class JEPABackbone(nn.Module):
    """JEPA backbone - no decoder, learns abstract dynamics in latent space.
    Trust = consistency between predicted and actual latent embeddings."""

    def __init__(self, obs_dim: int, act_dim: int, h_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim

        # Encoder: obs -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, h_dim), nn.LayerNorm(h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, latent_dim),
        )
        # Projector: latent -> embedding (for target network)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # Predictor: predict embedding from action + current latent
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + act_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, latent_dim),
        )
        # Action-conditioned transition in latent space
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + act_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, latent_dim),
        )
        # Trust head on prediction consistency
        self.trust_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def initial_state(self, B, device):
        return torch.zeros(B, 128, device=device)  # dummy state for compat

    def _encode(self, obs):
        return self.encoder(obs)

    def train_loss(self, obs_seq, act_seq):
        B, T = obs_seq.shape[:2]
        total_loss = torch.tensor(0.0, device=obs_seq.device)

        for t in range(T - 1):
            z_t = self._encode(obs_seq[:, t])
            z_next = self._encode(obs_seq[:, t + 1])
            target = self.projector(z_next).detach()  # target network stop-gradient

            pred = self.predictor(torch.cat([z_t, act_seq[:, t]], dim=-1))
            loss = F.mse_loss(pred, target)
            total_loss = total_loss + loss

        return total_loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        z = self._encode(obs)
        z_next = self._encode(next_obs)
        pred = self.predictor(torch.cat([z, act], dim=-1))
        target = self.projector(z_next)
        return F.mse_loss(pred, target, reduction="none").mean(dim=-1)

    @torch.no_grad()
    def compute_trust(self, obs, act, next_obs):
        z = self._encode(obs)
        z_next = self._encode(next_obs)
        pred = self.predictor(torch.cat([z, act], dim=-1))
        target = self.projector(z_next)
        consistency = F.cosine_similarity(pred, target, dim=-1)
        return ((consistency + 1) / 2).clamp(0, 1)


# ---------------------------------------------------------------------------
# BACKBONE 3: MLP World Model (simplest baseline)
# ---------------------------------------------------------------------------

class MLPBackbone(nn.Module):
    """Simple feedforward world model. No recurrence, no stochastic state."""

    def __init__(self, obs_dim: int, act_dim: int, h_dim: int = 256):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h_dim), nn.LayerNorm(h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, 1),
        )
        self.trust_head = nn.Sequential(
            nn.Linear(obs_dim * 2, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def initial_state(self, B, device):
        return torch.zeros(B, 1, device=device)  # dummy

    def train_loss(self, obs_seq, act_seq):
        B, T = obs_seq.shape[:2]
        total_loss = torch.tensor(0.0, device=obs_seq.device)
        for t in range(T - 1):
            inp = torch.cat([obs_seq[:, t], act_seq[:, t]], dim=-1)
            pred_next = self.net(inp)
            total_loss = total_loss + F.mse_loss(pred_next, obs_seq[:, t + 1])
        return total_loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        inp = torch.cat([obs, act], dim=-1)
        pred = self.net(inp)
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)

    @torch.no_grad()
    def compute_trust(self, obs, act, next_obs):
        inp = torch.cat([obs, act], dim=-1)
        pred = self.net(inp)
        err = F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)
        return torch.exp(-err).clamp(0, 1)


# ---------------------------------------------------------------------------
# TRUST METHOD 1: EMA (single-step, baseline)
# ---------------------------------------------------------------------------

class EMATrust:
    def __init__(self, alpha: float = 1.0, ema: float = 0.95):
        self.alpha, self.ema, self.errors = alpha, ema, {}

    def compute_trust(self, error: torch.Tensor, task_id: int) -> torch.Tensor:
        e = float(error.mean())
        self.errors[task_id] = self.ema * self.errors.get(task_id, e) + (1 - self.ema) * e
        return torch.exp(-self.alpha * error / (self.errors[task_id] + 1e-8)).clamp(0, 1)


# ---------------------------------------------------------------------------
# TRUST METHOD 2: Multi-Step Adaptive Trust
# ---------------------------------------------------------------------------

class MultiStepAdaptiveTrust:
    """Multi-step rollout trust verification (FFDC-inspired).
    
    At each step, verify consistency over k-step imagined rollouts.
    Adaptive k: expand when consistent, contract when divergent.
    """

    def __init__(self, backbone: nn.Module, max_k: int = 8, 
                 expand_threshold: float = 0.8, contract_threshold: float = 0.5,
                 alpha: float = 1.0):
        self.backbone = backbone
        self.max_k = max_k
        self.expand_threshold = expand_threshold
        self.contract_threshold = contract_threshold
        self.alpha = alpha
        self.current_k: dict[int, int] = {}  # per-task horizon

    def _rollout_error(self, obs: torch.Tensor, acts: torch.Tensor, task_id: int) -> torch.Tensor:
        """Compute k-step rollout prediction error."""
        k = self.current_k.get(task_id, 1)
        B = obs.shape[0]
        device = obs.device
        
        if hasattr(self.backbone, 'rssm'):
            # RSSM backbone
            h, z = self.backbone.initial_state(B, device)
            total_err = torch.zeros(B, device=device)
            h_cur, z_cur = h, z
            for step in range(min(k, acts.shape[1] if acts.dim() > 2 else 1)):
                if acts.dim() == 2:
                    a = acts
                else:
                    a = acts[:, step]
                h_cur, z_cur, obs_pred, _, _, _ = self.backbone._step(h_cur, z_cur, a, obs if step == 0 else None)
                total_err = total_err + F.mse_loss(obs_pred, obs, reduction="none").mean(dim=-1)
            return total_err / k
        else:
            # JEPA or MLP backbone
            err = self.backbone.predict_error(obs, acts[:, 0] if acts.dim() > 2 else acts, obs)
            return err

    def compute_trust(self, error: torch.Tensor, task_id: int) -> torch.Tensor:
        k = self.current_k.get(task_id, 1)
        e = float(error.mean())

        # Adaptive horizon adjustment
        if e < self.expand_threshold:
            self.current_k[task_id] = min(k + 1, self.max_k)
        elif e > self.contract_threshold:
            self.current_k[task_id] = max(k - 1, 1)

        return torch.exp(-self.alpha * error).clamp(0, 1)

    def get_horizon(self, task_id: int) -> int:
        return self.current_k.get(task_id, 1)


# ---------------------------------------------------------------------------
# TRUST METHOD 3: FFDC (Future Forward Dynamics Causal Attention)
# ---------------------------------------------------------------------------

class FFDCMultiStep:
    """FFDC-style verifier trained on k-step rollout data."""

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int = 128):
        self.verifier = nn.Sequential(
            nn.Linear(obs_dim * 2 + latent_dim + act_dim, 128),
            nn.SiLU(), nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.opt = torch.optim.Adam(self.verifier.parameters(), lr=1e-3)

    def train_step(self, pred_obs, actual_obs, latent, action, labels):
        inp = torch.cat([pred_obs, actual_obs, latent, action], dim=-1)
        pred = self.verifier(inp).squeeze(-1)
        loss = F.binary_cross_entropy(pred, labels.float())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    @torch.no_grad()
    def compute_trust(self, pred_obs, actual_obs, latent, action):
        inp = torch.cat([pred_obs, actual_obs, latent, action], dim=-1)
        return self.verifier(inp).squeeze(-1).clamp(0, 1)


# ---------------------------------------------------------------------------
# TRUST METHOD 4: Ensemble Disagreement
# ---------------------------------------------------------------------------

class EnsembleDisagreement:
    def __init__(self, obs_dim: int, n_heads: int = 5):
        self.heads = nn.ModuleList([nn.Linear(obs_dim, obs_dim) for _ in range(n_heads)])
        self.opt = torch.optim.Adam(self.heads.parameters(), lr=1e-3)

    def train_step(self, features, targets):
        preds = torch.stack([h(features) for h in self.heads], dim=0)
        loss = F.mse_loss(preds, targets.unsqueeze(0))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    @torch.no_grad()
    def compute_trust(self, features):
        preds = torch.stack([h(features) for h in self.heads], dim=0)
        variance = preds.var(dim=0).mean(dim=-1)
        return torch.exp(-variance).clamp(0, 1)


# ---------------------------------------------------------------------------
# POLICY
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

class TrustWeightedEWC:
    def __init__(self, model, lam: float = 1000.0):
        self.model, self.lam = model, lam
        self.fisher, self.optimal = {}, {}

    def consolidate(self):
        self.optimal = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.fisher = {n: torch.ones_like(p) for n, p in self.model.named_parameters()}

    def penalty(self):
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            loss += (self.fisher[n] * (p - self.optimal[n]) ** 2).sum()
        return self.lam * loss


# ---------------------------------------------------------------------------
# EXPERIMENT RUNNER
# ---------------------------------------------------------------------------

def run_experiment(
    backbone_name: str,
    trust_name: str,
    task_names: list[str],
    n_episodes: int = 15,
    max_steps: int = 50,
    n_epochs_per_task: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """Run one (backbone, trust) combination."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create first env to get dims
    env = gym.make(task_names[0], render_mode=None)
    sample_obs = np.asarray(env.reset()[0], dtype=np.float32).flatten()
    obs_dim = sample_obs.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    # Create backbone
    if backbone_name == "rssm":
        backbone = RSSMBackbone(obs_dim, act_dim).to(dev)
    elif backbone_name == "jepa":
        backbone = JEPABackbone(obs_dim, act_dim).to(dev)
    elif backbone_name == "mlp":
        backbone = MLPBackbone(obs_dim, act_dim).to(dev)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Create trust method
    if trust_name == "ema":
        trust = EMATrust()
    elif trust_name == "multi_step":
        trust = MultiStepAdaptiveTrust(backbone)
    elif trust_name == "ffdc":
        trust = FFDCMultiStep(obs_dim, act_dim)
    elif trust_name == "ensemble":
        trust = EnsembleDisagreement(obs_dim)
    elif trust_name == "none":
        trust = None
    else:
        raise ValueError(f"Unknown trust: {trust_name}")

    policy = Policy(obs_dim, act_dim).to(dev)
    ewc = TrustWeightedEWC(policy)

    all_task_accs = []

    for task_idx, task_name in enumerate(task_names):
        # Train backbone on this task's data
        bb_opt = torch.optim.Adam(backbone.parameters(), lr=lr)
        env = gym.make(task_name, render_mode=None)

        # Collect data for backbone training
        for _ in range(n_epochs_per_task):
            obs_data, act_data = [], []
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=task_idx * 1000 + ep)
                obs = np.asarray(obs, dtype=np.float32).flatten()
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                for _ in range(max_steps):
                    with torch.no_grad():
                        act = policy(obs_t)
                    next_obs, _, term, trunc, _ = env.step(act.cpu().numpy().flatten())
                    next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
                    obs_data.append(obs_t.squeeze(0))
                    act_data.append(act.squeeze(0))
                    obs_t = torch.from_numpy(next_obs).float().unsqueeze(0).to(dev)
                    if term or trunc:
                        break

            if len(obs_data) > 2:
                obs_t = torch.stack(obs_data)
                act_t = torch.stack(act_data)
                # Train in mini-batches
                for i in range(0, len(obs_data) - 1, 32):
                    chunk = min(32, len(obs_data) - 1 - i)
                    loss = backbone.train_loss(obs_t[i:i+chunk], act_t[i:i+chunk])
                    bb_opt.zero_grad()
                    loss.backward()
                    bb_opt.step()

        # Consolidate EWC
        ewc.consolidate()

        # Train policy
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=task_idx * 1000 + ep + 5000)
            obs = np.asarray(obs, dtype=np.float32).flatten()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)

            for step in range(max_steps):
                action = policy(obs_t)
                next_obs, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
                next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
                next_obs_t = torch.from_numpy(next_obs).float().unsqueeze(0).to(dev)

                # Compute trust weight
                trust_weight = 1.0
                if trust is not None:
                    with torch.no_grad():
                        err = backbone.predict_error(obs_t, action, next_obs_t)
                        if hasattr(trust, 'compute_trust') and trust_name == "multi_step":
                            tw = trust.compute_trust(err, task_idx)
                        elif trust_name == "ema":
                            tw = trust.compute_trust(err, task_idx)
                        elif trust_name == "ffdc":
                            tw = trust.compute_trust(obs_t, next_obs_t, action, action)
                        elif trust_name == "ensemble":
                            tw = trust.compute_trust(obs_t.squeeze(0))
                        else:
                            tw = torch.tensor(0.5)
                        trust_weight = float(tw.mean())

                # Simple policy gradient
                target = torch.from_numpy(env.action_space.sample()).float().to(dev).unsqueeze(0)
                loss = F.mse_loss(action, target) * trust_weight + ewc.penalty()
                policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
                policy_opt.zero_grad()
                loss.backward()
                policy_opt.step()

                obs_t = next_obs_t
                if term or trunc:
                    break

        # Evaluate
        eval_rewards = []
        for ep in range(10):
            obs, _ = env.reset(seed=task_idx * 10000 + ep + 90000)
            obs = np.asarray(obs, dtype=np.float32).flatten()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            total_r = 0
            for _ in range(max_steps):
                with torch.no_grad():
                    act = policy(obs_t)
                next_obs, r, term, trunc, _ = env.step(act.cpu().numpy().flatten())
                total_r += r
                obs_t = torch.from_numpy(np.asarray(next_obs, dtype=np.float32).flatten()).float().unsqueeze(0).to(dev)
                if term or trunc:
                    break
            eval_rewards.append(total_r)

        all_task_accs.append(float(np.mean(eval_rewards)))
        env.close()

        print(f"  Task {task_idx} ({task_name}): avg_reward={np.mean(eval_rewards):.3f}")

    avg = float(np.mean(all_task_accs))
    return {
        "backbone": backbone_name,
        "trust": trust_name,
        "task_rewards": all_task_accs,
        "avg_reward": avg,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

@app.main
def main(
    output: str = "sweep_results.json",
    n_episodes: int = typer.Option(15, help="Episodes per task"),
    max_steps: int = typer.Option(50, help="Max steps per episode"),
):
    """Run full backbone x trust sweep."""
    # ManiSkill Phase2 tasks (the harder ones where we lag ER)
    tasks = ["PushCube-v1", "LiftPegUpright-v1", "StackCube-v1"]

    backbones = ["rssm", "jepa", "mlp"]
    trusts = ["ema", "multi_step", "ffdc", "ensemble", "none"]

    results = []
    total = len(backbones) * len(trusts)
    idx = 0

    for bb in backbones:
        for tr in trusts:
            idx += 1
            print(f"\n=== [{idx}/{total}] {bb} + {tr} ===")
            t0 = time.time()
            try:
                r = run_experiment(bb, tr, tasks, n_episodes=n_episodes, max_steps=max_steps)
                r["time_sec"] = time.time() - t0
                results.append(r)
                print(f"  -> avg_reward={r['avg_reward']:.3f} ({r['time_sec']:.0f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"backbone": bb, "trust": tr, "error": str(e)})

    # Save results
    out_path = Path(output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{"Backbone":<10} {"Trust":<15} {"Avg Reward":>10}")
    print("-" * 40)
    for r in results:
        if "error" not in r:
            print(f"{r['backbone']:<10} {r['trust']:<15} {r['avg_reward']:>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    app()
