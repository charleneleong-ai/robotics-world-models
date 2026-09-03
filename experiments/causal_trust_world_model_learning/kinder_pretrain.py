#!/usr/bin/env python3
"""
KinDER: Dynamics-based world model pretraining + trust-weighted MPC.
Self-contained - no mani_skill import needed.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ""

import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import kinder
kinder.register_all_environments()
import gymnasium as gym


# ============================================================================
# BACKBONE CLASSES (inlined to avoid mani_skill import)
# ============================================================================

class WorldModelBackbone(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def train_loss(self, obs_seq, act_seq):
        raise NotImplementedError

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        raise NotImplementedError


class MLPBackbone(WorldModelBackbone):
    def __init__(self, obs_dim, act_dim, h=256):
        super().__init__(obs_dim, act_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, obs_dim),
        )

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        loss = torch.tensor(0.0, device=obs_seq.device)
        for t in range(T - 1):
            inp = torch.cat([obs_seq[:, t], act_seq[:, t]], dim=-1)
            pred = self.net(inp)
            loss = loss + F.mse_loss(pred, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        pred = self.net(torch.cat([obs, act], dim=-1))
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


class RSSMBackbone(WorldModelBackbone):
    def __init__(self, obs_dim, act_dim, h=256, stoch=32, deter=256):
        super().__init__(obs_dim, act_dim)
        self.h = h
        self.stoch = stoch
        self.deter = deter
        self.obs_net = nn.Linear(obs_dim, h)
        self.act_net = nn.Linear(act_dim, h)
        self.deter_net = nn.GRUCell(h + stoch, deter)
        self.stoch_net = nn.Linear(deter, stoch * 2)
        self.obs_pred = nn.Linear(deter + stoch, obs_dim)

    def _init_state(self, B, dev):
        return (torch.zeros(B, self.deter, device=dev),
                torch.zeros(B, self.stoch, device=dev))

    def _step(self, h, z, act, obs=None):
        inp = F.silu(self.obs_net(obs) + self.act_net(act)) if obs is not None else torch.zeros(h.shape[0], self.h, device=h.device)
        h_new = self.deter_net(torch.cat([inp, z], dim=-1), h)
        logits = self.stoch_net(h_new)
        mean, std = logits.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        z_new = mean + std * torch.randn_like(mean)
        return h_new, z_new

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        h, z = self._init_state(B, obs_seq.device)
        loss = 0.0
        for t in range(T):
            h, z = self._step(h, z, act_seq[:, t], obs_seq[:, t])
            pred = self.obs_pred(torch.cat([h, z], dim=-1))
            loss = loss + F.mse_loss(pred, obs_seq[:, t])
        return loss / T

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        h, z = self._init_state(obs.shape[0], obs.device)
        h, z = self._step(h, z, act, obs)
        pred = self.obs_pred(torch.cat([h, z], dim=-1))
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


class JEPABackbone(WorldModelBackbone):
    def __init__(self, obs_dim, act_dim, h=256, lat=128):
        super().__init__(obs_dim, act_dim)
        self.encoder = nn.Linear(obs_dim, h)
        self.predictor = nn.Linear(h + act_dim, lat)
        self.decoder = nn.Linear(lat, obs_dim)

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        loss = 0.0
        for t in range(T - 1):
            z = F.relu(self.encoder(obs_seq[:, t]))
            pred_z = self.predictor(torch.cat([z, act_seq[:, t]], dim=-1))
            target_z = F.relu(self.encoder(obs_seq[:, t + 1]))
            loss = loss + F.mse_loss(pred_z, target_z[:, :pred_z.shape[-1]])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        z = F.relu(self.encoder(obs))
        pred_z = self.predictor(torch.cat([z, act], dim=-1))
        target_z = F.relu(self.encoder(next_obs))
        return F.mse_loss(pred_z, target_z[:, :pred_z.shape[-1]], reduction="none").mean(dim=-1)


class DreamerV3Backbone(WorldModelBackbone):
    def __init__(self, obs_dim, act_dim, h=256, stoch=32, deter=256):
        super().__init__(obs_dim, act_dim)
        self.h = h
        self.stoch = stoch
        self.deter = deter
        self.obs_net = nn.Linear(obs_dim, h)
        self.act_net = nn.Linear(act_dim, h)
        self.deter_net = nn.GRUCell(h, deter)
        self.stoch_net = nn.Linear(deter, stoch)
        self.pred_net = nn.Linear(stoch + deter, obs_dim)

    def _init_state(self, B, dev):
        return (torch.zeros(B, self.deter, device=dev),
                torch.zeros(B, self.stoch, device=dev))

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        h, z = self._init_state(B, obs_seq.device)
        loss = 0.0
        for t in range(T):
            inp = F.silu(self.obs_net(obs_seq[:, t]) + self.act_net(act_seq[:, t]))
            h = self.deter_net(inp, h)
            z = self.stoch_net(h)
            pred = self.pred_net(torch.cat([h, z], dim=-1))
            loss = loss + F.mse_loss(pred, obs_seq[:, t])
        return loss / T

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        inp = F.silu(self.obs_net(obs) + self.act_net(act))
        h, z = self._init_state(obs.shape[0], obs.device)
        h = self.deter_net(inp, h)
        z = self.stoch_net(h)
        pred = self.pred_net(torch.cat([h, z], dim=-1))
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


class DiffusionBackbone(WorldModelBackbone):
    def __init__(self, obs_dim, act_dim, h=256, n_steps=5):
        super().__init__(obs_dim, act_dim)
        self.n_steps = n_steps
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2 + act_dim + 1, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, obs_dim),
        )

    def _add_noise(self, x, t):
        noise = torch.randn_like(x)
        return x + noise * t.float().unsqueeze(-1) / self.n_steps

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        loss = 0.0
        for t in range(T - 1):
            step = torch.randint(0, self.n_steps, (B,), device=obs_seq.device)
            noisy_next = self._add_noise(obs_seq[:, t + 1], step)
            inp = torch.cat([obs_seq[:, t], act_seq[:, t], noisy_next, step.float().unsqueeze(-1) / self.n_steps], dim=-1)
            pred = self.net(inp)
            loss = loss + F.mse_loss(pred, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        step = torch.zeros(obs.shape[0], device=obs.device)
        noisy_next = self._add_noise(next_obs, step.long())
        inp = torch.cat([obs, act, noisy_next, step.unsqueeze(-1) / self.n_steps], dim=-1)
        pred = self.net(inp)
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


class TransformerBackbone(WorldModelBackbone):
    def __init__(self, obs_dim, act_dim, h=128, n_heads=4, n_layers=2):
        super().__init__(obs_dim, act_dim)
        self.obs_proj = nn.Linear(obs_dim, h)
        self.act_proj = nn.Linear(act_dim, h)
        encoder_layer = nn.TransformerEncoderLayer(d_model=h, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(h, obs_dim)

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        obs_emb = self.obs_proj(obs_seq)
        act_emb = self.act_proj(act_seq)
        inp = obs_emb + act_emb
        out = self.transformer(inp)
        pred = self.out_proj(out[:, :-1])
        return F.mse_loss(pred, obs_seq[:, 1:])

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        obs_emb = self.obs_proj(obs).unsqueeze(1)
        act_emb = self.act_proj(act).unsqueeze(1)
        inp = obs_emb + act_emb
        out = self.transformer(inp)
        pred = self.out_proj(out.squeeze(1))
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


BACKBONES = {
    "mlp": MLPBackbone,
    "rssm": RSSMBackbone,
    "jepa": JEPABackbone,
    "dreamerv3": DreamerV3Backbone,
    "diffusion": DiffusionBackbone,
    "transformer": TransformerBackbone,
}


# ============================================================================
# TRUST CLASSES
# ============================================================================

class EMATrust:
    def __init__(self, alpha=1.0, ema=0.95):
        self.alpha = alpha
        self.ema = ema
        self.errors = {}

    def compute_trust(self, error, task_id=0):
        e = float(error.mean())
        self.errors[task_id] = self.ema * self.errors.get(task_id, e) + (1 - self.ema) * e
        return torch.exp(-self.alpha * error / (self.errors[task_id] + 1e-8)).clamp(0, 1)


class MultiStepAdaptiveTrust:
    def __init__(self, max_k=8, expand=0.3, contract=0.7):
        self.max_k = max_k
        self.expand = expand
        self.contract = contract
        self.k = {}

    def compute_trust(self, error, task_id=0):
        k = self.k.get(task_id, 1)
        e = float(error.mean())
        if e < self.expand:
            self.k[task_id] = min(k + 1, self.max_k)
        elif e > self.contract:
            self.k[task_id] = max(k - 1, 1)
        return torch.exp(-error).clamp(0, 1)

    def get_horizon(self, task_id=0):
        return self.k.get(task_id, 1)


class EnsembleDisagreement:
    def __init__(self, obs_dim, n_heads=5):
        self.heads = nn.ModuleList([nn.Linear(obs_dim, obs_dim) for _ in range(n_heads)])
        self.opt = torch.optim.Adam(self.heads.parameters(), lr=1e-3)

    def train_step(self, features, targets):
        preds = torch.stack([h(features) for h in self.heads], dim=0)
        loss = F.mse_loss(preds, targets.unsqueeze(0))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    def compute_trust(self, features):
        preds = torch.stack([h(features) for h in self.heads], dim=0)
        return torch.exp(-preds.var(dim=0).mean(dim=-1)).clamp(0, 1)


# ============================================================================
# DYNAMICS COLLECTION + PRETRAINING
# ============================================================================

def collect_dynamics_data(env, n_episodes=5, max_steps=200):
    all_obs, all_actions, all_next_obs = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for t in range(max_steps):
            act = env.action_space.sample()
            act_clipped = np.clip(act, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(act_clipped)
            all_obs.append(obs)
            all_actions.append(act_clipped)
            all_next_obs.append(next_obs)
            obs = next_obs
            if terminated or truncated:
                break
    return (np.array(all_obs, np.float32), np.array(all_actions, np.float32),
            np.array(all_next_obs, np.float32))


def pretrain_dynamics(wm, obs_data, act_data, next_obs_data, n_epochs=50, batch_size=64, lr=1e-3):
    device = next(wm.parameters()).device
    n = len(obs_data)
    optimizer = torch.optim.Adam(wm.parameters(), lr=lr)

    # Build sequences: (obs[t], act[t]) -> obs[t+1] as (B=1, T, D) batches
    T_seq = 10  # sequence length for train_loss
    for epoch in range(n_epochs):
        indices = np.random.permutation(n - T_seq)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, min(len(indices), 500), T_seq):
            idx = indices[start:start+1]
            i = idx[0]
            obs_seq = torch.tensor(obs_data[i:i+T_seq], device=device).unsqueeze(0)
            act_seq = torch.tensor(act_data[i:i+T_seq], device=device).unsqueeze(0)
            loss = wm.train_loss(obs_seq, act_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss/max(n_batches,1):.6f}")


# ============================================================================
# MPC EVALUATION
# ============================================================================

def mpc_rollout(wm, env, horizon=5, n_samples=100, max_steps=200):
    device = next(wm.parameters()).device
    obs, _ = env.reset()
    total_reward = 0.0

    for t in range(max_steps):
        best_action = env.action_space.sample()
        best_cost = float("inf")

        for _ in range(n_samples):
            act = env.action_space.sample()
            act_clipped = np.clip(act, env.action_space.low, env.action_space.high)
            act_t = torch.tensor(act_clipped, device=device).unsqueeze(0)
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)

            with torch.no_grad():
                if hasattr(wm, 'net'):
                    inp = torch.cat([obs_t, act_t], dim=-1)
                    pred_next = wm.net(inp)
                else:
                    pred_next = obs_t

            cost = float((pred_next ** 2).sum())
            if cost < best_cost:
                best_cost = cost
                best_action = act_clipped

        obs, reward, terminated, truncated, info = env.step(best_action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


def evaluate_trust(wm, obs_d, act_d, next_obs_d, trust_name, obs_dim, act_dim, device):
    if trust_name == "none":
        return 1.0
    elif trust_name == "ema":
        trust = EMATrust()
    elif trust_name == "multi_step":
        trust = MultiStepAdaptiveTrust()
    elif trust_name == "ensemble":
        trust = EnsembleDisagreement(obs_dim).to(device)
    else:
        return 1.0

    n = min(200, len(obs_d))
    idx = np.random.choice(len(obs_d), n, replace=False)
    obs = torch.tensor(obs_d[idx], device=device)
    act = torch.tensor(act_d[idx], device=device)
    next_obs = torch.tensor(next_obs_d[idx], device=device)

    with torch.no_grad():
        error = wm.predict_error(obs, act, next_obs)
        if trust_name in ["ema", "multi_step"]:
            scores = trust.compute_trust(error, task_id=0)
        elif trust_name == "ensemble":
            scores = trust.compute_trust(obs)
        else:
            scores = torch.ones(n, device=device)
    return float(scores.mean())


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--backbones", nargs="+", default=["mlp", "rssm"])
    parser.add_argument("--trusts", nargs="+", default=["none", "ema", "multi_step"])
    parser.add_argument("--collect-episodes", type=int, default=5)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--mpc-samples", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output", default="kinder_pretrain_results.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    from gymnasium import envs
    all_envs = sorted([spec.id for spec in envs.registry.values()
                       if "kinder" in spec.id.lower() and "2D" in spec.id])
    env_ids = all_envs[:args.n_envs]

    print("="*60)
    print("KinDER Dynamics Pretraining + Trust MPC")
    print("="*60)
    print(f"Envs: {len(env_ids)}: {env_ids}")
    print(f"Backbones: {args.backbones}")
    print(f"Trusts: {args.trusts}")
    print(f"Device: {device}")
    print()

    if HAS_WANDB and not args.no_wandb:
        wandb.init(project="continualwam", name=f"kinder_{device.type}", config=vars(args), reinit=True)

    results = []

    for bb_name in args.backbones:
        for trust_name in args.trusts:
            print(f"\n{'='*40}")
            print(f"Config: {bb_name} + {trust_name}")
            print(f"{'='*40}")

            env_rewards = {}
            for env_id in env_ids:
                print(f"\n--- {env_id} ---")
                env = gym.make(env_id)
                obs_dim = env.observation_space.shape[0]
                act_dim = env.action_space.shape[0]

                print("  Collecting dynamics...")
                obs_d, act_d, next_obs_d = collect_dynamics_data(env, args.collect_episodes)
                print(f"  {len(obs_d)} transitions")

                wm = BACKBONES[bb_name](obs_dim, act_dim).to(device)
                print(f"  Pretraining {bb_name}...")
                pretrain_dynamics(wm, obs_d, act_d, next_obs_d, args.pretrain_epochs)

                trust_val = evaluate_trust(wm, obs_d, act_d, next_obs_d, trust_name, obs_dim, act_dim, device)
                print(f"  Trust: {trust_val:.3f}")

                print(f"  MPC eval ({args.eval_episodes} eps)...")
                rews = [mpc_rollout(wm, env, n_samples=args.mpc_samples) for _ in range(args.eval_episodes)]
                avg = np.mean(rews)
                print(f"  Reward: {avg:.3f}")
                env_rewards[env_id] = avg

                if HAS_WANDB and not args.no_wandb:
                    wandb.log({f"{bb_name}_{trust_name}/{env_id}/reward": avg,
                               f"{bb_name}_{trust_name}/{env_id}/trust": trust_val})
                env.close()

            avg_reward = np.mean(list(env_rewards.values()))
            print(f"\n>>> Average reward: {avg_reward:.3f}")
            results.append({"benchmark": "kinder", "backbone": bb_name, "trust": trust_name,
                            "avg_reward": avg_reward, "env_rewards": env_rewards})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    for r in results:
        print(f"  {r['backbone']:12} + {r['trust']:12}: {r['avg_reward']:.3f}")

    if HAS_WANDB and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
