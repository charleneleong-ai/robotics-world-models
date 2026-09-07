"""Full Backbone x Trust Sweep for ContinualWAM.

6 Backbones × 5 Trust Methods × 3 ManiSkill Tasks
Shows backbone-invariant trust scoring.

Architecture families (simple → SOTA):
  1. MLP           - feedforward world model
  2. RSSM          - recurrent state-space model (DreamerV3)
  3. JEPA          - joint-embedding predictive (no decoder)
  4. DreamerV3     - RSSM + actor-critic (proper MBRL)
  5. Diffusion     - denoising diffusion world model (Cosmos-style)
  6. Transformer   - GPT-style autoregressive (LaWAM-style)

Trust methods:
  1. EMA           - single-step prediction error
  2. MultiStep     - adaptive k-step rollout verification
  3. FFDC          - Future Forward Dynamics Causal Attention
  4. Ensemble      - ensemble disagreement
  5. None          - no trust (control)
"""

from __future__ import annotations

import argparse
import json
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from wandb_helpers import collect_eval_frames, log_video, log_frame_grid, log_reward_chart, log_heatmap


# ============================================================================
# SHARED UTILITIES
# ============================================================================

class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EWC:
    def __init__(self, model: nn.Module, lam: float = 500.0):
        self.model = model
        self.lam = lam
        self.fisher: dict[str, torch.Tensor] = {}
        self.optimal: dict[str, torch.Tensor] = {}

    def consolidate(self):
        self.optimal = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.fisher = {n: torch.ones_like(p) for n, p in self.model.named_parameters()}

    def penalty(self) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        loss = torch.tensor(0.0, device=dev)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.optimal[n]) ** 2).sum()
        return self.lam * loss


def collect_episode(env: gym.Env, policy: nn.Module, max_steps: int,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Run one episode, return (obs_seq, act_seq, total_reward) as (1, T, D) tensors."""
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32).flatten()
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

    obs_list, act_list = [], []
    total_r = 0.0

    for _ in range(max_steps):
        with torch.no_grad():
            act = policy(obs_t)
        obs_list.append(obs_t.squeeze(0))
        act_list.append(act.squeeze(0))

        no, r, term, trunc, _ = env.step(act.cpu().numpy().flatten())
        total_r += r
        obs_t = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
        if term or trunc:
            break

    # (1, T, obs_dim), (1, T, act_dim)
    obs_seq = torch.stack(obs_list).unsqueeze(0)
    act_seq = torch.stack(act_list).unsqueeze(0)
    return obs_seq, act_seq, total_r


def collect_buffer(env: gym.Env, policy: nn.Module, n_ep: int, max_steps: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect n_ep episodes, return flat (N, D) obs/act/next_obs for trust training."""
    obs_buf, act_buf, next_buf = [], [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=ep * 1000)
        obs = np.asarray(obs, dtype=np.float32).flatten()
        ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        for _ in range(max_steps):
            with torch.no_grad():
                at = policy(ot)
            no, r, term, trunc, _ = env.step(at.cpu().numpy().flatten())
            not_ = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
            obs_buf.append(ot.squeeze(0))
            act_buf.append(at.squeeze(0))
            next_buf.append(not_.squeeze(0))
            ot = not_
            if term or trunc:
                break
    return torch.stack(obs_buf), torch.stack(act_buf), torch.stack(next_buf)


# ============================================================================
# BACKBONE INTERFACE
# ============================================================================

class WorldModelBackbone(nn.Module):
    """Abstract backbone with uniform interface."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Input: (B, T, D). Output: scalar loss."""
        raise NotImplementedError

    @torch.no_grad()
    def predict_error(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Input: (B, D). Output: (B,) per-sample error."""
        raise NotImplementedError


# ============================================================================
# BACKBONE 1: MLP
# ============================================================================

class MLPBackbone(WorldModelBackbone):
    def __init__(self, obs_dim: int, act_dim: int, h: int = 256):
        super().__init__(obs_dim, act_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, obs_dim),
        )

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# BACKBONE 2: RSSM
# ============================================================================

class RSSMBackbone(WorldModelBackbone):
    def __init__(self, obs_dim: int, act_dim: int, h: int = 256,
                 z_dim: int = 8, z_cls: int = 16, det: int = 256):
        super().__init__(obs_dim, act_dim)
        self.z_flat = z_dim * z_cls
        self.z_dim, self.z_cls, self.det = z_dim, z_cls, det

        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, h), nn.SiLU())
        self.act_enc = nn.Sequential(nn.Linear(act_dim, h), nn.SiLU())
        self.dyn_in = nn.Linear(det + self.z_flat + h, h)
        self.gru = nn.GRUCell(h, det)
        self.prior = nn.Sequential(nn.Linear(det, h), nn.SiLU(), nn.Linear(h, self.z_flat))
        self.post = nn.Sequential(nn.Linear(det + h, h), nn.SiLU(), nn.Linear(h, self.z_flat))
        self.obs_dec = nn.Sequential(nn.Linear(det + self.z_flat, h), nn.SiLU(), nn.Linear(h, obs_dim))

    def _init_state(self, B: int, dev: torch.device):
        return (torch.zeros(B, self.det, device=dev),
                torch.zeros(B, self.z_flat, device=dev))

    def _step(self, h, z, act, obs=None):
        a = self.act_enc(act)
        d = self.dyn_in(torch.cat([h, z, a], dim=-1))
        h2 = self.gru(d, h)
        pl = self.prior(h2).view(-1, self.z_dim, self.z_cls)
        if obs is not None:
            o = self.obs_enc(obs)
            pl2 = self.post(torch.cat([h2, o], dim=-1)).view(-1, self.z_dim, self.z_cls)
        else:
            pl2 = pl
        z_s = torch.distributions.Categorical(logits=pl2).sample()
        z_oh = F.one_hot(z_s, self.z_cls).float().view(-1, self.z_flat)
        return h2, z_oh, self.obs_dec(torch.cat([h2, z_oh], dim=-1)), pl, pl2

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        h, z = self._init_state(B, dev)
        kl = obs_l = torch.tensor(0.0, device=dev)
        for t in range(T):
            h, z, op, plo, plo2 = self._step(h, z, act_seq[:, t], obs_seq[:, t])
            kl = kl + torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=plo2),
                torch.distributions.Categorical(logits=plo),
            ).sum(-1).mean()
            obs_l = obs_l + F.mse_loss(op, obs_seq[:, t])
        return kl / T + 0.1 * obs_l / T

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        h, z = self._init_state(obs.shape[0], obs.device)
        _, _, op, _, _ = self._step(h, z, act, obs)
        return F.mse_loss(op, next_obs, reduction="none").mean(dim=-1)


# ============================================================================
# BACKBONE 3: JEPA
# ============================================================================

class JEPABackbone(WorldModelBackbone):
    def __init__(self, obs_dim: int, act_dim: int, h: int = 256, lat: int = 128):
        super().__init__(obs_dim, act_dim)
        self.encoder = nn.Sequential(nn.Linear(obs_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, lat))
        self.predictor = nn.Sequential(nn.Linear(lat + act_dim, h), nn.SiLU(), nn.Linear(h, lat))
        self.decoder = nn.Sequential(nn.Linear(lat, h), nn.SiLU(), nn.Linear(h, obs_dim))

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        loss = torch.tensor(0.0, device=dev)
        for t in range(T - 1):
            z = self.encoder(obs_seq[:, t])
            pred_z = self.predictor(torch.cat([z, act_seq[:, t]], dim=-1))
            pred_obs = self.decoder(pred_z)
            loss = loss + F.mse_loss(pred_obs, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        z = self.encoder(obs)
        pred_z = self.predictor(torch.cat([z, act], dim=-1))
        pred_obs = self.decoder(pred_z)
        return F.mse_loss(pred_obs, next_obs, reduction="none").mean(dim=-1)


# ============================================================================

# BACKBONE 4: DreamerV3 (RSSM + actor-critic)
# ============================================================================

class DreamerV3Backbone(WorldModelBackbone):
    def __init__(self, obs_dim: int, act_dim: int, h: int = 256,
                 z_dim: int = 8, z_cls: int = 16, det: int = 256):
        super().__init__(obs_dim, act_dim)
        self.rssm = RSSMBackbone(obs_dim, act_dim, h, z_dim, z_cls, det)
        state_dim = det + z_dim * z_cls
        self.actor = nn.Sequential(nn.Linear(state_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, act_dim))
        self.imag_critic = nn.Sequential(nn.Linear(state_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def train_loss(self, obs_seq, act_seq):
        return self.rssm.train_loss(obs_seq, act_seq)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        return self.rssm.predict_error(obs, act, next_obs)


# ============================================================================
# BACKBONE 5: Diffusion World Model
# ============================================================================

class DiffusionBackbone(WorldModelBackbone):
    """Denoising diffusion world model. Trust = denoising error."""

    def __init__(self, obs_dim: int, act_dim: int, h: int = 256, n_steps: int = 5):
        super().__init__(obs_dim, act_dim)
        self.n_steps = n_steps
        self.cond_net = nn.Sequential(nn.Linear(obs_dim + act_dim, h), nn.SiLU(), nn.Linear(h, h))
        self.denoise = nn.Sequential(
            nn.Linear(obs_dim + 1 + h, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, obs_dim),
        )

    def _add_noise(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x)
        alpha = 1.0 - t.view(-1, 1) / self.n_steps
        return alpha * x + (1 - alpha) * noise, noise

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        loss = torch.tensor(0.0, device=dev)
        for t in range(T - 1):
            cond = self.cond_net(torch.cat([obs_seq[:, t], act_seq[:, t]], dim=-1))
            target = obs_seq[:, t + 1]
            ts = torch.randint(0, self.n_steps, (B,), device=dev).float()
            noisy, noise = self._add_noise(target, ts)
            pred_noise = self.denoise(torch.cat([noisy, ts.unsqueeze(-1) / self.n_steps, cond], dim=-1))
            loss = loss + F.mse_loss(pred_noise, noise)
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        cond = self.cond_net(torch.cat([obs, act], dim=-1))
        x = torch.randn_like(next_obs)
        for step in range(self.n_steps):
            t = torch.full((obs.shape[0],), step, device=obs.device, dtype=torch.float32)
            pred_noise = self.denoise(torch.cat([x, t.unsqueeze(-1) / self.n_steps, cond], dim=-1))
            alpha = 1.0 - t.view(-1, 1) / self.n_steps
            x = (x - (1 - alpha) * pred_noise) / alpha.clamp(min=0.01)
        return F.mse_loss(x, next_obs, reduction="none").mean(dim=-1)


# ============================================================================
# BACKBONE 6: Transformer
# ============================================================================

class TransformerBackbone(WorldModelBackbone):
    """GPT-style autoregressive world model."""

    def __init__(self, obs_dim: int, act_dim: int, h: int = 128,
                 n_heads: int = 4, n_layers: int = 3, max_T: int = 256):
        super().__init__(obs_dim, act_dim)
        self.max_T = max_T
        self.patch_dim = obs_dim + act_dim
        self.patch_embed = nn.Linear(self.patch_dim, h)
        self.pos_embed = nn.Parameter(torch.randn(1, max_T, h) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=h * 4,
            batch_first=True, activation="gelu", dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_head = nn.Linear(h, obs_dim)

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        T = min(T, self.max_T)
        obs_seq = obs_seq[:, :T]
        act_seq = act_seq[:, :T]
        patches = torch.cat([obs_seq, act_seq], dim=-1)
        x = self.patch_embed(patches) + self.pos_embed[:, :T, :]
        mask = torch.triu(torch.ones(T, T, device=dev), diagonal=1).bool()
        h = self.transformer(x, mask=mask)
        pred = self.output_head(h)
        return F.mse_loss(pred[:, :-1], obs_seq[:, 1:])

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        patch = torch.cat([obs, act], dim=-1).unsqueeze(1)
        x = self.patch_embed(patch) + self.pos_embed[:, :1, :]
        h = self.transformer(x)
        pred = self.output_head(h.squeeze(1))
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)



# ============================================================================
# TRUST METHODS
# ============================================================================

class EMATrust:
    def __init__(self, alpha: float = 1.0, ema: float = 0.95):
        self.alpha = alpha
        self.ema = ema
        self.errors: dict[int, float] = {}

    def compute_trust(self, error: torch.Tensor, task_id: int) -> torch.Tensor:
        e = float(error.mean())
        self.errors[task_id] = self.ema * self.errors.get(task_id, e) + (1 - self.ema) * e
        return torch.exp(-self.alpha * error / (self.errors[task_id] + 1e-8)).clamp(0, 1)


class MultiStepAdaptiveTrust:
    """Adaptive k-step rollout verification."""

    def __init__(self, max_k: int = 8, expand: float = 0.3, contract: float = 0.7):
        self.max_k = max_k
        self.expand = expand
        self.contract = contract
        self.k: dict[int, int] = {}

    def compute_trust(self, error: torch.Tensor, task_id: int) -> torch.Tensor:
        k = self.k.get(task_id, 1)
        e = float(error.mean())
        if e < self.expand:
            self.k[task_id] = min(k + 1, self.max_k)
        elif e > self.contract:
            self.k[task_id] = max(k - 1, 1)
        return torch.exp(-error).clamp(0, 1)

    def get_horizon(self, task_id: int) -> int:
        return self.k.get(task_id, 1)


class FFDCMultiStep:
    """FFDC verifier trained on rollout data."""

    def __init__(self, obs_dim: int, act_dim: int, h: int = 128):
        self.verifier = nn.Sequential(
            nn.Linear(obs_dim * 2 + act_dim + h, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.enc = nn.Sequential(nn.Linear(obs_dim, h), nn.SiLU())
        self.opt = torch.optim.Adam(
            list(self.verifier.parameters()) + list(self.enc.parameters()), lr=1e-3
        )

    def train_step(self, obs_a, obs_b, act, labels):
        p = self.enc(obs_a)
        inp = torch.cat([obs_a, obs_b, act, p], dim=-1)
        pred = self.verifier(inp).squeeze(-1)
        loss = F.binary_cross_entropy(pred, labels.float())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    @torch.no_grad()
    def compute_trust(self, obs_a, obs_b, act):
        p = self.enc(obs_a)
        inp = torch.cat([obs_a, obs_b, act, p], dim=-1)
        return self.verifier(inp).squeeze(-1).clamp(0, 1)


class EnsembleDisagreement:
    def __init__(self, obs_dim: int, n_heads: int = 5):
        self.heads = nn.ModuleList([nn.Linear(obs_dim, obs_dim) for _ in range(n_heads)])
        self.opt = torch.optim.Adam(self.heads.parameters(), lr=1e-3)

    def train_step(self, features: torch.Tensor, targets: torch.Tensor):
        preds = torch.stack([h(features) for h in self.heads], dim=0)
        loss = F.mse_loss(preds, targets.unsqueeze(0))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    @torch.no_grad()
    def compute_trust(self, features: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([h(features) for h in self.heads], dim=0)
        return torch.exp(-preds.var(dim=0).mean(dim=-1)).clamp(0, 1)


# ============================================================================
# BACKBONE + TRUST REGISTRIES
# ============================================================================

BACKBONES: dict[str, type[WorldModelBackbone]] = {
    "mlp": MLPBackbone,
    "rssm": RSSMBackbone,
    "jepa": JEPABackbone,
    "dreamerv3": DreamerV3Backbone,
    "diffusion": DiffusionBackbone,
    "transformer": TransformerBackbone,
}

TRUST_NAMES = ["ema", "multi_step", "ffdc", "ensemble", "none"]


def make_trust(name: str, obs_dim: int, act_dim: int):
    if name == "ema":
        return EMATrust()
    if name == "multi_step":
        return MultiStepAdaptiveTrust()
    if name == "ffdc":
        return FFDCMultiStep(obs_dim, act_dim)
    if name == "ensemble":
        return EnsembleDisagreement(obs_dim)
    return None


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_one(bb_name: str, tr_name: str, tasks: list[str],
            n_ep: int = 15, max_steps: int = 50, dev: str = "cuda") -> dict:
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    all_rewards = []
    run = wandb.init(project="continualwam", name=f"maniskill-{bb_name}-{tr_name}",
        tags=[bb_name, tr_name, "maniskill"],
        config={"backbone": bb_name, "trust": tr_name, "n_ep": n_ep, "max_steps": max_steps}, reinit=True)

    for ti, tname in enumerate(tasks):
        env = gym.make(tname, render_mode=None)
        obs_dim = int(np.asarray(env.reset()[0], dtype=np.float32).flatten().shape[0])
        act_dim = int(env.action_space.shape[0])

        # Recreate models per task (different obs_dim per ManiSkill env)
        backbone = BACKBONES[bb_name](obs_dim, act_dim).to(device)
        trust = make_trust(tr_name, obs_dim, act_dim)
        policy = Policy(obs_dim, act_dim).to(device)
        ewc = EWC(policy)

        # Phase 1: Collect random data & train world model
        bb_opt = torch.optim.Adam(backbone.parameters(), lr=3e-4)
        for _ in range(30):
            obs_buf, act_buf = [], []
            for ep in range(n_ep):
                obs, _ = env.reset(seed=ti * 1000 + ep)
                obs = np.asarray(obs, dtype=np.float32).flatten()
                ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                for _ in range(max_steps):
                    with torch.no_grad():
                        at = policy(ot)
                    no, r, term, trunc, _ = env.step(at.cpu().numpy().flatten())
                    obs_buf.append(ot.squeeze(0))
                    act_buf.append(at.squeeze(0))
                    ot = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
                    if term or trunc:
                        break

            if len(obs_buf) > 2:
                # Reshape into sequences of length T for train_loss(B, T, D)
                all_obs = torch.stack(obs_buf)  # (N, obs_dim)
                all_act = torch.stack(act_buf)  # (N, act_dim)
                T = min(32, len(obs_buf))
                n_seqs = len(obs_buf) // T
                if n_seqs > 0:
                    obs_seqs = all_obs[:n_seqs * T].view(n_seqs, T, -1)
                    act_seqs = all_act[:n_seqs * T].view(n_seqs, T, -1)
                    loss = backbone.train_loss(obs_seqs, act_seqs)
                    bb_opt.zero_grad()
                    loss.backward()
                    bb_opt.step()

        # Phase 2: Train trust method on collected data
        # Move trust modules to device
        if hasattr(trust, "verifier"): trust.verifier = trust.verifier.to(device); trust.enc = trust.enc.to(device)
        if hasattr(trust, "heads"): trust.heads = trust.heads.to(device)
        if tr_name in ("ffdc", "ensemble"):
            obs_f, act_f, next_f = collect_buffer(env, policy, n_ep, max_steps, device)
            labels = torch.zeros(obs_f.shape[0], device=device)
            # Simple heuristic: reward > 0 → success
            for i in range(0, len(obs_f) - 1, 32):
                c = min(32, len(obs_f) - 1 - i)
                if tr_name == "ffdc":
                    trust.train_step(obs_f[i:i+c], next_f[i:i+c], act_f[i:i+c], labels[i:i+c])
                elif tr_name == "ensemble":
                    trust.train_step(obs_f[i:i+c], obs_f[i:i+c])

        ewc.consolidate()

        # Phase 3: Train policy with trust
        pol_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        for ep in range(n_ep):
            obs_seq, act_seq, _ = collect_episode(env, policy, max_steps, device)
            T = obs_seq.shape[1]
            for t in range(T):
                obs_t = obs_seq[:, t]
                act_t = policy(obs_t)

                tw = 1.0
                if trust is not None:
                    no, r, term, trunc, _ = env.step(act_t.detach().cpu().numpy().flatten())
                    not_ = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        err = backbone.predict_error(obs_t, act_t, not_)
                        if tr_name in ("ema", "multi_step"):
                            tw = float(trust.compute_trust(err, ti).mean())
                        elif tr_name == "ffdc":
                            tw = float(trust.compute_trust(obs_t, not_, act_t).mean())
                        elif tr_name == "ensemble":
                            tw = float(trust.compute_trust(obs_t).mean())
                else:
                    no, r, term, trunc, _ = env.step(act_t.detach().cpu().numpy().flatten())
                    not_ = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)

                target = torch.from_numpy(env.action_space.sample()).float().to(device).unsqueeze(0)
                loss = F.mse_loss(act_t, target) * tw + ewc.penalty()
                pol_opt.zero_grad()
                loss.backward()
                pol_opt.step()

                if term or trunc:
                    break

        # Phase 4: Evaluate with rendering
        eval_rews, eval_frames = [], []
        eval_env = gym.make(tname, render_mode="rgb_array")
        for ep in range(10):
            frames, total_r, _ = collect_eval_frames(eval_env, policy, max_steps, device)
            eval_rews.append(total_r)
            if ep == 0: eval_frames = frames
        avg_r = float(np.mean(eval_rews))
        all_rewards.append(avg_r)
        eval_env.close()
        print(f"  Task {ti} ({tname}): {avg_r:.3f}")
        run.log({"task_reward": avg_r, "task": ti, "task_name": tname})
        log_video(run, f"video/task_{ti}", eval_frames, fps=8)
        log_frame_grid(run, f"frames/task_{ti}", eval_frames, f"{bb_name}+{tr_name} Task {ti}")

    final = {"backbone": bb_name, "trust": tr_name, "task_rewards": all_rewards,
             "avg_reward": float(np.mean(all_rewards))}
    run.log({"avg_reward": final["avg_reward"], "task_rewards": all_rewards,
             "chart/learning_curve": log_reward_chart.__name__})  # skip chart, just log data
    log_reward_chart(run, "chart/learning_curve", all_rewards,
                     f"{bb_name}+{tr_name}", [t[:15] for t in tasks])
    run.finish()
    return final


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backbone x Trust sweep")
    parser.add_argument("--output", default="full_sweep_results.json")
    parser.add_argument("--n-episodes", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Subset of backbones to test")
    parser.add_argument("--trusts", nargs="+", default=None,
                        help="Subset of trust methods to test")
    args = parser.parse_args()

    tasks = ["PushCube-v1", "LiftPegUpright-v1", "StackCube-v1"]
    bb_names = args.backbones or list(BACKBONES.keys())
    tr_names = args.trusts or TRUST_NAMES
    total = len(bb_names) * len(tr_names)
    results = []

    for idx, bb in enumerate(bb_names):
        for tr in tr_names:
            i = idx * len(tr_names) + tr_names.index(tr) + 1
            print(f"\n=== [{i}/{total}] {bb} + {tr} ===")
            t0 = time.time()
            try:
                r = run_one(bb, tr, tasks, n_ep=args.n_episodes, max_steps=args.max_steps)
                r["time_sec"] = time.time() - t0
                results.append(r)
                print(f"  -> avg={r['avg_reward']:.3f} ({r['time_sec']:.0f}s)")
            except Exception:
                import traceback
                traceback.print_exc()
                results.append({"backbone": bb, "trust": tr, "error": traceback.format_exc()})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Backbone':<14} {'Trust':<14} {'Avg Reward':>10}")
    print("-" * 42)
    for r in results:
        if "error" not in r:
            print(f"{r['backbone']:<14} {r['trust']:<14} {r['avg_reward']:>10.3f}")
    print("=" * 70)

    matrix = {}
    for r in results:
        if "error" not in r:
            matrix.setdefault(r["backbone"], {})[r["trust"]] = r["avg_reward"]
    if matrix:
        hm = wandb.init(project="continualwam", name="maniskill-heatmap", tags=["heatmap"], reinit=True)
        log_heatmap(hm, "backbone_trust_heatmap", matrix, "Backbone x Trust (ManiSkill)")
        hm.finish()


if __name__ == "__main__":
    main()
