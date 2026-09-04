"""World model backbone implementations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelBackbone(nn.Module):
    """Base class for all world model backbones."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict_error(
        self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class MLPBackbone(WorldModelBackbone):
    """Feedforward world model."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__(obs_dim, act_dim, hidden)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = obs_seq.shape
        loss = torch.tensor(0.0, device=obs_seq.device)
        for t in range(T - 1):
            inp = torch.cat([obs_seq[:, t], act_seq[:, t]], dim=-1)
            pred = self.encoder(inp)
            loss = loss + F.mse_loss(pred, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([obs, act], dim=-1)
        pred = self.encoder(inp)
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


class RSSMBackbone(WorldModelBackbone):
    """Recurrent state-space model."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, state: int = 256):
        super().__init__(obs_dim, act_dim, hidden)
        self.state_dim = state
        self.obs_encoder = nn.Linear(obs_dim, hidden)
        self.rnn = nn.GRUCell(hidden + act_dim, state)
        self.pred_head = nn.Linear(state, obs_dim)
        self.kl_free: float = 1.0

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        h = torch.zeros(B, self.state_dim, device=dev)
        loss = torch.tensor(0.0, device=dev)
        for t in range(T - 1):
            o_enc = torch.relu(self.obs_encoder(obs_seq[:, t]))
            h = torch.relu(self.rnn(torch.cat([o_enc, act_seq[:, t]], dim=-1), h))
            pred = self.pred_head(h)
            loss = loss + F.mse_loss(pred, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        dev = obs.device
        h = torch.zeros(B, self.state_dim, device=dev)
        o_enc = torch.relu(self.obs_encoder(obs))
        h = torch.relu(self.rnn(torch.cat([o_enc, act], dim=-1), h))
        pred = self.pred_head(h)
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


class JEPABackbone(WorldModelBackbone):
    """Joint-embedding predictive architecture."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, latent: int = 128):
        super().__init__(obs_dim, act_dim, hidden)
        self.latent_dim = latent
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, latent))
        self.predictor = nn.Sequential(nn.Linear(latent + act_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, obs_dim))

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = obs_seq.shape
        loss = torch.tensor(0.0, device=obs_seq.device)
        for t in range(T - 1):
            z = self.encoder(obs_seq[:, t])
            pred_z = self.predictor(torch.cat([z, act_seq[:, t]], dim=-1))
            pred_obs = self.decoder(pred_z)
            loss = loss + F.mse_loss(pred_obs, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        pred_z = self.predictor(torch.cat([z, act], dim=-1))
        pred_obs = self.decoder(pred_z)
        return F.mse_loss(pred_obs, next_obs, reduction="none").mean(dim=-1)

    @torch.no_grad()
    def predict_error_latent(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        pred_z = self.predictor(torch.cat([z, act], dim=-1))
        z_next = self.encoder(next_obs)
        return F.mse_loss(pred_z, z_next, reduction="none").mean(dim=-1)


class DreamerV3Backbone(WorldModelBackbone):
    """RSSM + stochastic observations."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, state: int = 256, stochastic: int = 32):
        super().__init__(obs_dim, act_dim, hidden)
        self.state_dim = state
        self.stoch_dim = stochastic
        self.obs_encoder = nn.Linear(obs_dim, hidden)
        self.rnn = nn.GRUCell(hidden + act_dim, state)
        self.stoch_encoder = nn.Linear(state, stochastic)
        self.pred_head = nn.Linear(state + stochastic, obs_dim)
        self.kl_free: float = 1.0

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        h = torch.zeros(B, self.state_dim, device=dev)
        s = torch.zeros(B, self.stoch_dim, device=dev)
        loss = torch.tensor(0.0, device=dev)
        for t in range(T - 1):
            o_enc = torch.relu(self.obs_encoder(obs_seq[:, t]))
            h = torch.relu(self.rnn(torch.cat([o_enc, act_seq[:, t]], dim=-1), h))
            s = torch.relu(self.stoch_encoder(h))
            pred = self.pred_head(torch.cat([h, s], dim=-1))
            loss = loss + F.mse_loss(pred, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        dev = obs.device
        h = torch.zeros(B, self.state_dim, device=dev)
        s = torch.zeros(B, self.stoch_dim, device=dev)
        o_enc = torch.relu(self.obs_encoder(obs))
        h = torch.relu(self.rnn(torch.cat([o_enc, act], dim=-1), h))
        s = torch.relu(self.stoch_encoder(h))
        pred = self.pred_head(torch.cat([h, s], dim=-1))
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


class DiffusionBackbone(WorldModelBackbone):
    """Denoising diffusion world model."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, steps: int = 5):
        super().__init__(obs_dim, act_dim, hidden)
        self.diff_steps = steps
        self.noise_pred = nn.Sequential(
            nn.Linear(obs_dim * 2 + act_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        loss = torch.tensor(0.0, device=dev)
        for t in range(T - 1):
            noise = torch.randn_like(obs_seq[:, t + 1])
            alpha = torch.rand(B, 1, device=dev)
            noisy = alpha * obs_seq[:, t + 1] + (1 - alpha) * noise
            pred = self.noise_pred(torch.cat([noisy, obs_seq[:, t], act_seq[:, t], alpha], dim=-1))
            loss = loss + F.mse_loss(pred, noise)
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        dev = obs.device
        total_err = torch.zeros(B, device=dev)
        for i in range(self.diff_steps):
            alpha = torch.full((B, 1), (i + 1) / self.diff_steps, device=dev)
            noise = torch.randn_like(next_obs)
            noisy = alpha * next_obs + (1 - alpha) * noise
            pred = self.noise_pred(torch.cat([noisy, obs, act], dim=-1))
            total_err = total_err + F.mse_loss(pred, noise, reduction="none").mean(dim=-1)
        return total_err / self.diff_steps


class TransformerBackbone(WorldModelBackbone):
    """GPT-style autoregressive world model."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__(obs_dim, act_dim, hidden)
        self.embed = nn.Linear(obs_dim + act_dim, hidden)
        self.pos_embed = nn.Embedding(128, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, obs_dim)

    def train_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = obs_seq.shape
        x = self.embed(torch.cat([obs_seq, act_seq], dim=-1))
        pos = self.pos_embed(torch.arange(T, device=x.device)).unsqueeze(0)
        x = self.transformer(x + pos)
        pred = self.head(x[:, :-1])
        return F.mse_loss(pred, obs_seq[:, 1:])

    @torch.no_grad()
    def predict_error(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        seq = torch.stack([obs, next_obs], dim=1)
        act = torch.stack([act, torch.zeros_like(act)], dim=1)
        x = self.embed(torch.cat([seq, act], dim=-1))
        pos = self.pos_embed(torch.arange(2, device=x.device)).unsqueeze(0)
        x = self.transformer(x + pos)
        pred = self.head(x[:, 0])
        return F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)


BACKBONES: dict[str, type[WorldModelBackbone]] = {
    "mlp": MLPBackbone,
    "rssm": RSSMBackbone,
    "jepa": JEPABackbone,
    "dreamerv3": DreamerV3Backbone,
    "diffusion": DiffusionBackbone,
    "transformer": TransformerBackbone,
}


def get_backbone(name: str, obs_dim: int, act_dim: int, **kwargs) -> WorldModelBackbone:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONES.keys())}")
    return BACKBONES[name](obs_dim, act_dim, **kwargs)
