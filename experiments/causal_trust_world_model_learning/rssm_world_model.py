"""RSSM World Model for Continual Learning.

Recurrent State-Space Model (RSSM) following DreamerV3 architecture.
Learns latent dynamics from observation sequences.
Trust scoring via prediction error on held-out data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Independent
from typing import Optional


class RSSM(nn.Module):
    """Recurrent State-Space Model (DreamerV3-style).

    Maintains deterministic (h) and stochastic (z) state.
    Prior: p(z_t | h_t)
    Posterior: q(z_t | h_t, o_t)
    Dynamics: p(h_t | h_{t-1}, z_{t-1}, a_{t-1})
    Reward: p(r_t | h_t, z_t)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        stochastic_dim: int = 32,
        stochastic_classes: int = 32,
        deterministic_dim: int = 512,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        self.stochastic_classes = stochastic_classes
        self.deterministic_dim = deterministic_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
        )

        # Dynamics model: p(h_t | h_{t-1}, z_{t-1}, a_{t-1})
        self.dynamics_input = nn.Linear(
            deterministic_dim + stochastic_dim * stochastic_classes + hidden_dim,
            hidden_dim,
        )
        self.dynamics_gru = nn.GRUCell(hidden_dim, deterministic_dim)

        # Prior model: p(z_t | h_t) -- stochastic prediction
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_dim * stochastic_classes),
        )

        # Posterior model: q(z_t | h_t, o_t) -- uses observation
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_dim * stochastic_classes),
        )

        # Reward predictor: p(r_t | h_t, z_t)
        self.reward_net = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim * stochastic_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Continue predictor (done signal)
        self.continue_net = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim * stochastic_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return initial deterministic (h) and stochastic (z) state."""
        h = torch.zeros(batch_size, self.deterministic_dim, device=device)
        z = torch.zeros(
            batch_size,
            self.stochastic_dim * self.stochastic_classes,
            device=device,
        )
        return h, z

    def observe_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single step with observation (posterior). Used during training."""
        obs_enc = self.obs_encoder(obs)
        act_enc = self.action_encoder(action)

        # Dynamics: compute deterministic state
        dyn_input = torch.cat([prev_h, prev_z, act_enc], dim=-1)
        dyn_input = self.dynamics_input(dyn_input)
        h = self.dynamics_gru(dyn_input, prev_h)

        # Posterior: q(z_t | h_t, o_t)
        post_input = torch.cat([h, obs_enc], dim=-1)
        post_logits = self.posterior_net(post_input)
        post_logits = post_logits.view(
            -1, self.stochastic_dim, self.stochastic_classes
        )
        post_dist = Categorical(logits=post_logits)
        z_sampled = post_dist.sample()
        z_onehot = F.one_hot(z_sampled, self.stochastic_classes).float()
        z_flat = z_onehot.view(-1, self.stochastic_dim * self.stochastic_classes)

        # Prior: p(z_t | h_t)
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.view(
            -1, self.stochastic_dim, self.stochastic_classes
        )
        prior_dist = Categorical(logits=prior_logits)

        # Reward and continue prediction
        pred_input = torch.cat([h, z_flat], dim=-1)
        reward_pred = self.reward_net(pred_input).squeeze(-1)
        continue_pred = self.continue_net(pred_input).squeeze(-1)

        return {
            "h": h,
            "z": z_flat,
            "z_sampled": z_sampled,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "prior_dist": prior_dist,
            "post_dist": post_dist,
            "reward_pred": reward_pred,
            "continue_pred": continue_pred,
        }

    def imagine_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single step without observation (prior). Used during imagination."""
        act_enc = self.action_encoder(action)

        dyn_input = torch.cat([prev_h, prev_z, act_enc], dim=-1)
        dyn_input = self.dynamics_input(dyn_input)
        h = self.dynamics_gru(dyn_input, prev_h)

        # Prior: p(z_t | h_t)
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.view(
            -1, self.stochastic_dim, self.stochastic_classes
        )
        prior_dist = Categorical(logits=prior_logits)
        z_sampled = prior_dist.sample()
        z_onehot = F.one_hot(z_sampled, self.stochastic_classes).float()
        z_flat = z_onehot.view(-1, self.stochastic_dim * self.stochastic_classes)

        pred_input = torch.cat([h, z_flat], dim=-1)
        reward_pred = self.reward_net(pred_input).squeeze(-1)
        continue_pred = self.continue_net(pred_input).squeeze(-1)

        return {
            "h": h,
            "z": z_flat,
            "z_sampled": z_sampled,
            "prior_logits": prior_logits,
            "prior_dist": prior_dist,
            "reward_pred": reward_pred,
            "continue_pred": continue_pred,
        }

    def get_free_nats(self) -> float:
        """Free nats for KL balancing (DreamerV3 uses 1.0)."""
        return 1.0


class WorldModel(nn.Module):
    """Wrapper that combines RSSM with decoders for observation reconstruction."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        stochastic_dim: int = 32,
        stochastic_classes: int = 32,
        deterministic_dim: int = 512,
    ):
        super().__init__()
        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
            stochastic_classes=stochastic_classes,
            deterministic_dim=deterministic_dim,
        )

        state_dim = deterministic_dim + stochastic_dim * stochastic_classes

        # Observation decoder: reconstruct obs from state
        self.obs_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Trust head: predicts whether model's prediction is reliable
        self.trust_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def training_step(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one training step on a sequence.

        Args:
            observations: (B, T, obs_dim)
            actions: (B, T, action_dim)
            rewards: (B, T)
            dones: (B, T)

        Returns:
            Dictionary of losses and metrics.
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device

        h, z = self.rssm.initial_state(batch_size, device)

        prior_kls = []
        post_kls = []
        obs_losses = []
        reward_losses = []
        continue_losses = []
        trust_scores = []

        for t in range(seq_len):
            obs_t = observations[:, t]
            act_t = actions[:, t]
            reward_t = rewards[:, t]
            done_t = dones[:, t]

            result = self.rssm.observe_step(h, z, act_t, obs_t)
            h, z = result["h"], result["z"]

            # KL divergence between posterior and prior (with free nats)
            free_nats = self.rssm.get_free_nats()
            kl_post = result["post_dist"].log_prob(result["z_sampled"])
            kl_prior = result["prior_dist"].log_prob(result["z_sampled"])
            kl = torch.clamp(kl_post - kl_prior, min=free_nats).sum(dim=-1).mean()
            prior_kls.append(kl.item())

            # Observation reconstruction loss
            obs_pred = self.obs_decoder(z)
            obs_loss = F.mse_loss(obs_pred, obs_t).item()
            obs_losses.append(obs_loss)

            # Reward prediction loss
            reward_loss = F.mse_loss(result["reward_pred"], reward_t).item()
            reward_losses.append(reward_loss)

            # Continue prediction loss
            continue_loss = F.binary_cross_entropy_with_logits(
                result["continue_pred"], (1.0 - done_t)
            ).item()
            continue_losses.append(continue_loss)

            # Trust score
            trust = self.trust_head(z).mean().item()
            trust_scores.append(trust)

        # Total loss for backprop
        total_loss = sum(prior_kls) / seq_len + \
                     0.1 * sum(obs_losses) / seq_len + \
                     0.1 * sum(reward_losses) / seq_len + \
                     0.1 * sum(continue_losses) / seq_len

        return {
            "total_loss": total_loss,
            "prior_kl": sum(prior_kls) / seq_len,
            "obs_loss": sum(obs_losses) / seq_len,
            "reward_loss": sum(reward_losses) / seq_len,
            "continue_loss": sum(continue_losses) / seq_len,
            "trust_score": sum(trust_scores) / seq_len,
        }

    def compute_trust(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute trust scores for given observation-action pairs.

        Trust = low prediction error + high model confidence.

        Args:
            observations: (B, obs_dim)
            actions: (B, action_dim)

        Returns:
            trust_scores: (B,) in [0, 1]
        """
        batch_size = observations.shape[0]
        device = observations.device

        # Run RSSM to get full state
        h, z = self.rssm.initial_state(batch_size, device)
        result = self.rssm.observe_step(h, z, actions, observations)
        state = torch.cat([result["h"], result["z"]], dim=-1)

        # Trust from prediction confidence
        trust = self.trust_head(state).squeeze(-1)
        return trust

    def compute_prediction_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample prediction error for trust scoring.

        Args:
            observations: (B, obs_dim)
            actions: (B, action_dim)
            next_observations: (B, obs_dim)

        Returns:
            errors: (B,) prediction error per sample
        """
        batch_size = observations.shape[0]
        device = observations.device

        h, z = self.rssm.initial_state(batch_size, device)
        result = self.rssm.observe_step(h, z, actions, observations)

        # Predicted next observation (obs_decoder expects concatenated h+z)
        state = torch.cat([result["h"], result["z"]], dim=-1)
        obs_pred = self.obs_decoder(state)

        # Per-sample MSE error
        errors = F.mse_loss(obs_pred, next_observations, reduction="none").mean(dim=-1)
        return errors
