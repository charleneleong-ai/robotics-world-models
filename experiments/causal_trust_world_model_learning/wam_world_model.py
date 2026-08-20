"""World Action Model (WAM) with Trust Scoring and Mitigation.

Extends RSSM world model with:
1. Action generation head for planning
2. Trust scoring with calibration
3. Feedback correction using real-time observations
4. Verification using forward-inverse cycle consistency
5. Agentic decision layer (explore/exploit/help)

Based on:
- DreamerV3 (Hafner et al., 2023) - RSSM architecture
- Feedback World Model (2026) - online correction
- World Action Verifier (2026) - cycle consistency
- Conformal Prediction - calibration
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical stochastic states."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        stochastic_dim: int = 16,
        stochastic_classes: int = 16,
        deterministic_dim: int = 256,
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
        )

        # Prior network (predicts stochastic state from deterministic state)
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_dim * stochastic_classes),
        )

        # Posterior network (predicts stochastic state from deterministic + observation)
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_dim * stochastic_classes),
        )

        # Deterministic state transition (GRU)
        self.rnn = nn.GRUCell(
            stochastic_dim * stochastic_classes + action_dim,
            deterministic_dim,
        )

        # Observation decoder
        self.obs_decoder = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim * stochastic_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim * stochastic_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Done predictor
        self.done_head = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim * stochastic_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states."""
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step with observation (posterior)."""
        # Encode observation
        obs_enc = self.obs_encoder(obs)

        # Prior
        prior_input = prev_h
        prior_logits = self.prior_net(prior_input)
        prior_logits = prior_logits.view(-1, self.stochastic_dim, self.stochastic_classes)

        # Posterior
        posterior_input = torch.cat([prev_h, obs_enc], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(-1, self.stochastic_dim, self.stochastic_classes)

        # Sample stochastic state (straight-through gradient)
        z = self._sample_categorical(posterior_logits)

        # Deterministic state transition
        rnn_input = torch.cat([z, action], dim=-1)
        h = self.rnn(rnn_input, prev_h)

        return h, z, posterior_logits

    def imagine_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step without observation (prior)."""
        # Prior
        prior_logits = self.prior_net(prev_h)
        prior_logits = prior_logits.view(-1, self.stochastic_dim, self.stochastic_classes)

        # Sample stochastic state
        z = self._sample_categorical(prior_logits)

        # Deterministic state transition
        rnn_input = torch.cat([z, action], dim=-1)
        h = self.rnn(rnn_input, prev_h)

        return h, z, prior_logits

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode observation, reward, done from states."""
        # Concatenate states
        state = torch.cat([h, z], dim=-1)

        # Decode observation
        obs_pred = self.obs_decoder(state)

        # Predict reward
        reward_pred = self.reward_head(state)

        # Predict done
        done_pred = self.done_head(state)

        return obs_pred, reward_pred, done_pred

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with straight-through gradient."""
        # Gumbel-softmax for straight-through gradient
        z = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        z = z.view(-1, self.stochastic_dim * self.stochastic_classes)
        return z


class WorldActionModel(nn.Module):
    """World Action Model: RSSM + Action Generation + Trust Scoring.

    Components:
    1. RSSM: Learns latent dynamics
    2. Action Head: Generates actions for planning
    3. Trust Scorer: Computes trust scores from prediction error
    4. Feedback Corrector: Corrects trust scores using real-time observations
    5. Verifier: Uses forward-inverse cycle consistency
    6. Agentic Layer: Makes explore/exploit/help decisions
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        stochastic_dim: int = 16,
        stochastic_classes: int = 16,
        deterministic_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # RSSM for dynamics
        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
            stochastic_classes=stochastic_classes,
            deterministic_dim=deterministic_dim,
        )

        # Action generation head (for planning)
        self.action_head = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim * stochastic_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

        # Trust scorer
        self.trust_scorer = TrustScorerWithCalibration()

        # Feedback corrector
        self.feedback_corrector = FeedbackCorrector(obs_dim, hidden_dim)

        # Verifier (forward-inverse cycle consistency)
        self.verifier = ForwardInverseVerifier(obs_dim, action_dim, hidden_dim)

        # Agentic layer
        self.agentic_layer = AgenticDecisionLayer()

        # Inverse dynamics model (for verification)
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> dict:
        """Forward pass for training.

        Args:
            obs_seq: (B, T, obs_dim)
            action_seq: (B, T, action_dim)

        Returns:
            Dictionary of predictions and losses
        """
        B, T, _ = obs_seq.shape

        # Initialize states
        h, z = self.rssm.initial_state(B, obs_seq.device)

        # Storage for predictions
        obs_preds = []
        reward_preds = []
        done_preds = []
        posterior_logits_list = []
        prior_logits_list = []

        # Process sequence
        for t in range(T):
            obs = obs_seq[:, t]
            action = action_seq[:, t]

            # Observe step (posterior)
            h, z, posterior_logits = self.rssm.observe_step(h, z, action, obs)

            # Decode predictions
            obs_pred, reward_pred, done_pred = self.rssm.decode(h, z)

            # Prior for training
            _, _, prior_logits = self.rssm.imagine_step(h.detach(), z.detach(), action)

            obs_preds.append(obs_pred)
            reward_preds.append(reward_pred)
            done_preds.append(done_pred)
            posterior_logits_list.append(posterior_logits)
            prior_logits_list.append(prior_logits)

        # Stack predictions
        obs_preds = torch.stack(obs_preds, dim=1)
        reward_preds = torch.stack(reward_preds, dim=1).squeeze(-1)
        done_preds = torch.stack(done_preds, dim=1).squeeze(-1)

        return {
            "obs_preds": obs_preds,
            "reward_preds": reward_preds,
            "done_preds": done_preds,
            "posterior_logits": posterior_logits_list,
            "prior_logits": prior_logits_list,
        }

    def compute_trust_scores(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> dict:
        """Compute trust scores with calibration and feedback correction.

        Args:
            obs: (B, obs_dim)
            action: (B, action_dim)
            next_obs: (B, obs_dim)

        Returns:
            Dictionary with trust scores and diagnostics
        """
        # Forward pass to get predictions
        with torch.no_grad():
            h, z = self.rssm.initial_state(len(obs), obs.device)
            h, z, _ = self.rssm.observe_step(h, z, action, obs)
            obs_pred, _, _ = self.rssm.decode(h, z)

        # Compute prediction error
        pred_error = F.mse_loss(obs_pred, next_obs, reduction="none").mean(dim=-1)

        # Raw trust score (inverse of prediction error)
        raw_trust = torch.exp(-pred_error)

        # Calibrate trust scores
        calibrated_trust = self.trust_scorer.calibrate(raw_trust)

        # Feedback correction using real-time observations
        corrected_trust = self.feedback_corrector.correct(
            calibrated_trust, obs, next_obs, obs_pred
        )

        # Verification using forward-inverse cycle consistency
        verification_score = self.verifier.verify(obs, action, next_obs)

        # Combine trust scores
        final_trust = 0.5 * corrected_trust + 0.5 * verification_score

        return {
            "raw_trust": raw_trust,
            "calibrated_trust": calibrated_trust,
            "corrected_trust": corrected_trust,
            "verification_score": verification_score,
            "final_trust": final_trust,
            "pred_error": pred_error,
        }

    def generate_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate action from observation (for planning).

        Args:
            obs: (B, obs_dim)

        Returns:
            action: (B, action_dim)
        """
        h, z = self.rssm.initial_state(len(obs), obs.device)
        state = torch.cat([h, z], dim=-1)
        return self.action_head(state)

    def get_agentic_decision(
        self,
        trust_score: torch.Tensor,
        confidence_threshold: float = 0.7,
        exploration_threshold: float = 0.3,
    ) -> dict:
        """Make agentic decision based on trust score.

        Args:
            trust_score: (B,) trust scores
            confidence_threshold: threshold for high trust
            exploration_threshold: threshold for low trust

        Returns:
            Dictionary with decisions
        """
        return self.agentic_layer.decide(
            trust_score, confidence_threshold, exploration_threshold
        )


class TrustScorerWithCalibration(nn.Module):
    """Trust scoring with conformal prediction calibration."""

    def __init__(self, calibration_size: int = 100):
        super().__init__()
        self.calibration_size = calibration_size
        self.calibration_scores: list[float] = []
        self.quantile: Optional[float] = None

    def calibrate(self, raw_trust: torch.Tensor) -> torch.Tensor:
        """Calibrate trust scores using conformal prediction.

        Args:
            raw_trust: (B,) raw trust scores

        Returns:
            calibrated_trust: (B,) calibrated trust scores
        """
        # Store calibration scores
        if len(self.calibration_scores) < self.calibration_size:
            self.calibration_scores.extend(raw_trust.cpu().numpy().tolist())

        # Compute quantile for calibration
        if len(self.calibration_scores) >= self.calibration_size and self.quantile is None:
            scores = np.array(self.calibration_scores)
            self.quantile = np.percentile(scores, 90)  # 90% coverage

        # Apply calibration
        if self.quantile is not None:
            # Scale trust scores to achieve desired coverage
            calibrated = raw_trust / (self.quantile + 1e-8)
            calibrated = torch.clamp(calibrated, 0, 1)
        else:
            calibrated = raw_trust

        return calibrated


class FeedbackCorrector(nn.Module):
    """Correct trust scores using real-time observations.

    Based on Feedback World Model (2026):
    - Uses real observations to correct predictions online
    - Reduces prediction error by up to 76.4% under OOD conditions
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feedback_net = nn.Sequential(
            nn.Linear(obs_dim * 3, hidden_dim),  # obs, next_obs, pred_obs
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def correct(
        self,
        trust_score: torch.Tensor,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        pred_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Correct trust scores using real-time feedback.

        Args:
            trust_score: (B,) raw trust scores
            obs: (B, obs_dim) current observation
            next_obs: (B, obs_dim) next observation
            pred_obs: (B, obs_dim) predicted next observation

        Returns:
            corrected_trust: (B,) corrected trust scores
        """
        # Compute feedback signal
        feedback_input = torch.cat([obs, next_obs, pred_obs], dim=-1)
        feedback_signal = self.feedback_net(feedback_input).squeeze(-1)

        # Correct trust score
        corrected = trust_score * (0.5 + 0.5 * feedback_signal)

        return corrected


class ForwardInverseVerifier(nn.Module):
    """Verify trust scores using forward-inverse cycle consistency.

    Based on World Action Verifier (2026):
    - Decomposes prediction into state plausibility and action reachability
    - Uses cycle consistency for verification
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Forward model (state -> action -> next_state)
        self.forward_model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Inverse model (state, next_state -> action)
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def verify(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Verify using forward-inverse cycle consistency.

        Args:
            obs: (B, obs_dim)
            action: (B, action_dim)
            next_obs: (B, obs_dim)

        Returns:
            verification_score: (B,) verification scores
        """
        # Forward prediction
        forward_input = torch.cat([obs, action], dim=-1)
        pred_next_obs = self.forward_model(forward_input)

        # Inverse prediction
        inverse_input = torch.cat([obs, next_obs], dim=-1)
        pred_action = self.inverse_model(inverse_input)

        # Cycle consistency
        forward_error = F.mse_loss(pred_next_obs, next_obs, reduction="none").mean(dim=-1)
        inverse_error = F.mse_loss(pred_action, action, reduction="none").mean(dim=-1)

        # Verification score (low error = high verification)
        forward_score = torch.exp(-forward_error)
        inverse_score = torch.exp(-inverse_error)

        # Combined score
        verification_score = 0.5 * forward_score + 0.5 * inverse_score

        return verification_score


class AgenticDecisionLayer(nn.Module):
    """Agentic decision layer for explore/exploit/help decisions.

    Uses trust scores to make decisions:
    - High trust (>confidence_threshold): Execute action (exploit)
    - Medium trust: Explore (try something new)
    - Low trust (<exploration_threshold): Ask for help
    """

    def __init__(self):
        super().__init__()

    def decide(
        self,
        trust_score: torch.Tensor,
        confidence_threshold: float = 0.7,
        exploration_threshold: float = 0.3,
    ) -> dict:
        """Make agentic decision based on trust score.

        Args:
            trust_score: (B,) trust scores
            confidence_threshold: threshold for high trust
            exploration_threshold: threshold for low trust

        Returns:
            Dictionary with decisions
        """
        # High trust: exploit
        exploit = trust_score > confidence_threshold

        # Low trust: ask for help
        ask_help = trust_score < exploration_threshold

        # Medium trust: explore
        explore = ~exploit & ~ask_help

        return {
            "exploit": exploit,
            "explore": explore,
            "ask_help": ask_help,
            "trust_score": trust_score,
        }
