"""WM-guided exploration using Cross-Entropy Method (CEM) planning.

Uses the trained World Action Model to plan actions that maximize
reward + exploration bonus, replacing random data collection.

Architecture:
    1. Sample K candidate action sequences (K=200, horizon=8)
    2. Simulate each sequence through the WAM
    3. Score by: reward prediction + exploration_bonus(prediction_uncertainty)
    4. Select top-K, refit distribution, repeat for N iterations
    5. Execute the best action sequence in the real environment

Key insight: The WAM IS the planner — no separate value function needed.
The parallel action+state denoising heads naturally couple planning with
world modeling.
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class CEMConfig:
    """Cross-Entropy Method planner configuration."""
    horizon: int = 8
    num_samples: int = 200
    num_top_k: int = 20
    num_iterations: int = 5
    exploration_weight: float = 0.1
    action_bounds: tuple[float, float] = (-1.0, 1.0)
    momentum: float = 0.0
    noise_std: float = 0.5


class WMPlanner:
    """CEM-based planner that uses the WAM for action selection.

    The planner maintains a Gaussian distribution over action sequences,
    iteratively refines it by simulating candidates through the WAM,
    and selects actions that maximize predicted reward + exploration bonus.
    """

    def __init__(self, wam, config: CEMConfig | None = None):
        """
        Args:
            wam: trained DiffusionWAM model
            config: CEM hyperparameters
        """
        self.wam = wam
        self.config = config or CEMConfig()
        self.device = next(wam.parameters()).device

        # Initialize distribution parameters
        act_dim = wam.act_dim
        self.mean = torch.zeros(self.config.horizon, act_dim, device=self.device)
        self.std = torch.ones(self.config.horizon, act_dim, device=self.device) * self.config.noise_std

    def plan(self, state: np.ndarray, reward_fn=None) -> np.ndarray:
        """Plan a sequence of actions from the current state.

        Args:
            state: current observation [obs_dim] or [1, obs_dim]
            reward_fn: optional callable(state, action, next_state) -> reward
                       If None, uses prediction uncertainty as exploration bonus

        Returns:
            best_action [act_dim] — the first action of the best sequence
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        for _ in range(self.config.num_iterations):
            # Sample action sequences from current distribution
            actions = self.mean + self.std * torch.randn(
                self.config.num_samples, self.config.horizon, self.wam.act_dim,
                device=self.device
            )
            actions = actions.clamp(*self.config.action_bounds)

            # Simulate each sequence through the WAM
            scores = self._evaluate_sequences(state_t, actions, reward_fn)

            # Select top-K
            topk_vals, topk_idx = scores.topk(self.config.num_top_k)
            topk_actions = actions[topk_idx]

            # Refit distribution to top-K
            self.mean = topk_actions.mean(dim=0)
            self.std = topk_actions.std(dim=0).clamp(min=0.01)

        # Return the first action of the best sequence
        best_action = self.mean[0]
        return best_action.cpu().numpy()

    def _evaluate_sequences(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        reward_fn=None,
    ) -> torch.Tensor:
        """Simulate action sequences through WAM and compute scores.

        Args:
            state: [1, obs_dim]
            actions: [K, horizon, act_dim]
            reward_fn: optional reward function

        Returns:
            scores: [K] — higher is better
        """
        K = actions.shape[0]
        state_expanded = state.expand(K, -1)  # [K, obs_dim]

        total_reward = torch.zeros(K, device=self.device)
        total_uncertainty = torch.zeros(K, device=self.device)
        s = state_expanded

        for h in range(self.config.horizon):
            a = actions[:, h]  # [K, act_dim]

            # Predict next state
            s_next = self.wam.predict_next_state(s, a, num_steps=10)

            # Compute reward
            if reward_fn is not None:
                r = reward_fn(s, a, s_next)
                total_reward += r
            else:
                # Use prediction uncertainty as exploration bonus
                # Higher uncertainty = more interesting state
                uncertainty = (s_next - s).abs().mean(dim=1)
                total_uncertainty += uncertainty

            s = s_next

        # Score = reward + exploration bonus
        scores = total_reward + self.config.exploration_weight * total_uncertainty
        return scores

    def reset(self):
        """Reset distribution to prior."""
        act_dim = self.wam.act_dim
        self.mean = torch.zeros(self.config.horizon, act_dim, device=self.device)
        self.std = torch.ones(self.config.horizon, act_dim, device=self.device) * self.config.noise_std


class WMGuidedCollector:
    """Collector that uses WM planning instead of random actions.

    Alternates between:
    1. CEM planning (uses WAM to select actions)
    2. Execution (runs planned actions in real environment)
    3. Data collection (records transitions)
    """

    def __init__(
        self,
        env,
        wam,
        planner_config: CEMConfig | None = None,
        plan_every: int = 1,
        fallback_to_random: bool = True,
    ):
        """
        Args:
            env: ManiSkill3 environment
            wam: trained DiffusionWAM
            planner_config: CEM hyperparameters
            plan_every: re-plan every N steps (1 = every step)
            fallback_to_random: if True, use random actions when WAM is untrained
        """
        self.env = env
        self.wam = wam
        self.planner = WMPlanner(wam, planner_config)
        self.plan_every = plan_every
        self.fallback_to_random = fallback_to_random
        self._step_count = 0

    def collect_episode(self, max_steps: int = 200) -> dict:
        """Collect one episode using WM-guided exploration.

        Returns:
            dict with keys: obs, action, next_obs, reward, done, success
        """
        obs, _ = self.env.reset()
        obs_list, action_list, next_obs_list, reward_list = [], [], [], []

        for step in range(max_steps):
            obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs

            # Plan or use random
            if self._step_count % self.plan_every == 0:
                try:
                    action = self.planner.plan(obs_np)
                except Exception:
                    if self.fallback_to_random:
                        action = self.env.action_space.sample()
                    else:
                        raise
            self._step_count += 1

            # Execute
            if isinstance(action, np.ndarray):
                action_np = action
            else:
                action_np = np.array(action)

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated | truncated

            # Record
            obs_list.append(obs_np.flatten())
            action_list.append(action_np.flatten())
            next_obs_list.append(
                next_obs.cpu().numpy().flatten()
                if isinstance(next_obs, torch.Tensor)
                else np.array(next_obs).flatten()
            )
            reward_list.append(float(reward) if not isinstance(reward, torch.Tensor) else float(reward.item()))

            obs = next_obs
            if bool(done) if not isinstance(done, torch.Tensor) else bool(done.item()):
                break

        success = float(info.get("success", 0))
        if isinstance(success, torch.Tensor):
            success = float(success.item())

        return {
            "obs": np.array(obs_list),
            "action": np.array(action_list),
            "next_obs": np.array(next_obs_list),
            "reward": np.array(reward_list),
            "done": np.array([True] * len(obs_list)),
            "success": success,
            "total_reward": sum(reward_list),
        }

    def update_wam(self, new_wam):
        """Update the planner with a newly trained WAM."""
        self.wam = new_wam
        self.planner = WMPlanner(new_wam, self.planner.config)
        self._step_count = 0
