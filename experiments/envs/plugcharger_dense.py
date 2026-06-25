"""Dense-reward PlugCharger for model-based RL.

Stock `PlugCharger-v1` defines **no** `compute_dense_reward` — it inherits BaseEnv's
sparse default (≈ success → 1, else 0). So an RL agent gets zero learning signal until
it stumbles onto a successful two-prong insert by chance, which for that tolerance never
happens from scratch (observed: TD-MPC2 flat at R=0 for 6 h).

This subclass adds a staged shaped reward mirroring `PegInsertionSideEnv.compute_dense_reward`
(reach → grasp → align → seat → success bonus of 10, normalized by /10), with the one piece
PegInsertion never needed: an **angular** alignment term. Success here requires the charger's
two prongs to be co-oriented with the socket (`obj_to_goal_angle <= 0.2`), not just positioned.

Wire it in by importing this module before `gym.make` so registration runs, then train with
`env_id=PlugChargerDense-v1`:

    import experiments.envs.plugcharger_dense  # noqa: F401  (registers PlugChargerDense-v1)

The weights (1 / 3 / 5 / 10) and tanh scales (5, 2, 25) are starting points copied from
PegInsertion's tuned reward; validate that reward rises monotonically with task progress and
that the policy actually reaches the success bonus before trusting the numbers.
"""

from __future__ import annotations

from typing import Any

import torch
from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.utils.registration import register_env


@register_env("PlugChargerDense-v1", max_episode_steps=200)
class PlugChargerDenseEnv(PlugChargerEnv):
    # Stock PlugCharger restricts this to ["none", "sparse"]; re-enable dense now that
    # we provide compute_dense_reward (matches BaseEnv's default).
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        # Stage 1 — reach the charger with the gripper.
        tcp_to_charger = torch.linalg.norm(self.agent.tcp.pose.p - self.charger.pose.p, axis=1)
        reward = 1 - torch.tanh(5.0 * tcp_to_charger)

        # Stage 2 — grasp it.
        grasped = self.agent.is_grasping(self.charger)
        reward = reward + grasped

        # Stage 3 — align charger to socket: BOTH position and angle (the two prongs).
        dist, angle = self._compute_distance()  # success needs dist <= 5mm and angle <= 0.2 rad
        align = 1 - torch.tanh(5.0 * dist + 2.0 * angle)
        reward = reward + 3.0 * align * grasped
        aligned = grasped & (dist < 0.02) & (angle < 0.3)  # looser than success, gates stage 4

        # Stage 4 — seat the prongs: sharp falloff as the gap closes.
        seat = 1 - torch.tanh(25.0 * dist)
        reward = reward + 5.0 * seat * aligned

        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / 10.0
