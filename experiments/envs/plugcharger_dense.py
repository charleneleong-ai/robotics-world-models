"""Dense-reward PlugCharger for model-based RL.

Stock `PlugCharger-v1` defines **no** `compute_dense_reward` — it inherits BaseEnv's
sparse default (≈ success → 1, else 0). So an RL agent gets zero learning signal until
it stumbles onto a successful two-prong insert by chance, which for that tolerance never
happens from scratch (observed: TD-MPC2 flat at R=0 for 6 h).

This subclass adds a shaped reward in the spirit of `PegInsertionSideEnv.compute_dense_reward`
(reach → grasp → approach-success → success bonus, normalized), with the one piece PegInsertion
never needed: an **angular** alignment term. Success here requires the charger's two prongs to
be co-oriented with the socket (`obj_to_goal_angle <= 0.2`), not just positioned.

Wire it in by importing this module before `gym.make` so registration runs, then train with
`env_id=PlugChargerDense-v1`:

    import experiments.envs.plugcharger_dense  # noqa: F401  (registers PlugChargerDense-v1)

v1 used flat staged weights (reach=1/grasp=1/align=3/seat=5) and the policy farmed the early
stages (R 3.4 -> 104 over 1M steps, eval success flat at 0). v2 rebalanced so completion
dominates (precursors shrunk to 0.1, one success-tracking term + a 25 bonus). v3 adds a small
0.1 *touch* precursor so the early ladder is reach -> touch -> grasp -> insert — denser early
guidance, still farm-proof. v3 ran a full 2M: eval success 0, but `train/success_once` grazed
0.03 (1/32 envs) ~10× while `success_at_end` stayed ~0 — the policy *touched* success but never
*held* it. v4 fixes that: the success bonus is now **additive** (v3 overwrote reward to a flat 25,
killing the gradient inside the success region) so a held, centred insert always pays strictly
more than hovering at the boundary. Still validate that eval success actually appears before
trusting it — a rising reward alone is not enough.
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
        # v2, completion-dominant. v1 (reach=1 + grasp=1 + align=3 + seat=5 per step) let the
        # agent farm the early stages: a 200-step episode returned ~100 while never inserting
        # (R 3.4 -> 104 with eval success flat at 0). Fix: precursors shrunk to negligible, and
        # the one dominant per-step term now tracks the *success condition itself* (pos AND angle),
        # so the only way to accumulate reward is to actually approach a real insertion.

        # Tiny precursors — guide reach -> touch -> grasp, each 0.1 so the ladder is followable
        # but can't be farmed (ceiling ~0.3/step vs the ~24 reward at a real, held insert).
        tcp_to_charger = torch.linalg.norm(self.agent.tcp.pose.p - self.charger.pose.p, axis=1)
        reward = 0.1 * (1 - torch.tanh(5.0 * tcp_to_charger))
        # touch: either gripper finger in contact with the charger (same primitive is_grasping uses)
        lforce = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.charger), axis=1
        )
        rforce = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.charger), axis=1
        )
        reward = reward + 0.1 * ((lforce > 0.1) | (rforce > 0.1))
        grasped = self.agent.is_grasping(self.charger)
        reward = reward + 0.1 * grasped

        # Dominant signal: a CONTINUOUS approach term that keeps rising all the way to a perfect,
        # centred insert -- so there is always a gradient pulling the charger deeper into the socket
        # and *holding* it there. (v3 OVERWROTE reward to a flat 25 at success, erasing that gradient
        # inside the success region, so the policy touched success then drifted out -> success_at_end
        # ~ 0 despite 10 momentary success_once sparks. This is the v4 "reward the hold" fix.)
        dist, angle = self._compute_distance()  # success: dist <= 5mm AND angle <= 0.2 rad
        insertion = 1 - torch.tanh(8.0 * dist + 4.0 * angle)
        reward = reward + 4.0 * insertion * grasped

        # Additive success bonus (NOT an overwrite): a held, well-centred insert always pays strictly
        # more than hovering at the success boundary -> rewards holding, not just touching. Max reward
        # at a perfect insert ~= 0.3 (precursors) + 4 (insertion) + 20 (bonus) = 24.3.
        reward = reward + 20.0 * info["success"].float()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / 25.0
