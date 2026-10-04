"""Dense reward shaping for ManiSkill3 manipulation tasks.

Provides dense reward signals to supplement sparse success rewards:
- Keypoint distances (gripper to object, object to target)
- Contact rewards (finger contact, grasp detection)
- Progress rewards (task completion progress)
- Smoothness penalties
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np


@dataclass
class DenseRewardConfig:
    """Configuration for dense reward shaping."""
    # Keypoint distance weights
    gripper_to_object_weight: float = 1.0
    object_to_target_weight: float = 2.0
    gripper_to_target_weight: float = 0.5
    
    # Contact rewards
    contact_reward: float = 0.1
    grasp_reward: float = 0.5
    
    # Progress reward
    progress_weight: float = 1.0
    
    # Penalties
    action_penalty_weight: float = 0.01
    collision_penalty: float = 0.1
    
    # Normalization
    distance_scale: float = 0.1  # meters


class DenseRewardComputer:
    """Computes dense rewards for manipulation tasks."""
    
    def __init__(self, config: DenseRewardConfig, task_name: str):
        self.config = config
        self.task_name = task_name
        self._prev_distances: dict = {}
        self._prev_grasp: bool = False
    
    def compute_keypoint_distances(
        self, 
        obs: np.ndarray,
        task_name: str
    ) -> dict[str, float]:
        """Extract keypoint distances from observation.
        
        Observation format depends on task:
        - PickCube: [qpos(14), obj_pos(3), obj_quat(4), target_pos(3), ...]
        - PegInsertion: [qpos(14), peg_pos(3), peg_quat(4), hole_pos(3), ...]
        - PushCube: [qpos(14), cube_pos(3), target_pos(3), ...]
        - PlugCharger: [qpos(14), plug_pos(3), socket_pos(3), ...]
        """
        distances = {}
        
        if task_name == "PickCube-v1":
            # obs[14:17] = object position, obs[17:20] = target position
            # gripper pos = obs[0:3] (first 3 of qpos)
            gripper_pos = obs[:3]
            obj_pos = obs[14:17]
            target_pos = obs[17:20]
            
            d_gripper_obj = np.linalg.norm(gripper_pos - obj_pos)
            d_obj_target = np.linalg.norm(obj_pos - target_pos)
            d_gripper_target = np.linalg.norm(gripper_pos - target_pos)
            
            distances = {
                'gripper_to_object': d_gripper_obj,
                'object_to_target': d_obj_target,
                'gripper_to_target': d_gripper_target,
            }
            
        elif task_name == "PegInsertionSide-v1":
            # obs[14:17] = peg pos, obs[17:20] = hole pos
            gripper_pos = obs[:3]
            peg_pos = obs[14:17]
            hole_pos = obs[17:20]
            
            d_gripper_peg = np.linalg.norm(gripper_pos - peg_pos)
            d_peg_hole = np.linalg.norm(peg_pos - hole_pos)
            d_gripper_hole = np.linalg.norm(gripper_pos - hole_pos)
            
            distances = {
                'gripper_to_object': d_gripper_peg,
                'object_to_target': d_peg_hole,
                'gripper_to_target': d_gripper_hole,
            }
            
        elif task_name == "PushCube-v1":
            gripper_pos = obs[:3]
            cube_pos = obs[14:17]
            target_pos = obs[17:20]
            
            d_gripper_cube = np.linalg.norm(gripper_pos - cube_pos)
            d_cube_target = np.linalg.norm(cube_pos - target_pos)
            d_gripper_target = np.linalg.norm(gripper_pos - target_pos)
            
            distances = {
                'gripper_to_object': d_gripper_cube,
                'object_to_target': d_cube_target,
                'gripper_to_target': d_gripper_target,
            }
            
        elif task_name == "PlugCharger-v1":
            gripper_pos = obs[:3]
            plug_pos = obs[14:17]
            socket_pos = obs[17:20]
            
            d_gripper_plug = np.linalg.norm(gripper_pos - plug_pos)
            d_plug_socket = np.linalg.norm(plug_pos - socket_pos)
            d_gripper_socket = np.linalg.norm(gripper_pos - socket_pos)
            
            distances = {
                'gripper_to_object': d_gripper_plug,
                'object_to_target': d_plug_socket,
                'gripper_to_target': d_gripper_socket,
            }
        
        else:
            distances = {
                'gripper_to_object': 0.0,
                'object_to_target': 0.0,
                'gripper_to_target': 0.0,
            }
        
        return distances
    
    def compute_contact_rewards(
        self, 
        obs: np.ndarray,
        info: dict
    ) -> tuple[float, bool]:
        """Compute contact and grasp rewards."""
        contact_reward = 0.0
        is_grasping = False
        
        # Check for contact (from info if available)
        if 'contact' in info:
            contact = info['contact']
            if isinstance(contact, np.ndarray):
                if contact.any():
                    contact_reward = self.config.contact_reward
        
        # Grasp detection (gripper closed + object contact)
        # Gripper position is first 2 dims of action (for 2-finger gripper)
        # Or check gripper qpos (obs[12:14] typically)
        if len(info.get('grasp', [])) > 0:
            is_grasping = True
            contact_reward += self.config.grasp_reward
        elif info.get('is_grasping', False):
            is_grasping = True
            contact_reward += self.config.grasp_reward
        
        return contact_reward, is_grasping
    
    def compute_progress_reward(
        self, 
        current_distances: dict, 
        prev_distances: Optional[dict]
    ) -> float:
        """Compute progress reward based on distance reduction."""
        if prev_distances is None:
            return 0.0
        
        progress = 0.0
        # Reward for reducing object-to-target distance
        if 'object_to_target' in current_distances and 'object_to_target' in prev_distances:
            reduction = prev_distances['object_to_target'] - current_distances['object_to_target']
            progress += max(0, reduction) * self.config.progress_weight / self.config.distance_scale
        
        # Reward for reducing gripper-to-target distance
        if 'gripper_to_target' in current_distances and 'gripper_to_target' in prev_distances:
            reduction = prev_distances['gripper_to_target'] - current_distances['gripper_to_target']
            progress += max(0, reduction) * self.config.progress_weight / self.config.distance_scale * 0.5
        
        return progress
    
    def compute_action_penalty(self, action: np.ndarray) -> float:
        """Penalize large actions for smoothness."""
        return self.config.action_penalty_weight * np.mean(np.abs(action))
    
    def compute(self, obs: np.ndarray, action: np.ndarray, info: dict, prev_distances: Optional[dict] = None) -> dict:
        """Compute all dense reward components."""
        # Keypoint distances
        distances = self.compute_keypoint_distances(obs, self.task_name)
        
        # Contact/grasp
        contact_reward, is_grasping = self.compute_contact_rewards(obs, {})
        
        # Progress
        progress = self.compute_progress_reward(distances, self._prev_distances)
        
        # Action penalty
        action_penalty = self.compute_action_penalty(action)
        
        # Total dense reward
        dense_reward = 0.0
        scale = self.config.distance_scale
        
        # Distance-based rewards (negative distances -> positive rewards)
        gripper_obj = distances.get('gripper_to_object', 0.0)
        obj_target = distances.get('object_to_target', 0.0)
        gripper_target = distances.get('gripper_to_target', 0.0)
        
        dense_reward += self.config.gripper_to_object_weight * max(0, 1.0 - gripper_obj / scale)
        dense_reward += self.config.object_to_target_weight * max(0, 1.0 - obj_target / scale)
        dense_reward += self.config.gripper_to_target_weight * max(0, 1.0 - gripper_target / scale)
        
        dense_reward += contact_reward
        dense_reward += progress
        dense_reward -= action_penalty
        
        # Update state
        self._prev_distances = distances
        
        return {
            'dense_reward': dense_reward,
            'distance_rewards': {
                'gripper_to_object': self.config.gripper_to_object_weight * max(0, 1.0 - gripper_obj / scale),
                'object_to_target': self.config.object_to_target_weight * max(0, 1.0 - obj_target / scale),
                'gripper_to_target': self.config.gripper_to_target_weight * max(0, 1.0 - gripper_target / scale),
            },
            'contact_reward': contact_reward,
            'progress_reward': progress,
            'action_penalty': -action_penalty,
            'is_grasping': is_grasping,
        }
