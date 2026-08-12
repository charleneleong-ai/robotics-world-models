"""Continual learning via causal trust world model verification.

This module enables robots to learn new capabilities while preserving old knowledge
by using trust scoring to decide when to reuse old knowledge vs collect new data.

Key insight: Trust scoring tells you when old knowledge is still valid for new tasks,
and when you need to collect new real-world data to update the world model.

Usage:
    # Learn new task while preserving old knowledge
    python -m experiments.causal_trust_world_model_learning.continual_learning \\
        --old-tasks pickcube peginsertion \\
        --new-tasks plugcharger \\
        --trust-threshold 0.8
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning."""
    trust_threshold: float = 0.8  # Minimum trust to reuse old knowledge
    max_replay_buffer: int = 10000  # Maximum replay buffer size
    consolidation_interval: int = 100  # Steps between memory consolidation
    knowledge_transfer_threshold: float = 0.7  # Minimum trust for knowledge transfer


class ContinualLearner:
    """Continual learner that uses trust scoring to manage knowledge transfer.
    
    The continual learner:
    1. Tracks trust scores for old tasks on new data
    2. Decides when to reuse old knowledge vs collect new data
    3. Consolidates knowledge to prevent catastrophic forgetting
    4. Transfers knowledge across tasks when trust is high
    """
    
    def __init__(
        self,
        world_model_verifier: Any,
        vla_policy: Any,
        config: ContinualLearningConfig,
    ):
        self.verifier = world_model_verifier
        self.vla = vla_policy
        self.config = config
        
        # Task memory
        self.task_memories: dict[str, list[dict[str, Any]]] = {}
        self.task_trust_scores: dict[str, list[float]] = {}
        
        # Replay buffer for experience replay
        self.replay_buffer: list[dict[str, Any]] = []
    
    def learn_new_task(
        self,
        task_name: str,
        environment: Any,
        num_episodes: int = 100,
    ) -> dict[str, Any]:
        """Learn a new task while preserving old knowledge.
        
        Args:
            task_name: Name of the new task
            environment: Task environment
            num_episodes: Number of episodes to collect
            
        Returns:
            Dictionary with learning statistics
        """
        stats = {
            "episodes_collected": 0,
            "old_knowledge_reused": 0,
            "new_data_collected": 0,
            "trust_scores": [],
        }
        
        for episode in range(num_episodes):
            # Collect experience
            episode_data = self._collect_episode(environment)
            
            # Check if old knowledge can be reused
            trust_score = self._evaluate_trust_for_task(
                task_name, episode_data
            )
            
            if trust_score >= self.config.trust_threshold:
                # Reuse old knowledge
                stats["old_knowledge_reused"] += 1
                self._reuse_old_knowledge(task_name, episode_data, trust_score)
            else:
                # Collect new data
                stats["new_data_collected"] += 1
                self._collect_new_data(task_name, episode_data)
            
            stats["trust_scores"].append(trust_score)
            stats["episodes_collected"] += 1
            
            # Consolidate memory periodically
            if episode % self.config.consolidation_interval == 0:
                self._consolidate_memory(task_name)
        
        # Store task memory
        self.task_memories[task_name] = self._get_task_memory(task_name)
        self.task_trust_scores[task_name] = stats["trust_scores"]
        
        return stats
    
    def _collect_episode(self, environment: Any) -> dict[str, Any]:
        """Collect one episode of experience."""
        obs = environment.reset()
        episode_data = {"observations": [], "actions": [], "rewards": []}
        
        done = False
        while not done:
            # Generate action using VLA policy
            # The VLA policy can be either a VLAPolicy object or a simple model
            if hasattr(self.vla, 'generate_candidates'):
                # VLAPolicy interface
                candidates = self.vla.generate_candidates(obs, "")
                action = candidates[0] if candidates else torch.zeros(1, obs.shape[-1])
            else:
                # Simple model interface
                action = self.vla.predict(obs)
            
            # Step environment
            next_obs, reward, done, _ = environment.step(action)
            
            # Store data
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            
            obs = next_obs
        
        return episode_data
    
    def _evaluate_trust_for_task(
        self,
        task_name: str,
        episode_data: dict[str, Any],
    ) -> float:
        """Evaluate trust score for old knowledge on new task.
        
        This checks if old knowledge from previous tasks is still valid
        for the new task.
        """
        trust_scores = []
        
        for obs, action in zip(
            episode_data["observations"],
            episode_data["actions"],
        ):
            # Use world model verifier to check trust
            verification = self.verifier.verify(obs, action)
            trust_scores.append(verification.trust_score)
        
        # Return average trust score
        return sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
    
    def _reuse_old_knowledge(
        self,
        task_name: str,
        episode_data: dict[str, Any],
        trust_score: float,
    ):
        """Reuse old knowledge when trust is high."""
        # Add to replay buffer with trust-weighted importance
        for obs, action, reward in zip(
            episode_data["observations"],
            episode_data["actions"],
            episode_data["rewards"],
        ):
            self.replay_buffer.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "task": task_name,
                "trust_score": trust_score,
                "source": "old_knowledge",
            })
        
        # Trim replay buffer if needed
        if len(self.replay_buffer) > self.config.max_replay_buffer:
            self.replay_buffer = self.replay_buffer[-self.config.max_replay_buffer:]
    
    def _collect_new_data(
        self,
        task_name: str,
        episode_data: dict[str, Any],
    ):
        """Collect new data when trust is low."""
        # Add to replay buffer as new data
        for obs, action, reward in zip(
            episode_data["observations"],
            episode_data["actions"],
            episode_data["rewards"],
        ):
            self.replay_buffer.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "task": task_name,
                "trust_score": 0.0,
                "source": "new_data",
            })
    
    def _consolidate_memory(self, task_name: str):
        """Consolidate memory to prevent catastrophic forgetting.
        
        This selects the most important experiences to keep in the replay buffer,
        prioritizing high-trust experiences and diverse tasks.
        """
        if not self.replay_buffer:
            return
        
        # Sort by trust score (descending) and task diversity
        task_counts = {}
        for exp in self.replay_buffer:
            task = exp["task"]
            task_counts[task] = task_counts.get(task, 0) + 1
        
        # Score experiences by trust and diversity
        scored_experiences = []
        for exp in self.replay_buffer:
            trust_score = exp["trust_score"]
            task = exp["task"]
            diversity_score = 1.0 / task_counts[task]
            
            # Combined score
            score = trust_score + diversity_score
            scored_experiences.append((score, exp))
        
        # Keep top experiences
        scored_experiences.sort(reverse=True, key=lambda x: x[0])
        self.replay_buffer = [
            exp for _, exp in scored_experiences[:self.config.max_replay_buffer]
        ]
    
    def _get_task_memory(self, task_name: str) -> list[dict[str, Any]]:
        """Get memory for a specific task."""
        return [
            exp for exp in self.replay_buffer
            if exp["task"] == task_name
        ]
    
    def transfer_knowledge(
        self,
        source_task: str,
        target_task: str,
        environment: Any,
    ) -> dict[str, Any]:
        """Transfer knowledge from source task to target task.
        
        Uses trust scoring to determine how much knowledge can be transferred.
        """
        # Get source task memory
        source_memory = self.task_memories.get(source_task, [])
        
        if not source_memory:
            return {"transfer_success": False, "reason": "no_source_memory"}
        
        # Evaluate trust for source knowledge on target task
        trust_scores = []
        for exp in source_memory[:100]:  # Sample for efficiency
            trust_score, _, _ = self.verifier.verify(
                exp["observation"], exp["action"]
            )
            trust_scores.append(trust_score)
        
        avg_trust = sum(trust_scores) / len(trust_scores)
        
        if avg_trust >= self.config.knowledge_transfer_threshold:
            # Transfer knowledge
            for exp in source_memory:
                self.replay_buffer.append({
                    "observation": exp["observation"],
                    "action": exp["action"],
                    "reward": exp["reward"],
                    "task": target_task,
                    "trust_score": exp["trust_score"],
                    "source": f"transfer_from_{source_task}",
                })
            
            return {
                "transfer_success": True,
                "knowledge_transferred": len(source_memory),
                "average_trust": avg_trust,
            }
        else:
            return {
                "transfer_success": False,
                "reason": "trust_too_low",
                "average_trust": avg_trust,
            }
