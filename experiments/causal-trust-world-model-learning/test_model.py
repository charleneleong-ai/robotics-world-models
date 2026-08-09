"""Unit tests for causal trust world model learning components.

Usage:
    python -m pytest experiments/causal_trust_world_model_learning/test_model.py -v
"""
import pytest
import torch

from experiments.causal_trust_world_model_learning.world_model_verifier import (
    WorldModelVerifier,
    VerificationResult,
)
from experiments.causal_trust_world_model_learning.trust_scoring import TrustScorer
from experiments.causal_trust_world_model_learning.causal_attribution import (
    CausalAttributionEngine,
    FailureMechanism,
)
from experiments.causal_trust_world_model_learning.recovery_strategies import (
    RecoveryStrategies,
)
from experiments.causal_trust_world_model_learning.vla_policy import VLAPolicy
from experiments.causal_trust_world_model_learning.continual_learning import (
    ContinualLearner,
)


class MockWorldModel:
    """Mock world model for testing."""

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return obs + action * 0.1


class MockVLA:
    """Mock VLA policy for testing."""

    def predict(self, obs: torch.Tensor, instruction: str = "") -> torch.Tensor:
        return torch.randn(obs.shape[0], obs.shape[1])


class TestWorldModelVerifier:
    def test_verify_returns_trust_score(self):
        world_model = MockWorldModel()
        verifier = WorldModelVerifier(world_model)

        obs = torch.randn(1, 10)
        action = torch.randn(1, 4)
        result = verifier.verify(obs, action)

        assert isinstance(result, VerificationResult)
        assert 0 <= result.trust_score <= 1
        assert result.predicted_next.shape == obs.shape

    def test_verify_trust_score_range(self):
        world_model = MockWorldModel()
        verifier = WorldModelVerifier(world_model)

        for _ in range(10):
            obs = torch.randn(1, 10)
            action = torch.randn(1, 4)
            result = verifier.verify(obs, action)
            assert 0 <= result.trust_score <= 1


class TestTrustScorer:
    def test_compute_trust_score(self):
        scorer = TrustScorer()

        trust_score = scorer.compute_trust_score(
            physics_consistency=0.8,
            ood_score=0.9,
            calibration_error=0.05,
            prediction_confidence=0.85,
        )

        assert 0 <= trust_score.overall <= 1
        assert isinstance(trust_score.is_trustworthy, bool)

    def test_trust_threshold(self):
        scorer = TrustScorer(trust_threshold=0.7)

        high_trust = scorer.compute_trust_score(
            physics_consistency=0.9,
            ood_score=0.95,
            calibration_error=0.01,
            prediction_confidence=0.9,
        )
        assert high_trust.is_trustworthy

        low_trust = scorer.compute_trust_score(
            physics_consistency=0.3,
            ood_score=0.4,
            calibration_error=0.5,
            prediction_confidence=0.3,
        )
        assert not low_trust.is_trustworthy


class TestCausalAttribution:
    def test_diagnose_returns_attribution(self):
        engine = CausalAttributionEngine()

        obs = torch.randn(1, 10)
        action = torch.randn(1, 4)
        predicted_next = torch.randn(1, 10)
        actual_next = torch.randn(1, 10)

        attribution = engine.diagnose(obs, action, predicted_next, actual_next)

        assert isinstance(attribution.mechanism, FailureMechanism)
        assert 0 <= attribution.confidence <= 1
        assert 0 <= attribution.severity <= 1

    def test_contact_failure_detection(self):
        engine = CausalAttributionEngine()

        obs = torch.randn(1, 10)
        action = torch.randn(1, 4)
        predicted_next = obs + action * 0.1
        actual_next = obs + action * 0.5  # Large difference

        attribution = engine.diagnose(obs, action, predicted_next, actual_next)

        assert attribution.mechanism in [
            FailureMechanism.CONTACT,
            FailureMechanism.VISUAL,
            FailureMechanism.DYNAMIC,
        ]


class TestRecoveryStrategies:
    def test_apply_recovery(self):
        recovery = RecoveryStrategies()

        obs = torch.randn(1, 10)
        causal_attribution = CausalAttribution(
            mechanism=FailureMechanism.CONTACT,
            confidence=0.8,
            features={"contact_score": 0.9},
            recovery_recommendation="Adjust contact parameters",
            severity=0.3,
        )

        result = recovery.apply_recovery(obs, causal_attribution)

        assert result.success
        assert result.recovered_state.shape == obs.shape

    def test_mechanism_specific_recovery(self):
        recovery = RecoveryStrategies()

        obs = torch.randn(1, 10)

        # Test contact recovery
        contact_attribution = CausalAttribution(
            mechanism=FailureMechanism.CONTACT,
            confidence=0.8,
            features={},
            recovery_recommendation="Adjust contact",
            severity=0.3,
        )
        result = recovery.apply_recovery(obs, contact_attribution)
        assert result.strategy_used.startswith("adjust_contact")

        # Test visual recovery
        visual_attribution = CausalAttribution(
            mechanism=FailureMechanism.VISUAL,
            confidence=0.8,
            features={},
            recovery_recommendation="Re-sample visual",
            severity=0.3,
        )
        result = recovery.apply_recovery(obs, visual_attribution)
        assert result.strategy_used == "resample_visual"


class TestVLAPolicy:
    def test_select_action(self):
        vla = VLAPolicy(MockVLA(), num_candidates=4, trust_threshold=0.7)

        obs = torch.randn(1, 10)
        instruction = "pick up the cube"

        world_model = MockWorldModel()
        verifier = WorldModelVerifier(world_model)

        action, trust_score = vla.select_action(obs, instruction, verifier)

        assert action.shape == obs.shape
        assert 0 <= trust_score <= 1

    def test_statistics(self):
        vla = VLAPolicy(MockVLA(), num_candidates=4)

        obs = torch.randn(1, 10)
        instruction = "pick up the cube"

        world_model = MockWorldModel()
        verifier = WorldModelVerifier(world_model)

        for _ in range(5):
            vla.select_action(obs, instruction, verifier)

        stats = vla.get_statistics()
        assert stats["total_actions"] == 5
        assert 0 <= stats["trusted_rate"] <= 1


class TestContinualLearning:
    def test_learn_new_task(self):
        world_model = MockWorldModel()
        vla = MockVLA()
        verifier = WorldModelVerifier(world_model)

        continual_learner = ContinualLearner(
            world_model_verifier=verifier,
            vla_policy=VLAPolicy(vla),
            config=ContinualLearningConfig(),
        )

        stats = continual_learner.learn_new_task(
            task_name="test_task",
            environment=MockEnvironment(),
            num_episodes=5,
        )

        assert stats["episodes_collected"] == 5
        assert len(stats["trust_scores"]) == 5


class MockEnvironment:
    """Mock environment for testing."""

    def reset(self):
        return torch.randn(1, 10)

    def step(self, action):
        next_obs = torch.randn(1, 10)
        reward = torch.tensor(1.0)
        done = False
        info = {}
        return next_obs, reward, done, info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
