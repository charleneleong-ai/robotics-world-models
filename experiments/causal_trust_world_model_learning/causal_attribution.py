"""Causal Attribution Engine for world model failures.

This module diagnoses the mechanism of world model failures, enabling targeted 
recovery strategies.

Usage:
    # Initialize causal attribution engine
    engine = CausalAttributionEngine()
    
    # Diagnose failure mechanism
    attribution = engine.diagnose(
        observation, action, predicted_next, actual_next
    )
    
    # Get recovery recommendation
    recommendation = engine.get_recommendation(attribution)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class FailureMechanism(Enum):
    """Types of failure mechanisms."""
    CONTACT = "contact"
    VISUAL = "visual"
    DYNAMIC = "dynamic"
    PHYSICS = "physics"
    OOD = "ood"
    UNKNOWN = "unknown"


@dataclass
class CausalAttribution:
    """Result of causal attribution."""
    mechanism: FailureMechanism
    confidence: float
    features: dict[str, float]
    recovery_recommendation: str
    severity: float


class ContactFailureDetector:
    """Detects contact-related failures."""
    
    def __init__(self, contact_threshold: float = 0.1):
        self.contact_threshold = contact_threshold
    
    def detect(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> float:
        """Detect contact failure.
        
        Returns:
            Contact failure score (0-1, higher = more likely contact failure)
        """
        # Check for contact inconsistencies
        # In practice, this would analyze:
        # - Contact forces
        # - Interpenetration
        # - Friction violations
        
        # Simplified: check if prediction error is localized
        error = (predicted_next - actual_next).abs()
        localization = self._compute_localization(error)
        
        return localization
    
    def _compute_localization(self, error: torch.Tensor) -> float:
        """Compute error localization (contact errors are typically localized)."""
        # Simple localization measure
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        if mean_error == 0:
            return 0.0
        
        localization = max_error / mean_error
        return min(1.0, localization / 10.0)


class VisualFailureDetector:
    """Detects visual-related failures."""
    
    def __init__(self, visual_threshold: float = 0.1):
        self.visual_threshold = visual_threshold
    
    def detect(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> float:
        """Detect visual failure.
        
        Returns:
            Visual failure score (0-1, higher = more likely visual failure)
        """
        # Check for visual inconsistencies
        # In practice, this would analyze:
        # - Visual features
        # - Object appearance
        # - Lighting changes
        
        # Simplified: check if error affects visual features
        visual_features = self._extract_visual_features(observation)
        visual_error = self._compute_visual_error(visual_features, predicted_next, actual_next)
        
        return visual_error
    
    def _extract_visual_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract visual features from observation."""
        # Simplified: use first half of observation as visual features
        dim = observation.shape[-1] // 2
        return observation[..., :dim]
    
    def _compute_visual_error(
        self,
        visual_features: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> float:
        """Compute visual error."""
        # Simplified visual error
        error = (predicted_next - actual_next).abs().mean().item()
        return min(1.0, error / 5.0)


class DynamicFailureDetector:
    """Detects dynamics-related failures."""
    
    def __init__(self, dynamic_threshold: float = 0.1):
        self.dynamic_threshold = dynamic_threshold
    
    def detect(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> float:
        """Detect dynamics failure.
        
        Returns:
            Dynamics failure score (0-1, higher = more likely dynamics failure)
        """
        # Check for dynamics inconsistencies
        # In practice, this would analyze:
        # - Velocity consistency
        # - Acceleration patterns
        # - Force balance
        
        # Simplified: check if error is consistent over time
        dynamics_score = self._compute_dynamics_consistency(
            observation, predicted_next, actual_next
        )
        
        return dynamics_score
    
    def _compute_dynamics_consistency(
        self,
        observation: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> float:
        """Compute dynamics consistency."""
        # Simplified dynamics check
        predicted_velocity = predicted_next - observation
        actual_velocity = actual_next - observation
        
        velocity_error = (predicted_velocity - actual_velocity).abs().mean().item()
        return min(1.0, velocity_error / 5.0)


class CausalAttributionEngine:
    """Causal attribution engine for world model failures.
    
    This module diagnoses the mechanism of world model failures by analyzing:
    1. Contact-related failures
    2. Visual-related failures
    3. Dynamics-related failures
    """
    
    def __init__(
        self,
        contact_weight: float = 0.33,
        visual_weight: float = 0.33,
        dynamic_weight: float = 0.34,
    ):
        """Initialize causal attribution engine.
        
        Args:
            contact_weight: Weight for contact failure detection
            visual_weight: Weight for visual failure detection
            dynamic_weight: Weight for dynamics failure detection
        """
        self.contact_weight = contact_weight
        self.visual_weight = visual_weight
        self.dynamic_weight = dynamic_weight
        
        # Failure detectors
        self.contact_detector = ContactFailureDetector()
        self.visual_detector = VisualFailureDetector()
        self.dynamic_detector = DynamicFailureDetector()
    
    def diagnose(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> CausalAttribution:
        """Diagnose failure mechanism.
        
        Args:
            observation: Current state
            action: Action taken
            predicted_next: Predicted next state
            actual_next: Actual next state
            
        Returns:
            CausalAttribution with mechanism and confidence
        """
        # Detect failure mechanisms
        contact_score = self.contact_detector.detect(
            observation, action, predicted_next, actual_next
        )
        visual_score = self.visual_detector.detect(
            observation, action, predicted_next, actual_next
        )
        dynamic_score = self.dynamic_detector.detect(
            observation, action, predicted_next, actual_next
        )
        
        # Determine primary mechanism
        scores = {
            FailureMechanism.CONTACT: contact_score * self.contact_weight,
            FailureMechanism.VISUAL: visual_score * self.visual_weight,
            FailureMechanism.DYNAMIC: dynamic_score * self.dynamic_weight,
        }
        
        primary_mechanism = max(scores, key=scores.get)
        confidence = scores[primary_mechanism]
        
        # Compute severity
        severity = self._compute_severity(
            observation, action, predicted_next, actual_next
        )
        
        # Get recovery recommendation
        recovery_recommendation = self._get_recovery_recommendation(
            primary_mechanism, severity
        )
        
        return CausalAttribution(
            mechanism=primary_mechanism,
            confidence=confidence,
            features={
                "contact_score": contact_score,
                "visual_score": visual_score,
                "dynamic_score": dynamic_score,
            },
            recovery_recommendation=recovery_recommendation,
            severity=severity,
        )
    
    def _compute_severity(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> float:
        """Compute failure severity."""
        error = (predicted_next - actual_next).abs().mean().item()
        return min(1.0, error / 10.0)
    
    def _get_recovery_recommendation(
        self,
        mechanism: FailureMechanism,
        severity: float,
    ) -> str:
        """Get recovery recommendation based on mechanism and severity."""
        recommendations = {
            FailureMechanism.CONTACT: {
                "low": "Adjust contact parameters slightly",
                "medium": "Retry with modified forces",
                "high": "Reset and re-plan with contact-aware controller",
            },
            FailureMechanism.VISUAL: {
                "low": "Re-sample visual observations",
                "medium": "Reset to known visual state",
                "high": "Collect more visual data for this scenario",
            },
            FailureMechanism.DYNAMIC: {
                "low": "Re-plan trajectory with current model",
                "medium": "Collect real-world data to update dynamics",
                "high": "Switch to conservative controller",
            },
        }
        
        # Determine severity level
        if severity < 0.3:
            severity_level = "low"
        elif severity < 0.7:
            severity_level = "medium"
        else:
            severity_level = "high"
        
        return recommendations.get(mechanism, {}).get(
            severity_level, "Unknown mechanism"
        )
