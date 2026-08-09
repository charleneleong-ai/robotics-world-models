"""Multi-signal trust scoring for world model predictions.

This module combines multiple signals into a unified trust score that quantifies 
when to trust world model predictions.

Usage:
    # Initialize trust scorer
    scorer = TrustScorer()
    
    # Compute trust score
    trust_score = scorer.compute_trust_score(
        physics_consistency=0.8,
        ood_score=0.9,
        calibration_error=0.05
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrustSignals:
    """Container for trust signals."""
    physics_consistency: float
    ood_score: float
    calibration_error: float
    prediction_confidence: float
    historical_accuracy: float


@dataclass
class TrustScore:
    """Container for trust score and breakdown."""
    overall: float
    physics_component: float
    ood_component: float
    calibration_component: float
    confidence_component: float
    historical_component: float
    is_trustworthy: bool


class TrustScorer:
    """Multi-signal trust scorer for world model predictions.
    
    This module combines multiple signals into a unified trust score:
    1. Physics consistency (energy, momentum, contact)
    2. Out-of-distribution detection
    3. Calibration error
    4. Prediction confidence
    5. Historical accuracy
    """
    
    def __init__(
        self,
        physics_weight: float = 0.3,
        ood_weight: float = 0.25,
        calibration_weight: float = 0.2,
        confidence_weight: float = 0.15,
        historical_weight: float = 0.1,
        trust_threshold: float = 0.7,
    ):
        """Initialize trust scorer.
        
        Args:
            physics_weight: Weight for physics consistency
            ood_weight: Weight for OOD detection
            calibration_weight: Weight for calibration
            confidence_weight: Weight for prediction confidence
            historical_weight: Weight for historical accuracy
            trust_threshold: Minimum trust score to consider trustworthy
        """
        self.physics_weight = physics_weight
        self.ood_weight = ood_weight
        self.calibration_weight = calibration_weight
        self.confidence_weight = confidence_weight
        self.historical_weight = historical_weight
        self.trust_threshold = trust_threshold
        
        # Historical tracking
        self.historical_accuracy = 0.5  # Start with neutral
        self.accuracy_history: list[float] = []
    
    def compute_trust_score(
        self,
        physics_consistency: float,
        ood_score: float,
        calibration_error: float,
        prediction_confidence: float = 1.0,
    ) -> TrustScore:
        """Compute trust score from signals.
        
        Args:
            physics_consistency: Physics consistency score (0-1)
            ood_score: OOD score (0-1, higher = more in-distribution)
            calibration_error: Calibration error (lower is better)
            prediction_confidence: Prediction confidence (0-1)
            
        Returns:
            TrustScore with overall score and breakdown
        """
        # Normalize calibration error (lower is better)
        calibration_score = max(0, 1 - calibration_error)
        
        # Compute components
        physics_component = self.physics_weight * physics_consistency
        ood_component = self.ood_weight * ood_score
        calibration_component = self.calibration_weight * calibration_score
        confidence_component = self.confidence_weight * prediction_confidence
        historical_component = self.historical_weight * self.historical_accuracy
        
        # Compute overall score
        overall = (
            physics_component +
            ood_component +
            calibration_component +
            confidence_component +
            historical_component
        )
        
        # Clip to [0, 1]
        overall = min(1.0, max(0.0, overall))
        
        return TrustScore(
            overall=overall,
            physics_component=physics_component,
            ood_component=ood_component,
            calibration_component=calibration_component,
            confidence_component=confidence_component,
            historical_component=historical_component,
            is_trustworthy=overall >= self.trust_threshold,
        )
    
    def update_historical_accuracy(self, was_correct: bool):
        """Update historical accuracy tracking.
        
        Args:
            was_correct: Whether the last prediction was correct
        """
        # Exponential moving average
        alpha = 0.1
        self.historical_accuracy = (
            alpha * float(was_correct) +
            (1 - alpha) * self.historical_accuracy
        )
        self.accuracy_history.append(float(was_correct))
    
    def get_calibration_error(self) -> float:
        """Get current calibration error."""
        if len(self.accuracy_history) < 10:
            return 0.0
        
        # Simple calibration error
        recent_accuracy = sum(self.accuracy_history[-100:]) / len(self.accuracy_history[-100:])
        return abs(recent_accuracy - self.historical_accuracy)
    
    def reset(self):
        """Reset trust scorer state."""
        self.historical_accuracy = 0.5
        self.accuracy_history.clear()
