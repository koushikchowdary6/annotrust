"""Thresholds for a scan.

The numbers are heuristics. They are named here so a report can record the
exact values that produced it. See docs/engineering-decisions.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from annotrust.errors import AnnotrustError


@dataclass(frozen=True)
class ScanConfig:
    """Knobs for Dawid-Skene and the detectors.

    ``use_cleanlab`` is off unless the caller asks. Cleanlab is an optional
    extra and is imported only then.
    """

    max_iter: int = 100
    tol: float = 1e-5
    smoothing: float = 0.05
    min_labels: int = 12
    min_pair_overlap: int = 10
    collusion_min_kappa: float = 0.75
    collusion_min_excess: float = 0.15
    collusion_min_joint_wrong: int = 3
    collusion_min_score: float = 0.25
    collusion_high_score: float = 0.45
    collusion_high_kappa: float = 0.85
    collusion_excess_weight: float = 0.7
    collusion_wrong_weight: float = 0.3
    poison_min_rate: float = 0.40
    poison_min_concentration: float = 0.60
    poison_min_other_accuracy: float = 0.70
    poison_min_source_count: float = 12.0
    poison_min_flip_count: float = 6.0
    poison_high_rate: float = 0.70
    careless_max_accuracy: float = 0.55
    careless_high_accuracy: float = 0.40
    streak_min_length: int = 12
    streak_min_fraction: float = 0.50
    streak_max_entropy: float = 0.35
    streak_max_accuracy: float = 0.85
    fast_seconds: float = 2.0
    fast_fraction: float = 0.50
    min_timed_gaps: int = 8
    item_entropy: float = 0.65
    item_overturn_confidence: float = 0.60
    model_disagree_confidence: float = 0.85
    use_cleanlab: bool = False

    def validate(self) -> None:
        """Reject settings that would crash the fit or invert a comparison."""
        if self.max_iter < 1:
            raise AnnotrustError("max_iter must be at least 1")
        if self.smoothing < 0:
            raise AnnotrustError("smoothing must be non-negative")
        if self.min_labels < 1:
            raise AnnotrustError("min_labels must be at least 1")
        if self.min_pair_overlap < 1:
            raise AnnotrustError("min_pair_overlap must be at least 1")
        if self.tol < 0:
            raise AnnotrustError("tol must be non-negative")
