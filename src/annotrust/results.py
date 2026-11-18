"""Scan results and the JSON shape."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from annotrust.config import ScanConfig
from annotrust.errors import AnnotrustError

SEVERITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3}


def highest_severity(levels: list[str | None]) -> str | None:
    """Return the worse of the given severities."""
    present = [level for level in levels if level is not None]
    if not present:
        return None
    return max(present, key=lambda level: SEVERITY_RANK[level])


def _num(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return round(number, digits)


@dataclass(frozen=True)
class Flag:
    """One reason an annotator was marked."""

    kind: str
    severity: str
    summary: str


@dataclass(frozen=True)
class Flip:
    """One off-diagonal concentration in a confusion matrix.

    ``rate`` is P(emit ``target`` | true class ``source``). ``concentration``
    is that rate divided by the row's off-diagonal mass.
    """

    source: str
    target: str
    rate: float
    concentration: float
    other_accuracy: float
    source_count: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "source": self.source,
            "target": self.target,
            "rate": _num(self.rate) or 0.0,
            "concentration": _num(self.concentration) or 0.0,
            "other_accuracy": _num(self.other_accuracy) or 0.0,
            "source_count": _num(self.source_count) or 0.0,
        }


@dataclass(frozen=True)
class AnnotatorReport:
    """Per-annotator audit row."""

    annotator_id: str
    n_labels: int
    accuracy: float | None
    ds_accuracy: float | None
    normalized_entropy: float | None
    longest_streak: int
    fast_fraction: float | None
    median_gap_seconds: float | None
    confusion: tuple[tuple[float, ...], ...]
    flips: tuple[Flip, ...]
    flags: tuple[Flag, ...]
    primary: str
    severity: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "annotator_id": self.annotator_id,
            "n_labels": self.n_labels,
            "accuracy": _num(self.accuracy),
            "ds_accuracy": _num(self.ds_accuracy),
            "normalized_entropy": _num(self.normalized_entropy),
            "longest_streak": self.longest_streak,
            "fast_fraction": _num(self.fast_fraction),
            "median_gap_seconds": _num(self.median_gap_seconds),
            "confusion": [[_num(value) or 0.0 for value in row] for row in self.confusion],
            "flips": [flip.to_dict() for flip in self.flips],
            "flags": [asdict(flag) for flag in self.flags],
            "primary": self.primary,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class PairReport:
    """Pairwise kappa and the collusion score on shared items."""

    annotator_a: str
    annotator_b: str
    overlap: int
    kappa: float
    kappa_a_consensus: float
    kappa_b_consensus: float
    excess: float
    joint_wrong: int
    both_differ: int
    wrong_agreement: float
    score: float
    flagged: bool
    severity: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "annotator_a": self.annotator_a,
            "annotator_b": self.annotator_b,
            "overlap": self.overlap,
            "kappa": _num(self.kappa),
            "kappa_a_consensus": _num(self.kappa_a_consensus),
            "kappa_b_consensus": _num(self.kappa_b_consensus),
            "excess": _num(self.excess),
            "joint_wrong": self.joint_wrong,
            "both_differ": self.both_differ,
            "wrong_agreement": _num(self.wrong_agreement),
            "score": _num(self.score),
            "flagged": self.flagged,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ItemReport:
    """Label-quality scores for one item.

    ``suspect`` means the item was flagged. A low ``confidence`` by itself is
    not a claim that the item is mislabeled.
    """

    item_id: str
    estimated_label: str
    confidence: float
    normalized_entropy: float
    majority_label: str
    majority_fraction: float
    n_annotators: int
    suspect: bool
    severity: str | None
    reasons: tuple[str, ...]
    model_label: str | None
    model_confidence: float | None
    cleanlab_score: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "estimated_label": self.estimated_label,
            "confidence": _num(self.confidence),
            "normalized_entropy": _num(self.normalized_entropy),
            "majority_label": self.majority_label,
            "majority_fraction": _num(self.majority_fraction),
            "n_annotators": self.n_annotators,
            "suspect": self.suspect,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "model_label": self.model_label,
            "model_confidence": _num(self.model_confidence),
            "cleanlab_score": _num(self.cleanlab_score),
        }


@dataclass(frozen=True)
class ScanResult:
    """Full audit of one annotation table."""

    version: str
    source: str | None
    n_annotations: int
    n_items: int
    n_annotators: int
    classes: tuple[str, ...]
    warnings: tuple[str, ...]
    iterations: int
    log_likelihood: float
    prior: tuple[float, ...]
    annotators: tuple[AnnotatorReport, ...]
    pairs: tuple[PairReport, ...]
    items: tuple[ItemReport, ...]
    config: ScanConfig

    def severity_counts(self) -> dict[str, int]:
        """Count annotators by their worst flag. Items are not included."""
        counts = {"high": 0, "medium": 0, "low": 0}
        for annotator in self.annotators:
            if annotator.severity is not None:
                counts[annotator.severity] += 1
        return counts

    def fails(self, level: str) -> bool:
        """True when an annotator's severity is at or above ``level``.

        Flagged items do not affect this. ``--fail-on`` is a CI gate for
        suspicious annotators, and ordinary item disagreement would trip it
        on almost every file.
        """
        if level not in SEVERITY_RANK:
            raise AnnotrustError("fail-on must be low, medium, or high")
        threshold = SEVERITY_RANK[level]
        return any(
            annotator.severity is not None and SEVERITY_RANK[annotator.severity] >= threshold
            for annotator in self.annotators
        )

    def to_dict(self) -> dict[str, object]:
        counts = self.severity_counts()
        suspect_items = sum(1 for item in self.items if item.suspect)
        flagged_pairs = sum(1 for pair in self.pairs if pair.flagged)
        return {
            "annotrust": self.version,
            "source": self.source,
            "summary": {
                "annotations": self.n_annotations,
                "items": self.n_items,
                "annotators": self.n_annotators,
                "classes": list(self.classes),
                "annotator_flags": counts,
                "flagged_pairs": flagged_pairs,
                "pairs_scored": len(self.pairs),
                "suspect_items": suspect_items,
                "dawid_skene_iterations": self.iterations,
                "log_likelihood": _num(self.log_likelihood, 4),
            },
            "warnings": list(self.warnings),
            "consensus": {
                "method": "dawid-skene",
                "iterations": self.iterations,
                "log_likelihood": _num(self.log_likelihood, 4),
                "prior": [
                    {"label": label, "probability": _num(probability)}
                    for label, probability in zip(self.classes, self.prior, strict=True)
                ],
            },
            "annotators": [annotator.to_dict() for annotator in self.annotators],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "items": [item.to_dict() for item in self.items],
            "config": asdict(self.config),
            "exit_code_note": (
                "--fail-on compares annotator severity only. Flagged items do not change the exit code."
            ),
        }
