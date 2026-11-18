"""Precision and recall for a set of flagged ids."""

from __future__ import annotations


def classification_counts(truth: set[str], predicted: set[str]) -> dict[str, float | int | None]:
    """Score one attack type.

    Precision is ``None`` when nothing was flagged. Recall is ``None`` when
    nothing was planted. Callers should print those as ``n/a`` rather than 0,
    because 0 would claim a measured rate that was not observed.
    """
    true_positive = len(truth & predicted)
    false_positive = len(predicted - truth)
    false_negative = len(truth - predicted)
    precision: float | None = None
    recall: float | None = None
    if true_positive + false_positive:
        precision = true_positive / (true_positive + false_positive)
    if true_positive + false_negative:
        recall = true_positive / (true_positive + false_negative)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
    }
