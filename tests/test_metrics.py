"""Precision and recall stay undefined instead of pretending to be zero."""

from __future__ import annotations

from annotrust.metrics import classification_counts


def test_perfect_match() -> None:
    counts = classification_counts({"a", "b"}, {"a", "b"})
    assert counts == {"tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0}


def test_no_predictions_has_no_precision() -> None:
    counts = classification_counts({"a"}, set())
    assert counts["precision"] is None
    assert counts["recall"] == 0.0
    assert counts["fn"] == 1


def test_no_planted_cases_has_no_recall() -> None:
    counts = classification_counts(set(), set())
    assert counts["precision"] is None
    assert counts["recall"] is None
