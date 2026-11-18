"""Collusion score: agreement on errors, not agreement on the truth."""

from __future__ import annotations

import numpy as np

from annotrust.collusion import score_pairs
from annotrust.config import ScanConfig


def test_flags_a_pair_that_agrees_on_the_wrong_label() -> None:
    n_items = 30
    truth = np.arange(n_items) % 3
    labels = np.full((n_items, 4), -1, dtype=np.int64)
    labels[:, 0] = truth
    labels[:, 1] = truth
    labels[:, 2] = (truth + 1) % 3
    labels[:, 3] = (truth + 1) % 3
    pairs = score_pairs(labels, truth.astype(np.int64), ["h0", "h1", "c0", "c1"], ScanConfig())
    flagged = {(pair.annotator_a, pair.annotator_b) for pair in pairs if pair.flagged}
    assert ("c0", "c1") in flagged
    assert ("h0", "h1") not in flagged


def test_does_not_flag_two_accurate_annotators() -> None:
    n_items = 40
    truth = np.arange(n_items) % 4
    labels = np.column_stack([truth, truth, truth])
    pairs = score_pairs(
        labels.astype(np.int64),
        truth.astype(np.int64),
        ["a", "b", "c"],
        ScanConfig(),
    )
    assert pairs
    assert not any(pair.flagged for pair in pairs)


def test_skips_pairs_without_enough_overlap() -> None:
    labels = np.full((6, 2), -1, dtype=np.int64)
    labels[:3, 0] = 0
    labels[3:, 1] = 1
    consensus = np.zeros(6, dtype=np.int64)
    assert score_pairs(labels, consensus, ["a", "b"], ScanConfig(min_pair_overlap=4)) == []
