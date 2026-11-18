"""Cohen's kappa, including the undefined chance-agreement case."""

from __future__ import annotations

import numpy as np
import pytest

from annotrust.kappa import cohen_kappa


def test_known_five_pair_example() -> None:
    left = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    right = np.array([0, 1, 1, 1, 0], dtype=np.int64)
    # p_o = 0.8, p_e = 0.48, kappa = 0.32 / 0.52
    assert cohen_kappa(left, right, 2) == pytest.approx(0.32 / 0.52)


def test_perfect_agreement_is_one() -> None:
    labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    assert cohen_kappa(labels, labels.copy(), 3) == pytest.approx(1.0)


def test_complete_disagreement_on_a_balanced_pair_is_minus_one() -> None:
    left = np.array([0, 0, 1, 1], dtype=np.int64)
    right = np.array([1, 1, 0, 0], dtype=np.int64)
    assert cohen_kappa(left, right, 2) == pytest.approx(-1.0)


def test_undefined_chance_term_with_perfect_agreement() -> None:
    left = np.array([0, 0, 0], dtype=np.int64)
    assert cohen_kappa(left, left.copy(), 2) == pytest.approx(1.0)


def test_undefined_chance_term_without_agreement() -> None:
    # Both raters use only one class, but not the same class: p_e is 0, not 1.
    # A single class on one side and the other class on the other side:
    # margins are (1, 0) and (0, 1), so p_e is 0 and kappa is (0 - 0) / 1 = 0.
    left = np.array([0, 0], dtype=np.int64)
    right = np.array([1, 1], dtype=np.int64)
    assert cohen_kappa(left, right, 2) == pytest.approx(0.0)


def test_rejects_bad_input() -> None:
    left = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="same shape"):
        cohen_kappa(left, np.array([0], dtype=np.int64), 2)
    with pytest.raises(ValueError, match="at least one"):
        cohen_kappa(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 2)
    with pytest.raises(ValueError, match="outside"):
        cohen_kappa(np.array([0, 3], dtype=np.int64), np.array([0, 1], dtype=np.int64), 2)
    with pytest.raises(ValueError, match="positive"):
        cohen_kappa(left, left, 0)
