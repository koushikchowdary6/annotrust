"""Targeted flips are concentrated. A constant spammer is not one."""

from __future__ import annotations

import numpy as np

from annotrust.config import ScanConfig
from annotrust.poison import find_targeted_flips

CLASSES = ["a", "b", "c", "d"]


def test_flags_a_concentrated_flip() -> None:
    matrix = np.eye(4)
    matrix[0] = np.array([0.05, 0.9, 0.03, 0.02])
    counts = np.array([40.0, 40.0, 40.0, 40.0])
    flips = find_targeted_flips(matrix, counts, CLASSES, ScanConfig())
    assert len(flips) == 1
    assert flips[0].source == "a"
    assert flips[0].target == "b"
    assert flips[0].rate > 0.8


def test_does_not_flag_a_constant_spammer_as_targeted() -> None:
    matrix = np.zeros((4, 4))
    matrix[:, 0] = 1.0
    counts = np.full(4, 30.0)
    assert find_targeted_flips(matrix, counts, CLASSES, ScanConfig()) == []


def test_ignores_a_weak_off_diagonal() -> None:
    matrix = np.full((4, 4), 0.02)
    np.fill_diagonal(matrix, 0.94)
    counts = np.full(4, 50.0)
    assert find_targeted_flips(matrix, counts, CLASSES, ScanConfig()) == []


def test_ignores_a_small_source_count() -> None:
    matrix = np.eye(4)
    matrix[0] = np.array([0.0, 1.0, 0.0, 0.0])
    counts = np.array([3.0, 40.0, 40.0, 40.0])
    assert find_targeted_flips(matrix, counts, CLASSES, ScanConfig()) == []
