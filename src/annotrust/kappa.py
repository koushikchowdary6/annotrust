"""Cohen's kappa for a pair of categorical ratings.

Cohen, J. (1960). A coefficient of agreement for nominal scales.
Educational and Psychological Measurement, 20(1), 37-46.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cohen_kappa(left: NDArray[np.int64], right: NDArray[np.int64], n_classes: int) -> float:
    """Return Cohen's kappa on paired integer labels.

    ``left`` and ``right`` use class indices in ``[0, n_classes)``. When the
    chance term is 1, the classical ratio is undefined: perfect agreement
    returns 1 and any other agreement returns 0.
    """
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if left.shape != right.shape:
        raise ValueError("kappa inputs must be the same shape")
    if left.size == 0:
        raise ValueError("kappa requires at least one pair")
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= n_classes) or np.any(right >= n_classes):
        raise ValueError("labels are outside the class range")

    index = left.astype(np.int64) * n_classes + right.astype(np.int64)
    table = np.bincount(index, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    total = float(table.sum())
    observed = float(np.trace(table)) / total
    left_margin = table.sum(axis=1) / total
    right_margin = table.sum(axis=0) / total
    chance = float(np.dot(left_margin, right_margin))
    if abs(1.0 - chance) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else 0.0
    return float((observed - chance) / (1.0 - chance))
