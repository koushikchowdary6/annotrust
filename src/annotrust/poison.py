"""Targeted label-flip detection from a Dawid-Skene confusion matrix.

A targeted flip is one source class landing on one other class, while the
annotator stays accurate on the remaining classes. A spammer who emits one
label for everything produces several peaked rows, but the other diagonals
are near zero, so they are not called a targeted flip.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from annotrust.config import ScanConfig
from annotrust.results import Flip


def find_targeted_flips(
    confusion_row: NDArray[np.float64],
    source_counts: NDArray[np.float64],
    classes: list[str],
    config: ScanConfig,
) -> list[Flip]:
    """Return flips in ``confusion_row`` that clear the config thresholds.

    ``confusion_row[source, predicted]`` is one annotator's matrix. ``source_counts``
    is the posterior mass of each true class on the items that annotator labeled.
    """
    n_classes = confusion_row.shape[0]
    if confusion_row.shape != (n_classes, n_classes):
        raise ValueError("confusion matrix must be square")
    if source_counts.shape != (n_classes,):
        raise ValueError("source_counts length must match the matrix")

    diagonals = np.diag(confusion_row)
    flips: list[Flip] = []
    for source in range(n_classes):
        source_count = float(source_counts[source])
        if source_count < config.poison_min_source_count:
            continue
        off = confusion_row[source].copy()
        off[source] = 0.0
        off_mass = float(off.sum())
        if off_mass <= 1e-12:
            continue
        target = int(np.argmax(off))
        rate = float(confusion_row[source, target])
        if rate * source_count < config.poison_min_flip_count:
            continue
        if rate < config.poison_min_rate:
            continue
        concentration = rate / off_mass
        if concentration < config.poison_min_concentration:
            continue
        weights = source_counts.copy()
        weights[source] = 0.0
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-12:
            continue
        other_accuracy = float(np.dot(weights, diagonals) / weight_sum)
        if other_accuracy < config.poison_min_other_accuracy:
            continue
        flips.append(
            Flip(
                source=classes[source],
                target=classes[target],
                rate=rate,
                concentration=concentration,
                other_accuracy=other_accuracy,
                source_count=source_count,
            )
        )
    flips.sort(key=lambda flip: flip.rate, reverse=True)
    return flips
