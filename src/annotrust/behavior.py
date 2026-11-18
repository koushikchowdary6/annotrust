"""Per-annotator speed and label-pattern summaries.

The speed rule is a threshold on inter-arrival time. It is in the spirit of
crowdsourcing spam studies that separate workers who submit faster than a
person can read the item (Vuurens, de Vries, and Eickhoff, 2011), not a
reproduction of their classifier. A streak is the longest run of one label
in time order, or in file order when timestamps are missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import entropy


@dataclass(frozen=True)
class Behavior:
    """Pattern statistics for one annotator."""

    normalized_entropy: float
    longest_streak: int
    fast_fraction: float | None
    median_gap_seconds: float | None
    n_gaps: int


def annotator_behavior(
    frame: pd.DataFrame,
    classes: list[str],
    *,
    timing_enabled: bool,
    fast_seconds: float,
    min_timed_gaps: int,
) -> dict[str, Behavior]:
    """Summarize every annotator present in ``frame``."""
    class_index = {label: index for index, label in enumerate(classes)}
    n_classes = len(classes)
    normalizer = math.log2(n_classes) if n_classes > 1 else 1.0
    summaries: dict[str, Behavior] = {}
    for annotator_id, group in frame.groupby("annotator_id", sort=False):
        labels = [str(label) for label in group["label"].tolist()]
        counts = np.zeros(n_classes, dtype=np.float64)
        for label in labels:
            counts[class_index[label]] += 1.0
        probabilities = counts / counts.sum()
        bits = float(entropy(probabilities, base=2))
        normalized = bits / normalizer if normalizer else 0.0
        if timing_enabled and "timestamp" in group.columns:
            ordered = group.sort_values("timestamp", kind="mergesort")
            ordered_labels = [str(label) for label in ordered["label"].tolist()]
            gaps = _gaps(ordered["timestamp"].tolist())
        else:
            ordered = group.sort_values("_row", kind="mergesort")
            ordered_labels = [str(label) for label in ordered["label"].tolist()]
            gaps = []
        fast_fraction: float | None = None
        median: float | None = None
        if timing_enabled and len(gaps) >= min_timed_gaps:
            gap_array = np.asarray(gaps, dtype=np.float64)
            median = float(np.median(gap_array))
            fast_fraction = float(np.mean(gap_array < fast_seconds))
        elif timing_enabled and gaps:
            median = float(np.median(np.asarray(gaps, dtype=np.float64)))
        summaries[str(annotator_id)] = Behavior(
            normalized_entropy=normalized,
            longest_streak=_longest_streak(ordered_labels),
            fast_fraction=fast_fraction,
            median_gap_seconds=median,
            n_gaps=len(gaps),
        )
    return summaries


def _longest_streak(labels: list[str]) -> int:
    if not labels:
        return 0
    best = 1
    run = 1
    for previous, current in zip(labels, labels[1:], strict=False):
        if current == previous:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _gaps(timestamps: list[object]) -> list[float]:
    gaps: list[float] = []
    for left, right in zip(timestamps, timestamps[1:], strict=False):
        if not isinstance(left, datetime) or not isinstance(right, datetime):
            continue
        seconds = (right - left).total_seconds()
        if seconds < 0:
            continue
        gaps.append(float(seconds))
    return gaps
