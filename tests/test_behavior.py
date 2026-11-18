"""Speed and streak summaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from annotrust.behavior import annotator_behavior


def test_fast_gaps_and_constant_streak() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(10):
        rows.append(
            {
                "item_id": f"i{index}",
                "annotator_id": "fast",
                "label": "a",
                "timestamp": start + timedelta(seconds=0.2 * index),
                "_row": index,
            }
        )
    frame = pd.DataFrame(rows)
    summary = annotator_behavior(
        frame,
        ["a", "b"],
        timing_enabled=True,
        fast_seconds=2.0,
        min_timed_gaps=4,
    )["fast"]
    assert summary.fast_fraction == 1.0
    assert summary.longest_streak == 10
    assert summary.normalized_entropy == 0.0
    assert summary.median_gap_seconds is not None
    assert summary.median_gap_seconds < 1.0


def test_timing_disabled_leaves_speed_empty() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "annotator_id": ["a", "a"],
            "label": ["x", "y"],
            "_row": [0, 1],
        }
    )
    summary = annotator_behavior(
        frame,
        ["x", "y"],
        timing_enabled=False,
        fast_seconds=2.0,
        min_timed_gaps=1,
    )["a"]
    assert summary.fast_fraction is None
    assert summary.longest_streak == 1
