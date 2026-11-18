"""End-to-end scans, including the cases the benchmark says we miss."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from annotrust.config import ScanConfig
from annotrust.report import render_html, render_json, render_text
from annotrust.scan import scan
from annotrust.simulate import simulate, write_simulation


def _primaries(result: object) -> dict[str, str]:
    return {annotator.annotator_id: annotator.primary for annotator in result.annotators}  # type: ignore[attr-defined]


def test_reference_seed_recovers_every_planted_role() -> None:
    simulation = simulate(seed=0)
    result = scan(simulation.frame, source="memory")
    primaries = _primaries(result)
    for annotator in simulation.annotators:
        if annotator.role == "honest":
            assert primaries[annotator.annotator_id] == "honest"
        else:
            assert primaries[annotator.annotator_id] == annotator.role
    careless = [row for row in result.annotators if row.annotator_id.startswith("careless")]
    assert all(any(flag.kind == "fast" for flag in row.flags) for row in careless)
    assert result.fails("high")
    assert not result.fails("low") or result.fails("high")
    assert any(pair.flagged for pair in result.pairs)
    text = render_text(result)
    assert "poisoner_00" in text
    assert "flip class_0->class_1" in text
    assert "Exit status" in text
    html = render_html(result)
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()
    payload = render_json(result)
    assert '"method": "dawid-skene"' in payload


def test_csv_roundtrip_keeps_primaries(tmp_path: Path) -> None:
    simulation = simulate(seed=0, timestamps=True)
    path = tmp_path / "ann.csv"
    write_simulation(simulation, path)
    from_file = scan(path)
    from_frame = scan(simulation.frame)
    assert _primaries(from_file) == _primaries(from_frame)


def test_weak_flip_is_not_called_poisoning() -> None:
    simulation = simulate(seed=0, poison_rate=0.15, timestamps=False)
    result = scan(simulation.frame)
    poisoners = {row.annotator_id for row in result.annotators if row.primary == "poisoning"}
    assert "poisoner_00" not in poisoners


def test_copying_an_accurate_leader_is_not_called_collusion() -> None:
    simulation = simulate(seed=0, leader_accuracy=0.95, collude_rate=1.0, timestamps=False)
    result = scan(simulation.frame)
    colluding = {row.annotator_id for row in result.annotators if row.primary == "colluding"}
    assert "colluder_00" not in colluding
    assert "colluder_01" not in colluding


def test_majority_flip_can_accuse_the_honest_annotator() -> None:
    """Seed 0 of the panel used in docs/benchmark.md. The flip is absorbed."""
    simulation = simulate(
        seed=0,
        timestamps=False,
        n_items=200,
        labels_per_item=6,
        n_honest=2,
        n_careless=0,
        n_colluders=0,
        n_poisoners=5,
        poison_rate=1.0,
    )
    result = scan(simulation.frame)
    primaries = _primaries(result)
    assert primaries["poisoner_00"] != "poisoning"
    assert "honest_00" in {row.annotator_id for row in result.annotators if row.primary == "poisoning"}


def test_small_panel_flags_a_flip_and_a_fast_spammer() -> None:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    classes = ["a", "b", "c", "d"]
    for item in range(40):
        truth = classes[item % 4]
        for annotator in ("h0", "h1", "h2"):
            rows.append(
                {
                    "item_id": f"i{item}",
                    "annotator_id": annotator,
                    "label": truth,
                    "timestamp": start + pd.Timedelta(seconds=item * 30 + 10),
                }
            )
        rows.append(
            {
                "item_id": f"i{item}",
                "annotator_id": "flipper",
                "label": "b" if truth == "a" else truth,
                "timestamp": start + pd.Timedelta(seconds=item * 30 + 12),
            }
        )
        rows.append(
            {
                "item_id": f"i{item}",
                "annotator_id": "spam",
                "label": "a",
                "timestamp": start + pd.Timedelta(seconds=0.2 * item),
            }
        )
    frame = pd.DataFrame(rows)
    frame["model_prob_a"] = 0.1
    frame["model_prob_b"] = 0.1
    frame["model_prob_c"] = 0.1
    frame["model_prob_d"] = 0.7
    result = scan(frame, ScanConfig(poison_min_source_count=8, poison_min_flip_count=4, min_labels=10))
    primaries = _primaries(result)
    assert primaries["flipper"] == "poisoning"
    assert primaries["spam"] == "careless"
    spam = next(row for row in result.annotators if row.annotator_id == "spam")
    assert any(flag.kind == "fast" for flag in spam.flags)
    assert any("constant-label streak" in flag.summary for flag in spam.flags)


def test_scan_records_mocked_cleanlab_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import numpy as np

    rank = types.ModuleType("cleanlab.rank")

    def fake_scores(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return np.full(len(labels), 0.25)

    rank.get_label_quality_scores = fake_scores  # type: ignore[attr-defined]
    package = types.ModuleType("cleanlab")
    monkeypatch.setitem(sys.modules, "cleanlab", package)
    monkeypatch.setitem(sys.modules, "cleanlab.rank", rank)
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["yes", "yes", "no", "no"],
            "model_prob_no": [0.2, 0.2, 0.8, 0.8],
            "model_prob_yes": [0.8, 0.8, 0.2, 0.2],
        }
    )
    result = scan(frame, ScanConfig(min_labels=1, use_cleanlab=True))
    assert [item.cleanlab_score for item in result.items] == [0.25, 0.25]


def test_cleanlab_requested_without_probabilities_warns() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["yes", "yes", "no", "no"],
        }
    )
    result = scan(frame, ScanConfig(min_labels=1, use_cleanlab=True))
    assert any("cleanlab comparison was requested" in warning for warning in result.warnings)


def test_single_annotator_warning_and_html_escape() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "annotator_id": ["<script>alert(1)</script>"] * 4,
            "label": ["yes", "no", "yes", "no"],
        }
    )
    result = scan(frame, ScanConfig(min_labels=1))
    assert any("Only one annotator" in warning for warning in result.warnings)
    html = render_html(result)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_no_overlapping_annotators_warns() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "annotator_id": ["a", "b"],
            "label": ["yes", "no"],
        }
    )
    result = scan(frame, ScanConfig(min_labels=1))
    assert any("more than one annotator" in warning for warning in result.warnings)


def test_identical_timestamps_skip_speed() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["yes", "no", "yes", "no"],
            "timestamp": ["2026-01-01T00:00:00Z"] * 4,
        }
    )
    result = scan(frame, ScanConfig(min_labels=1))
    assert any("do not vary" in warning for warning in result.warnings)


def test_model_disagreement_flags_an_item() -> None:
    rows = []
    for item, label in enumerate(["yes", "yes", "no", "no"]):
        for annotator in ("a", "b", "c"):
            rows.append(
                {
                    "item_id": f"i{item}",
                    "annotator_id": annotator,
                    "label": label,
                    "model_prob_yes": 0.95 if label == "no" else 0.05,
                    "model_prob_no": 0.05 if label == "no" else 0.95,
                }
            )
    result = scan(pd.DataFrame(rows), ScanConfig(min_labels=1))
    assert any("model_disagrees" in item.reasons for item in result.items)


def test_fail_levels_ignore_items() -> None:
    simulation = simulate(
        seed=0,
        n_items=30,
        n_honest=4,
        n_careless=0,
        n_colluders=0,
        n_poisoners=0,
        labels_per_item=3,
        timestamps=False,
    )
    result = scan(simulation.frame)
    assert result.fails("high") is False
    assert all(row.primary == "honest" for row in result.annotators)
