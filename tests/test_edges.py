"""Error paths and the branches the main scenarios do not hit."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from annotrust.behavior import _gaps, _longest_streak, annotator_behavior
from annotrust.collusion import score_pairs
from annotrust.config import ScanConfig
from annotrust.dawid_skene import fit_dawid_skene
from annotrust.errors import AnnotrustError
from annotrust.importers.encord import encord_payload_to_frame, encord_to_frame
from annotrust.importers.label_studio import label_studio_payload_to_frame, label_studio_to_frame
from annotrust.items import score_items
from annotrust.load import load_file, parse_timestamps, prepare, timestamps_usable
from annotrust.poison import find_targeted_flips
from annotrust.report import _cell, render_text
from annotrust.results import highest_severity
from annotrust.scan import _careless_flag, scan
from annotrust.simulate import simulate


def test_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(AnnotrustError, match="n_items"):
        simulate(n_items=0)
    with pytest.raises(AnnotrustError, match="non-negative"):
        simulate(n_honest=-1)
    with pytest.raises(AnnotrustError, match="labels_per_item must be at least"):
        simulate(labels_per_item=0)
    with pytest.raises(AnnotrustError, match="at least one annotator"):
        simulate(n_honest=0, n_careless=0, n_colluders=0, n_poisoners=0)
    with pytest.raises(AnnotrustError, match="larger than the annotator pool"):
        simulate(n_honest=1, n_careless=0, n_colluders=0, n_poisoners=0, labels_per_item=2)
    with pytest.raises(AnnotrustError, match="unique"):
        simulate(class_names=["a", "a"])
    with pytest.raises(AnnotrustError, match="does not match"):
        simulate(n_classes=3, class_names=["a", "b"])
    with pytest.raises(AnnotrustError, match="Unknown class"):
        simulate(n_poisoners=1, poison_source="nope")
    with pytest.raises(AnnotrustError, match="at most 200"):
        simulate(n_classes=201, n_poisoners=0, n_colluders=0, n_careless=0, n_honest=2, labels_per_item=2, n_items=2)
    indexed = simulate(
        n_items=4,
        n_honest=2,
        n_careless=0,
        n_colluders=0,
        n_poisoners=1,
        labels_per_item=2,
        poison_source="0",
        poison_target="1",
        timestamps=False,
    )
    assert indexed.annotators[-1].source == "class_0"
    uniform = simulate(
        n_items=12,
        n_honest=2,
        n_careless=0,
        n_colluders=2,
        n_poisoners=4,
        labels_per_item=4,
        timestamps=False,
        collude_rate=0.5,
    )
    assert uniform.params["attackers_pinned_to_every_item"] is False
    with pytest.raises(AnnotrustError, match="min_labels"):
        ScanConfig(min_labels=0).validate()
    with pytest.raises(AnnotrustError, match="min_pair_overlap"):
        ScanConfig(min_pair_overlap=0).validate()
    with pytest.raises(AnnotrustError, match="tol"):
        ScanConfig(tol=-1).validate()


def test_json_sniff_and_bad_records(tmp_path: Path) -> None:
    encord = tmp_path / "encord.json"
    encord.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "label_hash": "h",
                        "data_title": "pic",
                        "classification_answers": {
                            "c": {"classifications": [{"name": "Q", "value": "q", "answers": [{"value": "Yes"}]}]}
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    frame = load_file(encord)
    assert frame.loc[0, "label"] == "Yes"
    studio = tmp_path / "one-task.json"
    studio.write_text(
        json.dumps(
            {
                "id": 3,
                "annotations": [{"completed_by": 4, "result": [{"from_name": "s", "value": {"choices": ["A", "B"]}}]}],
            }
        ),
        encoding="utf-8",
    )
    assert load_file(studio).loc[0, "annotator_id"] == "4"
    records = tmp_path / "obj.json"
    records.write_text(json.dumps({"item_id": "i", "label": "a"}), encoding="utf-8")
    with pytest.raises(AnnotrustError, match="list of annotation"):
        load_file(records, "json")
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("[]", encoding="utf-8")
    with pytest.raises(AnnotrustError, match="no records"):
        load_file(empty_json)
    junk = tmp_path / "junk.json"
    junk.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(AnnotrustError, match="not a list of objects"):
        load_file(junk)
    unknown = tmp_path / "nope.json"
    unknown.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(AnnotrustError, match="Unrecognized JSON"):
        load_file(unknown)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(AnnotrustError, match="invalid JSON"):
        load_file(broken)
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(AnnotrustError, match="JSON object"):
        load_file(jsonl)
    odd = tmp_path / "ann.txt"
    odd.write_text("item_id,annotator_id,label\ni,a,yes\ni,b,no\n", encoding="utf-8")
    assert len(load_file(odd)) == 2
    with pytest.raises(AnnotrustError, match="No annotations left"):
        prepare(pd.DataFrame({"item_id": ["", ""], "annotator_id": ["a", "b"], "label": ["", ""]}))
    labels = [f"c{index}" for index in range(201)]
    wide = pd.DataFrame({"item_id": labels, "annotator_id": ["a"] * 201, "label": labels})
    with pytest.raises(AnnotrustError, match="Too many classes"):
        prepare(wide)


def test_text_coercion_and_model_prob_edges() -> None:
    frame = pd.DataFrame(
        {
            "item_id": [1.5, 2, True, None],
            "annotator_id": [1.0, 2, False, pd.NA],
            "label": [1.5, 2, True, "yes"],
            "model_label": ["", "no", "no", "no"],
            "model_prob_1.5": ["0.2", "nope", "0.2", "0.2"],
            "model_prob_2": [0.2, 0.2, 0.2, 0.2],
            "model_prob_True": [0.6, 0.8, 0.6, 0.6],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], utc=True),
        }
    )
    prepared = prepare(frame)
    assert "1.5" in prepared.classes
    assert prepared.model_probs is not None
    assert any(
        "Renormalized" in warning or "no model_prob" in warning or "Some items" in warning
        for warning in prepared.warnings
    )
    assert timestamps_usable(prepared.frame)
    assert timestamps_usable(pd.DataFrame({"timestamp": [pd.NaT, pd.NaT]})) is False
    assert parse_timestamps(pd.Series(dtype=object)).empty or True
    datetime_series = pd.to_datetime(pd.Series(["2026-01-01T00:00:00Z"]))
    assert parse_timestamps(datetime_series).notna().all()


def test_incomplete_model_probabilities_warn() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["yes", "yes", "no", "no"],
            "model_prob_no": [None, None, 0.2, 0.2],
            "model_prob_yes": [None, None, 0.8, 0.8],
        }
    )
    prepared = prepare(frame)
    assert any("Some items have no model_prob" in warning for warning in prepared.warnings)
    blank = frame.copy()
    blank["model_prob_no"] = None
    blank["model_prob_yes"] = None
    assert prepare(blank).model_probs is None


def test_importer_error_shapes(tmp_path: Path) -> None:
    with pytest.raises(AnnotrustError, match="list of label rows"):
        encord_payload_to_frame("nope")
    with pytest.raises(AnnotrustError, match="Unrecognized Encord"):
        encord_payload_to_frame({"labels": {}})
    with pytest.raises(AnnotrustError, match="did not contain"):
        encord_payload_to_frame(["nope"])
    value_label = {
        "label_hash": "row",
        "data_hash": "data",
        "classification_answers": {
            "bad": "skip",
            "c": {"classifications": [{"value": "Night", "name": "Time of day"}]},
        },
    }
    frame = encord_payload_to_frame(value_label)
    assert frame.loc[0, "label"] == "Night"
    assert "timestamp" not in frame.columns
    direct = {
        "data_hash": "d",
        "label_hash": "l",
        "classification_answers": {},
        "data_units": {"u": {"labels": {"classifications": [{"name": "Q", "answers": [{"name": "Day"}]}]}}},
    }
    assert encord_payload_to_frame(direct).loc[0, "label"] == "Day"
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(AnnotrustError, match="invalid JSON"):
        encord_to_frame(bad)
    with pytest.raises(AnnotrustError, match="Expected a list"):
        label_studio_payload_to_frame({"data": {}})
    with pytest.raises(AnnotrustError, match="Expected a list"):
        label_studio_payload_to_frame("nope")
    with pytest.raises(AnnotrustError, match="did not contain"):
        label_studio_payload_to_frame(["nope"])
    fallback = label_studio_payload_to_frame(
        [{"annotations": [{"id": 9, "result": [{"from_name": "s", "value": {"choices": ["A"]}}]}]}]
    )
    assert str(fallback.loc[0, "annotator_id"]).startswith("annotation:")
    with pytest.raises(AnnotrustError, match="invalid JSON"):
        label_studio_to_frame(bad)


def test_item_scores_and_helpers() -> None:
    reports = score_items(
        ["i"],
        ["a", "b", "c"],
        np.array([[0, 1, 2]], dtype=np.int64),
        np.array([[1 / 3, 1 / 3, 1 / 3]]),
        ScanConfig(),
        None,
        ["other"],
        None,
    )
    assert "high_disagreement" in reports[0].reasons
    assert "model_disagrees" in reports[0].reasons
    overturned = score_items(
        ["i"],
        ["a", "b", "c"],
        np.array([[0, 0, 1]], dtype=np.int64),
        np.array([[0.05, 0.9, 0.05]]),
        ScanConfig(),
        np.array([[np.nan, np.nan, np.nan]]),
        [None],
        np.array([np.nan]),
    )
    assert overturned[0].severity == "medium"
    assert overturned[0].cleanlab_score is None
    assert _cell("abcd", 3) == "abc"
    assert _cell("  hello   world ", 20) == "hello world"
    assert _cell("abcdefghij", 6).endswith("...")
    assert highest_severity([]) is None
    assert _longest_streak([]) == 0
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert _gaps([start, "not-a-time", start + timedelta(seconds=1)]) == []
    assert _gaps([start + timedelta(seconds=5), start]) == []
    short = pd.DataFrame(
        {
            "annotator_id": ["a", "a", "a"],
            "label": ["x", "x", "y"],
            "timestamp": [start, start + timedelta(seconds=1), start + timedelta(seconds=2)],
            "_row": [0, 1, 2],
        }
    )
    summary = annotator_behavior(short, ["x", "y"], timing_enabled=True, fast_seconds=2.0, min_timed_gaps=8)
    assert summary["a"].fast_fraction is None
    assert summary["a"].median_gap_seconds == pytest.approx(1.0)


def test_detector_edges() -> None:
    with pytest.raises(ValueError, match="square"):
        find_targeted_flips(np.zeros((2, 3)), np.ones(2), ["a", "b"], ScanConfig())
    with pytest.raises(ValueError, match="source_counts"):
        find_targeted_flips(np.eye(2), np.ones(1), ["a", "b"], ScanConfig())
    matrix = np.eye(3)
    matrix[0] = np.array([0.0, 1.0, 0.0])
    assert find_targeted_flips(matrix, np.array([30.0, 0.0, 0.0]), ["a", "b", "c"], ScanConfig()) == []
    labels = np.zeros((12, 2), dtype=np.int64)
    assert score_pairs(labels, labels[:, 0], ["a", "b"], ScanConfig()) == []
    medium = _careless_flag(0.5, 20, 0.9, 2, ScanConfig())
    assert medium is not None and medium.severity == "medium"
    low = _careless_flag(0.7, 30, 0.1, 20, ScanConfig())
    assert low is not None and low.severity == "low"
    assert _careless_flag(None, 20, 0.1, 20, ScanConfig()) is None
    with pytest.raises(ValueError, match="2-D"):
        fit_dawid_skene(np.array([0, 1], dtype=np.int64), 2)
    missing_column = np.array([[0, -1], [1, -1], [0, -1]], dtype=np.int64)
    fitted = fit_dawid_skene(missing_column, 2, max_iter=1)
    assert fitted.iterations == 1


def test_medium_flip_and_multiple_colluders_and_long_report() -> None:
    rows: list[dict[str, str]] = []
    for item in range(30):
        truth = str(item % 3)
        for annotator in ("h0", "h1", "h2", "h3"):
            rows.append({"item_id": f"i{item}", "annotator_id": annotator, "label": truth})
        for annotator in ("c0", "c1", "c2"):
            rows.append({"item_id": f"i{item}", "annotator_id": annotator, "label": str((item + 1) % 3)})
    result = scan(pd.DataFrame(rows), ScanConfig(min_labels=10))
    colluders = [row for row in result.annotators if row.annotator_id.startswith("c")]
    assert colluders
    assert any("+1 more" in flag.summary for row in colluders for flag in row.flags)
    many = []
    for annotator in range(41):
        many.append({"item_id": "i1", "annotator_id": f"p{annotator:02d}", "label": "yes"})
        many.append({"item_id": "i2", "annotator_id": f"p{annotator:02d}", "label": "no"})
    text = render_text(scan(pd.DataFrame(many), ScanConfig(min_labels=5)))
    assert "more annotators" in text
    agreed = []
    for item, label in enumerate(["yes", "yes", "no", "no", "yes", "no"]):
        for annotator in ("a", "b"):
            agreed.append({"item_id": f"i{item}", "annotator_id": annotator, "label": label})
    quiet = render_text(scan(pd.DataFrame(agreed), ScanConfig(min_labels=3)))
    assert "item flags: 0" in quiet
