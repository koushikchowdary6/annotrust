"""File loading and timestamp parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from annotrust.errors import AnnotrustError
from annotrust.load import load_file, parse_timestamps, prepare


def test_csv_normalizes_integral_floats_and_drops_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "ann.csv"
    path.write_text(
        "item_id,annotator_id,label\n1,10,0\n1,10,1\n2,10,0\n2,11,\n",
        encoding="utf-8",
    )
    frame = load_file(path)
    prepared = prepare(frame)
    assert prepared.classes == ["0", "1"]
    assert any("duplicate" in warning for warning in prepared.warnings)
    assert any("empty" in warning for warning in prepared.warnings)
    kept = prepared.frame
    row = kept[(kept["item_id"] == "1") & (kept["annotator_id"] == "10")]
    assert row["label"].tolist() == ["1"]


def test_jsonl_and_json_records(tmp_path: Path) -> None:
    jsonl = tmp_path / "ann.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "i", "annotator_id": "a", "label": "yes"}),
                "",
                json.dumps({"item_id": "i", "annotator_id": "b", "label": "no"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prepared = prepare(load_file(jsonl))
    assert set(prepared.classes) == {"yes", "no"}

    array_path = tmp_path / "ann.json"
    array_path.write_text(
        json.dumps(
            [
                {"item_id": "i", "annotator_id": "a", "label": "yes"},
                {"item_id": "i", "annotator_id": "b", "label": "no"},
            ]
        ),
        encoding="utf-8",
    )
    prepared = prepare(load_file(array_path))
    assert prepared.item_ids == ["i"]


def test_tsv(tmp_path: Path) -> None:
    path = tmp_path / "ann.tsv"
    path.write_text("item_id\tannotator_id\tlabel\ni\ta\tyes\ni\tb\tno\n", encoding="utf-8")
    assert len(prepare(load_file(path)).annotator_ids) == 2


def test_missing_column_names_what_was_found() -> None:
    with pytest.raises(AnnotrustError, match="Missing column"):
        prepare(pd.DataFrame({"item_id": ["i"], "label": ["a"]}))


def test_one_class_is_rejected() -> None:
    frame = pd.DataFrame({"item_id": ["i1", "i2"], "annotator_id": ["a", "a"], "label": ["only", "only"]})
    with pytest.raises(AnnotrustError, match="at least 2"):
        prepare(frame)


def test_empty_and_bad_json(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(AnnotrustError, match="no JSONL"):
        load_file(empty)
    broken = tmp_path / "bad.jsonl"
    broken.write_text("{nope}\n", encoding="utf-8")
    with pytest.raises(AnnotrustError, match="invalid JSON"):
        load_file(broken)
    missing = tmp_path / "nope.csv"
    with pytest.raises(AnnotrustError, match="File not found"):
        load_file(missing)
    with pytest.raises(AnnotrustError, match="Unknown format"):
        load_file(tmp_path / "x.csv", "parquet")


def test_timestamp_units() -> None:
    iso = parse_timestamps(pd.Series(["2026-01-01T00:00:00Z", "2026-01-01T00:00:05Z"]))
    assert iso.notna().all()
    seconds = parse_timestamps(pd.Series([1_700_000_000, 1_700_000_010]))
    assert (seconds.iloc[1] - seconds.iloc[0]).total_seconds() == pytest.approx(10)
    millis = parse_timestamps(pd.Series([1_700_000_000_000, 1_700_000_010_000]))
    assert (millis.iloc[1] - millis.iloc[0]).total_seconds() == pytest.approx(10)
    offsets = parse_timestamps(pd.Series([0, 2.5]))
    assert (offsets.iloc[1] - offsets.iloc[0]).total_seconds() == pytest.approx(2.5)


def test_unparseable_timestamps_are_dropped() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["yes", "yes", "no", "no"],
            "timestamp": ["not-a-time", "still-no", "nope", "nah"],
        }
    )
    prepared = prepare(frame)
    assert "timestamp" not in prepared.frame.columns
    assert any("could not be parsed" in warning for warning in prepared.warnings)


def test_model_probabilities_are_aligned() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["no", "no", "yes", "yes"],
            "model_label": ["yes", "yes", "yes", "yes"],
            "model_prob_no": [0.2, 0.2, 0.1, 0.4],
            "model_prob_yes": [0.8, 0.8, 0.9, 0.6],
        }
    )
    prepared = prepare(frame)
    assert prepared.model_probs is not None
    assert prepared.model_labels == ["yes", "yes"]
    assert any("differ" in warning for warning in prepared.warnings)


def test_model_columns_that_do_not_match_classes_are_ignored() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1"],
            "annotator_id": ["a", "b"],
            "label": ["no", "yes"],
            "model_prob_other": [0.5, 0.5],
        }
    )
    prepared = prepare(frame)
    assert prepared.model_probs is None
    assert any("Ignored model_prob" in warning for warning in prepared.warnings)
