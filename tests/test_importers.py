"""Unofficial Encord and Label Studio readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotrust.errors import AnnotrustError
from annotrust.importers.encord import encord_payload_to_frame, encord_to_frame
from annotrust.importers.label_studio import label_studio_payload_to_frame, label_studio_to_frame
from annotrust.load import load_file


def test_encord_classification_answers_use_the_documented_answer_name() -> None:
    payload = [
        {
            "label_hash": "row-1",
            "data_hash": "data-1",
            "created_at": "2024-01-01 00:00:00",
            "classification_answers": {
                "OFX2jzwE": {
                    "classificationHash": "OFX2jzwE",
                    "classifications": [
                        {
                            "name": "Time of day",
                            "value": "time_of_day",
                            "answers": [{"name": "Night", "value": "night", "featureHash": "j/7dpSmi"}],
                            "createdBy": "ada@example.com",
                            "createdAt": "Tue, 10 Oct 2023 14:16:14 GMT",
                        }
                    ],
                }
            },
        }
    ]
    frame = encord_payload_to_frame(payload)
    assert frame.loc[0, "item_id"] == "data-1:Time of day"
    assert frame.loc[0, "annotator_id"] == "ada@example.com"
    assert frame.loc[0, "label"] == "Night"


def test_encord_without_created_by_uses_the_label_row() -> None:
    payload = [
        {
            "label_hash": "row-9",
            "data_hash": "data-9",
            "classification_answers": {
                "h1": {
                    "classifications": [
                        {
                            "name": "Quality",
                            "value": "quality",
                            "answers": [{"name": "Good"}, {"name": "Fair"}],
                        }
                    ]
                }
            },
        }
    ]
    frame = encord_payload_to_frame(payload)
    assert frame.loc[0, "annotator_id"] == "row-9"
    assert frame.loc[0, "label"] == "Fair|Good"


def test_encord_falls_back_to_data_units_and_rejects_geometries() -> None:
    payload = {
        "label_hash": "row-2",
        "data_hash": "data-2",
        "classification_answers": {},
        "data_units": {
            "data-2": {
                "labels": {
                    "0": {
                        "objects": [{"name": "Cherries", "shape": "bounding_box"}],
                        "classifications": [
                            {
                                "name": "Time of day",
                                "value": "time_of_day",
                                "answers": "Day",
                                "createdBy": "bea@example.com",
                            }
                        ],
                    }
                }
            }
        },
    }
    frame = encord_payload_to_frame(payload)
    assert frame.loc[0, "label"] == "Day"
    assert frame.loc[0, "annotator_id"] == "bea@example.com"

    geometry_only = {
        "label_hash": "row-3",
        "data_hash": "data-3",
        "classification_answers": {},
        "data_units": {"data-3": {"labels": {"objects": [{"name": "Cherries", "shape": "bounding_box"}]}}},
    }
    with pytest.raises(AnnotrustError, match="geometr"):
        encord_payload_to_frame(geometry_only)


def test_encord_file_and_auto_format(tmp_path: Path) -> None:
    payload = [
        {
            "label_hash": "row-1",
            "data_hash": "data-1",
            "classification_answers": {
                "h": {"classifications": [{"name": "Q", "value": "q", "answers": [{"name": "Yes"}]}]}
            },
        },
        {
            "label_hash": "row-2",
            "data_hash": "data-1",
            "classification_answers": {
                "h": {"classifications": [{"name": "Q", "value": "q", "answers": [{"name": "No"}]}]}
            },
        },
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    frame = encord_to_frame(path)
    assert len(frame) == 2
    loaded = load_file(path)
    assert set(loaded["label"]) == {"Yes", "No"}


def test_label_studio_choices_taxonomy_and_cancelled(tmp_path: Path) -> None:
    payload = [
        {
            "id": 7,
            "annotations": [
                {
                    "id": 1,
                    "completed_by": {"email": "ada@example.com"},
                    "created_at": "2024-05-01T00:00:00.000000Z",
                    "result": [
                        {
                            "from_name": "sentiment",
                            "type": "choices",
                            "value": {"choices": ["Positive"]},
                        },
                        {"from_name": "region", "type": "rectangle", "value": {"x": 1}},
                    ],
                },
                {
                    "id": 2,
                    "was_cancelled": True,
                    "completed_by": 3,
                    "result": [
                        {
                            "from_name": "sentiment",
                            "type": "choices",
                            "value": {"choices": ["Negative"]},
                        }
                    ],
                },
            ],
        },
        {
            "id": 8,
            "completions": [
                {
                    "completed_by": "bea",
                    "result": [
                        {
                            "from_name": "topic",
                            "type": "taxonomy",
                            "value": {"taxonomy": [["News", "World"], ["Sports"]]},
                        }
                    ],
                }
            ],
        },
    ]
    frame = label_studio_payload_to_frame(payload)
    sentiment = frame[frame["item_id"] == "7:sentiment"]
    assert sentiment["label"].tolist() == ["Positive"]
    assert sentiment["annotator_id"].tolist() == ["ada@example.com"]
    topic = frame[frame["item_id"] == "8:topic"].iloc[0]
    assert topic["label"] == "Sports|World"
    assert topic["annotator_id"] == "bea"

    path = tmp_path / "ls.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(label_studio_to_frame(path)) == 2
    with pytest.raises(AnnotrustError, match="no choices"):
        label_studio_payload_to_frame([{"id": 1, "annotations": [{"result": [{"value": {"text": ["hi"]}}]}]}])
