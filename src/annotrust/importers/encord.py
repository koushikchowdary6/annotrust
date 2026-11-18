"""Best-effort reader for Encord label JSON.

Unofficial. This is not an Encord product and Encord does not endorse it.
The field names follow the public JSON export documentation:
https://docs.encord.com/platform-documentation/Annotate/annotate-export/annotate-export-json

What it reads
    Label rows (a list, or a dict with ``labels`` / ``label_rows``).
    Whole-frame classifications from ``classification_answers``. The chosen
    value is ``answers[].name`` (the documented example uses ``Night`` under
    the question ``Time of day``). ``createdBy`` / ``created_by`` becomes
    ``annotator_id`` when it is present. The documented
    ``classification_answers`` example does not include ``createdBy``; in that
    case the label row's ``label_hash`` is the annotator id, which treats one
    label row as one submission.

    If ``classification_answers`` yields nothing, classifications nested under
    ``data_units`` are tried. Rows whose ``value`` is only the question slug
    (the docs say that ``value`` repeats ``name``) are skipped, because the
    selected answer is not in the file.

What it does not read
    Bounding boxes, polygons, keypoints, and other geometries. Those are not
    a single class label shared by several annotators. A geometry-only export
    raises ``AnnotrustError`` rather than inventing rows.

Item ids are ``{data_hash}:{question name}`` so two questions on one asset
are not pooled into one label set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from annotrust.errors import AnnotrustError

_PERSON_KEYS = ("createdBy", "created_by", "lastEditedBy", "last_edited_by", "user_email", "email")
_TIME_KEYS = ("createdAt", "created_at", "lastEditedAt", "last_edited_at", "updated_at")


def encord_to_frame(path: Path) -> pd.DataFrame:
    """Read an Encord JSON export from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotrustError(f"{path}: invalid JSON") from exc
    except OSError as exc:
        raise AnnotrustError(f"Could not read {path}: {exc.strerror or exc}") from exc
    return encord_payload_to_frame(payload)


def encord_payload_to_frame(payload: object) -> pd.DataFrame:
    """Convert a decoded Encord export to annotation rows."""
    rows = _label_rows(payload)
    records: list[dict[str, object]] = []
    for row in rows:
        records.extend(_row_records(row))
    if not records:
        raise AnnotrustError(
            "Encord export contained no classification answers. "
            "This unofficial reader uses classification_answers (answers[].name) "
            "and does not import geometries. "
            "It is a best-effort parser of the public JSON export, not an Encord tool."
        )
    return pd.DataFrame(records)


def _label_rows(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if "classification_answers" in payload or "data_units" in payload or "label_hash" in payload:
            rows = [payload]
        else:
            nested = payload.get("labels", payload.get("label_rows"))
            if not isinstance(nested, list):
                raise AnnotrustError(
                    "Unrecognized Encord JSON. Expected a list of label rows, "
                    "or an object with classification_answers, data_units, or label_hash."
                )
            rows = nested
    else:
        raise AnnotrustError("Unrecognized Encord JSON. Expected a list of label rows.")
    parsed = [row for row in rows if isinstance(row, dict)]
    if not parsed:
        raise AnnotrustError("Encord JSON did not contain any label-row objects.")
    return parsed


def _row_records(row: dict[str, object]) -> list[dict[str, object]]:
    data_hash = _text(row.get("data_hash")) or _text(row.get("data_title")) or _text(row.get("label_hash"))
    label_hash = _text(row.get("label_hash")) or "label-row"
    if not data_hash:
        data_hash = label_hash
    from_answers = _from_classification_answers(row, data_hash, label_hash)
    if from_answers:
        return from_answers
    return _from_data_units(row, data_hash, label_hash)


def _from_classification_answers(
    row: dict[str, object],
    data_hash: str,
    label_hash: str,
) -> list[dict[str, object]]:
    raw = row.get("classification_answers")
    if not isinstance(raw, dict):
        return []
    records: list[dict[str, object]] = []
    for payload in raw.values():
        if not isinstance(payload, dict):
            continue
        classifications = payload.get("classifications")
        if not isinstance(classifications, list):
            continue
        for classification in classifications:
            if not isinstance(classification, dict):
                continue
            record = _classification_record(
                classification,
                data_hash=data_hash,
                fallback_annotator=label_hash,
                fallback_timestamp=_text(row.get("created_at")) or _text(row.get("last_edited_at")),
            )
            if record is not None:
                records.append(record)
    return records


def _from_data_units(row: dict[str, object], data_hash: str, label_hash: str) -> list[dict[str, object]]:
    units = row.get("data_units")
    if not isinstance(units, dict):
        return []
    records: list[dict[str, object]] = []
    for unit in units.values():
        if not isinstance(unit, dict):
            continue
        for classification in _unit_classifications(unit.get("labels", unit)):
            record = _classification_record(
                classification,
                data_hash=data_hash,
                fallback_annotator=label_hash,
                fallback_timestamp=_text(row.get("created_at")) or _text(row.get("last_edited_at")),
            )
            if record is not None:
                records.append(record)
    return records


def _unit_classifications(labels: object) -> list[dict[str, object]]:
    if not isinstance(labels, dict):
        return []
    direct = labels.get("classifications")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    found: list[dict[str, object]] = []
    for value in labels.values():
        if isinstance(value, dict) and isinstance(value.get("classifications"), list):
            found.extend(item for item in value["classifications"] if isinstance(item, dict))
    return found


def _classification_record(
    classification: dict[str, object],
    *,
    data_hash: str,
    fallback_annotator: str,
    fallback_timestamp: str,
) -> dict[str, object] | None:
    label = _classification_label(classification)
    if label is None:
        return None
    question = _text(classification.get("name")) or "classification"
    person = _person(classification) or fallback_annotator
    timestamp = _time(classification) or fallback_timestamp
    record: dict[str, object] = {
        "item_id": f"{data_hash}:{question}",
        "annotator_id": person,
        "label": label,
    }
    if timestamp:
        record["timestamp"] = timestamp
    return record


def _classification_label(classification: dict[str, object]) -> str | None:
    from_answers = _answer_names(classification.get("answers"))
    if from_answers:
        if len(from_answers) == 1:
            return from_answers[0]
        return "|".join(sorted(set(from_answers)))
    name = _text(classification.get("name"))
    value = _text(classification.get("value"))
    # The public docs say classification ``value`` repeats the question name.
    # Using it as the label would record the question, not the answer.
    if value and value != name:
        return value
    return None


def _answer_names(answers: object) -> list[str]:
    names: list[str] = []
    if isinstance(answers, str):
        text = answers.strip()
        return [text] if text else []
    if not isinstance(answers, list):
        return names
    for answer in answers:
        if isinstance(answer, str) and answer.strip():
            names.append(answer.strip())
        elif isinstance(answer, dict):
            text = _text(answer.get("name")) or _text(answer.get("value"))
            if text:
                names.append(text)
    return names


def _person(payload: dict[str, object]) -> str:
    for key in _PERSON_KEYS:
        text = _text(payload.get(key))
        if text:
            return text
    return ""


def _time(payload: dict[str, object]) -> str:
    for key in _TIME_KEYS:
        text = _text(payload.get(key))
        if text:
            return text
    return ""


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""
