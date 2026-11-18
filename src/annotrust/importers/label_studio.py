"""Best-effort reader for Label Studio JSON exports.

Unofficial. This is not a HumanSignal product and HumanSignal does not
endorse it. The shape follows the public export documentation for a JSON
list of tasks: each task has ``annotations`` (older exports say
``completions``) and each annotation has ``result``.

What it reads
    ``choices`` and ``taxonomy`` results. ``completed_by`` (an id, a string,
    or an object with ``email``) is the annotator. ``created_at`` is the
    timestamp. Cancelled annotations (``was_cancelled``) are skipped.
    Predictions are ignored.

Item ids are ``{task id}:{from_name}`` so two questions in one task are not
pooled. Several selected choices become one label joined with ``|`` in sorted
order.

What it does not read
    Brushes, rectangles, text areas, and other result types that are not a
    single classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from annotrust.errors import AnnotrustError


def label_studio_to_frame(path: Path) -> pd.DataFrame:
    """Read a Label Studio JSON export from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotrustError(f"{path}: invalid JSON") from exc
    except OSError as exc:
        raise AnnotrustError(f"Could not read {path}: {exc.strerror or exc}") from exc
    return label_studio_payload_to_frame(payload)


def label_studio_payload_to_frame(payload: object) -> pd.DataFrame:
    """Convert decoded Label Studio tasks to annotation rows."""
    tasks = _tasks(payload)
    records: list[dict[str, object]] = []
    for index, task in enumerate(tasks):
        task_id = task.get("id", index)
        annotations = task.get("annotations", task.get("completions", []))
        if not isinstance(annotations, list):
            continue
        for annotation in annotations:
            if not isinstance(annotation, dict) or annotation.get("was_cancelled") is True:
                continue
            records.extend(_annotation_records(task_id, annotation))
    if not records:
        raise AnnotrustError(
            "Label Studio export contained no choices or taxonomy results. "
            "This unofficial reader skips geometries, text areas, and predictions. "
            "It is a best-effort parser of the public JSON export, not a HumanSignal tool."
        )
    return pd.DataFrame(records)


def _tasks(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        if "annotations" in payload or "completions" in payload:
            return [payload]
        raise AnnotrustError("Unrecognized Label Studio JSON. Expected a list of tasks.")
    if not isinstance(payload, list):
        raise AnnotrustError("Unrecognized Label Studio JSON. Expected a list of tasks.")
    tasks = [task for task in payload if isinstance(task, dict)]
    if not tasks:
        raise AnnotrustError("Label Studio JSON did not contain any tasks.")
    return tasks


def _annotation_records(task_id: object, annotation: dict[str, object]) -> list[dict[str, object]]:
    results = annotation.get("result", [])
    if not isinstance(results, list):
        return []
    grouped: dict[str, list[str]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        parsed = _result_label(result)
        if parsed is None:
            continue
        from_name, label = parsed
        grouped.setdefault(from_name, []).append(label)
    if not grouped:
        return []
    annotator = _annotator(annotation)
    timestamp = annotation.get("created_at") or annotation.get("updated_at")
    records: list[dict[str, object]] = []
    for from_name, labels in grouped.items():
        flat: list[str] = []
        for label in labels:
            flat.extend(part for part in label.split("|") if part)
        unique = sorted(set(flat))
        record: dict[str, object] = {
            "item_id": f"{task_id}:{from_name}",
            "annotator_id": annotator,
            "label": unique[0] if len(unique) == 1 else "|".join(unique),
        }
        if isinstance(timestamp, str) and timestamp.strip():
            record["timestamp"] = timestamp.strip()
        records.append(record)
    return records


def _result_label(result: dict[str, object]) -> tuple[str, str] | None:
    value = result.get("value")
    if not isinstance(value, dict):
        return None
    from_name = str(result.get("from_name") or "label")
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        names = [str(item) for item in choices if str(item).strip()]
        if names:
            return from_name, names[0] if len(names) == 1 else "|".join(sorted(names))
    taxonomy = value.get("taxonomy")
    if isinstance(taxonomy, list) and taxonomy:
        leaves: list[str] = []
        for item in taxonomy:
            if isinstance(item, list) and item:
                leaves.append(str(item[-1]))
            elif isinstance(item, str) and item.strip():
                leaves.append(item.strip())
        if leaves:
            return from_name, leaves[0] if len(leaves) == 1 else "|".join(sorted(leaves))
    return None


def _annotator(annotation: dict[str, object]) -> str:
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        for key in ("email", "username", "id"):
            value = who.get(key)
            if value not in (None, ""):
                return str(value)
    if isinstance(who, str) and who.strip():
        return who.strip()
    if isinstance(who, int) and not isinstance(who, bool):
        return str(who)
    if annotation.get("id") is not None:
        return f"annotation:{annotation['id']}"
    return "unknown"
