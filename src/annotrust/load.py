"""Read annotation files and normalize them to one row per label."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from annotrust.errors import AnnotrustError
from annotrust.importers.encord import encord_payload_to_frame
from annotrust.importers.label_studio import label_studio_payload_to_frame

REQUIRED_COLUMNS = ("item_id", "annotator_id", "label")
FORMATS = ("auto", "csv", "tsv", "jsonl", "json", "encord", "label-studio")


@dataclass
class PreparedAnnotations:
    """Arrays the detectors share.

    ``label_index`` is ``(n_items, n_annotators)`` with ``-1`` for a missing
    label. Class indices follow ``classes``, which are sorted as strings.
    """

    frame: pd.DataFrame
    item_ids: list[str]
    annotator_ids: list[str]
    classes: list[str]
    label_index: NDArray[np.int64]
    model_probs: NDArray[np.float64] | None
    model_labels: list[str | None] | None
    warnings: list[str]


def load_file(path: Path, file_format: str = "auto") -> pd.DataFrame:
    """Load a local annotation file. No network access."""
    if file_format not in FORMATS:
        raise AnnotrustError(
            "Unknown format " + file_format + ". Use auto, csv, tsv, jsonl, json, encord, or label-studio."
        )
    if not path.is_file():
        raise AnnotrustError(f"File not found: {path}")
    resolved = _guess_format(path) if file_format == "auto" else file_format
    try:
        if resolved == "csv":
            return _read_table(path, "\t" if path.suffix.lower() == ".tsv" else ",")
        if resolved == "tsv":
            return _read_table(path, "\t")
        if resolved == "jsonl":
            return _read_jsonl(path)
        payload = _read_json(path)
        if resolved == "json":
            return _records_frame(payload, path)
        if resolved == "encord":
            return encord_payload_to_frame(payload)
        if resolved == "label-studio":
            return label_studio_payload_to_frame(payload)
    except AnnotrustError:
        raise
    except OSError as exc:
        raise AnnotrustError(f"Could not read {path}: {exc.strerror or exc}") from exc
    except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise AnnotrustError(f"Could not parse {path}: {exc}") from exc
    raise AnnotrustError(f"Unknown format {resolved}")


def prepare(frame: pd.DataFrame) -> PreparedAnnotations:
    """Normalize a dataframe of annotations."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        found = ", ".join(str(column) for column in frame.columns) or "(none)"
        needed = ", ".join(missing)
        raise AnnotrustError(
            f"Missing column(s): {needed}. Required columns are item_id, annotator_id, label. "
            f"Optional: timestamp, model_label, model_prob_<class>. Found: {found}."
        )
    warnings: list[str] = []
    work = frame.copy()
    for column in REQUIRED_COLUMNS:
        work[column] = [_as_text(value) for value in work[column].tolist()]
    complete = (work["item_id"] != "") & (work["annotator_id"] != "") & (work["label"] != "")
    dropped_incomplete = int((~complete).sum())
    if dropped_incomplete:
        warnings.append(f"Dropped {dropped_incomplete} rows with an empty item_id, annotator_id, or label.")
    work = work.loc[complete].copy()
    if work.empty:
        raise AnnotrustError("No annotations left after dropping incomplete rows.")

    duplicate = work.duplicated(["item_id", "annotator_id"], keep="last")
    dropped_duplicates = int(duplicate.sum())
    if dropped_duplicates:
        warnings.append(f"Dropped {dropped_duplicates} duplicate item_id/annotator_id rows; kept the last.")
    work = work.loc[~duplicate].copy()
    work["_row"] = np.arange(len(work))

    if "timestamp" in work.columns:
        parsed = parse_timestamps(work["timestamp"])
        if int(parsed.notna().sum()) == 0:
            warnings.append("timestamp column could not be parsed; speed checks were skipped.")
            work = work.drop(columns=["timestamp"])
        else:
            work["timestamp"] = parsed

    classes = sorted({label for label in work["label"].tolist() if label})
    if len(classes) < 2:
        raise AnnotrustError("Need at least 2 distinct labels to audit agreement.")
    if len(classes) > 200:
        raise AnnotrustError(f"Too many classes ({len(classes)}). Annotrust caps categorical label sets at 200.")

    item_ids = sorted(set(work["item_id"].tolist()))
    annotator_ids = sorted(set(work["annotator_id"].tolist()))
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    annotator_index = {annotator_id: index for index, annotator_id in enumerate(annotator_ids)}
    class_index = {label: index for index, label in enumerate(classes)}
    label_index = np.full((len(item_ids), len(annotator_ids)), -1, dtype=np.int64)
    for item_id, annotator_id, label in zip(
        work["item_id"].tolist(),
        work["annotator_id"].tolist(),
        work["label"].tolist(),
        strict=True,
    ):
        label_index[item_index[item_id], annotator_index[annotator_id]] = class_index[label]

    model_probs, model_labels = _model_inputs(work, item_ids, classes, warnings)
    return PreparedAnnotations(
        frame=work,
        item_ids=item_ids,
        annotator_ids=annotator_ids,
        classes=classes,
        label_index=label_index,
        model_probs=model_probs,
        model_labels=model_labels,
        warnings=warnings,
    )


def timestamps_usable(frame: pd.DataFrame) -> bool:
    """True when the column exists and has at least two distinct times."""
    if "timestamp" not in frame.columns:
        return False
    column = frame["timestamp"]
    present = column.dropna()
    if present.empty:
        return False
    return int(present.nunique()) >= 2


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse ISO-8601 strings, unix seconds, or unix milliseconds.

    Values are treated as milliseconds when most of the column is numeric and
    the median is above 1e11. Smaller numbers are seconds, including offsets
    from an arbitrary origin. Only the gaps matter for the speed check.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_fraction = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_fraction > 0.8:
        median = float(numeric.median())
        unit = "ms" if median > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format")
        return pd.to_datetime(series, utc=True, errors="coerce")


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _read_table(path: Path, separator: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, sep=separator)


def _read_jsonl(path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AnnotrustError(f"{path}: line {number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise AnnotrustError(f"{path}: line {number}: expected a JSON object")
            records.append(payload)
    if not records:
        raise AnnotrustError(f"{path}: no JSONL records")
    return pd.DataFrame(records)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotrustError(f"{path}: invalid JSON") from exc
    except OSError as exc:
        raise AnnotrustError(f"Could not read {path}: {exc.strerror or exc}") from exc


def _records_frame(payload: object, path: Path) -> pd.DataFrame:
    if isinstance(payload, dict):
        raise AnnotrustError(
            f"{path}: JSON object is not a list of annotation records. "
            "Pass --format encord or --format label-studio if this is an export."
        )
    if not isinstance(payload, list) or not payload:
        raise AnnotrustError(f"{path}: JSON annotation file has no records")
    if not isinstance(payload[0], dict):
        raise AnnotrustError(f"{path}: JSON annotation records must be objects")
    return pd.DataFrame(payload)


def _guess_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        payload = _read_json(path)
        return _sniff_json(payload)
    return "csv"


def _sniff_json(payload: object) -> str:
    if isinstance(payload, dict):
        if "classification_answers" in payload or "data_units" in payload or "label_hash" in payload:
            return "encord"
        if "annotations" in payload or "completions" in payload:
            return "label-studio"
        for key in ("labels", "label_rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return _sniff_json(value)
        if "item_id" in payload and "label" in payload:
            return "json"
        raise AnnotrustError("Unrecognized JSON object. Pass --format csv, jsonl, json, encord, or label-studio.")
    if not isinstance(payload, list) or not payload:
        raise AnnotrustError("JSON file has no records.")
    first = payload[0]
    if not isinstance(first, dict):
        raise AnnotrustError("JSON file is not a list of objects.")
    if "annotations" in first or "completions" in first:
        return "label-studio"
    if "classification_answers" in first or "data_units" in first or "label_hash" in first:
        return "encord"
    if "item_id" in first and "label" in first:
        return "json"
    raise AnnotrustError("Unrecognized JSON. Pass --format csv, jsonl, json, encord, or label-studio.")


def _model_inputs(
    frame: pd.DataFrame,
    item_ids: list[str],
    classes: list[str],
    warnings: list[str],
) -> tuple[NDArray[np.float64] | None, list[str | None] | None]:
    model_labels = _first_model_labels(frame, item_ids) if "model_label" in frame.columns else None
    expected = [f"model_prob_{label}" for label in classes]
    present_named = [column for column in frame.columns if str(column).startswith("model_prob_")]
    if not present_named:
        return None, model_labels
    if present_named != expected and set(present_named) != set(expected):
        warnings.append(
            "Ignored model_prob_* columns because they do not match the class names one-for-one. "
            "Expected model_prob_<class> for every class."
        )
        return None, model_labels
    matrix = np.full((len(item_ids), len(classes)), np.nan, dtype=np.float64)
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    numeric_columns = {
        label: pd.to_numeric(frame[f"model_prob_{label}"], errors="coerce").to_numpy(dtype=np.float64)
        for label in classes
    }
    conflict = False
    for row_number, item_id in enumerate(frame["item_id"].tolist()):
        index = item_index[str(item_id)]
        current = np.asarray([numeric_columns[label][row_number] for label in classes], dtype=np.float64)
        if np.isnan(current).any():
            continue
        if np.all(np.isnan(matrix[index])):
            matrix[index] = current
        elif not np.allclose(matrix[index], current, atol=1e-6, equal_nan=False):
            conflict = True
    if conflict:
        warnings.append("model_prob_* values differ across annotators of the same item; kept the first.")
    observed = ~np.isnan(matrix).any(axis=1)
    if not bool(observed.any()):
        warnings.append("model_prob_* columns were present but no row had a full numeric probability vector.")
        return None, model_labels
    if not bool(observed.all()):
        warnings.append("Some items have no model_prob_* values; model comparison skips them.")
    row_sums = np.nansum(matrix, axis=1, keepdims=True)
    bad_sum = observed & (np.abs(row_sums.squeeze(-1) - 1.0) > 0.05)
    if bool(np.any(bad_sum)):
        warnings.append("Renormalized model_prob_* rows that did not sum to 1 within 0.05.")
    safe = np.where(row_sums <= 0, 1.0, row_sums)
    matrix = matrix / safe
    return matrix, model_labels


def _first_model_labels(frame: pd.DataFrame, item_ids: list[str]) -> list[str | None]:
    found: dict[str, str] = {}
    for item_id, raw in zip(frame["item_id"].tolist(), frame["model_label"].tolist(), strict=True):
        text = _as_text(raw)
        if text and item_id not in found:
            found[item_id] = text
    return [found.get(item_id) for item_id in item_ids]
