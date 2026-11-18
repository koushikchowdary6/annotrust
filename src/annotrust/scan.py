"""Run the audit on one annotation table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from annotrust.behavior import Behavior, annotator_behavior
from annotrust.cleanlab_compare import compare_item_scores
from annotrust.collusion import score_pairs
from annotrust.config import ScanConfig
from annotrust.dawid_skene import fit_dawid_skene
from annotrust.items import score_items
from annotrust.load import load_file, prepare, timestamps_usable
from annotrust.poison import find_targeted_flips
from annotrust.results import AnnotatorReport, Flag, PairReport, ScanResult, highest_severity
from annotrust.version import __version__

_KIND_RANK = {"poisoning": 4, "colluding": 3, "careless": 2, "fast": 1}


def scan(
    data: object,
    config: ScanConfig | None = None,
    *,
    source: str | None = None,
    file_format: str = "auto",
) -> ScanResult:
    """Audit a dataframe or a path to CSV, JSONL, JSON, Encord, or Label Studio.

    ``data`` is a path or a pandas DataFrame with ``item_id``, ``annotator_id``,
    and ``label``.
    """
    settings = config if config is not None else ScanConfig()
    settings.validate()
    if isinstance(data, pd.DataFrame):
        origin = source
        prepared = prepare(data)
    else:
        path = Path(str(data))
        origin = source if source is not None else str(path)
        prepared = prepare(load_file(path, file_format))
    warnings = list(prepared.warnings)
    if prepared.label_index.shape[1] == 1:
        warnings.append(
            "Only one annotator. Consensus equals their labels, so agreement, "
            "collusion, and flip checks are not informative. Speed and streak checks can still fire."
        )
    else:
        per_item = (prepared.label_index >= 0).sum(axis=1)
        if int(np.max(per_item)) < 2:
            warnings.append(
                "No item was labeled by more than one annotator. Consensus equals that annotator, "
                "so agreement, collusion, and flip checks are not informative."
            )

    fitted = fit_dawid_skene(
        prepared.label_index,
        len(prepared.classes),
        max_iter=settings.max_iter,
        tol=settings.tol,
        smoothing=settings.smoothing,
    )
    map_labels = np.argmax(fitted.posterior, axis=1).astype(np.int64)
    timing = timestamps_usable(prepared.frame)
    if "timestamp" in prepared.frame.columns and not timing:
        warnings.append("Timestamps do not vary; speed checks were skipped.")
    behavior = annotator_behavior(
        prepared.frame,
        prepared.classes,
        timing_enabled=timing,
        fast_seconds=settings.fast_seconds,
        min_timed_gaps=settings.min_timed_gaps,
    )
    pairs = score_pairs(prepared.label_index, map_labels, prepared.annotator_ids, settings)
    cleanlab_scores = _cleanlab(prepared.model_probs, map_labels, settings, warnings)
    items = score_items(
        prepared.item_ids,
        prepared.classes,
        prepared.label_index,
        fitted.posterior,
        settings,
        prepared.model_probs,
        prepared.model_labels,
        cleanlab_scores,
    )
    annotators = _annotators(
        prepared.annotator_ids,
        prepared.classes,
        prepared.label_index,
        map_labels,
        fitted.confusion,
        fitted.posterior,
        fitted.prior,
        behavior,
        pairs,
        settings,
    )
    return ScanResult(
        version=__version__,
        source=origin,
        n_annotations=int(len(prepared.frame)),
        n_items=len(prepared.item_ids),
        n_annotators=len(prepared.annotator_ids),
        classes=tuple(prepared.classes),
        warnings=tuple(warnings),
        iterations=fitted.iterations,
        log_likelihood=fitted.log_likelihood,
        prior=tuple(float(value) for value in fitted.prior),
        annotators=tuple(annotators),
        pairs=tuple(pairs),
        items=tuple(items),
        config=settings,
    )


def _cleanlab(
    model_probs: np.ndarray | None,
    map_labels: np.ndarray,
    config: ScanConfig,
    warnings: list[str],
) -> np.ndarray | None:
    if not config.use_cleanlab:
        return None
    if model_probs is None:
        warnings.append("cleanlab comparison was requested but no model_prob_<class> columns were found.")
        return None
    observed = ~np.isnan(model_probs).any(axis=1)
    if not bool(observed.all()):
        warnings.append("cleanlab scores were skipped for items with incomplete model probabilities.")
    scores = np.full(model_probs.shape[0], np.nan, dtype=np.float64)
    if bool(observed.any()):
        scores[observed] = compare_item_scores(map_labels[observed].astype(np.int64), model_probs[observed])
    return scores


def _annotators(
    annotator_ids: list[str],
    classes: list[str],
    label_index: np.ndarray,
    map_labels: np.ndarray,
    confusion: np.ndarray,
    posterior: np.ndarray,
    prior: np.ndarray,
    behavior_map: dict[str, Behavior],
    pair_reports: list[PairReport],
    config: ScanConfig,
) -> list[AnnotatorReport]:
    reports: list[AnnotatorReport] = []
    for index, annotator_id in enumerate(annotator_ids):
        observed_rows = label_index[:, index] >= 0
        observed = label_index[observed_rows, index]
        n_labels = int(observed.size)
        if n_labels:
            accuracy = float(np.mean(observed == map_labels[observed_rows]))
            source_counts = posterior[observed_rows].sum(axis=0)
        else:
            accuracy = None
            source_counts = np.zeros(len(classes), dtype=np.float64)
        ds_accuracy = float(np.dot(prior, np.diag(confusion[index])))
        pattern = behavior_map[annotator_id]
        flips = []
        if n_labels >= config.min_labels:
            flips = find_targeted_flips(confusion[index], source_counts, classes, config)
        flags: list[Flag] = []
        if flips:
            strong = any(flip.rate >= config.poison_high_rate for flip in flips)
            summary = "; ".join(
                f"flip {flip.source}->{flip.target} rate={flip.rate:.2f} "
                f"conc={flip.concentration:.2f} n={flip.source_count:.0f}"
                for flip in flips
            )
            flags.append(Flag(kind="poisoning", severity="high" if strong else "medium", summary=summary))
        else:
            careless = _careless_flag(accuracy, n_labels, pattern.normalized_entropy, pattern.longest_streak, config)
            if careless is not None:
                flags.append(careless)
        collusion = _collusion_flag(annotator_id, pair_reports)
        if collusion is not None:
            flags.append(collusion)
        if pattern.fast_fraction is not None and pattern.fast_fraction >= config.fast_fraction:
            median = pattern.median_gap_seconds
            median_text = f", median {median:.2f}s" if median is not None else ""
            flags.append(
                Flag(
                    kind="fast",
                    severity="low",
                    summary=(
                        f"fast labels {pattern.fast_fraction:.0%} of gaps under {config.fast_seconds:.0f}s{median_text}"
                    ),
                )
            )
        primary = _primary(flags, n_labels, config.min_labels)
        severity = highest_severity([flag.severity for flag in flags])
        matrix = tuple(tuple(float(value) for value in row) for row in confusion[index])
        reports.append(
            AnnotatorReport(
                annotator_id=annotator_id,
                n_labels=n_labels,
                accuracy=accuracy,
                ds_accuracy=ds_accuracy,
                normalized_entropy=pattern.normalized_entropy,
                longest_streak=pattern.longest_streak,
                fast_fraction=pattern.fast_fraction,
                median_gap_seconds=pattern.median_gap_seconds,
                confusion=matrix,
                flips=tuple(flips),
                flags=tuple(flags),
                primary=primary,
                severity=severity,
            )
        )
    reports.sort(key=lambda report: (_severity_sort(report.severity), report.annotator_id))
    return reports


def _careless_flag(
    accuracy: float | None,
    n_labels: int,
    normalized_entropy: float,
    longest_streak: int,
    config: ScanConfig,
) -> Flag | None:
    if accuracy is None or n_labels < config.min_labels:
        return None
    streak_cut = max(config.streak_min_length, int(config.streak_min_fraction * n_labels))
    degenerate = normalized_entropy <= config.streak_max_entropy and longest_streak >= streak_cut
    low_accuracy = accuracy <= config.careless_max_accuracy
    patterned = degenerate and accuracy <= config.streak_max_accuracy
    if not low_accuracy and not patterned:
        return None
    summary = f"low agreement {accuracy:.2f}"
    if degenerate:
        summary += f", constant-label streak {longest_streak}"
    if accuracy <= config.careless_high_accuracy:
        severity = "high"
    elif low_accuracy:
        severity = "medium"
    else:
        severity = "low"
    return Flag(kind="careless", severity=severity, summary=summary)


def _collusion_flag(annotator_id: str, pairs: list[PairReport]) -> Flag | None:
    matched = [pair for pair in pairs if pair.flagged and annotator_id in {pair.annotator_a, pair.annotator_b}]
    if not matched:
        return None
    matched.sort(key=lambda pair: pair.score, reverse=True)
    top = matched[0]
    other = top.annotator_b if top.annotator_a == annotator_id else top.annotator_a
    extra = f" (+{len(matched) - 1} more)" if len(matched) > 1 else ""
    summary = (
        f"pair {other} kappa={top.kappa:.2f} excess={top.excess:.2f} "
        f"joint_wrong={top.joint_wrong} score={top.score:.2f}{extra}"
    )
    return Flag(kind="colluding", severity=top.severity or "medium", summary=summary)


def _primary(flags: list[Flag], n_labels: int, min_labels: int) -> str:
    if not flags:
        return "honest" if n_labels >= min_labels else "insufficient"
    return max(flags, key=lambda flag: (_severity_sort_value(flag.severity), _KIND_RANK[flag.kind])).kind


def _severity_sort(severity: str | None) -> int:
    if severity is None:
        return 3
    return {"high": 0, "medium": 1, "low": 2}[severity]


def _severity_sort_value(severity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}[severity]
