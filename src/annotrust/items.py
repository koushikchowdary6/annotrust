"""Item-level label quality from the Dawid-Skene posterior.

A flagged item is a suspicion. High posterior entropy means the annotators
disagreed. ``majority_overturned`` means the modal label is not the latent
class the EM settled on, and the posterior is confident. Neither one proves
the stored label is wrong.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy

from annotrust.config import ScanConfig
from annotrust.results import ItemReport


def score_items(
    item_ids: list[str],
    classes: list[str],
    label_index: NDArray[np.int64],
    posterior: NDArray[np.float64],
    config: ScanConfig,
    model_probs: NDArray[np.float64] | None,
    model_labels: list[str | None] | None,
    cleanlab_scores: NDArray[np.float64] | None,
) -> list[ItemReport]:
    """Score every item."""
    n_classes = len(classes)
    normalizer = math.log2(n_classes) if n_classes > 1 else 1.0
    reports: list[ItemReport] = []
    for index, item_id in enumerate(item_ids):
        observed = label_index[index]
        observed = observed[observed >= 0]
        counts = np.bincount(observed, minlength=n_classes)
        majority_index = int(np.argmax(counts))
        majority_fraction = float(counts[majority_index] / counts.sum()) if counts.sum() else 0.0
        probabilities = posterior[index]
        estimated_index = int(np.argmax(probabilities))
        confidence = float(probabilities[estimated_index])
        normalized_entropy = float(entropy(probabilities, base=2)) / normalizer if normalizer else 0.0
        reasons: list[str] = []
        if int(observed.size) >= 3 and normalized_entropy >= config.item_entropy:
            reasons.append("high_disagreement")
        if (
            int(observed.size) >= 3
            and majority_index != estimated_index
            and confidence >= config.item_overturn_confidence
        ):
            reasons.append("majority_overturned")
        model_label, model_confidence = _model_call(
            classes,
            model_probs[index] if model_probs is not None else None,
            model_labels[index] if model_labels is not None else None,
        )
        if model_label is not None and model_label != classes[estimated_index]:
            confident_model = model_confidence is None or model_confidence >= config.model_disagree_confidence
            if confident_model:
                reasons.append("model_disagrees")
        severity: str | None = None
        if "majority_overturned" in reasons and confidence >= 0.75:
            severity = "medium"
        elif reasons:
            severity = "low"
        cleanlab_score = None
        if cleanlab_scores is not None and not math.isnan(float(cleanlab_scores[index])):
            cleanlab_score = float(cleanlab_scores[index])
        reports.append(
            ItemReport(
                item_id=item_id,
                estimated_label=classes[estimated_index],
                confidence=confidence,
                normalized_entropy=normalized_entropy,
                majority_label=classes[majority_index],
                majority_fraction=majority_fraction,
                n_annotators=int(observed.size),
                suspect=bool(reasons),
                severity=severity,
                reasons=tuple(reasons),
                model_label=model_label,
                model_confidence=model_confidence,
                cleanlab_score=cleanlab_score,
            )
        )
    return reports


def _model_call(
    classes: list[str],
    probs: NDArray[np.float64] | None,
    model_label: str | None,
) -> tuple[str | None, float | None]:
    if probs is not None and not np.isnan(probs).any():
        choice = int(np.argmax(probs))
        return classes[choice], float(probs[choice])
    if model_label is not None:
        return model_label, None
    return None, None
