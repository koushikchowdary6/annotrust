"""Collusion score from pairwise kappa and agreement on consensus errors.

Cohen's kappa measures agreement beyond chance. Two careful annotators also
have a high kappa with the consensus, so the excess over that consensus kappa
is the part that is not explained by both being right. The score then adds
the rate at which the pair agrees with each other given that both differ from
the consensus. Copying a correct annotator does not clear the excess test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from annotrust.config import ScanConfig
from annotrust.kappa import cohen_kappa
from annotrust.results import PairReport


def score_pairs(
    label_index: NDArray[np.int64],
    map_labels: NDArray[np.int64],
    annotator_ids: list[str],
    config: ScanConfig,
) -> list[PairReport]:
    """Score every pair that shares at least ``min_pair_overlap`` items."""
    n_annotators = label_index.shape[1]
    observed_max = int(np.max(label_index)) if label_index.size else -1
    map_max = int(np.max(map_labels)) if map_labels.size else 0
    n_classes = max(observed_max, map_max, 0) + 1
    if n_classes < 2:
        return []
    pairs: list[PairReport] = []
    for left in range(n_annotators):
        for right in range(left + 1, n_annotators):
            shared = (label_index[:, left] >= 0) & (label_index[:, right] >= 0)
            overlap = int(shared.sum())
            if overlap < config.min_pair_overlap:
                continue
            left_labels = label_index[shared, left]
            right_labels = label_index[shared, right]
            consensus = map_labels[shared]
            kappa_pair = cohen_kappa(left_labels, right_labels, n_classes)
            kappa_left = cohen_kappa(left_labels, consensus, n_classes)
            kappa_right = cohen_kappa(right_labels, consensus, n_classes)
            excess = kappa_pair - max(kappa_left, kappa_right)
            joint_wrong = int(np.sum((left_labels == right_labels) & (left_labels != consensus)))
            both_differ = int(np.sum((left_labels != consensus) & (right_labels != consensus)))
            wrong_agreement = (joint_wrong / both_differ) if both_differ else 0.0
            score = config.collusion_excess_weight * max(0.0, excess)
            score += config.collusion_wrong_weight * wrong_agreement
            flagged = (
                kappa_pair >= config.collusion_min_kappa
                and excess >= config.collusion_min_excess
                and joint_wrong >= config.collusion_min_joint_wrong
                and score >= config.collusion_min_score
            )
            severity: str | None = None
            if flagged:
                strong = score >= config.collusion_high_score and kappa_pair >= config.collusion_high_kappa
                severity = "high" if strong else "medium"
            first, second = annotator_ids[left], annotator_ids[right]
            if second < first:
                first, second = second, first
                kappa_left, kappa_right = kappa_right, kappa_left
            pairs.append(
                PairReport(
                    annotator_a=first,
                    annotator_b=second,
                    overlap=overlap,
                    kappa=kappa_pair,
                    kappa_a_consensus=kappa_left,
                    kappa_b_consensus=kappa_right,
                    excess=excess,
                    joint_wrong=joint_wrong,
                    both_differ=both_differ,
                    wrong_agreement=wrong_agreement,
                    score=score,
                    flagged=flagged,
                    severity=severity,
                )
            )
    pairs.sort(key=lambda pair: (not pair.flagged, -pair.score, pair.annotator_a, pair.annotator_b))
    return pairs
