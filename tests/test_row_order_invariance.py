"""Regression coverage for row-order independence of annotation risk findings.

Annotation exports may arrive in arbitrary order. Reordering records must not
change the reported annotator risk classification or flagged item identities.
"""

from __future__ import annotations

import pytest

from annotrust.scan import scan
from annotrust.simulate import simulate


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reordered_annotations_preserve_risk_findings(seed: int) -> None:
    frame = simulate(seed=seed, timestamps=False).frame
    baseline = scan(frame)
    reordered = scan(frame.sample(frac=1.0, random_state=seed + 100).reset_index(drop=True))

    def annotator_findings(result: object) -> dict[str, tuple[str, str | None, tuple[tuple[str, str], ...]]]:
        return {
            row.annotator_id: (
                row.primary,
                row.severity,
                tuple(sorted((flag.kind, flag.severity) for flag in row.flags)),
            )
            for row in result.annotators  # type: ignore[attr-defined]
        }

    def item_findings(result: object) -> dict[str, tuple[str, ...]]:
        return {
            row.item_id: tuple(sorted(row.reasons))
            for row in result.items  # type: ignore[attr-defined]
            if row.reasons
        }

    assert baseline.n_annotations == reordered.n_annotations
    assert baseline.n_items == reordered.n_items
    assert annotator_findings(baseline) == annotator_findings(reordered)
    assert item_findings(baseline) == item_findings(reordered)
