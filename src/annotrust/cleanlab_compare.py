"""Optional comparison against cleanlab label-quality scores.

Northcutt, Jiang, and Chuang (2021) describe confident learning, which is
what cleanlab implements. This module does not import cleanlab unless a
caller asks for the comparison. The core scan does not need it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from annotrust.errors import AnnotrustError


class CleanlabUnavailableError(AnnotrustError):
    """The optional cleanlab extra is not installed."""


def compare_item_scores(
    consensus_labels: NDArray[np.int64],
    model_probs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return one cleanlab quality score per item.

    ``consensus_labels`` are integer class ids. ``model_probs`` rows align
    with those ids and should sum to 1. Install the extra with
    ``pip install 'annotrust[cleanlab]'``.
    """
    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError as exc:
        raise CleanlabUnavailableError(
            "cleanlab is not installed. Install the optional extra: pip install 'annotrust[cleanlab]'"
        ) from exc
    scores = get_label_quality_scores(consensus_labels, model_probs)
    return np.asarray(scores, dtype=np.float64)
