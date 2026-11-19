"""Invalid model probabilities must not reach scoring or normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annotrust.load import prepare


@pytest.mark.parametrize("bad", [-0.2, float("inf"), float("-inf")])
def test_invalid_probability_vector_is_ignored(bad: float) -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["yes", "yes", "no", "no"],
            "model_prob_no": [bad, bad, 0.8, 0.8],
            "model_prob_yes": [1.0, 1.0, 0.2, 0.2],
        }
    )
    result = prepare(frame)
    assert result.model_probs is not None
    assert np.isnan(result.model_probs[result.item_ids.index("i1")]).all()
    np.testing.assert_allclose(result.model_probs[result.item_ids.index("i2")], [0.8, 0.2])
    assert any("Ignored invalid model_prob_* vector" in warning for warning in result.warnings)


def test_zero_mass_probability_vector_is_ignored() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["a", "b", "a", "b"],
            "label": ["yes", "yes", "no", "no"],
            "model_prob_no": [0.0, 0.0, 0.7, 0.7],
            "model_prob_yes": [0.0, 0.0, 0.3, 0.3],
        }
    )
    result = prepare(frame)
    assert result.model_probs is not None
    assert np.isnan(result.model_probs[result.item_ids.index("i1")]).all()
    np.testing.assert_allclose(result.model_probs[result.item_ids.index("i2")], [0.7, 0.3])
