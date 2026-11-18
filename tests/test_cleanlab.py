"""The cleanlab extra is optional and stays unimported until asked."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from annotrust.cleanlab_compare import CleanlabUnavailableError, compare_item_scores


def test_missing_cleanlab_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cleanlab", None)
    monkeypatch.setitem(sys.modules, "cleanlab.rank", None)
    with pytest.raises(CleanlabUnavailableError, match="optional extra"):
        compare_item_scores(np.array([0, 1]), np.array([[0.2, 0.8], [0.7, 0.3]]))


def test_compare_uses_the_installed_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    rank = types.ModuleType("cleanlab.rank")

    def fake_scores(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
        assert labels.tolist() == [1, 0]
        assert probs.shape == (2, 2)
        return np.array([0.2, 0.9])

    rank.get_label_quality_scores = fake_scores  # type: ignore[attr-defined]
    package = types.ModuleType("cleanlab")
    package.rank = rank  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cleanlab", package)
    monkeypatch.setitem(sys.modules, "cleanlab.rank", rank)
    scores = compare_item_scores(np.array([1, 0], dtype=np.int64), np.array([[0.1, 0.9], [0.8, 0.2]]))
    assert scores.tolist() == [0.2, 0.9]
