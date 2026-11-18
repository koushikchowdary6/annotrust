"""Dawid-Skene recovers a majority and a confusion row."""

from __future__ import annotations

import numpy as np

from annotrust.dawid_skene import _m_step, fit_dawid_skene


def _panel(n_items: int, n_honest: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    truth = np.arange(n_items) % n_classes
    labels = np.full((n_items, n_honest + 1), -1, dtype=np.int64)
    for annotator in range(n_honest):
        labels[:, annotator] = truth
    labels[:, -1] = (truth + 1) % n_classes
    return labels, truth


def test_majority_survives_one_systematically_wrong_annotator() -> None:
    labels, truth = _panel(n_items=90, n_honest=4, n_classes=3)
    fitted = fit_dawid_skene(labels, 3, max_iter=30)
    estimate = fitted.posterior.argmax(axis=1)
    assert float(np.mean(estimate == truth)) > 0.95
    assert np.allclose(fitted.posterior.sum(axis=1), 1.0)
    bad = fitted.confusion[-1]
    assert float(np.mean(np.diag(bad))) < 0.2
    honest = fitted.confusion[0]
    assert float(np.mean(np.diag(honest))) > 0.9
    assert fitted.iterations >= 1
    assert np.isfinite(fitted.log_likelihood)


def test_confusion_row_names_the_flip_a_single_accuracy_cannot() -> None:
    labels, truth = _panel(n_items=60, n_honest=5, n_classes=3)
    fitted = fit_dawid_skene(labels, 3)
    row = fitted.confusion[-1, 0]
    assert int(np.argmax(row)) == 1
    accuracy = float(np.mean(labels[:, -1] == truth))
    assert accuracy < 0.1


def test_zero_smoothing_empty_row_falls_back_to_uniform() -> None:
    labels = np.array([[0, 0], [1, 1]], dtype=np.int64)
    posterior = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    confusion, _prior = _m_step(labels, posterior, smoothing=0.0)
    assert confusion.shape == (2, 3, 3)
    assert np.allclose(confusion[:, 2], 1.0 / 3.0)


def test_rejects_a_degenerate_matrix() -> None:
    import pytest

    with pytest.raises(ValueError):
        fit_dawid_skene(np.zeros((0, 1), dtype=np.int64), 2)
    with pytest.raises(ValueError):
        fit_dawid_skene(np.array([[0]], dtype=np.int64), 1)
