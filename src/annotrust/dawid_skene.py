"""Dawid-Skene EM for categorical labels from several annotators.

Dawid, A. P. and Skene, A. M. (1979). Maximum likelihood estimation of
observer error-rates using the EM algorithm. Applied Statistics, 28(1), 20-28.

The implementation follows that EM. The E-step is the posterior of the latent
class given every annotator's label. The M-step updates the class prior and
each annotator's confusion matrix, with Dirichlet smoothing so an unseen cell
is not a hard zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


@dataclass(frozen=True)
class DawidSkeneResult:
    """Fitted consensus and per-annotator confusion.

    ``posterior`` is ``(n_items, n_classes)``. ``confusion[k, i, j]`` is
    P(annotator k emits class j | true class i). Rows sum to 1.
    """

    posterior: NDArray[np.float64]
    prior: NDArray[np.float64]
    confusion: NDArray[np.float64]
    iterations: int
    log_likelihood: float


def fit_dawid_skene(
    label_index: NDArray[np.int64],
    n_classes: int,
    *,
    max_iter: int = 50,
    tol: float = 1e-5,
    smoothing: float = 0.05,
) -> DawidSkeneResult:
    """Run EM from a majority-vote initialization.

    ``label_index`` is ``(n_items, n_annotators)`` with ``-1`` for a missing
    label. Every column should contain at least one observed label.
    """
    if label_index.ndim != 2:
        raise ValueError("label_index must be 2-D")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    n_items, n_annotators = label_index.shape
    if n_items == 0 or n_annotators == 0:
        raise ValueError("label_index is empty")

    posterior = _majority_posterior(label_index, n_classes, smoothing)
    previous_ll = -np.inf
    confusion = np.zeros((n_annotators, n_classes, n_classes), dtype=np.float64)
    prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    log_likelihood = float("-inf")
    iterations = 0
    for iteration in range(1, max_iter + 1):
        confusion, prior = _m_step(label_index, posterior, smoothing)
        posterior, log_likelihood = _e_step(label_index, confusion, prior)
        iterations = iteration
        if abs(log_likelihood - previous_ll) < tol:
            break
        previous_ll = log_likelihood
    return DawidSkeneResult(
        posterior=posterior,
        prior=prior,
        confusion=confusion,
        iterations=iterations,
        log_likelihood=log_likelihood,
    )


def _majority_posterior(
    label_index: NDArray[np.int64],
    n_classes: int,
    smoothing: float,
) -> NDArray[np.float64]:
    n_items = label_index.shape[0]
    counts = np.full((n_items, n_classes), smoothing, dtype=np.float64)
    for annotator in range(label_index.shape[1]):
        observed = label_index[:, annotator]
        valid = np.flatnonzero(observed >= 0)
        if valid.size == 0:
            continue
        counts[valid, observed[valid]] += 1.0
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


def _m_step(
    label_index: NDArray[np.int64],
    posterior: NDArray[np.float64],
    smoothing: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_classes = posterior.shape[1]
    prior = posterior.mean(axis=0)
    prior = np.maximum(prior, 1e-12)
    prior = prior / prior.sum()

    n_annotators = label_index.shape[1]
    confusion = np.empty((n_annotators, n_classes, n_classes), dtype=np.float64)
    for annotator in range(n_annotators):
        observed = label_index[:, annotator]
        valid = observed >= 0
        weights = posterior[valid]
        chosen = observed[valid]
        accumulated = np.full((n_classes, n_classes), smoothing, dtype=np.float64)
        for label in range(n_classes):
            mask = chosen == label
            if np.any(mask):
                accumulated[:, label] += weights[mask].sum(axis=0)
        row_sums = accumulated.sum(axis=1, keepdims=True)
        empty = row_sums.squeeze(-1) <= 0
        accumulated = accumulated / np.maximum(row_sums, 1e-12)
        if np.any(empty):
            accumulated[empty] = 1.0 / n_classes
        confusion[annotator] = accumulated
    return confusion, prior.astype(np.float64)


def _e_step(
    label_index: NDArray[np.int64],
    confusion: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    n_items, n_annotators = label_index.shape
    n_classes = confusion.shape[1]
    log_posterior = np.broadcast_to(np.log(np.clip(prior, 1e-12, None)), (n_items, n_classes)).copy()
    for annotator in range(n_annotators):
        observed = label_index[:, annotator]
        valid = np.flatnonzero(observed >= 0)
        if valid.size == 0:
            continue
        chosen = observed[valid]
        log_emission = np.log(np.clip(confusion[annotator][:, chosen], 1e-12, 1.0)).T
        log_posterior[valid] += log_emission
    item_ll = np.asarray(logsumexp(log_posterior, axis=1), dtype=np.float64)
    refreshed = np.exp(log_posterior - item_ll[:, None])
    refreshed = np.clip(refreshed, 0.0, None)
    refreshed /= np.maximum(refreshed.sum(axis=1, keepdims=True), 1e-12)
    return refreshed, float(item_ll.sum())
