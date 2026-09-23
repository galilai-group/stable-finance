"""Covariance forecasts: the risk half of a portfolio decision.

A forecast says which names to prefer. A risk model says how much of each to
hold, and it is a genuinely separate choice: the same ranking allocated under
a diagonal model and under a correlated one produces different portfolios,
different turnover, and different Sharpe ratios. Keeping the estimators here
rather than inside an allocator is what lets a study vary one while holding
the other fixed.

Every estimator returns ``(n_horizons, n_assets, n_assets)`` on the asset axis
of the history it was given, and every one is projected to the positive
semidefinite cone so the allocation objective stays convex. Estimation uses
only the decisions in the supplied history, so a caller that passes a prior
period gets a strictly out-of-sample risk model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stable_finance.data import Embeddings, ForwardReturns

__all__ = [
    "RISK_MODELS",
    "diagonal_covariance",
    "embedding_covariance",
    "estimate_risk",
    "ledoit_wolf_covariance",
]


def _project_psd(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize and clip negative eigenvalues, keeping a strict floor.

    Pairwise-complete estimates and shrunk correlation forecasts are both
    capable of returning an indefinite matrix. A tiny positive floor -- rather
    than a clip at zero -- keeps the risk term strictly convex, so an asset
    whose estimated variance is numerically zero cannot attract unbounded
    weight.
    """
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues.max())
    if largest <= 0:
        raise ValueError("covariance estimate has no positive variance")
    eigenvalues = np.maximum(eigenvalues, 1e-8 * largest)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _observed_moments(
    x: NDArray[np.float64], min_samples: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-asset means, the zero-filled centered panel, and finite counts."""
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    if (counts < min_samples).any():
        raise ValueError(
            f"an asset has {int(counts.min())} finite returns; need {min_samples}"
        )
    mean = np.where(finite, x, 0.0).sum(axis=0) / counts
    centered = np.where(finite, x - mean, 0.0)
    return mean, centered, counts.astype(np.float64)


def diagonal_covariance(
    history: ForwardReturns, *, min_samples: int = 2
) -> NDArray[np.float64]:
    """Per-asset variance with every correlation forced to zero.

    The honest floor of risk modelling: it assumes nothing it has not measured
    directly. Inverse-volatility and equal-risk-contribution allocators both
    reduce to ``w ~ 1/sigma`` under it, and minimum variance to ``w ~
    1/sigma^2``, which is why those pairings collapse into one option rather
    than three.
    """
    n_decisions, n_assets, n_horizons = history.values.shape
    out = np.zeros((n_horizons, n_assets, n_assets), dtype=np.float64)
    for horizon in range(n_horizons):
        _, centered, counts = _observed_moments(
            history.values[:, :, horizon], min_samples
        )
        variance = (centered * centered).sum(axis=0) / np.maximum(counts - 1, 1)
        largest = float(variance.max())
        if largest <= 0:
            raise ValueError("returns have no variance")
        out[horizon] = np.diag(np.maximum(variance, 1e-8 * largest))
    return out


def ledoit_wolf_covariance(
    history: ForwardReturns, *, min_samples: int = 2
) -> NDArray[np.float64]:
    """Sample covariance shrunk toward a scaled identity by the LW intensity.

    The standard fix for a covariance estimated on fewer decisions than it has
    assets, where the sample matrix is singular and its extreme eigenvalues
    are mostly estimation error. The shrinkage intensity is chosen by the
    data, not by the caller -- that is the whole point of the estimator, and
    it is why this is a different option from ``shrunk_sample`` rather than a
    setting of it.

    MISSING OBSERVATIONS ARE ZERO-FILLED AFTER CENTERING, which biases a
    sparsely observed asset's covariance toward zero. The presence screen in
    :mod:`stable_finance.universe` is what keeps that bias small; without one,
    a name quoted on a handful of decisions would look riskless.
    """
    from sklearn.covariance import LedoitWolf

    n_decisions, n_assets, n_horizons = history.values.shape
    out = np.empty((n_horizons, n_assets, n_assets), dtype=np.float64)
    for horizon in range(n_horizons):
        _, centered, _ = _observed_moments(
            history.values[:, :, horizon], min_samples
        )
        estimate = LedoitWolf(assume_centered=True).fit(centered).covariance_
        out[horizon] = _project_psd(estimate)
    return out


def embedding_covariance(
    history: ForwardReturns,
    embeddings: Embeddings,
    *,
    shrinkage: float = 0.5,
    min_samples: int = 2,
) -> NDArray[np.float64]:
    """The encoder's own representation used as a correlation forecast.

    Correlations are the cosine similarities of each asset's mean embedding
    over the history window, shrunk toward the identity by ``shrinkage`` and
    rescaled by trailing volatilities. This asks whether a representation
    trained to predict returns also carries the co-movement structure a
    portfolio needs -- a claim about the embedding that no return-only
    estimator can make, and one that can just as easily come out worse than
    the sample covariance.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not np.array_equal(history.assets, embeddings.assets):
        raise ValueError("assets are not aligned")

    features = embeddings.values
    finite = np.isfinite(features)
    counts = finite.any(axis=2).sum(axis=0)
    if (counts == 0).any():
        raise ValueError("an asset has no finite embedding rows")
    with np.errstate(invalid="ignore"):
        mean_embedding = np.where(finite, features, 0.0).sum(axis=0) / np.maximum(
            finite.sum(axis=0), 1
        )
    norms = np.linalg.norm(mean_embedding, axis=1, keepdims=True)
    if not (norms > 0).all():
        raise ValueError("an asset has a zero mean embedding")
    unit = mean_embedding / norms
    correlation = np.clip(unit @ unit.T, -1.0, 1.0)
    n_assets = len(history.assets)
    correlation = (1 - shrinkage) * correlation + shrinkage * np.eye(n_assets)
    np.fill_diagonal(correlation, 1.0)

    n_horizons = history.values.shape[2]
    out = np.empty((n_horizons, n_assets, n_assets), dtype=np.float64)
    for horizon in range(n_horizons):
        _, centered, sample_counts = _observed_moments(
            history.values[:, :, horizon], min_samples
        )
        variance = (centered * centered).sum(axis=0) / np.maximum(sample_counts - 1, 1)
        sigma = np.sqrt(np.maximum(variance, 0.0))
        largest = float(sigma.max())
        if largest <= 0:
            raise ValueError("returns have no variance")
        sigma = np.maximum(sigma, 1e-8 * largest)
        out[horizon] = _project_psd(correlation * np.outer(sigma, sigma))
    return out


def estimate_risk(
    name: str,
    history: ForwardReturns,
    *,
    embeddings: Embeddings | None = None,
    shrinkage: float = 0.3,
    min_samples: int = 2,
) -> NDArray[np.float64]:
    """Dispatch to one named risk model; see :data:`RISK_MODELS`."""
    from stable_finance.allocation import estimate_covariance

    if name == "diagonal":
        return diagonal_covariance(history, min_samples=min_samples)
    if name == "shrunk_sample":
        return estimate_covariance(
            history, shrinkage=shrinkage, min_samples=max(min_samples, 2)
        )
    if name == "ledoit_wolf":
        return ledoit_wolf_covariance(history, min_samples=min_samples)
    if name == "embedding":
        if embeddings is None:
            raise ValueError("the 'embedding' risk model needs embeddings")
        return embedding_covariance(
            history, embeddings, shrinkage=shrinkage, min_samples=min_samples
        )
    raise ValueError(f"unknown risk model {name!r}; choose from {RISK_MODELS}")


#: The risk models a backtest configuration may name.
RISK_MODELS = ("diagonal", "shrunk_sample", "ledoit_wolf", "embedding")
