"""Risk management: which names a portfolio may hold, and how they co-move.

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

THE UNIVERSE SCREENS are the other half of risk management. Spread costs are
heavy-tailed across a cross-section, so how tight a tradable universe to hold
is one of the decisions that moves a Sharpe ratio most and is dictated by the
model least. A screen is a boolean mask on the ``(decision, asset)`` grid,
applied before selection, so it composes with any selection rule and any
allocator. Which names the data can price at all -- quoted, and present often
enough to estimate -- is the dataset's to say, in
:mod:`stable_finance.dataset.universe`; the screens here start from that.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.universe import tradable

from stable_finance.evaluation.embeddings.contract import Embeddings
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = [
    "RISK_MODELS",
    "UNIVERSE_REQUIRES",
    "UNIVERSE_SCREENS",
    "apply_screen",
    "diagonal_covariance",
    "embedding_covariance",
    "estimate_risk",
    "ledoit_wolf_covariance",
    "liquidity_screen",
    "nonlinear_shrinkage_covariance",
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
    :func:`stable_finance.dataset.universe.presence_screen` is what keeps that bias small; without one,
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


_SQRT5 = np.sqrt(5.0)


def _analytical_shrinkage(
    centered: NDArray[np.float64], n_effective: int
) -> NDArray[np.float64]:
    """Ledoit and Wolf's (2020) closed form on an already centred panel.

    A line-by-line port of the authors' ``analytical_shrinkage.m``: the
    sample spectrum is smoothed with an Epanechnikov kernel whose bandwidth
    is proportional to each eigenvalue (``h = n^(-1/3)``), and each eigenvalue
    is replaced by the oracle value that the kernel estimates of the spectral
    density and its Hilbert transform imply (their equations 4.3 and C.4-C.8).
    Eigenvectors are the sample's, untouched.
    """
    p = centered.shape[1]
    n = n_effective
    sample = centered.T @ centered / n
    eigenvalues, eigenvectors = np.linalg.eigh((sample + sample.T) / 2)
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    # Only the top min(p, n) eigenvalues are informative; when p > n the rest
    # are structurally zero and get their own closed form below. A relative
    # floor keeps the kernel bandwidths positive where rounding (or the
    # zero-filled panel) leaves an informative eigenvalue at or below zero.
    lam = eigenvalues[max(0, p - n):]
    largest = float(lam.max())
    if largest <= 0:
        raise ValueError("returns have no variance")
    lam = np.maximum(lam, 1e-12 * largest)

    h = n ** (-1.0 / 3.0)
    bandwidth = h * lam[np.newaxis, :]  # H[i, j] = h * lambda_j
    x = (lam[:, np.newaxis] - lam[np.newaxis, :]) / bandwidth
    density = (3 / 4 / _SQRT5) * np.mean(
        np.maximum(1 - x**2 / 5, 0.0) / bandwidth, axis=1
    )
    # The kernel's Hilbert transform has a log singularity at |x| = sqrt(5),
    # where its (1 - x^2/5) factor vanishes; the reference code drops the log
    # term there, which is what zeroing it does.
    edge = np.isclose(np.abs(x), _SQRT5)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.abs((_SQRT5 - x) / (_SQRT5 + x)))
    log_term = np.where(edge, 0.0, log_term)
    hilbert = (-3 / 10 / np.pi) * x + (3 / 4 / _SQRT5 / np.pi) * (
        1 - x**2 / 5
    ) * log_term
    hilbert = np.mean(hilbert / bandwidth, axis=1)

    ratio = p / n
    if p <= n:
        shrunk = lam / (
            (np.pi * ratio * lam * density) ** 2
            + (1 - ratio - np.pi * ratio * lam * hilbert) ** 2
        )
    else:
        hilbert_null = (
            (1 / np.pi)
            * (
                3 / 10 / h**2
                + 3 / 4 / _SQRT5 / h * (1 - 1 / 5 / h**2)
                * np.log((1 + _SQRT5 * h) / (1 - _SQRT5 * h))
            )
            * np.mean(1 / lam)
        )
        null = 1 / (np.pi * (p - n) / n * hilbert_null)
        informative = lam / (np.pi**2 * lam**2 * (density**2 + hilbert**2))
        shrunk = np.concatenate([np.full(p - n, null), informative])
    return (eigenvectors * shrunk) @ eigenvectors.T


def nonlinear_shrinkage_covariance(
    history: ForwardReturns, *, min_samples: int = 2
) -> NDArray[np.float64]:
    """Sample eigenvectors with each eigenvalue shrunk by its own amount.

    Linear shrinkage (:func:`ledoit_wolf_covariance`) pulls every sample
    eigenvalue toward the grand mean by ONE data-chosen intensity, which is
    the best a single number can do but still too little for the extreme
    eigenvalues and too much for the bulk. Nonlinear shrinkage keeps the
    sample eigenvectors and replaces each eigenvalue by an estimate of the
    variance the portfolio along that eigenvector will actually realize, so
    the correction is largest exactly where estimation error is. Ledoit and
    Wolf (2020), "Analytical nonlinear shrinkage of large-dimensional
    covariance matrices", Annals of Statistics 48(5), give that estimate in
    closed form -- a kernel estimate of the sample spectral density and its
    Hilbert transform -- so it costs one eigendecomposition rather than the
    numerical inversion of the earlier QuEST estimator, and it is defined
    when there are more assets than decisions.

    The panel is centred with each asset's own mean, so the effective sample
    size is one less than the number of decisions, as in the authors' code
    for demeaned data. The closed form needs at least 12 of them: below that
    the kernel bandwidth ``n^(-1/3)`` is too wide for the null-eigenvalue
    term to be defined.

    MISSING OBSERVATIONS ARE ZERO-FILLED AFTER CENTERING, which biases a
    sparsely observed asset's covariance toward zero. The presence screen in
    :func:`stable_finance.dataset.universe.presence_screen` is what keeps that bias small; without one,
    a name quoted on a handful of decisions would look riskless.
    """
    n_decisions, n_assets, n_horizons = history.values.shape
    n_effective = n_decisions - 1
    if n_effective < 12:
        raise ValueError(
            f"nonlinear shrinkage needs at least 13 decisions; got {n_decisions}"
        )
    out = np.empty((n_horizons, n_assets, n_assets), dtype=np.float64)
    for horizon in range(n_horizons):
        _, centered, _ = _observed_moments(
            history.values[:, :, horizon], min_samples
        )
        out[horizon] = _project_psd(_analytical_shrinkage(centered, n_effective))
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

    if name == "diagonal":
        return diagonal_covariance(history, min_samples=min_samples)
    if name == "shrunk_sample":
        return estimate_covariance(
            history, shrinkage=shrinkage, min_samples=max(min_samples, 2)
        )
    if name == "ledoit_wolf":
        return ledoit_wolf_covariance(history, min_samples=min_samples)
    if name == "nonlinear_shrinkage":
        return nonlinear_shrinkage_covariance(history, min_samples=min_samples)
    if name == "embedding":
        if embeddings is None:
            raise ValueError("the 'embedding' risk model needs embeddings")
        return embedding_covariance(
            history, embeddings, shrinkage=shrinkage, min_samples=min_samples
        )
    raise ValueError(f"unknown risk model {name!r}; choose from {RISK_MODELS}")


#: The risk models a backtest configuration may name.
RISK_MODELS = (
    "diagonal", "shrunk_sample", "ledoit_wolf", "nonlinear_shrinkage", "embedding",
)

#: What each risk model needs of the market: nothing, since all of them are
#: estimated from returns (``embedding`` also needs the embeddings, which are
#: not market data and are checked where they are used).
RISK_MODEL_REQUIRES: dict[str, tuple[str, ...]] = {name: () for name in RISK_MODELS}


def estimate_covariance(
    returns: ForwardReturns,
    *,
    shrinkage: float = 0.1,
    min_samples: int = 2,
) -> NDArray[np.float64]:
    """Per-horizon asset covariance estimated from a realized-return history.

    Decisions are the samples. Each pair of assets is estimated over the
    decisions where both are finite, the result is shrunk toward its diagonal
    by ``shrinkage``, and negative eigenvalues are clipped so the allocation
    objective is convex. The returned array has shape
    ``(n_horizons, n_assets, n_assets)``.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    n_decisions, n_assets, n_horizons = returns.values.shape
    covariances = np.empty((n_horizons, n_assets, n_assets), dtype=np.float64)
    for horizon in range(n_horizons):
        x = returns.values[:, :, horizon]
        finite = np.isfinite(x)
        counts = finite.sum(axis=0)
        if (counts < min_samples).any():
            asset = returns.assets[int(np.argmin(counts))]
            raise ValueError(
                f"asset {asset!r} has {int(counts.min())} finite returns at "
                f"horizon {returns.horizons[horizon]!r}; need {min_samples}"
            )
        mean = np.where(finite, x, 0.0).sum(axis=0) / counts
        centered = np.where(finite, x - mean, 0.0)
        pairs = finite.T.astype(np.float64) @ finite.astype(np.float64)
        sample = (centered.T @ centered) / np.maximum(pairs - 1, 1)
        sample[pairs < 2] = 0.0
        diagonal = np.diag(np.diag(sample))
        shrunk = (1 - shrinkage) * sample + shrinkage * diagonal
        eigenvalues, eigenvectors = np.linalg.eigh((shrunk + shrunk.T) / 2)
        if eigenvalues.max() <= 0:
            raise ValueError(
                f"returns at horizon {returns.horizons[horizon]!r} have no variance"
            )
        # A tiny floor keeps the risk term strictly convex so that assets with
        # (numerically) zero variance cannot receive unbounded weight.
        eigenvalues = np.maximum(eigenvalues, 1e-8 * eigenvalues.max())
        covariances[horizon] = (eigenvectors * eigenvalues) @ eigenvectors.T
    return covariances


# ── universe screens ──────────────────────────────────────────────────────────

def liquidity_screen(
    half_spread: ArrayLike, *, keep: float
) -> NDArray[np.bool_]:
    """Keep the tightest ``keep`` fraction of each decision's cross-section.

    The cut is a CROSS-SECTIONAL FRACTION rather than an absolute spread in
    basis points, because the level of spreads moves by an order of magnitude
    across a long history -- an absolute threshold would screen out almost
    everything in a 2008 and almost nothing in a 2019, making the universe a
    proxy for the calendar. A fraction keeps the screen's meaning fixed.
    """
    if not 0.0 < keep <= 1.0:
        raise ValueError("keep must lie in (0, 1]")
    spread = np.asarray(half_spread, dtype=np.float64)
    usable = tradable(spread)
    if keep == 1.0:
        return usable
    mask = np.zeros_like(usable)
    for decision in range(spread.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) == 0:
            continue
        # At least one name survives any non-empty cross-section: a screen
        # that can empty a decision would silently turn into a missing period
        # rather than a tighter universe.
        count = max(int(np.floor(len(row) * keep)), 1)
        tightest = row[np.argsort(spread[decision, row], kind="stable")[:count]]
        mask[decision, tightest] = True
    return mask


def apply_screen(name: str, half_spread: ArrayLike | None,
                 *, shape: tuple[int, int] | None = None) -> NDArray[np.bool_]:
    """Dispatch to one named universe screen; see :data:`UNIVERSE_SCREENS`.

    ``half_spread=None`` means the market has no quotes at all, which is not
    the same as every name being unquoted. Only ``all`` is defined then, and it
    admits every name on a ``shape`` grid: with no quote to say a name could
    not trade, the frictionless backtest that is the only one possible trades
    whatever has a forecast and an outcome.
    """
    if half_spread is None:
        if UNIVERSE_REQUIRES.get(name):
            from stable_finance.dataset.schema import MissingRolesError

            raise MissingRolesError(
                f"universe={name!r} screens on the quoted half-spread, and the "
                "market data has none")
        if name != "all":
            raise ValueError(f"unknown universe {name!r}; choose from {UNIVERSE_SCREENS}")
        if shape is None:
            raise ValueError("pass shape when there is no half_spread")
        return np.ones(shape, dtype=bool)
    if name == "all":
        return tradable(half_spread)
    if name == "tight_50pct":
        return liquidity_screen(half_spread, keep=0.5)
    if name == "tight_25pct":
        return liquidity_screen(half_spread, keep=0.25)
    raise ValueError(f"unknown universe {name!r}; choose from {UNIVERSE_SCREENS}")


#: The universe screens a backtest configuration may name.
UNIVERSE_SCREENS = ("all", "tight_50pct", "tight_25pct")

#: What each screen needs of the market. ``all`` keeps the quoted names when
#: there are quotes and every name when there are none; the liquidity screens
#: rank on the quoted half-spread and cannot be formed without it.
UNIVERSE_REQUIRES: dict[str, tuple[str, ...]] = {
    "all": (), "tight_50pct": ("half_spread",), "tight_25pct": ("half_spread",),
}
