"""How much of each selected name to hold, given a risk model.

Selection says which names are in the book and on which side; weighting sizes
them. The two are separate axes because they fail differently: a good ranking
weighted badly concentrates into whichever name the covariance estimate
happened to call safest, and an equal-weighted book of the wrong names is
merely flat.

All rules here are PATH-INDEPENDENT -- each decision is sized from that
decision's forecast and risk model alone, with no reference to the portfolio
currently held. The cost-aware mean-variance allocator in
:mod:`stable_finance.allocation` is the path-dependent alternative, and the
difference between them is exactly the value of trading off turnover against
edge. Legs are sized independently and scaled so a long-short book is dollar
neutral at the requested gross exposure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["WEIGHTING_RULES", "weight_book"]

#: The path-independent weighting rules a backtest configuration may name.
#: ``mean_variance`` is handled by the cost-aware allocator instead.
WEIGHTING_RULES = ("equal", "inverse_vol", "min_var", "erc", "rank")


def _equal(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.ones(len(score), dtype=np.float64)


def _inverse_vol(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    volatility = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    floor = volatility[volatility > 0].min() if (volatility > 0).any() else 1.0
    return 1.0 / np.maximum(volatility, floor * 1e-6)


def _min_var(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimum-variance weights within the leg, clipped to the long side.

    ``Sigma^-1 1`` is the unconstrained minimum-variance solution. Its negative
    entries would place a short inside the long leg -- a position the selection
    rule explicitly did not ask for -- so they are clipped. Clipping rather
    than solving the constrained problem keeps this a closed form; the fallback
    to inverse-volatility covers the case where clipping empties the leg.
    """
    try:
        raw = np.linalg.solve(sigma, np.ones(len(score)))
    except np.linalg.LinAlgError:
        return _inverse_vol(sigma, score)
    clipped = np.maximum(raw, 0.0)
    return clipped if clipped.sum() > 0 else _inverse_vol(sigma, score)


def _erc(sigma: NDArray[np.float64], score: NDArray[np.float64],
         *, max_iter: int = 500, tol: float = 1e-10) -> NDArray[np.float64]:
    """Equal risk contribution, by the standard fixed-point iteration.

    Each name contributes the same share of forecast portfolio variance, so
    the risk contribution ``w_i * (Sigma w)_i`` is equal across the leg. The
    multiplicative update ``w <- sqrt(w / (Sigma w))`` followed by
    renormalization is the standard fixed point for that condition: it
    converges for a positive definite Sigma and, on a diagonal one, lands on
    ``w ~ 1/sigma`` in a single step.

    NOT ``w <- w / (Sigma w)``, which is the neighbouring update for inverse
    VARIANCE: on a diagonal model that one converges to ``w ~ 1/sigma^2``,
    which is the minimum-variance portfolio wearing this function's name.
    """
    n = len(score)
    weights = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        marginal = sigma @ weights
        if not np.all(np.isfinite(marginal)) or np.any(marginal <= 0):
            return _inverse_vol(sigma, score)
        updated = np.sqrt(weights / marginal)
        updated /= updated.sum()
        if np.max(np.abs(updated - weights)) < tol:
            return updated
        weights = updated
    return weights


def _rank(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weights proportional to the within-leg forecast rank.

    The only rule that reads the forecast's strength rather than only its
    sign, and the one control that uses no covariance model at all: if IC
    tracks this book's Sharpe but not an allocator's, the allocator is what
    broke the relationship.
    """
    order = np.argsort(np.argsort(score, kind="stable")).astype(np.float64)
    return order + 1.0


_RULES = {"equal": _equal, "inverse_vol": _inverse_vol, "min_var": _min_var,
          "erc": _erc, "rank": _rank}


def weight_book(
    name: str,
    signs: ArrayLike,
    forecast: ArrayLike,
    covariance: ArrayLike,
    *,
    gross_exposure: float = 2.0,
) -> NDArray[np.float64]:
    """Size a signed book into weights at a fixed gross exposure.

    ``signs`` is the selection's ``(decision, asset)`` output, ``covariance``
    the ``(n_assets, n_assets)`` risk model for this horizon. A long-short
    decision splits ``gross_exposure`` evenly between its legs and is dollar
    neutral by construction; a decision with only one leg carries the whole
    exposure on that side. Decisions whose book is empty get zero weights,
    which the evaluator reads as a period that was not traded.
    """
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")
    if name not in _RULES:
        raise ValueError(f"unknown weighting {name!r}; choose from {WEIGHTING_RULES}")
    rule = _RULES[name]

    side = np.asarray(signs, dtype=np.int8)
    values = np.asarray(forecast, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    if side.ndim != 2 or values.shape != side.shape:
        raise ValueError("signs and forecast must be equally shaped 2-D arrays")
    if sigma.shape != (side.shape[1], side.shape[1]):
        raise ValueError(
            f"covariance must have shape {(side.shape[1], side.shape[1])}, "
            f"got {sigma.shape}"
        )

    weights = np.zeros(side.shape, dtype=np.float64)
    for decision in range(side.shape[0]):
        legs = [s for s in (1, -1) if np.any(side[decision] == s)]
        if not legs:
            continue
        share = gross_exposure / len(legs)
        for leg in legs:
            members = np.flatnonzero(side[decision] == leg)
            score = values[decision, members]
            # A short leg's best name is its LOWEST forecast, so the rank rule
            # must see the leg oriented the way it is traded.
            raw = rule(sigma[np.ix_(members, members)], score * leg)
            total = float(np.sum(np.abs(raw)))
            if total <= 0 or not np.isfinite(total):
                raw, total = np.ones(len(members)), float(len(members))
            weights[decision, members] = leg * share * np.abs(raw) / total
    return weights
