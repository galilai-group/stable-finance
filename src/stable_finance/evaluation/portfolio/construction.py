"""Portfolio construction: from a forecast and a risk model to target weights.

In practice this is one step, done one of two ways. A sort picks which names
to hold and on which side, then a simple rule sizes them -- the decile
portfolios of the academic literature. An optimizer does both at once from the
forecast, the risk model and a risk aversion. Both live here, and both are
frictionless: what trading costs is the trades stage's question.

SELECTION picks which side of the book each name is on. A forecast orders the
cross-section; it does not say how much of that order to trade. Holding the
whole cross-section long-short diversifies best and dilutes the signal most;
holding only the tails concentrates the signal and the risk. That trade-off is
a decision the model does not make, so it is an axis.

Selection returns a sign array on the ``(decision, asset)`` grid -- ``+1``
long, ``-1`` short, ``0`` not held -- which an allocator then sizes. Cuts are
taken per decision on the names that survived the universe screen, so a
tighter universe changes which names are in the tails rather than how many.

WEIGHTING sizes the selected names, given a risk model. Selection and
weighting stay separate configuration axes, even in one module, because they
fail differently: a good ranking weighted badly concentrates into whichever
name the covariance estimate happened to call safest, and an equal-weighted
book of the wrong names is merely flat.

All weighting rules are PATH-INDEPENDENT -- each decision is sized from that
decision's forecast and risk model alone, with no reference to the portfolio
currently held. The cost-aware rebalance in
:mod:`stable_finance.evaluation.trades.rebalance` is the path-dependent
alternative, and the difference between them is exactly the value of trading
off turnover against edge. Legs are sized independently and scaled so a
long-short book is dollar neutral at the requested gross exposure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = [
    "SELECTION_REQUIRES",
    "SELECTION_RULES",
    "WEIGHTING_REQUIRES",
    "WEIGHTING_RULES",
    "mean_variance_target",
    "select_book",
    "tail_selection",
    "weight_book",
]


# ── selection: which names, on which side ──────────────────────────────────

def tail_selection(
    forecast: ArrayLike,
    eligible: ArrayLike,
    *,
    fraction: float,
    long_only: bool = False,
) -> NDArray[np.int8]:
    """Long the top ``fraction`` of each decision, short the bottom.

    The cut is a fraction of that decision's eligible cross-section, not an
    absolute forecast threshold: forecasts from a rank target, a raw-return
    probe and a trained head live on entirely different scales, and a fraction
    is the only cut that means the same thing for all three.
    """
    if not 0.0 < fraction <= 0.5:
        raise ValueError("fraction must lie in (0, 0.5]")
    values = np.asarray(forecast, dtype=np.float64)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    if values.shape != usable.shape or values.ndim != 2:
        raise ValueError("forecast and eligible must be equally shaped 2-D arrays")

    signs = np.zeros(values.shape, dtype=np.int8)
    for decision in range(values.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) < 2:
            continue
        count = max(int(np.floor(len(row) * fraction)), 1)
        # A leg cannot exceed half the cross-section, or the long and short
        # tails would overlap and a name would be held on both sides.
        count = min(count, len(row) // 2)
        if count == 0:
            continue
        order = row[np.argsort(values[decision, row], kind="stable")]
        signs[decision, order[-count:]] = 1
        if not long_only:
            signs[decision, order[:count]] = -1
    return signs


def select_book(
    name: str, forecast: ArrayLike, eligible: ArrayLike
) -> NDArray[np.int8]:
    """Dispatch to one named selection rule; see :data:`SELECTION_RULES`.

    ``all`` holds every eligible name, signed by whether its forecast is above
    or below that decision's cross-sectional median -- the dollar-neutral book
    a reader would write down from the ranking alone.
    """
    values = np.asarray(forecast, dtype=np.float64)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    if name == "all":
        signs = np.zeros(values.shape, dtype=np.int8)
        for decision in range(values.shape[0]):
            row = np.flatnonzero(usable[decision])
            if len(row) < 2:
                continue
            centre = np.median(values[decision, row])
            signs[decision, row] = np.where(values[decision, row] >= centre, 1, -1)
        return signs
    if name == "quintile":
        return tail_selection(values, usable, fraction=0.2)
    if name == "decile":
        return tail_selection(values, usable, fraction=0.1)
    if name == "decile_long_only":
        return tail_selection(values, usable, fraction=0.1, long_only=True)
    raise ValueError(f"unknown selection {name!r}; choose from {SELECTION_RULES}")


#: The selection rules a backtest configuration may name.
SELECTION_RULES = ("all", "quintile", "decile", "decile_long_only")

#: What each named rule needs of the market: nothing, since each cuts the
#: forecast ranking alone. ``threshold_selection`` and ``persistent_selection``
#: compare the forecast with a cost and so need ``half_spread`` when given
#: one; they are not configuration axes.
SELECTION_REQUIRES: dict[str, tuple[str, ...]] = {name: () for name in SELECTION_RULES}


# ── weighting: how much of each ─────────────────────────────────────────────

#: The path-independent weighting rules a backtest configuration may name.
#: ``mean_variance`` is handled by the cost-aware allocator instead.
WEIGHTING_RULES = ("equal", "inverse_vol", "min_var", "erc", "rank")

#: What each rule needs of the market. None reads it today; a rule sized off
#: the bar range (Parkinson or Garman-Klass volatility, say) would declare
#: ``("high", "low")`` here, or ``("open", "high", "low", "close")``, and the
#: backtest would refuse a market without them.
WEIGHTING_REQUIRES: dict[str, tuple[str, ...]] = {name: () for name in WEIGHTING_RULES}


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


def mean_variance_target(
    forecast: ForwardReturns,
    covariance: ArrayLike,
    *,
    risk_aversion: float = 1.0,
) -> PortfolioWeights:
    """The frictionless mean-variance holding, ``Sigma^-1 mu / risk_aversion``.

    What the portfolio would hold if trading were free. ``covariance`` has
    shape ``(n_horizons, n_assets, n_assets)`` and must be invertible (an
    estimate from :func:`~stable_finance.evaluation.portfolio.risk.estimate_covariance`
    is). A missing forecast is a zero expected return. How far to actually
    move toward this target, given costs, is the trades stage's decision
    (:func:`~stable_finance.evaluation.trades.rebalance.rebalance`).
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    mu = np.nan_to_num(np.asarray(forecast.values, dtype=np.float64), nan=0.0)
    sigma = np.asarray(covariance, dtype=np.float64)
    if sigma.shape != (mu.shape[2], mu.shape[1], mu.shape[1]):
        raise ValueError(
            f"covariance must have shape {(mu.shape[2], mu.shape[1], mu.shape[1])}, "
            f"got {sigma.shape}")
    target = np.empty_like(mu)
    for horizon in range(mu.shape[2]):
        target[:, :, horizon] = np.linalg.solve(sigma[horizon], mu[:, :, horizon].T).T / risk_aversion
    return PortfolioWeights(target, forecast.decisions, forecast.assets, forecast.horizons)
