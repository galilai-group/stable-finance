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
WEIGHTING_RULES = ("equal", "inverse_vol", "min_var", "min_var_qp", "erc", "hrp",
                   "rank", "signal")

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


#: How ill-conditioned a leg's covariance may be before :func:`_min_var_qp`
#: refuses it, as the ratio of its largest eigenvalue to its smallest. Beyond
#: this a linear solve keeps only a few significant digits and the minimum-
#: variance book is noise. Every risk model in
#: :mod:`~stable_finance.evaluation.portfolio.risk` floors its eigenvalues at
#: ``1e-8`` of the largest, and a principal submatrix's eigenvalues lie inside
#: the full matrix's range, so a leg cut from one sits four orders of
#: magnitude inside this limit.
_MAX_CONDITION = 1e12


def _min_var_qp(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    """The long-only minimum-variance portfolio within the leg, solved exactly.

    ``min w' Sigma w`` subject to ``w >= 0`` and ``sum w = 1``: the optimum
    that :func:`_min_var` approximates by clipping ``Sigma^-1 1``. Clipping
    zeroes the negative entries but leaves the survivors where the
    UNCONSTRAINED solution put them, which is no longer optimal once the
    short positions they were hedging against are gone. Jagannathan and Ma
    (2003) show that the no-short constraint acts as shrinkage of the
    covariance toward a structured target, which is why this book is often
    the better-behaved out of sample despite the estimation error the
    constraint does not see.

    Solved by a primal active-set method (Nocedal and Wright 2006, Algorithm
    16.3), which is exact. The working set ``A`` is the names allowed a
    nonzero weight. Each round solves the budget-constrained problem on
    ``A``, ``Sigma_AA w_A = c 1`` with ``sum w_A = 1``, and then changes
    ``A`` by exactly ONE name:

    * if stepping from the iterate to that solution would push a held name
      below zero, step only as far as the first name to reach zero -- lowest
      index on a tie -- and remove it;
    * otherwise move to the solution, which is then feasible, and readmit the
      single excluded name with the most negative KKT slack ``(Sigma w)_i -
      w' Sigma w``: the name whose marginal variance most undercuts the
      common level the held names share;
    * stop when the solution is feasible and no excluded name's slack is
      negative (to a relative ``1e-10``) -- the KKT conditions of the problem.

    The ratio test in the first case is what makes this terminate. The
    shortcut of jumping straight to each solution and dropping its most
    negative weight is NOT safe: on some two-factor covariances (a 12-name one
    is in the tests) it readmits and re-drops the same few names forever. Here every iterate is
    feasible and the variance never rises, so, barring exact ties, no working
    set comes round twice. A name usually leaves once and rarely returns, so the
    rounds are capped at ``2 n``, well clear of the ``1.3 n`` that thousands
    of random factor, block and sample covariances need at worst; hitting the
    cap raises a ``ValueError`` rather than returning an unsettled book.

    The rounds start from a warm start rather than the whole leg, because a
    long-only minimum-variance book usually holds a small fraction of its
    names, and removing the rest one solve at a time is most of the work:
    on a month of 300-name legs it is fifty times slower. The warm start
    repeatedly solves on the remaining names and drops EVERY name the
    solution shorts. The set only shrinks, so it ends within ``n`` solves,
    at a feasible book -- the start the exact rounds need. It is not the
    answer, since a name dropped early may belong in the book once the names
    it was hedging are gone; the rounds readmit it.

    The leg's covariance must be positive definite, to within
    :data:`_MAX_CONDITION`: then every working set's system is a principal
    submatrix of it, also positive definite, and no solve is singular. A
    covariance that fails the check -- two copies of one name, a NaN, a model
    with no positive variance -- raises a ``ValueError`` naming the leg's
    eigenvalue range rather than falling back to some other book; the risk
    models never produce one. A one-name leg is the whole leg, whatever its
    variance. No randomness and stable tie-breaks: the same leg always gets
    the same book.
    """
    n = len(score)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.isfinite(sigma).all():
        raise ValueError(f"min_var_qp: the {n}-name leg's covariance is not finite")
    sigma = (sigma + sigma.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(sigma)
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if not smallest > largest / _MAX_CONDITION:
        raise ValueError(
            f"min_var_qp needs a positive definite leg covariance; the {n}-name "
            f"leg's eigenvalues span [{smallest:.3g}, {largest:.3g}], singular or "
            f"conditioned worse than {_MAX_CONDITION:.0e}")

    active = np.ones(n, dtype=bool)
    while True:
        index = np.flatnonzero(active)
        raw = np.linalg.solve(sigma[np.ix_(index, index)], np.ones(len(index)))
        shorted = raw < 0
        if not shorted.any():
            break
        active[index[shorted]] = False
    weights = np.zeros(n)
    weights[index] = raw / raw.sum()
    for _ in range(2 * n):
        index = np.flatnonzero(active)
        raw = np.linalg.solve(sigma[np.ix_(index, index)], np.ones(len(index)))
        # 1' Sigma_AA^-1 1 > 0 for a positive definite Sigma_AA, and it is the
        # reciprocal of the working set's minimum variance.
        target = np.zeros(n)
        target[index] = raw / raw.sum()
        step = target - weights
        falling = index[step[index] < 0]
        if len(falling):
            ratios = weights[falling] / -step[falling]
            first = int(np.argmin(ratios))
            if ratios[first] < 1.0:
                weights = np.maximum(weights + ratios[first] * step, 0.0)
                weights[falling[first]] = 0.0
                active[falling[first]] = False
                continue
        weights = target
        marginal = sigma @ weights
        level = float(weights @ marginal)
        slack = marginal - level
        violated = np.flatnonzero(~active & (slack < -1e-10 * level))
        if len(violated) == 0:
            return weights
        active[violated[np.argmin(slack[violated])]] = True
    raise ValueError(
        f"min_var_qp: the active set did not settle within {2 * n} rounds on a "
        f"{n}-name leg; the leg's covariance is degenerate")


def _cluster_variance(sigma: NDArray[np.float64], members: NDArray[np.intp],
                      variance: NDArray[np.float64]) -> float:
    inverse = 1.0 / variance[members]
    weights = inverse / inverse.sum()
    return float(weights @ sigma[np.ix_(members, members)] @ weights)


def _hrp(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hierarchical Risk Parity (Lopez de Prado 2016, JPM 42(4):59-69).

    A minimum-variance book inverts the covariance, and with it every
    estimation error in the off-diagonal; inverse volatility ignores the
    off-diagonal entirely. HRP uses the correlation structure only to decide
    which names compete for the same risk budget, and never inverts:

    1. correlation ``rho`` from the leg's covariance, and the distance
       ``d_ij = sqrt((1 - rho_ij) / 2)``, zero for perfectly correlated names;
    2. single-linkage clustering of ``d``, by
       :func:`scipy.cluster.hierarchy.linkage` on the condensed distances;
    3. quasi-diagonalization: order the names as the dendrogram's leaves
       (:func:`scipy.cluster.hierarchy.leaves_list`), so similar names sit
       next to each other;
    4. recursive bisection: split the ordered list in halves, give each half
       a share of its parent's weight inversely proportional to its cluster
       variance -- the variance of the half held at inverse-variance weights
       -- and recurse until every name stands alone.

    Under a diagonal covariance every split is by summed inverse variance, so
    the book is exactly inverse-variance, the minimum-variance answer for
    that model. A one-name leg is the whole leg, and a two-name leg splits
    by inverse variance whatever the correlation. scipy's single linkage is
    deterministic for a given distance matrix, so the leaf order -- and the
    book -- is a deterministic function of the covariance.
    """
    # scipy ships with scikit-learn; imported here, as risk.py imports
    # sklearn, so importing the package does not pay for clustering.
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = len(score)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    variance = np.diag(sigma).astype(np.float64)
    positive = variance[np.isfinite(variance) & (variance > 0)]
    floor = positive.min() * 1e-12 if len(positive) else 1.0
    variance = np.where(np.isfinite(variance) & (variance > 0), variance, floor)
    volatility = np.sqrt(variance)
    correlation = np.clip(sigma / np.outer(volatility, volatility), -1.0, 1.0)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.sqrt(np.maximum((1.0 - correlation) / 2.0, 0.0))
    # squareform reads one triangle, so make the matrix the exact symmetric,
    # zero-diagonal, non-negative distance it assumes.
    distance = np.maximum((distance + distance.T) / 2.0, 0.0)
    np.fill_diagonal(distance, 0.0)

    link = linkage(squareform(distance, checks=False), method="single")
    order = leaves_list(link)
    weights = np.ones(n, dtype=np.float64)
    pending = [order]
    while pending:
        members = pending.pop(0)
        if len(members) < 2:
            continue
        half = len(members) // 2
        left, right = members[:half], members[half:]
        left_var = _cluster_variance(sigma, left, variance)
        right_var = _cluster_variance(sigma, right, variance)
        total = left_var + right_var
        share = right_var / total if total > 0 and np.isfinite(total) else 0.5
        weights[left] *= share
        weights[right] *= 1.0 - share
        pending.extend((left, right))
    return weights


def _rank(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weights proportional to the within-leg forecast rank.

    Reads the forecast's ordering rather than only its sign, and uses no
    covariance model at all: if IC tracks this book's Sharpe but not an
    allocator's, the allocator is what broke the relationship. Ranks are
    shift-invariant, so the centring :func:`weight_book` applies to the score
    leaves this book unchanged.
    """
    order = np.argsort(np.argsort(score, kind="stable")).astype(np.float64)
    return order + 1.0


def _signal(sigma: NDArray[np.float64], score: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weights proportional to the forecast's distance from the centre.

    The characteristic-weighted (or z-score-weighted) long-short book: each
    name is held in proportion to how far its forecast sits from that
    decision's cross-sectional median, in the direction its leg trades.
    Where :func:`_rank` keeps only the ordering, this book keeps the spacing
    too, so a forecast that is confident about a few names concentrates on
    them. It ignores the covariance entirely -- the forecast-only control
    for the risk-model-driven rules. A leg whose every name sits on the
    centre (or on the wrong side of it) falls back to equal weights.
    """
    distance = np.where(np.isfinite(score), np.maximum(score, 0.0), 0.0)
    if distance.sum() <= 0:
        return np.ones(len(score), dtype=np.float64)
    return distance


_RULES = {"equal": _equal, "inverse_vol": _inverse_vol, "min_var": _min_var,
          "min_var_qp": _min_var_qp, "erc": _erc, "hrp": _hrp, "rank": _rank,
          "signal": _signal}


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

    Each rule sees its leg's forecast CENTRED on the median of that
    decision's book (every name with a nonzero side and a finite forecast)
    and multiplied by the leg's sign, so a positive score always means "more
    of what this leg is betting on". Centring matters only to rules that read
    magnitudes (``signal``); ranks are shift-invariant.
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
        book = values[decision, side[decision] != 0]
        book = book[np.isfinite(book)]
        centre = float(np.median(book)) if len(book) else 0.0
        for leg in legs:
            members = np.flatnonzero(side[decision] == leg)
            # Centred on the book's median and oriented the way the leg is
            # traded: a short leg's best name is its LOWEST forecast, and a
            # signal-weighted leg sizes by distance from the centre, not from
            # whatever zero the forecast's scale happens to have.
            score = (values[decision, members] - centre) * leg
            raw = rule(sigma[np.ix_(members, members)], score)
            total = float(np.sum(np.abs(raw)))
            if total <= 0 or not np.isfinite(total):
                raw, total = np.ones(len(members)), float(len(members))
            weights[decision, members] = leg * share * np.abs(raw) / total
    return weights


def _per_decision_risk_aversion(
    risk_aversion: float | ArrayLike, n_decisions: int
) -> NDArray[np.float64]:
    """One positive, finite risk aversion per decision, from a scalar or an array.

    A risk aversion calibrated to the forecast's scale over the WHOLE panel
    reads every later decision's forecast into the earliest one's sizing; a
    caller avoiding that look-ahead calibrates one ``lambda_t`` per decision
    from data available at ``t``. A scalar is broadcast, and reproduces the
    single-lambda result exactly. Shared with the trades stage so both
    validate it the same way.
    """
    array = np.asarray(risk_aversion, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(n_decisions, float(array))
    elif array.shape != (n_decisions,):
        raise ValueError(
            "risk_aversion must be a scalar or a 1-D array with one value per "
            f"decision ({n_decisions}), got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("risk_aversion must be finite")
    if np.any(array <= 0):
        raise ValueError("risk_aversion must be positive")
    return array


def mean_variance_target(
    forecast: ForwardReturns,
    covariance: ArrayLike,
    *,
    risk_aversion: float | ArrayLike = 1.0,
) -> PortfolioWeights:
    """The frictionless mean-variance holding, ``Sigma^-1 mu_t / lambda_t``.

    What the portfolio would hold if trading were free. ``covariance`` has
    shape ``(n_horizons, n_assets, n_assets)`` and must be invertible (an
    estimate from :func:`~stable_finance.evaluation.portfolio.risk.estimate_covariance`
    is). ``risk_aversion`` is one positive scalar for the whole panel, or a
    1-D array of ``n_decisions`` values, one ``lambda_t`` per decision, which
    scales only that decision's target. A missing forecast is a zero expected
    return. How far to actually move toward this target, given costs, is the
    trades stage's decision
    (:func:`~stable_finance.evaluation.trades.rebalance.rebalance`).
    """
    mu = np.nan_to_num(np.asarray(forecast.values, dtype=np.float64), nan=0.0)
    aversion = _per_decision_risk_aversion(risk_aversion, mu.shape[0])
    sigma = np.asarray(covariance, dtype=np.float64)
    if sigma.shape != (mu.shape[2], mu.shape[1], mu.shape[1]):
        raise ValueError(
            f"covariance must have shape {(mu.shape[2], mu.shape[1], mu.shape[1])}, "
            f"got {sigma.shape}")
    target = np.empty_like(mu)
    for horizon in range(mu.shape[2]):
        target[:, :, horizon] = (
            np.linalg.solve(sigma[horizon], mu[:, :, horizon].T).T / aversion[:, None])
    return PortfolioWeights(target, forecast.decisions, forecast.assets, forecast.horizons)
