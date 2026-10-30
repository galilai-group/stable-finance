"""Signal transforms: what a forecast says the book should express.

Between a forecast and portfolio construction sits a stage that trades
nothing and yet changes the portfolio: it decides what the forecast is ALLOWED
TO MEAN. A transform here takes a ``(decision, asset)`` array of forecasts and
returns another on the same grid, so it composes with every selection rule,
every weighting rule and the cost-aware allocator downstream without any of
them knowing it ran.

ALPHA REFINEMENT puts a forecast in return units. Many forecasts are not:
a probe trained on a rank target emits ranks, a noised oracle emits scores of
whatever scale its noise had, and a classifier emits log-odds. A sort does
not care, but every rule that compares the forecast with something measured
in returns does -- mean-variance divides it by a covariance, edge-to-cost and
threshold rules subtract a spread from it. Feeding such a rule a rank is a
unit error that silently picks a risk aversion. Grinold's rule,
``alpha = IC * sigma * score``, is the standard conversion: standardise the
cross-section, scale by the skill the forecast is believed to have, and
scale by how much each name actually moves. Configuration reaches it by
name, through :func:`alpha_from_rule`: ``grinold`` standardises with the
z-score, ``grinold_rank`` with the Gaussian rank score, which no single
outlier can dominate.

NEUTRALIZATION removes the exposures the forecast never meant to take. A
forecast that ranks high-beta names first in a rising market looks skilful
and is merely long the market; one that favours a single industry is an
industry bet wearing a stock-picking label. Residualising the forecast on
those exposures, cross-sectionally and before construction, keeps the book
from expressing a bet the model did not make. The intercept in every
regression also makes the residual dollar neutral.

Everything here is deterministic numpy and standard library, NaN marks a
missing value on input and an unusable one on output, and every estimate is
formed per decision from that decision's cross-section alone.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation._grid import _rankdata
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = [
    "ALPHA_RULES",
    "NEUTRALIZATIONS",
    "SIGNAL_REQUIRES",
    "alpha_from_rule",
    "apply_signal",
    "estimate_beta",
    "industry_exposures",
    "neutralize",
    "refine_alpha",
]


#: The alpha rules a backtest configuration may name, both Grinold's rule and
#: differing only in the cross-sectional score. ``grinold`` uses the z-score,
#: ``(x - mean) / std``, which keeps the forecast's spacing and so lets one
#: extreme forecast claim an extreme alpha; ``grinold_rank`` uses the Gaussian
#: rank score, ``Phi^-1(rank / (n + 1))``, which keeps only the ordering and
#: is robust to fat tails in the forecast. See :func:`alpha_from_rule`.
ALPHA_RULES = ("grinold", "grinold_rank")

#: The :func:`refine_alpha` score each named alpha rule standardises with.
_ALPHA_SCORE_OF = {"grinold": "zscore", "grinold_rank": "gauss_rank"}

#: The neutralizations a backtest configuration may name.
NEUTRALIZATIONS = ("beta", "industry", "beta_industry")

#: What each neutralization needs of the market. Beta is estimated from the
#: return history alone, so it needs nothing; the industry options need an
#: ``industry`` code per name and decision, which is market data.
SIGNAL_REQUIRES: dict[str, tuple[str, ...]] = {
    "beta": (),
    "industry": ("industry",),
    "beta_industry": ("industry",),
}

_NORMAL = NormalDist()


def _usable(values: NDArray[np.float64], eligible: ArrayLike | None) -> NDArray[np.bool_]:
    usable = np.isfinite(values)
    if eligible is not None:
        mask = np.asarray(eligible, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(
                f"eligible must have shape {values.shape}, got {mask.shape}")
        usable &= mask
    return usable


# ── alpha refinement ────────────────────────────────────────────────────────

def _zscore(row: NDArray[np.float64]) -> NDArray[np.float64] | None:
    spread = float(np.std(row))
    if not spread > 0:
        return None
    return (row - row.mean()) / spread


def _gauss_rank(row: NDArray[np.float64]) -> NDArray[np.float64] | None:
    ranks = _rankdata(row)
    if np.all(ranks == ranks[0]):
        return None
    # Average ranks take few distinct values; invert each once, and tied names
    # share one quantile exactly.
    unique, inverse = np.unique(ranks, return_inverse=True)
    n = len(row)
    quantiles = np.array([_NORMAL.inv_cdf(r / (n + 1)) for r in unique])
    return quantiles[inverse]


_SCORES = {"zscore": _zscore, "gauss_rank": _gauss_rank}


def refine_alpha(
    forecast: ArrayLike,
    volatility: ArrayLike,
    *,
    ic: float,
    score: str = "zscore",
    eligible: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Grinold's rule, ``alpha[t, i] = ic * sigma[i] * z[t, i]``.

    Grinold (1994), "Alpha is volatility times IC times score", *Journal of
    Portfolio Management* 20(4):9-16. ``z`` is the per-decision
    cross-sectional score of ``forecast`` over the eligible finite names:
    ``zscore`` is ``(x - mean) / std`` (``ddof=0``), ``gauss_rank`` is
    ``Phi^-1(rank / (n + 1))`` with average ranks for ties, which keeps an
    outlier from claiming a huge alpha. ``volatility`` is each name's sigma,
    shape ``(n_assets,)`` or ``(n_decisions, n_assets)``, in return units at
    the forecast's horizon.

    Sigma should ideally be RESIDUAL volatility -- the part of a name's risk a
    stock-specific forecast can explain -- but the caller passes what it has;
    the backtest passes ``sqrt(diag Sigma)`` of its risk model. ``ic`` is the
    believed correlation of forecast with outcome; a negative ``ic`` flips
    the signal, which is correct for a forecast known to be contrarian and a
    silent bug otherwise.

    Names that are ineligible or not finite are NaN, and so is every name at
    a decision with fewer than two usable names or no dispersion: such a
    cross-section carries no ranking to scale.
    """
    if not np.isfinite(ic):
        raise ValueError("ic must be finite")
    if score not in _SCORES:
        raise ValueError(f"unknown score {score!r}; choose from {tuple(_SCORES)}")
    values = np.asarray(forecast, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("forecast must be a 2-D (decision, asset) array")
    sigma = np.asarray(volatility, dtype=np.float64)
    if sigma.shape not in ((values.shape[1],), values.shape):
        raise ValueError(
            f"volatility must have shape {(values.shape[1],)} or {values.shape}, "
            f"got {sigma.shape}")
    if np.any(sigma[np.isfinite(sigma)] < 0):
        raise ValueError("volatility must be non-negative")
    sigma = np.broadcast_to(sigma, values.shape)
    usable = _usable(values, eligible)

    scored = np.full(values.shape, np.nan)
    transform = _SCORES[score]
    for decision in range(values.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) < 2:
            continue
        z = transform(values[decision, row])
        if z is not None:
            scored[decision, row] = z
    return ic * sigma * scored


def alpha_from_rule(
    name: str,
    forecast: ArrayLike,
    volatility: ArrayLike,
    *,
    ic: float,
    eligible: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Dispatch to one named alpha rule; see :data:`ALPHA_RULES`.

    Both rules are :func:`refine_alpha`, ``ic * sigma * score``; the name
    picks the score. ``grinold`` standardises with the z-score and
    ``grinold_rank`` with the Gaussian rank score, so a configuration can ask
    for the fat-tail-robust conversion without the score becoming an axis of
    its own: a score only means something inside an alpha rule. ``forecast``,
    ``volatility``, ``ic`` and ``eligible`` are as for :func:`refine_alpha`.
    """
    if name not in _ALPHA_SCORE_OF:
        raise ValueError(f"unknown alpha {name!r}; choose from {ALPHA_RULES}")
    return refine_alpha(forecast, volatility, ic=ic, score=_ALPHA_SCORE_OF[name],
                        eligible=eligible)


# ── exposures ───────────────────────────────────────────────────────────────

def estimate_beta(history: ForwardReturns) -> NDArray[np.float64]:
    """Each asset's OLS beta on the equal-weighted cross-sectional mean.

    The market is the mean of the finite returns at each decision of the
    supplied panel, so the estimate is DATASET-AGNOSTIC: it needs no index,
    no market capitalisation and no benchmark, only the panel itself, and so
    runs on any input the backtest can. Each beta uses the decisions where
    both the asset and the market are finite. Returns
    ``(n_horizons, n_assets)``; an asset with fewer than two such decisions,
    or whose market has no variance over them, is NaN.

    As with the risk models, estimation sees only the decisions it is given,
    so passing a prior period gives an out-of-sample beta.
    """
    values = np.asarray(history.values, dtype=np.float64)
    _, n_assets, n_horizons = values.shape
    out = np.full((n_horizons, n_assets), np.nan)
    for horizon in range(n_horizons):
        x = values[:, :, horizon]
        finite = np.isfinite(x)
        counts = finite.sum(axis=1)
        market = np.where(finite, x, 0.0).sum(axis=1) / np.maximum(counts, 1)
        market[counts == 0] = np.nan
        for asset in range(n_assets):
            pair = finite[:, asset] & np.isfinite(market)
            # A constant market has no variance; test it exactly, since the
            # centered sum of identical values need not round to zero.
            if pair.sum() < 2 or np.ptp(market[pair]) == 0:
                continue
            m = market[pair] - market[pair].mean()
            r = x[pair, asset] - x[pair, asset].mean()
            variance = float(m @ m)
            if variance > 0:
                out[horizon, asset] = float(m @ r) / variance
    return out


def industry_exposures(codes: ArrayLike) -> NDArray[np.float64]:
    """One-hot industry dummies from ``(decision, asset)`` industry codes.

    Returns ``(n_decisions, n_assets, n_industries)`` over the sorted unique
    finite codes in the whole array. NO DUMMY IS DROPPED: together they are
    collinear with the intercept, which :func:`neutralize` resolves with a
    minimum-norm least-squares solve rather than an arbitrary base industry.
    A NaN code -- industry unknown -- gives an all-NaN row, so that name is
    left out of neutralization at that decision rather than guessed.
    """
    values = np.asarray(codes, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("codes must be a 2-D (decision, asset) array")
    known = np.isfinite(values)
    industries = np.unique(values[known])
    dummies = (values[:, :, None] == industries[None, None, :]).astype(np.float64)
    dummies[~known] = np.nan
    return dummies


# ── neutralization ──────────────────────────────────────────────────────────

def neutralize(
    values: ArrayLike,
    exposures: ArrayLike,
    *,
    eligible: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Per-decision cross-sectional OLS residual on an intercept and exposures.

    Grinold and Kahn's alpha neutralization: the part of the forecast that no
    linear combination of the exposures explains. ``exposures`` has shape
    ``(n_decisions, n_assets, k)``, or ``(n_assets, k)`` to apply the same
    exposures at every decision. Each decision regresses on the names that
    are eligible and finite in ``values`` and in every exposure; the rest are
    NaN, as is every name at a decision with fewer than two such names.

    The solve is ``np.linalg.lstsq``, so a rank-deficient design -- industry
    dummies alongside the intercept, an industry absent from a decision, a
    constant exposure -- gives the minimum-norm fit instead of an error. The
    residual is the same for every least-squares solution, so the choice does
    not leak into the result. The intercept makes each residual row sum to
    zero.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be a 2-D (decision, asset) array")
    factors = np.asarray(exposures, dtype=np.float64)
    if factors.ndim == 2 and factors.shape[0] == x.shape[1]:
        factors = np.broadcast_to(factors, (x.shape[0], *factors.shape))
    if factors.ndim != 3 or factors.shape[:2] != x.shape:
        raise ValueError(
            f"exposures must have shape {(*x.shape, 'k')} or "
            f"{(x.shape[1], 'k')}, got {np.shape(exposures)}")
    usable = _usable(x, eligible) & np.isfinite(factors).all(axis=2)

    out = np.full(x.shape, np.nan)
    for decision in range(x.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) < 2:
            continue
        design = np.column_stack([np.ones(len(row)), factors[decision, row]])
        target = x[decision, row]
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        out[decision, row] = target - design @ coefficients
    return out


def apply_signal(
    name: str,
    values: ArrayLike,
    *,
    eligible: ArrayLike,
    beta: ArrayLike | None = None,
    industry: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Dispatch to one named neutralization; see :data:`NEUTRALIZATIONS`.

    ``beta`` is one horizon's row of :func:`estimate_beta`, shape
    ``(n_assets,)`` (or ``(n_decisions, n_assets)`` for a rolling estimate);
    ``industry`` is the ``(decision, asset)`` industry codes. ``beta_industry``
    regresses on both at once, which is not the same as neutralizing one
    after the other when beta differs by industry.
    """
    if name not in NEUTRALIZATIONS:
        raise ValueError(f"unknown neutralization {name!r}; choose from {NEUTRALIZATIONS}")
    x = np.asarray(values, dtype=np.float64)
    parts = []
    if name in ("beta", "beta_industry"):
        if beta is None:
            raise ValueError(
                f"neutralization={name!r} needs beta; estimate it with estimate_beta")
        b = np.asarray(beta, dtype=np.float64)
        if b.shape == (x.shape[1],):
            b = np.broadcast_to(b, x.shape)
        if b.shape != x.shape:
            raise ValueError(
                f"beta must have shape {(x.shape[1],)} or {x.shape}, got {b.shape}")
        parts.append(b[:, :, None])
    if name in ("industry", "beta_industry"):
        if industry is None:
            from stable_finance.dataset.schema import MissingRolesError

            raise MissingRolesError(
                f"neutralization={name!r} needs an industry code per name, and "
                "the market data has none")
        parts.append(industry_exposures(industry))
    return neutralize(x, np.concatenate(parts, axis=2), eligible=eligible)
