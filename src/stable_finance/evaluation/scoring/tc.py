"""How much of a forecast reaches the book: the transfer coefficient (TC).

Clarke, de Silva and Thorley (2002, *Financial Analysts Journal* 58(5); 2006,
*Journal of Investment Management*) extend Grinold's fundamental law to a
portfolio that cannot hold exactly what its forecast asks for::

    IR  ~=  TC * IC * sqrt(breadth)

The IC says how good the forecast is; the TC says what fraction of it the
weights actually held still express. A Sharpe that falls short of what the IC
promised is either a forecast that does not pay or a book that does not
express it, and only the TC tells the two apart -- which is why the harness
reports it beside the IC rather than leaving the gap to be guessed at.

The TC compares the two in RISK-ADJUSTED units. A forecast ``alpha`` and the
active weights ``w`` held against it live in different spaces: ``Sigma^-1
alpha`` is what a mean-variance investor would hold, so the forecast is
whitened by ``Sigma^-1/2`` and the weights coloured by ``Sigma^1/2``, and in
that common space the frictionless optimum is the forecast itself. The TC at
one decision is the cosine of the angle between them::

    TC = (Sigma^1/2 w) . (Sigma^-1/2 alpha) / (|Sigma^1/2 w| |Sigma^-1/2 alpha|)
       = w' alpha / sqrt(w' Sigma w * alpha' Sigma^-1 alpha)

which is the form in which the fundamental law holds as an identity: the
book's expected active return ``w' alpha`` is its risk times the forecast's
best attainable ratio times the TC. For a diagonal covariance it is
``sum(w alpha) / sqrt(sum(sigma^2 w^2) * sum(alpha^2 / sigma^2))``.

The roots never have to be formed: the identity needs only ``Sigma w`` and
``Sigma^-1 alpha``. ``Sigma^-1`` is computed once, and a decision that
leaves some names out is scored against the inverse of its own block of the
covariance, obtained from the full inverse by the block-inverse identity in
time linear in the names kept when only a few are missing. Zero-filling the
missing names instead would invent a view of zero on them.

An unconstrained, frictionless mean-variance book ``w ∝ Sigma^-1 alpha`` has
``TC = 1`` exactly. Every departure from it lowers the number: selecting only
the tails, long-only or per-leg constraints, a cost-aware solver that trades
only part of the way, and holding a book past the decision that formed it.
The TC is invariant to positive rescaling of either the weights or the
forecast, so it measures the SHAPE of the book against the shape of the view,
never the gross exposure or the units the forecast was quoted in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["TransferCoefficient", "transfer_coefficient"]

#: Fewest names a decision needs before its coefficient is reported: with two
#: names a dollar-neutral book has one degree of freedom and agrees with any
#: forecast up to sign.
_MIN_NAMES = 3


@dataclass(frozen=True)
class TransferCoefficient:
    """Per-decision transfer coefficients and their decision-level summary.

    ``per_decision`` is NaN where a decision had too few usable names or a
    flat book or forecast; ``mean`` and ``standard_error`` read only the
    decisions that were scored. The standard error treats decisions as
    independent, as the IC's does, so it is optimistic for a slowly turning
    book whose TC is serially correlated.
    """

    per_decision: NDArray[np.float64]
    mean: float
    standard_error: float


def _inverse(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a covariance, with eigenvalues floored relative to the largest.

    A singular or slightly indefinite estimate (a sample covariance from fewer
    periods than names, or roundoff) is inverted along its null directions by
    a large finite factor rather than by infinity.
    """
    symmetric = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    largest = float(values.max()) if len(values) else 0.0
    floor = max(largest, 0.0) * 1e-12 or np.finfo(np.float64).tiny
    return (vectors / np.maximum(values, floor)) @ vectors.T


def _block_solve(precision: NDArray[np.float64], covariance: NDArray[np.float64],
                 names: NDArray[np.bool_], alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(Sigma_SS)^-1 alpha_S`` for the kept names ``S``, from the full inverse.

    With ``P = Sigma^-1`` and ``M`` the names left out, the inverse of the kept
    block is ``P_SS - P_SM (P_MM)^-1 P_MS``. That costs a solve in the number
    left out, so it is used while they are few; past half the names the
    block is inverted directly.
    """
    missing = ~names
    if not missing.any():
        return precision @ alpha
    if missing.sum() * 2 > len(names):
        return _inverse(covariance[np.ix_(names, names)]) @ alpha
    p_ss = precision[np.ix_(names, names)]
    p_sm = precision[np.ix_(names, missing)]
    p_mm = precision[np.ix_(missing, missing)]
    return p_ss @ alpha - p_sm @ np.linalg.solve(p_mm, p_sm.T @ alpha)


def transfer_coefficient(
    weights: ArrayLike,
    alpha: ArrayLike,
    covariance: ArrayLike,
    *,
    eligible: ArrayLike | None = None,
) -> TransferCoefficient:
    """Correlation of risk-adjusted weights with risk-adjusted forecasts.

    Args:
        weights: ``(n_decisions, n_assets)`` active weights held.
        alpha: Forecasts of identical shape, in any positive scale.
        covariance: ``(n_assets, n_assets)`` risk model the book was built on.
        eligible: Optional boolean mask of the weights' shape; names outside
            it are left out, as are names not finite in both weights and alpha.

    Each decision is scored over its usable names against the inverse of
    that block of the covariance (see the module docstring). Decisions with
    fewer than three usable names, or with a flat book or forecast, are NaN.
    """
    weights = np.asarray(weights, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if weights.ndim != 2 or weights.shape != alpha.shape:
        raise ValueError("weights and alpha must be equally shaped 2-D arrays")
    n_assets = weights.shape[1]
    if covariance.shape != (n_assets, n_assets):
        raise ValueError(f"covariance must have shape {(n_assets, n_assets)}, "
                         f"got {covariance.shape}")
    usable = np.isfinite(weights) & np.isfinite(alpha)
    if eligible is not None:
        mask = np.asarray(eligible, dtype=bool)
        if mask.shape != weights.shape:
            raise ValueError(f"eligible must have shape {weights.shape}, got {mask.shape}")
        usable &= mask

    if not np.isfinite(covariance).all():
        return TransferCoefficient(per_decision=np.full(weights.shape[0], np.nan),
                                   mean=np.nan, standard_error=np.nan)
    precision = _inverse(covariance)
    per_decision = np.full(weights.shape[0], np.nan)
    for t, names in enumerate(usable):
        if names.sum() < _MIN_NAMES:
            continue
        w, a = weights[t, names], alpha[t, names]
        risk = float(w @ covariance[np.ix_(names, names)] @ w) if not names.all() \
            else float(w @ covariance @ w)
        attainable = float(a @ _block_solve(precision, covariance, names, a))
        denominator = np.sqrt(risk * attainable)
        if not np.isfinite(denominator) or denominator <= 0:
            continue
        per_decision[t] = float(np.clip((w @ a) / denominator, -1.0, 1.0))

    scored = per_decision[np.isfinite(per_decision)]
    mean = float(scored.mean()) if len(scored) else np.nan
    standard_error = (float(scored.std(ddof=1) / np.sqrt(len(scored)))
                      if len(scored) > 1 else np.nan)
    return TransferCoefficient(per_decision=per_decision, mean=mean,
                               standard_error=standard_error)
