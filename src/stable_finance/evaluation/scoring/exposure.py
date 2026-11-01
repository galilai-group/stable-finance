"""Exposure and concentration of supplied target or filled portfolio weights."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.evaluation.portfolio.contract import PortfolioWeights

__all__ = ["portfolio_exposure_report"]


def portfolio_exposure_report(
    weights: PortfolioWeights,
    *,
    factor_loadings: ArrayLike | None = None,
    factor_names: ArrayLike | None = None,
    factor_lower: ArrayLike | None = None,
    factor_upper: ArrayLike | None = None,
    tolerance: float = 1e-8,
) -> dict:
    """Report the input book without estimating factors or changing positions.

    Book metrics have shape ``(decision, horizon)``. Static factor loadings
    are ``(asset, factor)`` in the supplied asset order; factor diagnostics
    have shape ``(decision, horizon, factor)``. Bounds are scalar or one per
    factor and both are required when checking limits. Outward infinities
    leave a side open. Factor names default to ``factor_0``, ``factor_1``, ...

    Missing loadings on a nonzero position make that factor exposure unknown.
    Coverage is the fraction of gross with known loadings, one for cash.
    A known violation takes precedence over an unknown factor in the book's
    status. Without bounds the status is ``not_checked``. Targets and output
    from ``positions_from_fills`` can be checked separately with this function.
    ``tolerance`` is in portfolio-weight units: each factor's raw threshold
    is tolerance times its largest absolute known loading (one for an all-zero
    or unknown column), reported as ``factor_tolerance``. Exposures, slacks
    and violations stay in the original factor units.
    """
    if not isinstance(weights, PortfolioWeights):
        raise TypeError("weights must be PortfolioWeights")
    values = weights.values
    if not np.isfinite(values).all():
        raise ValueError("portfolio weights must be finite")
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be a finite nonnegative scalar")
    absolute = np.abs(values)
    with np.errstate(over="ignore", invalid="ignore"):
        gross, net = absolute.sum(axis=1), values.sum(axis=1)
    if not np.isfinite(gross).all() or not np.isfinite(net).all():
        raise ValueError("portfolio exposure exceeds the finite numerical range")
    fractions = np.divide(absolute, gross[:, None, :], out=np.zeros_like(values),
                          where=gross[:, None, :] > 0)
    hhi = np.square(fractions).sum(axis=1)
    report = dict(decisions=weights.decisions.copy(), assets=weights.assets.copy(),
        horizons=weights.horizons.copy(), gross=gross, net=net,
        max_abs_weight=absolute.max(axis=1), names_held=(absolute != 0).sum(axis=1),
        concentration_hhi=hhi,
        effective_names=np.divide(1., hhi, out=np.zeros_like(hhi), where=hhi > 0),
        factor_names=None, factor_exposure=None, factor_coverage=None,
        factor_lower=None, factor_upper=None, factor_tolerance=None,
        lower_slack=None, upper_slack=None,
        violations=None, constraint_status=np.full(gross.shape, "not_checked"))
    if factor_loadings is None:
        if any(value is not None for value in (factor_names, factor_lower, factor_upper)):
            raise ValueError("factor names and bounds require factor_loadings")
        return report
    raw = np.asarray(factor_loadings)
    if (raw.dtype.kind not in "iuf" or raw.ndim != 2
            or raw.shape[0] != len(weights.assets) or raw.shape[1] == 0):
        raise ValueError("factor_loadings must have shape (N,F) in the portfolio asset order")
    loadings = np.asarray(raw, dtype=float)
    if np.isinf(loadings).any():
        raise ValueError("factor_loadings may contain NaN for unknown values, but not infinity")
    count = loadings.shape[1]
    names = np.asarray([f"factor_{i}" for i in range(count)] if factor_names is None else factor_names)
    if (names.shape != (count,)
            or any(not isinstance(name, (str, np.str_)) or not name.strip() for name in names)
            or len(set(names.tolist())) != count):
        raise ValueError("factor_names must contain one unique nonempty string per factor")
    known = np.isfinite(loadings)
    unknown = np.einsum("dnh,nf->dhf", (values != 0).astype(int), (~known).astype(int)) > 0
    with np.errstate(over="ignore", invalid="ignore"):
        exposure = np.einsum("dnh,nf->dhf", values, np.where(known, loadings, 0.))
    if not np.isfinite(exposure[~unknown]).all():
        raise ValueError("factor exposure exceeds the finite numerical range")
    exposure[unknown] = np.nan
    covered = np.einsum("dnh,nf->dhf", absolute, known.astype(float))
    coverage = np.divide(covered, gross[:, :, None], out=np.ones_like(covered),
                         where=gross[:, :, None] > 0)
    report.update(factor_names=names.copy(), factor_exposure=exposure,
                  factor_coverage=np.clip(coverage, 0., 1.))
    if factor_lower is None and factor_upper is None:
        return report
    if factor_lower is None or factor_upper is None:
        raise ValueError("factor_lower and factor_upper must be supplied together")

    def bound(value, name):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.shape not in ((), (count,)):
            raise ValueError(f"{name} must be scalar or have shape (F,)")
        result = np.broadcast_to(np.asarray(raw, dtype=float), (count,)).copy()
        if np.isnan(result).any():
            raise ValueError(f"{name} cannot contain NaN")
        return result

    lower, upper = bound(factor_lower, "factor_lower"), bound(factor_upper, "factor_upper")
    if np.isposinf(lower).any() or np.isneginf(upper).any() or np.any(lower > upper):
        raise ValueError("factor bounds must satisfy lower <= upper and use outward infinities")
    lower_slack, upper_slack = exposure-lower, upper-exposure
    violations = np.maximum(np.maximum(-lower_slack, -upper_slack), 0.)
    scale = np.max(np.where(known, np.abs(loadings), 0.), axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        thresholds = tolerance * np.where(scale > 0., scale, 1.)
    if not np.isfinite(thresholds).all():
        raise ValueError("factor tolerance exceeds the finite numerical range")
    failed = np.any(violations > thresholds, axis=2)
    status = np.where(failed, "fail", np.where(unknown.any(axis=2), "unknown", "pass"))
    report.update(factor_lower=lower, factor_upper=upper, factor_tolerance=thresholds,
        lower_slack=lower_slack,
        upper_slack=upper_slack, violations=violations, constraint_status=status)
    return report
