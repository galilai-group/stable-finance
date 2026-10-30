"""Read-only diagnostics of supplied portfolio weights, not forecast scores.

The report measures the target or held book supplied by the caller. It never
estimates beta, changes a position, fits a model or declares that targets were
executed. Factor data must be aligned and known at the corresponding decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import date, datetime
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation.portfolio.contract import PortfolioWeights

__all__ = ["PortfolioExposureLimits", "PortfolioExposureReport", "portfolio_exposure_report"]


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PortfolioExposureLimits:
    """Optional diagnostic limits; none are imposed by default.

    ``max_abs_industry`` bounds the absolute NET weight of each known
    industry. ``max_hhi`` bounds concentration of absolute weights normalized
    by gross. A zero net/beta/industry limit requests the respective neutrality
    check. Limits diagnose supplied positions; they never modify them.
    """

    gross_limit: float | None = None
    max_weight: float | None = None
    max_abs_net: float | None = None
    max_abs_beta: float | None = None
    max_abs_industry: float | None = None
    max_hhi: float | None = None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{field.name} must be a finite nonnegative number or None")
        if self.max_hhi is not None and self.max_hhi > 1:
            raise ValueError("max_hhi must lie in [0,1]")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value[()]) if value.ndim == 0 else [_json_value(v) for v in value]
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("floating value exceeds the finite JSON numerical range")
        return converted
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, PortfolioExposureLimits):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


@dataclass(frozen=True)
class PortfolioExposureReport:
    """Per-decision/horizon exposure and concentration, with explicit coverage.

    Scalar diagnostics have shape ``(D,H)``. Industry arrays have shape
    ``(D,H,G)``, with the last axis named by ``industry_labels``. Long and
    short exposures are both positive amounts. HHI is based on absolute
    weights divided by gross, so cash has HHI and effective names equal to
    zero. Missing held beta/classification gives incomplete factor coverage;
    it never silently becomes a zero factor exposure.

    ``violations`` contains raw nonnegative excess for each requested limit,
    or NaN where a factor limit cannot be evaluated. ``constraint_status`` is
    pass, fail, unknown, or not_checked. Any known excess above tolerance
    gives fail; otherwise an unknown requested constraint prevents pass.
    Arrays and the violation mapping are read-only copies.
    """

    decisions: NDArray
    assets: NDArray
    horizons: NDArray
    gross: NDArray
    net: NDArray
    long_exposure: NDArray
    short_exposure: NDArray
    max_abs_weight: NDArray
    names_held: NDArray
    concentration_hhi: NDArray
    effective_names: NDArray
    beta_exposure: NDArray
    beta_coverage: NDArray
    industry_labels: NDArray
    industry_net: NDArray
    industry_gross: NDArray
    unknown_industry_net: NDArray
    unknown_industry_gross: NDArray
    industry_coverage: NDArray
    limits: PortfolioExposureLimits
    tolerance: float
    violations: Mapping[str, NDArray]
    constraint_status: NDArray

    def as_dict(self) -> dict:
        """JSON-safe arrays and axes; missing values become null.

        Floating metadata outside Python's finite JSON number range raises
        explicitly rather than emitting Infinity or losing a numeric label.
        """
        return {field.name: _json_value(getattr(self, field.name)) for field in fields(self)}


def _validate_weights(weights):
    if not isinstance(weights, PortfolioWeights):
        raise TypeError("weights must be PortfolioWeights")
    values = np.asarray(weights.values, dtype=float)
    if (values.ndim != 3 or min(values.shape) == 0 or not np.isfinite(values).all()
            or values.shape != tuple(len(getattr(weights, name)) for name in ("decisions", "assets", "horizons"))):
        raise ValueError("weights must be finite on nonempty aligned (D,N,H) axes")
    for name in ("decisions", "assets", "horizons"):
        axis = np.asarray(getattr(weights, name))
        if axis.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        try:
            unique = len(np.unique(axis)) == len(axis)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must have comparable unique labels") from error
        if not unique or any(item is None for item in axis):
            raise ValueError(f"{name} must have unique nonmissing labels")
        if any(isinstance(item, Real) and not isinstance(item, Integral)
               and not np.isfinite(item) for item in axis):
            raise ValueError(f"{name} must have finite labels")
        if axis.dtype.kind in "iufc" and not np.isfinite(axis).all():
            raise ValueError(f"{name} must have finite labels")
        if axis.dtype.kind == "M" and np.isnat(axis).any():
            raise ValueError(f"{name} cannot contain NaT")
    horizons = np.asarray(weights.horizons)
    if (horizons.dtype.kind not in "iuf" or not np.isfinite(horizons).all()
            or np.any(horizons <= 0) or np.any(horizons != np.floor(horizons))):
        raise ValueError("horizons must contain positive integer seconds")
    return values


def _beta(beta, shape):
    if beta is None:
        return np.full(shape, np.nan)
    raw = np.asarray(beta)
    if raw.dtype.kind not in "iuf" or np.isinf(raw).any():
        raise ValueError("beta must contain finite numbers or NaN")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(raw, dtype=float)
    if np.isinf(values).any():
        raise ValueError("beta exceeds the finite numerical range")
    d, n, _ = shape
    if values.shape == (n,):
        values = values[None, :, None]
    elif values.shape == (d, n):
        values = values[:, :, None]
    elif values.shape != shape:
        raise ValueError("beta must have shape (N,), (D,N), or explicit (D,N,H)")
    return np.broadcast_to(values, shape)


def _industry(industry, d, n):
    if industry is None:
        return np.full((d, n), None), np.zeros((d, n), bool), np.asarray([], dtype=str)
    raw = np.asarray(industry, dtype=object)
    if raw.shape == (n,):
        raw = np.broadcast_to(raw[None], (d, n))
    elif raw.shape != (d, n):
        raise ValueError("industry must have shape (N,) or (D,N)")
    clean = np.full((d, n), None, dtype=object)
    known = np.zeros((d, n), bool)
    labels, kinds = set(), set()
    for index in np.ndindex(raw.shape):
        value = raw[index]
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            continue
        if isinstance(value, str):
            if value == "":
                continue
            kind = "string"
        elif isinstance(value, Integral) and not isinstance(value, bool):
            kind = "numeric"
        elif isinstance(value, Real) and not isinstance(value, bool):
            if np.isnan(value):
                continue
            if not np.isfinite(value):
                raise ValueError("industry codes must be finite numbers, strings, or missing")
            kind = "numeric"
        else:
            raise ValueError("industry codes must be finite numbers, strings, or missing")
        kinds.add(kind)
        labels.add(value)
        known[index], clean[index] = True, value
    if len(kinds) > 1:
        raise ValueError("industry codes must be homogeneous strings or numbers")
    return clean, known, np.asarray(sorted(labels), dtype=object)


def _coverage(unknown_gross, gross):
    # Equivalent sums on C/F-strided input can differ by a few ulps. Coverage
    # is a fraction; clipping only that diagnostic preserves its domain.
    # Constraint evaluability separately uses exact nonzero missing holdings.
    ratio = np.divide(gross-unknown_gross, gross, out=np.ones_like(gross), where=gross > 0)
    return np.clip(ratio, 0., 1.)


def portfolio_exposure_report(
    weights: PortfolioWeights,
    *,
    beta: ArrayLike | None = None,
    industry: ArrayLike | None = None,
    limits: PortfolioExposureLimits | None = None,
    tolerance: float = 1e-8,
) -> PortfolioExposureReport:
    """Diagnose supplied target or held weights without changing them.

    Beta accepts static ``(N,)``, decision-varying ``(D,N)``, or explicit
    ``(D,N,H)`` values. A ``(H,N)`` beta estimate is not implicitly inferred;
    transpose/broadcast it explicitly. Industry codes are static ``(N,)``
    or decision-varying ``(D,N)`` and are common across horizon books. String
    and numeric codes are supported, with None/NaN/empty string missing.
    Arrays must already follow the weights' asset/decision order; this
    function neither estimates factors nor checks their historical source.

    Betas and industry codes on unheld names may be missing. Missing data on
    any nonzero held name makes the respective requested factor constraint
    unknown. Cash has exact zero exposure and full coverage. Without limits,
    diagnostics are still returned but status is not_checked. Report native
    constructor outputs as targets; supply actual executed weights to report
    actual holdings. Forecast-score neutralization alone guarantees neither.
    """
    values = _validate_weights(weights)
    if limits is None:
        limits = PortfolioExposureLimits()
    if not isinstance(limits, PortfolioExposureLimits):
        raise TypeError("limits must be PortfolioExposureLimits")
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    d, n, h = values.shape
    absolute = np.abs(values)
    with np.errstate(over="ignore", invalid="ignore"):
        gross, net = absolute.sum(axis=1), values.sum(axis=1)
        long = np.maximum(values, 0).sum(axis=1)
        short = np.maximum(-values, 0).sum(axis=1)
    if not all(np.isfinite(array).all() for array in (gross, net, long, short)):
        raise ValueError("portfolio exposure exceeds the finite numerical range")
    normalized = np.divide(absolute, gross[:, None, :],
                           out=np.zeros_like(absolute), where=gross[:, None, :] > 0)
    hhi = (normalized*normalized).sum(axis=1)
    effective = np.divide(1., hhi, out=np.zeros_like(hhi), where=hhi > 0)

    factors = _beta(beta, values.shape)
    beta_known = np.isfinite(factors)
    beta_unknown = np.where(beta_known, 0., absolute).sum(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        beta_net = (values*np.where(beta_known, factors, 0.)).sum(axis=1)
    if not np.isfinite(beta_net).all():
        raise ValueError("beta exposure exceeds the finite numerical range")
    beta_net = np.where(beta_unknown > 0, np.nan, beta_net)

    codes, industry_known, labels = _industry(industry, d, n)
    membership = codes[:, :, None] == labels[None, None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        industry_net = np.einsum("dnh,dng->dhg", values, membership)
        industry_gross = np.einsum("dnh,dng->dhg", absolute, membership)
        unknown_net = np.where(industry_known[:, :, None], 0., values).sum(axis=1)
        unknown_gross = np.where(industry_known[:, :, None], 0., absolute).sum(axis=1)
    industry_max = np.max(np.abs(industry_net), axis=2, initial=0.)
    industry_max = np.where(unknown_gross > 0, np.nan, industry_max)

    metrics = dict(gross_limit=gross, max_weight=absolute.max(axis=1),
        max_abs_net=np.abs(net), max_abs_beta=np.abs(beta_net),
        max_abs_industry=industry_max, max_hhi=hhi)
    violations = {name: np.maximum(metric-getattr(limits, name), 0.)
                  for name, metric in metrics.items() if getattr(limits, name) is not None}
    status = np.full((d, h), "not_checked", dtype="U11")
    if violations:
        amounts = np.stack(list(violations.values()))
        unknown = np.any(~np.isfinite(amounts), axis=0)
        failed = np.any(amounts > tolerance, axis=0)
        status[:] = "pass"
        status[unknown], status[failed] = "unknown", "fail"
    data = dict(decisions=weights.decisions, assets=weights.assets, horizons=weights.horizons,
        gross=gross, net=net, long_exposure=long, short_exposure=short,
        max_abs_weight=absolute.max(axis=1), names_held=np.count_nonzero(values, axis=1),
        concentration_hhi=hhi, effective_names=effective, beta_exposure=beta_net,
        beta_coverage=_coverage(beta_unknown, gross), industry_labels=labels,
        industry_net=industry_net, industry_gross=industry_gross,
        unknown_industry_net=unknown_net, unknown_industry_gross=unknown_gross,
        industry_coverage=_coverage(unknown_gross, gross), constraint_status=status)
    return PortfolioExposureReport(**{name: _readonly(value) for name, value in data.items()},
        limits=limits, tolerance=float(tolerance),
        violations=MappingProxyType({name: _readonly(value) for name, value in violations.items()}))
