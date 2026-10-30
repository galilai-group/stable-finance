"""Declared linear factor exposure budgets with explicit asset and time axes.

This module estimates no factor, changes no weights and imports only NumPy and
the standard library. Low-level solvers may load it directly in an isolated
execution namespace. Missing loadings never become a free held exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["FactorExposureConstraints", "ResolvedFactorConstraints",
           "FactorExposureReport", "factor_constraint_arrays", "factor_exposure_report"]


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _labels(value, name, *, strings=False):
    labels = np.asarray(value)
    if labels.ndim != 1 or not len(labels):
        raise ValueError(f"{name} must be a nonempty one-dimensional label axis")
    if (any(item is None or (isinstance(item, str) and not item.strip()) for item in labels)
            or any(isinstance(item, Real) and not np.isfinite(item) for item in labels)):
        raise ValueError(f"{name} must contain nonmissing finite labels")
    if strings and any(not isinstance(item, (str, np.str_)) for item in labels):
        raise ValueError(f"{name} must contain nonempty strings")
    try:
        unique = len(np.unique(labels)) == len(labels)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain comparable unique labels") from error
    if not unique:
        raise ValueError(f"{name} must contain unique labels")
    return _readonly(labels)


def _clock(value, name):
    values = np.asarray(value)
    if values.dtype.kind in "iuf":
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite timestamps")
        return values, "numeric"
    if values.dtype.kind == "M":
        if np.isnat(values).any():
            raise ValueError(f"{name} cannot contain NaT")
        return values, "datetime64"
    if values.dtype == object and all(isinstance(item, datetime) for item in values.flat):
        aware = [item.utcoffset() is not None for item in values.flat]
        if any(aware) and not all(aware):
            raise ValueError(f"{name} cannot mix timezone-aware and naive timestamps")
        if aware and aware[0]:
            normalized = np.asarray([item.astimezone(timezone.utc) for item in values.flat],
                                    dtype=object).reshape(values.shape)
            return normalized, "aware_datetime"
        return values, "naive_datetime"
    raise ValueError(f"{name} must use numeric seconds or datetime timestamps")


def _decision_axis(value, name):
    values, kind = _clock(value, name)
    if values.ndim != 1 or not len(values) or np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{name} must be nonempty, strictly increasing and unique")
    return _readonly(values), kind


def _horizons(value):
    raw = np.asarray(value)
    if (raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iu"
            or np.any(raw < 1) or np.any(raw >= 2**63)):
        raise ValueError("horizons must contain positive integer seconds")
    values = np.asarray(raw, dtype=np.int64)
    if np.any(values[1:] <= values[:-1]):
        raise ValueError("horizons must be strictly increasing and unique")
    return _readonly(values)


def _bounds(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape not in ((), (shape[-1],), shape):
        raise ValueError(f"{name} must be scalar, (F,), or the full declared factor-bound shape {shape}")
    values = np.broadcast_to(np.asarray(raw, dtype=float), shape).copy()
    if np.isnan(values).any():
        raise ValueError(f"{name} cannot contain NaN")
    return values


def _ordered_bounds(lower, upper):
    if np.isposinf(lower).any() or np.isneginf(upper).any():
        raise ValueError("lower may use -inf and upper may use +inf, not the reverse")
    if np.any(lower > upper):
        raise ValueError("factor lower bounds cannot exceed upper bounds")


def factor_constraint_arrays(loadings, lower, upper, *, stages, n_assets,
                             factor_names=None, required=None):
    """Validate one decision's factor budgets and return ``(A, l, u, names)``.

    A is static ``(N,F)`` or explicit ``(K,N,F)``; returned A is ``(K,N,F)``.
    Bounds are scalar, ``(F,)`` or ``(K,F)`` and return as ``(K,F)``. Missing
    coefficients are permitted only on hard-excluded ``required=False``
    coordinates; those coordinates become zero for numerical multiplication.
    Required is a boolean ``(N,)`` or ``(K,N)`` mask, defaulting to all true.
    Coefficient infinity is always rejected. Null inputs return four Nones.
    """
    k, n = _count(stages, "stages"), _count(n_assets, "n_assets")
    if loadings is None:
        if any(value is not None for value in (lower, upper, factor_names)):
            raise ValueError("factor bounds and names require factor_loadings")
        return None, None, None, None
    raw = np.asarray(loadings)
    if raw.dtype.kind not in "iuf" or raw.ndim not in (2, 3) or raw.shape[-1] < 1:
        raise ValueError("factor_loadings must be numeric (N,F) or (K,N,F)")
    f = raw.shape[-1]
    if raw.shape not in ((n, f), (k, n, f)):
        raise ValueError("factor_loadings must follow the exact (N,F) or (K,N,F) axes")
    values = np.broadcast_to(np.asarray(raw, dtype=float), (k, n, f)).copy()
    if np.isinf(values).any():
        raise ValueError("factor loadings cannot contain infinity")
    needed = np.ones((k, n), dtype=bool)
    if required is not None:
        raw_required = np.asarray(required)
        if raw_required.dtype.kind != "b" or raw_required.shape not in ((n,), (k, n)):
            raise ValueError("required must be a boolean (N,) or (K,N) mask")
        needed = np.broadcast_to(raw_required, (k, n))
    if not np.isfinite(values[needed]).all():
        raise ValueError("factor loadings must be known for every potentially held coordinate")
    values[~needed] = 0.
    lo, hi = _bounds(lower, (k, f), "factor_lower"), _bounds(upper, (k, f), "factor_upper")
    _ordered_bounds(lo, hi)
    names = (_labels([f"factor_{i}" for i in range(f)], "factor_names", strings=True)
             if factor_names is None else _labels(factor_names, "factor_names", strings=True))
    if len(names) != f:
        raise ValueError("factor_names must match the factor axis")
    return tuple(_readonly(value) for value in (values, lo, hi, names))


@dataclass(frozen=True)
class ResolvedFactorConstraints:
    """Declared budgets aligned to `(decision, stage, asset, factor)` order."""

    loadings: ArrayLike
    lower: ArrayLike
    upper: ArrayLike
    factor_names: ArrayLike
    available_at: ArrayLike
    decisions: ArrayLike
    assets: ArrayLike
    horizons: ArrayLike

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class FactorExposureConstraints:
    """Immutable point-in-time factor loadings and lower/upper exposure budgets.

    Loadings support `(N,F)`, time-varying `(D,N,F)` with `decisions`, staged
    `(K,N,F)` with `horizons`, or `(D,K,N,F)` with both axes. A static matrix
    can broadcast over declared axes, for example for time-varying budgets.
    Three-dimensional loadings require exactly one explicit leading axis.
    Bounds are scalar, `(F,)` or the FULL declared leading axes plus `(F,)`;
    a `(D,F)` bound is not silently interpreted as `(K,F)` when D equals K.

    `available_at` is the latest information time for the whole matrix/budget
    at each declared coordinate: scalar or full declared leading shape. None
    declares availability at the requested decision, matching the beta API.
    No factor, industry membership or availability is estimated here.
    """

    loadings: ArrayLike
    lower: ArrayLike
    upper: ArrayLike
    assets: ArrayLike
    factor_names: ArrayLike
    decisions: ArrayLike | None = None
    horizons: ArrayLike | None = None
    available_at: ArrayLike | None = None

    def __post_init__(self):
        assets = _labels(self.assets, "assets")
        names = _labels(self.factor_names, "factor_names", strings=True)
        decisions = None if self.decisions is None else _decision_axis(self.decisions, "factor decisions")[0]
        horizons = None if self.horizons is None else _horizons(self.horizons)
        leading = (() if decisions is None else (len(decisions),)) + (() if horizons is None else (len(horizons),))
        raw = np.asarray(self.loadings)
        tail = (len(assets), len(names))
        if raw.dtype.kind not in "iuf" or raw.shape not in (tail, leading+tail):
            raise ValueError("loadings must match static (N,F) or the complete explicitly declared leading axes")
        if raw.ndim == 3 and (decisions is None) == (horizons is None):
            raise ValueError("three-dimensional loadings need exactly one of decisions or horizons")
        values = np.broadcast_to(np.asarray(raw, dtype=float), leading+tail).copy()
        if np.isinf(values).any():
            raise ValueError("factor loadings cannot contain infinity")
        lo, hi = _bounds(self.lower, leading+(len(names),), "lower"), _bounds(self.upper, leading+(len(names),), "upper")
        _ordered_bounds(lo, hi)
        available = None
        if self.available_at is not None:
            available, kind = _clock(self.available_at, "factor available_at")
            if available.shape not in ((), leading):
                raise ValueError("factor available_at must be scalar or follow all declared leading axes")
            if decisions is not None and kind != _clock(decisions, "factor decisions")[1]:
                raise ValueError("factor availability and decisions must use the same timestamp basis")
            available = np.broadcast_to(available, leading).copy()
        for name, value in (("assets", assets), ("factor_names", names), ("loadings", values),
                            ("lower", lo), ("upper", hi), ("decisions", decisions),
                            ("horizons", horizons), ("available_at", available)):
            object.__setattr__(self, name, None if value is None else _readonly(value))

    def resolve(self, *, decisions, assets, horizons, eligible=None):
        """Select exact requested labels, validate causality, and align budgets.

        Source time/stage axes may contain more labels than requested. Only
        exact labels are selected; there is no forward fill or interpolation.
        Asset order must already match exactly, rather than being reordered.
        Eligible is an explicit boolean `(N,)`, `(D,N)` or `(D,K,N)` hard-zero
        mask. Missing coefficients on any potentially held coordinate raise.
        """
        requested, kind = _decision_axis(decisions, "requested decisions")
        requested_horizons = _horizons(horizons)
        if not np.array_equal(np.asarray(assets), self.assets):
            raise ValueError("factor assets must match the exact requested asset order")
        d, k, n, f = len(requested), len(requested_horizons), len(self.assets), len(self.factor_names)

        def selected(source, target, name):
            if source is None:
                return None
            if name == "decisions" and _clock(source, name)[1] != kind:
                raise ValueError("factor and requested decisions must use the same timestamp basis")
            indices = np.searchsorted(source, target)
            if np.any(indices >= len(source)) or not np.array_equal(source[indices], target):
                raise ValueError(f"requested {name} must exist exactly in the factor contract")
            return indices

        di = selected(self.decisions, requested, "decisions")
        ki = selected(self.horizons, requested_horizons, "horizons")

        def align(value, tail):
            values = value
            if di is not None:
                values = np.take(values, di, axis=0)
            if ki is not None:
                values = np.take(values, ki, axis=1 if di is not None else 0)
            if di is None:
                values = np.expand_dims(values, 0)
            if ki is None:
                values = np.expand_dims(values, 1)
            return np.broadcast_to(values, (d, k)+tail).copy()

        loadings = align(self.loadings, (n, f))
        lower, upper = align(self.lower, (f,)), align(self.upper, (f,))
        required = np.ones((d, k, n), dtype=bool)
        if eligible is not None:
            raw = np.asarray(eligible)
            if raw.dtype.kind != "b":
                raise ValueError("eligible must contain booleans")
            if raw.shape == (n,):
                required = np.broadcast_to(raw, (d, k, n))
            elif raw.shape == (d, n):
                required = np.broadcast_to(raw[:, None], (d, k, n))
            elif raw.shape == (d, k, n):
                required = raw
            else:
                raise ValueError("eligible must be (N,), (D,N), or explicit (D,K,N)")
        for row in range(d):
            loadings[row], lower[row], upper[row], _ = factor_constraint_arrays(
                loadings[row], lower[row], upper[row], stages=k, n_assets=n,
                factor_names=self.factor_names, required=required[row])
        if self.available_at is None:
            available = np.broadcast_to(requested[:, None], (d, k)).copy()
        else:
            if _clock(self.available_at, "factor available_at")[1] != kind:
                raise ValueError("factor availability and requested decisions must use the same timestamp basis")
            available = align(self.available_at, ())
            if np.any(available > requested[:, None]):
                raise ValueError("factor available_at is later than a requested decision")
        return ResolvedFactorConstraints(loadings, lower, upper, self.factor_names,
            available, requested, self.assets, requested_horizons)


@dataclass(frozen=True)
class FactorExposureReport:
    """Named realized or target exposures in `(D,K,F)` order, not `(D,N,H)`."""

    exposures: ArrayLike
    lower: ArrayLike
    upper: ArrayLike
    lower_slack: ArrayLike
    upper_slack: ArrayLike
    violations: ArrayLike
    status: ArrayLike
    factor_names: ArrayLike
    decisions: ArrayLike
    assets: ArrayLike
    horizons: ArrayLike

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def factor_exposure_report(weights, constraints, tolerance=1e-8):
    """Report supplied weights without estimating factors or repairing holdings.

    Weights must expose native `values (D,N,K)`, decisions, assets and horizons.
    A missing coefficient on any nonzero held coordinate raises. Missing
    coefficients on exact zero positions are harmless and contribute zero.
    This reports the input book only; target feasibility does not assert that
    the target was executed or that actual marked holdings remain feasible.
    """
    if not isinstance(constraints, FactorExposureConstraints):
        raise TypeError("constraints must be FactorExposureConstraints")
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be a finite nonnegative scalar")
    values = np.asarray(weights.values)
    expected = (len(weights.decisions), len(weights.assets), len(weights.horizons))
    if values.shape != expected or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("weights must be finite on their exact (D,N,K) axes")
    stage_weights = np.asarray(values, dtype=float).transpose(0, 2, 1)
    resolved = constraints.resolve(decisions=weights.decisions, assets=weights.assets,
        horizons=weights.horizons, eligible=stage_weights != 0)
    exposures = np.einsum("dkn,dknf->dkf", stage_weights, resolved.loadings)
    if not np.isfinite(exposures).all():
        raise ValueError("factor exposure multiplication exceeds finite range")
    lower_slack, upper_slack = exposures-resolved.lower, resolved.upper-exposures
    violations = np.maximum(np.maximum(-lower_slack, -upper_slack), 0.)
    status = np.where(np.any(violations > tolerance, axis=2), "fail", "pass")
    return FactorExposureReport(exposures, resolved.lower, resolved.upper,
        lower_slack, upper_slack, violations, status, resolved.factor_names,
        resolved.decisions, resolved.assets, resolved.horizons)
