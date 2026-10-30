"""Risk for a decision-price, reference-notional multi-horizon holding plan.

All horizon payoffs use the known DECISION price as denominator, including
when entry is delayed. Adjacent differences are exact price-increment payoffs
per unit of decision notional. They are not simple returns rebased at a
future entry or interval beginning. Covariance keeps cross-asset and
cross-interval dependence, in interval-major order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import LedoitWolf

from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = ["JointReturnPanel", "JointRiskModel", "estimate_joint_risk"]

_PANEL_BASIS = "decision_price_payoff"
_RETURN_BASIS = "decision_price_payoff_increments"


def _readonly(value: ArrayLike) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _delay(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("entry_delay_seconds must be a nonnegative integer")
    return int(value)


def _axes(assets: ArrayLike, horizons: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    names, raw = np.asarray(assets), np.asarray(horizons)
    if names.ndim != 1 or not len(names):
        raise ValueError("assets must be a non-empty one-dimensional axis")
    try:
        unique = len(np.unique(names)) == len(names)
    except (TypeError, ValueError) as error:
        raise ValueError("assets must be comparable unique labels") from error
    if not unique or any(value is None for value in names):
        raise ValueError("assets must be unique non-missing labels")
    if any(isinstance(value, Real) and not np.isfinite(value) for value in names):
        raise ValueError("assets must be finite labels")
    if names.dtype.kind in "fc" and not np.isfinite(names).all():
        raise ValueError("assets must be finite labels")
    if (raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iuf"
            or not np.isfinite(raw).all() or np.any(raw <= 0)
            or np.any(raw != np.floor(raw))
            or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError("horizons must contain finite positive integer seconds")
    if raw.dtype.kind == "f" and np.any(raw >= float(2**63)):
        raise ValueError("horizons exceed the positive integer-second range")
    with np.errstate(invalid="ignore", over="ignore"):
        durations = raw.astype(np.int64)
    if np.any(durations <= 0):
        raise ValueError("horizons exceed the positive integer-second range")
    if np.any(np.diff(durations) <= 0):
        raise ValueError("horizons must be strictly increasing and unique")
    return _readonly(names), _readonly(durations)


def _clock(values: ArrayLike, name: str) -> tuple[np.ndarray, str]:
    array = np.asarray(values)
    if array.dtype.kind in "iuf":
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite timestamps")
        return array, "numeric"
    if array.dtype.kind == "M":
        if np.isnat(array).any():
            raise ValueError(f"{name} cannot contain NaT")
        return array, "datetime64"
    if array.dtype == object and all(isinstance(value, datetime) for value in array.flat):
        aware = [value.utcoffset() is not None for value in array.flat]
        if any(aware) and not all(aware):
            raise ValueError(f"{name} cannot mix timezone-aware and naive datetimes")
        if aware and aware[0]:
            normalized = np.asarray([value.astimezone(timezone.utc) for value in array.flat],
                                    dtype=object).reshape(array.shape)
            return normalized, "aware_datetime"
        return array, "naive_datetime"
    raise ValueError(f"{name} must use numeric seconds, numpy datetime64, or native datetime timestamps")


def _scalar_clock(value, name: str):
    values, kind = _clock(value, name)
    if values.ndim != 0:
        raise ValueError(f"{name} must be a scalar timestamp")
    return values[()], kind


def _same_clock(left_kind: str, right_kind: str) -> None:
    if left_kind != right_kind:
        raise ValueError("timestamps must use the same time basis and timezone awareness")


def _earliest_availability(decisions: np.ndarray, horizons: np.ndarray, kind: str,
                           delay: int = 0) -> np.ndarray:
    durations = [int(horizon)+delay for horizon in horizons]
    try:
        if kind == "numeric":
            # Promotion avoids signed integer wrap while retaining the exact
            # ordinary epoch-second values used by market data.
            result = decisions[:, None].astype(np.longdouble) + np.asarray(durations, dtype=np.longdouble)[None]
            if not np.isfinite(result).all():
                raise ValueError("decision plus horizon exceeds the timestamp range")
            return result
        if kind == "datetime64":
            result = decisions[:, None] + np.asarray(durations, dtype="timedelta64[s]")[None]
            if np.isnat(result).any() or np.any(result < decisions[:, None]):
                raise ValueError("decision plus horizon exceeds the timestamp range")
            return result
        return np.asarray([[decision + timedelta(seconds=duration) for duration in durations]
                           for decision in decisions], dtype=object)
    except (OverflowError, TypeError) as error:
        raise ValueError("decision plus horizon exceeds the timestamp range") from error


@dataclass(frozen=True)
class JointReturnPanel:
    """Cumulative future payoffs in known decision-price units, ``(D,N,K)``.

    At anchor t, delay delta, and horizon h, a realized value is
    ``(P[t+delta+h] - P[t+delta]) / P[t]``. A prediction is the conditional
    mean of exactly that payoff using information available at t. For delta
    zero this equals an ordinary cumulative simple return. For positive delay
    it DOES NOT equal a simple return divided by the unknown future entry
    price; the two conventions are never converted implicitly.

    The explicit contract prevents passing an ambiguous ForwardReturns object
    into joint reference-share optimization. Missing values may be NaN. Risk
    estimation inspects only mature rows; forecast construction must handle
    its own missing forecasts explicitly. Arrays are copied read-only.
    """

    values: ArrayLike
    decisions: ArrayLike
    assets: ArrayLike
    horizons: ArrayLike
    entry_delay_seconds: int = 0
    return_basis: str = _PANEL_BASIS

    def __post_init__(self):
        names, horizons = _axes(self.assets, self.horizons)
        contract = ForwardReturns(self.values, self.decisions, names, horizons)
        decisions, _ = _clock(contract.decisions, "decisions")
        if not len(decisions) or np.any(decisions[1:] <= decisions[:-1]):
            raise ValueError("decisions must be nonempty, strictly increasing and unique")
        if self.return_basis != _PANEL_BASIS:
            raise ValueError(f"return_basis must be {_PANEL_BASIS!r}")
        object.__setattr__(self, "entry_delay_seconds", _delay(self.entry_delay_seconds))
        object.__setattr__(self, "values", _readonly(contract.values))
        object.__setattr__(self, "decisions", _readonly(contract.decisions))
        object.__setattr__(self, "assets", names)
        object.__setattr__(self, "horizons", horizons)


@dataclass(frozen=True)
class JointRiskModel:
    """Immutable full covariance of disjoint decision-price payoffs.

    ``covariance`` accepts ``(K*N,K*N)`` or ``(K,N,K,N)`` and is stored as a
    flat interval-major matrix: coordinate ``k*N+i`` is asset i's payoff in
    interval k. ``information_cutoff`` is the latest observation availability
    used to estimate it, not an inferred return-window end. A supplied model
    must declare that cutoff explicitly. Consumers must call
    ``assert_available`` before using it at a forecast decision.

    Tiny negative eigenvalues attributable to floating-point roundoff are
    removed; material asymmetry or non-convexity raises. Supplied singular or
    zero PSD matrices are permitted as explicit models. Estimated constant
    coordinates instead raise, rather than silently inventing a risk floor.
    No current holdings, transaction costs or future execution data enter this
    portfolio-stage contract.
    """

    covariance: ArrayLike
    assets: ArrayLike
    horizons: ArrayLike
    information_cutoff: object
    source_observations: int = 0
    shrinkage: float = 0.0
    estimator: str = "supplied"
    entry_delay_seconds: int = 0
    risk_source: str = "supplied"
    return_basis: str = _RETURN_BASIS

    def __post_init__(self):
        names, durations = _axes(self.assets, self.horizons)
        n, k = len(names), len(durations)
        covariance = np.array(self.covariance, dtype=np.float64, copy=True)
        if covariance.shape == (k, n, k, n):
            covariance = covariance.reshape(k*n, k*n)
        if covariance.shape != (k*n, k*n) or not np.isfinite(covariance).all():
            raise ValueError(f"covariance must be finite with shape {(k*n, k*n)} or {(k, n, k, n)}")
        scale = max(float(np.abs(covariance).max()), np.finfo(float).tiny)
        if np.max(np.abs(covariance-covariance.T)) > 1e-10*scale:
            raise ValueError("covariance must be symmetric")
        covariance = (covariance+covariance.T)/2
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[0] < -1e-10*scale:
            raise ValueError("covariance must be positive semidefinite")
        if eigenvalues[0] < 0:
            covariance = (eigenvectors*np.maximum(eigenvalues, 0)) @ eigenvectors.T
            covariance = (covariance+covariance.T)/2
        cutoff, _ = _scalar_clock(self.information_cutoff, "information_cutoff")
        if (isinstance(self.source_observations, (bool, np.bool_))
                or not isinstance(self.source_observations, Integral) or self.source_observations < 0):
            raise ValueError("source_observations must be a nonnegative integer")
        if (isinstance(self.shrinkage, (bool, np.bool_)) or not isinstance(self.shrinkage, Real)
                or not np.isfinite(self.shrinkage) or not 0 <= self.shrinkage <= 1):
            raise ValueError("shrinkage must lie in [0,1]")
        if not isinstance(self.estimator, str) or not self.estimator:
            raise ValueError("estimator must be a nonempty description")
        if self.return_basis != _RETURN_BASIS:
            raise ValueError(f"return_basis must be {_RETURN_BASIS!r}")
        if self.risk_source not in ("supplied", "unconditional_payoff_covariance",
                                     "historical_forecast_residual_covariance"):
            raise ValueError("risk_source must describe a supplied, unconditional, or residual model")
        covariance = np.ascontiguousarray(covariance)
        covariance.setflags(write=False)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "assets", names)
        object.__setattr__(self, "horizons", durations)
        object.__setattr__(self, "information_cutoff", cutoff)
        object.__setattr__(self, "entry_delay_seconds", _delay(self.entry_delay_seconds))

    @property
    def blocks(self) -> np.ndarray:
        """Read-only ``(interval,asset,interval,asset)`` covariance view."""
        return self.covariance.reshape(len(self.horizons), len(self.assets),
                                       len(self.horizons), len(self.assets))

    def assert_available(self, as_of) -> None:
        """Reject use before the last risk input became observable."""
        cutoff, cutoff_kind = _scalar_clock(self.information_cutoff, "information_cutoff")
        decision, decision_kind = _scalar_clock(as_of, "as_of")
        _same_clock(cutoff_kind, decision_kind)
        if cutoff > decision:
            raise ValueError("risk information_cutoff is later than the forecast decision")


def estimate_joint_risk(
    history: JointReturnPanel,
    *,
    available_at: ArrayLike,
    as_of,
    shrinkage: float | str = "ledoit_wolf",
    min_samples: int = 2,
    historical_forecasts: JointReturnPanel | None = None,
    forecast_available_at: ArrayLike | None = None,
) -> JointRiskModel:
    """Estimate complete-path cross-interval risk using only mature labels.

    ``history`` uses cumulative decision-price payoffs and its declared entry
    latency. ``available_at[d,k]`` explicitly states when that horizon's label
    was observable, at least anchor + delay + horizon seconds. It may
    be later due to publication or execution-label conventions. A path enters
    the estimate only when ALL its horizons are available by ``as_of`` and
    ALL its asset/horizon returns are finite. Missing paths are dropped as a
    whole, preserving a common covariance sample. Mature infinities raise;
    unavailable future return values are never inspected.

    Adjacent differences convert cumulative returns to disjoint payoffs,
    exactly. The path matrix is flattened interval-major. Shrinkage is
    applied to its sample CORRELATION, then rescaled by sample standard
    deviations. This preserves different marginal risks for unequal interval
    durations. ``'ledoit_wolf'`` estimates the intensity from standardized past
    paths; a number in [0,1] fixes it (zero gives the sample covariance).
    With ``historical_forecasts`` and ``forecast_available_at`` supplied,
    estimate the covariance of payoff-prediction errors. Forecast axes, basis
    and delay must match exactly, and each forecast must have been available
    by its anchor. No model is fitted here. Without forecasts, raw payoff
    covariance is an explicitly stationary unconditional proxy, not an
    asserted conditional risk model. Overlapping paths estimate moments but
    are not claimed to be independent statistical observations.
    """
    if not isinstance(history, JointReturnPanel):
        raise TypeError("history must be JointReturnPanel with an explicit decision-price payoff basis")
    names, horizons = _axes(history.assets, history.horizons)
    if (isinstance(min_samples, (bool, np.bool_)) or not isinstance(min_samples, Integral)
            or min_samples < 2):
        raise ValueError("min_samples must be an integer at least two")
    if isinstance(shrinkage, str):
        if shrinkage != "ledoit_wolf":
            raise ValueError("shrinkage must be 'ledoit_wolf' or a number in [0,1]")
    elif (isinstance(shrinkage, (bool, np.bool_)) or not isinstance(shrinkage, Real)
          or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1):
        raise ValueError("shrinkage must be 'ledoit_wolf' or a number in [0,1]")
    decisions, clock_kind = _clock(history.decisions, "history decisions")
    available, available_kind = _clock(available_at, "available_at")
    cutoff, cutoff_kind = _scalar_clock(as_of, "as_of")
    _same_clock(clock_kind, available_kind)
    _same_clock(clock_kind, cutoff_kind)
    d, n, k = history.values.shape
    if not d or available.shape != (d, k):
        raise ValueError(f"available_at must have shape {(d, k)} for a nonempty history")
    if np.any(decisions[1:] <= decisions[:-1]):
        raise ValueError("history decisions must be strictly increasing and unique")
    if np.any(available[:, 1:] < available[:, :-1]):
        raise ValueError("available_at must be nondecreasing across horizons")
    earliest = _earliest_availability(decisions, horizons, clock_kind, history.entry_delay_seconds)
    if np.any(available < earliest):
        raise ValueError("available_at cannot precede the anchor plus delay and horizon")
    mature = np.all(available <= cutoff, axis=1)
    observed = np.asarray(history.values[mature], dtype=np.float64)
    if np.isinf(observed).any():
        raise ValueError("mature history cannot contain infinity")
    if (historical_forecasts is None) != (forecast_available_at is None):
        raise ValueError("historical_forecasts and forecast_available_at must be supplied together")
    if historical_forecasts is not None:
        if not isinstance(historical_forecasts, JointReturnPanel):
            raise TypeError("historical_forecasts must be JointReturnPanel")
        forecast_decisions, forecast_kind = _clock(historical_forecasts.decisions, "historical forecast decisions")
        _same_clock(clock_kind, forecast_kind)
        if (not np.array_equal(forecast_decisions, decisions)
                or not np.array_equal(historical_forecasts.assets, names)
                or not np.array_equal(historical_forecasts.horizons, horizons)
                or historical_forecasts.entry_delay_seconds != history.entry_delay_seconds
                or historical_forecasts.return_basis != history.return_basis):
            raise ValueError("historical forecasts must match history axes, delay and payoff basis exactly")
        forecast_times, forecast_time_kind = _clock(forecast_available_at, "forecast_available_at")
        _same_clock(clock_kind, forecast_time_kind)
        if forecast_times.shape != (d, k) or np.any(forecast_times > decisions[:, None]):
            raise ValueError("forecast_available_at must have shape (D,K) and be no later than each anchor")
        predicted = np.asarray(historical_forecasts.values[mature], dtype=np.float64)
        if np.isinf(predicted).any():
            raise ValueError("mature historical forecasts cannot contain infinity")
        with np.errstate(over="ignore", invalid="ignore"):
            observed = observed-predicted
        if np.isinf(observed).any():
            raise ValueError("historical forecast residuals exceed the finite payoff range")
    complete = np.isfinite(observed).all(axis=(1, 2))
    selected = observed[complete]
    count = len(selected)
    if count < min_samples:
        raise ValueError(f"joint risk needs {min_samples} complete mature paths; got {count}")
    with np.errstate(over="ignore", invalid="ignore"):
        increments = np.diff(selected, axis=2, prepend=np.zeros((count, n, 1)))
    if not np.isfinite(increments).all():
        raise ValueError("payoff increments exceed the finite numerical range")
    paths = increments.transpose(0, 2, 1).reshape(count, k*n)
    centered = paths-paths.mean(axis=0)
    variances = np.sum(centered*centered, axis=0)/(count-1)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("every asset/interval coordinate needs positive finite observed variance")
    standard_deviations = np.sqrt(variances)
    standardized = centered/standard_deviations
    correlation = standardized.T @ standardized/(count-1)
    intensity = (float(LedoitWolf(assume_centered=True).fit(standardized).shrinkage_)
                 if isinstance(shrinkage, str) else float(shrinkage))
    correlation = (1-intensity)*correlation+intensity*np.eye(k*n)
    covariance = correlation*np.outer(standard_deviations, standard_deviations)
    selected_times = available[mature][complete]
    information_cutoff = np.max(selected_times)
    return JointRiskModel(covariance, names, horizons, information_cutoff,
        source_observations=count, shrinkage=intensity,
        estimator="standardized_ledoit_wolf_correlation" if isinstance(shrinkage, str)
                  else "fixed_correlation_shrinkage",
        entry_delay_seconds=history.entry_delay_seconds,
        risk_source="unconditional_payoff_covariance" if historical_forecasts is None
                    else "historical_forecast_residual_covariance")
