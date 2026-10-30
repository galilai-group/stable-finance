"""Combine existing per-horizon targets into one contemporaneous net book.

Construction stays with the native selection/weighting and mean-variance
functions. This module adds only a fixed-capital combination, with no model
fitting, risk estimation, holding policy, trade state or transaction costs.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.evaluation.portfolio.contract import PortfolioWeights

__all__ = ["combine_horizon_targets"]


def _readonly(values):
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _unique_axis(values, name):
    try:
        unique = len(np.unique(values)) == len(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain comparable unique labels") from error
    if not unique or any(value is None for value in values):
        raise ValueError(f"{name} must contain unique nonmissing labels")
    if any(isinstance(value, Real) and not np.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite labels")
    if values.dtype.kind in "iufc" and not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite labels")
    if values.dtype.kind == "M" and np.isnat(values).any():
        raise ValueError(f"{name} cannot contain NaT")


def combine_horizon_targets(
    targets: PortfolioWeights,
    *,
    capital_weights: ArrayLike | None = None,
    target_horizon_seconds: int,
) -> PortfolioWeights:
    """Return ``sum_h alpha_h * targets[d, :, h]`` on a ``(D,N,1)`` grid.

    ``targets`` is the output of existing portfolio constructors, with one
    desired book per source horizon at the same decisions and asset order.
    ``capital_weights`` is a fixed nonnegative vector summing to one, in the
    source-horizon order; omitted fractions are equal. Declare these fractions
    before evaluating outcomes. The function neither learns nor normalizes
    them. A zero-weight source still needs a finite, explicitly supplied book.

    All target weights must be finite. NaNs are rejected rather than treated
    as cash or triggering redistribution. Callers can explicitly supply zero
    books to reserve capital as cash. Any common convex position constraints
    on the component books are preserved by their convex combination. No new
    caps, neutrality adjustment or full-investment scaling is introduced.

    This is a contemporaneous target mix, not staggered vintages or a promise
    to hold a source book for its forecast horizon. ``target_horizon_seconds``
    explicitly labels the downstream evaluation interval. Realized returns
    used to score the output must match that single interval; overlapping
    source-horizon outcomes are never summed or averaged here. The native
    trades/execution stage sees only this one net target, so opposing source
    positions cancel before it decides trades and charges costs.

    Source decisions, assets and horizons must be nonempty and unique;
    horizons are positive integer seconds. Returned arrays are copied and
    read-only. Source ordering is preserved and input arrays are unchanged.
    """
    if not isinstance(targets, PortfolioWeights):
        raise TypeError("targets must be PortfolioWeights")
    values = np.asarray(targets.values, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) == 0:
        raise ValueError("targets must have nonempty decision, asset and horizon axes")
    expected = tuple(len(getattr(targets, name)) for name in ("decisions", "assets", "horizons"))
    if values.shape != expected:
        raise ValueError(f"targets must retain shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("target weights must be finite; supply explicit cash books for missing inputs")
    for name in ("decisions", "assets", "horizons"):
        _unique_axis(np.asarray(getattr(targets, name)), name)
    horizons = np.asarray(targets.horizons)
    if (horizons.dtype.kind not in "iuf" or not np.isfinite(horizons).all()
            or np.any(horizons <= 0) or np.any(horizons != np.floor(horizons))):
        raise ValueError("source horizons must contain positive integer seconds")
    if (isinstance(target_horizon_seconds, (bool, np.bool_))
            or not isinstance(target_horizon_seconds, Integral)
            or not 0 < target_horizon_seconds <= np.iinfo(np.int64).max):
        raise ValueError("target_horizon_seconds must be a positive int64 number of seconds")
    count = values.shape[2]
    if capital_weights is None:
        fractions = np.full(count, 1./count)
    else:
        raw = np.asarray(capital_weights)
        if raw.dtype.kind not in "iuf":
            raise ValueError("capital_weights must be numeric nonnegative fractions")
        fractions = np.asarray(raw, dtype=np.float64)
        if (fractions.shape != (count,) or not np.isfinite(fractions).all()
                or np.any(fractions < 0)
                or not np.isclose(fractions.sum(), 1., rtol=0., atol=1e-12)):
            raise ValueError("capital_weights must have shape (H,), be nonnegative and sum to one")
    with np.errstate(over="ignore", invalid="ignore"):
        combined = np.einsum("dnh,h->dn", values, fractions)
    if not np.isfinite(combined).all():
        raise ValueError("combined target exceeds the finite numerical range")
    return PortfolioWeights(_readonly(combined[:, :, None]),
        _readonly(targets.decisions), _readonly(targets.assets),
        _readonly([int(target_horizon_seconds)]))
