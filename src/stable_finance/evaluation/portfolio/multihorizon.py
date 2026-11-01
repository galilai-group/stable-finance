"""Mix per-horizon target portfolios using fixed capital fractions."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.dataset.anchors import TO_CLOSE
from stable_finance.evaluation.portfolio.contract import PortfolioWeights

__all__ = ["combine_horizon_targets"]


def combine_horizon_targets(
    targets: PortfolioWeights,
    *,
    capital_weights: ArrayLike | None = None,
    target_horizon: int,
) -> PortfolioWeights:
    """Combine simultaneous targets into one book on the existing weight grid.

    Capital fractions follow the source horizon order, are nonnegative and
    sum to one; the default is equal fractions. Choose them before evaluating
    outcomes. Missing target weights raise rather than reallocating capital.
    An explicit zero book reserves its share of capital as cash.

    Opposing positions cancel without restoring gross exposure. The output
    horizon labels the downstream scoring interval, in seconds or TO_CLOSE;
    it does not prescribe how long to hold each source book. Score the net
    book with returns for that interval through the existing harness.
    """
    if not isinstance(targets, PortfolioWeights):
        raise TypeError("targets must be PortfolioWeights")
    values = targets.values
    if not np.isfinite(values).all():
        raise ValueError("target weights must be finite; use an explicit zero book for cash")
    if (isinstance(target_horizon, (bool, np.bool_))
            or not isinstance(target_horizon, Integral)
            or not (target_horizon == TO_CLOSE
                    or 0 < target_horizon <= np.iinfo(np.int64).max)):
        raise ValueError("target_horizon must be positive integer seconds or TO_CLOSE")
    count = len(targets.horizons)
    if capital_weights is None:
        fractions = np.full(count, 1.0 / count)
    else:
        raw = np.asarray(capital_weights)
        if raw.dtype.kind not in "iuf":
            raise ValueError("capital_weights must be numeric fractions")
        fractions = np.asarray(raw, dtype=float)
        if (fractions.shape != (count,) or not np.isfinite(fractions).all()
                or np.any(fractions < 0)
                or not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1e-12)):
            raise ValueError("capital_weights must have shape (H,), be nonnegative and sum to one")
    with np.errstate(over="ignore", invalid="ignore"):
        combined = np.einsum("dnh,h->dn", values, fractions)
    if not np.isfinite(combined).all():
        raise ValueError("combined target exceeds the finite numerical range")
    return PortfolioWeights(combined[:, :, None], targets.decisions.copy(),
                            targets.assets.copy(), np.asarray([target_horizon], dtype=np.int64))
