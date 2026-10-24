"""Which names the data can price at all: present in the panel, and quoted.

These are facts about the data rather than choices about the portfolio. A name
with no two-sided market at a decision cannot be traded at that instant, and a
name priced on a handful of decisions has no estimable covariance row. Both are
read straight off the market data with no parameter a study would vary, which
is what makes them the dataset's to state. How much tighter to make the
universe than that -- the liquidity screens -- is a risk-management decision
and lives in :mod:`stable_finance.evaluation.portfolio.risk`.

Every mask here reads only quantities observable at the decision itself, so
neither introduces lookahead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "presence_screen",
    "tradable",
]


def tradable(half_spread: ArrayLike) -> NDArray[np.bool_]:
    """Names with a usable two-sided market at the decision.

    A missing or negative half-spread means the book was not two-sided, which
    is a real market state rather than a data defect: the name simply cannot
    be traded at that instant and must not be silently treated as free.
    """
    spread = np.asarray(half_spread, dtype=np.float64)
    if spread.ndim != 2:
        raise ValueError("half_spread must be two-dimensional")
    return np.isfinite(spread) & (spread >= 0)


def presence_screen(
    observed: ArrayLike, *, minimum: float
) -> NDArray[np.bool_]:
    """Assets priced on at least ``minimum`` of the decisions, as a 1-D mask.

    Applied to the asset axis rather than per decision. A name quoted on a
    handful of decisions is not a small position -- it is an unestimable
    covariance row and a portfolio that cannot rebalance, so it is excluded
    from the panel entirely rather than screened out decision by decision.
    """
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum must lie in [0, 1]")
    seen = np.asarray(observed, dtype=bool)
    if seen.ndim != 2:
        raise ValueError("observed must be two-dimensional")
    return seen.mean(axis=0) >= minimum
