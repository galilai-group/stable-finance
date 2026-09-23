"""Which names a portfolio is allowed to hold at each decision.

Spread costs are heavy-tailed across a cross-section, so the tradable
universe is one of the decisions that moves a Sharpe ratio most and is
dictated by the model least. Screening happens before weighting and is
expressed as a boolean mask on the ``(decision, asset)`` grid, so a screen can
be composed with any selection rule and any allocator.

Every screen here reads only quantities observable at the decision itself --
the name's own quoted half-spread, or how often it was priced over a prior
window -- so no screen introduces lookahead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "UNIVERSE_SCREENS",
    "apply_screen",
    "liquidity_screen",
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


def liquidity_screen(
    half_spread: ArrayLike, *, keep: float
) -> NDArray[np.bool_]:
    """Keep the tightest ``keep`` fraction of each decision's cross-section.

    The cut is a CROSS-SECTIONAL FRACTION rather than an absolute spread in
    basis points, because the level of spreads moves by an order of magnitude
    across this archive's history -- an absolute threshold would screen out
    almost everything in 2008 and almost nothing in 2019, making the universe
    a proxy for the calendar. A fraction keeps the screen's meaning fixed.
    """
    if not 0.0 < keep <= 1.0:
        raise ValueError("keep must lie in (0, 1]")
    spread = np.asarray(half_spread, dtype=np.float64)
    usable = tradable(spread)
    if keep == 1.0:
        return usable
    mask = np.zeros_like(usable)
    for decision in range(spread.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) == 0:
            continue
        # At least one name survives any non-empty cross-section: a screen
        # that can empty a decision would silently turn into a missing period
        # rather than a tighter universe.
        count = max(int(np.floor(len(row) * keep)), 1)
        tightest = row[np.argsort(spread[decision, row], kind="stable")[:count]]
        mask[decision, tightest] = True
    return mask


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


def apply_screen(name: str, half_spread: ArrayLike) -> NDArray[np.bool_]:
    """Dispatch to one named universe screen; see :data:`UNIVERSE_SCREENS`."""
    if name == "all":
        return tradable(half_spread)
    if name == "tight_50pct":
        return liquidity_screen(half_spread, keep=0.5)
    if name == "tight_25pct":
        return liquidity_screen(half_spread, keep=0.25)
    raise ValueError(f"unknown universe {name!r}; choose from {UNIVERSE_SCREENS}")


#: The universe screens a backtest configuration may name.
UNIVERSE_SCREENS = ("all", "tight_50pct", "tight_25pct")
