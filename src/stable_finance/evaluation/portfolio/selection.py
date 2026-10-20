"""Which side of the book each name is on, from a cross-sectional forecast.

A forecast orders the cross-section; it does not say how much of that order to
trade. Holding the whole cross-section long-short diversifies best and dilutes
the signal most; holding only the tails concentrates the signal and the risk.
That trade-off is a decision the model does not make, so it is an axis.

Selection returns a sign array on the ``(decision, asset)`` grid -- ``+1``
long, ``-1`` short, ``0`` not held -- which an allocator then sizes. Cuts are
taken per decision on the names that survived the universe screen, so a
tighter universe changes which names are in the tails rather than how many.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = ["SELECTION_REQUIRES", "SELECTION_RULES", "select_book", "tail_selection"]


def tail_selection(
    forecast: ArrayLike,
    eligible: ArrayLike,
    *,
    fraction: float,
    long_only: bool = False,
) -> NDArray[np.int8]:
    """Long the top ``fraction`` of each decision, short the bottom.

    The cut is a fraction of that decision's eligible cross-section, not an
    absolute forecast threshold: forecasts from a rank target, a raw-return
    probe and a trained head live on entirely different scales, and a fraction
    is the only cut that means the same thing for all three.
    """
    if not 0.0 < fraction <= 0.5:
        raise ValueError("fraction must lie in (0, 0.5]")
    values = np.asarray(forecast, dtype=np.float64)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    if values.shape != usable.shape or values.ndim != 2:
        raise ValueError("forecast and eligible must be equally shaped 2-D arrays")

    signs = np.zeros(values.shape, dtype=np.int8)
    for decision in range(values.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) < 2:
            continue
        count = max(int(np.floor(len(row) * fraction)), 1)
        # A leg cannot exceed half the cross-section, or the long and short
        # tails would overlap and a name would be held on both sides.
        count = min(count, len(row) // 2)
        if count == 0:
            continue
        order = row[np.argsort(values[decision, row], kind="stable")]
        signs[decision, order[-count:]] = 1
        if not long_only:
            signs[decision, order[:count]] = -1
    return signs


def select_book(
    name: str, forecast: ArrayLike, eligible: ArrayLike
) -> NDArray[np.int8]:
    """Dispatch to one named selection rule; see :data:`SELECTION_RULES`.

    ``all`` holds every eligible name, signed by whether its forecast is above
    or below that decision's cross-sectional median -- the dollar-neutral book
    a reader would write down from the ranking alone.
    """
    values = np.asarray(forecast, dtype=np.float64)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    if name == "all":
        signs = np.zeros(values.shape, dtype=np.int8)
        for decision in range(values.shape[0]):
            row = np.flatnonzero(usable[decision])
            if len(row) < 2:
                continue
            centre = np.median(values[decision, row])
            signs[decision, row] = np.where(values[decision, row] >= centre, 1, -1)
        return signs
    if name == "quintile":
        return tail_selection(values, usable, fraction=0.2)
    if name == "decile":
        return tail_selection(values, usable, fraction=0.1)
    if name == "decile_long_only":
        return tail_selection(values, usable, fraction=0.1, long_only=True)
    raise ValueError(f"unknown selection {name!r}; choose from {SELECTION_RULES}")


#: The selection rules a backtest configuration may name.
SELECTION_RULES = ("all", "quintile", "decile", "decile_long_only")

#: What each named rule needs of the market: nothing, since each cuts the
#: forecast ranking alone. ``threshold_selection`` and ``persistent_selection``
#: compare the forecast with a cost and so need ``half_spread`` when given
#: one; they are not configuration axes.
SELECTION_REQUIRES: dict[str, tuple[str, ...]] = {name: () for name in SELECTION_RULES}
