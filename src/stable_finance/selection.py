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

__all__ = [
    "SELECTION_RULES", "persistent_selection", "select_book", "tail_selection",
    "threshold_selection",
]


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


def threshold_selection(
    forecast: ArrayLike,
    eligible: ArrayLike,
    *,
    multiple: float,
    cost: ArrayLike,
    long_only: bool = False,
) -> NDArray[np.int8]:
    """Hold a name only when its own forecast clears its own trading cost.

    A quantile cut spends the same on every name it selects, because a fixed
    fraction of the cross-section is held whatever the spread happens to be.
    But the spread is a cost per trade and the edge scales with the forecast,
    so the decision to hold a name is a comparison between the two, made per
    name: hold ``i`` when ``|edge_i| > multiple * cost_i``. A wide-spread name
    must therefore carry a larger forecast to earn its place, which is the
    screen a quantile cut cannot express -- the tail of a forecast
    distribution is not the tail of an edge-to-cost distribution whenever the
    two are correlated, and in equities they are.

    ``forecast`` MUST be in the units of the realized return, the same units
    as ``cost``; a rank-unit forecast compared against a return-unit spread
    makes this threshold meaningless rather than merely mis-scaled.

    The edge is the forecast less that decision's cross-sectional median over
    the eligible names, matching the centring ``select_book("all")`` uses: the
    common level is a market call the cross-sectional forecast is not making.
    ``multiple`` is left free rather than fixed at the round-trip 2x because
    the holding period, and so the number of crossings an edge must fund, is
    itself a configuration axis.
    """
    if not np.isfinite(multiple) or multiple < 0:
        raise ValueError("multiple must be finite and non-negative")
    values = np.asarray(forecast, dtype=np.float64)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    if values.shape != usable.shape or values.ndim != 2:
        raise ValueError("forecast and eligible must be equally shaped 2-D arrays")
    charge = np.asarray(cost, dtype=np.float64)
    if charge.shape != values.shape:
        raise ValueError(f"cost must have shape {values.shape}, got {charge.shape}")
    # A name without a quote has no cost to clear, so it cannot be shown to be
    # worth trading; treating a missing cost as zero would select it always.
    usable &= np.isfinite(charge)

    signs = np.zeros(values.shape, dtype=np.int8)
    for decision in range(values.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) < 2:
            continue
        edge = values[decision, row] - np.median(values[decision, row])
        clears = np.abs(edge) > multiple * charge[decision, row]
        if long_only:
            clears &= edge > 0
        signs[decision, row[clears]] = np.sign(edge[clears]).astype(np.int8)
    return signs


def persistent_selection(
    forecast: ArrayLike,
    eligible: ArrayLike,
    *,
    enter: float,
    exit_: float,
    cost: ArrayLike,
    long_only: bool = False,
) -> NDArray[np.int8]:
    """Open on conviction, and hold until the forecast turns against you.

    Every other rule here re-decides the whole book at every decision, which
    is only sensible when trading is free. It is not: a name whose edge sits
    near the entry threshold flips in and out and pays a full round trip each
    time, for a forecast that barely moved. Separating the decision to OPEN
    from the decision to CLOSE is what removes that churn, and it is why this
    rule carries state across decisions while the others are pointwise.

    ``enter`` is the multiple of a name's own cost its edge must clear to open
    a position. ``exit_`` is the multiple the edge must reach ON THE OPPOSITE
    SIDE before the position is closed, so one signed number spans the three
    behaviours worth having:

    ``exit_ == 0``
        close as soon as the forecast changes sign -- hold a name until it is
        expected to go the other way.
    ``exit_ > 0``
        tolerate a mild adverse forecast, closing only on a reversal large
        enough to be worth the round trip. Cheapest in turnover.
    ``exit_ < 0``
        close when conviction merely decays, before any reversal, at
        ``|exit_|`` times cost. Closest to the pointwise rules.

    Holding a name is free, so the exit test deliberately does not re-check
    ``enter``: a position already paid for is worth keeping on weaker evidence
    than it took to open, which is the whole asymmetry a cost creates.

    A name that stops being eligible is closed, because an unquoted market is
    not a position one can choose to keep. Returned signs are the book HELD
    after each decision, so an allocator sizes them exactly as it sizes the
    pointwise rules' output.
    """
    if not np.isfinite(enter) or enter < 0:
        raise ValueError("enter must be finite and non-negative")
    if not np.isfinite(exit_):
        raise ValueError("exit_ must be finite")
    values = np.asarray(forecast, dtype=np.float64)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    if values.shape != usable.shape or values.ndim != 2:
        raise ValueError("forecast and eligible must be equally shaped 2-D arrays")
    charge = np.asarray(cost, dtype=np.float64)
    if charge.shape != values.shape:
        raise ValueError(f"cost must have shape {values.shape}, got {charge.shape}")
    usable &= np.isfinite(charge)

    signs = np.zeros(values.shape, dtype=np.int8)
    held = np.zeros(values.shape[1], dtype=np.int8)
    for decision in range(values.shape[0]):
        row = np.flatnonzero(usable[decision])
        held[~usable[decision]] = 0
        if len(row) < 2:
            signs[decision] = held
            continue
        edge = np.zeros(values.shape[1])
        edge[row] = values[decision, row] - np.median(values[decision, row])

        # Edge measured in the direction of the position: positive while the
        # forecast still favours what is held, negative once it has turned.
        favour = held * edge
        keep = (held != 0) & usable[decision] & (favour > -exit_ * charge[decision])
        held = np.where(keep, held, 0).astype(np.int8)

        opening = usable[decision] & (held == 0) & (
            np.abs(edge) > enter * charge[decision])
        if long_only:
            opening &= edge > 0
        held = np.where(opening, np.sign(edge), held).astype(np.int8)
        signs[decision] = held
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
