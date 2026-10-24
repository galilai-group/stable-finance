"""Whether a trade is worth making: forecasts held against their own cost.

A portfolio stage says what to hold. These rules decide whether moving there
pays: ``edge_to_cost`` measures how far each forecast clears the cost of
acting on it, ``threshold_selection`` holds a name only when it does, and
``persistent_selection`` separates the decision to open a position from the
decision to close it, so a name near the bar does not pay a round trip each
time its forecast wobbles.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import MarketData, market_quantity, requires


@requires("half_spread")
def edge_to_cost(
    forecast: ArrayLike,
    half_spread: ArrayLike | MarketData,
    *,
    eligible: ArrayLike | None = None,
) -> np.ndarray:
    """How far each name's forecast clears the cost of acting on it.

    The ratio of ``|forecast - cross-sectional median|`` to that name's own
    one-way cost, on the ``(decision, asset)`` grid. Above 1 the expected move
    exceeds a single crossing; a round trip needs 2.

    THIS IS THE STATISTIC TO LEAD WITH when reporting whether a forecast is
    tradable, in preference to a Sharpe ratio, and the reason is sample size.
    A Sharpe is a time-series statistic over a few hundred periods, with a
    standard error near ``sqrt((1 + S**2 / 2) / years)``; at two or three
    years a true Sharpe below 0.5 cannot be distinguished from zero however
    carefully it is computed, so reporting one as the headline claims
    precision the data does not carry. This ratio is cross-sectional over
    every (decision, name) pair -- hundreds of thousands of them -- and is
    pinned orders of magnitude more tightly. It also answers a question a
    Sharpe cannot: WHY a forecast is or is not tradable, in units a reader
    can check against a quoted market.

    ``forecast`` MUST be in the units of the realized return, as
    ``half_spread`` is; a rank-unit forecast makes this ratio meaningless
    rather than merely mis-scaled. A name with no two-sided market has no
    finite cost to clear and is returned as NaN rather than as an infinite
    edge, so a summary over the grid is not dominated by untradable names.
    ``half_spread`` may be :class:`MarketData` with a spread or both quotes.
    """
    half_spread = market_quantity(half_spread, "half_spread", stage=edge_to_cost)
    values = np.asarray(forecast, dtype=np.float64)
    spread = np.asarray(half_spread, dtype=np.float64)
    if values.ndim != 2 or spread.shape != values.shape:
        raise ValueError("forecast and half_spread must be equally shaped 2-D arrays")
    if np.any(np.isfinite(spread) & (spread < 0)):
        raise ValueError("half_spread cannot be negative")
    usable = np.isfinite(values) & np.isfinite(spread) & (spread > 0)
    if eligible is not None:
        usable &= np.asarray(eligible, dtype=bool)

    out = np.full(values.shape, np.nan)
    for decision in range(values.shape[0]):
        row = np.flatnonzero(usable[decision])
        if len(row) < 2:
            continue
        edge = values[decision, row] - np.median(values[decision, row])
        out[decision, row] = np.abs(edge) / spread[decision, row]
    return out


@requires("half_spread")
def threshold_selection(
    forecast: ArrayLike,
    eligible: ArrayLike,
    *,
    multiple: float,
    cost: ArrayLike | MarketData,
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
    cost = market_quantity(cost, "half_spread", stage=threshold_selection)
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


@requires("half_spread")
def persistent_selection(
    forecast: ArrayLike,
    eligible: ArrayLike,
    *,
    enter: float,
    exit_: float,
    cost: ArrayLike | MarketData,
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
    cost = market_quantity(cost, "half_spread", stage=persistent_selection)
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
