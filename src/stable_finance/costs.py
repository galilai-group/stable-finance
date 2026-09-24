"""What a trade costs, as a fraction of mid per unit of weight traded.

The cost model is the single decision with the largest effect on a Sharpe
ratio and the smallest connection to the model being evaluated, which is why
it is an axis rather than a constant. Every model here returns a cost on the
``(decision, asset)`` grid that the allocator charges when it moves a weight
and the evaluator charges when it books a fill, so a configuration cannot
optimise against one cost and be measured under another.

Costs are one-way, per unit of weight traded: a full round trip pays twice.
``mid`` is the frictionless upper bound every backtest reports by default;
the ``cross`` family pays the name's own quoted half-spread, optionally plus
a flat allowance for fees and impact.

``multiple`` scales any model's cost and is the continuous version of the
same axis. It is the EFFECTIVE-TO-QUOTED SPREAD RATIO (E/Q): the standard
microstructure measure of execution quality, defined as the effective
half-spread actually paid over the half-spread that was quoted when the
order was sent. ``effective_half_spread`` is the same statement written out. It is preferred over the flat add-ons for reasoning about size or
execution quality, for two reasons. A flat ``+1bp`` charges a 2 bp-spread
mega-cap and a 40 bp-spread small-cap the same penalty, whereas impact scales
WITH illiquidity and the quoted spread is the only observable proxy for it.
And one scalar spans the whole range: ``0`` is frictionless, values below 1
are the fraction of notional that had to cross rather than rest, ``1`` is
crossing everything, and above 1 is the size regime where a trade moves the
price it gets. A net return at any multiple is ``gross - multiple * cost``,
so a sensitivity curve costs one backtest rather than one per point.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "COST_MODELS",
    "EQ_EMPIRICAL_RANGE",
    "breakeven_eq",
    "effective_half_spread",
    "trading_cost",
]

#: The range the effective-to-quoted spread ratio is usually measured in for
#: liquid US equities. A marketable order does not reliably pay the full
#: quoted half-spread: some of it executes against price improvement, hidden
#: midpoint liquidity or a quote that moved in its favour, so the realised
#: (effective) half-spread is typically a FRACTION of the quoted one. Venues
#: differ enough that this is a range and not a constant, which is precisely
#: why it belongs on an axis: a result that survives only at the bottom of
#: this band is a result about venue selection, not about a forecast.
EQ_EMPIRICAL_RANGE = (0.5, 1.0)

#: The cost models a backtest configuration may name, and their flat add-on in
#: basis points on top of the quoted half-spread. ``mid`` charges nothing.
COST_MODELS = ("mid", "cross", "cross+0.5bps", "cross+1bps", "cross+2bps")

_ADD_ON_BPS = {"cross": 0.0, "cross+0.5bps": 0.5, "cross+1bps": 1.0,
               "cross+2bps": 2.0}


def trading_cost(
    name: str, half_spread: ArrayLike, *, multiple: float = 1.0
) -> NDArray[np.float64]:
    """Cost per unit of weight traded under one named model.

    A name without a two-sided market gets an infinite cost rather than a
    missing one. Infinity is the honest value: the allocator's proximal step
    reads it as "this asset cannot be traded at this decision" and holds the
    previous weight, whereas a NaN would propagate into the objective and a
    zero would quietly make an unquoted name the cheapest thing to trade.

    The add-on is charged only where a quote exists, so ``cross+2bps`` never
    prices a trade the market could not have filled.

    ``multiple`` scales the whole charge, add-on included. An UNQUOTED name
    stays at infinite cost whatever the multiple, including zero: a multiple
    is a statement about execution quality or size, not a claim that a market
    which was not quoting can be traded in. That is also why ``multiple=0``
    is not a synonym for ``mid`` -- ``mid`` asserts the whole universe trades
    free, while ``cross`` at zero keeps the untradable names untradable.
    """
    if not np.isfinite(multiple) or multiple < 0:
        raise ValueError("multiple must be finite and non-negative")
    if name == "mid":
        spread = np.asarray(half_spread, dtype=np.float64)
        return np.zeros(spread.shape, dtype=np.float64)
    if name not in _ADD_ON_BPS:
        raise ValueError(f"unknown cost model {name!r}; choose from {COST_MODELS}")

    spread = np.asarray(half_spread, dtype=np.float64)
    if spread.ndim != 2:
        raise ValueError("half_spread must be two-dimensional")
    if np.any(np.isfinite(spread) & (spread < 0)):
        raise ValueError("half_spread cannot be negative")
    quoted = np.isfinite(spread)
    charge = (spread + _ADD_ON_BPS[name] / 1e4) * multiple
    return np.where(quoted, charge, np.inf)


def effective_half_spread(
    quoted_half_spread: ArrayLike, eq: float
) -> NDArray[np.float64]:
    """The half-spread actually paid, at an effective-to-quoted ratio of ``eq``.

    E/Q is the microstructure measure of execution quality: the effective
    half-spread ``|fill - mid|`` divided by the half-spread quoted when the
    order was sent. It is one scalar spanning every execution regime a
    strategy can be run in -- ``0`` is a midpoint fill, ``EQ_EMPIRICAL_RANGE``
    is where a marketable order in a liquid US name usually lands, ``1`` is
    paying the full quoted spread, and above ``1`` is the regime where the
    order is large enough to move the price it gets.

    This is the same scaling ``trading_cost(multiple=...)`` applies, named.
    It exists separately because a study that reports a result AS A FUNCTION
    of E/Q is making a different and stronger claim than one that picks a
    cost model: it says under which execution regimes the result holds, and
    a reader who knows their own venue's E/Q can read off their own answer.

    An unquoted name stays untradable at every ratio, including zero, for the
    reason given in ``trading_cost``.
    """
    if not np.isfinite(eq) or eq < 0:
        raise ValueError("eq must be finite and non-negative")
    spread = np.asarray(quoted_half_spread, dtype=np.float64)
    if np.any(np.isfinite(spread) & (spread < 0)):
        raise ValueError("half_spread cannot be negative")
    return np.where(np.isfinite(spread), spread * eq, np.inf)


def breakeven_eq(gross: ArrayLike, cost_at_unit_eq: ArrayLike) -> float:
    """The E/Q at which the expected net return reaches zero.

    Costs are linear in E/Q, so a strategy's whole sensitivity to execution
    quality collapses to one number: it makes money exactly while

        E/Q < mean(gross) / mean(cost at E/Q = 1)

    This is the quantity to report. A Sharpe ratio quoted at one cost
    assumption invites the reader to argue about the assumption; a break-even
    E/Q hands them the assumption as a threshold and lets them compare it to
    what their own execution achieves. It is also where the weak-form
    efficiency reading becomes quantitative -- a break-even close to 1 says
    the edge is almost exactly the cost of taking it.

    Returns ``inf`` when nothing is ever charged and ``nan`` when the gross
    return is not positive, since no execution quality rescues that.
    """
    gross = np.asarray(gross, dtype=np.float64)
    cost = np.asarray(cost_at_unit_eq, dtype=np.float64)
    if gross.shape != cost.shape:
        raise ValueError("gross and cost must have the same shape")
    mean_gross, mean_cost = float(np.mean(gross)), float(np.mean(cost))
    if mean_gross <= 0:
        return float("nan")
    return float("inf") if mean_cost <= 0 else mean_gross / mean_cost
