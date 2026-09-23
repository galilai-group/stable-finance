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
same axis. It is preferred over the flat add-ons for reasoning about size or
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

__all__ = ["COST_MODELS", "trading_cost"]

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
