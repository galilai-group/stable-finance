"""How an order becomes a fill: at the touch, or at the return's own benchmark.

The fill rule is half of the execution model (:mod:`.costs` is the other):
an assumption about the market, not a fact recorded in the dataset, and used
by the trades stage to turn orders into fills. Each mode declares what it
needs of the market in :data:`FILL_REQUIRES`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.dataset.schema import MissingRolesError
from stable_finance.evaluation.market_data import (
    MarketData,
    require_market_aligned,
    requires,
    _validated_quotes,
)
from stable_finance.evaluation.trades.contract import Fills, Orders


#: What each fill mode needs of the market. ``touch`` prices a buy at the
#: ask and a sell at the bid, so it needs both. ``benchmark`` fills at the
#: price the realized return starts from, which needs no column at all.
FILL_REQUIRES: dict[str, tuple[str, ...]] = {
    "touch": ("bid_price", "ask_price"),
    "benchmark": (),
}


@requires("bid_price", "ask_price")
def simulate_fills(orders: Orders, best_bid: ArrayLike | MarketData | None = None,
                   best_ask: ArrayLike | None = None, *, at: str = "touch") -> Fills:
    """Fill every order in full, at the touch or at the benchmark, or not at all.

    ``at="touch"``: buys execute at the best ask and sells at the best bid.
    Quotes have shape ``(n_decisions, n_assets)`` and apply to every horizon.
    An order whose quote is missing or non-positive is left unfilled. Pass the
    two quote arrays, or :class:`MarketData` in place of both; a market
    without ``bid_price`` and ``ask_price`` is refused, since a fill at the
    touch has no price without them and a half-spread alone cannot say which
    side of mid each fill landed.

    ``at="benchmark"``: every order is worked over the objective's
    measurement window and fills at that window's price -- the price the
    realized return is measured from (the one-minute VWAP after the decision,
    by default). It therefore costs nothing relative to the return, and needs
    no market column. Its realism depends on two things the simulator cannot
    see: the order being small against the window's volume, and the return's
    price being one an order can actually get (a VWAP can be targeted; a
    time-averaged midpoint, the price of a univariate series, cannot without
    resting orders that may not fill). Pass market data with ``mid`` (or both
    quotes) to record decision-time prices, in which case a name with no
    midpoint at a decision is left unfilled; with no market, every order
    fills and prices are recorded in units of the benchmark (1.0).

    Depth, partial fills, market impact, and commissions are outside this
    simulator.
    """
    if at not in FILL_REQUIRES:
        raise ValueError(f"at must be one of {tuple(FILL_REQUIRES)}")
    if isinstance(best_bid, MarketData):
        if best_ask is not None:
            raise ValueError("pass market data or two quote arrays, not both")
        market = best_bid
        try:
            market.require(*FILL_REQUIRES[at], purpose=f"simulate_fills(at={at!r})")
        except MissingRolesError as error:
            raise MissingRolesError(
                f"{error}; at='benchmark' fills without quotes, at the price the "
                "return is measured from") from None
        require_market_aligned(market, orders)
        if at == "touch":
            best_bid, best_ask = market.get("bid_price"), market.get("ask_price")
        else:
            best_bid = market.get("mid") if market.has("mid") else None
    elif at == "touch" and (best_bid is None or best_ask is None):
        raise TypeError("simulate_fills(at='touch') needs best_bid and best_ask, "
                        "or market data with both; at='benchmark' needs neither")
    elif at == "benchmark" and best_bid is not None:
        # Two quote arrays given for a benchmark fill: record their midpoint.
        if best_ask is None:
            raise TypeError("pass both quotes, market data, or nothing for at='benchmark'")
        bid, ask = _validated_quotes(best_bid, best_ask)
        best_bid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2, np.nan)

    requested = np.nan_to_num(orders.values, nan=0.0)
    expected = orders.values.shape[:2]
    if at == "benchmark":
        if best_bid is None:
            mid = np.ones(expected)
        else:
            mid = np.asarray(best_bid, dtype=np.float64)
            if mid.shape != expected:
                raise ValueError(f"mid must have shape {expected}")
        priced = np.isfinite(mid) & (mid > 0)
        fillable = (requested != 0) & priced[:, :, None]
        price = np.where(fillable, mid[:, :, None], np.nan)
        return Fills(np.where(fillable, requested, 0.0), price, price,
                     orders.decisions, orders.assets, orders.horizons)

    bid, ask = _validated_quotes(best_bid, best_ask)
    if bid.shape != expected:
        raise ValueError(f"best_bid and best_ask must have shape {expected}")
    quoted = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
    fillable = (requested != 0) & quoted[:, :, None]
    quantity = np.where(fillable, requested, 0.0)
    touch = np.where(requested > 0, ask[:, :, None], bid[:, :, None])
    price = np.where(fillable, touch, np.nan)
    mid = np.where(fillable, ((bid + ask) / 2)[:, :, None], np.nan)
    return Fills(quantity, price, mid, orders.decisions, orders.assets, orders.horizons)
