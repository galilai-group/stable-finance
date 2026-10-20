"""The execution model: orders, how they fill, and what a trade costs.

One assumption about the market, written once and used twice. The trades
stage turns target weights into :class:`Orders`; :func:`simulate_fills` says
where each order executed; :func:`trading_cost` says what that execution cost
per unit of weight, and is what the cost-aware allocator is charged, so a
configuration cannot optimise against one cost and be measured under another.

EXECUTION QUALITY IS ONE NUMBER, the EFQ: the effective half-spread actually
paid over the half-spread quoted when the order was sent. It is stored as a
FRACTION, because it is used as a multiplier; the literature quotes it as a
percent, and figures should display it that way.

* ``efq=0`` fills at the midpoint, at the price the realized return is
  measured from. It needs no market column, which is what lets a univariate
  series or a quote-less dataset run a backtest at all.
* ``0 < efq < 1`` is the fraction of the spread an order pays when part of it
  rests rather than crosses; :data:`EFQ_INFORMED_RANGE` is where an informed
  order routed to the market lands.
* ``efq=1`` crosses the whole quoted spread: buys at the ask, sells at the bid.
* ``efq > 1`` is where size and crowding live (see
  :func:`effective_half_spread`).

``efq`` may be a scalar or an array that broadcasts to ``(decision, asset)``,
so a per-name or per-decision measure of execution quality plugs in without a
new model. Costs are one-way, per unit of weight traded: a round trip pays
twice. A net return at any EFQ is ``gross - efq * cost``, so a sensitivity
curve costs one backtest rather than one per point.

Depth, partial fills, market impact beyond the spread, and commissions are
outside this model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.schema import MissingRolesError
from stable_finance.evaluation.market_data import (
    MarketData,
    require_market_aligned,
    requires,
    _validated_quotes,
)
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = [
    "COST_MODELS",
    "COST_MODEL_REQUIRES",
    "EFQ_DIRECT_ACCESS",
    "EFQ_INFORMED_RANGE",
    "EFQ_LIT_EXCHANGE",
    "Fills",
    "Orders",
    "effective_half_spread",
    "fill_requires",
    "orders_from_weights",
    "simulate_fills",
    "trading_cost",
]


#: EFQ measured for direct market access, from Levy (2022), a randomized
#: controlled trial that routed real orders through Interactive Brokers:
#: 52.15% on buys and 55.29% on sells (Table 3). This is what an order pays
#: when it is routed to the market rather than sold to a wholesaler.
EFQ_DIRECT_ACCESS = 0.55

#: EFQ on the lit exchanges, from the same trial's venue breakdown
#: (Table 3 Panel C): 78.94% at NASDAQ and 88.25% at NYSE. An order that
#: cannot rest in a dark pool pays close to the whole quoted spread.
EFQ_LIT_EXCHANGE = 0.88

#: The band to report an INFORMED strategy in, direct access to lit exchange.
#:
#: Levy's much lower figures -- 20% pooled across PFOF brokers, and 0.55% at
#: TD Ameritrade, i.e. essentially a midpoint fill -- are NOT available here,
#: and the reason is the mechanism that produces them. Price improvement is
#: offered because retail flow is uninformed: a wholesaler who can separate
#: retail from the rest will quote inside the spread to one and not the
#: other (Kyle 1985; Glosten and Milgrom 1985). A cross-sectional forecast
#: is exactly the informed side of that split, so the retail numbers are a
#: floor it cannot claim. The same trial's decomposition says the same
#: thing arithmetically: of a 10.53 bp effective spread, 7.22 bp is
#: permanent price impact and only 3.30 bp is realised spread (Table 1), and
#: impact is the part an informed order cannot route around.
EFQ_INFORMED_RANGE = (EFQ_DIRECT_ACCESS, EFQ_LIT_EXCHANGE)

#: The cost models a backtest configuration may name. ``mid`` charges nothing
#: anywhere; ``cross`` charges the name's own quoted half-spread, scaled by
#: the EFQ. Other measures (per-name impact, say) belong here as new names.
COST_MODELS = ("mid", "cross")

#: What each cost model needs of the market.
COST_MODEL_REQUIRES: dict[str, tuple[str, ...]] = {
    "mid": (),
    "cross": ("half_spread",),
}


def _efq_array(efq: ArrayLike, name: str = "efq") -> NDArray[np.float64]:
    """An EFQ as a float array, refused unless finite and non-negative."""
    array = np.asarray(efq, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    return array


@dataclass(frozen=True)
class Orders:
    """Signed trade requests on the :class:`PortfolioWeights` grid.

    ``values`` is the requested change in weight, as a fraction of portfolio
    equity: positive buys, negative sells, zero and NaN mean no order. Convert
    to shares outside this contract by multiplying by equity and dividing by
    the reference price.
    """

    values: NDArray[np.floating]
    decisions: NDArray
    assets: NDArray
    horizons: NDArray[np.integer]

    def __post_init__(self) -> None:
        contract = ForwardReturns(self.values, self.decisions, self.assets, self.horizons)
        object.__setattr__(self, "values", contract.values)
        object.__setattr__(self, "decisions", contract.decisions)
        object.__setattr__(self, "assets", contract.assets)
        object.__setattr__(self, "horizons", contract.horizons)


@dataclass(frozen=True)
class Fills:
    """Executions produced from :class:`Orders`.

    ``quantity`` is the filled weight change (zero where nothing traded),
    ``price`` the execution price (NaN where nothing traded), and ``mid_price``
    the reference midpoint used to measure execution cost. All three share the
    ``(n_decisions, n_assets, n_horizons)`` grid.
    """

    quantity: NDArray[np.floating]
    price: NDArray[np.floating]
    mid_price: NDArray[np.floating]
    decisions: NDArray
    assets: NDArray
    horizons: NDArray[np.integer]

    def __post_init__(self) -> None:
        contract = ForwardReturns(self.quantity, self.decisions, self.assets, self.horizons)
        price = np.asarray(self.price, dtype=np.float64)
        mid_price = np.asarray(self.mid_price, dtype=np.float64)
        for name, array in (("price", price), ("mid_price", mid_price)):
            if array.shape != contract.values.shape:
                raise ValueError(
                    f"{name} must have shape {contract.values.shape}, got {array.shape}"
                )
        if not np.isfinite(contract.values).all():
            raise ValueError("quantity must be finite; use 0 for unfilled orders")
        traded = contract.values != 0
        if not (np.isfinite(price[traded]) & (price[traded] > 0)).all():
            raise ValueError("every non-zero fill needs a positive price")
        if not (np.isfinite(mid_price[traded]) & (mid_price[traded] > 0)).all():
            raise ValueError("every non-zero fill needs a positive mid_price")
        object.__setattr__(self, "quantity", contract.values)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "mid_price", mid_price)
        object.__setattr__(self, "decisions", contract.decisions)
        object.__setattr__(self, "assets", contract.assets)
        object.__setattr__(self, "horizons", contract.horizons)


def _initial_positions(initial: ArrayLike | None, shape: tuple[int, int]) -> np.ndarray:
    if initial is None:
        return np.zeros(shape, dtype=np.float64)
    array = np.asarray(initial, dtype=np.float64)
    try:
        array = np.broadcast_to(array, shape).astype(np.float64)
    except ValueError as error:
        raise ValueError(
            f"initial must broadcast to {shape}, got shape {array.shape}"
        ) from error
    if not np.isfinite(array).all():
        raise ValueError("initial positions must be finite")
    return array


def orders_from_weights(
    weights: PortfolioWeights, *, initial: ArrayLike | None = None
) -> Orders:
    """Orders that move the portfolio along the target weight path.

    Each decision's order is the target weight minus the previous decision's
    target weight; the first decision trades from ``initial`` (default cash),
    which broadcasts to ``(n_assets, n_horizons)``. A NaN target is treated as
    zero exposure. Orders are generated from targets, not fills, so a missed
    fill is not retried at the next decision; feed realized positions back
    through ``initial`` to do that.
    """
    target = np.nan_to_num(weights.values, nan=0.0)
    n_decisions, n_assets, n_horizons = target.shape
    start = _initial_positions(initial, (n_assets, n_horizons))
    previous = np.concatenate([start[None], target[:-1]], axis=0)
    return Orders(
        target - previous, weights.decisions, weights.assets, weights.horizons
    )


def fill_requires(efq: ArrayLike = 1.0) -> tuple[str, ...]:
    """What :func:`simulate_fills` needs of the market at this EFQ.

    A midpoint fill needs nothing: it is the return's own starting price. Any
    positive EFQ is priced off the spread, which needs both quotes or a
    midpoint with a half-spread.
    """
    return () if not _efq_array(efq).any() else ("mid", "half_spread")


@requires("mid", "half_spread")
def simulate_fills(orders: Orders, best_bid: ArrayLike | MarketData | None = None,
                   best_ask: ArrayLike | None = None, *, efq: ArrayLike = 1.0) -> Fills:
    """Fill every order in full at ``mid * (1 +/- efq * half_spread)``, or not at all.

    Buys pay above the midpoint and sells receive below it, by ``efq`` times
    the quoted half-spread. ``efq=1`` is the touch: a buy at the ask and a
    sell at the bid. ``efq=0`` is a midpoint fill, worked over the objective's
    measurement window at the price the realized return is measured from, and
    so costs nothing relative to that return. ``efq`` is a scalar or an array
    that broadcasts to ``(n_decisions, n_assets)``.

    Pass the two quote arrays, or :class:`MarketData` in place of both. A
    positive EFQ needs a spread: a market without one is refused, since a fill
    has no price without it. At ``efq=0`` no market is needed. With one, the
    decision-time midpoint is recorded and a name without one is left unfilled;
    without one every order fills and prices are recorded in units of the
    benchmark (1.0). At a positive EFQ, an order whose quote is missing or
    non-positive is left unfilled.

    A midpoint fill's realism depends on two things the simulator cannot see:
    the order being small against the window's volume, and the return's price
    being one an order can actually get (a VWAP can be targeted; a
    time-averaged midpoint, the price of a univariate series, cannot without
    resting orders that may not fill).
    """
    efq = _efq_array(efq)
    expected = orders.values.shape[:2]
    try:
        efq = np.broadcast_to(efq, expected)
    except ValueError:
        raise ValueError(f"efq must broadcast to {expected}, got shape {efq.shape}") from None
    crossing = bool(efq.any())

    mid = spread = None
    if isinstance(best_bid, MarketData):
        if best_ask is not None:
            raise ValueError("pass market data or two quote arrays, not both")
        market = best_bid
        try:
            market.require(*fill_requires(efq), purpose="simulate_fills(efq>0)")
        except MissingRolesError as error:
            raise MissingRolesError(
                f"{error}; efq=0 fills without quotes, at the price the "
                "return is measured from") from None
        require_market_aligned(market, orders)
        if market.has("mid"):
            mid = market.get("mid")
        if crossing:
            spread = market.get("half_spread")
    elif best_bid is not None or best_ask is not None:
        if best_bid is None or best_ask is None:
            raise TypeError("pass both quotes, market data, or nothing")
        bid, ask = _validated_quotes(best_bid, best_ask)
        if bid.shape != expected:
            raise ValueError(f"best_bid and best_ask must have shape {expected}")
        quoted = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
        mid = np.where(quoted, (bid + ask) / 2, np.nan)
        spread = np.where(quoted, (ask - bid) / 2 / np.where(quoted, mid, 1.0), np.nan)
    elif crossing:
        raise TypeError("simulate_fills(efq>0) needs best_bid and best_ask, or market "
                        "data with a spread; efq=0 needs neither")

    if mid is None:
        mid = np.ones(expected)
    mid = np.asarray(mid, dtype=np.float64)
    if mid.shape != expected:
        raise ValueError(f"mid must have shape {expected}")
    priced = np.isfinite(mid) & (mid > 0)
    if crossing:
        spread = np.asarray(spread, dtype=np.float64)
        priced &= np.isfinite(spread) | (efq == 0)
        paid = np.where(efq == 0, 0.0, np.nan_to_num(spread, nan=0.0) * efq)
    else:
        paid = np.zeros(expected)

    requested = np.nan_to_num(orders.values, nan=0.0)
    fillable = (requested != 0) & priced[:, :, None]
    side = np.sign(requested)
    price = np.where(fillable, mid[:, :, None] * (1 + side * paid[:, :, None]), np.nan)
    reference = np.where(fillable, mid[:, :, None], np.nan)
    return Fills(np.where(fillable, requested, 0.0), price, reference,
                 orders.decisions, orders.assets, orders.horizons)


def trading_cost(
    name: str, half_spread: ArrayLike, *, multiple: ArrayLike = 1.0
) -> NDArray[np.float64]:
    """Cost per unit of weight traded under one named model.

    A name without a two-sided market gets an infinite cost rather than a
    missing one. Infinity is the honest value: the allocator's proximal step
    reads it as "this asset cannot be traded at this decision" and holds the
    previous weight, whereas a NaN would propagate into the objective and a
    zero would quietly make an unquoted name the cheapest thing to trade.

    ``multiple`` is the EFQ. It may be a scalar or an array that broadcasts
    to ``(decision, asset)``, so a per-name or per-decision execution quality
    needs no new cost model. An UNQUOTED name
    stays at infinite cost whatever the multiple, including zero: a multiple
    is a statement about execution quality or size, not a claim that a market
    which was not quoting can be traded in. That is also why ``multiple=0``
    is not a synonym for ``mid`` -- ``mid`` asserts the whole universe trades
    free, while ``cross`` at zero keeps the untradable names untradable.
    """
    multiple = _efq_array(multiple, "multiple")
    if name == "mid":
        spread = np.asarray(half_spread, dtype=np.float64)
        return np.zeros(spread.shape, dtype=np.float64)
    if name not in COST_MODELS:
        raise ValueError(f"unknown cost model {name!r}; choose from {COST_MODELS}")

    spread = np.asarray(half_spread, dtype=np.float64)
    if spread.ndim != 2:
        raise ValueError("half_spread must be two-dimensional")
    if np.any(np.isfinite(spread) & (spread < 0)):
        raise ValueError("half_spread cannot be negative")
    quoted = np.isfinite(spread)
    charge = np.broadcast_to(np.where(quoted, spread, 0.0) * multiple, spread.shape)
    return np.where(quoted, charge, np.inf)


def effective_half_spread(
    quoted_half_spread: ArrayLike, efq: ArrayLike
) -> NDArray[np.float64]:
    """The half-spread actually paid, at an effective-to-quoted ratio ``efq``.

    EFQ is the microstructure measure of execution quality: the effective
    half-spread ``|fill - mid|`` divided by the half-spread quoted when the
    order was sent. It is one scalar spanning every execution regime a
    strategy can be run in -- ``0`` is a midpoint fill (0% in the units the
    literature quotes), ``EFQ_INFORMED_RANGE`` is where an informed order
    routed to the market lands, and ``1`` is paying the whole quoted spread.

    ABOVE 1 IS NOT AN EXTRAPOLATION. Two ordinary mechanisms put EFQ past
    100%, and both apply to a strategy like this one. An order larger than
    the quoted depth walks the book, so the average fill is worse than the
    touch it started at -- the quoted half-spread prices the first lot, not
    the whole order. And a signal is rarely traded alone: when others act on
    the same information, they compete for the same liquidity in the same
    direction, so the quote moves away before the order completes and the
    realised half-spread exceeds the one that was displayed. EFQ above 100%
    is therefore where capacity and crowding live, which is why the axis is
    worth sweeping past the point where a single small order would stop.

    This is the same scaling ``trading_cost(multiple=...)`` applies, named.
    It exists separately because a study that reports a result AS A FUNCTION
    of EFQ is making a different and stronger claim than one that picks a
    cost model: it says under which execution regimes the result holds, and
    a reader who knows their own venue's EFQ can read off their own answer.

    An unquoted name stays untradable at every ratio, including zero, for the
    reason given in ``trading_cost``.
    """
    efq = _efq_array(efq)
    spread = np.asarray(quoted_half_spread, dtype=np.float64)
    if np.any(np.isfinite(spread) & (spread < 0)):
        raise ValueError("half_spread cannot be negative")
    quoted = np.isfinite(spread)
    return np.where(quoted, np.where(quoted, spread, 0.0) * efq, np.inf)
