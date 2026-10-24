"""The execution model: orders, how they fill, and what a trade costs.

One assumption about the market, written once and used twice. The trades
stage turns target weights into :class:`Orders`; :func:`execute` says where
each order filled; :func:`trading_cost` says what that costs per unit of
weight, and is what the cost-aware allocator is charged, so a configuration
cannot optimise against one cost and be measured under another.

EACH MODEL NEEDS A LEVEL OF MARKET DATA (see
:data:`~stable_finance.dataset.market.MARKET_LEVELS`), and is refused without
it:

``mid``    fills at the price the realized return is measured from, at no
           cost. Needs nothing: a bare price series runs.
``cross``  pays ``efq`` times the half-spread: quoted, or estimated from bars.
``book``   walks the depth for the order's size, so a large order pays more
           per share than a small one and may only partly fill. Needs depth
           and the portfolio's ``equity``, to turn weight into shares.
``impact`` the price-impact model of Frazzini, Israel and Moskowitz (2018),
           calibrated on $1.7 trillion of live institutional trades: cost in
           basis points is ``a + b x + c sqrt(x)`` in the order's size ``x``
           as a percent of the name's average daily dollar volume, and the
           order is worked over a schedule of decisions (one trading day in
           the paper) at the midpoint as it evolves. Needs ``adv`` and
           ``equity``. See :func:`impact_cost`. This is the simple
           institutional model.
``impact_stock``
           the same paper's full regression: the size terms plus the
           stock's market cap and idiosyncratic volatility, and the market's
           volatility and a time trend. Needs those too, as of each
           decision. The stock-specific institutional model; see
           :func:`impact_stock_cost`.

An order may carry a LIMIT, the most slippage it accepts as a fraction of the
decision price. The part a model can fill within it fills as usual. The rest
rests, and fills at the limit only if the execution window traded through it
(``window_best_buy`` and ``window_best_sell``); otherwise it is left unfilled.
Queue position is not modelled, so a resting fill is an upper bound.

EXECUTION QUALITY IS ONE NUMBER, the EFQ: the effective half-spread actually
paid over the half-spread quoted when the order was sent. It is stored as a
FRACTION, because it is used as a multiplier; the literature quotes it as a
percent, and figures should display it that way.

* ``efq=0`` fills at the midpoint, at the price the realized return is
  measured from.
* ``0 < efq < 1`` is the fraction of the spread an order pays when part of it
  rests rather than crosses; :data:`EFQ_INFORMED_RANGE` is where an informed
  order routed to the market lands.
* ``efq=1`` crosses the whole quoted spread: buys at the ask, sells at the bid.
* ``efq > 1`` is where size and crowding live (see
  :func:`effective_half_spread`).

``efq`` may be a scalar or an array that broadcasts to ``(decision, asset)``,
so a per-name or per-decision measure of execution quality plugs in without a
new model. Costs are one-way, per unit of weight traded: a round trip pays
twice. Market impact beyond the visible book, and commissions, are outside
every model here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import (
    ADV,
    BOOK_QUANTITIES,
    CHARACTERISTICS,
    WINDOW_QUANTITIES,
    MarketData,
    require_market_aligned,
    requires,
    _validated_quotes,
)
from stable_finance.dataset.schema import MissingRolesError
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
    "IMPACT_COEFFICIENTS",
    "IMPACT_STOCK_COEFFICIENTS",
    "ImpactStockCoefficients",
    "effective_half_spread",
    "execute",
    "impact_cost",
    "impact_stock_cost",
    "execution_requires",
    "fill_path",
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

#: The execution models a backtest configuration may name, as its
#: ``cost_model``. ``mid`` charges nothing; ``cross`` charges the name's own
#: half-spread, scaled by the EFQ; ``book`` walks the depth. Other measures
#: (per-name impact, say) belong here as new names.
COST_MODELS = ("mid", "cross", "book", "impact", "impact_stock")

#: What each model needs of the market.
COST_MODEL_REQUIRES: dict[str, tuple[str, ...]] = {
    "mid": (),
    "cross": ("half_spread",),
    "book": BOOK_QUANTITIES,
    "impact": (ADV,),
    "impact_stock": (ADV, "market_cap", *CHARACTERISTICS),
}

#: ``(a, b, c)`` of the price-impact model, in basis points: cost per dollar
#: traded is ``a + b x + c sqrt(x)`` for an order of ``x`` percent of the
#: name's average daily dollar volume. ``b`` and ``c`` are the U.S. estimates
#: of Frazzini, Israel and Moskowitz (2018), Table VII column (10), from live
#: trades over 1998-2016. ``a`` absorbs the paper's level terms (time trend,
#: market cap, idiosyncratic and market volatility) at their sample medians,
#: which the paper does not report; zero reproduces its realized U.S.
#: averages within a basis point (7.1 bps at 0.5% of daily volume, 15.4 bps
#: at 2.4%). The calibration covers 0.1% to 13% of daily volume.
IMPACT_COEFFICIENTS = (0.0, -0.53, 11.21)


class ImpactStockCoefficients(NamedTuple):
    """The stock-specific impact model: Table VII column (10), and where its level is pinned.

    Cost in basis points is

        a + size x + sqrt_size sqrt(x)
          + log_market_cap (log(1 + ME) - ref_log_market_cap)
          + idio_vol (sigma - ref_idio_vol)
          + market_vol (VIX - ref_market_vol)
          + time_trend (t - ref_time_trend)

    with ME in billions, volatilities annualized in percent, and ``t`` in
    months since June 1926. The paper reports the slopes but never the
    intercept, so the level is pinned instead: a name at the reference
    characteristics pays the simple model's cost, with ``a`` on top of that.
    """

    a: float
    size: float
    sqrt_size: float
    log_market_cap: float
    idio_vol: float
    market_vol: float
    time_trend: float
    ref_log_market_cap: float
    ref_idio_vol: float
    ref_market_vol: float
    ref_time_trend: float


#: The U.S. estimates of Table VII column (10), and reference levels. The
#: firm-size reference is the paper's own median traded firm, $4.9 billion
#: (Table I). The others are NOT reported by the paper and are ordinary
#: values for its sample: 30% idiosyncratic volatility, 17% market
#: volatility, and the sample's middle month (May 2007; June 1926 is 1).
#: Pass your own :class:`ImpactStockCoefficients` to move them.
IMPACT_STOCK_COEFFICIENTS = ImpactStockCoefficients(
    a=0.0, size=-0.53, sqrt_size=11.21, log_market_cap=-0.14, idio_vol=0.31,
    market_vol=0.12, time_trend=0.01,
    ref_log_market_cap=float(np.log1p(4.9)), ref_idio_vol=30.0, ref_market_vol=17.0,
    ref_time_trend=972.0,
)

#: Where a fill's reference price comes from, in order of preference: the
#: midpoint, else the last close, else the VWAP. With none of them, a
#: frictionless fill is recorded in units of the return's own price (1.0).
_REFERENCE_PRICES = ("mid", "close", "vwap")


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
    #: The most slippage each order accepts, as a fraction of the decision
    #: price: a buy fills only at or below ``price * (1 + limit)``, a sell only
    #: at or above ``price * (1 - limit)``. Negative asks for a better price
    #: than the decision's. NaN, or no array, is a market order.
    limit: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        contract = ForwardReturns(self.values, self.decisions, self.assets, self.horizons)
        if self.limit is not None:
            limit = np.asarray(self.limit, dtype=np.float64)
            try:
                limit = np.broadcast_to(limit, contract.values.shape).copy()
            except ValueError:
                raise ValueError(f"limit must broadcast to {contract.values.shape}, "
                                 f"got {limit.shape}") from None
            if np.isinf(limit).any() or (limit <= -1).any():
                raise ValueError("limit must be finite, NaN for a market order, "
                                 "and above -1")
            object.__setattr__(self, "limit", limit if np.isfinite(limit).any() else None)
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
    weights: PortfolioWeights, *, initial: ArrayLike | None = None,
    limit: ArrayLike | None = None,
) -> Orders:
    """Orders that move the portfolio along the target weight path.

    Each decision's order is the target weight minus the previous decision's
    target weight; the first decision trades from ``initial`` (default cash),
    which broadcasts to ``(n_assets, n_horizons)``. A NaN target is treated as
    zero exposure. Orders are generated from targets, not fills, so a missed
    fill is not retried at the next decision; feed realized positions back
    through ``initial`` to do that, or use :func:`fill_path`, which
    does. ``limit`` is passed to every order (see :class:`Orders`).
    """
    target = np.nan_to_num(weights.values, nan=0.0)
    n_decisions, n_assets, n_horizons = target.shape
    start = _initial_positions(initial, (n_assets, n_horizons))
    previous = np.concatenate([start[None], target[:-1]], axis=0)
    return Orders(
        target - previous, weights.decisions, weights.assets, weights.horizons,
        limit=limit,
    )


def fill_requires(efq: ArrayLike = 1.0) -> tuple[str, ...]:
    """What :func:`simulate_fills` needs of the market at this EFQ.

    A midpoint fill needs nothing: it is the return's own starting price. Any
    positive EFQ is priced off the spread, quoted or estimated from bars.
    """
    return () if not _efq_array(efq).any() else ("half_spread",)


def _reference_price(market: MarketData) -> NDArray[np.float64] | None:
    """The decision-time price fills are measured against, if the market has one."""
    for quantity in _REFERENCE_PRICES:
        if market.has(quantity):
            return market.get(quantity)
    return None


@requires("half_spread")
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
    has no price without it. At ``efq=0`` no market is needed. Fills are
    measured against the decision-time midpoint, else the close, else the
    VWAP (the first the market has), and a name without that price at a
    decision is left unfilled; with no price at all every order fills and
    prices are recorded in units of the benchmark (1.0). At a positive EFQ,
    an order whose spread is missing is left unfilled.

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
        mid = _reference_price(market)
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


def execution_requires(model: str, *, limits: bool = False) -> tuple[str, ...]:
    """What :func:`execute` needs of the market under ``model``.

    Limit orders also need the execution window's best prices, to say
    whether a resting order was reached.
    """
    if model not in COST_MODELS:
        raise ValueError(f"unknown execution model {model!r}; choose from {COST_MODELS}")
    return COST_MODEL_REQUIRES[model] + (WINDOW_QUANTITIES if limits else ())


def execute(orders: Orders, market: MarketData | None = None, *, model: str = "cross",
            efq: ArrayLike = 1.0, equity: float | None = None, schedule: int = 0,
            impact: tuple[float, float, float] | None = None) -> Fills:
    """Fill ``orders`` under one execution model (see the module docstring).

    ``efq`` applies to ``cross``; ``equity`` is the portfolio's value in the
    market's currency, required by ``book`` and ``impact``. ``schedule`` and
    ``impact`` apply to ``impact``: the number of decisions each order is
    worked over (``0`` fills at the decision) and the model's coefficients
    (:data:`IMPACT_COEFFICIENTS` by default). A market that lacks what the
    model needs is refused with :class:`MissingRolesError`.
    """
    limits = orders.limit is not None
    if model in ("impact", "impact_stock") and limits:
        raise ValueError("the impact model works orders over a schedule and has no "
                         "limit; pass market orders")
    needed = execution_requires(model, limits=limits)
    if market is None:
        market = MarketData.empty(orders.decisions, orders.assets)
    market.require(*needed, purpose=f"execute(model={model!r})")
    require_market_aligned(market, orders)
    if limits and _reference_price(market) is None:
        raise MissingRolesError("a limit order is priced off the decision price, "
                                f"which needs one of {', '.join(_REFERENCE_PRICES)}")
    if model == "book":
        return _walk_book(orders, market, equity)
    if model in ("impact", "impact_stock"):
        return _impact_fills(orders, market, equity, schedule, impact, model)
    fills = simulate_fills(orders, market, efq=efq if model == "cross" else 0.0)
    return _rest(orders, market, fills) if limits else fills


def _caps(orders: Orders, reference: NDArray) -> tuple[NDArray, NDArray]:
    """Each order's side and its worst acceptable price (NaN for a market order)."""
    side = np.sign(np.nan_to_num(orders.values, nan=0.0))
    limit = np.full(orders.values.shape, np.nan) if orders.limit is None else orders.limit
    return side, reference[:, :, None] * (1 + side * limit)


def _rest(orders: Orders, market: MarketData, fills: Fills) -> Fills:
    """Hold a fill to its limit; rest what could not fill within it."""
    reference = _reference_price(market)
    side, cap = _caps(orders, reference)
    limited = np.isfinite(cap)
    within = (fills.quantity != 0) & ((side * (cap - fills.price)) >= 0)
    keep = ~limited | within
    best_buy = market.get("window_best_buy")[:, :, None]
    best_sell = market.get("window_best_sell")[:, :, None]
    # A resting order fills only if the window traded THROUGH its limit.
    reached = np.where(side > 0, best_buy < cap, best_sell > cap)
    resting = limited & ~within & reached & (side != 0) & np.isfinite(reference)[:, :, None]
    requested = np.nan_to_num(orders.values, nan=0.0)
    quantity = np.where(keep, fills.quantity, np.where(resting, requested, 0.0))
    price = np.where(keep, fills.price, np.where(resting, cap, np.nan))
    mid = np.where(keep, fills.mid_price, np.where(resting, reference[:, :, None], np.nan))
    return Fills(quantity, price, mid, orders.decisions, orders.assets, orders.horizons)


def _walk_book(orders: Orders, market: MarketData, equity: float | None) -> Fills:
    """Fill each order level by level through the book, up to its limit."""
    if equity is None or not np.isfinite(equity) or equity <= 0:
        raise ValueError("the book model needs equity: the portfolio's value in the "
                         "book's currency, to turn weight into shares")
    mid = market.get("mid")
    side, cap = _caps(orders, mid)
    requested = np.nan_to_num(orders.values, nan=0.0)
    priced = (np.isfinite(mid) & (mid > 0))[:, :, None] & (requested != 0)
    wanted = np.where(priced, np.abs(requested) * equity / np.where(priced, mid[:, :, None], 1.0), 0.0)

    buy = side[..., None] > 0
    level_price = np.where(buy, market.get("book_ask_price")[:, :, None, :],
                           market.get("book_bid_price")[:, :, None, :])
    level_size = np.where(buy, market.get("book_ask_size")[:, :, None, :],
                          market.get("book_bid_size")[:, :, None, :])
    usable = np.isfinite(level_price) & np.isfinite(level_size) & (level_size > 0)
    capped = np.isfinite(cap)[..., None]
    usable &= ~capped | (side[..., None] * (cap[..., None] - level_price) >= 0)
    size = np.where(usable, level_size, 0.0)
    before = np.cumsum(size, axis=-1) - size
    take = np.clip(wanted[..., None] - before, 0.0, size)
    shares = take.sum(axis=-1)
    notional = (take * np.where(usable, level_price, 0.0)).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        walked = np.where(shares > 0, notional / shares, np.nan)
    quantity = side * shares * np.where(priced, mid[:, :, None], 0.0) / equity

    # What the book could not fill within the limit rests, as for any model.
    if orders.limit is not None:
        best = np.where(side > 0, market.get("window_best_buy")[:, :, None],
                        market.get("window_best_sell")[:, :, None])
        reached = capped[..., 0] & priced & np.where(side > 0, best < cap, best > cap)
        rest = np.where(reached, requested - quantity, 0.0)
        total = quantity + rest
        with np.errstate(invalid="ignore", divide="ignore"):
            walked = np.where(total != 0, (np.nan_to_num(walked) * quantity
                                           + np.nan_to_num(cap) * rest) / total, np.nan)
        quantity = total
    filled = quantity != 0
    return Fills(np.where(filled, quantity, 0.0), np.where(filled, walked, np.nan),
                 np.where(filled, mid[:, :, None], np.nan),
                 orders.decisions, orders.assets, orders.horizons)


def _equity(equity, model: str) -> float:
    if equity is None or not np.isfinite(equity) or equity <= 0:
        raise ValueError(f"the {model} model needs equity: the portfolio's value in the "
                         "market's currency, to turn weight into dollars")
    return float(equity)


def impact_cost(size: ArrayLike, *, coefficients: tuple[float, float, float] | None = None
                ) -> NDArray[np.float64]:
    """Price impact of an order, as a fraction of the dollars traded.

    ``size`` is the order as a percent of the name's average daily dollar
    volume, ``100 * dollars / adv``. The model is equation (8) of Frazzini,
    Israel and Moskowitz (2018), *Trading Costs*: in basis points,

        cost = a + b size + c sqrt(size)

    which they fit to the realized cost of live institutional orders (the
    average fill against the price when trading began) and which matched
    three brokers' independent cost tables out of sample. The square root
    is the concavity of price impact in size -- a trade twice as large costs
    less than twice as much per dollar -- which the paper tests against a
    linear form and prefers. What the coefficients embed is the paper's
    patience: orders were worked with resting limit orders over about one
    trading day, so this is the cost of trading THAT way, not of crossing
    the spread at once. The cost does not depend on the horizon chosen here
    for that reason; the horizon decides what the position earns meanwhile.

    A size of zero costs ``a``, the size-free part (commissions, say). The
    paper calibrates up to 13% of daily volume, and its linear term is small
    and negative (not significant), so far beyond that range the fitted curve
    would turn down. The cost is held at its peak from there on: an order
    past the calibration never gets cheaper for being larger.
    """
    a, b, c = IMPACT_COEFFICIENTS if coefficients is None else coefficients
    x = np.asarray(size, dtype=np.float64)
    if (x < 0).any():
        raise ValueError("size is unsigned")
    if b < 0 < c:
        x = np.minimum(x, (c / (2 * b)) ** 2)
    return (a + b * x + c * np.sqrt(x)) / 1e4


def impact_stock_cost(size: ArrayLike, market_cap: ArrayLike, idio_vol: ArrayLike,
                      market_vol: ArrayLike, time_trend: ArrayLike, *,
                      coefficients: ImpactStockCoefficients | None = None
                      ) -> NDArray[np.float64]:
    """Price impact with the stock's and the market's characteristics, as a fraction.

    The full model of Frazzini, Israel and Moskowitz (2018), Table VII
    column (10): :func:`impact_cost`'s size terms, plus a cost that falls
    with the firm's market cap (``market_cap`` in the market's currency),
    rises with its idiosyncratic volatility and with the market's
    volatility (both annualized, in percent), and drifts with a time trend
    (months since June 1926). See :class:`ImpactStockCoefficients` for how
    the level is pinned. A cost is never negative: a large, quiet name at a
    tiny size is free, not paid to trade.
    """
    k = IMPACT_STOCK_COEFFICIENTS if coefficients is None else ImpactStockCoefficients(*coefficients)
    base = impact_cost(size, coefficients=(k.a, k.size, k.sqrt_size))
    log_me = np.log1p(np.asarray(market_cap, dtype=np.float64) / 1e9)
    extra = (k.log_market_cap * (log_me - k.ref_log_market_cap)
             + k.idio_vol * (np.asarray(idio_vol, dtype=np.float64) - k.ref_idio_vol)
             + k.market_vol * (np.asarray(market_vol, dtype=np.float64) - k.ref_market_vol)
             + k.time_trend * (np.asarray(time_trend, dtype=np.float64) - k.ref_time_trend))
    return np.maximum(base + extra / 1e4, 0.0)


def _impact_fills(orders: Orders, market: MarketData, equity, schedule: int,
                  coefficients, model: str = "impact") -> Fills:
    """Work each order over ``schedule`` decisions at the evolving mid, plus impact.

    Orders in the same name within one schedule are one parent order: the
    paper's unit is a stock's order for the day, and the concave cost means
    slicing it into child orders and charging each alone would undercharge.
    Each order is charged the increase in its parent's total cost, per unit
    of weight, and that rate is paid as its slices fill. An order against a
    parent in flight cancels the parent's unfilled slices, which are then
    not paid for; only what fills is charged.

    What is outstanding in a name fills in equal slices over the ``schedule``
    decisions after the latest order, so one order ramps the position
    linearly. ``schedule=0`` fills at the decision, which is the paper's own
    accounting (full position at the arrival price, charged the formula).
    Slices past the last decision are never filled, and a decision with no
    mid fills nothing and leaves the rest for the decisions after it.
    """
    equity = _equity(equity, "impact")
    if not isinstance(schedule, (int, np.integer)) or isinstance(schedule, bool) or schedule < 0:
        raise ValueError("schedule must be a non-negative number of decisions")
    mid = market.get("mid") if market.has("mid") else _reference_price(market)
    if mid is None:
        raise MissingRolesError("the impact model prices fills off the midpoint, which "
                                f"needs one of {', '.join(_REFERENCE_PRICES)}")
    adv = market.get(ADV)
    requested = np.nan_to_num(orders.values, nan=0.0)
    n_decisions, n_assets, n_horizons = requested.shape
    priced = np.isfinite(mid) & (mid > 0) & np.isfinite(adv) & (adv > 0)
    if model == "impact_stock":
        traits = [market.get(name)[:, :, None] for name in ("market_cap", *CHARACTERISTICS)]
        priced &= np.all([np.isfinite(x[:, :, 0]) for x in traits], axis=0)
        cost = lambda size: impact_stock_cost(size, *traits, coefficients=coefficients)  # noqa: E731
    else:
        cost = lambda size: impact_cost(size, coefficients=coefficients)  # noqa: E731
    priced = priced[:, :, None]
    q = np.where(priced, requested, 0.0)

    # The parent: net flow in the name over this and the previous
    # ``schedule`` decisions. Its cost at size Q is Q * impact(100 Q equity / adv).
    flow = np.cumsum(q, axis=0)
    before = np.zeros_like(flow)
    span = schedule + 1
    before[span:] = flow[:-span]
    parent = flow - before                     # includes this order
    prior = parent - q                         # the parent without it

    def total(weight):
        size = 100.0 * np.abs(weight) * equity / np.where(priced, adv[:, :, None], 1.0)
        return np.abs(weight) * cost(size)
    charge = np.maximum(total(parent) - total(prior), 0.0)     # fraction of equity
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(q != 0, charge / np.abs(q), 0.0)        # per unit of weight

    quantity = np.zeros_like(q)
    price = np.full(q.shape, np.nan)
    reference = np.full(q.shape, np.nan)
    shape = (n_assets, n_horizons)
    outstanding, pooled, remaining = np.zeros(shape), np.zeros(shape), np.zeros(shape, int)
    for t in range(n_decisions):
        m = mid[t][:, None]
        can = np.isfinite(m) & (m > 0)
        # Fill the slice due at this decision from what was outstanding.
        due = (remaining > 0) & can & (outstanding != 0)
        fill = np.where(due, outstanding / np.maximum(remaining, 1), 0.0)
        paid = pooled.copy()                   # the rate the slice was charged at
        outstanding = outstanding - fill
        remaining = np.where(due, remaining - 1, remaining)
        # Then take in this decision's orders, which fill from the next
        # decision on -- or now, when there is no schedule.
        arriving = q[t]
        if schedule == 0:
            fill = fill + arriving
            paid = rate[t]
        else:
            same = (np.sign(outstanding) == np.sign(arriving)) | (outstanding == 0)
            new = outstanding + arriving
            flipped = (np.sign(new) != np.sign(outstanding)) & (outstanding != 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                merged = (np.abs(outstanding) * pooled + np.abs(arriving) * rate[t]) / np.abs(new)
            pooled = np.where(arriving == 0, pooled,
                              np.where(same, np.nan_to_num(merged),
                                       np.where(flipped, rate[t], pooled)))
            outstanding = new
            remaining = np.where(arriving != 0, schedule, remaining)
        side = np.sign(fill)
        filled = fill != 0
        quantity[t] = fill
        price[t] = np.where(filled, m * (1 + side * paid), np.nan)
        reference[t] = np.where(filled, m, np.nan)
        pooled = np.where(outstanding == 0, 0.0, pooled)
    return Fills(quantity, price, reference, orders.decisions, orders.assets, orders.horizons)


def fill_path(weights: PortfolioWeights, market: MarketData | None = None, *,
              model: str = "cross", efq: ArrayLike = 1.0, equity: float | None = None,
              initial: ArrayLike | None = None, limit: ArrayLike | None = None) -> Fills:
    """Trade toward each target from the position actually held, decision by decision.

    Unlike :func:`orders_from_weights`, an order that only partly fills is
    not forgotten: the next decision's order is the next target minus what is
    really held. This is the path to use whenever fills can fall short --
    the book, limit orders, or names that stop quoting.
    """
    if market is not None:
        require_market_aligned(market, weights)
    target = np.nan_to_num(weights.values, nan=0.0)
    n_decisions, n_assets, n_horizons = target.shape
    held = _initial_positions(initial, (n_assets, n_horizons))
    efq = np.broadcast_to(_efq_array(efq), (n_decisions, n_assets))
    columns = {} if market is None else market.columns
    quantity, price, mid = (np.zeros(target.shape), np.full(target.shape, np.nan),
                            np.full(target.shape, np.nan))
    for t in range(n_decisions):
        row = slice(t, t + 1)
        step_limit = (None if limit is None else
                      np.broadcast_to(np.asarray(limit, dtype=np.float64), target.shape)[row])
        order = Orders(target[row] - held[None], weights.decisions[row], weights.assets,
                       weights.horizons, limit=step_limit)
        step = None if market is None else MarketData(
            {k: v[row] for k, v in columns.items()}, market.decisions[row], market.assets,
            market.estimated)
        fill = execute(order, step, model=model, efq=efq[row], equity=equity)
        quantity[t], price[t], mid[t] = fill.quantity[0], fill.price[0], fill.mid_price[0]
        held = held + fill.quantity[0]
    return Fills(quantity, price, mid, weights.decisions, weights.assets, weights.horizons)


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

    ``book`` is charged as ``cross`` here: the touch is its best linear
    approximation, and the allocator's objective must stay linear in the
    trade. The book itself prices the fills (:func:`execute`). ``impact`` is
    not a spread model at all and is refused here; see :func:`impact_cost`.
    """
    if name in ("impact", "impact_stock"):
        raise ValueError(f"the {name} model's cost depends on order size; use impact_cost")
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
