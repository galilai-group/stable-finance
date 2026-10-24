"""The observable market, by level, and how a stage says what it needs.

Market data comes in a hierarchy, from least to most informative:

``mid``     a price series and nothing else. Returns can be measured and a
            book traded frictionlessly at the price the returns assume.
``bars``    OHLCV bars. A spread can be ESTIMATED from high, low and close
            (:func:`bar_half_spread`), which is what lets costs be charged.
``quotes``  the best bid and ask, observed. The spread is quoted, not guessed.
``book``    depth: several price levels with their sizes on each side, which
            is what an order larger than the touch walks.

A richer level derives the poorer ones where it can: a book's first level is
the best bid and ask, and those give the midpoint and the half-spread. So a
stage written for quotes runs on book data unchanged.

:class:`MarketData` holds one ``(decision, asset)`` array per quantity -- a
book quantity adds a trailing level axis -- as observed AT the decision, with
no lookahead. The two ``window_*`` quantities are the exception, and are named
for it: they summarize the execution window after the decision, and only the
execution model may read them. A stage declares what it needs with
:func:`requires` and is refused with
:class:`~stable_finance.dataset.schema.MissingRolesError` when it is absent,
rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.schema import ROLES, MissingRolesError

__all__ = [
    "ADV",
    "BOOK_QUANTITIES",
    "CHARACTERISTICS",
    "DERIVATIONS",
    "MARKET_LEVELS",
    "MARKET_QUANTITIES",
    "WINDOW_QUANTITIES",
    "MarketData",
    "bar_half_spread",
    "market_quantity",
    "require_market_aligned",
    "requires",
]

#: Depth, one ``(decision, asset, level)`` array each; level 0 is the touch.
#: Sizes are in shares. A missing level is NaN.
BOOK_QUANTITIES = ("book_bid_price", "book_bid_size", "book_ask_price", "book_ask_size")

#: The best price a resting buy, and a resting sell, could have filled at
#: during the execution window after the decision: the lowest ask (or low, or
#: price) and the highest bid (or high, or price). They are what a limit order
#: is filled against, and they are AFTER the decision.
WINDOW_QUANTITIES = ("window_best_buy", "window_best_sell")

#: Average daily dollar volume of the name, as of the decision: the scale an
#: order's size is measured against by the price-impact model. In the paper
#: it is calibrated to (Frazzini, Israel and Moskowitz, 2018) it is the past
#: year's average; :func:`~stable_finance.evaluation.inputs.evaluation_inputs`
#: computes it over a lookback you choose.
ADV = "adv"

#: Stock and market characteristics the stock-specific impact model reads,
#: each as of the decision: idiosyncratic volatility (annualized %, the
#: residual of the name's daily returns on the market's over a trailing
#: year), market volatility (annualized %, of the market's daily returns),
#: and the time trend (months since June 1926, which is 1). Market cap is a
#: dataset role, in the market's currency.
CHARACTERISTICS = ("idio_vol", "market_vol", "time_trend")

#: Quantities a stage can require of the market: every dataset role, the
#: half-spread over mid (quoted, or estimated from bars), average daily
#: dollar volume, the characteristics, depth, and the execution-window
#: extremes.
MARKET_QUANTITIES = (*ROLES, "half_spread", ADV, *CHARACTERISTICS, *BOOK_QUANTITIES,
                     *WINDOW_QUANTITIES)

#: What each level of market data needs, from least to most informative.
MARKET_LEVELS: dict[str, tuple[str, ...]] = {
    "mid": (),
    "bars": ("high", "low", "close"),
    "quotes": ("bid_price", "ask_price"),
    "book": BOOK_QUANTITIES,
}

#: Quantities derived when not supplied directly, and from what. Derivations
#: chain: a book gives the touch, and the touch gives mid and half-spread.
DERIVATIONS: dict[str, tuple[str, ...]] = {
    "mid": ("bid_price", "ask_price"),
    "half_spread": ("bid_price", "ask_price"),
    "bid_price": ("book_bid_price",),
    "ask_price": ("book_ask_price",),
    "bid_size": ("book_bid_size",),
    "ask_size": ("book_ask_size",),
}


def _one_dimensional(name: str, value: ArrayLike) -> NDArray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


@dataclass(frozen=True, eq=False)
class MarketData:
    """The observable market at each decision, one array per quantity.

    A stage that needs a market quantity says so and calls :meth:`require`,
    so a dataset without that column is REFUSED rather than guessed at. The
    distinction matters because the two absences mean different things: a
    NaN inside a supplied column is a name that was not quoting at that
    decision, which every stage handles; a missing column is a dataset that
    never had the information, and filling it with NaN or zero would turn
    "unknown cost" into "untradable" or "free".

    ``columns`` maps a quantity in :data:`MARKET_QUANTITIES` to a
    ``(decision, asset)`` array, or ``(decision, asset, level)`` for a book
    quantity. ``estimated`` names the columns that were estimated rather than
    observed, such as a half-spread from bars. Build one with
    :func:`stable_finance.evaluation.inputs.evaluation_inputs`, from a
    dataset's own reader, or directly.
    """

    columns: dict
    decisions: NDArray
    assets: NDArray
    estimated: frozenset = field(default=frozenset())

    def __post_init__(self) -> None:
        decisions = _one_dimensional("decisions", self.decisions)
        assets = _one_dimensional("assets", self.assets)
        shape = (len(decisions), len(assets))
        columns = {}
        levels = None
        for name, value in dict(self.columns).items():
            if name not in MARKET_QUANTITIES:
                raise ValueError(
                    f"unknown market quantity {name!r}; known: {MARKET_QUANTITIES}")
            array = np.asarray(value, dtype=np.float64)
            if name in BOOK_QUANTITIES:
                if array.ndim != 3 or array.shape[:2] != shape or array.shape[2] < 1:
                    raise ValueError(f"{name} must have shape {shape} + (levels,), "
                                     f"got {array.shape}")
                if levels is not None and array.shape[2] != levels:
                    raise ValueError("every book quantity needs the same number of levels")
                levels = array.shape[2]
            elif array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            columns[name] = array
        for name in ("half_spread", ADV, "book_bid_size", "book_ask_size"):
            if name in columns and np.any(np.isfinite(columns[name]) & (columns[name] < 0)):
                raise ValueError(f"{name} cannot be negative")
        estimated = frozenset(self.estimated)
        if not estimated <= set(columns):
            raise ValueError("estimated names a quantity the market does not hold")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "estimated", estimated)

    @classmethod
    def empty(cls, decisions: ArrayLike, assets: ArrayLike) -> "MarketData":
        """A market about which nothing is observed: only prices from returns."""
        return cls({}, decisions, assets)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.decisions), len(self.assets))

    @property
    def available(self) -> tuple[str, ...]:
        """Every quantity :meth:`get` can return, supplied or derived."""
        return tuple(q for q in MARKET_QUANTITIES if self.has(q))

    @property
    def levels(self) -> tuple[str, ...]:
        """Every level of :data:`MARKET_LEVELS` this market supports."""
        return tuple(level for level, needed in MARKET_LEVELS.items() if self.has(*needed))

    @property
    def level(self) -> str:
        """The most informative level this market supports."""
        return self.levels[-1]

    def has(self, *quantities: str) -> bool:
        for quantity in quantities:
            if quantity not in MARKET_QUANTITIES:
                raise KeyError(f"unknown market quantity {quantity!r}")
            if quantity in self.columns:
                continue
            sources = DERIVATIONS.get(quantity)
            if sources is None or not self.has(*sources):
                return False
        return True

    def require(self, *quantities: str, purpose: str) -> None:
        """Raise :class:`MissingRolesError` unless every quantity is available."""
        missing = [q for q in quantities if not self.has(q)]
        if missing:
            def spell(q):
                sources = DERIVATIONS.get(q)
                return f"{q} (or {' and '.join(sources)})" if sources else q
            raise MissingRolesError(
                f"{purpose} needs {', '.join(spell(q) for q in missing)}; this "
                f"market data provides {', '.join(self.available) or 'nothing'}")

    def get(self, quantity: str) -> NDArray[np.float64]:
        """One quantity on the grid; raises if it is unavailable."""
        self.require(quantity, purpose=f"reading {quantity!r}")
        if quantity in self.columns:
            return self.columns[quantity]
        if quantity in ("bid_price", "ask_price", "bid_size", "ask_size"):
            return self.get(DERIVATIONS[quantity][0])[:, :, 0]
        bid, ask = self.get("bid_price"), self.get("ask_price")
        # A crossed or one-sided book has no midpoint and no spread. A negative
        # half-spread would book a REBATE for crossing it.
        broken = ~np.isfinite(bid) | ~np.isfinite(ask) | (bid <= 0) | (ask < bid)
        mid = np.where(broken, np.nan, (bid + ask) / 2)
        if quantity == "mid":
            return mid
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(broken, np.nan, (ask - bid) / 2 / mid)

    __getitem__ = get

    def select(self, *quantities: str) -> "MarketData":
        """A copy holding only ``quantities`` -- to test what a stage needs."""
        return MarketData({q: self.get(q) for q in quantities}, self.decisions,
                          self.assets, self.estimated & set(quantities))


def bar_half_spread(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> NDArray[np.float64]:
    """Per-pair half-spread terms from consecutive bars (Abdi and Ranaldo, 2017).

    The close of a bar sits at the bid or the ask, and the midpoint of its
    high and low, and of the next bar's, brackets the efficient price. With
    log prices ``c`` and ``eta = (log high + log low) / 2``, the squared spread
    is ``4 (c_t - eta_t)(c_t - eta_{t+1})`` in expectation. This returns that
    term for each consecutive pair, ``len - 1`` values; average them over a
    lookback, floor the mean at zero, take the square root and halve it for
    the half-spread. A single term is noisy and can be negative; only the
    average is an estimate.
    """
    h, lo, c = (np.log(np.asarray(a, dtype=np.float64)) for a in (high, low, close))
    eta = (h + lo) / 2
    return 4.0 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])


def requires(*quantities: str):
    """Declare what a stage needs of the market, as ``stage.requires``.

    The declaration is what lets a caller ask a stage, before running it,
    whether their data can support it -- ``market.has(*stage.requires)`` --
    and it is the same list the stage passes to :meth:`MarketData.require`
    when it is handed market data. A custom stage built on the library (an
    allocator sized off ``high`` and ``low``, say) declares its own the same
    way and gets the same refusal.
    """
    unknown = set(quantities) - set(MARKET_QUANTITIES)
    if unknown:
        raise ValueError(f"unknown market quantities {sorted(unknown)}")

    def mark(stage):
        stage.requires = tuple(quantities)
        return stage
    return mark


def market_quantity(value, quantity: str, *, stage, like=None) -> NDArray[np.float64]:
    """An array argument that may instead be :class:`MarketData`.

    Stages accept either the bare array they always took or market data, from
    which they read ``quantity`` after checking ``stage.requires``.
    """
    if isinstance(value, MarketData):
        value.require(*stage.requires, purpose=stage.__name__)
        if like is not None:
            require_market_aligned(value, like)
        return value.get(quantity)
    return value


def require_market_aligned(market: MarketData, other) -> None:
    """Raise unless ``market`` describes the decisions and assets of ``other``."""
    for axis in ("decisions", "assets"):
        if not np.array_equal(getattr(market, axis), getattr(other, axis)):
            raise ValueError(f"market data {axis} are not aligned")


def _validated_quotes(
    best_bid: ArrayLike,
    best_ask: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equally shaped quote arrays after validating finite markets."""
    bid = np.asarray(best_bid, dtype=np.float64)
    ask = np.asarray(best_ask, dtype=np.float64)
    if bid.ndim != 2 or bid.shape != ask.shape:
        raise ValueError("best_bid and best_ask must be equally shaped 2-D arrays")
    crossed = np.isfinite(bid) & np.isfinite(ask) & (ask < bid)
    if crossed.any():
        raise ValueError("best_ask cannot be below best_bid")
    return bid, ask
