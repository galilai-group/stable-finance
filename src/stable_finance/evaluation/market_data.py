"""The observable market as evaluation reads it, and how a stage says what it needs.

:class:`MarketData` holds one ``(decision, asset)`` array per dataset role,
plus ``half_spread``. A stage that reads the market declares the quantities
it needs with :func:`requires`, and is refused with
:class:`~stable_finance.dataset.schema.MissingRolesError` when they are
absent, rather than guessing. This module depends on nothing in the pipeline,
so every stage can import it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.spec.schema import (
    ROLES,
    MissingRolesError,
)
from stable_finance.evaluation._grid import _one_dimensional

__all__ = ["DERIVATIONS", "MARKET_QUANTITIES", "MarketData", "market_quantity",
           "require_market_aligned", "requires"]

#: Quantities a stage can require of the market: every dataset role, plus the
#: quoted half-spread over mid, which a caller may supply directly.
MARKET_QUANTITIES = (*ROLES, "half_spread")

#: Quantities that can be derived when not supplied directly, and from what.
DERIVATIONS: dict[str, tuple[str, ...]] = {
    "mid": ("bid_price", "ask_price"),
    "half_spread": ("bid_price", "ask_price"),
}


@dataclass(frozen=True, eq=False)
class MarketData:
    """The observable market at each decision, one ``(decision, asset)`` array per role.

    A stage that needs a market quantity says so and calls :meth:`require`,
    so a dataset without that column is REFUSED rather than guessed at. The
    distinction matters because the two absences mean different things: a
    NaN inside a supplied column is a name that was not quoting at that
    decision, which every stage handles; a missing column is a dataset that
    never had the information, and filling it with NaN or zero would turn
    "unknown cost" into "untradable" or "free".

    ``columns`` maps a quantity in :data:`MARKET_QUANTITIES` to an array.
    ``mid`` and ``half_spread`` are derived from ``bid_price`` and
    ``ask_price`` when not given. Build one from the dataset with
    :func:`stable_finance.evaluation.from_dataset.market_from_days` (whatever roles the
    dataset's schema had), from a panel cache with
    :meth:`stable_finance.evaluation.from_dataset.MarketPanel.market`, or directly.
    """

    columns: dict
    decisions: NDArray
    assets: NDArray

    def __post_init__(self) -> None:
        decisions = _one_dimensional("decisions", self.decisions)
        assets = _one_dimensional("assets", self.assets)
        shape = (len(decisions), len(assets))
        columns = {}
        for name, value in dict(self.columns).items():
            if name not in MARKET_QUANTITIES:
                raise ValueError(
                    f"unknown market quantity {name!r}; known: {MARKET_QUANTITIES}")
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            columns[name] = array
        if "half_spread" in columns:
            spread = columns["half_spread"]
            if np.any(np.isfinite(spread) & (spread < 0)):
                raise ValueError("half_spread cannot be negative")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "assets", assets)

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

    def has(self, *quantities: str) -> bool:
        for quantity in quantities:
            if quantity not in MARKET_QUANTITIES:
                raise KeyError(f"unknown market quantity {quantity!r}")
            if quantity in self.columns:
                continue
            sources = DERIVATIONS.get(quantity)
            if sources is None or not all(s in self.columns for s in sources):
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
        bid, ask = self.columns["bid_price"], self.columns["ask_price"]
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
        return MarketData({q: self.get(q) for q in quantities}, self.decisions, self.assets)


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
