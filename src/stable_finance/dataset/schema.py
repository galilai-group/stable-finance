"""Schema for the Polygon-derived 1 Hz market dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FeatureSchema:
    """Column semantics needed to reconstruct and transform market sessions."""

    columns: tuple[str, ...]
    forward_fill: frozenset[str]
    zero_fill: frozenset[str]
    aggregation: tuple[tuple[str, str], ...]
    normalization: tuple[tuple[str, tuple[str, ...], bool], ...]

    def __post_init__(self) -> None:
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be non-empty and unique")
        known = set(self.columns)
        referenced = self.forward_fill | self.zero_fill
        if not referenced <= known:
            raise ValueError(f"fill rules reference unknown columns: {referenced - known}")
        rules = dict(self.aggregation)
        if set(rules) != known:
            raise ValueError("aggregation must define exactly one rule per column")
        allowed = {"last", "max", "min", "sum", "vwap"}
        if set(rules.values()) - allowed:
            raise ValueError(f"unknown aggregation rules: {set(rules.values()) - allowed}")

    def index(self, column: str) -> int:
        try:
            return self.columns.index(column)
        except ValueError as error:
            raise KeyError(column) from error

    def indices(self, columns: set[str] | frozenset[str]) -> list[int]:
        return [index for index, name in enumerate(self.columns) if name in columns]

    @property
    def forward_fill_indices(self) -> list[int]:
        return self.indices(self.forward_fill)

    @property
    def zero_fill_indices(self) -> list[int]:
        return self.indices(self.zero_fill)

    @property
    def aggregation_rules(self) -> dict[str, str]:
        return dict(self.aggregation)

    def normalization_groups(self) -> list[tuple[list[int], bool]]:
        groups = []
        for _, columns, apply_log1p in self.normalization:
            indices = [self.index(column) for column in columns if column in self.columns]
            if indices:
                groups.append((indices, apply_log1p))
        return groups


MARKET_SCHEMA = FeatureSchema(
    columns=(
        "bid_price", "vwap_all", "high", "low", "ask_price",
        "bid_size", "ask_size", "volume", "n",
    ),
    forward_fill=frozenset({
        "bid_price", "vwap_all", "high", "low", "ask_price",
        "bid_size", "ask_size",
    }),
    zero_fill=frozenset({"volume", "n"}),
    aggregation=(
        ("bid_price", "last"), ("vwap_all", "vwap"), ("high", "max"),
        ("low", "min"), ("ask_price", "last"), ("bid_size", "last"),
        ("ask_size", "last"), ("volume", "sum"), ("n", "sum"),
    ),
    normalization=(
        ("price", ("bid_price", "ask_price", "vwap_all", "high", "low"), False),
        ("order_book_size", ("bid_size", "ask_size"), True),
        ("trade_size", ("volume",), True),
        ("trade_count", ("n",), True),
    ),
)


@dataclass(frozen=True)
class MarketSession:
    """One dense ticker session, independent of its storage backend.

    Keeping ``date`` explicit is intentional: a future multi-session framer
    can insert learned open/close boundary events while v1 consumers continue
    to operate on single sessions.
    """

    ticker: str
    date: str
    timestamps: NDArray[np.integer]
    features: NDArray[np.floating]
    schema: FeatureSchema = MARKET_SCHEMA

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps)
        features = np.asarray(self.features)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if features.shape != (len(timestamps), len(self.schema.columns)):
            raise ValueError(
                f"features must have shape {(len(timestamps), len(self.schema.columns))}, "
                f"got {features.shape}"
            )
        if len(timestamps) > 1 and np.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "features", features)

