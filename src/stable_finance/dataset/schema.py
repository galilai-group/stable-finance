"""What the columns of a market time series mean, and what its price is.

SCHEMA. What the columns of a market time series mean.

A :class:`FeatureSchema` names the physical columns of a dense session and
says, per column, how gaps are filled, how bars are aggregated to a coarser
resolution, and which columns are normalized together. It also maps columns
to ROLES -- ``mid``, ``vwap``, ``bid_price``, ``volume`` and so on -- which is
how every target, view builder and execution input finds what it needs without
knowing any dataset's column names.

No role is required. A schema with only a ``mid`` column is a univariate
price series, and it supports exactly the quantities a price series can
support: returns and realized volatility, but not spread changes or anything
priced at the touch. Asking for one of those raises :class:`MissingRolesError`
naming the roles that are absent, instead of failing on an index lookup.

Build a schema from roles and let the defaults supply the rules::

    FeatureSchema.from_roles(mid="price")                       # univariate
    FeatureSchema.from_roles(close="c", high="h", low="l", volume="v")

or spell every rule out with the constructor, as the Market-1T preset does
(:data:`stable_finance.dataset.datasets.market1t.MARKET_SCHEMA`).

PRICE. What "the price" of a bar is, as a choice rather than a column name.

A forward return compares the price measured over a short window at the
decision with the same measurement at the horizon. Which price is a method:

``vwap``
    Volume-weighted trade price: the ``vwap`` column weighted by ``volume``.
    Windows with no traded volume have no price. This is Market-1T's default
    and the measurement every stored Market-1T target uses.
``mid``
    Time-averaged midpoint: the ``mid`` column when the schema has one, else
    ``(bid_price + ask_price) / 2``. The only method a univariate series
    supports, since a single price column is read as the midpoint.
``close``
    Time-averaged ``close`` column, for bar data with no quotes or volume.

Every method reads its window as a weighted mean from prefix sums, so one
code path serves all three and the window length is the anchor spec's
``measurement_window_seconds``: 60 seconds by default, so the default
Market-1T price is the one-minute VWAP.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray



# ── schema ───────────────────────────────────────────────────────────────────

#: Aggregation rules a column may use when bars are coarsened.
AGGREGATIONS = frozenset({"first", "last", "max", "min", "sum", "mean", "vwap"})

#: Every role a column can play, with the rules a column in that role gets
#: from :meth:`FeatureSchema.from_roles`: (aggregation, fill, normalization
#: group, log1p before normalizing). ``fill`` is ``ffill`` for a level that
#: persists between observations and ``zero`` for a flow that did not happen.
ROLE_DEFAULTS: dict[str, tuple[str, str, str, bool]] = {
    "mid": ("last", "ffill", "price", False),
    "open": ("first", "ffill", "price", False),
    "high": ("max", "ffill", "price", False),
    "low": ("min", "ffill", "price", False),
    "close": ("last", "ffill", "price", False),
    "vwap": ("vwap", "ffill", "price", False),
    "bid_price": ("last", "ffill", "price", False),
    "ask_price": ("last", "ffill", "price", False),
    "bid_size": ("last", "ffill", "order_book_size", True),
    "ask_size": ("last", "ffill", "order_book_size", True),
    "volume": ("sum", "zero", "trade_size", True),
    "n_trades": ("sum", "zero", "trade_count", True),
    "market_cap": ("last", "ffill", "market_cap", True),
}
ROLES = tuple(ROLE_DEFAULTS)

#: Column names read as a role when a schema states no roles of its own:
#: every role's own name, plus Market-1T's two names that differ from it.
ROLE_ALIASES: dict[str, str] = {**{role: role for role in ROLES},
                                "vwap_all": "vwap", "n": "n_trades"}
_GROUP_ORDER = ("price", "order_book_size", "trade_size", "trade_count", "market_cap")


class MissingRolesError(ValueError):
    """A quantity was requested that the schema's columns cannot support."""


@dataclass(frozen=True)
class FeatureSchema:
    """Column semantics needed to reconstruct and transform market sessions.

    ``roles`` maps a role in :data:`ROLES` to the column that plays it. Left
    empty, roles are inferred from column names (:data:`ROLE_ALIASES`), so a
    schema written before roles existed keeps working; pass roles explicitly
    whenever a column's name is not its role.
    """

    columns: tuple[str, ...]
    forward_fill: frozenset[str]
    zero_fill: frozenset[str]
    aggregation: tuple[tuple[str, str], ...]
    normalization: tuple[tuple[str, tuple[str, ...], bool], ...]
    roles: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "forward_fill", frozenset(self.forward_fill))
        object.__setattr__(self, "zero_fill", frozenset(self.zero_fill))
        object.__setattr__(self, "aggregation", tuple(tuple(r) for r in self.aggregation))
        object.__setattr__(self, "normalization", tuple(
            (name, tuple(columns), bool(log1p)) for name, columns, log1p in self.normalization))
        if isinstance(self.roles, Mapping):
            object.__setattr__(self, "roles", tuple(self.roles.items()))
        object.__setattr__(self, "roles", tuple(tuple(r) for r in self.roles))
        if not self.roles:
            inferred: dict[str, str] = {}
            for column in self.columns:
                role = ROLE_ALIASES.get(column)
                if role is not None and role not in inferred:
                    inferred[role] = column
            object.__setattr__(self, "roles", tuple(inferred.items()))
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be non-empty and unique")
        known = set(self.columns)
        referenced = self.forward_fill | self.zero_fill
        if not referenced <= known:
            raise ValueError(f"fill rules reference unknown columns: {referenced - known}")
        rules = dict(self.aggregation)
        if set(rules) != known:
            raise ValueError("aggregation must define exactly one rule per column")
        if set(rules.values()) - AGGREGATIONS:
            raise ValueError(f"unknown aggregation rules: {set(rules.values()) - AGGREGATIONS}")
        names = [role for role, _ in self.roles]
        if len(set(names)) != len(names):
            raise ValueError("each role can be assigned to one column only")
        unknown_roles = set(names) - set(ROLES)
        if unknown_roles:
            raise ValueError(f"unknown roles {sorted(unknown_roles)}; known: {ROLES}")
        missing = {column for _, column in self.roles} - known
        if missing:
            raise ValueError(f"roles reference unknown columns: {sorted(missing)}")

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_roles(
        cls,
        columns: tuple[str, ...] | list[str] | None = None,
        *,
        extra: Mapping[str, str] | None = None,
        **roles: str,
    ) -> "FeatureSchema":
        """A schema whose rules follow from the role each column plays.

        ``roles`` maps role to column name, for instance ``mid="price"``.
        ``extra`` adds columns that play no role, mapping each to its
        aggregation rule; a ``sum`` column is zero-filled, every other one is
        forward-filled, and each is normalized on its own. ``columns`` fixes
        the physical column order and defaults to the roles in the order
        given, then the extras.
        """
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)}; known: {ROLES}")
        if not roles and not extra:
            raise ValueError("a schema needs at least one column")
        extra = dict(extra or {})
        by_column: dict[str, tuple[str, str, str, bool]] = {}
        for role, column in roles.items():
            if column in by_column:
                raise ValueError(f"column {column!r} is assigned two roles")
            by_column[column] = ROLE_DEFAULTS[role]
        for column, rule in extra.items():
            if column in by_column:
                raise ValueError(f"column {column!r} is both a role and an extra")
            if rule not in AGGREGATIONS:
                raise ValueError(f"unknown aggregation rule {rule!r} for {column!r}")
            by_column[column] = (rule, "zero" if rule == "sum" else "ffill", column, False)
        order = tuple(columns) if columns is not None else tuple(by_column)
        if set(order) != set(by_column) or len(order) != len(by_column):
            raise ValueError("columns must list every role and extra column exactly once")
        groups: dict[str, list[str]] = {}
        log1p: dict[str, bool] = {}
        for column in order:
            _, _, group, takes_log = by_column[column]
            groups.setdefault(group, []).append(column)
            log1p[group] = takes_log
        ordered = [g for g in _GROUP_ORDER if g in groups] + [
            g for g in groups if g not in _GROUP_ORDER]
        return cls(
            columns=order,
            forward_fill=frozenset(c for c in order if by_column[c][1] == "ffill"),
            zero_fill=frozenset(c for c in order if by_column[c][1] == "zero"),
            aggregation=tuple((c, by_column[c][0]) for c in order),
            normalization=tuple((g, tuple(groups[g]), log1p[g]) for g in ordered),
            roles=tuple(roles.items()),
        )

    @classmethod
    def univariate(cls, column: str = "price") -> "FeatureSchema":
        """One price column, read as the midpoint."""
        return cls.from_roles(mid=column)

    def to_dict(self) -> dict:
        """A JSON-serializable description, written beside every store."""
        return {
            "columns": list(self.columns),
            "forward_fill": sorted(self.forward_fill),
            "zero_fill": sorted(self.zero_fill),
            "aggregation": dict(self.aggregation),
            "normalization": [[name, list(columns), log1p]
                              for name, columns, log1p in self.normalization],
            "roles": dict(self.roles),
        }

    @classmethod
    def from_dict(cls, value: Mapping | list | tuple) -> "FeatureSchema":
        """Inverse of :meth:`to_dict`.

        A bare list of column names is what stores written before schemas
        were self-describing recorded. The only such stores are Market-1T, so
        that list must match the preset and the preset is returned.
        """
        if isinstance(value, (list, tuple)):
            from stable_finance.dataset.datasets.market1t.spec import MARKET_SCHEMA

            if tuple(value) != MARKET_SCHEMA.columns:
                raise ValueError(
                    f"a store that records only column names must be Market-1T; got {value}")
            return MARKET_SCHEMA
        aggregation = value["aggregation"]
        return cls(
            columns=tuple(value["columns"]),
            forward_fill=frozenset(value.get("forward_fill", ())),
            zero_fill=frozenset(value.get("zero_fill", ())),
            aggregation=tuple((c, aggregation[c]) for c in value["columns"]),
            normalization=tuple((n, tuple(c), bool(l)) for n, c, l in value.get("normalization", ())),
            roles=tuple(dict(value.get("roles", {})).items()),
        )

    # -- columns ---------------------------------------------------------------

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

    # -- roles -----------------------------------------------------------------

    @property
    def role_map(self) -> dict[str, str]:
        return dict(self.roles)

    def column_for(self, role: str) -> str | None:
        """The column playing ``role``, or None when the schema has none."""
        if role not in ROLE_DEFAULTS:
            raise KeyError(f"unknown role {role!r}; known: {ROLES}")
        return self.role_map.get(role)

    def has(self, *roles: str) -> bool:
        return all(self.column_for(role) is not None for role in roles)

    def role_index(self, role: str) -> int:
        """Column index of ``role``; raises :class:`MissingRolesError` if absent."""
        column = self.column_for(role)
        if column is None:
            raise MissingRolesError(
                f"this schema has no {role!r} column (roles: {self.describe_roles()})")
        return self.index(column)

    def optional_index(self, role: str) -> int | None:
        column = self.column_for(role)
        return None if column is None else self.index(column)

    def require(self, *roles: str, purpose: str) -> None:
        missing = [role for role in roles if not self.has(role)]
        if missing:
            raise MissingRolesError(
                f"{purpose} needs {', '.join(missing)}; this schema maps "
                f"{self.describe_roles()}")

    def describe_roles(self) -> str:
        return ", ".join(f"{r}={c}" for r, c in self.roles) or "no roles"


@dataclass(frozen=True)
class MarketSession:
    """One dense ticker session, independent of its storage backend.

    ``schema`` defaults to the Market-1T preset. ``bar_seconds`` is the
    spacing of ``timestamps``. Keeping ``date`` explicit is intentional: a
    future multi-session framer can insert learned open/close boundary events
    while v1 consumers continue to operate on single sessions.
    """

    ticker: str
    date: str
    timestamps: NDArray[np.integer]
    features: NDArray[np.floating]
    schema: FeatureSchema | None = None
    bar_seconds: int = 1

    def __post_init__(self) -> None:
        if self.schema is None:
            object.__setattr__(self, "schema", default_schema())
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
        if self.bar_seconds < 1:
            raise ValueError("bar_seconds must be positive")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "features", features)


def default_schema(schema: FeatureSchema | None = None) -> FeatureSchema:
    """``schema``, or Market-1T's when none is given.

    The one place a generic module falls back to Market-1T: every function
    that takes a schema accepts None and resolves it here, so the default is
    visible as a default rather than baked into each signature.
    """
    if schema is not None:
        return schema
    from stable_finance.dataset.datasets.market1t.spec import MARKET_SCHEMA

    return MARKET_SCHEMA


def __getattr__(name: str):
    # MARKET_SCHEMA lives with the rest of Market-1T; this keeps the historical
    # ``from stable_finance.dataset.schema import MARKET_SCHEMA`` working
    # without a circular import at module load.
    if name == "MARKET_SCHEMA":
        return default_schema()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── price ────────────────────────────────────────────────────────────────────

PRICE_METHODS = ("vwap", "mid", "close")


def has_mid(schema: FeatureSchema) -> bool:
    """Can a midpoint be read, directly or from both sides of the book?"""
    return schema.has("mid") or schema.has("bid_price", "ask_price")


def default_price(schema: FeatureSchema) -> str:
    """The richest price the schema supports: vwap, then mid, then close."""
    if schema.has("vwap", "volume"):
        return "vwap"
    if has_mid(schema):
        return "mid"
    if schema.has("close"):
        return "close"
    raise MissingRolesError(
        "no price can be formed: a schema needs vwap and volume, mid, "
        f"bid_price and ask_price, or close (roles: {schema.describe_roles()})")


def resolve_price(schema: FeatureSchema, price: str | None) -> str:
    """Validate ``price`` against ``schema``; None picks :func:`default_price`."""
    if price is None:
        return default_price(schema)
    if price not in PRICE_METHODS:
        raise ValueError(f"unknown price method {price!r}; known: {PRICE_METHODS}")
    if price == "vwap":
        schema.require("vwap", "volume", purpose="the vwap price")
    elif price == "mid":
        if not has_mid(schema):
            raise MissingRolesError(
                "the mid price needs a mid column or both bid_price and ask_price "
                f"(roles: {schema.describe_roles()})")
    else:
        schema.require("close", purpose="the close price")
    return price


def mid_series(features: np.ndarray, schema: FeatureSchema) -> np.ndarray:
    """The midpoint at every row, float64."""
    index = schema.optional_index("mid")
    if index is not None:
        return np.asarray(features[..., index], dtype=np.float64)
    if not schema.has("bid_price", "ask_price"):
        raise MissingRolesError(
            "a midpoint needs a mid column or both bid_price and ask_price "
            f"(roles: {schema.describe_roles()})")
    return (np.asarray(features[..., schema.role_index("bid_price")], dtype=np.float64)
            + np.asarray(features[..., schema.role_index("ask_price")], dtype=np.float64)) / 2.0


def price_values(features: np.ndarray, schema: FeatureSchema, method: str) -> np.ndarray:
    """The per-row price a method weights; NaN where it is unobserved."""
    if method == "vwap":
        return np.asarray(features[..., schema.role_index("vwap")], dtype=np.float64)
    if method == "mid":
        return mid_series(features, schema)
    if method == "close":
        return np.asarray(features[..., schema.role_index("close")], dtype=np.float64)
    raise ValueError(f"unknown price method {method!r}")


def price_cumsums(features: np.ndarray, schema: FeatureSchema,
                  method: str) -> tuple[np.ndarray, np.ndarray]:
    """Prefix sums of weight * price and of weight, each with a leading zero.

    ``vwap`` weights by traded volume, the others weight every observed row
    equally. A window's price is the ratio of the two differences, and a
    window whose weight is zero has no price.
    """
    values = price_values(features, schema, method)
    if method == "vwap":
        weights = np.nan_to_num(
            np.asarray(features[:, schema.role_index("volume")], dtype=np.float64), nan=0.0)
    else:
        weights = np.isfinite(values).astype(np.float64)
    numerator = np.nan_to_num(values * weights, nan=0.0)
    return (
        np.concatenate([[0.0], np.cumsum(numerator)]),
        np.concatenate([[0.0], np.cumsum(weights)]),
    )


def windowed_price(cs_num: np.ndarray, cs_weight: np.ndarray, length: int,
                   idx, window: int):
    """The price over ``[idx, idx + window)`` for one or many ``idx``."""
    indices = np.asarray(idx, dtype=np.int64)
    lo, hi = indices, indices + int(window)
    valid = (lo >= 0) & (hi <= length)
    lo_safe = np.clip(lo, 0, length)
    hi_safe = np.clip(hi, 0, length)
    denominator = cs_weight[hi_safe] - cs_weight[lo_safe]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            valid & (denominator > 0),
            (cs_num[hi_safe] - cs_num[lo_safe])
            / np.where(denominator > 0, denominator, 1.0),
            np.nan,
        )
