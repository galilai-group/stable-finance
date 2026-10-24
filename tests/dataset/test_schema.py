"""Schemas say what columns mean by role; Market-1T is one preset."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance.dataset import (
    MARKET_SCHEMA,
    FeatureSchema,
    MarketSession,
    default_price,
    supported_targets,
)



def test_supported_schema_owns_column_semantics():
    assert MARKET_SCHEMA.columns[0] == "bid_price"
    assert MARKET_SCHEMA.index("ask_price") == 4
    assert MARKET_SCHEMA.aggregation_rules["vwap_all"] == "vwap"
    assert MARKET_SCHEMA.forward_fill_indices == list(range(7))
    assert MARKET_SCHEMA.zero_fill_indices == [7, 8]


def test_market_session_retains_explicit_boundaries_for_future_framing():
    session = MarketSession(
        "ABC", "2023-01-03", np.array([1, 2]), np.ones((2, 9))
    )
    assert session.date == "2023-01-03"
    assert session.bar_seconds == 1
    assert MarketSession(
        "ABC", "2023-01-03", np.array([1, 61]), np.ones((2, 9)),
        bar_seconds=60,
    ).bar_seconds == 60
    with pytest.raises(ValueError, match="strictly increasing"):
        MarketSession("ABC", "2023-01-03", [2, 1], np.ones((2, 9)))


DATE = "2024-03-05"          # a Tuesday


def test_univariate_schema_is_one_mid_column():
    schema = FeatureSchema.univariate("px")
    assert schema.columns == ("px",)
    assert schema.role_map == {"mid": "px"}
    assert schema.forward_fill == {"px"} and not schema.zero_fill
    assert schema.aggregation_rules == {"px": "last"}
    assert schema.normalization_groups() == [([0], False)]
    assert default_price(schema) == "mid"


def test_from_roles_derives_rules_from_each_role():
    schema = FeatureSchema.from_roles(open="o", high="h", low="l", close="c", volume="v",
                                      extra={"oi": "last"})
    assert schema.aggregation_rules == {"o": "first", "h": "max", "l": "min", "c": "last",
                                        "v": "sum", "oi": "last"}
    assert schema.zero_fill == {"v"}
    assert [name for name, *_ in schema.normalization] == ["price", "trade_size", "oi"]
    assert default_price(schema) == "close"


def test_schema_round_trips_through_its_description():
    for schema in (MARKET_SCHEMA, FeatureSchema.from_roles(mid="m", bid_price="b", ask_price="a")):
        assert FeatureSchema.from_dict(schema.to_dict()) == schema
    # A store that recorded only column names is Market-1T.
    assert FeatureSchema.from_dict(list(MARKET_SCHEMA.columns)) is MARKET_SCHEMA
    with pytest.raises(ValueError):
        FeatureSchema.from_dict(["price"])


def test_a_schema_without_roles_reads_them_off_its_column_names():
    schema = FeatureSchema(
        columns=("n_trades", "vwap", "volume", "extra"), forward_fill=frozenset({"vwap"}),
        zero_fill=frozenset({"n_trades", "volume"}),
        aggregation=(("n_trades", "sum"), ("vwap", "vwap"), ("volume", "sum"),
                     ("extra", "last")),
        normalization=(),
    )
    assert schema.role_map == {"n_trades": "n_trades", "vwap": "vwap", "volume": "volume"}


def test_market1t_roles_cover_every_quantity():
    assert supported_targets(MARKET_SCHEMA) == ("return", "spread_change", "volatility_change")
    assert default_price(MARKET_SCHEMA) == "vwap"


def test_session_default_schema_is_market1t():
    s = MarketSession("A", DATE, np.arange(3), np.zeros((3, 9)))
    assert s.schema is MARKET_SCHEMA

