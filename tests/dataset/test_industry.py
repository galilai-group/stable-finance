"""Industry is a schema role like market cap, and Market-1T supplies it from its map."""
from __future__ import annotations

import numpy as np
import pytest
from helpers import market1t_store

from stable_finance.dataset import (
    MARKET_QUANTITIES,
    ROLES,
    FeatureSchema,
    MarketData,
    aggregate,
    market_from_days,
)
from stable_finance.dataset.schema import ROLE_DEFAULTS


def test_industry_is_a_role_kept_as_a_label():
    assert "industry" in ROLES and "industry" in MARKET_QUANTITIES
    assert ROLE_DEFAULTS["industry"] == ("last", "ffill", "classification", False)
    schema = FeatureSchema.from_roles(mid="px", volume="v", industry="ff49", market_cap="cap")
    assert schema.column_for("industry") == "ff49"
    assert schema.aggregation_rules["ff49"] == "last"
    assert "ff49" in schema.forward_fill and "ff49" not in schema.zero_fill
    assert schema.normalization[-1] == ("classification", ("ff49",), False)
    assert [name for name, *_ in schema.normalization] == [
        "price", "trade_size", "market_cap", "classification"]
    # A column named after the role plays it, and the schema round-trips.
    inferred = FeatureSchema(columns=("mid", "industry"), forward_fill=frozenset({"mid", "industry"}),
                             zero_fill=frozenset(), aggregation=(("mid", "last"), ("industry", "last")),
                             normalization=())
    assert inferred.role_map == {"mid": "mid", "industry": "industry"}
    assert FeatureSchema.from_dict(schema.to_dict()) == schema


def test_coarser_bars_keep_the_code_rather_than_averaging_it():
    schema = FeatureSchema.from_roles(vwap="p", volume="v", industry="code")
    bars = np.array([[10.0, 1.0, 12.0], [11.0, 3.0, 12.0], [11.0, 2.0, 30.0], [12.0, 2.0, 30.0]])
    coarse = aggregate(bars, 2, schema=schema)
    np.testing.assert_array_equal(coarse[:, schema.role_index("industry")], [12.0, 30.0])


def test_market_data_holds_an_industry_column():
    codes = np.array([[12.0, 7.0, np.nan], [12.0, 30.0, 7.0]])
    market = MarketData({"industry": codes}, np.arange(2), np.array(["A", "B", "C"]))
    assert market.has("industry") and market.available == ("industry",)
    np.testing.assert_array_equal(market.get("industry"), codes)
    assert not MarketData.empty(np.arange(2), np.array(["A", "B", "C"])).has("industry")


def test_market_from_days_joins_the_industry_map_by_month(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    paths = market1t_store(tmp_path / "store", ["2023-01-31", "2023-02-01"], n_tickers=4)
    # The map as ingest.build_industry_map writes it; T01's January code is
    # NULL, T03 is unlabelled in January, ZZZ is not in the store and March
    # is outside the span.
    industry_map = tmp_path / "industry_map.parquet"
    pd.DataFrame({
        "month": ["2023-01", "2023-01", "2023-01", "2023-02", "2023-02", "2023-02", "2023-03"],
        "ticker": ["T00", "T01", "T02", "T00", "T03", "ZZZ", "T01"],
        "ff49": pd.array([12, None, 7, 30, 44, 1, 9], dtype="Int16"),
    }).to_parquet(industry_map, index=False)

    market = market_from_days(paths, industry_map=industry_map)
    assert market.has("industry") and list(market.assets) == ["T00", "T01", "T02", "T03"]
    codes = market.get("industry")
    january = np.char.startswith(market.decisions.astype(str), "2023-01-31")
    assert january.any() and (~january).any()
    nan = np.nan
    np.testing.assert_array_equal(codes[january], np.tile([12.0, nan, 7.0, nan], (january.sum(), 1)))
    np.testing.assert_array_equal(codes[~january],
                                  np.tile([30.0, nan, nan, 44.0], ((~january).sum(), 1)))
    # Without the map, nothing changes and no industry is claimed.
    plain = market_from_days(paths)
    assert not plain.has("industry")
    np.testing.assert_array_equal(plain.get("bid_price"), market.get("bid_price"))


def test_a_table_carries_its_industry_column_through_as_of_each_decision():
    from stable_finance.evaluation.inputs import evaluation_inputs

    frame = {"timestamp": np.array([0, 60, 120, 0, 60, 120]),
             "ticker": np.array(["A", "A", "A", "B", "B", "B"]),
             "px": np.array([10.0, 10.1, 10.2, 20.0, 20.1, 20.2]),
             "code": np.array([12.0, 12.0, 30.0, 7.0, 7.0, np.nan])}
    schema = FeatureSchema.from_roles(mid="px", industry="code")
    _, market = evaluation_inputs(frame, schema, [60, 120], [60])
    np.testing.assert_array_equal(market.get("industry"), [[12.0, 7.0], [30.0, np.nan]])
