"""Downstream stages declare the market columns they need and refuse without them."""
from __future__ import annotations

import numpy as np
import pytest

import stable_finance as sf
from stable_finance import (
    DEFAULT_BACKTEST,
    ForwardReturns,
    MarketData,
    MissingRolesError,
    Orders,
    PortfolioWeights,
    allocate,
    cross_spread_sharpe,
    edge_to_cost,
    enumerate_configs,
    evaluate_weights,
    market_from_days,
    run_backtest,
    simulate_fills,
    sweep_configs,
)
from stable_finance.dataset import Objective, Timeline

D, A = 40, 12
DECISIONS, ASSETS = np.arange(D), np.array([f"S{i}" for i in range(A)])


def _quotes(seed=0):
    rng = np.random.default_rng(seed)
    mid = 50 + rng.normal(0, 1, (D, A)).cumsum(axis=0) * 0.1
    half = 0.01 + 0.02 * rng.random((D, A))
    return mid - half, mid + half


def _returns(seed=1, scale=1e-3):
    rng = np.random.default_rng(seed)
    return ForwardReturns(rng.normal(0, scale, (D, A, 1)), DECISIONS, ASSETS, [300])


def _forecast(realized, seed=2):
    rng = np.random.default_rng(seed)
    noisy = realized.values + rng.normal(0, 2e-3, realized.values.shape)
    return ForwardReturns(noisy, DECISIONS, ASSETS, [300])


MID = DEFAULT_BACKTEST.replace(cost_model="mid")


# ── the contract ─────────────────────────────────────────────────────────────


def test_mid_and_half_spread_are_derived_from_the_book():
    bid, ask = _quotes()
    market = MarketData({"bid_price": bid, "ask_price": ask}, DECISIONS, ASSETS)
    assert market.available == ("mid", "bid_price", "ask_price", "half_spread")
    np.testing.assert_allclose(market["mid"], (bid + ask) / 2)
    np.testing.assert_allclose(market["half_spread"], (ask - bid) / (bid + ask))
    crossed = MarketData({"bid_price": ask, "ask_price": bid}, DECISIONS, ASSETS)
    assert np.isnan(crossed["half_spread"]).all()       # never a rebate


def test_a_missing_column_is_refused_and_named():
    market = MarketData({"mid": np.ones((D, A))}, DECISIONS, ASSETS)
    with pytest.raises(MissingRolesError,
                       match=r"half_spread \(or bid_price and ask_price\).*provides mid"):
        market.require("half_spread", purpose="test")
    with pytest.raises(ValueError, match="unknown market quantity"):
        MarketData({"spread": np.ones((D, A))}, DECISIONS, ASSETS)
    with pytest.raises(ValueError, match="shape"):
        MarketData({"mid": np.ones((D, A + 1))}, DECISIONS, ASSETS)


def test_every_stage_states_what_it_needs():
    assert simulate_fills.requires == ("bid_price", "ask_price")
    assert cross_spread_sharpe.requires == ("bid_price", "ask_price")
    assert edge_to_cost.requires == ("half_spread",)
    assert evaluate_weights.requires == ("half_spread",)
    assert allocate.requires == ("half_spread",)
    assert sf.evaluate_forward_returns.requires == () and sf.fit.requires == ()
    assert DEFAULT_BACKTEST.requires == {"cost_model": ("half_spread",)}
    assert MID.requires == {}
    assert MID.replace(universe="tight_25pct").requires == {"universe": ("half_spread",)}


def test_a_custom_stage_declares_and_is_refused_the_same_way():
    @sf.requires("open", "high", "low", "close")
    def range_weighted(forecast, market):
        market.require(*range_weighted.requires, purpose="range_weighted")
        return forecast

    bid, ask = _quotes()
    quotes_only = MarketData({"bid_price": bid, "ask_price": ask}, DECISIONS, ASSETS)
    assert not quotes_only.has(*range_weighted.requires)
    with pytest.raises(MissingRolesError, match="open, high, low, close"):
        range_weighted(None, quotes_only)
    with pytest.raises(ValueError):
        sf.requires("spread")


# ── return -> portfolio ──────────────────────────────────────────────────────


def test_backtest_without_quotes_refuses_the_spread_charging_default():
    realized = _returns()
    with pytest.raises(MissingRolesError, match="cost_model='cross'.*cost_model='mid' runs without quotes"):
        run_backtest(_forecast(realized), realized)
    with pytest.raises(MissingRolesError, match="universe='tight_50pct'"):
        run_backtest(_forecast(realized), realized, config=MID.replace(universe="tight_50pct"))


def test_frictionless_backtest_needs_no_market_and_trades_every_name():
    realized = _returns()
    forecast = _forecast(realized)
    bare = run_backtest(forecast, realized, config=MID)
    # The same run with a zero spread everywhere: every name tradable, nothing charged.
    zero = run_backtest(forecast, realized, config=MID, half_spread=np.zeros((D, A)))
    assert bare.as_dict() == zero.as_dict()
    assert bare.mean_names_held == A


def test_quotes_and_their_half_spread_give_the_same_backtest():
    realized = _returns()
    forecast = _forecast(realized)
    bid, ask = _quotes()
    market = MarketData({"bid_price": bid, "ask_price": ask}, DECISIONS, ASSETS)
    by_market = run_backtest(forecast, realized, market=market)
    by_spread = run_backtest(forecast, realized, half_spread=market["half_spread"])
    assert by_market.as_dict() == by_spread.as_dict()
    with pytest.raises(ValueError, match="not both"):
        run_backtest(forecast, realized, market=market, half_spread=market["half_spread"])
    misaligned = MarketData({"bid_price": bid, "ask_price": ask}, DECISIONS + 1, ASSETS)
    with pytest.raises(ValueError, match="aligned"):
        run_backtest(forecast, realized, market=misaligned)


def test_the_config_space_shrinks_to_what_the_market_supports():
    no_quotes = MarketData.empty(DECISIONS, ASSETS)
    space = enumerate_configs(market=no_quotes)
    assert space and all(c.cost_model == "mid" and c.efq == 0.0 for c in space)
    assert {c.universe for c in space} == {"all"}
    full = enumerate_configs()
    assert len(space) < len(full)
    assert all(c in full or c.replace(cost_model="cross", efq=0.0) in full for c in space)

    realized = _returns()
    forecast = _forecast(realized)
    with pytest.raises(MissingRolesError):
        sweep_configs(forecast, realized, full[:3])
    kept = sweep_configs(forecast, realized, (MID, DEFAULT_BACKTEST), on_error="skip")
    assert [r.config for r in kept] == [MID]


# ── portfolio -> trades ──────────────────────────────────────────────────────


def test_fills_need_both_sides_of_the_book():
    bid, ask = _quotes()
    weights = np.random.default_rng(3).normal(0, 0.1, (D, A, 1))
    orders = sf.orders_from_weights(PortfolioWeights(weights, DECISIONS, ASSETS, [300]))
    market = MarketData({"bid_price": bid, "ask_price": ask}, DECISIONS, ASSETS)
    a, b = simulate_fills(orders, market), simulate_fills(orders, bid, ask)
    np.testing.assert_array_equal(a.price, b.price)

    spread_only = MarketData({"half_spread": market["half_spread"]}, DECISIONS, ASSETS)
    univariate = MarketData({"mid": market["mid"]}, DECISIONS, ASSETS)
    for lacking in (spread_only, univariate):
        with pytest.raises(MissingRolesError, match=r"simulate_fills\(at='touch'\) needs bid_price.*at='benchmark' fills"):
            simulate_fills(orders, lacking)
        with pytest.raises(MissingRolesError, match="cross_spread_sharpe"):
            cross_spread_sharpe(PortfolioWeights(weights, DECISIONS, ASSETS, [300]),
                                _returns(), lacking)
    with pytest.raises(TypeError):
        simulate_fills(orders, bid)


def test_mid_fills_need_no_quotes_and_reproduce_the_mid_marked_sharpe():
    """A cost-less fill is at the price the returns already assume: no column needed."""
    rng = np.random.default_rng(3)
    weights = PortfolioWeights(rng.normal(0, 0.1, (D, A, 1)), DECISIONS, ASSETS, [300])
    orders = sf.orders_from_weights(weights)
    realized = _returns()
    expect = sf.mid_price_sharpe(weights, realized)
    bid, ask = _quotes()
    univariate = MarketData({"mid": (bid + ask) / 2}, DECISIONS, ASSETS)
    for fills in (simulate_fills(orders, at="benchmark"),
                  simulate_fills(orders, MarketData.empty(DECISIONS, ASSETS), at="benchmark"),
                  simulate_fills(orders, univariate, at="benchmark"),
                  simulate_fills(orders, bid, ask, at="benchmark")):
        assert np.array_equal(fills.price, fills.mid_price, equal_nan=True)
        np.testing.assert_allclose(sf.positions_from_fills(fills).values, weights.values)
        result = sf.evaluate_fills(fills, realized)[0]
        assert (result.costs == 0).all()
        np.testing.assert_allclose(result.sharpe, expect[0])
    # A name with no midpoint at a decision is not observed there, so it does not fill.
    gappy = univariate.get("mid").copy()
    gappy[5, 3] = np.nan
    fills = simulate_fills(orders, MarketData({"mid": gappy}, DECISIONS, ASSETS), at="benchmark")
    assert fills.quantity[5, 3, 0] == 0 and fills.quantity[6, 3, 0] != 0
    with pytest.raises(ValueError, match="at must be"):
        simulate_fills(orders, at="vwap")


def test_cost_reading_stages_accept_a_spread_or_quotes_and_refuse_neither():
    bid, ask = _quotes()
    market = MarketData({"bid_price": bid, "ask_price": ask}, DECISIONS, ASSETS)
    univariate = MarketData({"mid": market["mid"]}, DECISIONS, ASSETS)
    realized = _returns()
    mu = _forecast(realized).values[:, :, 0]
    np.testing.assert_array_equal(edge_to_cost(mu, market),
                                  edge_to_cost(mu, market["half_spread"]))
    weights = PortfolioWeights(np.full((D, A, 1), 1 / A), DECISIONS, ASSETS, [300])
    assert (evaluate_weights(weights, realized, half_spread=market)[0].sharpe
            == evaluate_weights(weights, realized, half_spread=market["half_spread"])[0].sharpe)
    sigma = np.eye(A)[None] * 1e-4
    w_market = allocate(_forecast(realized), sigma, cost=market).values
    w_array = allocate(_forecast(realized), sigma, cost=market["half_spread"]).values
    np.testing.assert_array_equal(w_market, w_array)
    for call in (lambda: edge_to_cost(mu, univariate),
                 lambda: evaluate_weights(weights, realized, half_spread=univariate),
                 lambda: allocate(_forecast(realized), sigma, cost=univariate),
                 lambda: sf.threshold_selection(mu, np.ones_like(mu, bool), multiple=1.0,
                                                cost=univariate)):
        with pytest.raises(MissingRolesError):
            call()


# ── from the dataset ─────────────────────────────────────────────────────────


def _store(tmp_path, schema_roles, n=12):
    from stable_finance.dataset import FeatureSchema, FixedSession, sessions_from_frame
    from stable_finance.dataset import write_daystore

    cal = FixedSession()
    o, _ = cal.bounds("2024-03-05")
    rng = np.random.default_rng(4)
    rows = {"t": [], "sym": [], **{c: [] for c in schema_roles.values()}}
    for k in range(n):
        mid = 20 + k + rng.normal(0, 0.01, 390).cumsum()
        rows["t"].append(o + 60 * np.arange(390))
        rows["sym"].append(np.full(390, f"S{k:02d}"))
        for role, column in schema_roles.items():
            rows[column].append({"mid": mid, "bid_price": mid - 0.01,
                                 "ask_price": mid + 0.01}[role])
    frame = {key: np.concatenate(v) for key, v in rows.items()}
    schema = FeatureSchema.from_roles(**schema_roles)
    sessions = sessions_from_frame(frame, schema, Timeline(cal, bar_seconds=60), time="t", ticker="sym",
                                   stamp="start")
    return write_daystore(sessions, tmp_path, Timeline(cal, bar_seconds=60),
                          objective=Objective(horizons=(300,)))


def test_a_univariate_store_supports_only_frictionless_trading(tmp_path):
    days = _store(tmp_path, {"mid": "px"})
    market = market_from_days(days, anchors=[3600, 7200])
    assert market.available == ("mid",)
    assert market.shape == (2, 12)
    with pytest.raises(MissingRolesError):
        simulate_fills(Orders(np.zeros((2, 12, 1)), market.decisions, market.assets, [300]),
                       market)
    assert not DEFAULT_BACKTEST.supported_by(market) and MID.supported_by(market)


def test_a_quoted_store_carries_its_book_downstream(tmp_path):
    days = _store(tmp_path, {"bid_price": "b", "ask_price": "a"})
    market = market_from_days(days, anchors=[3600])
    assert market.has("bid_price", "ask_price", "half_spread", "mid")
    from stable_finance.dataset.storage.daystore import DayRecord

    record = DayRecord(days[0])
    row = record.anchor_row(3600)
    expect = np.asarray(record.features[:, record.schema.role_index("bid_price"), row])
    np.testing.assert_allclose(market["bid_price"][0], expect)


def test_a_panel_cache_without_quotes_says_so(tmp_path):
    from stable_finance import MarketPanel
    from stable_finance.dataset import (
        FixedSession, PanelCache, ViewSpec, build_session_panel, panel_cache_key,
        sessions_from_series,
    )
    from helpers import IdentityStats as _IdentityStats

    cal = FixedSession()
    o, _ = cal.bounds("2024-03-05")
    sessions = [s for k in range(3) for s in sessions_from_series(
        o + np.arange(400), 10 + k + np.linspace(0, 1, 400), timeline=Timeline(cal, bar_seconds=1), ticker=f"S{k}",
        stamp="start")]
    rows = [r for s in sessions for r in build_session_panel(
        s, [150, 200], _IdentityStats(), timeline=cal, target_types=("return",),
        horizons=(10,), view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)))]
    assert rows and all(r.quote is None for r in rows)
    key = panel_cache_key(anchors_per_day=2)
    cache = PanelCache(tmp_path)
    cache.build("2024-03", key, [(np.stack([r.view for r in rows]), rows)])
    assert cache.load("2024-03", key).quotes is None

    panel = MarketPanel.from_cache(tmp_path, "2024-03", anchors_per_day=2)
    assert not panel.has_quotes and "half_spread" not in panel.take([0, 1])
    with pytest.raises(MissingRolesError):
        panel.half_spread()
    market = panel.market([0], ["d"], ["S0"], decision_axis=["d"], asset_axis=["S0"])
    assert market.available == ()
