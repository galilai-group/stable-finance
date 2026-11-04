"""The price-impact execution model of Frazzini, Israel and Moskowitz (2018)."""
from __future__ import annotations

import numpy as np
import pytest

import stable_finance as sf
from stable_finance import (
    DEFAULT_BACKTEST,
    IMPACT_COEFFICIENTS,
    MarketData,
    MissingRolesError,
    Orders,
    PortfolioWeights,
    execute,
    impact_cost,
    run_backtest,
)

EQUITY = 1e6
D, A = 24, 3
DECISIONS = np.arange(D)
ASSETS = np.array(["X", "Y", "Z"])


def _market(adv=1e8, mid=None):
    mid = np.full((D, A), 100.0) if mid is None else mid
    return MarketData({"mid": mid, "adv": np.full((D, A), adv)}, DECISIONS, ASSETS)


def _orders(q):
    return Orders(q, DECISIONS, ASSETS, [60])


def test_the_formula_reproduces_the_papers_realized_us_costs():
    """Table X and section IV: 7.1 bps at 0.5% of daily volume, 15.4 at 2.4%."""
    bps = impact_cost([0.5, 2.4]) * 1e4
    assert bps[0] == pytest.approx(7.1, abs=1.0) and bps[1] == pytest.approx(15.4, abs=1.0)
    assert IMPACT_COEFFICIENTS == (0.0, -0.53, 11.21)
    # Concave: twice the size costs less than twice as much per dollar... in total,
    # and more per dollar.
    one, two = impact_cost(1.0), impact_cost(2.0)
    assert one < two < 2 * one
    with pytest.raises(ValueError, match="unsigned"):
        impact_cost(-1.0)


def test_instant_execution_is_the_papers_accounting():
    """schedule=0: the full position at the decision, charged the formula."""
    q = np.zeros((D, A, 1)); q[3, 1, 0] = 0.05                 # $50k, 0.05% of $100M ADV
    fills = execute(_orders(q), _market(), model="impact", equity=EQUITY, schedule=0)
    assert fills.quantity[3, 1, 0] == 0.05 and (fills.quantity != 0).sum() == 1
    assert fills.price[3, 1, 0] == pytest.approx(100 * (1 + impact_cost(0.05)))
    # A sell receives below the mid by the same amount.
    fills = execute(_orders(-q), _market(), model="impact", equity=EQUITY, schedule=0)
    assert fills.price[3, 1, 0] == pytest.approx(100 * (1 - impact_cost(0.05)))


def test_a_scheduled_order_ramps_the_position_linearly_at_the_evolving_mid():
    rng = np.random.default_rng(0)
    mid = 100 * np.exp(rng.normal(0, 1e-3, (D, A)).cumsum(axis=0))
    q = np.zeros((D, A, 1)); q[2, 0, 0] = 0.04
    fills = execute(_orders(q), _market(mid=mid), model="impact", equity=EQUITY, schedule=4)
    held = sf.positions_from_fills(fills).values[:, 0, 0]
    np.testing.assert_allclose(held[2:8], [0.0, 0.01, 0.02, 0.03, 0.04, 0.04])
    # Each slice is filled off that decision's own mid, plus the parent's impact.
    rate = impact_cost(100 * 0.04 * EQUITY / 1e8)
    np.testing.assert_allclose(fills.price[3:7, 0, 0], mid[3:7, 0] * (1 + rate))
    np.testing.assert_allclose(fills.mid_price[3:7, 0, 0], mid[3:7, 0])
    # The cost charged by the P&L is the formula's, whatever the mid did.
    result = sf.evaluate_fills(fills, sf.ForwardReturns(np.zeros((D, A, 1)), DECISIONS, ASSETS, [60]))[0]
    assert result.costs.sum() == pytest.approx(0.04 * rate)


def test_child_orders_within_a_schedule_are_charged_as_one_parent():
    """Concavity would reward slicing; the parent's size is what is priced."""
    whole = np.zeros((D, A, 1)); whole[5, 0, 0] = 0.3
    sliced = np.zeros((D, A, 1)); sliced[3:6, 0, 0] = 0.1
    zero = sf.ForwardReturns(np.zeros((D, A, 1)), DECISIONS, ASSETS, [60])
    def cost(q, schedule):
        fills = execute(_orders(q), _market(), model="impact", equity=EQUITY, schedule=schedule)
        return sf.evaluate_fills(fills, zero)[0].costs.sum()
    assert cost(sliced, 4) == pytest.approx(cost(whole, 4))
    # Outside one schedule they are separate parents, and cheaper in total.
    assert cost(sliced, 0) < cost(whole, 0)
    # An order against a parent in flight cancels its unfilled slices, which
    # are then not paid for: one slice of 0.3 fills, then 0.2 net remains.
    reverse = np.zeros((D, A, 1)); reverse[5, 0, 0] = 0.3; reverse[6, 0, 0] = -0.1
    fills = execute(_orders(reverse), _market(), model="impact", equity=EQUITY, schedule=4)
    assert sf.positions_from_fills(fills).values[-1, 0, 0] == pytest.approx(0.2)
    assert cost(reverse, 4) == pytest.approx(cost(whole, 4) * 0.2 / 0.3)


def test_slices_past_the_end_or_without_a_mid_are_not_filled():
    q = np.zeros((D, A, 1)); q[D - 2, 2, 0] = 0.02; q[4, 1, 0] = 0.02
    mid = np.full((D, A), 100.0); mid[6, 1] = np.nan
    fills = execute(_orders(q), _market(mid=mid), model="impact", equity=EQUITY, schedule=4)
    assert fills.quantity[D - 1, 2, 0] == pytest.approx(0.005)        # one slice of four
    assert fills.quantity[6, 1, 0] == 0 and fills.quantity[5:9, 1, 0].sum() == pytest.approx(0.015)


def test_the_model_states_what_it_needs():
    q = np.zeros((D, A, 1)); q[0, 0, 0] = 0.01
    with pytest.raises(MissingRolesError, match="adv"):
        execute(_orders(q), MarketData({"mid": np.full((D, A), 100.0)}, DECISIONS, ASSETS),
                model="impact", equity=EQUITY)
    with pytest.raises(ValueError, match="equity"):
        execute(_orders(q), _market(), model="impact")
    with pytest.raises(ValueError, match="limit"):
        execute(Orders(q, DECISIONS, ASSETS, [60], limit=0.0), _market(), model="impact",
                equity=EQUITY)
    with pytest.raises(ValueError, match="equity"):
        DEFAULT_BACKTEST.replace(cost_model="impact")
    with pytest.raises(ValueError, match="impact"):
        DEFAULT_BACKTEST.replace(cost_model="impact", equity=EQUITY, impact=(1.0, 2.0))
    with pytest.raises(ValueError, match="trading_cost|impact_cost"):
        sf.trading_cost("impact", np.ones((1, 1)))


def test_the_harness_works_orders_over_a_trading_day_read_from_the_days():
    rng = np.random.default_rng(1)
    realized = sf.ForwardReturns(rng.normal(0, 1e-3, (D, A, 1)), DECISIONS, ASSETS, [60])
    weights = PortfolioWeights(np.where(np.arange(D)[:, None, None] >= 4, 0.2, 0.0)
                               * np.ones((D, A, 1)), DECISIONS, ASSETS, [60])
    days = np.arange(D) // 6                                           # six decisions a day
    config = DEFAULT_BACKTEST.replace(cost_model="impact", equity=EQUITY, annualization="decisions")
    report = run_backtest(weights, realized, market=_market(), days=days, config=config,
                          with_series=True)
    assert "execution_days" not in config.label and "impact" not in report.as_dict()
    # One day of six decisions: the position reaches 0.2 six decisions after the order.
    instant = run_backtest(weights, realized, market=_market(), days=days,
                           config=config.replace(execution_days=0), with_series=True)
    assert instant.mean_gross_exposure > report.mean_gross_exposure
    assert "execution_days=0" in instant.config.label
    assert instant.mean_cost_bps == pytest.approx(report.mean_cost_bps, rel=1e-6)
    with pytest.raises(ValueError, match="days"):
        run_backtest(weights, realized, market=_market(), config=config)
    # A forecast runs the cost-aware allocator against the model's planning cost.
    forecast = sf.oracle_forecast(realized, 0.5, seed=2)
    planned = run_backtest(forecast, realized, market=_market(), days=days, config=config)
    assert np.isfinite(planned.sharpe_net)


# ── the stock-specific institutional model ──────────────────────────────────

from stable_finance import IMPACT_STOCK_COEFFICIENTS, ImpactStockCoefficients, impact_stock_cost  # noqa: E402
from stable_finance.dataset import FeatureSchema  # noqa: E402

REF = IMPACT_STOCK_COEFFICIENTS


def _stock_market(market_cap=4.9e9, idio_vol=30.0, market_vol=17.0, time_trend=972.0, **kw):
    base = _market(**kw)
    traits = {"market_cap": market_cap, "idio_vol": idio_vol, "market_vol": market_vol,
              "time_trend": time_trend}
    return MarketData({**base.columns, **{k: np.full((D, A), v) for k, v in traits.items()}},
                      DECISIONS, ASSETS)


def test_the_stock_model_is_the_simple_model_at_the_reference_name():
    np.testing.assert_allclose(
        impact_stock_cost([0.5, 2.4], 4.9e9, 30.0, 17.0, 972.0), impact_cost([0.5, 2.4]))
    # Column (10): cheaper for larger firms, dearer for volatile names and markets.
    bigger = impact_stock_cost(1.0, 100e9, 30.0, 17.0, 972.0)
    volatile = impact_stock_cost(1.0, 4.9e9, 60.0, 17.0, 972.0)
    crisis = impact_stock_cost(1.0, 4.9e9, 30.0, 40.0, 972.0)
    assert bigger < impact_cost(1.0) < volatile and crisis > impact_cost(1.0)
    assert volatile - impact_cost(1.0) == pytest.approx(0.31 * 30 / 1e4)
    assert crisis - impact_cost(1.0) == pytest.approx(0.12 * 23 / 1e4)
    assert bigger - impact_cost(1.0) == pytest.approx(-0.14 * (np.log1p(100) - np.log1p(4.9)) / 1e4)
    # Never a rebate.
    assert impact_stock_cost(0.0, 1e12, 5.0, 10.0, 972.0) == 0.0
    assert ImpactStockCoefficients(*REF) == REF and len(REF) == 11


def test_the_stock_model_needs_the_characteristics_and_prices_them_per_name():
    q = np.zeros((D, A, 1)); q[2, :, 0] = 0.05
    with pytest.raises(MissingRolesError, match="market_cap"):
        execute(_orders(q), _market(), model="impact_stock", equity=EQUITY)
    market = _stock_market()
    columns = dict(market.columns)
    columns["idio_vol"] = np.tile([20.0, 30.0, 60.0], (D, 1))
    market = MarketData(columns, DECISIONS, ASSETS)
    fills = execute(_orders(q), market, model="impact_stock", equity=EQUITY, schedule=0)
    slip = (fills.price[2, :, 0] - 100) / 100
    assert slip[0] < slip[1] < slip[2]
    assert slip[1] == pytest.approx(impact_cost(100 * 0.05 * EQUITY / 1e8))
    with pytest.raises(ValueError, match="11 finite"):
        DEFAULT_BACKTEST.replace(cost_model="impact_stock", equity=EQUITY, impact=(0, 1, 2))


def test_the_harness_runs_the_stock_model():
    rng = np.random.default_rng(3)
    realized = sf.ForwardReturns(rng.normal(0, 1e-3, (D, A, 1)), DECISIONS, ASSETS, [60])
    forecast = sf.oracle_forecast(realized, 0.5, seed=4)
    days = np.arange(D) // 6
    config = DEFAULT_BACKTEST.replace(cost_model="impact_stock", equity=EQUITY,
                                      annualization="decisions")
    report = run_backtest(forecast, realized, market=_stock_market(), days=days, config=config)
    simple = run_backtest(forecast, realized, market=_stock_market(),
                          days=days, config=config.replace(cost_model="impact"))
    assert np.isfinite(report.sharpe_net)
    assert report.mean_cost_bps == pytest.approx(simple.mean_cost_bps)   # reference name
    with pytest.raises(MissingRolesError, match="impact_stock"):
        run_backtest(forecast, realized, market=_market(), days=days, config=config)


def test_evaluation_inputs_computes_the_characteristics():
    rng = np.random.default_rng(5)
    T0 = 1_600_000_000 - 1_600_000_000 % 86_400
    n_days, per_day, names = 60, 20, 4
    rows = {"timestamp": [], "ticker": [], "px": [], "vol": [], "cap": []}
    market_daily = rng.normal(0, 0.01, n_days)
    for a in range(names):
        p = 100.0
        for d in range(n_days):
            p *= 1 + market_daily[d] * (0.5 + a / 2) + rng.normal(0, 0.005 * (a + 1))
            for k in range(per_day):
                rows["timestamp"].append(T0 + 86_400 * d + 34_200 + 60 * k)
                rows["ticker"].append(f"S{a}"); rows["px"].append(p); rows["vol"].append(1000.0)
                rows["cap"].append(1e9 * (a + 1))
    frame = {k: np.asarray(v) for k, v in rows.items()}
    schema = FeatureSchema.from_roles(mid="px", volume="vol", market_cap="cap")
    decisions = T0 + 86_400 * np.array([25, 40, 59]) + 34_200 + 60 * 5
    realized, market = sf.evaluation_inputs(frame, schema, decisions, [300],
                                            adv_lookback=10 * 86_400, vol_lookback=30 * 86_400)
    assert market.has(*sf.COST_MODEL_REQUIRES["impact_stock"])
    idio, mvol = market.get("idio_vol"), market.get("market_vol")
    assert np.isfinite(idio).all() and np.isfinite(mvol).all()
    # The noisiest name (largest residual sigma by construction) is more
    # idiosyncratic than the quietest; with four names the equal-weighted
    # market absorbs some of each name's own noise, so only the ends are ordered.
    assert (idio[:, 0] < idio[:, -1]).all() and (idio > 1).all() and (idio < 100).all()
    assert (mvol[:, 0] == mvol[:, 1]).all() and 5 < mvol[0, 0] < 40
    np.testing.assert_array_equal(market.get("market_cap")[0], 1e9 * np.arange(1, 5))
    np.testing.assert_array_equal(market.get("time_trend")[:, 0],
                                  (np.asarray(decisions, "datetime64[s]").astype("datetime64[M]")
                                   - np.datetime64("1926-06", "M")).astype(int) + 1)
    assert market.get("time_trend")[0, 0] == pytest.approx(1132, abs=1)   # Sep 2020


def test_volatilities_align_names_by_day_whatever_time_each_last_traded():
    """A name that stops trading at noon and one that trades to the close share a day."""
    rng = np.random.default_rng(7)
    T0 = 1_600_000_000 - 1_600_000_000 % 86_400
    n_days = 40
    common = rng.normal(0, 0.01, n_days)
    rows = {"timestamp": [], "ticker": [], "px": []}
    for a, last_minute in enumerate((30, 390)):            # early and late names
        p = 100.0
        for d in range(n_days):
            p *= 1 + common[d] + rng.normal(0, 0.002)
            for k in (0, last_minute):
                rows["timestamp"].append(T0 + 86_400 * d + 34_200 + 60 * k)
                rows["ticker"].append(f"S{a}"); rows["px"].append(p)
    frame = {k: np.asarray(v) for k, v in rows.items()}
    decision = np.array([T0 + 86_400 * (n_days - 1) + 34_200 + 60 * 400])
    _, market = sf.evaluation_inputs(frame, FeatureSchema.from_roles(mid="px"), decision, [60],
                                     vol_lookback=n_days * 86_400)
    idio, mvol = market.get("idio_vol")[0], market.get("market_vol")[0]
    # Both names move with the common factor: the market is volatile (about
    # 16%) and each name's residual against it is small (about 3%), which
    # misaligned days would turn into residuals as large as the market.
    assert 10 < mvol[0] < 25 and (idio < 8).all()
    # Nothing is known before the late name's close.
    _, early = sf.evaluation_inputs(frame, FeatureSchema.from_roles(mid="px"),
                                    decision - 60 * 100, [60], vol_lookback=n_days * 86_400)
    assert np.isfinite(early.get("idio_vol")).all()
