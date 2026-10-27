import numpy as np
import pytest

from stable_finance import (
    DEFAULT_BACKTEST,
    ForwardReturns,
    MarketData,
    Orders,
    PortfolioWeights,
    fill_path,
    positions_from_fills,
    run_backtest,
)

AXIS = np.array([300])


def grid(values, assets=("A",)):
    v = np.asarray(values, dtype=np.float64)
    return v.reshape(len(v), len(assets), 1)


def quotes(bid, ask, n_assets=1):
    bid, ask = np.asarray(bid, dtype=np.float64), np.asarray(ask, dtype=np.float64)
    shape = (len(bid), n_assets)
    return MarketData({"bid_price": np.broadcast_to(bid[:, None], shape).copy(),
                       "ask_price": np.broadcast_to(ask[:, None], shape).copy()},
                      np.arange(len(bid)), np.array(["A", "B"][:n_assets]))


def test_drift_holding_earns_the_compounded_price_change():
    decisions, assets = np.arange(3), np.array(["A"])
    r = [0.10, 0.10, np.nan]                              # 100 -> 110 -> 121
    realized = ForwardReturns(grid(r), decisions, assets, AXIS)
    orders = Orders(grid([0.5, 0.0, -0.5]), decisions, assets, AXIS)
    mid = DEFAULT_BACKTEST.replace(cost_model="mid", annualization="decisions")
    held = run_backtest(orders, realized, config=mid.replace(holding="drift"), with_series=True)
    rebal = run_backtest(orders, realized, config=mid, with_series=True)
    assert held.series.gross.sum() == pytest.approx(0.5 * (1.21 - 1))
    assert rebal.series.gross.sum() == pytest.approx(0.5 * 0.20)


def test_drift_holds_an_unchanged_target_instead_of_rebalancing_it():
    decisions, assets = np.arange(4), np.array(["A"])
    drift = grid([0.10, -0.05, 0.02, np.nan])
    target = PortfolioWeights(grid([1.0, 1.0, 1.0, 0.0]), decisions, assets, AXIS)
    fills = fill_path(target, model="mid", drift=drift)
    q = fills.quantity[:, 0, 0]
    assert q[0] == 1.0 and q[1] == q[2] == 0.0
    assert q[3] == pytest.approx(-(1.10 * 0.95 * 1.02))   # closes exactly what the position became
    held = positions_from_fills(fills, drift=drift).values[:, 0, 0]
    assert held[-1] == pytest.approx(0.0)


def test_a_missed_exit_is_retried_by_default_and_left_open_when_cancelled():
    decisions, assets = np.arange(3), np.array(["A"])
    market = quotes([99, np.nan, 99], [101, np.nan, 101])   # no quote when the exit is sent
    realized = ForwardReturns(np.zeros((3, 1, 1)), decisions, assets, AXIS)
    orders = Orders(grid([0.5, -0.5, 0.0]), decisions, assets, AXIS)
    config = DEFAULT_BACKTEST.replace(cost_model="cross", annualization="decisions")
    retried = run_backtest(orders, realized, market=market, config=config, with_series=True)
    cancelled = run_backtest(orders, realized, market=market, config=config, unfilled="cancel",
                             with_series=True)
    assert retried.mean_turnover == pytest.approx(1.0 / 3)      # in and out
    assert cancelled.mean_turnover == pytest.approx(0.5 / 3)    # in only: still open
    with pytest.raises(ValueError, match="unfilled"):
        run_backtest(orders, realized, market=market, config=config, unfilled="never")


def test_retry_and_cancel_agree_when_every_order_fills():
    rng = np.random.default_rng(0)
    decisions, assets = np.arange(20), np.array(["A", "B"])
    market = quotes(np.full(20, 99.0), np.full(20, 101.0), n_assets=2)
    realized = ForwardReturns(rng.normal(0, 0.01, (20, 2, 1)), decisions, assets, AXIS)
    orders = Orders(rng.normal(0, 0.1, (20, 2, 1)), decisions, assets, AXIS)
    config = DEFAULT_BACKTEST.replace(cost_model="cross", annualization="decisions")
    a = run_backtest(orders, realized, market=market, config=config)
    b = run_backtest(orders, realized, market=market, config=config, unfilled="cancel")
    assert a.sharpe_net == pytest.approx(b.sharpe_net)
    assert a.mean_cost_bps == pytest.approx(b.mean_cost_bps)


def test_holding_is_validated_and_kept_out_of_old_labels():
    with pytest.raises(ValueError, match="holding"):
        DEFAULT_BACKTEST.replace(holding="sometimes")
    assert "holding" not in DEFAULT_BACKTEST.label
    assert "holding=drift" in DEFAULT_BACKTEST.replace(holding="drift").label
    decisions, assets = np.arange(2), np.array(["A"])
    realized = ForwardReturns(np.zeros((2, 1, 1)), decisions, assets, AXIS)
    orders = Orders(grid([0.5, -0.5]), decisions, assets, AXIS)
    with pytest.raises(ValueError, match="drift"):
        run_backtest(orders, realized, config=DEFAULT_BACKTEST.replace(
            cost_model="impact", holding="drift", equity=1e6))


def test_the_daily_pnl_series_is_labelled_by_the_callers_days():
    decisions, assets = np.arange(4), np.array(["A"])
    realized = ForwardReturns(grid([0.01, 0.02, 0.03, 0.04]), decisions, assets, AXIS)
    weights = PortfolioWeights(np.ones((4, 1, 1)), decisions, assets, AXIS)
    days = np.array(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    report = run_backtest(weights, realized, config=DEFAULT_BACKTEST.replace(cost_model="mid"),
                          days=days, with_series=True)
    assert list(report.series.periods) == ["2024-01-02", "2024-01-03"]
    assert report.series.gross == pytest.approx([0.03, 0.07])
