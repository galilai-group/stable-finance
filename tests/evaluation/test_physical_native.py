"""Native physical execution keeps actual, pending and cash states distinct."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.physical import (
    PhysicalExecutionConfig, PricePanel, execute_portfolio_targets,
)
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.scoring.holdings_attribution import execution_attribution


def targets(values, decisions, assets=None):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    names = np.asarray(assets if assets is not None else [f"A{i}" for i in range(array.shape[1])])
    return PortfolioWeights(array[:, :, None], decisions, names, [1])


def prices(values, *, times=None, assets=None):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    return PricePanel(array, np.arange(len(array)) if times is None else times,
                      [f"A{i}" for i in range(array.shape[1])] if assets is None else assets)


def quotes(panel, spread=1.):
    return MarketData({"mid":panel.values, "bid_price":panel.values-spread,
                       "ask_price":panel.values+spread}, panel.times, panel.assets)


def run(goal, marks, market=None, *, delay=0, model="mid", efq=.7, end=None, **kwargs):
    return execute_portfolio_targets(goal, marks, market,
        config=PhysicalExecutionConfig(model=model, delay_seconds=delay, efq=efq),
        initial_equity=1000., strategy_end=end, periods_per_year=252., **kwargs)


def assert_accounting(ledger, marks):
    np.testing.assert_allclose(ledger.cash+(ledger.shares*marks.values).sum(axis=1),
                               ledger.posttrade_nav, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(ledger.nav[1:], ledger.nav[:-1]*(1+ledger.net_returns),
                               rtol=1e-12, atol=1e-10)
    gross = np.sum(ledger.shares[:-1]*np.diff(marks.values, axis=0))/ledger.initial_equity
    cost = (ledger.arrival_slippage.sum()+ledger.commissions.sum())/ledger.initial_equity
    assert ledger.total_return == pytest.approx(gross-cost, abs=1e-12)
    if ledger.liquidated:
        np.testing.assert_allclose(ledger.shares[-1], 0., atol=1e-10)
        np.testing.assert_allclose(ledger.pending_shares[-1], 0., atol=1e-10)


class ObservedPolicy:
    def __init__(self):
        self.states = []

    def validate(self, n_decisions, n_assets):
        self.shape = (n_decisions, n_assets)

    def plan(self, target, previous, pending, decision_index, n_decisions):
        self.states.append((np.array(previous), np.array(pending)))
        return target


def test_price_panel_copies_inputs_and_prevents_alias_mutation():
    value = np.full((3, 2), 100.)
    time, names = np.arange(3), np.array(["A", "B"])
    panel = PricePanel(value, time, names)
    value[0, 0], time[0], names[0] = 0., -100, "other"
    assert panel.values[0, 0] == 100. and panel.times[0] == 0 and panel.assets[0] == "A"
    for item in (panel.values, panel.times, panel.assets):
        assert not item.flags.writeable


@pytest.mark.parametrize("kind", ["negative", "missing", "shape", "duplicate_assets", "missing_asset",
                                  "nan_asset", "empty_asset", "duplicate_times", "nat_time", "string_time"])
def test_bad_price_or_clock_contract_is_rejected(kind):
    value, times, names = np.full((3, 2), 100.), np.arange(3), np.array(["A", "B"], dtype=object)
    if kind == "negative": value[0, 0] = 0.
    elif kind == "missing": value[0, 0] = np.nan
    elif kind == "shape": value = value[:2]
    elif kind == "duplicate_assets": names[1] = "A"
    elif kind == "missing_asset": names[1] = None
    elif kind == "nan_asset": names[1] = np.nan
    elif kind == "empty_asset": names[1] = ""
    elif kind == "duplicate_times": times[1] = times[0]
    elif kind == "nat_time": times = np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[D]")
    else: times = ["a", "b", "c"]
    with pytest.raises((ValueError, TypeError)):
        PricePanel(value, times, names)


def test_mid_ledger_marks_actual_shares_and_current_nav():
    panel = prices([100., 110., 110., 110.])
    policy = ObservedPolicy()
    ledger = run(targets([.5, .5], [0, 1]), panel, trade_policy=policy)
    assert ledger.total_return == pytest.approx(.05)
    assert ledger.decision_nav[1] == pytest.approx(1050.)
    assert ledger.actual_decision_weights[1, 0] == pytest.approx(550/1050)
    assert ledger.shares[0, 0] == 5.
    assert ledger.shares[1, 0] == pytest.approx(1050*.5/110)
    assert policy.shape == (2, 1)
    np.testing.assert_array_equal(policy.states[1][0], ledger.actual_decision_weights[1])
    assert_accounting(ledger, panel)


def test_current_pending_orders_are_committed_but_not_filled_positions():
    panel = prices(np.full(9, 100.))
    policy = ObservedPolicy()
    ledger = run(targets([.2, .2, 0.], [0, 2, 4]), panel, delay=3, end=5,
                 trade_policy=policy)
    np.testing.assert_array_equal(policy.states[1][0], [0.])
    np.testing.assert_array_equal(policy.states[1][1], [.2])
    assert ledger.requested_shares[2, 0] == 0.
    assert ledger.filled_shares[3, 0] == 2.
    assert ledger.filled_shares[7, 0] == -2.
    assert_accounting(ledger, panel)


def test_same_time_arrivals_settle_before_the_policy_reads_current_state():
    panel = prices(np.full(7, 100.))
    policy = ObservedPolicy()
    ledger = run(targets([.2, .2], [0, 2]), panel, delay=2, end=4, trade_policy=policy)
    np.testing.assert_array_equal(policy.states[1][0], [.2])
    np.testing.assert_array_equal(policy.states[1][1], [0.])
    assert ledger.requested_shares[2, 0] == 0.
    assert_accounting(ledger, panel)


def test_missing_arrival_quote_misses_only_that_name_and_retries_actual_state():
    panel = prices(np.full((26, 2), 100.))
    bid, ask = panel.values-1., panel.values+1.
    bid[5, 0] = ask[5, 0] = np.nan
    market = MarketData({"mid":panel.values, "bid_price":bid, "ask_price":ask}, panel.times, panel.assets)
    policy = ObservedPolicy()
    ledger = run(targets([[.2, -.2], [.2, -.2]], [0, 10]), panel, market,
                 model="cross", delay=5, end=20, trade_policy=policy)
    assert ledger.filled_shares[5, 0] == 0.
    assert ledger.unfilled_shares[5, 0] == 2.
    assert ledger.filled_shares[5, 1] == -2.
    assert ledger.actual_decision_weights[1, 0] == 0.
    assert ledger.pending_decision_weights[1, 0] == 0.
    assert ledger.requested_shares[10, 0] > 0.
    assert ledger.filled_shares[15, 0] > 0.
    assert_accounting(ledger, panel)


def test_spread_and_commission_are_charged_once_including_terminal_fill():
    panel = prices(np.full(3, 100.))
    ledger = run(targets([.5], [0]), panel, quotes(panel), model="cross", efq=.5,
                 cost_bps=10.)
    assert ledger.arrival_slippage.sum() == pytest.approx(5.)
    assert ledger.commissions.sum() == pytest.approx(1.)
    assert ledger.total_return == pytest.approx(-.006)
    assert len(ledger.order_records) == 2
    assert ledger.order_records[-1].terminal
    assert_accounting(ledger, panel)
    report = execution_attribution(ledger, panel, beta=[1.])
    assert report["totals"]["net_return"] == pytest.approx(-.006)
    assert report["totals"]["cost_pnl"] == pytest.approx(6.)


def test_initial_weights_define_real_shares_cash_and_observed_initial_state():
    panel = prices(np.array([[100., 50.], [100., 50.], [100., 50.]]))
    policy = ObservedPolicy()
    start = np.array([.4, -.2])
    ledger = run(targets([[.4, -.2]], [0]), panel, initial_weights=start, trade_policy=policy)
    np.testing.assert_allclose(ledger.shares[0], [4., -4.])
    assert ledger.cash[0] == pytest.approx(800.)
    np.testing.assert_array_equal(ledger.actual_decision_weights[0], start)
    assert ledger.requested_shares[0].sum() == 0.
    assert not ledger.initial_weights.flags.writeable
    start[:] = 0.
    np.testing.assert_array_equal(ledger.initial_weights, [.4, -.2])
    assert ledger.total_return == 0.
    assert_accounting(ledger, panel)


def test_initial_holdings_drift_before_first_target_is_sent():
    panel = prices([100., 110., 110., 110.])
    policy = ObservedPolicy()
    ledger = run(targets([.5], [1]), panel, initial_weights=[.5], trade_policy=policy)
    assert ledger.decision_nav[0] == pytest.approx(1050.)
    assert policy.states[0][0][0] == pytest.approx(550/1050)
    assert ledger.total_return == pytest.approx(.05)
    assert_accounting(ledger, panel)


@pytest.mark.parametrize("initial", [0., [0., 0.], [np.nan], [np.inf], [True]])
def test_initial_weights_require_one_finite_real_weight_per_asset(initial):
    panel = prices(np.full(3, 100.))
    with pytest.raises(ValueError, match="initial_weights"):
        run(targets([0.], [0]), panel, initial_weights=initial)


@pytest.mark.parametrize("clock", ["numeric", "numpy_datetime", "aware_datetime"])
def test_exact_second_arrivals_on_supported_clocks(clock):
    time = np.arange(7)
    if clock == "numpy_datetime":
        time = np.datetime64("2020-01-01T00:00:00")+time.astype("timedelta64[s]")
    elif clock == "aware_datetime":
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        time = np.array([start+timedelta(seconds=int(s)) for s in time], dtype=object)
    panel = prices(np.full(7, 100.), times=time)
    ledger = run(targets([.1], time[[0]]), panel, delay=2, end=time[4])
    assert ledger.filled_shares[2, 0] == 1.
    assert ledger.filled_shares[6, 0] == -1.
    assert_accounting(ledger, panel)


def test_five_second_arrival_cannot_be_rounded_to_minute_data():
    panel = prices(np.full(4, 100.), times=np.arange(4)*60)
    with pytest.raises(ValueError, match="exact arrival"):
        run(targets([.1], [0]), panel, delay=5, end=120)


def test_nonuniform_price_marks_are_refused_and_delay_needs_close_boundary():
    panel = prices(np.full(4, 100.), times=[0, 1, 3, 4])
    with pytest.raises(ValueError, match="uniform spacing"):
        run(targets([.1], [0]), panel)
    panel = prices(np.full(5, 100.))
    with pytest.raises(ValueError, match="strategy_end"):
        run(targets([.1], [0]), panel, delay=1)


@pytest.mark.parametrize("kwargs", [dict(model="book"), dict(schedule=1), dict(limit=.1),
                                  dict(efq=-.1), dict(delay_seconds=True)])
def test_unsupported_physical_execution_settings_fail_explicitly(kwargs):
    with pytest.raises((ValueError, TypeError)):
        PhysicalExecutionConfig(**kwargs)


def test_terminal_missing_quote_leaves_an_explicit_failure():
    panel = prices(np.full(4, 100.))
    bid, ask = panel.values-1., panel.values+1.
    bid[-1] = ask[-1] = np.nan
    market = MarketData({"mid":panel.values, "bid_price":bid, "ask_price":ask}, panel.times, panel.assets)
    with pytest.raises(ValueError, match="terminal liquidation"):
        run(targets([.1], [0]), panel, market, model="cross")


def test_unliquidated_run_reports_open_physical_position_without_invented_close():
    panel = prices(np.full(3, 100.))
    ledger = run(targets([.1], [0]), panel, liquidate=False)
    assert not ledger.liquidated
    np.testing.assert_array_equal(ledger.shares[-1], [1.])
    assert not any(row.terminal for row in ledger.order_records)
    assert_accounting(ledger, panel)


def test_policy_contract_and_nonfinite_plans_are_rejected_before_sending():
    panel, goal = prices(np.full(3, 100.)), targets([.1], [0])
    with pytest.raises(TypeError, match="validate and plan"):
        run(goal, panel, trade_policy=object())
    policy = ObservedPolicy()
    policy.plan = lambda *args:np.array([np.nan])
    with pytest.raises(ValueError, match="finite planned weight"):
        run(goal, panel, trade_policy=policy)
