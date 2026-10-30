"""Native objective backtesting preserves rewards, physical state and accounting."""

from dataclasses import replace

import numpy as np
import pytest

from stable_finance import DEFAULT_BACKTEST
from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.physical import PricePanel
from stable_finance.evaluation.portfolio.joint_constraints import JointPortfolioConstraints
from stable_finance.evaluation.portfolio.joint_risk import JointReturnPanel, JointRiskModel
from stable_finance.evaluation.return_risk_backtest import (
    BacktestExecutionState, ReturnRiskBacktestSpec, run_return_risk_backtest,
)
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.trades.objective import solve_return_risk_rebalance


def inputs(*, joint=False, delay=2):
    names, decisions = np.array(["A", "B"]), np.array([0, 10, 20])
    horizons = np.array([10, 20] if joint else [10])
    values = np.empty((3, 2, len(horizons)))
    values[:, :, 0] = [.02, -.02]
    if joint:
        values[:, :, 1] = [.03, -.03]
    covariance = .1*np.eye(2)
    if joint:
        covariance = np.block([[.1*np.eye(2), .04*np.eye(2)],
                               [.04*np.eye(2), .2*np.eye(2)]])
    forecast = JointReturnPanel(values, decisions, names, horizons, entry_delay_seconds=delay)
    risk = JointRiskModel(covariance, names, horizons, -1, entry_delay_seconds=delay)
    prices = PricePanel(np.full((31+delay, 2), 100.), np.arange(31+delay), names)
    market = MarketData({"mid":prices.values, "bid_price":prices.values-.01,
                         "ask_price":prices.values+.01}, prices.times, names)
    spec = ReturnRiskBacktestSpec(risk, strategy_end=30)
    config = DEFAULT_BACKTEST.replace(cost_model="mid", annualization="decisions")
    return forecast, prices, market, spec, config


def run(forecast, prices, market, spec, config, **kwargs):
    return run_return_risk_backtest(forecast, prices, market=market,
        objective=spec, config=config, **kwargs)


def test_single_endpoint_retains_original_alpha_and_native_trace():
    forecast, prices, market, spec, config = inputs()
    report = run(forecast, prices, market, spec, config, with_series=True)
    trace = report.objective_trace
    direct = solve_return_risk_rebalance([.02, -.02], spec.risk.covariance,
        previous=[0., 0.], cost=0., risk_aversion=5.)
    np.testing.assert_allclose(trace.ledger.planned_weights[0], direct.weights, atol=1e-10)
    assert report.horizon == 10 and report.n_periods == len(prices.times)-1
    assert trace.reference_plans.shape == (3, 1, 2)
    assert trace.ledger.liquidated
    np.testing.assert_allclose(trace.final_state.weights, 0., atol=1e-12)
    np.testing.assert_allclose(np.prod(1+report.series.net)-1, trace.ledger.total_return, atol=1e-12)
    assert np.isnan(report.information_coefficient) and np.isnan(report.transfer_coefficient)
    assert "objective_trace" not in report.as_dict()
    metadata = trace.report_metadata
    assert metadata["input_mode"] == "return_risk_physical"
    assert metadata["objective_gross_limit"] == 1.
    assert metadata["entry_delay_seconds"] == 2
    assert metadata["report_period"] == "physical_marks"
    assert not trace.reference_plans.flags.writeable and not trace.planning_cost.flags.writeable


def test_cross_interval_risk_changes_current_plan_and_prefix_is_preserved():
    forecast, prices, market, spec, config = inputs(joint=True)
    spec = replace(spec, stage_counts=[2, 1, 1])
    full = run(forecast, prices, market, spec, config).objective_trace
    diagonal = spec.risk.covariance.copy()
    diagonal[:2, 2:] = diagonal[2:, :2] = 0.
    risk = JointRiskModel(diagonal, forecast.assets, forecast.horizons, -1,
                          entry_delay_seconds=forecast.entry_delay_seconds)
    separate = run(forecast, prices, market, replace(spec, risk=risk), config).objective_trace
    expected_current = np.linalg.solve(np.array([[.1, .04], [.04, .2]]), [.02/5., .01/5.])[0]
    np.testing.assert_allclose(full.reference_plans[0, 0], [expected_current, -expected_current], atol=1e-10)
    np.testing.assert_allclose(separate.reference_plans[0, 0], [.04, -.04], atol=1e-10)
    assert np.max(np.abs(full.reference_plans[0, 0]-separate.reference_plans[0, 0])) > 1e-4
    np.testing.assert_array_equal(full.stage_counts, [2, 1, 1])
    np.testing.assert_array_equal(full.reference_plans[1:, 1], 0.)
    np.testing.assert_array_equal(full.ledger.planned_weights, full.reference_plans[:, 0])


def test_native_run_backtest_dispatch_uses_the_same_original_objective():
    from stable_finance.evaluation.backtest import run_backtest

    forecast, prices, market, spec, config = inputs(joint=True)
    direct = run(forecast, prices, market, spec, config)
    native = run_backtest(forecast, prices, market=market, objective=spec, config=config)
    np.testing.assert_array_equal(native.objective_trace.reference_plans,
                                  direct.objective_trace.reference_plans)
    np.testing.assert_array_equal(native.objective_trace.ledger.nav, direct.objective_trace.ledger.nav)


def test_initial_actual_holdings_are_not_a_previous_target():
    forecast, prices, market, spec, config = inputs()
    state = BacktestExecutionState(equity=1000., weights=[.03, -.03])
    trace = run(forecast, prices, market, spec, config, execution_state=state).objective_trace
    np.testing.assert_allclose(trace.ledger.actual_decision_weights[0], [.03, -.03], atol=1e-12)
    np.testing.assert_allclose(trace.ledger.decision_nav[0], 1000.)
    np.testing.assert_allclose(trace.ledger.planned_trade_weights[0],
        trace.ledger.planned_weights[0]-np.array([.03, -.03]), atol=1e-12)
    assert not trace.initial_state.weights.flags.writeable


def test_pending_orders_are_current_commitments_and_not_paid_twice():
    forecast, prices, market, spec, config = inputs(delay=5)
    forecast = JointReturnPanel(forecast.values, [0, 1, 2], forecast.assets,
        forecast.horizons, entry_delay_seconds=5)
    report = run(forecast, prices, market, spec, config.replace(cost_model="cross"))
    trace = report.objective_trace
    assert np.abs(trace.ledger.pending_decision_weights[1]).sum() > .01
    np.testing.assert_array_equal(trace.ledger.actual_decision_weights[1], 0.)
    assert trace.certificates[1].estimated_cost == pytest.approx(0., abs=1e-12)
    np.testing.assert_allclose(trace.ledger.planned_trade_weights[1], 0., atol=1e-10)
    np.testing.assert_allclose(trace.ledger.cash+(trace.ledger.shares*prices.values).sum(axis=1),
                               trace.ledger.posttrade_nav, atol=1e-12)


def test_missed_arrival_fill_and_current_missing_quote_feed_actual_state():
    forecast, prices, _, spec, config = inputs()
    marks = prices.values.copy()
    marks[5:, 1] = 110.
    prices = PricePanel(marks, prices.times, prices.assets)
    bid, ask = marks-.01, marks+.01
    bid[2, 0] = ask[2, 0] = np.nan
    bid[20, 1] = ask[20, 1] = np.nan
    market = MarketData({"mid":marks, "bid_price":bid, "ask_price":ask}, prices.times, prices.assets)
    report = run(forecast, prices, market, spec, config.replace(cost_model="cross"))
    trace, ledger = report.objective_trace, report.objective_trace.ledger
    assert ledger.unfilled_shares[2, 0] > 0 and ledger.filled_shares[2, 0] == 0
    assert ledger.filled_shares[2, 1] < 0
    assert np.max(np.abs(ledger.actual_decision_weights[1]-ledger.planned_weights[0])) > .01
    assert not trace.tradable[2, 1]
    assert ledger.requested_shares[20, 1] == pytest.approx(0., abs=1e-12)
    np.testing.assert_allclose(ledger.planned_weights[2, 1], ledger.actual_decision_weights[2, 1], atol=1e-12)
    np.testing.assert_allclose(ledger.nav[1:], ledger.nav[:-1]*(1+ledger.net_returns), atol=1e-12)


def test_daily_series_compounds_cash_nav_and_preserves_calendar_labels():
    forecast, prices, _, spec, config = inputs()
    marks = prices.values.copy()
    marks[:, 0] = 100.*1.02**np.arange(len(marks))
    prices = PricePanel(marks, prices.times, prices.assets)
    market = MarketData({"mid":marks, "bid_price":marks-.01, "ask_price":marks+.01}, prices.times, prices.assets)
    days = np.where(prices.times < 16, "session-A", "session-B")
    report = run(forecast, prices, market, spec,
        config.replace(cost_model="cross", annualization="daily"), days=days, with_series=True)
    ledger, series = report.objective_trace.ledger, report.series
    np.testing.assert_array_equal(series.periods, ["session-A", "session-B"])
    assert report.periods_per_year == 252. and report.n_periods == 2
    np.testing.assert_allclose(series.gross-series.cost, series.net, atol=1e-12)
    np.testing.assert_allclose(np.prod(1+series.net)-1, ledger.total_return, atol=1e-12)
    assert abs(series.net.sum()-ledger.total_return) > 1e-8


def test_future_prices_and_future_forecasts_cannot_change_earlier_plans():
    forecast, prices, market, spec, config = inputs(joint=True)
    original = run(forecast, prices, market, spec, config).objective_trace
    changed_values = forecast.values.copy()
    changed_values[2] *= -3
    changed = JointReturnPanel(changed_values, forecast.decisions, forecast.assets,
        forecast.horizons, entry_delay_seconds=forecast.entry_delay_seconds)
    marks = prices.values.copy()
    marks[12:, 0] *= 1.2
    later_prices = PricePanel(marks, prices.times, prices.assets)
    later_market = MarketData({"mid":marks}, prices.times, prices.assets)
    later = run(changed, later_prices, later_market, spec, config).objective_trace
    np.testing.assert_array_equal(original.reference_plans[:2], later.reference_plans[:2])
    assert np.max(np.abs(original.reference_plans[2]-later.reference_plans[2])) > .01


def test_risk_or_beta_cannot_be_used_before_its_availability():
    forecast, prices, market, spec, config = inputs()
    future_risk = JointRiskModel(spec.risk.covariance, forecast.assets, forecast.horizons, 1,
                                entry_delay_seconds=forecast.entry_delay_seconds)
    with pytest.raises(ValueError, match="information_cutoff"):
        run(forecast, prices, market, replace(spec, risk=future_risk), config)
    with pytest.raises(ValueError, match="beta_available_at"):
        run(forecast, prices, market, replace(spec, beta=[1., 2.], max_abs_beta=.1,
                                             beta_available_at=1), config)


@pytest.mark.parametrize("prefix", [True, [True, True, True], [1., 1., 1.], [0, 1, 1], [3, 1, 1], [1, 1], np.uint64(2**64-1)])
def test_invalid_or_ambiguous_stage_prefixes_are_rejected(prefix):
    forecast, prices, market, spec, config = inputs(joint=True)
    with pytest.raises(ValueError, match="stage_counts"):
        run(forecast, prices, market, replace(spec, stage_counts=prefix), config)


@pytest.mark.parametrize("overrides", [dict(rebalance="daily"), dict(rebalance="horizon"),
    dict(shrinkage=.2), dict(max_iter=100), dict(risk_model="diagonal"),
    dict(alpha="grinold", alpha_ic=.03), dict(neutralize="beta"),
    dict(cost_model="impact", equity=1000.), dict(gross_exposure=.5)])
def test_legacy_model_overrides_are_not_silently_ignored(overrides):
    forecast, prices, market, spec, config = inputs()
    with pytest.raises(ValueError):
        run(forecast, prices, market, spec, config.replace(**overrides))


def test_ambiguous_forward_returns_and_misaligned_risk_are_rejected():
    forecast, prices, market, spec, config = inputs()
    ordinary = ForwardReturns(forecast.values, forecast.decisions, forecast.assets, forecast.horizons)
    with pytest.raises(TypeError, match="JointReturnPanel"):
        run(ordinary, prices, market, spec, config)
    risk = JointRiskModel(spec.risk.covariance, forecast.assets[::-1], forecast.horizons, -1,
                          entry_delay_seconds=forecast.entry_delay_seconds)
    with pytest.raises(ValueError, match="assets"):
        run(forecast, prices, market, replace(spec, risk=risk), config)
    with pytest.raises(TypeError, match="config"):
        run_return_risk_backtest(forecast, prices, objective=spec, config=None)


def test_nonzero_initial_pending_and_wrong_state_axes_are_explicit_errors():
    with pytest.raises(ValueError, match="arrival records"):
        BacktestExecutionState(pending_weights=[.01, -.01])
    forecast, prices, market, spec, config = inputs()
    with pytest.raises(ValueError, match="asset axis"):
        run(forecast, prices, market, spec, config,
            execution_state=BacktestExecutionState(weights=[0., 0., 0.]))


@pytest.mark.parametrize("index", [-1, 3, True, .5])
def test_native_policy_cannot_select_future_or_negative_objective_rows(index):
    forecast, prices, market, spec, config = inputs()
    policy = run(forecast, prices, market, spec, config).objective_trace.ledger.trade_policy
    with pytest.raises(ValueError, match="decision_index"):
        policy.plan([0., 0.], [0., 0.], [0., 0.], index, len(forecast.decisions))


def test_flat_objective_metadata_contains_python_json_scalars():
    import json

    forecast, prices, market, spec, config = inputs()
    spec = replace(spec, beta=[1., 1.], max_abs_beta=np.int64(0), beta_available_at=-1)
    trace = run(forecast, prices, market, spec, config).objective_trace
    assert trace.report_metadata["objective_max_abs_beta"] == 0.
    assert isinstance(trace.report_metadata["objective_max_abs_beta"], float)
    assert json.loads(json.dumps(trace.report_metadata))["input_mode"] == "return_risk_physical"


def test_frozen_current_constraints_are_infeasible_without_hidden_relaxation():
    forecast, prices, _, spec, config = inputs()
    spread = np.full(prices.values.shape, np.nan)
    with pytest.raises(RuntimeError, match="infeasible"):
        run_return_risk_backtest(forecast, prices, objective=spec,
            config=config.replace(cost_model="cross"), half_spread=spread,
            execution_state=BacktestExecutionState(weights=[.2, -.2]))
