"""Native factor constraints preserve payoff, stage prefixes and physical state."""

from dataclasses import replace

import numpy as np
import pytest

from stable_finance import DEFAULT_BACKTEST
from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.backtest import run_backtest
from stable_finance.evaluation.physical import PricePanel
from stable_finance.evaluation.portfolio.factors import FactorExposureConstraints
from stable_finance.evaluation.portfolio.joint_risk import JointReturnPanel, JointRiskModel
from stable_finance.evaluation.return_risk_backtest import (
    BacktestExecutionState, ReturnRiskBacktestSpec, run_return_risk_backtest,
)


def _inputs(*, joint=False, delay=2):
    assets, decisions = np.array(["A", "B"]), np.array([0, 10, 20])
    horizons = np.array([10, 20] if joint else [10])
    values = np.empty((3, 2, len(horizons)))
    values[:, :, 0] = [.02, -.02]
    covariance = .1*np.eye(2)
    if joint:
        values[:, :, 1] = [.03, -.03]
        covariance = np.block([[.1*np.eye(2), .04*np.eye(2)],
                               [.04*np.eye(2), .2*np.eye(2)]])
    forecast = JointReturnPanel(values, decisions, assets, horizons, entry_delay_seconds=delay)
    risk = JointRiskModel(covariance, assets, horizons, -1, entry_delay_seconds=delay)
    prices = PricePanel(np.full((31+delay, 2), 100.), np.arange(31+delay), assets)
    market = MarketData({"mid": prices.values, "bid_price": prices.values-.01,
                         "ask_price": prices.values+.01}, prices.times, assets)
    spec = ReturnRiskBacktestSpec(risk, strategy_end=30)
    config = DEFAULT_BACKTEST.replace(cost_model="mid", annualization="decisions")
    return forecast, prices, market, spec, config


def _run(forecast, prices, market, spec, config, **kwargs):
    return run_backtest(forecast, prices, market=market, objective=spec,
                        config=config, **kwargs)


def _static(forecast, lower=-.02, upper=.02, **kwargs):
    return FactorExposureConstraints([[1.], [0.]], lower, upper,
        assets=forecast.assets, factor_names=["market_proxy"], **kwargs)


def _assert_active_bounds(trace):
    resolved = trace.factor_constraints
    assert resolved is not None
    for row, prefix in enumerate(trace.stage_counts):
        exposure = np.einsum("kn,knf->kf", trace.reference_plans[row, :prefix],
                             resolved.loadings[row, :prefix])
        np.testing.assert_allclose(trace.certificates[row].factor_exposures,
                                   exposure, atol=1e-10)
        assert np.all(exposure >= resolved.lower[row, :prefix]-1e-8)
        assert np.all(exposure <= resolved.upper[row, :prefix]+1e-8)


def test_single_interval_constraints_reach_frictionless_targets_and_actual_trade_solve():
    forecast, prices, market, spec, config = _inputs()
    factors = _static(forecast, lower=.01, upper=.02, available_at=-1)
    state = BacktestExecutionState(weights=[.03, -.03])
    report = _run(forecast, prices, market, replace(spec, factor_constraints=factors),
                   config, execution_state=state, with_series=True)
    trace = report.objective_trace
    np.testing.assert_allclose(trace.targets.values[:, :, 0], [[.02, -.02]]*3, atol=1e-10)
    np.testing.assert_allclose(trace.ledger.planned_weights, [[.02, -.02]]*3, atol=1e-10)
    np.testing.assert_allclose(trace.ledger.planned_trade_weights[0], [-.01, .01], atol=1e-10)
    np.testing.assert_allclose(trace.ledger.actual_decision_weights[0], [.03, -.03], atol=1e-12)
    _assert_active_bounds(trace)
    assert trace.factor_constraints.loadings.shape == (3, 1, 2, 1)
    assert tuple(trace.factor_constraints.factor_names) == ("market_proxy",)
    assert not trace.factor_constraints.loadings.flags.writeable
    assert np.prod(1+report.series.net)-1 == pytest.approx(trace.ledger.total_return, abs=1e-12)


def test_joint_prefix_slices_factor_loadings_and_budgets_before_each_solve():
    forecast, prices, market, spec, config = _inputs(joint=True)
    loadings = np.broadcast_to(np.array([[[1.], [0.]], [[2.], [0.]]]), (3, 2, 2, 1)).copy()
    lower = np.array([[[-.01], [.025]], [[-.008], [.5]], [[-.006], [.5]]])
    upper = np.array([[[.01], [.06]], [[.008], [.6]], [[.006], [.6]]])
    factors = FactorExposureConstraints(loadings, lower, upper,
        assets=forecast.assets, factor_names=["stage_factor"],
        decisions=forecast.decisions, horizons=forecast.horizons,
        available_at=np.broadcast_to(forecast.decisions[:, None], (3, 2)))
    spec = replace(spec, stage_counts=[2, 1, 1], factor_constraints=factors)
    trace = _run(forecast, prices, market, spec, config).objective_trace
    # Inactive future bounds deliberately exclude every allowed future book.
    # Their presence must not constrain the declared one-stage prefixes.
    np.testing.assert_array_equal(trace.stage_counts, [2, 1, 1])
    np.testing.assert_allclose(trace.reference_plans[0],
        [[.01, -.01], [.0125, -.0125]], atol=1e-9)
    np.testing.assert_allclose(trace.reference_plans[1:, 0],
        [[.008, -.008], [.006, -.006]], atol=1e-9)
    np.testing.assert_array_equal(trace.reference_plans[1:, 1], 0.)
    assert [certificate.factor_exposures.shape for certificate in trace.certificates] == [
        (2, 1), (1, 1), (1, 1)]
    _assert_active_bounds(trace)


def test_joint_generic_factors_retain_full_cross_interval_risk_and_original_payoffs():
    forecast, prices, market, spec, config = _inputs(joint=True)
    spec = replace(spec, factor_constraints=_static(forecast, -.09, .09))
    full = _run(forecast, prices, market, spec, config, with_series=True)
    diagonal = spec.risk.covariance.copy()
    diagonal[:2, 2:] = diagonal[2:, :2] = 0.
    risk = JointRiskModel(diagonal, forecast.assets, forecast.horizons, -1,
                          entry_delay_seconds=forecast.entry_delay_seconds)
    separate = _run(forecast, prices, market, replace(spec, risk=risk), config)
    expected = np.linalg.solve(np.array([[.1, .04], [.04, .2]]), [.02/5., .01/5.])
    trace = full.objective_trace
    np.testing.assert_allclose(trace.reference_plans[0],
        np.column_stack([expected, -expected]), atol=1e-10)
    assert abs(trace.reference_plans[0, 0, 0]-
               separate.objective_trace.reference_plans[0, 0, 0]) > 1e-4
    plan, certificate = trace.reference_plans[0].ravel(), trace.certificates[0]
    assert certificate.expected_payoff == pytest.approx(np.array([.02, -.02, .01, -.01]) @ plan)
    assert certificate.variance == pytest.approx(plan @ spec.risk.covariance @ plan)
    assert certificate.risk_penalty == pytest.approx(2.5*certificate.variance)
    np.testing.assert_array_equal(full.series.gross, 0.)
    _assert_active_bounds(trace)


@pytest.mark.parametrize("joint", [False, True])
def test_native_legacy_beta_and_generic_factor_constraints_both_apply(joint):
    forecast, prices, market, spec, config = _inputs(joint=joint)
    spec = replace(spec, beta=[1., 0.], max_abs_beta=.015,
        factor_constraints=_static(forecast, lower=.01, upper=.02))
    trace = _run(forecast, prices, market, spec, config).objective_trace
    np.testing.assert_allclose(trace.ledger.planned_weights, [[.015, -.015]]*3, atol=1e-9)
    for certificate in trace.certificates:
        assert np.max(np.abs(certificate.beta_exposures)) <= .015+1e-8
        assert np.min(certificate.factor_exposures) >= .01-1e-8
    _assert_active_bounds(trace)


def test_later_factor_changes_cannot_change_earlier_native_plans():
    forecast, prices, market, spec, config = _inputs(joint=True)
    loadings = np.broadcast_to(np.array([[1.], [0.]]), (3, 2, 1)).copy()
    factors = FactorExposureConstraints(loadings, -.02, .02,
        assets=forecast.assets, factor_names=["time_factor"],
        decisions=forecast.decisions, available_at=forecast.decisions)
    original = _run(forecast, prices, market,
        replace(spec, factor_constraints=factors), config).objective_trace
    changed_loadings = loadings.copy()
    changed_loadings[2, 0, 0] = 4.
    changed = replace(factors, loadings=changed_loadings)
    later = _run(forecast, prices, market,
        replace(spec, factor_constraints=changed), config).objective_trace
    np.testing.assert_array_equal(original.reference_plans[:2], later.reference_plans[:2])
    assert abs(original.reference_plans[2, 0, 0]-later.reference_plans[2, 0, 0]) > .01
    _assert_active_bounds(later)


def test_native_factor_availability_is_checked_before_any_execution():
    forecast, prices, market, spec, config = _inputs(joint=True)
    factors = FactorExposureConstraints([[1.], [0.]], -.02, .02,
        assets=forecast.assets, factor_names=["late_factor"],
        decisions=forecast.decisions, available_at=[0, 10, 21])
    with pytest.raises(ValueError, match="available_at|later|availability"):
        _run(forecast, prices, market, replace(spec, factor_constraints=factors), config)


@pytest.mark.parametrize("mismatch", ["assets", "decisions", "horizons"])
def test_native_factor_contract_requires_exact_declared_axes(mismatch):
    forecast, prices, market, spec, config = _inputs(joint=True)
    kwargs = dict(assets=forecast.assets, factor_names=["axis_factor"],
                  decisions=forecast.decisions, horizons=forecast.horizons)
    if mismatch == "assets":
        kwargs["assets"] = forecast.assets[::-1]
    elif mismatch == "decisions":
        kwargs["decisions"] = [0, 10, 21]
    else:
        kwargs["horizons"] = [10, 21]
    factors = FactorExposureConstraints([[1.], [0.]], -.02, .02, **kwargs)
    with pytest.raises(ValueError, match=mismatch+"|asset order"):
        _run(forecast, prices, market, replace(spec, factor_constraints=factors), config)


def test_generic_constraints_reach_native_pending_state_without_charging_it_again():
    forecast, prices, market, spec, config = _inputs(delay=5)
    forecast = JointReturnPanel(forecast.values, [0, 1, 2], forecast.assets,
        forecast.horizons, entry_delay_seconds=5)
    spec = replace(spec, factor_constraints=_static(forecast))
    trace = _run(forecast, prices, market, spec,
        config.replace(cost_model="cross")).objective_trace
    assert np.abs(trace.ledger.pending_decision_weights[1]).sum() > .01
    np.testing.assert_array_equal(trace.ledger.actual_decision_weights[1], 0.)
    np.testing.assert_allclose(trace.ledger.planned_trade_weights[1], 0., atol=1e-10)
    assert trace.certificates[1].estimated_cost == pytest.approx(0., abs=1e-12)
    _assert_active_bounds(trace)


def test_generic_constraints_cannot_trade_a_frozen_actual_holding_to_invent_feasibility():
    forecast, prices, _, spec, config = _inputs()
    spread = np.full(prices.values.shape, np.nan)
    spec = replace(spec, factor_constraints=_static(forecast))
    with pytest.raises(RuntimeError, match="infeasible"):
        run_return_risk_backtest(forecast, prices, objective=spec,
            config=config.replace(cost_model="cross"), half_spread=spread,
            execution_state=BacktestExecutionState(weights=[.03, -.03]))


def test_default_no_factor_spec_preserves_plans_ledger_and_flat_metadata():
    forecast, prices, market, spec, config = _inputs(joint=True)
    omitted = _run(forecast, prices, market, spec, config)
    explicit = _run(forecast, prices, market, replace(spec, factor_constraints=None), config)
    left, right = omitted.objective_trace, explicit.objective_trace
    assert left.factor_constraints is right.factor_constraints is None
    assert left.report_metadata == right.report_metadata
    assert set(left.report_metadata) == {
        "input_mode", "objective_risk_aversion", "objective_gross_limit",
        "objective_max_weight", "objective_dollar_neutral", "objective_max_abs_beta",
        "objective_risk_estimator", "objective_risk_source", "initial_equity",
        "entry_delay_seconds", "report_period"}
    np.testing.assert_array_equal(left.reference_plans, right.reference_plans)
    np.testing.assert_array_equal(left.ledger.nav, right.ledger.nav)
    np.testing.assert_array_equal(left.ledger.trade_notional, right.ledger.trade_notional)
    assert all(result.factor_exposures is None for result in left.certificates)
