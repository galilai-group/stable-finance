"""Original alpha and full joint risk survive the actual-state trading handoff."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stable_finance.evaluation.trades.objective import solve_return_risk_rebalance


LOOSE = dict(gross_limit=2., max_weight=1.)


def test_single_interval_matches_independent_scalar_original_alpha_solution():
    # w=(a,-a): 2*mu*a - gamma*a^2 - 2*c*abs(a).
    result = solve_return_risk_rebalance([.3, -.3], np.eye(2),
        previous=[0., 0.], cost=.1, risk_aversion=2., **LOOSE)
    np.testing.assert_allclose(result.weights, [.1, -.1], atol=1e-9)
    assert result.objective == pytest.approx(.02)
    assert result.expected_payoff == pytest.approx(.06)
    assert result.risk_penalty == pytest.approx(.02)
    assert result.estimated_cost == pytest.approx(.02)
    assert result.turnover == pytest.approx(.2)
    assert result.optimality_gap < 1e-10
    assert result.max_constraint_violation < 1e-8


def test_capped_target_does_not_replace_original_forecast_magnitude():
    from stable_finance.evaluation.trades.constrained import solve_constrained_rebalance

    sigma, original_mu = 1e-5*np.eye(2), np.array([1e-4, -1e-4])
    # The original no-cost optimum is capped at +/-0.1. Inferring alpha
    # from that capped book shrinks it to +/-5e-6, below the same spread.
    capped = np.array([.1, -.1])
    tracking = solve_constrained_rebalance(capped, sigma, previous=[0., 0.], cost=2e-5)
    corrected = solve_return_risk_rebalance(original_mu, sigma,
        previous=[0., 0.], cost=2e-5)
    np.testing.assert_array_equal(tracking.weights, [0., 0.])
    np.testing.assert_allclose(corrected.weights, capped, atol=1e-9)
    assert corrected.objective > 1e-5


def test_actual_plus_pending_is_committed_and_not_charged_as_new_order():
    previous, pending = np.array([.15, -.15]), np.array([.05, -.05])
    committed = previous+pending
    result = solve_return_risk_rebalance(2*committed, np.eye(2),
        previous=previous, pending=pending, cost=.1, risk_aversion=2., **LOOSE)
    np.testing.assert_array_equal(result.weights, committed)
    assert result.estimated_cost == result.turnover == 0.


def test_future_reference_positions_are_reoptimized_with_cross_interval_risk():
    eye, rho = np.eye(2), .5
    omega = np.block([[eye, rho*eye], [rho*eye, eye]])
    q = np.array([[0., 0.], [.1, -.1]])
    result = solve_return_risk_rebalance(q, omega, previous=[0., 0.],
        cost=.001, risk_aversion=1., **LOOSE)
    a = (-rho*.1+.001)/(1-rho**2)
    b = .1-rho*a
    np.testing.assert_allclose(result.reference_plan,
                               [[a, -a], [b, -b]], atol=1e-8)
    independent = solve_return_risk_rebalance(q, np.eye(4), previous=[0., 0.],
        cost=.001, risk_aversion=1., **LOOSE)
    np.testing.assert_array_equal(independent.weights, [0., 0.])
    np.testing.assert_allclose(independent.reference_plan[1], [.1, -.1], atol=1e-8)


def test_high_current_cost_allows_current_cash_and_positive_future_reference_utility():
    result = solve_return_risk_rebalance([[0., 0.], [.1, -.1]], np.eye(4),
        previous=[0., 0.], cost=100., risk_aversion=1., **LOOSE)
    np.testing.assert_array_equal(result.weights, [0., 0.])
    np.testing.assert_allclose(result.reference_plan[1], [.1, -.1], atol=1e-8)
    assert result.objective == pytest.approx(.01)
    assert result.estimated_cost == result.turnover == 0.


def test_zero_cost_agrees_with_original_native_joint_objective_and_solution():
    from stable_finance.evaluation.portfolio.joint import JointPortfolioConfig, _solve_plan

    q = np.array([[.04, -.02, .01], [.03, -.01, -.02]])
    omega = np.block([[.2*np.eye(3), .04*np.eye(3)],
                      [.04*np.eye(3), .3*np.eye(3)]])
    beta = np.array([2., 1., .5])
    native, native_gap = _solve_plan(q.ravel(), omega, 2, 3,
        np.ones((2, 3), dtype=bool), JointPortfolioConfig(),
        np.broadcast_to(beta, (2, 3)), .1)
    result = solve_return_risk_rebalance(q, omega, previous=[.02, -.01, -.01],
        cost=0., beta=beta, max_abs_beta=.1)
    native_utility = q.ravel() @ native-2.5*native @ omega @ native
    assert result.objective == pytest.approx(native_utility, abs=1e-10)
    np.testing.assert_allclose(result.reference_plan.ravel(), native, atol=1e-7)
    assert result.optimality_gap+native_gap < 1e-9


def test_full_recentered_linear_residual_is_equivalent_to_original_objective():
    q = np.array([[.3, -.3], [.12, -.12]])
    omega = np.array([[2., 0., .4, 0.], [0., 2., 0., .4],
                      [.4, 0., 1., 0.], [0., .4, 0., 1.]])
    anchor = np.array([[.1, -.1], [.02, -.02]])
    gamma, cost, initial = 3., np.array([.01, .02]), np.array([.03, -.03])
    result = solve_return_risk_rebalance(q, omega, previous=initial, cost=cost,
        risk_aversion=gamma, **LOOSE)
    x, t = result.reference_plan.ravel(), anchor.ravel()
    g = gamma*omega @ t-q.ravel()
    direct_anchor = q.ravel() @ t-gamma/2*t @ omega @ t-cost @ np.abs(anchor[0]-initial)
    delta = x-t
    change = (-gamma/2*delta @ omega @ delta-g @ delta
              -cost @ (np.abs(result.weights-initial)-np.abs(anchor[0]-initial)))
    assert result.objective-direct_anchor == pytest.approx(change, abs=1e-14)
    without_gradient = -gamma/2*delta @ omega @ delta-cost @ (
        np.abs(result.weights-initial)-np.abs(anchor[0]-initial))
    assert abs(change-without_gradient) > 1e-4


def test_stage_specific_beta_limits_are_inside_full_solve():
    q = [[.4, -.4], [.3, -.3]]
    beta = [[2., .5], [3., 1.]]
    result = solve_return_risk_rebalance(q, np.eye(4), previous=[0., 0.],
        cost=.001, risk_aversion=1., beta=beta, max_abs_beta=.1, **LOOSE)
    np.testing.assert_allclose(result.reference_plan[0], [.1/1.5, -.1/1.5], atol=1e-8)
    np.testing.assert_allclose(result.reference_plan[1], [.05, -.05], atol=1e-8)
    np.testing.assert_allclose(result.beta_exposures, [.1, .1], atol=1e-8)


@pytest.mark.parametrize("cost", [np.nan, np.inf])
def test_missing_current_cost_does_not_freeze_future_reference_positions(cost):
    result = solve_return_risk_rebalance([[.2, -.2], [.1, -.1]], np.eye(4),
        previous=[.03, -.03], cost=cost, risk_aversion=1., **LOOSE)
    np.testing.assert_array_equal(result.weights, [.03, -.03])
    np.testing.assert_allclose(result.reference_plan[1], [.1, -.1], atol=1e-8)
    assert result.estimated_cost == result.turnover == 0.


def test_partial_freeze_retains_covariance_cross_terms_and_exact_committed_position():
    sigma = np.array([[2., .7, .1], [.7, 1., .2], [.1, .2, .8]])
    result = solve_return_risk_rebalance([.1, -.02, .04], sigma,
        previous=[.02, -.01, -.01], pending=[.01, 0., 0.],
        cost=.001, tradable=[False, True, True], risk_aversion=1., **LOOSE)
    assert result.weights[0] == .03
    assert abs(result.weights.sum()) < 1e-9
    assert result.objective == pytest.approx(
        np.array([.1, -.02, .04]) @ result.weights
        -.5*result.weights @ sigma @ result.weights
        -.001*np.abs(result.weights-np.array([.03, -.01, -.01])).sum(), abs=1e-12)


@pytest.mark.parametrize("kwargs", [
    dict(previous=[.2, -.2], tradable=[False, False]),
    dict(previous=[.08, -.04], tradable=[False, False]),
    dict(previous=[.08, -.08], tradable=[False, True], beta=[2., .5], max_abs_beta=.01),
])
def test_frozen_current_infeasibility_has_no_fabricated_reference_plan(kwargs):
    result = solve_return_risk_rebalance([[.1, -.1], [.1, -.1]], np.eye(4),
        cost=.001, **kwargs)
    assert result.status == "infeasible"
    assert result.weights is result.reference_plan is result.objective is None


def test_high_cost_does_not_preserve_an_infeasible_drifted_current_book():
    result = solve_return_risk_rebalance([.1, -.1], np.eye(2),
        previous=[.11, -.11], cost=100.)
    np.testing.assert_allclose(result.weights, [.1, -.1], atol=1e-8)
    assert result.turnover == pytest.approx(.02, abs=1e-8)


def test_singular_psd_and_zero_risk_linear_objectives_are_supported():
    singular = solve_return_risk_rebalance([.1, -.1], np.ones((2, 2)),
        previous=[0., 0.], cost=.001)
    np.testing.assert_allclose(singular.weights, [.1, -.1], atol=1e-8)
    linear = solve_return_risk_rebalance([.1, -.1], np.zeros((2, 2)),
        previous=[0., 0.], cost=.001)
    np.testing.assert_allclose(linear.weights, [.1, -.1], atol=1e-8)


def test_common_forecast_shift_does_not_change_dollar_neutral_solution():
    first = solve_return_risk_rebalance([[.1, -.1], [.05, -.05]], np.eye(4),
        previous=[0., 0.], cost=.001, risk_aversion=1., **LOOSE)
    shifted = solve_return_risk_rebalance([[100.1, 99.9], [200.05, 199.95]], np.eye(4),
        previous=[0., 0.], cost=.001, risk_aversion=1., **LOOSE)
    np.testing.assert_allclose(first.reference_plan, shifted.reference_plan, atol=1e-9)
    assert first.objective == pytest.approx(shifted.objective, abs=1e-11)


def test_non_neutral_single_interval_matches_per_asset_soft_threshold():
    q = np.array([.2, -.1, .08])
    risk = np.diag([2., 1., .5])
    charge = np.array([.01, .02, .005])
    expected = np.sign(q)*np.maximum(np.abs(q)-charge, 0.)/(3*np.diag(risk))
    result = solve_return_risk_rebalance(q, risk, previous=np.zeros(3), cost=charge,
        risk_aversion=3., dollar_neutral=False, **LOOSE)
    np.testing.assert_allclose(result.weights, expected, atol=1e-9)


def test_output_is_independent_readonly_copy_of_forecast_and_state():
    forecast, actual = np.array([.1, -.1]), np.zeros(2)
    result = solve_return_risk_rebalance(forecast, np.eye(2), previous=actual,
        cost=.001, beta=[1., 1.], max_abs_beta=.1)
    stored = result.reference_plan.copy()
    forecast[:] = 9
    actual[:] = 9
    np.testing.assert_array_equal(result.reference_plan, stored)
    assert not result.weights.flags.writeable
    assert not result.reference_plan.flags.writeable
    assert not result.beta_exposures.flags.writeable


@pytest.mark.parametrize("kwargs", [
    dict(incremental_forecasts=[]), dict(incremental_forecasts=[True, False]),
    dict(incremental_forecasts=[np.nan, 0.]), dict(covariance=np.eye(3)),
    dict(covariance=[[1., .2], [0., 1.]]), dict(covariance=[[1., 0.], [0., -.1]]),
    dict(previous=[np.nan, 0.]), dict(pending=[0.]), dict(cost=-np.inf),
    dict(cost=True), dict(cost=[[.1, .1]]), dict(risk_aversion=0),
    dict(gross_limit=-1), dict(max_weight=True), dict(dollar_neutral=1),
    dict(beta=[1., 1.]), dict(max_abs_beta=.1),
    dict(beta=[np.nan, 1.], max_abs_beta=.1),
    dict(beta=[[1., 1.], [1., 1.]], max_abs_beta=.1),
    dict(tradable=[1, 0]), dict(tradable=[True]),
])
def test_invalid_input_cannot_change_objective_or_constraints_silently(kwargs):
    inputs = dict(incremental_forecasts=[.1, -.1], covariance=np.eye(2), previous=[0., 0.])
    inputs.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        solve_return_risk_rebalance(**inputs)


def test_module_can_be_loaded_in_an_isolated_execution_namespace():
    path = Path(__file__).resolve().parents[3]/"src/stable_finance/evaluation/trades/objective.py"
    spec = importlib.util.spec_from_file_location("isolated_original_objective_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = module.solve_return_risk_rebalance([.1, -.1], np.eye(2), previous=[0., 0.])
        assert result.status == "solved"
    finally:
        sys.modules.pop(spec.name, None)


def test_backend_success_string_cannot_override_independent_primal_certificate(monkeypatch):
    class FakeSolver:
        def setup(self, **kwargs):
            pass
        def warm_start(self, **kwargs):
            pass
        def solve(self, **kwargs):
            return SimpleNamespace(x=np.r_[[.2, -.2], [.2, .2], [.2, .2]],
                                   info=SimpleNamespace(status="solved"))
    monkeypatch.setattr("osqp.OSQP", FakeSolver)
    with pytest.raises(RuntimeError, match="violates constraints"):
        solve_return_risk_rebalance([.3, -.3], np.eye(2), previous=[0., 0.], cost=.01)


def test_nonoptimal_feasible_backend_point_cannot_pass_gap_certificate(monkeypatch):
    class FakeSolver:
        def setup(self, **kwargs):
            pass
        def warm_start(self, **kwargs):
            pass
        def solve(self, **kwargs):
            return SimpleNamespace(x=np.zeros(6), info=SimpleNamespace(status="solved"))
    monkeypatch.setattr("osqp.OSQP", FakeSolver)
    with pytest.raises(RuntimeError, match="optimality gap"):
        solve_return_risk_rebalance([.3, -.3], np.eye(2), previous=[0., 0.], cost=.01)


def test_backend_error_uses_same_objective_fallback_without_relaxing_limits(monkeypatch):
    class FailedSolver:
        def setup(self, **kwargs):
            raise ValueError("primary unavailable")
    monkeypatch.setattr("osqp.OSQP", FailedSolver)
    result = solve_return_risk_rebalance([.3, -.3], np.eye(2), previous=[0., 0.], cost=.01)
    np.testing.assert_allclose(result.weights, [.058, -.058], atol=1e-8)
    assert result.status == "solved" and result.max_constraint_violation < 1e-8
