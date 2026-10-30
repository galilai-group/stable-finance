"""Current-state target tracking preserves costs and explicit position limits."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stable_finance.evaluation.trades.constrained import (
    RebalanceConstraints, solve_constrained_rebalance,
)


LOOSE = RebalanceConstraints(gross_limit=2., max_weight=1., dollar_neutral=False)
NEUTRAL = RebalanceConstraints(gross_limit=2., max_weight=1., dollar_neutral=True)


def test_two_name_solution_matches_independent_scalar_soft_threshold():
    # w=(a,-a); gamma/2*2(a-.3)^2 + 2*c*abs(a), giving a=.3-c/gamma.
    result = solve_constrained_rebalance([.3, -.3], np.eye(2),
        previous=[0., 0.], cost=.1, risk_aversion=2., constraints=NEUTRAL)
    np.testing.assert_allclose(result.weights, [.25, -.25], atol=1e-9)
    assert result.status == "solved"
    assert result.tracking_risk == pytest.approx(.005)
    assert result.estimated_cost == pytest.approx(.05)
    assert result.objective == pytest.approx(.055)
    assert result.turnover == pytest.approx(.5)
    assert result.optimality_gap < 1e-8


def test_diagonal_non_neutral_solution_matches_per_name_analytic_optimum():
    goal = np.array([.4, -.2, .15])
    previous = np.array([.1, -.05, .2])
    variance, charge = np.array([2., 1., .5]), np.array([.1, .04, .02])
    move = goal-previous
    expected = previous+np.sign(move)*np.maximum(abs(move)-charge/(3*variance), 0.)
    result = solve_constrained_rebalance(goal, np.diag(variance), previous=previous,
        cost=charge, risk_aversion=3., constraints=LOOSE)
    np.testing.assert_allclose(result.weights, expected, atol=1e-9)


def test_high_cost_retains_feasible_actual_plus_pending_exactly():
    previous, pending = np.array([.1, -.1]), np.array([.05, -.05])
    result = solve_constrained_rebalance([.3, -.3], np.eye(2), previous=previous,
        pending=pending, cost=10., constraints=NEUTRAL)
    np.testing.assert_array_equal(result.weights, previous+pending)
    assert result.turnover == result.estimated_cost == 0.
    assert result.optimality_gap == 0.


def test_pending_is_not_charged_twice_or_treated_as_filled_state():
    first = solve_constrained_rebalance([.3, -.3], np.eye(2), previous=[.1, -.1],
        pending=[.1, -.1], cost=.1, risk_aversion=2., constraints=NEUTRAL)
    second = solve_constrained_rebalance([.3, -.3], np.eye(2), previous=[.2, -.2],
        cost=.1, risk_aversion=2., constraints=NEUTRAL)
    np.testing.assert_allclose(first.weights, second.weights, atol=1e-10)
    assert first.turnover == pytest.approx(.1)
    assert first.estimated_cost == pytest.approx(.01)


def test_beta_is_binding_inside_tracking_solve():
    limits = RebalanceConstraints(2., 1., True, .1)
    result = solve_constrained_rebalance([.3, -.3], np.eye(2), previous=[0., 0.],
        cost=.01, risk_aversion=2., constraints=limits, beta=[2., .5])
    expected = .1/1.5
    np.testing.assert_allclose(result.weights, [expected, -expected], atol=1e-9)
    assert result.beta_exposure == pytest.approx(.1, abs=1e-9)
    assert result.max_constraint_violation < 1e-8


def test_gross_and_name_constraints_are_not_undone_by_normalization():
    result = solve_constrained_rebalance([.8, -.8, .5, -.5], np.eye(4),
        previous=np.zeros(4), cost=.001,
        constraints=RebalanceConstraints(.3, .1, True))
    assert np.abs(result.weights).sum() <= .3+1e-8
    assert np.abs(result.weights).max() <= .1+1e-8
    assert abs(result.weights.sum()) < 1e-8


def test_zero_cost_matches_feasible_target_exactly_and_is_readonly():
    goal = np.array([.075, -.075])
    result = solve_constrained_rebalance(goal, np.eye(2), previous=[.03, -.03],
        cost=0., constraints=RebalanceConstraints(max_abs_beta=.1), beta=[1., .5])
    np.testing.assert_array_equal(result.weights, goal)
    assert result.objective == result.tracking_risk == result.estimated_cost == 0.
    assert not result.weights.flags.writeable
    goal[:] = 99
    np.testing.assert_array_equal(result.weights, [.075, -.075])


def test_correlated_risk_matches_independent_scalar_neutral_problem():
    sigma = np.array([[2., .4], [.4, 1.]])
    target = np.array([.3, -.1])
    direction = np.array([1., -1.])
    expected = (direction @ sigma @ target-.1)/(direction @ sigma @ direction)
    result = solve_constrained_rebalance(target, sigma, previous=[0., 0.],
        cost=.1, risk_aversion=2., constraints=NEUTRAL)
    np.testing.assert_allclose(result.weights, direction*expected, atol=1e-9)


@pytest.mark.parametrize("cost", [[np.nan, .01, .01], [np.inf, .01, .01]])
def test_missing_cost_freezes_actual_plus_pending_and_keeps_remaining_hedges(cost):
    result = solve_constrained_rebalance([0., -.1, .1], np.eye(3),
        previous=[.05, -.05, 0.], pending=[.02, 0., 0.], cost=cost,
        constraints=RebalanceConstraints(.5, .2, True, .1), beta=[2., 1., .5])
    assert result.weights[0] == .05+.02
    assert abs(result.weights.sum()) < 1e-8
    assert abs(result.beta_exposure) <= .1+1e-8


def test_explicit_nontradable_mask_freezes_even_with_zero_cost():
    result = solve_constrained_rebalance([0., 0.], np.eye(2), previous=[.05, -.05],
        cost=0., tradable=[False, True], constraints=NEUTRAL)
    np.testing.assert_allclose(result.weights, [.05, -.05], atol=1e-10)
    assert result.weights[0] == .05


@pytest.mark.parametrize("previous,limits,beta", [
    ([.15, -.15], RebalanceConstraints(), None),
    ([.08, -.08], RebalanceConstraints(.1, .1, True), None),
    ([.08, -.08], RebalanceConstraints(max_abs_beta=.01), [2., .5]),
    ([.08, -.04], RebalanceConstraints(), None),
])
def test_frozen_infeasibility_is_explicit_and_has_no_plan(previous, limits, beta):
    result = solve_constrained_rebalance([0., 0.], np.eye(2), previous=previous,
        cost=np.nan, constraints=limits, beta=beta)
    assert result.status == "infeasible"
    assert result.weights is result.objective is result.optimality_gap is None
    assert "Frozen" in result.message


def test_partial_freeze_can_make_hedging_infeasible_even_with_feasible_frozen_name():
    # Fixed +.08 requires -.08 in the only free name, violating beta .01.
    result = solve_constrained_rebalance([0., 0.], np.eye(2), previous=[.08, -.08],
        cost=.01, tradable=[False, True],
        constraints=RebalanceConstraints(max_abs_beta=.01), beta=[2., .5])
    assert result.status == "infeasible"
    assert result.weights is None


def test_high_cost_cannot_keep_an_infeasible_drifted_book():
    result = solve_constrained_rebalance([.05, -.05], np.eye(2),
        previous=[.11, -.11], cost=100., constraints=RebalanceConstraints())
    np.testing.assert_allclose(result.weights, [.1, -.1], atol=1e-8)
    assert result.turnover == pytest.approx(.02, abs=1e-8)


def test_singular_psd_risk_retains_exact_zero_cost_target():
    goal = np.array([.08, -.08])
    result = solve_constrained_rebalance(goal, np.ones((2, 2)), previous=[0., 0.])
    np.testing.assert_array_equal(result.weights, goal)


def test_zero_risk_with_cost_chooses_feasible_committed_book():
    result = solve_constrained_rebalance([.1, -.1], np.zeros((2, 2)),
        previous=[.05, -.05], cost=.1)
    np.testing.assert_array_equal(result.weights, [.05, -.05])
    assert result.objective == result.turnover == 0.


def test_zero_risk_and_infeasible_committed_book_is_a_valid_linear_solve():
    result = solve_constrained_rebalance([0., 0.], np.zeros((2, 2)),
        previous=[.15, -.15], cost=.1)
    np.testing.assert_allclose(result.weights, [.1, -.1], atol=1e-10)
    assert result.estimated_cost == pytest.approx(.01)


@pytest.mark.parametrize("kwargs", [
    {"target": []}, {"target": [True, False]}, {"target": [np.nan, 0.]},
    {"previous": [np.inf, 0.]}, {"pending": [0.]},
    {"cost": [-.1, 0.]}, {"cost": -np.inf}, {"cost": [[.1, .1]]},
    {"cost": True}, {"risk_aversion": 0}, {"risk_aversion": True},
    {"covariance": [[1., .2], [0., 1.]]}, {"covariance": [[1., 0.], [0., -.1]]},
    {"covariance": np.ones((3, 3))}, {"covariance": [[np.nan, 0.], [0., 1.]]},
    {"tradable": [1, 0]}, {"tradable": [True]},
    {"beta": [1., .5]}, {"constraints": RebalanceConstraints(max_abs_beta=.1)},
    {"constraints": RebalanceConstraints(max_abs_beta=.1), "beta": [np.nan, .5]},
])
def test_invalid_inputs_rejected(kwargs):
    base = dict(target=[.05, -.05], covariance=np.eye(2), previous=[0., 0.])
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        solve_constrained_rebalance(**base)


@pytest.mark.parametrize("kwargs", [
    {"gross_limit": -1}, {"max_weight": np.nan}, {"max_abs_beta": -.1},
    {"max_weight": True}, {"dollar_neutral": 1},
])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        RebalanceConstraints(**kwargs)


def test_exact_zero_cost_target_needs_no_optional_optimizer(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "osqp" or name.startswith("scipy"):
            raise ImportError("deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = solve_constrained_rebalance([.05, -.05], np.eye(2), previous=[0., 0.])
    np.testing.assert_array_equal(result.weights, [.05, -.05])


def test_module_is_directly_loadable_without_other_package_modules():
    path = Path(__file__).resolve().parents[3]/"src/stable_finance/evaluation/trades/constrained.py"
    spec = importlib.util.spec_from_file_location("isolated_constrained_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = module.solve_constrained_rebalance([.05, -.05], np.eye(2), previous=[0., 0.])
        np.testing.assert_array_equal(result.weights, [.05, -.05])
    finally:
        sys.modules.pop(spec.name, None)


def test_solver_success_string_cannot_bypass_constraint_certificate(monkeypatch):
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
        solve_constrained_rebalance([.3, -.3], np.eye(2), previous=[0., 0.],
            cost=.01, constraints=RebalanceConstraints())


def test_nonoptimal_feasible_solver_point_cannot_pass_gap_certificate(monkeypatch):
    class FakeSolver:
        def setup(self, **kwargs):
            pass
        def warm_start(self, **kwargs):
            pass
        def solve(self, **kwargs):
            return SimpleNamespace(x=np.zeros(6), info=SimpleNamespace(status="solved"))

    monkeypatch.setattr("osqp.OSQP", FakeSolver)
    with pytest.raises(RuntimeError, match="optimality gap"):
        solve_constrained_rebalance([.3, -.3], np.eye(2), previous=[0., 0.],
            cost=.01, constraints=NEUTRAL)


def test_asset_permutation_keeps_physical_problem_unchanged():
    goal, previous = np.array([.2, -.1, -.1]), np.array([.05, -.02, -.03])
    sigma = np.array([[2., .3, .1], [.3, 1., .2], [.1, .2, .8]])
    charge, beta = np.array([.02, .03, .01]), np.array([2., 1., .5])
    limits = RebalanceConstraints(1., .4, True, .1)
    first = solve_constrained_rebalance(goal, sigma, previous=previous,
        cost=charge, beta=beta, constraints=limits)
    order = np.array([2, 0, 1])
    second = solve_constrained_rebalance(goal[order], sigma[np.ix_(order, order)],
        previous=previous[order], cost=charge[order], beta=beta[order], constraints=limits)
    np.testing.assert_allclose(second.weights, first.weights[order], atol=1e-9)
    assert second.objective == pytest.approx(first.objective)
