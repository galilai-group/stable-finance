"""Generic factors survive original-objective and tracking trade decisions."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stable_finance.evaluation.trades import constrained, objective


def _solve(kind, *, signal=(.4, -.4), previous=(0., 0.), pending=None,
           covariance=None, cost=.01, beta=None, max_abs_beta=None,
           gross_limit=2., max_weight=1., dollar_neutral=True, **kwargs):
    sigma = np.eye(len(previous)) if covariance is None else covariance
    common = dict(previous=previous, pending=pending, cost=cost,
                  risk_aversion=1., beta=beta, **kwargs)
    if kind == "original":
        return objective.solve_return_risk_rebalance(signal, sigma,
            gross_limit=gross_limit, max_weight=max_weight,
            dollar_neutral=dollar_neutral, max_abs_beta=max_abs_beta, **common)
    return constrained.solve_constrained_rebalance(signal, sigma,
        constraints=constrained.RebalanceConstraints(gross_limit=gross_limit,
            max_weight=max_weight, dollar_neutral=dollar_neutral,
            max_abs_beta=max_abs_beta), **common)


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_generic_beta_equivalent_bound_is_solved_inside_original_problem(kind):
    legacy = _solve(kind, beta=[1., 0.], max_abs_beta=.08)
    generic = _solve(kind, factor_loadings=[[1.], [0.]],
        factor_lower=-.08, factor_upper=.08, factor_names=("market",))
    np.testing.assert_allclose(generic.weights, [.08, -.08], atol=1e-9)
    np.testing.assert_allclose(generic.weights, legacy.weights, atol=1e-9)
    assert generic.objective == pytest.approx(legacy.objective, abs=1e-10)
    assert generic.factor_names == ("market",)
    assert generic.max_constraint_violation < 1e-8
    assert generic.optimality_gap < 1e-8


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_multiple_asymmetric_factors_enforce_positive_lower_and_equality(kind):
    loadings = np.array([[1., 0.], [0., 1.]])
    result = _solve(kind, dollar_neutral=False, factor_loadings=loadings,
        factor_lower=[.12, -.05], factor_upper=[.2, -.05],
        factor_names=("value", "industry"))
    np.testing.assert_allclose(result.weights, [.2, -.05], atol=1e-9)
    exposure = result.factor_exposures.reshape(-1)
    np.testing.assert_allclose(exposure, loadings.T @ result.weights, atol=1e-12)
    np.testing.assert_allclose(result.factor_lower_slack.reshape(-1), [.08, 0.], atol=1e-9)
    np.testing.assert_allclose(result.factor_upper_slack.reshape(-1), [0., 0.], atol=1e-9)
    for array in (result.factor_exposures, result.factor_lower_slack, result.factor_upper_slack):
        assert not array.flags.writeable


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_beta_and_generic_factor_limits_intersect_even_when_name_is_beta(kind):
    result = _solve(kind, beta=[1., 0.], max_abs_beta=.1,
        factor_loadings=[[1.], [0.]], factor_lower=.05,
        factor_upper=.2, factor_names=("beta",))
    np.testing.assert_allclose(result.weights, [.1, -.1], atol=1e-9)
    impossible = _solve(kind, beta=[1., 0.], max_abs_beta=.1,
        factor_loadings=[[1.], [0.]], factor_lower=.15, factor_upper=.2)
    assert impossible.status == "infeasible" and impossible.weights is None


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_frozen_actual_plus_pending_exposure_shifts_free_asset_budget(kind):
    # The first committed holding is .25, so the second must be -.15
    # to satisfy the factor equality .25+w[1]=.10.
    result = _solve(kind, signal=(.4, -.4, 0.), previous=(.2, 0., 0.),
        pending=(.05, 0., 0.), cost=[np.nan, .01, .01],
        factor_loadings=[[1.], [1.], [0.]], factor_lower=.1,
        factor_upper=.1)
    np.testing.assert_allclose(result.weights, [.25, -.15, -.1], atol=1e-9)
    assert result.weights[0] == .25
    assert result.turnover == pytest.approx(.25, abs=1e-9)
    assert result.estimated_cost == pytest.approx(.0025, abs=1e-10)


@pytest.mark.parametrize("kind", ["original", "tracking"])
@pytest.mark.parametrize("frozen", [False, True])
def test_nonzero_factor_floor_cannot_silently_return_cash(kind, frozen):
    result = _solve(kind, signal=(0., 0.), gross_limit=0.,
        tradable=[not frozen, not frozen], factor_loadings=[[1.], [0.]],
        factor_lower=.05, factor_upper=.1)
    assert result.status == "infeasible"
    assert result.weights is None and result.objective is None
    assert result.factor_exposures is None


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_frozen_infeasible_factor_has_no_fabricated_plan(kind):
    result = _solve(kind, previous=(.2, -.2), tradable=[False, False],
        factor_loadings=[[1.], [0.]], factor_lower=-.1, factor_upper=.1)
    assert result.status == "infeasible" and result.weights is None


def test_zero_cost_exact_tracking_target_does_not_bypass_new_factor_constraint():
    result = _solve("tracking", cost=0., factor_loadings=[[1.], [0.]],
        factor_lower=-.1, factor_upper=.1)
    np.testing.assert_allclose(result.weights, [.1, -.1], atol=1e-9)
    assert result.message != "Exact feasible zero-cost target"


def test_every_joint_stage_uses_its_own_factor_loadings_and_budgets():
    q = np.array([[.8, -.8], [.9, -.9]])
    loadings = np.array([[[1.], [0.]], [[2.], [0.]]])
    result = _solve("original", signal=q, covariance=np.eye(4), cost=.1,
        factor_loadings=loadings, factor_lower=[[.1], [-.5]],
        factor_upper=[[.2], [-.3]], factor_names=("stage_factor",))
    np.testing.assert_allclose(result.reference_plan,
        [[.2, -.2], [-.15, .15]], atol=1e-9)
    np.testing.assert_allclose(result.factor_exposures, [[.2], [-.3]], atol=1e-9)
    assert result.expected_payoff == pytest.approx(.05, abs=1e-9)
    assert result.estimated_cost == pytest.approx(.04, abs=1e-9)
    assert result.variance == pytest.approx(.125, abs=1e-9)
    assert result.objective == pytest.approx(-.0525, abs=1e-9)


def test_joint_factors_preserve_full_cross_interval_risk_and_raw_payoff():
    eye = np.eye(2)
    sigma = np.block([[eye, .5*eye], [.5*eye, eye]])
    q = np.array([[0., 0.], [.3, -.3]])
    result = _solve("original", signal=q, covariance=sigma, cost=.01,
        factor_loadings=[[1.], [0.]], factor_lower=[[-.05], [-1.]],
        factor_upper=[[.05], [1.]])
    # w_k=(x_k,-x_k): with x0 bound at -.05, x1=.3-.5*x0=.325.
    np.testing.assert_allclose(result.reference_plan,
        [[-.05, .05], [.325, -.325]], atol=1e-9)
    plan = result.reference_plan.ravel()
    assert result.expected_payoff == pytest.approx(q.ravel() @ plan)
    assert result.variance == pytest.approx(plan @ sigma @ plan)
    assert result.estimated_cost == pytest.approx(.001, abs=1e-10)


def test_joint_multiple_factors_and_frozen_exposures_keep_stage_factor_axis_order():
    loadings = np.array([[[1., 0.], [1., 1.], [0., 1.]],
                         [[2., 0.], [0., 3.], [1., 1.]]])
    expected = np.array([[.25, -.15, -.1], [.1, -.2, .1]])
    exposure = np.einsum("knf,kn->kf", loadings, expected)
    result = _solve("original", signal=[[.4, -.4, 0.], [.2, -.4, .2]],
        previous=(.2, 0., 0.), pending=(.05, 0., 0.),
        covariance=np.eye(6), cost=[np.nan, .01, .01],
        factor_loadings=loadings, factor_lower=exposure, factor_upper=exposure,
        factor_names=("style", "sector"))
    np.testing.assert_allclose(result.reference_plan, expected, atol=1e-9)
    np.testing.assert_allclose(result.factor_exposures, exposure, atol=1e-9)
    assert result.factor_names == ("style", "sector")
    assert result.estimated_cost == pytest.approx(.0025, abs=1e-10)


def test_conditional_no_current_trade_reoptimization_retains_future_factor_bounds(monkeypatch):
    implementation = objective.solve_return_risk_rebalance
    calls = []

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return implementation(*args, **kwargs)

    monkeypatch.setattr(objective, "solve_return_risk_rebalance", recording)
    result = implementation([[0., 0.], [.2, -.2]], np.eye(4),
        previous=[0., 0.], cost=5., gross_limit=2., max_weight=1.,
        risk_aversion=1., factor_loadings=[[1.], [0.]],
        factor_lower=[[0.], [.15]], factor_upper=[[0.], [.15]],
        factor_names=("future_floor",))
    np.testing.assert_array_equal(result.weights, [0., 0.])
    np.testing.assert_allclose(result.reference_plan[1], [.15, -.15], atol=1e-9)
    assert result.turnover == result.estimated_cost == 0.
    assert calls and calls[0]["factor_names"] == ("future_floor",)
    np.testing.assert_array_equal(calls[0]["factor_lower"], [[0.], [.15]])


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_one_sided_factor_limits_allow_only_outward_infinity(kind):
    result = _solve(kind, factor_loadings=[[1.], [0.]],
        factor_lower=-np.inf, factor_upper=.08)
    np.testing.assert_allclose(result.weights, [.08, -.08], atol=1e-9)
    assert np.isposinf(result.factor_lower_slack).all()
    assert np.isfinite(result.factor_upper_slack).all()


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_missing_factor_is_permitted_only_for_frozen_zero_position(kind):
    result = _solve(kind, signal=(0., -.4, .4), previous=(0., 0., 0.),
        tradable=[False, True, True], factor_loadings=[[np.nan], [1.], [0.]],
        factor_lower=-.08, factor_upper=.08)
    np.testing.assert_allclose(result.weights, [0., -.08, .08], atol=1e-9)
    with pytest.raises(ValueError, match="finite|missing|loading"):
        _solve(kind, previous=(.01, -.01), tradable=[False, True],
            factor_loadings=[[np.nan], [0.]], factor_lower=-.1, factor_upper=.1)


@pytest.mark.parametrize("kind", ["original", "tracking"])
@pytest.mark.parametrize("kwargs", [
    dict(factor_loadings=[[1.], [0.]]),
    dict(factor_lower=-.1, factor_upper=.1),
    dict(factor_loadings=[[1.]], factor_lower=-.1, factor_upper=.1),
    dict(factor_loadings=[[1.], [np.nan]], factor_lower=-.1, factor_upper=.1),
    dict(factor_loadings=[[1.], [np.inf]], factor_lower=-.1, factor_upper=.1),
    dict(factor_loadings=[[1.], [0.]], factor_lower=.2, factor_upper=.1),
    dict(factor_loadings=[[1.], [0.]], factor_lower=np.inf, factor_upper=np.inf),
    dict(factor_loadings=[[1.], [0.]], factor_lower=-np.inf, factor_upper=-np.inf),
    dict(factor_loadings=[[1.], [0.]], factor_lower=np.nan, factor_upper=.1),
    dict(factor_loadings=[[1., 0.], [0., 1.]], factor_lower=-.1,
         factor_upper=.1, factor_names=("same", "same")),
])
def test_malformed_factors_are_rejected_before_solving(kind, kwargs):
    with pytest.raises((ValueError, TypeError)):
        _solve(kind, **kwargs)


@pytest.mark.parametrize("kind,module", [("original", objective), ("tracking", constrained)])
def test_omitted_generic_factors_do_not_load_helper_or_change_beta_path(kind, module, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy solve must not load generic factor validation")

    monkeypatch.setattr(module, "_factor_arrays", forbidden)
    legacy = _solve(kind, beta=[1., 0.], max_abs_beta=.08)
    explicit = _solve(kind, beta=[1., 0.], max_abs_beta=.08,
        factor_loadings=None, factor_lower=None, factor_upper=None, factor_names=None)
    np.testing.assert_array_equal(explicit.weights, legacy.weights)
    assert explicit.objective == legacy.objective
    assert explicit.factor_exposures is explicit.factor_names is None


def test_result_constructors_keep_old_positional_fields_compatible():
    original = objective.ReturnRiskTradeResult("infeasible", *([None]*11), "legacy")
    tracking = constrained.ConstrainedRebalanceResult("infeasible", *([None]*8), "legacy")
    for result in (original, tracking):
        assert result.message == "legacy"
        assert result.factor_exposures is result.factor_names is None
        assert result.factor_lower_slack is result.factor_upper_slack is None


@pytest.mark.parametrize("filename,solver", [
    ("objective", "solve_return_risk_rebalance"),
    ("constrained", "solve_constrained_rebalance"),
])
def test_standalone_namespace_can_resolve_new_factor_contract_by_path(filename, solver, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "src/stable_finance/evaluation/trades" / (filename+".py")
    key = "isolated_generic_factor_"+filename
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    original_import = builtins.__import__

    def without_package(name, *args, **kwargs):
        if name.startswith("stable_finance"):
            raise ModuleNotFoundError("old namespace", name="stable_finance")
        return original_import(name, *args, **kwargs)

    try:
        spec.loader.exec_module(module)
        monkeypatch.setattr(builtins, "__import__", without_package)
        result = getattr(module, solver)([.3, -.3], np.eye(2),
            previous=[0., 0.], factor_loadings=[[1.], [0.]],
            factor_lower=-.03, factor_upper=.03)
        np.testing.assert_allclose(result.weights, [.03, -.03], atol=1e-9)
    finally:
        sys.modules.pop(key, None)


@pytest.mark.parametrize("kind", ["original", "tracking"])
def test_success_status_cannot_bypass_factor_primal_certificate(kind, monkeypatch):
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
        _solve(kind, factor_loadings=[[1.], [0.]], factor_lower=-.05, factor_upper=.05)
