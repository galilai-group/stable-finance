"""Generic factor budgets constrain the original joint payoff objective."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from stable_finance.evaluation.portfolio.factors import FactorExposureConstraints
from stable_finance.evaluation.portfolio.joint import (
    JointPortfolioConfig, construct_joint_portfolio,
)
from stable_finance.evaluation.portfolio.joint_constraints import JointPortfolioConstraints
from stable_finance.evaluation.portfolio.joint_risk import JointReturnPanel, JointRiskModel


CONFIG = JointPortfolioConfig(2., JointPortfolioConstraints(2., 1., True))


def problem(*, k=1, d=1, covariance=None, zero_forecast=False):
    increments = np.array([[.3, -.3], [.6, -.6]])[:k]
    if zero_forecast:
        increments[:] = 0.
    forecast = JointReturnPanel(
        np.tile(np.cumsum(increments, axis=0).T[None], (d, 1, 1)),
        np.arange(d)*300+100, ["A", "B"], np.arange(1, k+1)*300)
    risk = JointRiskModel(np.eye(k*2) if covariance is None else covariance,
                          forecast.assets, forecast.horizons, 0)
    return forecast, risk


def budgets(loadings=((2.,), (.5,)), lower=-.1, upper=.1, **kwargs):
    return FactorExposureConstraints(loadings, lower, upper,
                                     ["A", "B"], ["market"], **kwargs)


def run(constraints=None, *, k=1, d=1, covariance=None, zero_forecast=False,
        config=CONFIG, **kwargs):
    forecast, risk = problem(k=k, d=d, covariance=covariance, zero_forecast=zero_forecast)
    return construct_joint_portfolio(forecast, risk=risk, config=config,
                                     factor_constraints=constraints, **kwargs)


def test_generic_beta_equivalent_matches_the_legacy_beta_optimum():
    generic = run(budgets())
    legacy = run(beta=[2., .5], max_abs_beta=.1)
    np.testing.assert_allclose(generic.reference_plan, legacy.reference_plan, atol=1e-10)
    np.testing.assert_allclose(generic.factor_exposures, .1, atol=1e-10)
    np.testing.assert_allclose(generic.reference_plan[0, 0], [.1/1.5, -.1/1.5], atol=1e-10)
    assert generic.optimality_gap.max() < 1e-8


def test_omitting_generic_factors_keeps_legacy_results_and_metadata():
    implicit = run(beta=[2., .5], max_abs_beta=.1)
    explicit = run(None, beta=[2., .5], max_abs_beta=.1)
    np.testing.assert_array_equal(implicit.reference_plan, explicit.reference_plan)
    for field in ("factor_constraints", "factor_names", "factor_loadings", "factor_exposures",
                  "factor_lower", "factor_upper", "factor_available_at",
                  "factor_lower_slack", "factor_upper_slack"):
        assert getattr(explicit, field) is None


def test_loose_generic_factors_preserve_the_exact_unconstrained_target():
    before = run()
    after = run(budgets(lower=-1., upper=1.))
    np.testing.assert_array_equal(after.reference_plan, before.reference_plan)


def test_asymmetric_upper_bound_changes_the_optimum_in_return_units():
    result = run(budgets(lower=-.4, upper=.06))
    np.testing.assert_allclose(result.reference_plan[0, 0], [.04, -.04], atol=1e-10)
    np.testing.assert_allclose(result.factor_lower_slack, .46, atol=1e-10)
    np.testing.assert_allclose(result.factor_upper_slack, 0., atol=1e-10)
    np.testing.assert_allclose(result.objective, [.6*.04-2*.04**2], atol=1e-12)


def test_positive_lower_bound_is_enforced_with_zero_signal():
    result = run(budgets(lower=.12, upper=.3), zero_forecast=True)
    # Minimize the original quadratic risk: exposure=1.5*a forces a=.08.
    np.testing.assert_allclose(result.reference_plan[0, 0], [.08, -.08], atol=1e-9)
    np.testing.assert_allclose(result.factor_exposures, .12, atol=1e-9)


def test_zero_objective_still_finds_a_nonzero_feasible_book():
    result = run(budgets(lower=.12, upper=.12), zero_forecast=True,
                 covariance=np.zeros((2, 2)))
    np.testing.assert_allclose(result.reference_plan[0, 0], [.08, -.08], atol=1e-9)
    np.testing.assert_array_equal(result.objective, [0.])
    np.testing.assert_array_equal(result.optimality_gap, [0.])


@pytest.mark.parametrize("config,eligible", [
    (JointPortfolioConfig(2., JointPortfolioConstraints(0., 1., True)), None),
    (JointPortfolioConfig(2., JointPortfolioConstraints(2., 0., True)), None),
    (CONFIG, [False, False]),
])
def test_nonzero_required_exposure_rejects_a_forced_zero_book(config, eligible):
    with pytest.raises(RuntimeError, match="infeasible.*forced zero book"):
        run(budgets(lower=.1, upper=.2), config=config, eligible=eligible)


@pytest.mark.parametrize("config,eligible", [
    (JointPortfolioConfig(2., JointPortfolioConstraints(0., 1., True)), None),
    (JointPortfolioConfig(2., JointPortfolioConstraints(2., 0., True)), None),
    (CONFIG, [False, False]),
])
def test_forced_zero_book_is_valid_when_all_factor_budgets_admit_zero(config, eligible):
    result = run(budgets(), config=config, eligible=eligible)
    np.testing.assert_array_equal(result.reference_plan, 0.)
    np.testing.assert_array_equal(result.factor_exposures, 0.)


def test_cross_interval_risk_and_stage_equality_have_the_analytic_coupled_solution():
    covariance = np.array([[1., 0., .5, 0.], [0., 1., 0., .5],
                           [.5, 0., 1., 0.], [0., .5, 0., 1.]])
    constraint = budgets(lower=[[.075], [-np.inf]], upper=[[.075], [np.inf]],
                         horizons=[300, 600])
    result = run(constraint, k=2, covariance=covariance)
    # With w=(a,-a,b,-b), stage one fixes a=.05 and dU/db=1.2-4b-2a.
    np.testing.assert_allclose(result.reference_plan[0], [[.05, -.05], [.275, -.275]], atol=1e-9)
    np.testing.assert_allclose(result.factor_exposures[0, 0], [.075], atol=1e-10)
    assert result.optimality_gap.max() < 1e-8


def test_stage_and_decision_specific_budgets_apply_to_every_reference_block():
    loadings = np.array([[[[2.], [.5]], [[1.], [0.]]],
                         [[[1.], [0.]], [[2.], [.5]]]])
    upper = np.array([[[.09], [.12]], [[.05], [.15]]])
    constraint = budgets(loadings, lower=-upper, upper=upper,
                         decisions=[100, 400], horizons=[300, 600],
                         available_at=[[100, 90], [400, 390]])
    result = run(constraint, d=2, k=2)
    np.testing.assert_allclose(result.reference_plan[:, :, 0], [[.06, .12], [.05, .1]], atol=1e-9)
    np.testing.assert_allclose(result.factor_exposures, upper, atol=1e-9)
    np.testing.assert_array_equal(result.factor_available_at, [[100, 90], [400, 390]])
    assert result.current_targets.values.shape == (2, 2, 1)


def test_multiple_factors_intersect_including_a_dependent_equality():
    constraint = FactorExposureConstraints([[2., 1., 4.], [.5, 1., 1.]],
        [-.1, 0., -.12], [.1, 0., .12], ["A", "B"], ["market", "common", "style"])
    result = run(constraint)
    # Common loading duplicates net neutrality; style=2*market is tighter.
    np.testing.assert_allclose(result.reference_plan[0, 0], [.04, -.04], atol=1e-9)
    np.testing.assert_allclose(result.factor_exposures[0, 0], [.06, 0., .12], atol=1e-9)


def test_generic_and_legacy_beta_constraints_intersect_without_overwriting_names():
    constraint = FactorExposureConstraints([[1.], [0.]], -.04, .04,
                                          ["A", "B"], ["beta"])
    result = run(constraint, beta=[2., .5], max_abs_beta=.1)
    np.testing.assert_allclose(result.reference_plan[0, 0], [.04, -.04], atol=1e-9)
    np.testing.assert_allclose(result.factor_exposures, .04, atol=1e-9)
    np.testing.assert_allclose(result.beta_exposures, .06, atol=1e-9)
    np.testing.assert_array_equal(result.factor_names, ["beta"])


@pytest.mark.parametrize("constraint", [
    budgets(lower=2., upper=3.),
    budgets([[1.], [1.]], lower=.1, upper=.2),
    FactorExposureConstraints([[2., 2.], [.5, .5]], [.1, -.2], [.2, -.1],
                              ["A", "B"], ["positive", "negative"]),
])
def test_infeasible_factor_intersections_fail_before_returning_a_plan(constraint):
    with pytest.raises(RuntimeError, match="factor constraints are infeasible"):
        run(constraint)


def test_legacy_beta_and_generic_bounds_can_be_jointly_infeasible():
    with pytest.raises(RuntimeError, match="factor constraints are infeasible"):
        run(budgets(lower=.12, upper=.2), beta=[2., .5], max_abs_beta=.1)


@pytest.mark.parametrize("lower,upper,expected", [
    (-np.inf, .06, .04), (.3, np.inf, .2), (-np.inf, np.inf, .15),
])
def test_one_sided_and_unbounded_factor_budgets(lower, upper, expected):
    result = run(budgets(lower=lower, upper=upper))
    np.testing.assert_allclose(result.reference_plan[0, 0], [expected, -expected], atol=1e-9)


def test_unknown_loadings_only_allowed_on_explicitly_unheld_assets():
    forecast = JointReturnPanel([[[.3], [-.3], [.1]]], [100], ["A", "B", "C"], [300])
    risk = JointRiskModel(np.eye(3), forecast.assets, forecast.horizons, 0)
    constraint = FactorExposureConstraints([[2.], [.5], [np.nan]], -.1, .1,
                                          forecast.assets, ["market"])
    result = construct_joint_portfolio(forecast, risk=risk, config=CONFIG,
        eligible=[True, True, False], factor_constraints=constraint)
    assert result.reference_plan[0, 0, 2] == 0.
    assert result.factor_loadings[0, 0, 2, 0] == 0.
    assert np.isnan(result.factor_constraints.loadings[2, 0])
    assert np.isfinite(result.factor_exposures).all()
    with pytest.raises(ValueError, match="potentially held"):
        construct_joint_portfolio(forecast, risk=risk, config=CONFIG, factor_constraints=constraint)


def test_factor_axes_and_availability_are_checked_against_the_forecast():
    wrong_assets = FactorExposureConstraints([[2.], [.5]], -.1, .1,
                                            ["B", "A"], ["market"])
    with pytest.raises(ValueError, match="asset order"):
        run(wrong_assets)
    with pytest.raises(ValueError, match="later"):
        run(budgets(available_at=101))
    with pytest.raises(TypeError, match="FactorExposureConstraints"):
        run({"loadings": [[2.], [.5]], "lower": -.1, "upper": .1})


def test_factor_diagnostics_are_copied_readonly_and_preserve_the_original_contract():
    loadings = np.array([[2.], [.5]])
    constraint = budgets(loadings)
    result = run(constraint)
    loadings[:] = 99.
    assert result.factor_constraints is constraint
    np.testing.assert_array_equal(result.factor_loadings[0, 0, :, 0], [2., .5])
    for name in ("factor_names", "factor_loadings", "factor_exposures", "factor_lower",
                 "factor_upper", "factor_available_at", "factor_lower_slack", "factor_upper_slack"):
        assert not getattr(result, name).flags.writeable


def test_unbounded_factors_still_reject_overflowing_exposure_diagnostics():
    forecast = JointReturnPanel([[[10.], [10.]]], [100], ["A", "B"], [300])
    risk = JointRiskModel(np.eye(2), forecast.assets, forecast.horizons, 0)
    config = JointPortfolioConfig(2., JointPortfolioConstraints(2., 1., False))
    constraint = budgets([[1e308], [1e308]], lower=-np.inf, upper=np.inf)
    with pytest.raises(RuntimeError, match="factor exposure diagnostics.*finite"):
        construct_joint_portfolio(forecast, risk=risk, config=config,
                                  factor_constraints=constraint)


def test_dependent_factor_equality_uses_exact_solution_without_the_optional_backend(monkeypatch):
    original = builtins.__import__

    def no_osqp(name, *args, **kwargs):
        if name == "osqp" or name.startswith("osqp."):
            raise ImportError("optional backend deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_osqp)
    result = run(budgets([[1.], [1.]], lower=0., upper=0.))
    np.testing.assert_allclose(result.reference_plan[0, 0], [.15, -.15], atol=1e-12)


class InvalidSolver:
    def setup(self, **kwargs):
        pass

    def warm_start(self, **kwargs):
        pass

    def solve(self, **kwargs):
        weights = np.array([.1, -.1])
        return SimpleNamespace(x=np.r_[weights, np.abs(weights)],
                               info=SimpleNamespace(status="solved"))


def test_solver_claimed_success_cannot_bypass_the_factor_certificate(monkeypatch):
    monkeypatch.setattr("osqp.OSQP", InvalidSolver)
    with pytest.raises(RuntimeError, match="violates stage constraints"):
        run(budgets())


def test_nonconverged_generic_qp_falls_back_to_the_same_certified_objective(monkeypatch):
    class Nonconverged(InvalidSolver):
        def solve(self, **kwargs):
            result = super().solve(**kwargs)
            result.info.status = "maximum iterations reached"
            return result

    monkeypatch.setattr("osqp.OSQP", Nonconverged)
    result = run(budgets())
    np.testing.assert_allclose(result.reference_plan[0, 0], [.1/1.5, -.1/1.5], atol=1e-9)
    assert result.optimality_gap.max() < 1e-8


def test_fallback_success_also_requires_the_independent_factor_certificate(monkeypatch):
    class Nonconverged(InvalidSolver):
        def solve(self, **kwargs):
            result = super().solve(**kwargs)
            result.info.status = "maximum iterations reached"
            return result

    monkeypatch.setattr("osqp.OSQP", Nonconverged)
    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs:
        SimpleNamespace(x=np.zeros(4), success=True, message="claimed success"))
    with pytest.raises(RuntimeError, match="factor QP could not be independently certified"):
        run(budgets())
