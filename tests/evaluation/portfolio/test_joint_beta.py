"""Explicit exogenous beta bounds are part of the certified portfolio QP."""

import builtins
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from stable_finance import (
    JointPortfolioConfig, JointPortfolioConstraints, JointReturnPanel,
    JointRiskModel, construct_joint_portfolio,
)


CONFIG = JointPortfolioConfig(2., JointPortfolioConstraints(2., 1., True))


def problem(*, k=1, d=1, covariance=None, decisions=None, assets=None):
    increments = np.array([[.3, -.3], [.6, -.6]])[:k]
    values = np.tile(np.cumsum(increments, axis=0).T[None], (d, 1, 1))
    forecast = JointReturnPanel(values, np.arange(d)*300+100 if decisions is None else decisions,
        ["A", "B"] if assets is None else assets, np.arange(1, k+1)*300)
    risk = JointRiskModel(np.eye(k*2) if covariance is None else covariance,
        forecast.assets, forecast.horizons, 0 if decisions is None else decisions[0])
    return forecast, risk


def run(*, beta=None, max_abs_beta=None, beta_available_at=None, **kwargs):
    forecast, risk = problem(**kwargs)
    return construct_joint_portfolio(forecast, risk=risk, config=CONFIG,
        beta=beta, max_abs_beta=max_abs_beta, beta_available_at=beta_available_at)


def test_two_name_neutral_book_matches_independent_scalar_beta_limited_optimum():
    # w=(a,-a): unconstrained a=.3/2=.15; beta exposure=1.5*a.
    result = run(beta=[2., .5], max_abs_beta=.1, beta_available_at=0)
    expected = min(.15, .1/1.5)
    np.testing.assert_allclose(result.current_targets.values[0, :, 0], [expected, -expected], atol=1e-9)
    np.testing.assert_allclose(result.beta_exposures, .1, atol=1e-9)
    assert result.max_abs_beta == .1
    assert result.optimality_gap.max() < 1e-8


def test_zero_beta_bound_and_dollar_neutrality_force_cash_for_two_different_betas():
    result = run(beta=[2., .5], max_abs_beta=0.)
    np.testing.assert_allclose(result.reference_plan, 0., atol=1e-8)
    np.testing.assert_allclose(result.beta_exposures, 0., atol=1e-8)


def test_loose_beta_limit_preserves_the_exact_existing_target():
    before = run()
    after = run(beta=[2., .5], max_abs_beta=.3)
    np.testing.assert_array_equal(after.current_targets.values, before.current_targets.values)
    assert before.beta is before.beta_exposures is before.max_abs_beta is None


def test_beta_constraint_applies_to_every_stage_and_decision():
    result = run(k=2, d=2, beta=[2., .5], max_abs_beta=.1, beta_available_at=0)
    np.testing.assert_allclose(result.reference_plan[:, :, 0], .1/1.5, atol=1e-9)
    np.testing.assert_allclose(result.beta_exposures, .1, atol=1e-9)
    assert result.current_targets.values.shape == (2, 2, 1)
    assert result.reference_plan.shape == (2, 2, 2)


def test_explicit_stage_beta_uses_interval_major_order():
    beta = np.array([[[2., .5], [1., 0.]]])
    result = run(k=2, beta=beta, max_abs_beta=.1)
    np.testing.assert_allclose(result.reference_plan[0, :, 0], [.1/1.5, .1], atol=1e-9)
    np.testing.assert_allclose(result.beta_exposures, .1, atol=1e-9)


def test_decision_varying_beta_does_not_leak_a_later_exposure_backwards():
    result = run(d=2, beta=[[2., .5], [1., 1.]], max_abs_beta=.1)
    np.testing.assert_allclose(result.current_targets.values[:, 0, 0], [.1/1.5, .15], atol=1e-9)


def test_beta_neutrality_dependent_on_dollar_neutrality_needs_no_optional_backend(monkeypatch):
    original = builtins.__import__

    def no_osqp(name, *args, **kwargs):
        if name == "osqp" or name.startswith("osqp."):
            raise ImportError("optional backend deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_osqp)
    result = run(beta=[1., 1.], max_abs_beta=0.)
    np.testing.assert_allclose(result.current_targets.values[0, :, 0], [.15, -.15], atol=1e-12)
    np.testing.assert_allclose(result.beta_exposures, 0., atol=1e-12)


def test_zero_risk_linear_problem_still_enforces_beta_bound():
    result = run(covariance=np.zeros((2, 2)), beta=[2., .5], max_abs_beta=.1)
    np.testing.assert_allclose(result.current_targets.values[0, :, 0], [.1/1.5, -.1/1.5], atol=1e-9)


def test_missing_beta_only_permitted_on_excluded_coordinates():
    forecast = JointReturnPanel([[[.3], [-.3], [.1]]], [100], ["A", "B", "C"], [300])
    risk = JointRiskModel(np.eye(3), forecast.assets, forecast.horizons, 0)
    allowed = np.array([True, True, False])
    result = construct_joint_portfolio(forecast, risk=risk, config=CONFIG,
        eligible=allowed, beta=[2., .5, np.nan], max_abs_beta=.1)
    assert result.reference_plan[0, 0, 2] == 0.
    assert np.isnan(result.beta[0, 0, 2])
    assert np.isfinite(result.beta_exposures).all()
    with pytest.raises(ValueError, match="finite for every eligible"):
        construct_joint_portfolio(forecast, risk=risk, config=CONFIG,
            beta=[2., .5, np.nan], max_abs_beta=.1)


@pytest.mark.parametrize("kwargs", [
    dict(beta=[1., 1.]), dict(max_abs_beta=.1), dict(beta_available_at=0),
    dict(beta=[1., 1.], max_abs_beta=-.1),
    dict(beta=[1., 1.], max_abs_beta=np.inf),
    dict(beta=[1., 1.], max_abs_beta=True),
    dict(beta=[True, False], max_abs_beta=.1),
    dict(beta=[1., np.inf], max_abs_beta=.1),
    dict(beta=[1., np.nan], max_abs_beta=.1),
    dict(beta=[[1.], [2.]], max_abs_beta=.1),
])
def test_invalid_beta_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)


@pytest.mark.parametrize("available", [101, [100, 401], [[100, 101], [400, 400]]])
def test_beta_cannot_be_observed_after_its_decision(available):
    with pytest.raises(ValueError, match="later than"):
        run(k=2, d=2, beta=[2., .5], max_abs_beta=.1, beta_available_at=available)


@pytest.mark.parametrize("available", [0, [100, 400], [[100, 100], [400, 400]]])
def test_scalar_decision_and_stage_beta_availability_supported(available):
    result = run(k=2, d=2, beta=[2., .5], max_abs_beta=.1, beta_available_at=available)
    assert result.beta_available_at.shape == (2, 2)


def test_beta_availability_supports_timezone_aware_decisions():
    start = datetime(2020, 8, 3, 10, tzinfo=timezone.utc)
    result = run(d=2, decisions=[start, start+timedelta(seconds=300)],
        beta=[2., .5], max_abs_beta=.1, beta_available_at=start-timedelta(days=3))
    assert result.beta_available_at.shape == (2, 1)
    with pytest.raises(ValueError, match="same time basis"):
        run(d=2, decisions=[start, start+timedelta(seconds=300)],
            beta=[2., .5], max_abs_beta=.1, beta_available_at=np.datetime64("2020-07-31"))


def test_beta_metadata_is_copied_readonly_without_mutating_the_input():
    beta = np.array([2., .5])
    result = run(beta=beta, max_abs_beta=.1)
    beta[:] = 99.
    np.testing.assert_array_equal(result.beta[0, 0], [2., .5])
    for array in (result.beta, result.beta_exposures, result.beta_available_at):
        assert not array.flags.writeable


def test_asset_permutation_preserves_the_constrained_plan():
    forecast, risk = problem(k=2)
    original = construct_joint_portfolio(forecast, risk=risk, config=CONFIG,
        beta=[2., .5], max_abs_beta=.1)
    order = [1, 0]
    rows = [1, 0, 3, 2]
    permuted = JointReturnPanel(forecast.values[:, order], forecast.decisions,
        forecast.assets[order], forecast.horizons)
    permuted_risk = JointRiskModel(risk.covariance[np.ix_(rows, rows)],
        permuted.assets, permuted.horizons, 0)
    result = construct_joint_portfolio(permuted, risk=permuted_risk, config=CONFIG,
        beta=[.5, 2.], max_abs_beta=.1)
    np.testing.assert_allclose(result.reference_plan, original.reference_plan[:, :, order], atol=1e-9)


def test_block_diagonal_risk_with_beta_constraints_preserves_first_stage_decoupling():
    forecast, risk = problem(k=2)
    first = construct_joint_portfolio(forecast, risk=risk, config=CONFIG,
        beta=[2., .5], max_abs_beta=.1)
    changed = forecast.values.copy()
    changed[:, :, 1] *= -2
    later = JointReturnPanel(changed, forecast.decisions, forecast.assets, forecast.horizons)
    second = construct_joint_portfolio(later, risk=risk, config=CONFIG,
        beta=[2., .5], max_abs_beta=.1)
    np.testing.assert_allclose(first.current_targets.values, second.current_targets.values, atol=1e-9)


class FakeSolver:
    def setup(self, **kwargs):
        pass

    def warm_start(self, **kwargs):
        pass

    def solve(self, **kwargs):
        # Position/gross/net limits pass, but beta exposure is .15 > .1.
        weights = np.array([.1, -.1])
        return SimpleNamespace(x=np.r_[weights, np.abs(weights)], info=SimpleNamespace(status="solved"))


def test_solver_success_cannot_bypass_independent_beta_feasibility(monkeypatch):
    monkeypatch.setattr("osqp.OSQP", FakeSolver)
    with pytest.raises(RuntimeError, match="violates stage constraints"):
        run(beta=[2., .5], max_abs_beta=.1)


class NonconvergedSolver(FakeSolver):
    def solve(self, **kwargs):
        result = super().solve(**kwargs)
        result.info.status = "maximum iterations reached"
        return result


def test_nonconverged_beta_qp_falls_back_to_the_same_certified_optimum(monkeypatch):
    monkeypatch.setattr("osqp.OSQP", NonconvergedSolver)
    result = run(beta=[2., .5], max_abs_beta=.1)
    # The primary candidate (.1,-.1) is NOT accepted. Another algorithm
    # solves the original constrained objective, with unchanged limits.
    np.testing.assert_allclose(result.current_targets.values[0, :, 0],
                               [.1/1.5, -.1/1.5], atol=1e-9)
    np.testing.assert_allclose(result.beta_exposures, .1, atol=1e-9)
    assert result.optimality_gap.max() < 1e-8


def test_fallback_success_flag_cannot_bypass_the_independent_certificate(monkeypatch):
    monkeypatch.setattr("osqp.OSQP", NonconvergedSolver)
    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs:
        SimpleNamespace(x=np.zeros(4), success=True, message="claimed success"))
    with pytest.raises(RuntimeError, match="could not be independently certified"):
        run(beta=[2., .5], max_abs_beta=.1)


def test_certified_primary_warning_does_not_need_fallback(monkeypatch):
    class CertifiedWarning(NonconvergedSolver):
        def solve(self, **kwargs):
            weights = np.array([.1/1.5, -.1/1.5])
            return SimpleNamespace(x=np.r_[weights, np.abs(weights)],
                info=SimpleNamespace(status="maximum iterations reached"))

    monkeypatch.setattr("osqp.OSQP", CertifiedWarning)
    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs:
        pytest.fail("a certified primary point must not invoke fallback"))
    result = run(beta=[2., .5], max_abs_beta=.1)
    np.testing.assert_allclose(result.beta_exposures, .1, atol=1e-9)


def test_no_beta_numerical_failure_keeps_the_existing_rejection_path(monkeypatch):
    class InvalidWarning(FakeSolver):
        def solve(self, **kwargs):
            return SimpleNamespace(x=np.zeros(4),
                info=SimpleNamespace(status="maximum iterations reached"))

    monkeypatch.setattr("osqp.OSQP", InvalidWarning)
    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs:
        pytest.fail("no-beta solves must not gain a numerical fallback"))
    forecast, risk = problem()
    tight = JointPortfolioConfig(2., JointPortfolioConstraints(.1, .05, True))
    with pytest.raises(RuntimeError, match="optimality gap"):
        construct_joint_portfolio(forecast, risk=risk, config=tight)
