"""Analytic, causal and failure-path checks for joint portfolio targets."""

from types import SimpleNamespace
import builtins

import numpy as np
import pytest

from stable_finance.evaluation.portfolio import joint
from stable_finance.evaluation.portfolio.joint import (
    JointPortfolioConfig, construct_joint_portfolio,
)
from stable_finance.evaluation.portfolio.joint_risk import JointReturnPanel, JointRiskModel
from stable_finance.evaluation.portfolio.joint_constraints import JointPortfolioConstraints
from stable_finance.evaluation.portfolio.construction import mean_variance_target
from stable_finance.evaluation.returns.contract import ForwardReturns


LOOSE = JointPortfolioConstraints(10.0, 5.0, False)


def panel(values, *, decisions=None, assets=None, horizons=None, delay=0):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :, None]
    elif values.ndim == 2:
        values = values[None]
    d, n, k = values.shape
    return JointReturnPanel(values, np.arange(d)*10+100 if decisions is None else decisions,
        np.arange(n) if assets is None else assets,
        np.arange(1, k+1)*300 if horizons is None else horizons,
        entry_delay_seconds=delay)


def risk_for(forecast, covariance, *, cutoff=0, **kwargs):
    return JointRiskModel(covariance, forecast.assets, forecast.horizons, cutoff,
        entry_delay_seconds=forecast.entry_delay_seconds, **kwargs)


def run(values, covariance, *, config=None, eligible=None, **panel_kwargs):
    forecast = panel(values, **panel_kwargs)
    return construct_joint_portfolio(forecast, risk=risk_for(forecast, covariance),
        config=JointPortfolioConfig() if config is None else config, eligible=eligible)


@pytest.mark.parametrize("limits", [JointPortfolioConstraints(10, 5, True),
                                    JointPortfolioConstraints(.3, .1, True), LOOSE])
def test_one_horizon_matches_the_same_supplied_mean_variance_problem(limits):
    mu = np.array([.4, -.2, .1])
    sigma = np.array([[1, .2, .1], [.2, 1, .3], [.1, .3, 2]])
    config = JointPortfolioConfig(risk_aversion=2, constraints=limits)
    result = run(mu, sigma, config=config)
    # Independent equality-only closed form; the tight fixture's active
    # solution has the first two names at their symmetric concentration caps.
    reference = np.linalg.solve(sigma, mu)/2
    if limits.dollar_neutral:
        hedge = np.linalg.solve(sigma, np.ones(3))
        reference -= hedge*reference.sum()/hedge.sum()
    if limits.max_weight == .1:
        reference = np.array([.1, -.1, 0.])
    np.testing.assert_allclose(result.current_targets.values[0, :, 0], reference, atol=2e-8)
    np.testing.assert_allclose(result.reference_plan[0, 0], reference, atol=2e-8)


def test_unconstrained_one_horizon_reuses_the_native_target_contract():
    forecast = panel([.4, -.2, .1])
    sigma = np.array([[1., .2, .1], [.2, 1., .3], [.1, .3, 2.]])
    native_forecast = ForwardReturns(forecast.values, forecast.decisions,
                                    forecast.assets, forecast.horizons)
    native = mean_variance_target(native_forecast, sigma[None], risk_aversion=2.)
    result = construct_joint_portfolio(forecast, risk=risk_for(forecast, sigma),
        config=JointPortfolioConfig(2., LOOSE))
    assert type(result.current_targets) is type(native)
    np.testing.assert_allclose(result.current_targets.values, native.values, atol=1e-13)


def test_binding_joint_qp_reports_the_optional_backend_when_unavailable(monkeypatch):
    original_import = builtins.__import__

    def without_osqp(name, *args, **kwargs):
        if name == "osqp" or name.startswith("osqp."):
            raise ImportError("optional backend deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_osqp)
    # An inactive-limits solution does not require the optional QP package.
    free = run([.2], [[1.]], config=JointPortfolioConfig(2., LOOSE))
    assert free.current_targets.values[0, 0, 0] == pytest.approx(.1)
    forecast = panel([[1., 2.], [-1., -2.]])
    with pytest.raises(ImportError, match=r"stable-finance\[joint\]"):
        construct_joint_portfolio(forecast, risk=risk_for(forecast, np.eye(4)))


@pytest.mark.parametrize("kwargs", [
    {"gross_limit": -1}, {"gross_limit": True}, {"max_weight": np.inf},
    {"max_weight": -.1}, {"dollar_neutral": 1},
])
def test_joint_position_limits_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        JointPortfolioConstraints(**kwargs)


def test_full_cross_interval_risk_matches_the_exact_two_stage_solution():
    # One asset, two disjoint reference-price payoffs. Later means hedge
    # earlier risk through the OFF-DIAGONAL covariance, not an added cost.
    covariance = np.array([[2, .8], [.8, 1]])
    increments = np.array([.2, .4])
    config = JointPortfolioConfig(2, LOOSE)
    result = run([[.2, .6]], covariance, config=config)
    expected = np.linalg.solve(covariance, increments)/2
    np.testing.assert_allclose(result.reference_plan[0, :, 0], expected, atol=1e-13)
    np.testing.assert_allclose(result.incremental_forecasts[0, :, 0], increments, atol=1e-15)
    assert result.expected_pnl[0] == pytest.approx(increments @ expected)
    assert result.variance[0] == pytest.approx(expected @ covariance @ expected)
    assert result.objective[0] == pytest.approx(increments @ expected-expected @ covariance @ expected)
    changed = run([[.2, 1.2]], covariance, config=config)
    assert changed.current_targets.values[0, 0, 0] != pytest.approx(result.current_targets.values[0, 0, 0])


def test_block_diagonal_risk_correctly_decouples_the_current_target():
    config = JointPortfolioConfig(2, LOOSE)
    first = run([[.2, .6]], np.diag([2, 1]), config=config)
    later_changed = run([[.2, 1.2]], np.diag([2, 1]), config=config)
    np.testing.assert_allclose(first.current_targets.values, later_changed.current_targets.values,
                               atol=1e-13)
    assert first.reference_plan[0, 1, 0] != pytest.approx(later_changed.reference_plan[0, 1, 0])
    assert first.current_targets.values[0, 0, 0] == pytest.approx(.05)


def test_differencing_preserves_common_decision_price_payoff_units():
    result = run([[.1, .3]], np.eye(2), config=JointPortfolioConfig(5, LOOSE),
                 horizons=[300, 900], delay=5)
    np.testing.assert_allclose(result.incremental_forecasts[0, :, 0], [.1, .2], atol=1e-15)
    np.testing.assert_allclose(result.reference_plan[0, :, 0], [.02, .04], atol=1e-13)
    np.testing.assert_array_equal(result.interval_seconds, [300, 600])
    np.testing.assert_array_equal(result.current_targets.horizons, [300])
    # Reference shares at P_decision=100, equity=1000. A delayed entry of
    # 110 does not change the known send-time denominator to 110.
    quantity = result.current_targets.values[0, 0, 0]*1000/100
    assert quantity == pytest.approx(.2)


def test_neutrality_is_enforced_per_stage_rather_than_on_the_global_plan():
    # Global neutrality alone could go long both names now and short both
    # later. Every stage is separately neutral, so these means give cash.
    result = run([[1, 0], [1, 0]], np.eye(4))
    np.testing.assert_allclose(result.reference_plan, 0, atol=1e-12)


def test_gross_and_name_caps_bind_at_every_stage_without_a_global_budget():
    increments = np.tile([2., 1., -1., -2.], (3, 1))
    cumulative = np.cumsum(increments, axis=0).T
    limits = JointPortfolioConstraints(.4, .15, True)
    result = run(cumulative, np.eye(12), config=JointPortfolioConfig(.2, limits))
    np.testing.assert_allclose(result.reference_plan.sum(axis=2), 0, atol=1e-9)
    np.testing.assert_allclose(np.abs(result.reference_plan).sum(axis=2), .4, atol=1e-8)
    assert np.abs(result.reference_plan).max() <= .15+1e-9
    assert np.abs(result.reference_plan).sum() == pytest.approx(1.2, abs=1e-8)


def test_explicit_known_stage_eligibility_is_a_holding_constraint():
    config = JointPortfolioConfig(2, LOOSE)
    allowed = np.array([[[True, False], [False, True]]])
    result = run([[.2, .5], [.3, .7]], np.eye(4), config=config, eligible=allowed)
    assert result.reference_plan[0, 0, 1] == 0
    assert result.reference_plan[0, 1, 0] == 0
    assert result.reference_plan[0, 0, 0] > 0
    assert result.reference_plan[0, 1, 1] > 0


@pytest.mark.parametrize("mask", [True, [True, False], [[True, False]],
                                  [[[True, False], [True, False]]]])
def test_supported_eligibility_shapes_have_explicit_meanings(mask):
    result = run([[.2, .5], [.3, .7]], np.eye(4),
                 config=JointPortfolioConfig(2, LOOSE), eligible=mask)
    assert np.isfinite(result.reference_plan).all()
    if np.asarray(mask).ndim:
        np.testing.assert_array_equal(result.reference_plan[:, :, 1], 0)


@pytest.mark.parametrize("mask", [[1, 0], [True], np.ones((2, 2), dtype=bool),
                                  np.ones((1, 2, 1), dtype=bool)])
def test_ambiguous_or_nonboolean_eligibility_is_rejected(mask):
    with pytest.raises(ValueError, match="eligible"):
        run([[.2, .5], [.3, .7]], np.eye(4), eligible=mask)


@pytest.mark.parametrize("constraints,mask", [(JointPortfolioConstraints(0, .1), None),
    (JointPortfolioConstraints(1, 0), None), (JointPortfolioConstraints(), False)])
def test_zero_budget_or_no_eligible_names_gives_exact_cash(constraints, mask):
    result = run([[.2, .5], [.3, .7]], np.eye(4),
        config=JointPortfolioConfig(constraints=constraints), eligible=mask)
    np.testing.assert_array_equal(result.reference_plan, 0)
    np.testing.assert_array_equal(result.optimality_gap, 0)


def test_singular_zero_risk_is_a_bounded_linear_problem():
    config = JointPortfolioConfig(constraints=JointPortfolioConstraints(.8, .2, True))
    result = run([[2, 3], [-2, -3]], np.zeros((4, 4)), config=config)
    np.testing.assert_allclose(result.reference_plan[0], [[.2, -.2], [.2, -.2]], atol=1e-12)
    np.testing.assert_allclose(result.variance, 0, atol=1e-15)


def test_asset_permutation_preserves_the_unique_joint_solution():
    rng = np.random.default_rng(19)
    n, k = 4, 3
    factors = rng.normal(size=(n*k, n*k))
    covariance = factors @ factors.T+np.eye(n*k)
    values = rng.normal(size=(n, k))*.1
    config = JointPortfolioConfig(2, JointPortfolioConstraints(3, .8))
    first = run(values, covariance, config=config)
    order = np.array([2, 0, 3, 1])
    coordinates = np.concatenate([h*n+order for h in range(k)])
    permuted = run(values[order], covariance[np.ix_(coordinates, coordinates)],
                   config=config, assets=order)
    np.testing.assert_allclose(permuted.reference_plan, first.reference_plan[:, :, order], atol=1e-12)


@pytest.mark.parametrize("scale", [1e-8, 1., 1e8])
def test_positive_objective_rescaling_preserves_the_plan(scale):
    covariance = np.array([[2, .8], [.8, 1]])
    config = JointPortfolioConfig(2, LOOSE)
    first = run([[.2, .6]], covariance, config=config)
    scaled = run(np.array([[.2, .6]])*scale, covariance*scale, config=config)
    np.testing.assert_allclose(scaled.reference_plan, first.reference_plan, atol=1e-13)


def test_a_future_forecast_row_cannot_change_an_earlier_target():
    values = np.array([[[.2, .6]], [[.2, .8]]])
    config = JointPortfolioConfig(2, LOOSE)
    first = run(values, [[2, .8], [.8, 1]], config=config)
    values[1] = [[10, -10]]
    changed = run(values, [[2, .8], [.8, 1]], config=config)
    np.testing.assert_array_equal(first.reference_plan[0], changed.reference_plan[0])


def test_outputs_are_readonly_and_do_not_alias_the_forecast():
    raw = np.array([[[.2, .6]]])
    result = run(raw, [[2, .8], [.8, 1]], config=JointPortfolioConfig(2, LOOSE))
    snapshot = result.reference_plan.copy()
    raw[:] = 100
    np.testing.assert_array_equal(result.reference_plan, snapshot)
    for value in (result.reference_plan, result.current_targets.values, result.incremental_forecasts,
                  result.expected_pnl, result.variance, result.objective, result.optimality_gap):
        with pytest.raises(ValueError, match="read-only"):
            value.flat[0] = 0


@pytest.mark.parametrize("field", ["assets", "horizons", "entry_delay"])
def test_risk_axes_and_entry_delay_must_match_exactly(field):
    forecast = panel([[.2, .5], [-.2, -.5]])
    risk = JointRiskModel(np.eye(4), forecast.assets[::-1] if field == "assets" else forecast.assets,
        [300, 900] if field == "horizons" else forecast.horizons, 0,
        entry_delay_seconds=5 if field == "entry_delay" else 0)
    with pytest.raises(ValueError, match="aligned|match"):
        construct_joint_portfolio(forecast, risk=risk)


def test_risk_cannot_be_used_before_its_information_cutoff():
    forecast = panel([.1, -.1], decisions=[100])
    risk = risk_for(forecast, np.eye(2), cutoff=101)
    with pytest.raises(ValueError, match="later|available"):
        construct_joint_portfolio(forecast, risk=risk)


def test_plain_forward_returns_do_not_implicitly_declare_reference_payoff_units():
    ordinary = ForwardReturns(np.ones((1, 2, 1)), [100], [0, 1], [300])
    risk = JointRiskModel(np.eye(2), [0, 1], [300], 0)
    with pytest.raises(TypeError, match="JointReturnPanel"):
        construct_joint_portfolio(ordinary, risk=risk)


def test_missing_cumulative_forecast_is_not_zero_filled_before_differencing():
    forecast = panel([[.2, np.nan]])
    with pytest.raises(ValueError, match="finite"):
        construct_joint_portfolio(forecast, risk=risk_for(forecast, np.eye(2)))


@pytest.mark.parametrize("value", [True, 0, -1, np.inf, np.nan])
def test_invalid_risk_preferences_are_rejected(value):
    with pytest.raises(ValueError, match="risk_aversion"):
        JointPortfolioConfig(risk_aversion=value)


def test_costs_and_current_holdings_are_not_portfolio_inputs():
    forecast = panel([.2, -.2])
    risk = risk_for(forecast, np.eye(2))
    for key in ("cost", "previous", "pending", "turnover_penalty"):
        with pytest.raises(TypeError):
            construct_joint_portfolio(forecast, risk=risk, **{key: 0})


class FakeSolver:
    def __init__(self, candidate, status="solved"):
        self.candidate = candidate
        self.status = status
        self.settings = None

    def setup(self, **kwargs):
        self.settings = kwargs

    def warm_start(self, **kwargs):
        pass

    def solve(self, **kwargs):
        return SimpleNamespace(x=self.candidate, info=SimpleNamespace(status=self.status))


def active_problem():
    return panel([[1., 2.], [-1., -2.]])


def test_a_false_solver_success_is_rejected_by_the_independent_gap(monkeypatch):
    forecast = active_problem()
    monkeypatch.setattr("osqp.OSQP", lambda: FakeSolver(np.zeros(8)))
    with pytest.raises(RuntimeError, match="optimality gap"):
        construct_joint_portfolio(forecast, risk=risk_for(forecast, np.eye(4)))


def test_an_independently_certified_candidate_can_survive_a_status_warning(monkeypatch):
    forecast = active_problem()
    weights = np.array([.1, -.1, .1, -.1])
    fake = FakeSolver(np.r_[weights, np.abs(weights)], status="maximum iterations reached")
    monkeypatch.setattr("osqp.OSQP", lambda: fake)
    result = construct_joint_portfolio(forecast, risk=risk_for(forecast, np.eye(4)))
    np.testing.assert_allclose(result.reference_plan[0], weights.reshape(2, 2), atol=1e-15)
    assert fake.settings["adaptive_rho_interval"] == 50
    assert fake.settings["eps_abs"] == fake.settings["eps_rel"] == 1e-10


@pytest.mark.parametrize("candidate", [None, np.zeros(4), np.full(8, np.nan),
                                      np.r_[np.ones(4), np.ones(4)]])
def test_invalid_or_infeasible_solver_candidates_are_rejected(monkeypatch, candidate):
    forecast = active_problem()
    monkeypatch.setattr("osqp.OSQP", lambda: FakeSolver(candidate))
    with pytest.raises(RuntimeError, match="invalid|violates"):
        construct_joint_portfolio(forecast, risk=risk_for(forecast, np.eye(4)))


def test_certificate_failure_is_not_treated_as_a_solution(monkeypatch):
    forecast = active_problem()
    monkeypatch.setattr(joint, "linprog", lambda *args, **kwargs:
                        SimpleNamespace(success=False, message="certificate unavailable"))
    with pytest.raises(RuntimeError, match="certificate failed"):
        construct_joint_portfolio(forecast, risk=risk_for(forecast, np.eye(4)))
