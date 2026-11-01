import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from stable_finance import (
    ForwardReturns,
    MeanVarianceAllocator,
    allocate,
    estimate_covariance,
    mean_variance_target,
    rebalance,
)


def history(seed=0, n_decisions=200, n_assets=3, horizons=(300, 900)):
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.01, size=(n_decisions, n_assets, len(horizons)))
    return ForwardReturns(
        values, np.arange(n_decisions), [f"A{i}" for i in range(n_assets)], horizons
    )


def test_free_trading_recovers_the_closed_form_mean_variance_portfolio():
    returns = history()
    allocator = MeanVarianceAllocator(risk_aversion=2.0, shrinkage=0.0).fit(returns)
    forecast = ForwardReturns(
        np.full((1, 3, 2), 0.001), [0], returns.assets, returns.horizons
    )
    weights = allocator.predict(forecast)
    for horizon in range(2):
        expected = np.linalg.solve(
            allocator.covariance_[horizon], forecast.values[0, :, horizon]
        ) / 2.0
        np.testing.assert_allclose(weights.values[0, :, horizon], expected, rtol=1e-5)


def test_transaction_cost_shrinks_the_trade_toward_the_current_portfolio():
    # One asset: minimise -mu w + (g/2) s2 w^2 + c |w - p| has the closed form
    # w = p + soft(mu / (g s2) - p, c / (g s2)).
    mu, variance, gamma, cost, previous = 0.002, 0.0004, 1.0, 0.001, 1.0
    forecast = ForwardReturns([[[mu]]], [0], ["A"], [300])
    covariance = np.array([[[variance]]])
    free = allocate(forecast, covariance, risk_aversion=gamma, initial=[[previous]])
    np.testing.assert_allclose(free.values[0, 0, 0], mu / variance, rtol=1e-6)
    paid = allocate(
        forecast, covariance, risk_aversion=gamma, cost=cost, initial=[[previous]]
    )
    target = mu / (gamma * variance) - previous
    expected = previous + np.sign(target) * max(abs(target) - cost / (gamma * variance), 0)
    np.testing.assert_allclose(paid.values[0, 0, 0], expected, rtol=1e-6)
    assert previous < paid.values[0, 0, 0] < free.values[0, 0, 0]


def test_a_prohibitive_cost_leaves_the_current_portfolio_untouched():
    forecast = ForwardReturns(np.full((3, 2, 1), 0.01), [0, 1, 2], ["A", "B"], [300])
    covariance = np.array([np.eye(2) * 0.01])
    weights = allocate(
        forecast, covariance, cost=[[np.nan, 10.0]], initial=[[0.3], [-0.2]]
    )
    np.testing.assert_allclose(weights.values[:, 0, 0], 0.3)
    np.testing.assert_allclose(weights.values[:, 1, 0], -0.2)


def test_rebalances_are_path_dependent_under_costs():
    forecast = ForwardReturns(
        [[[0.01]], [[0.01]], [[-0.01]]], [0, 1, 2], ["A"], [300]
    )
    covariance = np.array([[[0.01]]])
    weights = allocate(forecast, covariance, cost=0.002).values[:, 0, 0]
    # Same forecast twice: no reason to trade again at the second decision.
    assert weights[0] == pytest.approx(weights[1])
    # A reversal trades through the cost band from the held position.
    assert weights[2] < 0
    assert abs(weights[2]) < abs(weights[0])


def test_missing_forecasts_are_zero_expected_return_and_cost_still_applies():
    forecast = ForwardReturns([[[np.nan]]], [0], ["A"], [300])
    covariance = np.array([[[0.01]]])
    free = allocate(forecast, covariance, initial=[[0.5]])
    np.testing.assert_allclose(free.values[0, 0, 0], 0.0, atol=1e-8)
    costly = allocate(forecast, covariance, cost=1.0, initial=[[0.5]])
    np.testing.assert_allclose(costly.values[0, 0, 0], 0.5)


def test_allocator_is_a_scikit_learn_estimator():
    allocator = MeanVarianceAllocator(risk_aversion=3.0, shrinkage=0.2)
    assert clone(allocator).get_params()["risk_aversion"] == 3.0
    forecast = ForwardReturns(np.zeros((1, 3, 2)), [0], ["A0", "A1", "A2"], [300, 900])
    with pytest.raises(NotFittedError):
        allocator.predict(forecast)
    allocator.fit(history())
    with pytest.raises(ValueError, match="assets are not aligned"):
        allocator.predict(
            ForwardReturns(np.zeros((1, 3, 2)), [0], ["X", "Y", "Z"], [300, 900])
        )


def test_allocate_validates_covariance_shape_and_cost_sign():
    forecast = ForwardReturns(np.zeros((1, 2, 1)), [0], ["A", "B"], [300])
    with pytest.raises(ValueError, match="covariance must have shape"):
        allocate(forecast, np.eye(2))
    with pytest.raises(ValueError, match="cost cannot be negative"):
        allocate(forecast, np.array([np.eye(2)]), cost=-1.0)


def test_solution_satisfies_the_optimality_conditions_on_a_correlated_problem():
    rng = np.random.default_rng(1)
    n_assets = 12
    factors = rng.normal(size=(n_assets, 3))
    sigma = factors @ factors.T * 1e-4 + np.diag(rng.uniform(1e-4, 4e-4, n_assets))
    mu = rng.normal(scale=1e-3, size=n_assets)
    previous = rng.normal(scale=0.2, size=n_assets)
    cost = rng.uniform(1e-4, 1e-3, size=n_assets)
    gamma = 3.0
    forecast = ForwardReturns(mu[None, :, None], [0], np.arange(n_assets), [300])
    w = allocate(
        forecast, sigma[None], risk_aversion=gamma, cost=cost,
        initial=previous[:, None], max_iter=20000, tol=1e-13,
    ).values[0, :, 0]
    gradient = gamma * sigma @ w - mu
    trade = w - previous
    traded = np.abs(trade) > 1e-8
    # Traded assets sit on the cost boundary with the opposite sign of the
    # trade; untraded assets sit strictly inside it.
    np.testing.assert_allclose(gradient[traded], -cost[traded] * np.sign(trade[traded]), atol=1e-7)
    assert (np.abs(gradient[~traded]) <= cost[~traded] + 1e-7).all()
    assert traded.any() and (~traded).any()


def test_rebalancing_toward_the_mean_variance_target_is_the_one_shot_solve():
    """The portfolio stage's target plus the trades stage's move is allocate."""
    from stable_finance import mean_variance_target, rebalance

    rng = np.random.default_rng(4)
    returns = history(seed=4, n_assets=5)
    sigma = estimate_covariance(returns, shrinkage=0.2)
    forecast = ForwardReturns(rng.normal(0, 1e-3, (30, 5, 2)), np.arange(30),
                              returns.assets, returns.horizons)
    cost = 1e-4 * (1 + rng.random((30, 5)))
    one_shot = allocate(forecast, sigma, risk_aversion=3.0, cost=cost).values
    target = mean_variance_target(forecast, sigma, risk_aversion=3.0)
    two_step = rebalance(target, sigma, risk_aversion=3.0, cost=cost).values
    np.testing.assert_allclose(two_step, one_shot, rtol=1e-9, atol=1e-12)


def test_without_costs_the_rebalance_lands_on_the_target():
    from stable_finance import mean_variance_target, rebalance

    returns = history(seed=5, n_assets=4)
    sigma = estimate_covariance(returns, shrinkage=0.1)
    forecast = ForwardReturns(np.random.default_rng(5).normal(0, 1e-3, (10, 4, 2)),
                              np.arange(10), returns.assets, returns.horizons)
    target = mean_variance_target(forecast, sigma, risk_aversion=2.0)
    held = rebalance(target, sigma, risk_aversion=2.0).values
    np.testing.assert_allclose(held, target.values, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(
        target.values[:, :, 0], np.linalg.solve(sigma[0], forecast.values[:, :, 0].T).T / 2.0)


# Per-decision risk aversion in the cost-aware rebalance.
def problem(seed=0, n_decisions=6, n_assets=4):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_assets, 2))
    sigma = factors @ factors.T * 1e-4 + np.diag(rng.uniform(1e-4, 3e-4, n_assets))
    forecast = ForwardReturns(rng.normal(0, 1e-3, (n_decisions, n_assets, 1)),
                              np.arange(n_decisions), [f"A{i}" for i in range(n_assets)],
                              [300])
    cost = 1e-4 * (1 + rng.random((n_decisions, n_assets)))
    return forecast, sigma[None], cost


def test_a_constant_array_reproduces_the_scalar_path_exactly():
    forecast, sigma, cost = problem()
    for solve in (allocate, lambda f, s, **k: rebalance(
            mean_variance_target(f, s, risk_aversion=k["risk_aversion"]), s, **k)):
        scalar = solve(forecast, sigma, risk_aversion=3.0, cost=cost).values
        array = solve(forecast, sigma, risk_aversion=np.full(6, 3.0), cost=cost).values
        np.testing.assert_array_equal(array, scalar)


def test_without_cost_the_path_is_the_per_decision_closed_form():
    forecast, sigma, _ = problem(seed=1)
    aversion = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    target = mean_variance_target(forecast, sigma, risk_aversion=aversion)
    # The target moves by up to 2x between decisions here, so tighten the
    # solver's step-size stopping rule to measure the answer, not the stop.
    solver = {"max_iter": 20000, "tol": 1e-13}
    held = rebalance(target, sigma, risk_aversion=aversion, **solver).values[:, :, 0]
    closed = np.linalg.solve(sigma[0], forecast.values[:, :, 0].T).T / aversion[:, None]
    np.testing.assert_allclose(held, closed, rtol=1e-7, atol=1e-10)
    direct = allocate(forecast, sigma, risk_aversion=aversion, **solver).values[:, :, 0]
    np.testing.assert_allclose(direct, closed, rtol=1e-7, atol=1e-10)


def test_the_two_step_and_one_shot_solves_agree_under_per_decision_risk_aversion():
    forecast, sigma, cost = problem(seed=2)
    aversion = np.linspace(1.0, 6.0, 6)
    one_shot = allocate(forecast, sigma, risk_aversion=aversion, cost=cost).values
    target = mean_variance_target(forecast, sigma, risk_aversion=aversion)
    two_step = rebalance(target, sigma, risk_aversion=aversion, cost=cost).values
    np.testing.assert_allclose(two_step, one_shot, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("bad, message", [
    (np.ones(5), "one value per decision"),
    (np.ones((6, 1)), "one value per decision"),
    (np.array([1.0, 1.0, -2.0, 1.0, 1.0, 1.0]), "positive"),
    (0.0, "positive"),
    (np.array([1.0, np.inf, 1.0, 1.0, 1.0, 1.0]), "finite"),
])
def test_rebalance_and_allocate_reject_a_malformed_risk_aversion(bad, message):
    forecast, sigma, _ = problem()
    target = mean_variance_target(forecast, sigma)
    with pytest.raises(ValueError, match=message):
        rebalance(target, sigma, risk_aversion=bad)
    with pytest.raises(ValueError, match=message):
        allocate(forecast, sigma, risk_aversion=bad)


# Optional signed factor budgets use the existing allocator and trade path.
def test_omitted_factors_keep_the_existing_fista_path_bitwise(monkeypatch):
    import importlib
    module = importlib.import_module("stable_finance.evaluation.trades.rebalance")
    forecast, sigma, cost = problem(seed=7)
    expected = allocate(forecast, sigma, cost=cost).values
    def unexpected(*args, **kwargs):
        raise AssertionError("the factor solver must not run without supplied factors")
    monkeypatch.setattr(module, "_factor_rebalance", unexpected)
    np.testing.assert_array_equal(allocate(forecast, sigma, cost=cost,
        factor_loadings=None, factor_lower=None, factor_upper=None).values, expected)
    target = mean_variance_target(forecast, sigma)
    expected_tracking = rebalance(target, sigma, cost=cost).values
    np.testing.assert_array_equal(rebalance(target, sigma, cost=cost,
        factor_loadings=None, factor_lower=None, factor_upper=None).values, expected_tracking)


def test_single_asset_cost_band_and_factor_bounds_have_the_scalar_optimum():
    forecast = ForwardReturns([[[.03]], [[-.02]], [[.03]]], [0, 1, 2], ["A"], [300])
    weights = allocate(forecast, [[[.1]]], risk_aversion=2., cost=.001,
        factor_loadings=[[1.]], factor_lower=.02, factor_upper=.05)
    np.testing.assert_allclose(weights.values[:, 0, 0], [.05, .02, .05], atol=1e-10)


def test_multiple_factors_include_asymmetric_and_dependent_equalities():
    forecast = ForwardReturns([[[.02], [-.02]]], [0], ["A", "B"], [300])
    loadings = np.array([[1., 1., 2.], [0., 1., 2.]])
    weights = allocate(forecast, [np.eye(2)*.1], risk_aversion=2., cost=.001,
        factor_loadings=loadings, factor_lower=[-.01, 0., 0.],
        factor_upper=[.01, 0., 0.]).values[0, :, 0]
    np.testing.assert_allclose(weights, [.01, -.01], atol=1e-9)


def test_factor_units_do_not_change_the_optimized_portfolio():
    forecast = ForwardReturns([[[.02], [-.02]]], [0], ["A", "B"], [300])
    results = []
    for scale in (1e-12, 1., 1e12):
        results.append(allocate(forecast, [np.eye(2)*.1], cost=.001,
            factor_loadings=np.array([[1.], [0.]])*scale,
            factor_lower=-.01*scale, factor_upper=.01*scale).values)
    np.testing.assert_allclose(results, np.broadcast_to(results[0], (3, 1, 2, 1)), atol=1e-10)


def test_positive_factor_floor_is_enforced_even_with_no_forecast():
    forecast = ForwardReturns([[[0.]]], [0], ["A"], [300])
    constrained = allocate(forecast, [[[.1]]], cost=.01,
        factor_loadings=[[1.]], factor_lower=.02, factor_upper=np.inf)
    np.testing.assert_allclose(constrained.values, .02, atol=1e-10)


def test_frozen_holdings_reduce_the_remaining_exposure_budget():
    forecast = ForwardReturns([[[.01], [.01]]], [0], ["A", "B"], [300])
    constrained = allocate(forecast, [np.eye(2)*.1], initial=[[.03], [0.]],
        cost=[[np.nan, .001]], factor_loadings=[[1.], [1.]],
        factor_lower=0., factor_upper=0.)
    assert constrained.values[0, 0, 0] == .03
    np.testing.assert_allclose(constrained.values[0, 1, 0], -.03, atol=1e-10)


def test_infeasible_frozen_exposure_reports_decision_and_horizon():
    forecast = ForwardReturns([[[.01], [.01]]], [0], ["A", "B"], [300])
    with pytest.raises(ValueError, match="decision 0, horizon 0:.*infeasible"):
        allocate(forecast, [np.eye(2)*.1], initial=[[.2], [0.]], cost=[[np.nan, .001]],
            factor_loadings=[[1.], [0.]], factor_lower=-.1, factor_upper=.1)


def test_all_frozen_feasible_book_is_exact_and_needs_no_positive_free_risk():
    forecast = ForwardReturns([[[.01], [-.01]]], [0], ["A", "B"], [300])
    kwargs = dict(cost=np.inf, factor_loadings=[[1.], [1.]], factor_lower=0., factor_upper=0.)
    held = allocate(forecast, [np.zeros((2, 2))], initial=[[.03], [-.03]], **kwargs)
    np.testing.assert_array_equal(held.values[0, :, 0], [.03, -.03])
    with pytest.raises(ValueError, match="all positions frozen"):
        allocate(forecast, [np.zeros((2, 2))], initial=[[.03], [.03]], **kwargs)


def test_factor_constraints_retain_raw_reward_per_decision_gamma_and_all_horizons():
    import stable_finance as sf
    forecast, sigma, cost = problem(seed=8)
    gamma = np.linspace(1., 3., len(forecast.decisions))
    loadings = np.array([[1., 1.], [0., 1.], [0., 1.], [0., 1.]])
    kwargs = dict(risk_aversion=gamma, cost=cost, factor_loadings=loadings,
                  factor_lower=[-.2, 0.], factor_upper=[.2, 0.])
    direct = allocate(forecast, sigma, **kwargs)
    target = mean_variance_target(forecast, sigma, risk_aversion=gamma)
    tracking = rebalance(target, sigma, **kwargs)
    np.testing.assert_allclose(tracking.values, direct.values, atol=1e-7)
    report = sf.portfolio_exposure_report(direct, factor_loadings=loadings,
        factor_names=["proxy_beta", "net"], factor_lower=[-.2, 0.], factor_upper=[.2, 0.])
    assert np.all(report["constraint_status"] == "pass")
    assert report["factor_exposure"].shape == (6, 1, 2)


def test_allocator_predict_accepts_factor_limits_without_changing_fit_or_estimator_params():
    returns = history(seed=10, n_assets=3)
    estimator = MeanVarianceAllocator(risk_aversion=2.).fit(returns)
    original_params = estimator.get_params().copy()
    original_covariance = estimator.covariance_.copy()
    forecast = ForwardReturns(np.full((2, 3, 2), .001), [0, 1], returns.assets, returns.horizons)
    result = estimator.predict(forecast, factor_loadings=[[1.], [0.], [0.]],
        factor_lower=-.01, factor_upper=.01)
    np.testing.assert_allclose(result.values[:, 0], .01, atol=1e-8)
    np.testing.assert_array_equal(estimator.covariance_, original_covariance)
    assert estimator.get_params() == original_params


@pytest.mark.parametrize("kwargs", [
    dict(factor_lower=0.),
    dict(factor_loadings=[[1.], [0.]], factor_lower=0.),
    dict(factor_loadings=[[[1.], [0.]]], factor_lower=0., factor_upper=1.),
    dict(factor_loadings=[[1.], [np.nan]], factor_lower=0., factor_upper=1.),
    dict(factor_loadings=[[1.], [0.]], factor_lower=2., factor_upper=1.),
    dict(factor_loadings=[[1.], [0.]], factor_lower=np.inf, factor_upper=np.inf),
])
def test_malformed_or_missing_factor_budgets_are_rejected(kwargs):
    forecast = ForwardReturns([[[.01], [-.01]]], [0], ["A", "B"], [300])
    with pytest.raises(ValueError, match="factor"):
        allocate(forecast, [np.eye(2)], **kwargs)


@pytest.mark.parametrize("sigma", [np.ones((2, 2)), np.array([[1., 2.], [2., 1.]]),
                                   np.array([[1., .1], [0., 1.]])])
def test_factor_path_rejects_nonconvex_or_unsupported_singular_covariance(sigma):
    forecast = ForwardReturns([[[.01], [-.01]]], [0], ["A", "B"], [300])
    with pytest.raises(ValueError, match="positive definite|symmetric"):
        allocate(forecast, sigma[None], factor_loadings=[[1.], [1.]],
                 factor_lower=0., factor_upper=0.)


@pytest.mark.parametrize("tolerance", [1e-10, 1.])
def test_numerical_success_cannot_bypass_primal_or_dual_verification(monkeypatch, tolerance):
    from types import SimpleNamespace
    import scipy.optimize
    forecast = ForwardReturns([[[.01]]], [0], ["A"], [300])
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *args, **kwargs:
        SimpleNamespace(x=np.zeros(3), success=True, message="invented success"))
    with pytest.raises(RuntimeError, match="could not be verified"):
        allocate(forecast, [[[.1]]], factor_loadings=[[1.]], factor_lower=.02,
                 factor_upper=.03, tol=tolerance)


def test_correlated_cost_and_factor_equalities_are_verified_past_objective_roundoff():
    # The equality reduces w to [x, net-x]; the last factor's upper bound
    # determines x. The optimum is analytic, without another optimizer.
    mu = [.0017527815278021578, -.0002974795970127126]
    sigma = np.array([[.0009139989376175733, .0008817618608741598],
                      [.0008817618608741598, .001072717465987089]])
    previous = np.array([-.16664010734686124, -.3923171380886157])
    cost = [.0006447925051085614, .0008103968071411466]
    loadings = np.array([[1., -.6480395980117798, .26982433278566864],
                         [1., 1.3586196073598233, -1.2867258229920049]])
    lower = [.04640079681626046, .08986286066340371, -.29091745504334676]
    upper = [.04640079681626046, .3628374733832325, -.08184803846884012]
    net = lower[0]
    x = (upper[2]-loadings[1, 2]*net)/(loadings[0, 2]-loadings[1, 2])
    forecast = ForwardReturns(np.array(mu)[None, :, None], [0], ["A", "B"], [300])
    weights = allocate(forecast, sigma[None], risk_aversion=2.3,
        initial=previous[:, None], cost=cost, factor_loadings=loadings,
        factor_lower=lower, factor_upper=upper).values[0, :, 0]
    np.testing.assert_allclose(weights, [x, net-x], atol=1e-11)
