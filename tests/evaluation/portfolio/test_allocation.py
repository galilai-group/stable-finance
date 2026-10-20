import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from stable_finance import (
    ForwardReturns,
    MeanVarianceAllocator,
    allocate,
    estimate_covariance,
)


def history(seed=0, n_decisions=200, n_assets=3, horizons=(300, 900)):
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.01, size=(n_decisions, n_assets, len(horizons)))
    return ForwardReturns(
        values, np.arange(n_decisions), [f"A{i}" for i in range(n_assets)], horizons
    )


def test_covariance_matches_numpy_on_a_complete_panel_without_shrinkage():
    returns = history()
    got = estimate_covariance(returns, shrinkage=0.0)
    for horizon in range(2):
        expected = np.cov(returns.values[:, :, horizon], rowvar=False)
        np.testing.assert_allclose(got[horizon], expected, rtol=1e-6, atol=1e-12)


def test_covariance_is_estimated_pairwise_and_shrunk_toward_the_diagonal():
    complete = history()
    values = complete.values.copy()
    values[:50, 0, :] = np.nan
    returns = ForwardReturns(
        values, complete.decisions, complete.assets, complete.horizons
    )
    full = estimate_covariance(returns, shrinkage=0.0)
    shrunk = estimate_covariance(returns, shrinkage=1.0)
    assert np.isfinite(full).all()
    np.testing.assert_allclose(np.diag(shrunk[0]), np.diag(full[0]))
    assert np.allclose(shrunk[0] - np.diag(np.diag(shrunk[0])), 0.0)
    # Eigenvalues stay non-negative after the pairwise estimate.
    assert np.linalg.eigvalsh(full[0]).min() >= 0


def test_covariance_rejects_assets_without_enough_observations():
    values = history(n_decisions=5).values.copy()
    values[:, 1, :] = np.nan
    returns = ForwardReturns(values, np.arange(5), ["A0", "A1", "A2"], [300, 900])
    with pytest.raises(ValueError, match="finite returns"):
        estimate_covariance(returns)


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
