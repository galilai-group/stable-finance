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


# The run-to-tolerance solver: accelerated proximal gradient with adaptive restart.
def ill_conditioned(seed=0, n_decisions=5, n_assets=10, condition=2e3):
    """A correlated covariance with a wide spread of variances and a known condition number."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(n_assets, n_assets)))
    spectrum = np.geomspace(1.0, 1.0 / condition, n_assets)
    vols = np.geomspace(1.0, 8.0, n_assets)
    sigma = 1e-4 * (vols[:, None] * (basis * spectrum) @ basis.T * vols[None, :])
    # The forecast whose frictionless book at risk aversion 3 is of order one,
    # so tolerances on weights mean the same thing at every condition number.
    book = rng.normal(0, 0.5, (n_decisions, n_assets))
    forecast = ForwardReturns(3.0 * (book @ sigma)[:, :, None],
                              np.arange(n_decisions), [f"A{i}" for i in range(n_assets)],
                              [300])
    cost = 1e-4 * (1 + rng.random((n_decisions, n_assets)))
    return forecast, sigma[None], cost


def optimality_gap(w, previous, sigma, signal, cost, gamma):
    """Largest violation of the problem's optimality conditions, in return units."""
    gradient = gamma * sigma @ w - signal
    trade = w - previous
    traded = trade != 0
    on_boundary = np.abs(gradient[traded] + cost[traded] * np.sign(trade[traded]))
    inside = np.maximum(np.abs(gradient[~traded]) - cost[~traded], 0.0)
    return max(on_boundary.max(initial=0.0), inside.max(initial=0.0))


def test_the_restart_solver_meets_the_optimality_conditions_at_every_decision():
    forecast, sigma, cost = ill_conditioned()
    held, n_iter = allocate(forecast, sigma, risk_aversion=3.0, cost=cost,
                            solver="restart", return_n_iter=True)
    previous = np.zeros(10)
    for decision in range(5):
        w = held.values[decision, :, 0]
        gap = optimality_gap(w, previous, sigma[0], forecast.values[decision, :, 0],
                             cost[decision], 3.0)
        assert gap < 1e-9
        previous = w
    assert (n_iter < 20000).all()


def test_the_restart_solver_converges_where_the_fixed_budget_runs_out():
    """FISTA stopped by its budget is an approximation; restart reaches the optimum."""
    forecast, sigma, cost = ill_conditioned(seed=1)
    exact = allocate(forecast, sigma, risk_aversion=3.0, cost=cost, max_iter=100000,
                     tol=1e-14).values
    capped, capped_iter = allocate(forecast, sigma, risk_aversion=3.0, cost=cost,
                                   max_iter=50, return_n_iter=True)
    restart, restart_iter = allocate(forecast, sigma, risk_aversion=3.0, cost=cost,
                                     solver="restart", return_n_iter=True)
    capped, restart = capped.values, restart.values
    assert (capped_iter == 50).all()
    assert (restart_iter < 20000).all()
    # The step tolerance (1e-10) bounds the distance to the optimum only up to
    # the contraction factor of the last iterations, hence 1e-7 here.
    assert np.abs(restart - exact).max() < 1e-7
    assert np.abs(capped - exact).max() > 1e4 * np.abs(restart - exact).max()


def test_without_costs_the_restart_solver_lands_on_the_closed_form():
    forecast, sigma, _ = ill_conditioned(seed=2)
    aversion = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    target = mean_variance_target(forecast, sigma, risk_aversion=aversion)
    held = rebalance(target, sigma, risk_aversion=aversion, solver="restart",
                     tol=1e-13).values
    np.testing.assert_allclose(held, target.values, rtol=1e-7, atol=1e-9)


def test_the_two_step_and_one_shot_solves_agree_under_the_restart_solver():
    forecast, sigma, cost = ill_conditioned(seed=3)
    aversion = np.linspace(1.0, 6.0, 5)
    one_shot = allocate(forecast, sigma, risk_aversion=aversion, cost=cost,
                        solver="restart").values
    target = mean_variance_target(forecast, sigma, risk_aversion=aversion)
    two_step = rebalance(target, sigma, risk_aversion=aversion, cost=cost,
                         solver="restart").values
    np.testing.assert_allclose(two_step, one_shot, rtol=1e-7, atol=1e-9)


def test_an_untradable_name_holds_its_weight_under_the_restart_solver():
    forecast = ForwardReturns(np.full((3, 2, 1), 0.01), [0, 1, 2], ["A", "B"], [300])
    covariance = np.array([np.eye(2) * 0.01])
    weights = allocate(forecast, covariance, cost=[[np.nan, 10.0]],
                       initial=[[0.3], [-0.2]], solver="restart")
    np.testing.assert_array_equal(weights.values[:, 0, 0], 0.3)
    np.testing.assert_array_equal(weights.values[:, 1, 0], -0.2)


def test_a_name_without_variance_keeps_the_unscaled_step():
    """A zero diagonal entry cannot scale the metric; the solve must still be finite."""
    sigma = np.diag([1e-4, 2e-4, 0.0])
    sigma[0, 1] = sigma[1, 0] = 5e-5
    forecast = ForwardReturns(np.array([[[1e-3], [-1e-3], [1e-4]]]), [0], ["A", "B", "C"],
                              [300])
    cost = np.array([[1e-4, 1e-4, 1e-3]])  # C's edge never pays its cost
    restart = allocate(forecast, sigma[None], cost=cost, solver="restart").values
    fista = allocate(forecast, sigma[None], cost=cost, max_iter=20000, tol=1e-14).values
    assert np.isfinite(restart).all()
    assert restart[0, 2, 0] == 0.0
    np.testing.assert_allclose(restart, fista, rtol=1e-7, atol=1e-9)


def test_iteration_counts_have_one_entry_per_decision_and_horizon():
    returns = history(seed=6, n_assets=4)
    sigma = estimate_covariance(returns, shrinkage=0.1)
    forecast = ForwardReturns(np.random.default_rng(6).normal(0, 1e-3, (7, 4, 2)),
                              np.arange(7), returns.assets, returns.horizons)
    cost = np.full((7, 4), 1e-4)
    for solver in (None, "restart"):
        held, n_iter = allocate(forecast, sigma, cost=cost, solver=solver,
                                return_n_iter=True)
        assert n_iter.shape == (7, 2) and n_iter.dtype == np.int64
        assert (n_iter >= 1).all()
        np.testing.assert_array_equal(
            held.values, allocate(forecast, sigma, cost=cost, solver=solver).values)
    _, capped = allocate(forecast, sigma, cost=cost, max_iter=3, return_n_iter=True)
    assert (capped == 3).all()


def test_the_default_solver_is_the_fixed_budget_solver_bit_for_bit():
    forecast, sigma, cost = ill_conditioned(seed=4)
    implicit = allocate(forecast, sigma, cost=cost, max_iter=40).values
    explicit = allocate(forecast, sigma, cost=cost, max_iter=40, solver=None).values
    np.testing.assert_array_equal(implicit, explicit)


def test_reaching_the_restart_cap_warns_and_max_iter_can_only_raise_it(monkeypatch):
    import importlib

    from sklearn.exceptions import ConvergenceWarning

    # The package re-exports the function under the module's name.
    module = importlib.import_module("stable_finance.evaluation.trades.rebalance")
    monkeypatch.setattr(module, "RESTART_MAX_ITER", 10)
    forecast, sigma, cost = ill_conditioned(seed=5, n_decisions=2)
    # tol=0 can never be met, so every decision runs to the cap.
    with pytest.warns(ConvergenceWarning, match="2 of 2 decisions"):
        _, n_iter = allocate(forecast, sigma, cost=cost, solver="restart", max_iter=1,
                             tol=0.0, return_n_iter=True)
    assert (n_iter == 10).all()
    with pytest.warns(ConvergenceWarning, match="cap of 25"):
        _, n_iter = allocate(forecast, sigma, cost=cost, solver="restart", max_iter=25,
                             tol=0.0, return_n_iter=True)
    assert (n_iter == 25).all()


def test_the_fixed_budget_solver_never_warns(recwarn):
    forecast, sigma, cost = ill_conditioned(seed=5, n_decisions=2)
    allocate(forecast, sigma, cost=cost, max_iter=2)
    assert not recwarn.list


def test_an_unknown_solver_is_refused():
    forecast, sigma, _ = ill_conditioned(n_decisions=1)
    target = mean_variance_target(forecast, sigma)
    with pytest.raises(ValueError, match="solver='newton'"):
        rebalance(target, sigma, solver="newton")
    with pytest.raises(ValueError, match="solver='newton'"):
        allocate(forecast, sigma, solver="newton")


def test_the_allocator_passes_its_solver_through():
    returns = history(seed=7)
    forecast = ForwardReturns(np.random.default_rng(7).normal(0, 1e-3, (4, 3, 2)),
                              np.arange(4), returns.assets, returns.horizons)
    allocator = MeanVarianceAllocator(risk_aversion=2.0, solver="restart")
    assert clone(allocator).get_params()["solver"] == "restart"
    allocator.fit(returns)
    held = allocator.predict(forecast, cost=1e-4)
    direct = allocate(forecast, allocator.covariance_, risk_aversion=2.0, cost=1e-4,
                      solver="restart")
    np.testing.assert_array_equal(held.values, direct.values)
    assert MeanVarianceAllocator().get_params()["solver"] is None
