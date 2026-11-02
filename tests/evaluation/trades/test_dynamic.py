import numpy as np
import pytest

from stable_finance import (
    ForwardReturns,
    PortfolioWeights,
    aim_portfolio,
    mean_variance_target,
    no_trade_band,
    partial_adjustment,
)


def problem(seed=0, n_decisions=30, n_assets=5, horizons=(300,)):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_assets, 2))
    sigma = factors @ factors.T * 1e-4 + np.diag(rng.uniform(1e-4, 3e-4, n_assets))
    mu = rng.normal(0, 1e-3, (n_decisions, n_assets, len(horizons)))
    forecast = ForwardReturns(mu, np.arange(n_decisions), [f"A{i}" for i in range(n_assets)],
                              list(horizons))
    return forecast, sigma


def target(seed=1, n_decisions=30, n_assets=5):
    rng = np.random.default_rng(seed)
    return PortfolioWeights(rng.normal(0, 0.1, (n_decisions, n_assets, 1)),
                            np.arange(n_decisions), [f"A{i}" for i in range(n_assets)], [300])


def test_one_horizon_without_cost_aversion_is_the_mean_variance_target():
    forecast, sigma = problem()
    aim = aim_portfolio(forecast, sigma, risk_aversion=3.0)
    expected = mean_variance_target(forecast, sigma[None], risk_aversion=3.0)
    np.testing.assert_allclose(aim.values, expected.values, rtol=1e-10, atol=1e-14)


def test_horizons_enter_per_decision_and_cost_aversion_shrinks_the_fast_signal():
    forecast, sigma = problem(horizons=(300, 3600))
    free = aim_portfolio(forecast, sigma, risk_aversion=2.0)
    per_decision = forecast.values[:, :, 0] + forecast.values[:, :, 1] * 300 / 3600
    np.testing.assert_allclose(free.values[:, :, 0],
                               np.linalg.solve(sigma, per_decision.T).T / 2.0, rtol=1e-10)
    costly = aim_portfolio(forecast, sigma, risk_aversion=2.0, cost_aversion=10.0)
    fast, slow = 1 / (1 + np.log(2) * 10.0), 1 / (1 + np.log(2) / 12 * 10.0)
    mixed = forecast.values[:, :, 0] * fast + forecast.values[:, :, 1] * 300 / 3600 * slow
    np.testing.assert_allclose(costly.values[:, :, 0], np.linalg.solve(sigma, mixed.T).T / 2.0,
                               rtol=1e-10)
    assert slow > fast


def test_a_name_without_any_forecast_has_no_aim_and_a_missing_horizon_contributes_nothing():
    forecast, sigma = problem(horizons=(300, 900))
    values = forecast.values.copy()
    values[3, 1, :] = np.nan
    values[4, 2, 1] = np.nan
    gapped = ForwardReturns(values, forecast.decisions, forecast.assets, forecast.horizons)
    aim = aim_portfolio(gapped, sigma, risk_aversion=1.0)
    assert np.isnan(aim.values[3, 1, 0])
    assert np.isfinite(aim.values[4]).all()


@pytest.mark.parametrize("kwargs, message", [
    ({"risk_aversion": 0.0}, "risk_aversion"),
    ({"risk_aversion": 1.0, "cost_aversion": -1.0}, "cost_aversion"),
    ({"risk_aversion": 1.0, "half_lives": [1.0, 2.0]}, "half_lives"),
])
def test_aim_portfolio_validates_its_arguments(kwargs, message):
    forecast, sigma = problem()
    with pytest.raises(ValueError, match=message):
        aim_portfolio(forecast, sigma, **kwargs)


def test_rate_one_trades_to_the_target_and_a_smaller_rate_closes_the_gap_geometrically():
    goal = target()
    np.testing.assert_array_equal(partial_adjustment(goal, 1.0).values, goal.values)
    constant = PortfolioWeights(np.ones((10, 2, 1)), np.arange(10), ["A", "B"], [300])
    held = partial_adjustment(constant, 0.5).values[:, 0, 0]
    np.testing.assert_allclose(held, 1 - 0.5 ** np.arange(1, 11))


def test_partial_adjustment_holds_a_name_whose_target_is_missing():
    goal = target()
    values = goal.values.copy()
    values[5, 0, 0] = np.nan
    held = partial_adjustment(PortfolioWeights(values, goal.decisions, goal.assets,
                                               goal.horizons), 0.3).values
    assert held[5, 0, 0] == held[4, 0, 0]
    with pytest.raises(ValueError, match="rate"):
        partial_adjustment(goal, 0.0)


def test_a_zero_band_trades_to_the_target_and_a_huge_band_never_trades():
    goal = target()
    np.testing.assert_array_equal(no_trade_band(goal, 0.0).values, goal.values)
    start = np.full((5, 1), 0.02)
    held = no_trade_band(goal, 10.0, initial=start).values
    np.testing.assert_array_equal(held, np.broadcast_to(start, held.shape))


def test_the_band_trades_only_to_its_edge_and_scales_with_cost():
    goal = PortfolioWeights(np.array([[[0.5]], [[0.52]], [[0.2]]]), [0, 1, 2], ["A"], [300])
    held = no_trade_band(goal, 0.1).values[:, 0, 0]
    np.testing.assert_allclose(held, [0.4, 0.42, 0.3])
    cost = np.array([[1.0], [1.0], [3.0]])
    wide = no_trade_band(goal, 0.1, cost=cost).values[:, 0, 0]
    np.testing.assert_allclose(wide, [0.4, 0.42, 0.42])


def test_an_untradable_name_holds_its_weight_in_the_band_rule():
    goal = target()
    cost = np.ones((30, 5))
    cost[7, 2] = np.inf
    held = no_trade_band(goal, 0.01, cost=cost).values
    assert held[7, 2, 0] == held[6, 2, 0]
