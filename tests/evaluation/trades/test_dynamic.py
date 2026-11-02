"""Trading rules that move toward a target: partial adjustment and the no-trade band."""
import numpy as np
import pytest

from stable_finance import (
    PortfolioWeights,
    no_trade_band,
    partial_adjustment,
)


def target(seed=1, n_decisions=30, n_assets=5):
    rng = np.random.default_rng(seed)
    return PortfolioWeights(rng.normal(0, 0.1, (n_decisions, n_assets, 1)),
                            np.arange(n_decisions), [f"A{i}" for i in range(n_assets)], [300])


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
    wide = no_trade_band(goal, 0.1, cost_belief=cost).values[:, 0, 0]
    np.testing.assert_allclose(wide, [0.4, 0.42, 0.42])


def test_an_untradable_name_holds_its_weight_in_the_band_rule():
    goal = target()
    cost = np.ones((30, 5))
    cost[7, 2] = np.inf
    held = no_trade_band(goal, 0.01, cost_belief=cost).values
    assert held[7, 2, 0] == held[6, 2, 0]
