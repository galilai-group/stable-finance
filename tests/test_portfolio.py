import numpy as np

from stable_finance.data import ForwardReturns, PortfolioWeights
from stable_finance.portfolio import evaluate_weights


def test_evaluate_weights_reports_gross_cost_and_net_per_horizon():
    axes = ([1, 2], ["A", "B"], [300])
    weights = PortfolioWeights(
        np.array([[[0.5], [-0.5]], [[0.0], [1.0]]]), *axes
    )
    realized = ForwardReturns(
        np.array([[[0.10], [-0.10]], [[0.20], [0.03]]]), *axes
    )
    result, = evaluate_weights(
        weights, realized, half_spread=np.full((2, 2), 0.001), periods_per_year=2
    )
    np.testing.assert_allclose(result.gross_returns, [0.10, 0.03])
    np.testing.assert_allclose(result.costs, [0.001, 0.002])
    np.testing.assert_allclose(result.net_returns, [0.099, 0.028])


def test_evaluate_weights_returns_one_result_for_each_horizon():
    values = np.ones((2, 1, 2))
    axes = ([1, 2], ["A"], [300, 900])
    assert len(evaluate_weights(PortfolioWeights(values, *axes),
                                ForwardReturns(values, *axes))) == 2

