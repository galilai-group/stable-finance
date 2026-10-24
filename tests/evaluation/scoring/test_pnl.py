import numpy as np

from stable_finance.evaluation.scoring.sharpe import (
    cross_spread_sharpe,
    mid_price_sharpe,
    sharpe_ratio,
)
from stable_finance.evaluation.scoring.pnl import evaluate_weights
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns


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


def test_sharpe_uses_sample_volatility_and_annualizes():
    returns = np.array([0.0, 0.02])
    assert np.isclose(sharpe_ratio(returns, periods_per_year=2), 1.0)


def _portfolio_panel():
    axes = ([1, 2], ["A", "B"], [300])
    weights = PortfolioWeights(
        np.array([[[0.5], [-0.5]], [[0.0], [1.0]]]), *axes
    )
    realized = ForwardReturns(
        np.array([[[0.10], [-0.10]], [[0.20], [0.03]]]), *axes
    )
    return weights, realized


def test_mid_price_sharpe_uses_frictionless_portfolio_returns():
    weights, realized = _portfolio_panel()
    expected = sharpe_ratio([0.10, 0.03], periods_per_year=2)
    assert np.isclose(
        mid_price_sharpe(weights, realized, periods_per_year=2)[0], expected
    )


def test_cross_spread_sharpe_executes_buys_at_ask_and_sells_at_bid():
    weights, realized = _portfolio_panel()
    bid = np.full((2, 2), 99.0)
    ask = np.full((2, 2), 101.0)
    # Gross returns are [0.10, 0.03]. Turnover is [1.0, 2.0], so crossing
    # costs are [0.01, 0.02] and net returns are [0.09, 0.01].
    expected = sharpe_ratio([0.09, 0.01], periods_per_year=2)
    assert np.isclose(
        cross_spread_sharpe(
            weights, realized, bid, ask, periods_per_year=2
        )[0],
        expected,
    )


def test_cross_spread_sharpe_does_not_treat_missing_trade_quote_as_free():
    weights, realized = _portfolio_panel()
    bid = np.array([[99.0, np.nan], [99.0, 99.0]])
    ask = np.array([[101.0, np.nan], [101.0, 101.0]])
    assert np.isnan(cross_spread_sharpe(weights, realized, bid, ask)[0])


def test_cross_spread_sharpe_rejects_crossed_quotes():
    weights, realized = _portfolio_panel()
    with np.testing.assert_raises_regex(ValueError, "below best_bid"):
        cross_spread_sharpe(
            weights,
            realized,
            [[101.0, 99.0], [99.0, 99.0]],
            [[99.0, 101.0], [101.0, 101.0]],
        )
