import numpy as np

from stable_finance.data import ForwardReturns
from stable_finance.metrics import (
    evaluate_forward_returns,
    grouped_rank_ic,
    sharpe_ratio,
)


def test_grouped_rank_ic_averages_cross_sections_and_handles_ties():
    predicted = [[1, 2, 2, 4], [4, 3, 2, 1]]
    realized = [[1, 2, 2, 4], [1, 2, 3, 4]]
    result = grouped_rank_ic(predicted, realized, min_assets=2)
    assert result.mean == 0.0
    assert result.standard_error == 1.0
    assert result.observations == 2


def test_grouped_rank_ic_drops_incomplete_small_cross_sections():
    result = grouped_rank_ic(
        [[1, np.nan], [1, 2]], [[2, 1], [1, 2]], min_assets=2
    )
    assert result.mean == 1.0
    assert result.observations == 1


def test_sharpe_uses_sample_volatility_and_annualizes():
    returns = np.array([0.0, 0.02])
    assert np.isclose(sharpe_ratio(returns, periods_per_year=2), 1.0)


def test_evaluate_forward_returns_scores_every_horizon():
    axes = ([1], ["A", "B"], [300, 900])
    predicted = ForwardReturns(
        np.array([[[1.0, 2.0], [2.0, 1.0]]]), *axes
    )
    realized = ForwardReturns(
        np.array([[[1.0, 1.0], [2.0, 2.0]]]), *axes
    )
    results = evaluate_forward_returns(predicted, realized, min_assets=2)
    assert [result.mean for result in results] == [1.0, -1.0]
