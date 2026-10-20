import numpy as np

from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.returns.scoring import (
    evaluate_forward_returns,
    grouped_rank_ic,
    grouped_rank_ic_by_label,
    paired_difference,
    pooled_estimates,
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


def test_flat_grouped_rank_ic_is_an_injectable_equivalent():
    predicted = np.array([1, 2, 2, 4, 4, 3, 2, 1])
    realized = np.array([1, 2, 2, 4, 1, 2, 3, 4])
    result = grouped_rank_ic_by_label(
        predicted, realized, np.repeat([10, 20], 4), min_assets=2
    )
    assert result.mean == 0.0
    assert result.standard_error == 1.0


def test_month_pooling_and_paired_differences():
    assert pooled_estimates([1, 2, 3]).mean == 2
    assert paired_difference([2, 4], [1, 1]).mean == 2


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
