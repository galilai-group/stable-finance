import numpy as np
import pytest

from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.scoring.ic import (
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


def overlapping_ics(seed=0, n_decisions=240, n_assets=40, steps=4):
    """A slow forecast of returns spanning ``steps`` decisions, two decisions unscored.

    The forecast is mostly a fixed characteristic, as a slow signal is, so
    neighbouring decisions score nearly the same view against returns that
    share ``steps - 1`` of their one-decision pieces.
    """
    rng = np.random.default_rng(seed)
    one = rng.normal(size=(n_decisions + steps, n_assets))
    realized = np.stack([one[t + 1:t + 1 + steps].sum(0) for t in range(n_decisions)])
    predicted = rng.normal(size=n_assets) + rng.normal(scale=0.1, size=realized.shape)
    predicted[[5, 17]] = np.nan
    return predicted, realized


def test_the_standard_error_is_unchanged_unless_lags_are_given():
    predicted, realized = overlapping_ics()
    plain = grouped_rank_ic(predicted, realized)
    assert grouped_rank_ic(predicted, realized, lags=None) == plain
    iid = grouped_rank_ic(predicted, realized, lags=0)
    assert iid.mean == plain.mean and iid.observations == plain.observations
    np.testing.assert_allclose(iid.standard_error, plain.standard_error, rtol=1e-12)


def test_the_overlap_aware_standard_error_is_newey_west_over_decisions():
    predicted, realized = overlapping_ics()
    lags = 5
    result = grouped_rank_ic(predicted, realized, lags=lags)
    # The per-decision ICs, each scored alone, at the rows they came from.
    ic = np.array([grouped_rank_ic(predicted[t:t + 1], realized[t:t + 1]).mean
                   for t in range(len(predicted))])
    rows = np.flatnonzero(np.isfinite(ic))
    x, n = ic[rows] - ic[rows].mean(), len(rows)
    # Every pair of scored decisions within ``lags`` rows, Bartlett-weighted;
    # the two unscored rows are gaps, not closed up.
    distance = np.abs(rows[:, None] - rows[None, :])
    weight = np.clip(1.0 - distance / (lags + 1.0), 0.0, None)
    expected = np.sqrt((weight * np.outer(x, x)).sum() / (n - 1) / n)
    np.testing.assert_allclose(result.standard_error, expected, rtol=1e-12)
    assert result.mean == grouped_rank_ic(predicted, realized).mean
    # Four-decision returns overlap, so their ICs are positively correlated
    # and the iid figure understates the uncertainty.
    assert result.standard_error > 1.5 * grouped_rank_ic(predicted, realized).standard_error


def test_lags_must_be_a_non_negative_integer():
    predicted, realized = overlapping_ics(n_decisions=20)
    for lags in (-1, 1.5, True):
        with pytest.raises(ValueError, match="lags"):
            grouped_rank_ic(predicted, realized, lags=lags)
