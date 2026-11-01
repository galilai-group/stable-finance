import math
from itertools import pairwise
from statistics import NormalDist

import numpy as np
import pytest

from stable_finance.evaluation.scoring.sharpe import sharpe_ratio
from stable_finance.evaluation.scoring.stats import (
    PBOResult,
    deflated_sharpe,
    deflated_sharpe_from_trials,
    expected_max_sharpe,
    min_track_record,
    pbo,
    probabilistic_sharpe,
    sharpe_difference,
)

PHI = NormalDist()


def _hand_moments(values):
    values = np.asarray(values, dtype=float)
    deviation = values - values.mean()
    m2 = np.mean(deviation ** 2)
    sharpe = values.mean() / values.std(ddof=1)
    return (sharpe, np.mean(deviation ** 3) / m2 ** 1.5,
            np.mean(deviation ** 4) / m2 ** 2, len(values))


def test_probabilistic_sharpe_equals_phi_of_the_hand_computed_statistic():
    returns = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02, np.nan]
    sr, g3, g4, n = _hand_moments(returns[:-1])
    benchmark = 0.05
    z = (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    assert probabilistic_sharpe(returns, benchmark=benchmark) == pytest.approx(PHI.cdf(z), rel=1e-12)


def test_per_period_sharpe_agrees_with_sharpe_ratio_at_one_period_per_year():
    returns = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02]
    assert _hand_moments(returns)[0] == pytest.approx(sharpe_ratio(returns, periods_per_year=1))


def test_probabilistic_sharpe_increases_with_sample_length_for_a_positive_sharpe():
    block = np.array([0.02, -0.01, 0.015, -0.005, 0.01])
    values = [probabilistic_sharpe(np.tile(block, k)) for k in (2, 5, 20, 80)]
    assert all(later > earlier for earlier, later in pairwise(values))
    assert values[-1] > 0.99


def test_probabilistic_sharpe_on_normal_data_matches_the_normal_theory_formula():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.03, 1.0, size=20_000)
    sr, _, _, n = _hand_moments(returns)
    normal = PHI.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr ** 2 / 2))
    assert probabilistic_sharpe(returns) == pytest.approx(normal, abs=0.01)


def test_probabilistic_sharpe_is_nan_when_undefined():
    assert math.isnan(probabilistic_sharpe([0.01]))
    assert math.isnan(probabilistic_sharpe([0.01, 0.01, 0.01]))
    assert math.isnan(probabilistic_sharpe([0.01, 0.02], benchmark=np.nan))


def test_min_track_record_is_infinite_at_or_below_the_benchmark():
    returns = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02]
    sr = _hand_moments(returns)[0]
    assert min_track_record(returns, benchmark=sr) == math.inf
    assert min_track_record([-0.01, 0.005, -0.02, 0.0]) == math.inf


def test_min_track_record_matches_formula_and_shrinks_as_sharpe_grows():
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 1.0, size=500)
    noise = (noise - noise.mean()) / noise.std(ddof=1)
    lengths = [min_track_record(noise + drift) for drift in (0.05, 0.1, 0.2, 0.4)]
    assert all(later < earlier for earlier, later in pairwise(lengths))

    sr, g3, g4, _ = _hand_moments(noise + 0.1)
    expected = 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (PHI.inv_cdf(0.9) / sr) ** 2
    assert min_track_record(noise + 0.1, confidence=0.9) == pytest.approx(expected)


def test_min_track_record_rejects_a_confidence_outside_the_unit_interval():
    with pytest.raises(ValueError, match="confidence"):
        min_track_record([0.01, 0.02, 0.0], confidence=1.0)


def test_expected_max_sharpe_grows_with_trials_and_variance_and_vanishes_without_either():
    by_trials = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 10_000)]
    assert all(later > earlier for earlier, later in pairwise(by_trials))
    by_variance = [expected_max_sharpe(100, v) for v in (0.001, 0.01, 0.1)]
    assert all(later > earlier for earlier, later in pairwise(by_variance))
    assert expected_max_sharpe(1000, 0.0) == 0.0
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert math.isnan(expected_max_sharpe(10, np.nan))


def test_expected_max_sharpe_tracks_the_true_expected_maximum_of_normals():
    # E[max of 100 iid N(0,1)] is about 2.5076; the approximation is close.
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.5076, abs=0.05)
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -1.0)


def test_deflated_sharpe_never_exceeds_the_undeflated_probabilistic_sharpe():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.05, 1.0, size=750)
    psr = probabilistic_sharpe(returns, benchmark=0.0)
    for n_trials in (2, 10, 1000):
        assert deflated_sharpe(returns, n_trials=n_trials, trial_sharpe_var=0.002) < psr
    assert deflated_sharpe(returns, n_trials=1, trial_sharpe_var=0.002) == psr


def test_deflated_sharpe_from_trials_uses_the_matrix_own_trial_count_and_variance():
    rng = np.random.default_rng(5)
    matrix = rng.normal(0.0, 1.0, size=(400, 30))
    matrix[:, 0] += 0.2
    sharpes = np.array([_hand_moments(column)[0] for column in matrix.T])
    result = deflated_sharpe_from_trials(matrix)
    assert result.shape == (30,)
    expected = deflated_sharpe(matrix[:, 4], n_trials=30, trial_sharpe_var=sharpes.var(ddof=1))
    assert result[4] == pytest.approx(expected)
    assert np.argmax(result) == 0
    assert np.all(result <= np.array([probabilistic_sharpe(c) for c in matrix.T]))


def test_pbo_of_independent_noise_trials_is_near_one_half():
    rng = np.random.default_rng(0)
    result = pbo(rng.normal(0.0, 1.0, size=(1000, 50)), n_blocks=10)
    assert isinstance(result, PBOResult)
    assert result.n_combinations == math.comb(10, 5)
    assert result.logits.shape == (252,)
    assert 0.25 < result.probability < 0.75


def test_pbo_is_one_when_the_in_sample_winner_is_always_the_out_of_sample_loser():
    rng = np.random.default_rng(1)
    n_blocks, block_length, n_trials = 8, 50, 20
    # Each trial's block means sum to zero, so its in-sample mean is exactly the
    # negative of its out-of-sample mean; shared noise preserves the ordering.
    block_means = rng.normal(0.0, 0.05, size=(n_blocks, n_trials))
    block_means -= block_means.mean(axis=0)
    shared = rng.normal(0.0, 1.0, size=(n_blocks * block_length, 1))
    matrix = shared + np.repeat(block_means, block_length, axis=0)
    result = pbo(matrix, n_blocks=n_blocks)
    assert result.probability >= 0.95
    assert np.all(result.logits < 0)


def test_pbo_is_zero_when_one_trial_dominates_every_block():
    rng = np.random.default_rng(2)
    matrix = rng.normal(0.0, 1.0, size=(800, 20))
    matrix[:, 7] += 0.5
    result = pbo(matrix, n_blocks=16)
    assert result.probability == 0.0
    assert result.probability_of_loss == 0.0
    assert np.all(result.out_of_sample_sharpe > 0.3)


def test_pbo_drops_nan_rows_and_trailing_remainder_rows():
    rng = np.random.default_rng(4)
    matrix = rng.normal(0.0, 1.0, size=(103, 5))
    matrix[10, 2] = np.nan
    result = pbo(matrix, n_blocks=4)
    assert result.block_length == 102 // 4
    # 102 complete rows fill four blocks of 25; the last two rows are unused.
    altered = matrix.copy()
    altered[-2:, :] = [[50.0, -50.0, 0.0, 9.0, 1.0], [-3.0, 7.0, 2.0, 0.0, 4.0]]
    np.testing.assert_array_equal(pbo(altered, n_blocks=4).logits, result.logits)
    np.testing.assert_array_equal(
        pbo(np.delete(matrix, 10, axis=0)[:100], n_blocks=4).logits, result.logits
    )


def test_pbo_rejects_odd_too_many_or_too_large_block_counts():
    matrix = np.random.default_rng(6).normal(size=(400, 4))
    with pytest.raises(ValueError, match="even"):
        pbo(matrix, n_blocks=7)
    with pytest.raises(ValueError, match="even"):
        pbo(matrix, n_blocks=0)
    with pytest.raises(ValueError, match="splits"):
        pbo(matrix, n_blocks=20)
    with pytest.raises(ValueError, match="too few"):
        pbo(matrix[:3], n_blocks=4)
    with pytest.raises(ValueError, match="two trials"):
        pbo(matrix[:, :1], n_blocks=4)


def test_sharpe_difference_of_identical_series_is_zero_with_p_value_one():
    returns = np.random.default_rng(8).normal(0.01, 1.0, size=300)
    result = sharpe_difference(returns, returns)
    assert result.difference == 0.0
    assert result.p_value == 1.0
    assert result.lags == math.floor(4 * (300 / 100) ** (2 / 9))


def test_sharpe_difference_detects_a_clearly_better_series():
    rng = np.random.default_rng(9)
    common = rng.normal(0.0, 1.0, size=2000)
    better = common + rng.normal(0.0, 0.3, size=2000) + 0.15
    worse = common + rng.normal(0.0, 0.3, size=2000)
    result = sharpe_difference(better, worse)
    assert result.difference > 0
    assert result.p_value < 1e-3
    assert sharpe_difference(worse, better).z == pytest.approx(-result.z)


def test_sharpe_difference_with_zero_lags_is_the_iid_delta_method():
    rng = np.random.default_rng(10)
    a, b = rng.normal(0.1, 1.0, size=500), rng.normal(0.0, 1.0, size=500)
    iid = sharpe_difference(a, b, lags=0)
    hac = sharpe_difference(a, b, lags=5)
    assert iid.difference == hac.difference
    assert iid.standard_error != hac.standard_error
    with pytest.raises(ValueError, match="lags"):
        sharpe_difference(a, b, lags=-1)
    with pytest.raises(ValueError, match="same length"):
        sharpe_difference(a, b[:-1])


def test_every_statistic_is_deterministic():
    rng = np.random.default_rng(12)
    matrix = rng.normal(0.0, 1.0, size=(320, 12))
    first, second = pbo(matrix, n_blocks=8), pbo(matrix, n_blocks=8)
    assert first.probability == second.probability
    np.testing.assert_array_equal(first.logits, second.logits)
    np.testing.assert_array_equal(deflated_sharpe_from_trials(matrix),
                                  deflated_sharpe_from_trials(matrix))
    assert sharpe_difference(matrix[:, 0], matrix[:, 1]) == sharpe_difference(matrix[:, 0], matrix[:, 1])
    assert probabilistic_sharpe(matrix[:, 0]) == probabilistic_sharpe(matrix[:, 0])
