"""Oracle forecasts: the realized ranking degraded to a known rank IC."""

import numpy as np
import pytest

from stable_finance import ForwardReturns, grouped_rank_ic, oracle_forecast
from stable_finance.evaluation.returns.scoring import _rankdata


def realized(seed=0, *, n_decisions=12, n_assets=300, ties=False):
    rng = np.random.default_rng(seed)
    values = rng.standard_t(3, size=(n_decisions, n_assets, 2)) * 1e-3
    if ties:
        values = np.round(values, 4)
    return ForwardReturns(
        values, np.arange(n_decisions), np.arange(n_assets), [300, 900]
    )


def per_decision_ic(forecast, outcome):
    return np.array([
        [grouped_rank_ic(forecast.values[d:d + 1, :, h],
                         outcome.values[d:d + 1, :, h]).mean
         for h in range(outcome.values.shape[2])]
        for d in range(outcome.values.shape[0])
    ])


@pytest.mark.parametrize("noise", ["tails", "middle"])
@pytest.mark.parametrize("ic", [0.01, 0.03, 0.5, -0.02])
def test_every_decision_scores_the_target_ic(noise, ic):
    outcome = realized()
    forecast = oracle_forecast(outcome, ic, noise=noise, seed=1)
    np.testing.assert_allclose(per_decision_ic(forecast, outcome), ic, atol=1e-5)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("ic", [0.02, 0.5])
def test_random_noise_scores_the_target_ic_on_average_with_sampling_scatter(ties, ic):
    outcome = realized(n_decisions=1500, n_assets=200, ties=ties)
    ics = per_decision_ic(oracle_forecast(outcome, ic, noise="random", seed=1), outcome)
    # Each decision scatters like a real forecast of this IC would on a
    # cross-section of this size, so the mean is pinned only to its standard
    # error: 1 / sqrt(199) / sqrt(3000) is about 0.0013.
    assert ics.mean() == pytest.approx(ic, abs=0.005)
    assert ics.std() == pytest.approx(1 / np.sqrt(199), rel=0.15)


@pytest.mark.parametrize("noise", ["tails", "middle"])
def test_sampling_scatter_gives_mirror_modes_the_random_distribution(noise):
    outcome = realized(n_decisions=1500, n_assets=200)
    forecast = oracle_forecast(outcome, 0.02, noise=noise, scatter="sampling", seed=1)
    ics = per_decision_ic(forecast, outcome)
    assert ics.mean() == pytest.approx(0.02, abs=0.005)
    assert ics.std() == pytest.approx(1 / np.sqrt(199), rel=0.15)
    # The placement survives: the scatter only moves how far along the
    # priority order each decision gets. The budget's remainder hands back a
    # few of the cheapest pairs, so a wrong region is mostly, not entirely,
    # wrong.
    kept = correct_names(forecast, outcome)
    extreme, centre = np.r_[0:10, 190:200], slice(80, 120)
    if noise == "tails":
        assert not kept[:, extreme].any()
    else:
        assert kept[:, extreme].all() and kept[:, centre].mean() < 0.3


def test_numeric_scatter_sets_the_standard_deviation_and_zero_is_exact():
    outcome = realized(n_decisions=800, n_assets=200)
    wide = per_decision_ic(
        oracle_forecast(outcome, 0.3, noise="tails", scatter=0.1, seed=2), outcome
    )
    assert wide.std() == pytest.approx(0.1, rel=0.15)
    exact = per_decision_ic(
        oracle_forecast(outcome, 0.3, noise="tails", scatter=0.0, seed=2), outcome
    )
    np.testing.assert_allclose(exact, 0.3, atol=1e-5)


@pytest.mark.parametrize("noise", ["tails", "middle"])
def test_ties_in_the_outcome_are_scored_as_the_metric_scores_them(noise):
    outcome = realized(ties=True)
    assert len(np.unique(outcome.values[0, :, 0])) < 100
    forecast = oracle_forecast(outcome, 0.02, noise=noise, seed=1)
    np.testing.assert_allclose(per_decision_ic(forecast, outcome), 0.02, atol=2e-4)


def test_ic_one_reproduces_the_realized_ranking():
    outcome = realized()
    forecast = oracle_forecast(outcome, 1.0, noise="tails")
    for d in range(outcome.values.shape[0]):
        np.testing.assert_array_equal(
            np.argsort(forecast.values[d, :, 0]), np.argsort(outcome.values[d, :, 0])
        )


def test_mirror_modes_reach_minus_one_by_full_reversal():
    outcome = realized()
    for noise in ("tails", "middle"):
        forecast = oracle_forecast(outcome, -1.0, noise=noise)
        np.testing.assert_allclose(per_decision_ic(forecast, outcome), -1.0, atol=1e-9)


def correct_names(forecast, outcome):
    """Which realized-rank positions kept their forecast rank, per decision."""
    kept = []
    for d in range(outcome.values.shape[0]):
        y, f = outcome.values[d, :, 0], forecast.values[d, :, 0]
        order = np.argsort(y)
        kept.append(_rankdata(f[order]) == np.arange(1, len(y) + 1))
    return np.array(kept)


def test_tails_noise_leaves_the_middle_correct_and_middle_noise_the_tails():
    outcome = realized()
    n = outcome.values.shape[1]
    centre, tails = slice(n // 2 - 30, n // 2 + 30), np.r_[0:20, n - 20:n]

    # The budget's last few pairs land on the other side, so "mostly".
    by_tails = correct_names(oracle_forecast(outcome, 0.03, noise="tails"), outcome)
    assert not by_tails[:, tails].any() and by_tails[:, centre].mean() > 0.8

    by_middle = correct_names(oracle_forecast(outcome, 0.03, noise="middle"), outcome)
    assert by_middle[:, tails].all() and by_middle[:, centre].mean() < 0.2


def test_random_noise_spreads_errors_over_the_whole_ranking():
    outcome = realized()
    kept = correct_names(oracle_forecast(outcome, 0.03, noise="random", seed=0), outcome)
    n = kept.shape[1]
    thirds = [kept[:, :n // 3].mean(), kept[:, n // 3:2 * n // 3].mean(),
              kept[:, 2 * n // 3:].mean()]
    assert max(thirds) - min(thirds) < 0.05


def test_random_noise_is_reproducible_from_its_seed():
    outcome = realized()
    first = oracle_forecast(outcome, 0.03, seed=7).values
    np.testing.assert_array_equal(first, oracle_forecast(outcome, 0.03, seed=7).values)
    assert not np.array_equal(first, oracle_forecast(outcome, 0.03, seed=8).values)


def test_values_are_realized_returns_shrunk_by_the_ic():
    outcome = realized()
    y = outcome.values[0, :, 0]
    raw = oracle_forecast(outcome, 0.03, noise="middle", scale=1.0).values[0, :, 0]
    np.testing.assert_allclose(np.sort(raw), np.sort(y), atol=1e-10)
    shrunk = oracle_forecast(outcome, 0.03, noise="middle").values[0, :, 0]
    np.testing.assert_allclose(shrunk - y.mean(), 0.03 * (raw - y.mean()), atol=1e-15)


def test_missing_outcomes_stay_missing_and_flat_cross_sections_are_unscored():
    values = realized().values.copy()
    values[0, :40, 0] = np.nan
    values[1, :, 1] = 0.0
    outcome = ForwardReturns(values, np.arange(12), np.arange(300), [300, 900])
    forecast = oracle_forecast(outcome, 0.02, noise="tails")
    np.testing.assert_array_equal(
        np.isnan(forecast.values[0, :, 0]), np.isnan(values[0, :, 0])
    )
    assert np.isnan(forecast.values[1, :, 1]).all()
    assert per_decision_ic(forecast, outcome)[0, 0] == pytest.approx(0.02, abs=1e-5)


def test_invalid_arguments_are_rejected():
    outcome = realized()
    with pytest.raises(ValueError, match="ic must"):
        oracle_forecast(outcome, 1.5)
    with pytest.raises(ValueError, match="noise must"):
        oracle_forecast(outcome, 0.02, noise="edges")
    with pytest.raises(ValueError, match="scale explicitly"):
        oracle_forecast(outcome, 0.0)
    with pytest.raises(ValueError, match="positive ic"):
        oracle_forecast(outcome, -0.02, noise="random", seed=0)
    with pytest.raises(ValueError, match="scatter must be None"):
        oracle_forecast(outcome, 0.02, noise="tails", scatter="gaussian")
    with pytest.raises(ValueError, match="own sampling scatter"):
        oracle_forecast(outcome, 0.02, noise="random", scatter=0.05)
    with pytest.raises(ValueError, match="non-negative"):
        oracle_forecast(outcome, 0.02, noise="tails", scatter=-0.1)
    # "sampling" on random is the mode's own behaviour, not an error.
    oracle_forecast(outcome, 0.02, noise="random", scatter="sampling", seed=0)
