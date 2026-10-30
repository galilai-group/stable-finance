"""Oracle forecasts: the realized ranking degraded to a known rank IC."""

import numpy as np
import pytest

from stable_finance import ForwardReturns, grouped_rank_ic, oracle_forecast
from stable_finance.evaluation._grid import _rankdata
from stable_finance.evaluation.returns import copula_forecast


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
    # The copula is its own function, copula_forecast, not a noise mode.
    with pytest.raises(ValueError, match="noise must"):
        oracle_forecast(outcome, 0.02, noise="gaussian")
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


# --- per-horizon IC --------------------------------------------------------


@pytest.mark.parametrize("noise,scatter", [
    ("random", None), ("tails", None), ("tails", "sampling"), ("middle", 0.05),
])
def test_constant_per_horizon_ic_is_bitwise_the_scalar(noise, scatter):
    outcome = realized(ties=True)
    scalar = oracle_forecast(outcome, 0.04, noise=noise, scatter=scatter, seed=3)
    vector = oracle_forecast(outcome, np.array([0.04, 0.04]), noise=noise,
                             scatter=scatter, seed=3)
    np.testing.assert_array_equal(scalar.values, vector.values)


@pytest.mark.parametrize("noise", ["tails", "middle"])
def test_mirror_modes_hit_each_horizons_ic_at_every_decision(noise):
    outcome = realized()
    ics = per_decision_ic(oracle_forecast(outcome, [0.3, -0.05], noise=noise), outcome)
    np.testing.assert_allclose(ics[:, 0], 0.3, atol=1e-5)
    np.testing.assert_allclose(ics[:, 1], -0.05, atol=1e-5)


def test_random_noise_hits_each_horizons_ic_on_average():
    outcome = realized(n_decisions=1500, n_assets=200)
    ics = per_decision_ic(oracle_forecast(outcome, [0.3, 0.02], seed=1), outcome)
    assert ics[:, 0].mean() == pytest.approx(0.3, abs=0.005)
    assert ics[:, 1].mean() == pytest.approx(0.02, abs=0.005)


def test_default_scale_follows_each_horizons_ic():
    outcome = realized()
    forecast = oracle_forecast(outcome, [0.03, 0.01], noise="middle").values
    raw = oracle_forecast(outcome, [0.03, 0.01], noise="middle", scale=1.0).values
    for h, ic in enumerate([0.03, 0.01]):
        y = outcome.values[0, :, h]
        np.testing.assert_allclose(forecast[0, :, h] - y.mean(),
                                   ic * (raw[0, :, h] - y.mean()), atol=1e-15)


def test_per_horizon_ic_is_validated_element_by_element():
    outcome = realized()
    with pytest.raises(ValueError, match="one value per horizon"):
        oracle_forecast(outcome, [0.02, 0.02, 0.02])
    with pytest.raises(ValueError, match="one value per horizon"):
        oracle_forecast(outcome, [[0.02, 0.02]])
    with pytest.raises(ValueError, match="ic must"):
        oracle_forecast(outcome, [0.02, 1.5], noise="tails")
    with pytest.raises(ValueError, match="ic must"):
        oracle_forecast(outcome, [0.02, np.nan], noise="tails")
    with pytest.raises(ValueError, match="positive ic"):
        oracle_forecast(outcome, [0.02, -0.02], noise="random", seed=0)
    with pytest.raises(ValueError, match="scale explicitly"):
        oracle_forecast(outcome, [0.02, 0.0], noise="tails")
    # An explicit scale admits a zero target, as for a scalar.
    oracle_forecast(outcome, [0.02, 0.0], noise="tails", scale=1.0)


# --- persistence -----------------------------------------------------------


def iid_panel(seed=0, *, n_decisions=40, n_assets=120, n_horizons=1):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_decisions, n_assets, n_horizons)) * 1e-3
    return ForwardReturns(values, np.arange(n_decisions), np.arange(n_assets),
                          np.arange(1, n_horizons + 1))


def rank_map(forecast, outcome, horizon=0):
    """Forecast rank at each realized-rank position, per decision."""
    return np.array([
        _rankdata(forecast.values[d, np.argsort(outcome.values[d, :, horizon]), horizon])
        for d in range(outcome.values.shape[0])
    ])


@pytest.mark.parametrize("noise,scatter", [
    ("random", None), ("tails", "sampling"), ("middle", 0.05),
])
def test_persistence_of_one_is_bitwise_the_default(noise, scatter):
    outcome = realized()
    default = oracle_forecast(outcome, 0.05, noise=noise, scatter=scatter, seed=4)
    for persistence, days in [(None, None), (1, None), (1, np.arange(12) // 5)]:
        again = oracle_forecast(outcome, 0.05, noise=noise, scatter=scatter, seed=4,
                                persistence=persistence, days=days)
        np.testing.assert_array_equal(default.values, again.values)


def test_persistent_random_noise_repeats_the_rank_space_permutation_within_blocks():
    outcome = iid_panel(n_horizons=2)
    forecast = oracle_forecast(outcome, 0.1, seed=5, persistence=4)
    for h in range(2):
        maps = rank_map(forecast, outcome, h)
        for start in range(0, 40, 4):
            block = maps[start:start + 4]
            assert (block == block[0]).all()
        # Fresh draws for every block, and for every horizon.
        assert not (maps[0] == maps[4]).all()
    assert not (rank_map(forecast, outcome, 0)[0] == rank_map(forecast, outcome, 1)[0]).all()


def test_persistence_leaves_the_per_decision_ic_law_and_makes_it_constant_in_a_block():
    outcome = iid_panel(n_decisions=1500, n_assets=200)
    ics = per_decision_ic(oracle_forecast(outcome, 0.05, seed=1, persistence=5), outcome)[:, 0]
    assert ics.mean() == pytest.approx(0.05, abs=0.01)
    assert ics.std() == pytest.approx(1 / np.sqrt(199), rel=0.15)
    np.testing.assert_allclose(ics.reshape(-1, 5), np.repeat(ics[::5, None], 5, axis=1),
                               atol=1e-12)


def test_persistence_raises_the_rank_space_error_autocorrelation_on_iid_returns():
    outcome = iid_panel(n_decisions=60)
    positions = np.arange(1, outcome.values.shape[1] + 1)

    def lag_one(forecast):
        errors = rank_map(forecast, outcome) - positions
        return np.mean([np.corrcoef(errors[d], errors[d + 1])[0, 1]
                        for d in range(len(errors) - 1)])

    independent = lag_one(oracle_forecast(outcome, 0.05, seed=2))
    persistent = lag_one(oracle_forecast(outcome, 0.05, seed=2, persistence=6))
    # Independent draws share only their pull to the middle, about
    # (1 - ic) / 2; within a block the error by rank position is identical.
    assert independent == pytest.approx(0.475, abs=0.05)
    assert persistent > 0.9 > independent + 0.4


def test_persistence_shares_the_scatter_draw_in_mirror_modes():
    outcome = iid_panel(n_decisions=12)
    ics = per_decision_ic(
        oracle_forecast(outcome, 0.1, noise="tails", scatter="sampling", seed=3,
                        persistence=3), outcome)[:, 0]
    blocks = ics.reshape(4, 3)
    np.testing.assert_allclose(blocks, np.repeat(blocks[:, :1], 3, axis=1), atol=1e-5)
    assert np.ptp(blocks[:, 0]) > 0.01
    # Without scatter there is nothing random to share.
    np.testing.assert_array_equal(
        oracle_forecast(outcome, 0.1, noise="tails", persistence=3).values,
        oracle_forecast(outcome, 0.1, noise="tails").values,
    )


def test_persistence_blocks_restart_at_each_new_day():
    outcome = iid_panel(n_decisions=10)
    days = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    maps = rank_map(oracle_forecast(outcome, 0.1, seed=6, persistence=3, days=days), outcome)
    # Blocks: [0 1 2] [3 4] | [5 6 7] | [8 9].
    for block in ([0, 1, 2], [3, 4], [5, 6, 7], [8, 9]):
        assert (maps[block] == maps[block[0]]).all()
    for first, second in ([2, 3], [4, 5], [7, 8]):
        assert not (maps[first] == maps[second]).all()
    # Without day labels the same decisions fall into index // 3 blocks.
    undivided = rank_map(oracle_forecast(outcome, 0.1, seed=6, persistence=3), outcome)
    assert (undivided[3:6] == undivided[3]).all()


def test_persistent_noise_is_reproducible_from_its_seed():
    outcome = iid_panel()
    first = oracle_forecast(outcome, 0.1, seed=7, persistence=5).values
    np.testing.assert_array_equal(
        first, oracle_forecast(outcome, 0.1, seed=7, persistence=5).values
    )
    assert not np.array_equal(
        first, oracle_forecast(outcome, 0.1, seed=8, persistence=5).values
    )
    # The documented block generator, reproduced by hand for one decision.
    from stable_finance.evaluation.returns.oracle import _cross_section
    expected = _cross_section(outcome.values[7, :, 0], 0.1, "random", 0.1, None,
                              np.random.default_rng([7, 0, 1]))
    np.testing.assert_array_equal(first[7, :, 0], expected)
    # A generator seed is reproducible too, through the root it draws.
    np.testing.assert_array_equal(
        oracle_forecast(outcome, 0.1, seed=np.random.default_rng(1), persistence=5).values,
        oracle_forecast(outcome, 0.1, seed=np.random.default_rng(1), persistence=5).values,
    )


def test_invalid_persistence_and_days_are_rejected():
    outcome = realized()
    for bad in (0, -2, 2.0, True):
        with pytest.raises(ValueError, match="persistence must"):
            oracle_forecast(outcome, 0.02, seed=0, persistence=bad)
    with pytest.raises(ValueError, match="days must"):
        oracle_forecast(outcome, 0.02, seed=0, persistence=2, days=np.arange(11))
    with pytest.raises(ValueError, match="days must"):
        oracle_forecast(outcome, 0.02, seed=0, persistence=2, days=np.zeros((12, 1)))
    oracle_forecast(outcome, 0.02, seed=0, persistence=np.int64(2), days=np.zeros(12))


# --- copula_forecast: errors that belong to the name -----------------------


def copula_panel(seed=0, n_decisions=200, n_assets=120):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_decisions, n_assets, 1))
    return ForwardReturns(values, np.arange(n_decisions),
                          [f"A{i}" for i in range(n_assets)], np.array([300]))


def lag_one_rank_autocorrelation(values):
    ranks = np.argsort(np.argsort(values, axis=1), axis=1).astype(float)
    a = ranks[1:] - ranks[1:].mean(1, keepdims=True)
    b = ranks[:-1] - ranks[:-1].mean(1, keepdims=True)
    return float(np.mean((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))))


@pytest.mark.parametrize("target", [0.05, 0.2, -0.3])
def test_copula_forecast_hits_its_ic_on_average(target):
    panel = copula_panel()
    forecast = copula_forecast(panel, target, seed=3)
    ic = grouped_rank_ic(forecast.values[:, :, 0], panel.values[:, :, 0], min_assets=2)
    assert abs(ic.mean - target) < 3 * ic.standard_error + 0.01
    # Never steered: each decision scatters like a real forecast of this IC.
    ics = per_decision_ic(forecast, panel)[:, 0]
    assert ics.std() == pytest.approx((1 - target**2) / np.sqrt(119), rel=0.2)


def test_copula_forecast_hits_each_horizons_ic_on_average():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(600, 200, 2))
    panel = ForwardReturns(values, np.arange(600), np.arange(200), [300, 900])
    ics = per_decision_ic(copula_forecast(panel, [0.3, 0.02], seed=1), panel)
    assert ics[:, 0].mean() == pytest.approx(0.3, abs=0.01)
    assert ics[:, 1].mean() == pytest.approx(0.02, abs=0.01)


def test_copula_half_life_makes_the_forecast_move_slowly_by_name():
    panel = copula_panel()
    fresh = copula_forecast(panel, 0.05, seed=1).values[:, :, 0]
    slow = copula_forecast(panel, 0.05, seed=1, half_life=12).values[:, :, 0]
    # The realized returns are independent across decisions, so all of the
    # slow forecast's persistence is its names' own noise.
    assert lag_one_rank_autocorrelation(fresh) < 0.1
    assert lag_one_rank_autocorrelation(slow) > 0.85
    # A single decision's IC has the same law whatever the half-life.
    ic = grouped_rank_ic(slow, panel.values[:, :, 0], min_assets=2)
    assert abs(ic.mean - 0.05) < 0.03


def test_copula_noise_restarts_at_each_day():
    panel = copula_panel(n_decisions=20)
    days = np.repeat([0, 1], 10)
    split = copula_forecast(panel, 0.1, seed=2, half_life=50, days=days)
    whole = copula_forecast(panel, 0.1, seed=2, half_life=50)
    np.testing.assert_array_equal(split.values[:10], whole.values[:10])
    assert not np.array_equal(split.values[10:], whole.values[10:])
    # Without a half-life every decision is fresh, so days change nothing.
    np.testing.assert_array_equal(copula_forecast(panel, 0.1, seed=2, days=days).values,
                                  copula_forecast(panel, 0.1, seed=2).values)


def test_copula_forecast_is_reproducible_from_its_seed():
    panel = copula_panel(n_decisions=10)
    first = copula_forecast(panel, 0.1, seed=5, half_life=3).values
    np.testing.assert_array_equal(first, copula_forecast(panel, 0.1, seed=5, half_life=3).values)
    assert not np.array_equal(first, copula_forecast(panel, 0.1, seed=6, half_life=3).values)
    np.testing.assert_array_equal(
        copula_forecast(panel, 0.1, seed=np.random.default_rng(1)).values,
        copula_forecast(panel, 0.1, seed=np.random.default_rng(1)).values,
    )
    # Each horizon has a generator of its own, so adding horizons leaves the
    # first one alone.
    wide = ForwardReturns(np.repeat(panel.values, 3, axis=2), panel.decisions,
                          panel.assets, [300, 900, 1800])
    np.testing.assert_array_equal(
        copula_forecast(wide, 0.1, seed=5, half_life=3).values[:, :, :1], first
    )


# Computed by the former ``oracle_forecast(..., noise="gaussian",
# persistence=3, ...)`` on the panel below, rounded to 1e-12; shape
# (decision, horizon, asset).
FORMER_GAUSSIAN_MODE = np.array([
    [[0.311937380898, 0.934861147124, 1.180497790584, 0.712909203996, 0.783249845019, 0.777123597878],
     [0.088958704923, 0.091509709749, 0.086301675842, 0.370501192876, -0.271542674429, -0.281863700826]],
    [[-0.114140676738, 0.223951235185, 0.163771690650, -0.185368866090, 0.024552386398, -0.254118402846],
     [-0.614379730334, -0.417239583414, -0.906505546721, -0.240710099904, -0.619475925916, -0.962489064109]],
    [[0.224159958684, 0.256699863993, -0.225442565273, -0.626271295879, -0.359823370927, -0.448924110941],
     [0.063240408190, -0.120601586145, -0.230102623594, np.nan, -0.083194477361, -0.529459887716]],
    [[-0.865528111712, 0.263992190399, 0.241501329800, 0.157197794053, -0.516867771926, 0.588559788680],
     [-1.090886197271, -0.840942415308, -0.548410849193, -0.612043760483, -0.394256259328, -0.703851742137]],
    [[-0.561187708778, -0.587460459706, -0.562976417119, -0.747877927341, -0.596460948967, 0.034369712817],
     [-0.248286185734, 0.004254201728, 0.052572069638, -0.015730045829, 0.207143769627, -0.015071593854]],
]).transpose(0, 2, 1)


def test_copula_forecast_reproduces_the_former_gaussian_mode():
    rng = np.random.default_rng(31)
    values = rng.standard_normal((5, 6, 2))
    values[2, 3, 1] = np.nan
    panel = ForwardReturns(values, np.arange(5), np.arange(6), [300, 900])
    forecast = copula_forecast(panel, [0.4, -0.3], half_life=3, days=[0, 0, 0, 1, 1],
                               seed=11).values
    np.testing.assert_array_equal(np.isnan(forecast), np.isnan(FORMER_GAUSSIAN_MODE))
    # The former mode added a 1e-9 ramp to break ties; on tie-free data that
    # is the only difference, so the rankings agree exactly.
    for d in range(5):
        for h in range(2):
            finite = np.isfinite(forecast[d, :, h])
            np.testing.assert_array_equal(np.argsort(forecast[d, finite, h]),
                                          np.argsort(FORMER_GAUSSIAN_MODE[d, finite, h]))
    np.testing.assert_allclose(forecast, FORMER_GAUSSIAN_MODE, rtol=0, atol=1e-8)


def test_copula_values_are_realized_returns_shrunk_by_the_ic():
    panel = copula_panel(n_decisions=5)
    y = panel.values[0, :, 0]
    raw = copula_forecast(panel, 0.2, seed=4, scale=1.0).values[0, :, 0]
    np.testing.assert_allclose(np.sort(raw), np.sort(y), atol=1e-15)
    shrunk = copula_forecast(panel, 0.2, seed=4).values[0, :, 0]
    np.testing.assert_allclose(shrunk - y.mean(), 0.2 * (raw - y.mean()), atol=1e-15)


def test_copula_tied_realized_returns_give_tied_forecasts():
    panel = copula_panel(n_decisions=30)
    tied = ForwardReturns(np.round(panel.values, 1), panel.decisions, panel.assets,
                          panel.horizons)
    forecast = copula_forecast(tied, 0.2, seed=4, half_life=5, scale=1.0)
    for d in range(30):
        y, f = tied.values[d, :, 0], forecast.values[d, :, 0]
        # The realized values handed out in score order, ties and all: no
        # jitter separates them.
        np.testing.assert_allclose(np.sort(f), np.sort(y), atol=1e-15)
        assert len(np.unique(f)) == len(np.unique(y)) < len(y)
    ic = grouped_rank_ic(forecast.values[:, :, 0], tied.values[:, :, 0], min_assets=2)
    assert np.isfinite(ic.mean) and ic.mean > 0


def test_copula_missing_outcomes_stay_missing_and_flat_cross_sections_are_unscored():
    values = copula_panel(n_decisions=6).values.copy()
    values[0, :40, 0] = np.nan
    values[1, :, 0] = 0.0
    panel = ForwardReturns(values, np.arange(6), np.arange(120), [300])
    forecast = copula_forecast(panel, 0.1, seed=0, half_life=4).values
    np.testing.assert_array_equal(np.isnan(forecast[0, :, 0]), np.isnan(values[0, :, 0]))
    assert np.isnan(forecast[1, :, 0]).all()
    assert np.isfinite(forecast[2:]).all()


def test_copula_forecast_validates_its_arguments():
    panel = copula_panel(n_decisions=10)
    for bad in (1.0, -1.0, 1.5, np.nan, [0.1, 0.1]):
        with pytest.raises(ValueError, match="ic must"):
            copula_forecast(panel, bad)
    with pytest.raises(ValueError, match="scale explicitly"):
        copula_forecast(panel, 0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        copula_forecast(panel, 0.1, scale=0.0)
    for bad in (0, -2.0, np.nan, True):
        with pytest.raises(ValueError, match="half_life must"):
            copula_forecast(panel, 0.1, half_life=bad)
    with pytest.raises(ValueError, match="days must"):
        copula_forecast(panel, 0.1, half_life=2, days=np.arange(9))
    # An explicit scale admits a zero target, and a half-life need not be
    # a whole number of decisions.
    copula_forecast(panel, 0.0, scale=1.0, half_life=2.5, days=np.zeros(10), seed=0)
