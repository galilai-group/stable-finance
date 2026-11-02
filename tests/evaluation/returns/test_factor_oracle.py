"""factor_forecast: a copula forecast whose errors are factor bets and whose IC drifts."""

from statistics import NormalDist

import numpy as np
import pytest

import stable_finance as sf
from stable_finance import ForwardReturns, MarketData, MissingRolesError
from stable_finance.evaluation.returns import copula_forecast, factor_forecast
from stable_finance.evaluation.returns.oracle import _factor_loadings, _factor_shares


def iid_panel(seed=0, n_decisions=300, n_assets=150, n_horizons=1, vol=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_decisions, n_assets, n_horizons))
    if vol is not None:
        values *= vol[None, :, None]
    return ForwardReturns(values, np.arange(n_decisions), [f"A{i}" for i in range(n_assets)],
                          [300, 900, 1800][:n_horizons])


def factor_panel(seed=0, n_decisions=400, n_assets=120, n_factors=3, loading=1.0):
    """Realized returns and an earlier history, both driven by the same factors."""
    rng = np.random.default_rng(seed)
    exposures = rng.normal(size=(n_assets, n_factors))

    def draw(n):
        return loading * rng.normal(size=(n, n_factors)) @ exposures.T + rng.normal(size=(n, n_assets))

    names = [f"A{i}" for i in range(n_assets)]
    history = ForwardReturns(draw(n_decisions)[:, :, None], np.arange(n_decisions) - n_decisions,
                             names, [300])
    realized = ForwardReturns(draw(n_decisions)[:, :, None], np.arange(n_decisions), names, [300])
    return realized, history


def ranks(values):
    return np.argsort(np.argsort(values, axis=1), axis=1).astype(float)


def rank_ic(forecast, outcome):
    """Per-decision rank IC of two tie-free (decision, asset) arrays."""
    a, b = ranks(forecast), ranks(outcome)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def normal_scores(values):
    inverse = np.vectorize(NormalDist().inv_cdf)
    return inverse((ranks(values) + 1) / (values.shape[1] + 1))


def forecast_errors(forecast, outcome):
    """Each decision's forecast normal scores with their part along the truth removed."""
    s, z = normal_scores(forecast), normal_scores(outcome)
    return s - ((s * z).sum(axis=1) / (z * z).sum(axis=1))[:, None] * z


def autocovariance(x, lag):
    x = x - x.mean()
    return float((x[lag:] * x[: len(x) - lag]).mean())


# --- off by default: the copula, draw for draw ------------------------------


@pytest.mark.parametrize("kwargs", [
    {},
    {"half_life": 4, "days": np.repeat([0, 1, 2], 20)},
    {"n_factors": 7, "ic_persistence": 0.9, "scale": 1.0},
])
def test_without_its_features_factor_forecast_is_bitwise_the_copula(kwargs):
    panel = iid_panel(n_decisions=60, n_assets=80, n_horizons=2)
    values = panel.values.copy()
    values[3, :10, 0] = np.nan
    values[5, :, 1] = 0.0
    panel = ForwardReturns(values, panel.decisions, panel.assets, panel.horizons)
    _, history = factor_panel(n_decisions=60, n_assets=80)
    copula_kwargs = {k: v for k, v in kwargs.items() if k in ("half_life", "days", "scale")}
    np.testing.assert_array_equal(
        factor_forecast(panel, [0.05, -0.2], factor_share=0.0, history=history, seed=7,
                        **kwargs).values,
        copula_forecast(panel, [0.05, -0.2], seed=7, **copula_kwargs).values,
    )


def test_factor_forecast_is_exported_from_stable_finance():
    assert sf.factor_forecast is factor_forecast
    assert {"factor_forecast", "QUALITY_VARIABLES"} <= set(sf.__all__)
    assert sf.QUALITY_VARIABLES == ("spread", "vol")


# --- the IC -----------------------------------------------------------------


@pytest.mark.parametrize("share", [0.3, 1.0])
@pytest.mark.parametrize("target", [0.05, 0.2, -0.1])
def test_factor_forecast_hits_its_ic_on_average(share, target):
    # Realized returns without factor structure: the errors' factor bets are
    # then unrelated to the outcome and the IC has the copula's law.
    realized = iid_panel(n_decisions=600, n_assets=150)
    _, history = factor_panel(n_assets=150)
    forecast = factor_forecast(realized, target, history=history, factor_share=share, seed=3)
    ics = rank_ic(forecast.values[:, :, 0], realized.values[:, :, 0])
    # At factor_share=1 the error is a three-dimensional pattern whose
    # spread across names is far from Gaussian, which moves a rank IC by a
    # few percent; the copula's own O(1/n) shortfall is about 1% here.
    assert abs(ics.mean() - target) < 3 * ics.std() / np.sqrt(len(ics)) + 0.04 * abs(target)
    copula = rank_ic(copula_forecast(realized, target, seed=3).values[:, :, 0],
                     realized.values[:, :, 0])
    assert ics.std() == pytest.approx(copula.std(), rel=0.15)


def test_factor_errors_widen_the_ic_spread_when_returns_share_the_factors():
    realized, history = factor_panel(loading=0.5, n_decisions=600)
    y = realized.values[:, :, 0]
    copula = rank_ic(factor_forecast(realized, 0.1, history=history, factor_share=0.0,
                                     seed=1).values[:, :, 0], y)
    bets = rank_ic(factor_forecast(realized, 0.1, history=history, factor_share=0.3,
                                   seed=1).values[:, :, 0], y)
    # The bet is right or wrong about many names at once: strategy risk that
    # breadth does not diversify.
    assert bets.std() > 1.5 * copula.std()
    # The mean falls short by about the variance the bets add to the
    # error's correlation with the outcome, and no more.
    assert bets.mean() == pytest.approx(copula.mean(), rel=0.1, abs=3 * bets.std() / np.sqrt(600))


def test_factor_share_raises_error_correlation_along_the_factors():
    realized = iid_panel(n_decisions=400, n_assets=120)
    _, history = factor_panel(n_assets=120)
    loadings = _factor_loadings(history.values[:, :, 0], 3)
    unit = loadings / np.linalg.norm(loadings, axis=1, keepdims=True)
    pairs = np.triu_indices(len(loadings), 1)
    for share in (0.0, 0.5, 1.0):
        forecast = factor_forecast(realized, 0.1, history=history, factor_share=share, seed=1)
        errors = forecast_errors(forecast.values[:, :, 0], realized.values[:, :, 0])
        # Each name's factor part is sqrt(share_i) times its unit loading
        # dotted with the factor shock.
        root = np.sqrt(_factor_shares(loadings, share)) if share else np.ones(len(unit))
        exposure = root[:, None] * unit
        # Across names, at each decision: the share of the error the
        # exposures explain.
        design = np.c_[np.ones(len(unit)), exposure]
        coef, *_ = np.linalg.lstsq(design, errors.T, rcond=None)
        residual = errors - (design @ coef).T
        explained = 1 - (residual**2).sum(1) / ((errors - errors.mean(1, keepdims=True))**2).sum(1)
        # Across time, for each pair of names: the errors' correlation is
        # the product of their exposures.
        slope = np.polyfit((exposure @ exposure.T)[pairs], np.corrcoef(errors.T)[pairs], 1)[0]
        if share == 0.0:
            assert explained.mean() < 0.05 and abs(slope) < 0.03
        else:
            assert explained.mean() == pytest.approx(share, abs=0.06)
            assert slope == pytest.approx(1.0, abs=0.12)


def test_factor_shares_follow_the_squared_loadings_and_average_the_share():
    rng = np.random.default_rng(0)
    loadings = rng.normal(size=(300, 3)) * np.exp(rng.normal(size=(300, 1)))
    size = (loadings**2).sum(axis=1)
    for share in (0.05, 0.3, 0.9):
        shares = _factor_shares(loadings, share)
        assert shares.mean() == pytest.approx(share)
        assert shares.max() <= 1.0
        # Uncapped names keep their squared loadings' proportions.
        free = shares < 1.0
        np.testing.assert_allclose(shares[free] / size[free], (shares / size)[free][0])
        assert np.all(size[~free] >= size[free].max())
    np.testing.assert_array_equal(_factor_shares(loadings, 1.0), 1.0)
    # A name without loadings cannot carry factor error.
    loadings[:30] = 0.0
    assert _factor_shares(loadings, 0.95).mean() == pytest.approx(0.9)


def test_the_ic_process_is_an_ar1_with_the_stated_moments():
    realized = iid_panel(n_decisions=3000, n_assets=300)
    y = realized.values[:, :, 0]
    sampling = (1 - 0.05**2) ** 2 / 299
    for persistence in (0.8, None):
        ics = rank_ic(factor_forecast(realized, 0.05, factor_share=0.0, ic_dispersion=0.05,
                                      ic_persistence=persistence, seed=4).values[:, :, 0], y)
        phi = persistence or 0.0
        assert ics.var() == pytest.approx(0.05**2 + sampling, rel=0.1)
        for lag in (1, 2, 5):
            assert autocovariance(ics, lag) == pytest.approx(0.05**2 * phi**lag, abs=4e-4)
        assert abs(ics.mean() - 0.05) < 3 * np.sqrt(ics.var() * (1 + phi) / (1 - phi) / 3000)
    # No dispersion, no drift, whatever the persistence.
    flat = rank_ic(factor_forecast(realized, 0.05, factor_share=0.0, ic_persistence=0.9,
                                   seed=4).values[:, :, 0], y)
    assert abs(autocovariance(flat, 1)) < 3e-4


def test_a_frozen_ic_process_draws_one_ic_per_day():
    realized = iid_panel(n_decisions=2000, n_assets=200)
    days = np.repeat(np.arange(40), 50)
    forecast = factor_forecast(realized, 0.2, factor_share=0.0, ic_dispersion=0.15,
                               ic_persistence=1.0, days=days, seed=2)
    ics = rank_ic(forecast.values[:, :, 0], realized.values[:, :, 0]).reshape(40, 50)
    # Within a day only the sampling scatter is left; between days the
    # target itself moves by ic_dispersion.
    assert ics.std(axis=1).mean() == pytest.approx((1 - 0.2**2) / np.sqrt(199), rel=0.1)
    assert ics.mean(axis=1).std(ddof=1) == pytest.approx(0.15, rel=0.3)


# --- quality ----------------------------------------------------------------


def thirds(key):
    order = np.argsort(key)
    return order[: len(key) // 3], order[-(len(key) // 3):]


@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_quality_spread_concentrates_accuracy_in_wide_spread_names(strength):
    realized = iid_panel(n_decisions=500, n_assets=240)
    rng = np.random.default_rng(5)
    half = np.exp(rng.normal(size=240) * 0.7) * 1e-4
    quoted = half[None, :] * np.exp(rng.normal(size=(500, 240)) * 0.05)
    market = MarketData({"bid_price": 100 * (1 - quoted), "ask_price": 100 * (1 + quoted)},
                        realized.decisions, realized.assets)
    # Out-of-sample loadings: the realized cross-sections would correlate
    # with loadings estimated on themselves, widening the IC's spread.
    history = iid_panel(seed=1, n_decisions=300, n_assets=240)
    forecast = factor_forecast(realized, 0.2, history=history, quality="spread",
                               quality_strength=strength, market=market, factor_share=0.3,
                               seed=2).values[:, :, 0]
    y = realized.values[:, :, 0]
    narrow, wide = thirds(half)
    assert rank_ic(forecast, y).mean() == pytest.approx(0.2, abs=0.01)
    # The thirds hold weights averaging 1 -+ 2/3 of the strength.
    assert rank_ic(forecast[:, wide], y[:, wide]).mean() == pytest.approx(
        0.2 * (1 + 2 / 3 * strength), rel=0.08)
    assert rank_ic(forecast[:, narrow], y[:, narrow]).mean() == pytest.approx(
        0.2 * (1 - 2 / 3 * strength), abs=0.015)


def test_quality_vol_ranks_names_by_the_history_volatility():
    rng = np.random.default_rng(6)
    vol = np.exp(rng.normal(size=240) * 0.5)
    realized = iid_panel(n_decisions=500, n_assets=240)
    history = iid_panel(seed=1, n_decisions=300, n_assets=240, vol=vol)
    forecast = factor_forecast(realized, 0.2, history=history, quality="vol",
                               factor_share=0.0, seed=2).values[:, :, 0]
    y = realized.values[:, :, 0]
    calm, volatile = thirds(vol)
    assert rank_ic(forecast[:, volatile], y[:, volatile]).mean() == pytest.approx(0.2 * 5 / 3,
                                                                                abs=0.02)
    assert rank_ic(forecast[:, calm], y[:, calm]).mean() == pytest.approx(0.2 / 3, abs=0.02)


def test_quality_keeps_the_ic_when_the_variable_tracks_the_tails():
    # Volatile names land in the tails of each cross-section, so weights
    # that merely average one would raise the IC with the accuracy.
    rng = np.random.default_rng(7)
    vol = np.exp(rng.normal(size=240) * 0.5)
    realized = iid_panel(n_decisions=500, n_assets=240, vol=vol)
    y = realized.values[:, :, 0]
    plain = rank_ic(factor_forecast(realized, 0.2, factor_share=0.0, seed=2).values[:, :, 0], y)
    graded = factor_forecast(realized, 0.2, quality="vol", factor_share=0.0, seed=2)
    assert rank_ic(graded.values[:, :, 0], y).mean() == pytest.approx(plain.mean(), rel=0.02)
    _, volatile = thirds(vol)
    lifted = rank_ic(graded.values[:, volatile, 0], y[:, volatile]).mean()
    assert lifted > rank_ic(factor_forecast(realized, 0.2, factor_share=0.0, seed=2)
                            .values[:, volatile, 0], y[:, volatile]).mean() + 0.03


# --- persistence, reproducibility, values -----------------------------------


def lag_one_rank_autocorrelation(values):
    a = ranks(values[1:])
    b = ranks(values[:-1])
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    return float(np.mean((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))))


def test_half_life_makes_the_factor_bets_persist_and_days_restart_them():
    realized = iid_panel(n_decisions=200, n_assets=120)
    _, history = factor_panel(n_assets=120)
    fresh = factor_forecast(realized, 0.05, history=history, factor_share=1.0, seed=1)
    slow = factor_forecast(realized, 0.05, history=history, factor_share=1.0, half_life=12,
                           seed=1)
    assert lag_one_rank_autocorrelation(fresh.values[:, :, 0]) < 0.15
    assert lag_one_rank_autocorrelation(slow.values[:, :, 0]) > 0.7
    days = np.repeat([0, 1], 100)
    split = factor_forecast(realized, 0.05, history=history, half_life=12, days=days, seed=1)
    whole = factor_forecast(realized, 0.05, history=history, half_life=12, seed=1)
    np.testing.assert_array_equal(split.values[:100], whole.values[:100])
    assert not np.array_equal(split.values[100:], whole.values[100:])


def test_factor_forecast_is_reproducible_from_its_seed():
    realized = iid_panel(n_decisions=30, n_assets=60)
    kwargs = {"factor_share": 0.4, "ic_dispersion": 0.02, "ic_persistence": 0.5,
              "quality": "vol", "half_life": 3}
    first = factor_forecast(realized, 0.1, seed=5, **kwargs).values
    np.testing.assert_array_equal(first, factor_forecast(realized, 0.1, seed=5, **kwargs).values)
    assert not np.array_equal(first, factor_forecast(realized, 0.1, seed=6, **kwargs).values)
    np.testing.assert_array_equal(
        factor_forecast(realized, 0.1, seed=np.random.default_rng(1), **kwargs).values,
        factor_forecast(realized, 0.1, seed=np.random.default_rng(1), **kwargs).values,
    )
    # Each horizon has generators of its own, and the loadings and
    # volatilities come from the first horizon: adding horizons leaves the
    # first one alone.
    wide = ForwardReturns(np.repeat(realized.values, 3, axis=2), realized.decisions,
                          realized.assets, [300, 900, 1800])
    np.testing.assert_array_equal(factor_forecast(wide, 0.1, seed=5, **kwargs).values[:, :, :1],
                                  first)


def test_loadings_do_not_depend_on_the_order_of_the_history():
    _, history = factor_panel()
    returns = history.values[:, :, 0]
    shuffled = returns[np.random.default_rng(0).permutation(len(returns))]
    loadings = _factor_loadings(returns, 3)
    np.testing.assert_allclose(_factor_loadings(shuffled, 3), loadings, atol=1e-10)
    assert (loadings**2).sum(axis=1).mean() == pytest.approx(1.0)


def test_values_are_realized_returns_shrunk_by_the_ic_and_gaps_stay_gaps():
    values = iid_panel(n_decisions=6, n_assets=120).values.copy()
    values[0, :40, 0] = np.nan
    values[1, :, 0] = 0.0
    realized = ForwardReturns(values, np.arange(6), np.arange(120), [300])
    raw = factor_forecast(realized, 0.2, seed=4, scale=1.0, ic_dispersion=0.05).values
    shrunk = factor_forecast(realized, 0.2, seed=4, ic_dispersion=0.05).values
    np.testing.assert_array_equal(np.isnan(raw[0, :, 0]), np.isnan(values[0, :, 0]))
    assert np.isnan(raw[1]).all()
    assert np.isfinite(raw[2:]).all()
    y = values[3, :, 0]
    np.testing.assert_allclose(np.sort(raw[3, :, 0]), np.sort(y), atol=1e-15)
    np.testing.assert_allclose(shrunk[3, :, 0] - y.mean(), 0.2 * (raw[3, :, 0] - y.mean()),
                               atol=1e-15)


def test_factor_forecast_validates_its_arguments():
    realized = iid_panel(n_decisions=10, n_assets=20)
    for bad in (1.0, -1.0, np.nan, [0.1, 0.1]):
        with pytest.raises(ValueError, match="ic must"):
            factor_forecast(realized, bad)
    with pytest.raises(ValueError, match="scale explicitly"):
        factor_forecast(realized, 0.0)
    for bad in (0, 20, 1.5, True):
        with pytest.raises(ValueError, match="n_factors must"):
            factor_forecast(realized, 0.1, n_factors=bad)
    for bad in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="factor_share must"):
            factor_forecast(realized, 0.1, factor_share=bad)
    for bad in (-0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match="ic_persistence must"):
            factor_forecast(realized, 0.1, ic_persistence=bad)
    for bad in (-0.01, np.inf, np.nan):
        with pytest.raises(ValueError, match="ic_dispersion must"):
            factor_forecast(realized, 0.1, ic_dispersion=bad)
    with pytest.raises(ValueError, match="quality must"):
        factor_forecast(realized, 0.1, quality="size")
    for bad in (-0.1, 2.0, np.nan):
        with pytest.raises(ValueError, match="quality_strength must"):
            factor_forecast(realized, 0.1, quality="vol", quality_strength=bad)
    for bad in (0, -2.0, np.nan, True):
        with pytest.raises(ValueError, match="half_life must"):
            factor_forecast(realized, 0.1, half_life=bad)
    with pytest.raises(ValueError, match="days must"):
        factor_forecast(realized, 0.1, days=np.arange(9))
    # Spread quality needs quotes on the realized grid.
    with pytest.raises(ValueError, match="needs market"):
        factor_forecast(realized, 0.1, quality="spread")
    with pytest.raises(MissingRolesError, match="half_spread"):
        factor_forecast(realized, 0.1, quality="spread",
                        market=MarketData.empty(realized.decisions, realized.assets))
    with pytest.raises(ValueError, match="not aligned"):
        factor_forecast(realized, 0.1, quality="spread",
                        market=MarketData.empty(realized.decisions, realized.assets[::-1]))
    # The history must be on the same names, with enough cross-sections.
    with pytest.raises(ValueError, match="not aligned"):
        factor_forecast(realized, 0.1, history=iid_panel(n_decisions=10, n_assets=21))
    with pytest.raises(TypeError, match="ForwardReturns"):
        factor_forecast(realized, 0.1, history=realized.values[:, :, 0])
    with pytest.raises(ValueError, match="too few"):
        factor_forecast(realized, 0.1, history=iid_panel(n_decisions=3, n_assets=20), n_factors=3)
    # Off switches need no inputs: no market without spread quality, any
    # strength without quality, an explicit scale admits a zero target.
    factor_forecast(realized, 0.0, scale=1.0, n_factors=5, factor_share=1.0,
                    ic_persistence=1.0, quality_strength=0.0, half_life=2.5, days=np.zeros(10),
                    seed=0)
