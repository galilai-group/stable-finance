"""Signal transforms: alpha refinement, beta estimation and neutralization."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pytest

from stable_finance import ForwardReturns
from stable_finance.dataset.schema import MissingRolesError
from stable_finance.evaluation.portfolio import signals
from stable_finance.evaluation.portfolio.signals import (
    ALPHA_RULES,
    NEUTRALIZATIONS,
    SIGNAL_REQUIRES,
    alpha_from_rule,
    apply_signal,
    estimate_beta,
    industry_exposures,
    neutralize,
    refine_alpha,
)


def forecast(seed=0, n_decisions=6, n_assets=8):
    return np.random.default_rng(seed).normal(size=(n_decisions, n_assets))


def test_zscore_rows_have_zero_mean_and_unit_std_over_eligible_names():
    values = forecast()
    eligible = np.ones(values.shape, dtype=bool)
    eligible[:, :2] = False
    values[0, 3] = np.nan
    z = refine_alpha(values, np.ones(values.shape[1]), ic=1.0, eligible=eligible)
    for decision in range(values.shape[0]):
        row = z[decision][np.isfinite(z[decision])]
        assert row.mean() == pytest.approx(0.0, abs=1e-12)
        assert row.std() == pytest.approx(1.0)
    assert np.isfinite(z[1, 2:]).all()


def test_refine_alpha_with_constant_sigma_preserves_order_and_scales_as_ic_times_sigma():
    values = forecast()
    unit = refine_alpha(values, np.ones(values.shape[1]), ic=1.0)
    scaled = refine_alpha(values, np.full(values.shape[1], 0.02), ic=0.05)
    np.testing.assert_allclose(scaled, 0.05 * 0.02 * unit)
    for decision in range(values.shape[0]):
        np.testing.assert_array_equal(
            np.argsort(scaled[decision]), np.argsort(values[decision]))


def test_negative_ic_flips_the_signal():
    values = forecast()
    sigma = np.full(values.shape[1], 0.01)
    np.testing.assert_allclose(
        refine_alpha(values, sigma, ic=-0.1), -refine_alpha(values, sigma, ic=0.1))


def test_per_decision_volatility_scales_each_name_by_its_own_sigma():
    values = forecast()
    sigma = np.random.default_rng(1).uniform(0.01, 0.05, size=values.shape)
    unit = refine_alpha(values, np.ones(values.shape[1]), ic=0.1)
    np.testing.assert_allclose(refine_alpha(values, sigma, ic=0.1), unit * sigma)


def test_gauss_rank_averages_ties_and_is_symmetric():
    values = np.array([[1.0, 2.0, 2.0, 3.0, 4.0]])
    z = refine_alpha(values, np.ones(5), ic=1.0, score="gauss_rank")[0]
    phi = NormalDist().inv_cdf
    np.testing.assert_allclose(z, [phi(1 / 6), phi(2.5 / 6), phi(2.5 / 6), phi(4 / 6), phi(5 / 6)])
    assert z[1] == z[2]
    symmetric = refine_alpha(np.array([[5.0, 1.0, 3.0, 2.0, 4.0]]), np.ones(5),
                             ic=1.0, score="gauss_rank")[0]
    np.testing.assert_allclose(np.sort(symmetric), -np.sort(symmetric)[::-1])
    assert symmetric[2] == pytest.approx(0.0, abs=1e-12)


def test_ineligible_and_missing_names_are_nan():
    values = forecast()
    values[2, 5] = np.nan
    eligible = np.ones(values.shape, dtype=bool)
    eligible[:, 0] = False
    for score in ("zscore", "gauss_rank"):
        alpha = refine_alpha(values, np.ones(values.shape[1]), ic=0.1,
                             score=score, eligible=eligible)
        assert np.isnan(alpha[:, 0]).all()
        assert np.isnan(alpha[2, 5])
        assert np.isfinite(alpha[:, 1:4]).all()


def test_decisions_without_two_names_or_dispersion_are_nan_rows():
    values = forecast(n_decisions=3)
    values[0, 1:] = np.nan
    values[1] = 7.0
    for score in ("zscore", "gauss_rank"):
        alpha = refine_alpha(values, np.ones(values.shape[1]), ic=0.1, score=score)
        assert np.isnan(alpha[:2]).all()
        assert np.isfinite(alpha[2]).all()


def test_refine_alpha_rejects_bad_options():
    values = forecast()
    with pytest.raises(ValueError, match="unknown score 'rank'"):
        refine_alpha(values, np.ones(values.shape[1]), ic=0.1, score="rank")
    with pytest.raises(ValueError, match="ic must be finite"):
        refine_alpha(values, np.ones(values.shape[1]), ic=float("nan"))
    with pytest.raises(ValueError, match="volatility must have shape"):
        refine_alpha(values, np.ones(3), ic=0.1)


def test_each_alpha_rule_dispatches_to_its_score():
    values = forecast()
    eligible = np.ones(values.shape, dtype=bool)
    eligible[:, 0] = False
    sigma = np.random.default_rng(2).uniform(0.01, 0.05, size=values.shape[1])
    for name, score in (("grinold", "zscore"), ("grinold_rank", "gauss_rank")):
        np.testing.assert_array_equal(
            alpha_from_rule(name, values, sigma, ic=0.05, eligible=eligible),
            refine_alpha(values, sigma, ic=0.05, score=score, eligible=eligible))
    with pytest.raises(ValueError, match="unknown alpha 'barra'"):
        alpha_from_rule("barra", values, sigma, ic=0.05)


def test_the_rank_rule_keeps_one_outlier_from_claiming_the_book():
    values = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 50.0]])
    sigma = np.ones(8)
    plain = alpha_from_rule("grinold", values, sigma, ic=1.0)[0]
    robust = alpha_from_rule("grinold_rank", values, sigma, ic=1.0)[0]
    # The z-score gives the outlier almost the whole dispersion and squeezes
    # the rest together; the Gaussian rank score spaces every name by rank.
    assert plain[-1] > 2.5 and np.ptp(plain[:-1]) < 0.1
    np.testing.assert_allclose(robust, -robust[::-1], atol=1e-12)
    assert robust[-1] == pytest.approx(NormalDist().inv_cdf(8 / 9))


def factor_panel(seed=0, n_decisions=400, betas=(0.5, 0.8, 1.0, 1.2, 1.5)):
    """Returns ``b_i * m + e`` whose equal-weighted mean is exactly ``m``."""
    rng = np.random.default_rng(seed)
    betas = np.asarray(betas)
    assert betas.mean() == pytest.approx(1.0)
    market = rng.normal(scale=0.02, size=(n_decisions, 1, 2))
    noise = rng.normal(scale=0.005, size=(n_decisions, len(betas), 2))
    noise -= noise.mean(axis=1, keepdims=True)
    values = betas[None, :, None] * market + noise
    history = ForwardReturns(values, np.arange(n_decisions),
                             [f"A{i}" for i in range(len(betas))], [300, 900])
    return history, betas


def test_estimate_beta_recovers_known_betas_on_a_factor_panel():
    history, betas = factor_panel()
    got = estimate_beta(history)
    assert got.shape == (2, len(betas))
    np.testing.assert_allclose(got, np.broadcast_to(betas, got.shape), atol=0.05)
    # Betas on the equal-weighted market average exactly one.
    np.testing.assert_allclose(got.mean(axis=1), 1.0)


def test_estimate_beta_is_nan_without_two_observations_or_market_variance():
    history, _ = factor_panel(n_decisions=30)
    values = history.values.copy()
    values[1:, 0, :] = np.nan
    values[:, :, 1] = 0.01
    sparse = ForwardReturns(values, history.decisions, history.assets, history.horizons)
    got = estimate_beta(sparse)
    assert np.isnan(got[0, 0]) and np.isfinite(got[0, 1:]).all()
    assert np.isnan(got[1]).all()


def test_neutralize_residuals_are_orthogonal_to_each_exposure_and_sum_to_zero():
    values = forecast(n_assets=20)
    exposures = np.random.default_rng(3).normal(size=(values.shape[0], 20, 2))
    eligible = np.ones(values.shape, dtype=bool)
    eligible[:, -3:] = False
    exposures[1, 4, 0] = np.nan
    residual = neutralize(values, exposures, eligible=eligible)
    assert np.isnan(residual[:, -3:]).all()
    assert np.isnan(residual[1, 4])
    for decision in range(values.shape[0]):
        used = np.isfinite(residual[decision])
        assert residual[decision, used].sum() == pytest.approx(0.0, abs=1e-12)
        for k in range(2):
            assert residual[decision, used] @ exposures[decision, used, k] == pytest.approx(
                0.0, abs=1e-10)


def test_neutralize_broadcasts_static_exposures_over_decisions():
    values = forecast()
    static = np.random.default_rng(4).normal(size=(values.shape[1], 1))
    tiled = np.broadcast_to(static, (*values.shape, 1))
    np.testing.assert_allclose(neutralize(values, static), neutralize(values, tiled))


def test_industry_neutralization_demeans_within_each_industry():
    values = forecast(n_assets=9)
    codes = np.tile([1.0, 1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 5.0, np.nan], (values.shape[0], 1))
    residual = apply_signal("industry", values, eligible=np.ones(values.shape, bool),
                            industry=codes)
    assert np.isnan(residual[:, 8]).all()
    for members in ([0, 1, 2], [3, 4], [5, 6, 7]):
        expected = values[:, members] - values[:, members].mean(axis=1, keepdims=True)
        np.testing.assert_allclose(residual[:, members], expected, atol=1e-12)


def test_industry_exposures_keep_every_dummy_and_blank_unknown_names():
    dummies = industry_exposures(np.array([[3.0, 1.0, np.nan], [1.0, 1.0, 3.0]]))
    assert dummies.shape == (2, 3, 2)
    np.testing.assert_array_equal(dummies[0, 0], [0.0, 1.0])
    np.testing.assert_array_equal(dummies[1, 1], [1.0, 0.0])
    assert np.isnan(dummies[0, 2]).all()


def test_rank_deficient_exposures_do_not_crash():
    values = forecast()
    column = np.random.default_rng(5).normal(size=(values.shape[1], 1))
    exposures = np.concatenate([column, 2 * column, np.ones_like(column)], axis=1)
    residual = neutralize(values, exposures)
    np.testing.assert_allclose(residual, neutralize(values, column), atol=1e-10)


def test_beta_industry_is_orthogonal_to_beta_and_every_industry():
    values = forecast(n_assets=12)
    beta = np.random.default_rng(6).uniform(0.5, 1.5, size=12)
    codes = np.tile(np.repeat([1.0, 2.0, 3.0], 4), (values.shape[0], 1))
    residual = apply_signal("beta_industry", values, eligible=np.ones(values.shape, bool),
                            beta=beta, industry=codes)
    np.testing.assert_allclose(residual @ beta, 0.0, atol=1e-10)
    for code in (1.0, 2.0, 3.0):
        np.testing.assert_allclose(residual[:, codes[0] == code].sum(axis=1), 0.0, atol=1e-10)


def test_apply_signal_names_its_missing_inputs():
    values = forecast()
    eligible = np.ones(values.shape, bool)
    with pytest.raises(ValueError, match="needs beta"):
        apply_signal("beta", values, eligible=eligible)
    with pytest.raises(MissingRolesError, match="industry"):
        apply_signal("industry", values, eligible=eligible)
    with pytest.raises(ValueError, match="unknown neutralization 'sector'"):
        apply_signal("sector", values, eligible=eligible)


def test_signal_constants_cover_every_option():
    assert ALPHA_RULES == ("grinold", "grinold_rank")
    assert set(SIGNAL_REQUIRES) == set(NEUTRALIZATIONS)
    # The scores are an argument of refine_alpha, not a configuration axis.
    assert "ALPHA_SCORES" not in signals.__all__
    assert "alpha_from_rule" in signals.__all__
    assert signals.__all__ == sorted(signals.__all__)


def test_outputs_are_deterministic():
    values = forecast(n_assets=12)
    history, _ = factor_panel()
    codes = np.tile(np.repeat([1.0, 2.0, 3.0], 4), (values.shape[0], 1))
    beta = np.linspace(0.5, 1.5, 12)
    eligible = np.ones(values.shape, bool)
    for run in (
        lambda: refine_alpha(values, np.ones(12), ic=0.1, score="gauss_rank"),
        lambda: refine_alpha(values, np.ones(12), ic=0.1),
        lambda: alpha_from_rule("grinold_rank", values, np.ones(12), ic=0.1),
        lambda: estimate_beta(history),
        lambda: apply_signal("beta_industry", values, eligible=eligible,
                             beta=beta, industry=codes),
    ):
        np.testing.assert_array_equal(run(), run())
