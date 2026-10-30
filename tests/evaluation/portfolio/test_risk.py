"""Risk management: the universe screens and covariance estimation."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance import ForwardReturns, estimate_covariance, liquidity_screen
from stable_finance.evaluation.portfolio.risk import (
    RISK_MODELS,
    diagonal_covariance,
    estimate_risk,
    factor_covariance,
    ledoit_wolf_covariance,
    nonlinear_shrinkage_covariance,
)


def history(seed=0, n_decisions=200, n_assets=3, horizons=(300, 900)):
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.01, size=(n_decisions, n_assets, len(horizons)))
    return ForwardReturns(
        values, np.arange(n_decisions), [f"A{i}" for i in range(n_assets)], horizons
    )


def test_covariance_matches_numpy_on_a_complete_panel_without_shrinkage():
    returns = history()
    got = estimate_covariance(returns, shrinkage=0.0)
    for horizon in range(2):
        expected = np.cov(returns.values[:, :, horizon], rowvar=False)
        np.testing.assert_allclose(got[horizon], expected, rtol=1e-6, atol=1e-12)


def test_covariance_is_estimated_pairwise_and_shrunk_toward_the_diagonal():
    complete = history()
    values = complete.values.copy()
    values[:50, 0, :] = np.nan
    returns = ForwardReturns(
        values, complete.decisions, complete.assets, complete.horizons
    )
    full = estimate_covariance(returns, shrinkage=0.0)
    shrunk = estimate_covariance(returns, shrinkage=1.0)
    assert np.isfinite(full).all()
    np.testing.assert_allclose(np.diag(shrunk[0]), np.diag(full[0]))
    assert np.allclose(shrunk[0] - np.diag(np.diag(shrunk[0])), 0.0)
    # Eigenvalues stay non-negative after the pairwise estimate.
    assert np.linalg.eigvalsh(full[0]).min() >= 0


def test_covariance_rejects_assets_without_enough_observations():
    values = history(n_decisions=5).values.copy()
    values[:, 1, :] = np.nan
    returns = ForwardReturns(values, np.arange(5), ["A0", "A1", "A2"], [300, 900])
    with pytest.raises(ValueError, match="finite returns"):
        estimate_covariance(returns)


# ── nonlinear shrinkage ─────────────────────────────────────────────────────

def correlated_history(seed=0, n_decisions=120, n_assets=20, horizons=(300,)):
    """Returns with a spread-out true spectrum, so shrinkage has work to do."""
    rng = np.random.default_rng(seed)
    scales = np.linspace(0.005, 0.03, n_assets)
    basis, _ = np.linalg.qr(rng.normal(size=(n_assets, n_assets)))
    loadings = basis * scales
    values = np.stack(
        [rng.normal(size=(n_decisions, n_assets)) @ loadings.T for _ in horizons],
        axis=2,
    )
    return ForwardReturns(
        values, np.arange(n_decisions), [f"A{i}" for i in range(n_assets)], horizons
    )


def test_nonlinear_shrinkage_is_symmetric_and_positive_definite():
    sigma = nonlinear_shrinkage_covariance(correlated_history())
    assert sigma.shape == (1, 20, 20)
    np.testing.assert_allclose(sigma[0], sigma[0].T, atol=1e-15)
    assert np.linalg.eigvalsh(sigma[0]).min() > 0


def test_nonlinear_shrinkage_keeps_the_sample_eigenvectors():
    returns = correlated_history()
    sample = np.cov(returns.values[:, :, 0], rowvar=False)
    _, eigenvectors = np.linalg.eigh(sample)
    rotated = eigenvectors.T @ nonlinear_shrinkage_covariance(returns)[0] @ eigenvectors
    off_diagonal = rotated - np.diag(np.diag(rotated))
    assert np.abs(off_diagonal).max() < 1e-8 * np.abs(np.diag(rotated)).max()


def test_nonlinear_shrinkage_pulls_eigenvalues_toward_their_mean():
    returns = correlated_history()
    sample = np.linalg.eigvalsh(np.cov(returns.values[:, :, 0], rowvar=False))
    shrunk = np.linalg.eigvalsh(nonlinear_shrinkage_covariance(returns)[0])
    assert shrunk.std() / shrunk.mean() < sample.std() / sample.mean()
    assert shrunk.max() < sample.max()
    assert shrunk.min() > sample.min()


def test_nonlinear_shrinkage_approaches_the_sample_covariance_on_a_long_history():
    returns = history(n_decisions=20_000, horizons=(300,))
    sample = np.cov(returns.values[:, :, 0], rowvar=False)
    sigma = nonlinear_shrinkage_covariance(returns)[0]
    np.testing.assert_allclose(sigma, sample, atol=0.01 * np.trace(sample) / 3)


def test_nonlinear_shrinkage_is_invertible_with_more_assets_than_decisions():
    returns = correlated_history(n_decisions=30, n_assets=60)
    sigma = nonlinear_shrinkage_covariance(returns)[0]
    assert np.isfinite(sigma).all()
    eigenvalues = np.linalg.eigvalsh(sigma)
    # The sample matrix has rank 29; the estimate is well conditioned.
    assert eigenvalues.min() > 1e-3 * eigenvalues.max()


def test_nonlinear_shrinkage_rejects_a_history_too_short_for_its_bandwidth():
    with pytest.raises(ValueError, match="at least 13 decisions"):
        nonlinear_shrinkage_covariance(history(n_decisions=12))


def test_nonlinear_shrinkage_is_dispatched_by_name():
    returns = correlated_history()
    np.testing.assert_array_equal(
        estimate_risk("nonlinear_shrinkage", returns),
        nonlinear_shrinkage_covariance(returns),
    )


@pytest.mark.parametrize("name", [m for m in RISK_MODELS if m != "embedding"])
def test_every_return_only_risk_model_is_symmetric_positive_definite(name):
    sigma = estimate_risk(name, correlated_history(n_decisions=60, n_assets=8))
    assert sigma.shape == (1, 8, 8)
    np.testing.assert_allclose(sigma[0], sigma[0].T, atol=1e-15)
    assert np.linalg.eigvalsh(sigma[0]).min() > 0


# ── factor model ────────────────────────────────────────────────────────────

def planted_factors(seed=0, n_decisions=2000, n_assets=40, n_factors=3):
    """Returns with a known factor structure, unequal volatilities and specific risk."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_assets, n_factors)) * np.linspace(1.0, 0.4, n_factors)
    loadings[:, 0] += 1.0  # a market factor every name loads on
    vols = rng.uniform(0.5, 2.0, size=n_assets) * 1e-3
    specific = rng.uniform(0.5, 1.5, size=n_assets) ** 2
    factors = rng.normal(size=(n_decisions, n_factors))
    noise = rng.normal(size=(n_decisions, n_assets)) * np.sqrt(specific)
    values = ((factors @ loadings.T + noise) * vols)[:, :, None]
    true = (loadings @ loadings.T + np.diag(specific)) * np.outer(vols, vols)
    returns = ForwardReturns(
        values, np.arange(n_decisions), [f"A{i}" for i in range(n_assets)], [300]
    )
    return returns, true, loadings * vols[:, None]


def minimum_variance(sigma, true):
    """The true variance of the fully invested minimum-variance portfolio under ``sigma``."""
    weights = np.linalg.solve(sigma, np.ones(len(sigma)))
    weights /= weights.sum()
    return float(weights @ true @ weights)


def test_factor_model_is_symmetric_positive_definite():
    for returns in (correlated_history(), correlated_history(n_decisions=30, n_assets=60)):
        sigma = factor_covariance(returns, n_factors=3)
        np.testing.assert_allclose(sigma[0], sigma[0].T, atol=1e-15)
        assert np.linalg.eigvalsh(sigma[0]).min() > 0


def test_factor_model_recovers_a_planted_factor_structure():
    returns, true, exposures = planted_factors()
    sigma = factor_covariance(returns, n_factors=3)[0]
    assert np.linalg.norm(sigma - true) < 0.08 * np.linalg.norm(true)
    # The leading eigenvectors of the fitted correlation span the planted
    # exposures (in correlation units): every principal angle is small.
    scale = np.sqrt(np.diag(sigma))
    _, eigenvectors = np.linalg.eigh(sigma / np.outer(scale, scale))
    planted, _ = np.linalg.qr(exposures / np.sqrt(np.diag(true))[:, None])
    cosines = np.linalg.svd(planted.T @ eigenvectors[:, -3:], compute_uv=False)
    assert cosines.min() > 0.95
    # Too few factors leaves real covariance out.
    one = factor_covariance(returns, n_factors=1)[0]
    assert np.linalg.norm(one - true) > 2 * np.linalg.norm(sigma - true)


def test_factor_model_beats_shrinkage_on_a_short_factor_history():
    # 120 decisions for 80 names: the sample covariance is nearly singular,
    # and the factor structure is what lets the minimum-variance book hedge.
    for seed in range(3):
        returns, true, _ = planted_factors(seed, n_decisions=120, n_assets=80)
        best = minimum_variance(true, true)
        factor = minimum_variance(factor_covariance(returns, n_factors=3)[0], true)
        shrunk = minimum_variance(ledoit_wolf_covariance(returns)[0], true)
        assert factor < 1.15 * best
        assert factor < 0.75 * shrunk


def test_factor_model_with_no_factors_is_the_diagonal_model():
    returns = correlated_history()
    values = returns.values.copy()
    values[:40, 3, 0] = np.nan
    values[70:75, 7, 0] = np.nan
    gappy = ForwardReturns(values, returns.decisions, returns.assets, returns.horizons)
    for panel in (returns, gappy):
        np.testing.assert_allclose(
            factor_covariance(panel, n_factors=0), diagonal_covariance(panel),
            rtol=1e-12, atol=0.0,
        )


def test_factor_model_keeps_every_variance_and_adds_only_covariances():
    returns, _, _ = planted_factors(n_decisions=300)
    values = returns.values.copy()
    values[:50, 0, 0] = np.nan
    gappy = ForwardReturns(values, returns.decisions, returns.assets, returns.horizons)
    np.testing.assert_allclose(
        np.diag(factor_covariance(gappy, n_factors=4)[0]),
        np.diag(diagonal_covariance(gappy)[0]), rtol=1e-10,
    )


def test_industry_factors_capture_a_planted_industry_structure():
    rng = np.random.default_rng(3)
    n_decisions, members = 4000, 30
    codes = np.repeat([10.0, 20.0, 30.0], members)
    strength = np.array([1.0, 0.7, 0.5])
    industry = rng.normal(size=(n_decisions, 3)) * strength
    group = np.repeat(np.arange(3), members)
    values = industry[:, group] + rng.normal(size=(n_decisions, len(codes)))
    returns = ForwardReturns(values[:, :, None] * 1e-3, np.arange(n_decisions),
                             [f"A{i}" for i in range(len(codes))], [300])
    sigma = factor_covariance(returns, n_factors=0, industry=codes)[0]
    scale = np.sqrt(np.diag(sigma))
    correlation = sigma / np.outer(scale, scale)
    share = strength**2 / (strength**2 + 1)
    expected = np.where(group[:, None] == group[None, :], share[group][:, None], 0.0)
    np.fill_diagonal(expected, 1.0)
    # An industry's factor is its members' average, which carries 1/30 of
    # their specific variance: within-industry correlations sit that much high.
    np.testing.assert_allclose(correlation, expected, atol=0.05)


def test_unclassified_and_single_name_industries_get_no_industry_factor():
    returns = correlated_history(n_assets=6)
    codes = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, 3.0])
    sigma = factor_covariance(returns, n_factors=0, industry=codes)[0]
    for name in (0, 5):
        others = np.delete(np.arange(6), name)
        assert np.abs(sigma[name, others]).max() < 1e-12 * sigma[name, name]
    # The two real industries are factors: their members co-move, and so do
    # the industries, whose factor covariance is estimated in full.
    assert sigma[1, 2] != 0.0 and sigma[3, 4] != 0.0 and sigma[1, 3] != 0.0


def test_factor_model_rejects_bad_arguments():
    returns = correlated_history(n_assets=6)
    for bad in (-1, 6, 2.5, True):
        with pytest.raises(ValueError, match="n_factors"):
            factor_covariance(returns, n_factors=bad)
    with pytest.raises(ValueError, match="one code per asset"):
        factor_covariance(returns, industry=np.ones((3, 6)))


def test_factor_model_is_dispatched_by_name():
    returns = correlated_history()
    codes = np.repeat([1.0, 2.0], 10)
    np.testing.assert_array_equal(estimate_risk("factor", returns),
                                  factor_covariance(returns))
    np.testing.assert_array_equal(
        estimate_risk("factor", returns, n_factors=2, industry=codes),
        factor_covariance(returns, n_factors=2, industry=codes),
    )


# ── universe screens ────────────────────────────────────────────────────────

def test_liquidity_screen_keeps_the_tightest_fraction_of_each_decision():
    spread = np.array([[1.0, 2.0, 3.0, 4.0]]) * 1e-4
    kept = liquidity_screen(spread, keep=0.5)
    np.testing.assert_array_equal(kept, [[True, True, False, False]])


def test_liquidity_screen_never_empties_a_non_empty_cross_section():
    spread = np.array([[1e-4, 2e-4]])
    assert liquidity_screen(spread, keep=0.01).sum() == 1


def test_screens_exclude_names_without_a_two_sided_market():
    spread = np.array([[1e-4, np.nan, 2e-4]])
    np.testing.assert_array_equal(
        liquidity_screen(spread, keep=1.0), [[True, False, True]]
    )
