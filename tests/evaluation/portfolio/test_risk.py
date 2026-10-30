"""Risk management: the universe screens and covariance estimation."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance import ForwardReturns, estimate_covariance, liquidity_screen
from stable_finance.evaluation.portfolio.risk import (
    RISK_MODELS,
    estimate_risk,
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
