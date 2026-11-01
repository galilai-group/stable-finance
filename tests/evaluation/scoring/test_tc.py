"""The transfer coefficient: one for the frictionless optimum, lower for anything else."""

import numpy as np
import pytest

from stable_finance.evaluation.scoring.tc import transfer_coefficient


def correlated_covariance(n_assets=8, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_assets, 3))
    volatility = rng.uniform(0.01, 0.05, size=n_assets)
    sigma = loadings @ loadings.T + np.diag(rng.uniform(0.5, 1.5, size=n_assets))
    scale = volatility / np.sqrt(np.diag(sigma))
    return sigma * np.outer(scale, scale)


def test_the_unconstrained_mean_variance_book_transfers_the_whole_forecast():
    sigma = correlated_covariance()
    alpha = np.random.default_rng(1).normal(size=(20, 8))
    weights = np.linalg.solve(sigma, alpha.T).T
    result = transfer_coefficient(weights, alpha, sigma)
    np.testing.assert_allclose(result.per_decision, 1.0, atol=1e-10)
    assert result.mean == pytest.approx(1.0, abs=1e-10)


def test_a_diagonal_covariance_compares_volatility_scaled_weights_with_scaled_alpha():
    rng = np.random.default_rng(2)
    volatility = rng.uniform(0.01, 0.05, size=6)
    weights = rng.normal(size=(5, 6))
    alpha = rng.normal(size=(5, 6))
    result = transfer_coefficient(weights, alpha, np.diag(volatility ** 2))
    expected = [(w @ a) / np.sqrt(np.sum((volatility * w) ** 2) * np.sum((a / volatility) ** 2))
                for w, a in zip(weights, alpha)]
    np.testing.assert_allclose(result.per_decision, expected, atol=1e-12)


def test_the_block_inverse_matches_inverting_the_kept_block_directly():
    sigma = correlated_covariance(seed=9)
    rng = np.random.default_rng(9)
    weights, alpha = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
    weights[0, 2] = alpha[1, 5] = alpha[2, [0, 1, 2, 3, 4]] = np.nan
    result = transfer_coefficient(weights, alpha, sigma)
    for t in range(3):
        keep = np.isfinite(weights[t]) & np.isfinite(alpha[t])
        w, a, block = weights[t, keep], alpha[t, keep], sigma[np.ix_(keep, keep)]
        expected = (w @ a) / np.sqrt((w @ block @ w) * (a @ np.linalg.solve(block, a)))
        assert result.per_decision[t] == pytest.approx(expected, abs=1e-10)


def test_rescaling_the_weights_or_the_forecast_leaves_the_coefficient_unchanged():
    sigma = correlated_covariance(seed=3)
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(10, 8))
    alpha = rng.normal(size=(10, 8))
    base = transfer_coefficient(weights, alpha, sigma).per_decision
    scaled = transfer_coefficient(7.5 * weights, 0.002 * alpha, sigma).per_decision
    np.testing.assert_allclose(scaled, base, atol=1e-12)


def test_a_decision_with_fewer_than_three_usable_names_is_nan():
    rng = np.random.default_rng(4)
    weights = rng.normal(size=(3, 4))
    alpha = rng.normal(size=(3, 4))
    alpha[0, :2] = np.nan
    eligible = np.ones((3, 4), bool)
    eligible[1, 1:] = False
    result = transfer_coefficient(weights, alpha, np.eye(4), eligible=eligible)
    assert np.isnan(result.per_decision[0]) and np.isnan(result.per_decision[1])
    assert np.isfinite(result.per_decision[2])
    assert result.mean == result.per_decision[2]
    assert np.isnan(result.standard_error)


def test_a_flat_book_has_no_transfer_coefficient():
    result = transfer_coefficient(np.zeros((2, 5)), np.ones((2, 5)) * np.arange(5), np.eye(5))
    assert np.all(np.isnan(result.per_decision)) and np.isnan(result.mean)


def test_an_equal_weight_long_short_book_on_heterogeneous_risk_transfers_less_than_all():
    rng = np.random.default_rng(5)
    volatility = np.linspace(0.005, 0.06, 10)
    alpha = rng.normal(size=(30, 10))
    ranks = alpha.argsort(axis=1).argsort(axis=1)
    weights = np.where(ranks >= 7, 1 / 6, np.where(ranks <= 2, -1 / 6, 0.0))
    result = transfer_coefficient(weights, alpha, np.diag(volatility ** 2))
    assert result.mean < 0.9
    assert np.isfinite(result.standard_error) and result.standard_error > 0


def test_a_subset_of_names_is_scored_against_its_own_block_of_the_covariance():
    sigma = correlated_covariance(seed=6)
    rng = np.random.default_rng(6)
    alpha = rng.normal(size=(1, 8))
    keep = np.r_[np.ones(5, bool), np.zeros(3, bool)]
    weights = np.full((1, 8), np.nan)
    weights[0, keep] = np.linalg.solve(sigma[np.ix_(keep, keep)], alpha[0, keep])
    result = transfer_coefficient(weights, alpha, sigma)
    assert result.per_decision[0] == pytest.approx(1.0, abs=1e-10)


def test_misshapen_inputs_are_refused():
    with pytest.raises(ValueError, match="equally shaped"):
        transfer_coefficient(np.ones((2, 3)), np.ones((2, 4)), np.eye(3))
    with pytest.raises(ValueError, match="covariance"):
        transfer_coefficient(np.ones((2, 3)), np.ones((2, 3)), np.eye(4))
