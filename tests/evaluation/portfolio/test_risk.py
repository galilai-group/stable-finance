"""Covariance estimation for the portfolio stage's risk models."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance import ForwardReturns, estimate_covariance



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
