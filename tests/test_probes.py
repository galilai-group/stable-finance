import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stable_finance import Embeddings, ForwardReturns, RidgeProbe


def panel(seed=0):
    rng = np.random.default_rng(seed)
    n_decisions, n_assets, n_features = 15, 4, 3
    X = rng.normal(size=(n_decisions, n_assets, n_features))
    y = np.stack(
        [2 * X[..., 0] - X[..., 1], X[..., 1] + 0.5 * X[..., 2]], axis=-1
    )
    decisions = np.arange(n_decisions)
    assets = np.array(["A", "B", "C", "D"])
    return (
        Embeddings(X, decisions, assets),
        ForwardReturns(y, decisions, assets, [300, 900]),
    )


def test_ridge_probe_matches_the_existing_scaler_then_ridge_recipe():
    embeddings, targets = panel()
    probe = RidgeProbe(alpha=[1.0, 10.0]).fit(embeddings, targets)
    got = probe.predict(embeddings)

    X = embeddings.values.reshape(-1, 3)
    y = targets.values.reshape(-1, 2)
    scaled = StandardScaler().fit_transform(X)
    expected = np.column_stack([
        Ridge(alpha=1.0).fit(scaled, y[:, 0]).predict(scaled),
        Ridge(alpha=10.0).fit(scaled, y[:, 1]).predict(scaled),
    ]).reshape(15, 4, 2)
    np.testing.assert_allclose(got.values, expected)
    np.testing.assert_array_equal(got.horizons, [300, 900])


def test_ridge_probe_filters_missing_targets_per_horizon():
    embeddings, targets = panel()
    values = targets.values.copy()
    values[:3, :, 1] = np.nan
    targets = ForwardReturns(
        values, targets.decisions, targets.assets, targets.horizons
    )
    predictions = (
        RidgeProbe(min_samples=20).fit(embeddings, targets).predict(embeddings)
    )
    assert np.isfinite(predictions.values).all()


def test_ridge_probe_preserves_missing_embedding_rows_as_nan():
    embeddings, targets = panel()
    probe = RidgeProbe().fit(embeddings, targets)
    values = embeddings.values.copy()
    values[0, 1, 0] = np.nan
    missing = Embeddings(values, embeddings.decisions, embeddings.assets)
    prediction = probe.predict(missing)
    assert np.isnan(prediction.values[0, 1]).all()
    assert np.isfinite(prediction.values[0, 0]).all()


def test_ridge_probe_rejects_misaligned_training_rows():
    embeddings, targets = panel()
    misaligned = ForwardReturns(
        targets.values, targets.decisions, targets.assets[::-1], targets.horizons
    )
    with pytest.raises(ValueError, match="assets are not aligned"):
        RidgeProbe().fit(embeddings, misaligned)


def test_ridge_probe_must_be_fitted_before_prediction():
    embeddings, _ = panel()
    with pytest.raises(NotFittedError):
        RidgeProbe().predict(embeddings)
