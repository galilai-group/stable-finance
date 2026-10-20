import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import stable_finance as sf
from stable_finance import (
    ColumnwiseRidge, Embeddings, ForwardReturns, RidgeProbe, StreamingRidge,
)
from stable_finance.dataset import Month


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


def test_ridge_probe_can_fit_from_a_month_provider():
    embeddings, targets = panel()

    class Source:
        def embeddings(self, month):
            assert month == Month(2020, 1)
            return embeddings

        def forward_returns(self, month):
            assert month == Month(2020, 1)
            return targets

    probe = RidgeProbe().fit_month("2020-01", Source())
    assert probe.fit_month_ == Month(2020, 1)
    assert np.isfinite(probe.predict(embeddings).values).all()


def test_ridge_probe_is_an_sklearn_estimator():
    probe = RidgeProbe(alpha=3.0, min_samples=10)
    copied = clone(probe)
    assert copied.get_params() == {"alpha": 3.0, "min_samples": 10}


def test_top_level_fit_maps_embeddings_to_forecasts():
    embeddings, targets = panel()
    fitted = sf.fit(embeddings, targets, alpha=[1.0, 10.0])
    assert isinstance(fitted, RidgeProbe)
    np.testing.assert_array_equal(
        fitted.predict(embeddings).horizons, targets.horizons
    )


def test_columnwise_ridge_accepts_irregular_cached_panels():
    embeddings, targets = panel()
    X = embeddings.values.reshape(-1, 3)
    y = targets.values.reshape(-1, 2)
    y[:5, 1] = np.nan
    model = ColumnwiseRidge(alpha=[1.0, 10.0]).fit(X, y)
    assert model.predict(X).shape == y.shape
    assert clone(model).get_params() == {"alpha": [1.0, 10.0], "min_samples": 20}


# ── the streaming ridge ─────────────────────────────────────────────────────


def streaming_data(seed=0, n=500, d=12, missing=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 4.0, size=d) + rng.normal(size=d)
    y = X @ rng.normal(size=d) + rng.normal(scale=0.5, size=n)
    if missing:
        X[rng.random(n) < 0.05] = np.nan          # unusable feature rows
        y[rng.random(n) < 0.05] = np.nan          # unlabelled rows
    return X, y


@pytest.mark.parametrize("alpha", [0.0, 1.0, 10.0, 1e3])
def test_streaming_matches_the_batch_ridge_exactly(alpha):
    """The whole justification for this estimator: same answer, less memory."""
    X, y = streaming_data()
    batch = ColumnwiseRidge(alpha=alpha, min_samples=10).fit(X, y[:, None])
    stream = StreamingRidge(alpha=alpha, min_samples=10)
    for start in range(0, len(X), 97):            # deliberately ragged blocks
        stream.partial_fit(X[start:start + 97], y[start:start + 97])
    stream.finalize()
    np.testing.assert_allclose(
        stream.predict(X), batch.predict(X)[:, 0], rtol=1e-8, atol=1e-8
    )


def test_streaming_matches_the_batch_ridge_with_missing_values():
    """Scaler rows and regression rows are different sets; both must match."""
    X, y = streaming_data(seed=3, missing=True)
    batch = ColumnwiseRidge(alpha=10.0, min_samples=10).fit(X, y[:, None])
    stream = StreamingRidge(alpha=10.0, min_samples=10)
    for start in range(0, len(X), 64):
        stream.partial_fit(X[start:start + 64], y[start:start + 64])
    stream.finalize()
    got, want = stream.predict(X), batch.predict(X)[:, 0]
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(want)
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-7, atol=1e-7)


def test_block_size_does_not_change_the_answer():
    X, y = streaming_data(seed=1)
    answers = []
    for size in (13, 100, len(X)):
        probe = StreamingRidge(alpha=5.0, min_samples=10)
        for start in range(0, len(X), size):
            probe.partial_fit(X[start:start + size], y[start:start + size])
        answers.append(probe.finalize().predict(X))
    for other in answers[1:]:
        np.testing.assert_allclose(answers[0], other, rtol=1e-9, atol=1e-9)


def test_row_count_is_reported():
    X, y = streaming_data(n=321)
    probe = StreamingRidge(alpha=1.0, min_samples=10)
    probe.partial_fit(X, y).finalize()
    assert probe.n_samples_ == 321


def test_a_constant_feature_does_not_divide_by_zero():
    X, y = streaming_data()
    X[:, 0] = 7.0
    probe = StreamingRidge(alpha=1.0, min_samples=10).partial_fit(X, y).finalize()
    assert np.isfinite(probe.predict(X)).all()


def test_predict_before_finalize_is_an_error():
    X, y = streaming_data()
    probe = StreamingRidge(alpha=1.0, min_samples=10).partial_fit(X, y)
    with pytest.raises(NotFittedError):
        probe.predict(X)


def test_partial_fit_after_finalize_is_an_error():
    X, y = streaming_data()
    probe = StreamingRidge(alpha=1.0, min_samples=10).partial_fit(X, y).finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        probe.partial_fit(X, y)


def test_too_few_labelled_rows_is_an_error():
    X, y = streaming_data(n=30)
    probe = StreamingRidge(alpha=1.0, min_samples=100).partial_fit(X, y)
    with pytest.raises(ValueError, match="need 100"):
        probe.finalize()


def test_feature_count_must_not_change_between_blocks():
    X, y = streaming_data()
    probe = StreamingRidge(alpha=1.0, min_samples=10).partial_fit(X, y)
    with pytest.raises(ValueError, match="must have 12 features"):
        probe.partial_fit(X[:, :5], y)


def test_a_block_of_targets_matches_fitting_each_one_alone():
    """The whole point of the block form: one pass, identical answers.

    Targets are given DIFFERENT missingness so the shared-gram shortcut would
    be caught -- a long forecast horizon runs off the end of a session and is
    absent on rows a short one keeps.
    """
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 7))
    y = np.column_stack([X @ rng.normal(size=7) + rng.normal(size=400)
                         for _ in range(3)])
    y[::5, 1] = np.nan
    y[:60, 2] = np.nan
    alphas = [0.5, 2.0, 8.0]

    block = StreamingRidge(alpha=alphas, min_samples=5)
    for start in range(0, len(X), 90):
        block.partial_fit(X[start:start + 90], y[start:start + 90])
    block.finalize()

    for target, alpha in enumerate(alphas):
        alone = StreamingRidge(alpha=alpha, min_samples=5)
        for start in range(0, len(X), 90):
            alone.partial_fit(X[start:start + 90], y[start:start + 90, target])
        alone.finalize()
        assert block.coefficients_[target] == pytest.approx(
            alone.coefficients_, rel=1e-10)
        assert block.intercept_[target] == pytest.approx(alone.intercept_, rel=1e-10)
        assert block.n_samples_[target] == alone.n_samples_
        assert block.predict(X)[:, target] == pytest.approx(
            alone.predict(X), rel=1e-10, nan_ok=True)


def test_a_single_target_still_returns_one_dimensional_output():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(120, 4))
    probe = StreamingRidge(alpha=1.0, min_samples=5)
    probe.partial_fit(X, X @ rng.normal(size=4)).finalize()
    assert probe.predict(X).shape == (120,)
    assert np.ndim(probe.intercept_) == 0
