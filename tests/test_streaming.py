import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from stable_finance import ColumnwiseRidge, StreamingRidge


def data(seed=0, n=500, d=12, missing=False):
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
    X, y = data()
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
    X, y = data(seed=3, missing=True)
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
    X, y = data(seed=1)
    answers = []
    for size in (13, 100, len(X)):
        probe = StreamingRidge(alpha=5.0, min_samples=10)
        for start in range(0, len(X), size):
            probe.partial_fit(X[start:start + size], y[start:start + size])
        answers.append(probe.finalize().predict(X))
    for other in answers[1:]:
        np.testing.assert_allclose(answers[0], other, rtol=1e-9, atol=1e-9)


def test_row_count_is_reported():
    X, y = data(n=321)
    probe = StreamingRidge(alpha=1.0, min_samples=10)
    probe.partial_fit(X, y).finalize()
    assert probe.n_samples_ == 321


def test_a_constant_feature_does_not_divide_by_zero():
    X, y = data()
    X[:, 0] = 7.0
    probe = StreamingRidge(alpha=1.0, min_samples=10).partial_fit(X, y).finalize()
    assert np.isfinite(probe.predict(X)).all()


def test_predict_before_finalize_is_an_error():
    X, y = data()
    probe = StreamingRidge(alpha=1.0, min_samples=10).partial_fit(X, y)
    with pytest.raises(NotFittedError):
        probe.predict(X)


def test_partial_fit_after_finalize_is_an_error():
    X, y = data()
    probe = StreamingRidge(alpha=1.0, min_samples=10).partial_fit(X, y).finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        probe.partial_fit(X, y)


def test_too_few_labelled_rows_is_an_error():
    X, y = data(n=30)
    probe = StreamingRidge(alpha=1.0, min_samples=100).partial_fit(X, y)
    with pytest.raises(ValueError, match="need 100"):
        probe.finalize()


def test_feature_count_must_not_change_between_blocks():
    X, y = data()
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
