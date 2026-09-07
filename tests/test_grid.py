from __future__ import annotations

import numpy as np

from stable_finance.dataset import sparse_to_dense_grid


FFILL = list(range(7))
ZERO_FILL = [7, 8]


def _grid(timestamps, features, *, trim=True, dtype=np.float64):
    return sparse_to_dense_grid(
        np.asarray(timestamps, dtype=np.int32),
        np.asarray(features, dtype=np.float32),
        100,
        110,
        FFILL,
        ZERO_FILL,
        trim_leading_nan=trim,
        dtype=dtype,
    )


def test_empty_sparse_input_has_no_session():
    assert _grid([], np.empty((0, 9))) is None


def test_sparse_rows_are_placed_filled_and_typed():
    features = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [11, 12, 13, 14, 15, 16, 17, 18, 19],
    ])
    seconds, dense = _grid([100, 105], features, trim=False, dtype=np.float32)
    np.testing.assert_array_equal(seconds, np.arange(100, 110, dtype=np.int32))
    assert dense.dtype == np.float32
    np.testing.assert_array_equal(dense[1:5, :7], np.tile(features[0, :7], (4, 1)))
    np.testing.assert_array_equal(dense[1:5, 7:], 0.0)
    np.testing.assert_array_equal(dense[5], features[1])


def test_last_preopen_row_only_seeds_forward_filled_columns():
    features = np.array([
        [1, 2, 3, 4, 5, 6, 7, 80, 90],
        [11, 12, 13, 14, 15, 16, 17, 18, 19],
    ])
    _, dense = _grid([99, 105], features, trim=False)
    np.testing.assert_array_equal(dense[0, :7], features[0, :7])
    np.testing.assert_array_equal(dense[0, 7:], 0.0)


def test_leading_missing_quotes_are_retained_or_trimmed_on_request():
    features = np.ones((1, 9))
    _, full = _grid([105], features, trim=False)
    seconds, trimmed = _grid([105], features, trim=True)
    assert np.isnan(full[0, :7]).all()
    assert seconds[0] == 105
    assert not np.isnan(trimmed[0]).any()


def test_postclose_observations_are_ignored():
    features = np.array([
        np.arange(9),
        np.arange(9) + 10,
        np.arange(9) + 20,
    ])
    _, dense = _grid([99, 105, 111], features, trim=False)
    assert dense[0, 0] == features[0, 0]
    assert dense[5, 0] == features[1, 0]
    assert dense[-1, 0] != features[2, 0]
