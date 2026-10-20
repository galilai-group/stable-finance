from __future__ import annotations

import numpy as np
import pytest

from stable_finance.dataset import (
    MARKET_SCHEMA,
    FixedSession,
    MarketSession,
    SessionPreprocessor,
    Timeline,
    sparse_to_dense_grid,
    timeline_bounds_est,
)


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


def test_sparse_grid_preserves_the_existing_fill_contract():
    timestamps = np.array([99, 101, 103], dtype=np.int32)
    features = np.array([
        [9, 9, 9, 9, 10, 1, 1, 7, 1],
        [10, 10, 10, 10, 11, 2, 2, 5, 1],
        [12, 12, 12, 12, 13, 3, 3, 6, 1],
    ], dtype=np.float32)
    seconds, dense = sparse_to_dense_grid(
        timestamps, features, 100, 105,
        MARKET_SCHEMA.forward_fill_indices,
        MARKET_SCHEMA.zero_fill_indices,
        trim_leading_nan=False,
    )
    np.testing.assert_array_equal(seconds, np.arange(100, 105, dtype=np.int32))
    np.testing.assert_array_equal(dense[:, 0], [9, 10, 10, 12, 12])
    np.testing.assert_array_equal(dense[:, 7], [0, 5, 0, 6, 0])


def test_session_preprocessor_returns_backend_neutral_session():
    start, _ = timeline_bounds_est("2023-01-03")
    raw = {
        "ticker": "ABC",
        "date": "2023-01-03",
        "ts_interval": np.array([start, start + 1], dtype=np.int32),
        "features": np.ones((2, 9), dtype=np.float32),
    }
    session = SessionPreprocessor().transform(raw)
    assert isinstance(session, MarketSession)
    assert session.ticker == "ABC"
    assert session.features.dtype == np.float64


def test_session_preprocessor_owns_bounded_cache_and_telemetry():
    start, _ = timeline_bounds_est("2023-01-03")
    raw = {
        "ticker": "ABC",
        "date": "2023-01-03",
        "ts_interval": np.array([start, start + 1], dtype=np.int32),
        "features": np.ones((2, 9), dtype=np.float32),
    }
    processor = SessionPreprocessor(cache_bytes=1 << 20)
    processor.transform(raw)
    processor.transform(raw)
    info = processor.cache_info()
    assert (info.hits, info.misses, info.entries) == (1, 1, 1)
    assert 0 < info.bytes <= info.max_bytes


def test_preprocessor_refuses_calendar_and_schedule_together(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("date,status,start_time,end_time\n")
    from stable_finance.dataset import MarketSchedule

    with pytest.raises(ValueError):
        SessionPreprocessor(timeline=Timeline(FixedSession()), schedule=MarketSchedule(csv))
