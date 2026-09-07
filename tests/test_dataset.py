import datetime as dt

import numpy as np
import pytest

from stable_finance.dataset import (
    MARKET_SCHEMA,
    MarketSession,
    Month,
    SessionPreprocessor,
    ViewMetadata,
    ViewSpec,
    aggregate,
    build_norm_groups,
    infer_period_frequency,
    next_month,
    normalize,
    period_directories,
    resample_session,
    sparse_to_dense_grid,
    standard_open_est,
    timeline_bounds_est,
)


def test_month_is_a_first_class_fit_period():
    month = Month.parse("2020-12")
    assert month.first_day == dt.date(2020, 12, 1)
    assert month.last_day == dt.date(2020, 12, 31)
    assert next_month(month) == "2021-01"


@pytest.mark.parametrize(
    ("date", "open_utc", "close_utc"),
    [
        ("2023-03-10", "14:30:01", "21:00:01"),
        ("2023-03-13", "13:30:01", "20:00:01"),
        ("2023-11-03", "13:30:01", "20:00:01"),
        ("2023-11-06", "14:30:01", "21:00:01"),
    ],
)
def test_session_bounds_use_the_correct_dst_offset(date, open_utc, close_utc):
    """Pin absolute UTC times, not only the always-6.5-hour session length."""
    start, end = timeline_bounds_est(date)
    as_utc = lambda value: dt.datetime.fromtimestamp(
        value, dt.timezone.utc
    ).strftime("%H:%M:%S")
    assert as_utc(start) == open_utc
    assert as_utc(end) == close_utc
    assert standard_open_est(date) == start


def test_partition_discovery_is_storage_backend_independent(tmp_path):
    root = tmp_path / "1Hz_mosaic_mnth"
    (root / "2020" / "01").mkdir(parents=True)
    (root / "2020" / "03").mkdir(parents=True)
    assert infer_period_frequency(root) == "mnth"
    assert period_directories(root, "2020-01-01", "2020-03-31") == [
        root / "2020" / "01",
        root / "2020" / "03",
    ]


def test_view_geometry_is_validated_as_a_hyperparameter():
    assert ViewSpec().scale_range == (0.5, 1.0)
    with pytest.raises(ValueError, match="scale_range"):
        ViewSpec(scale_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="aggregation_seconds"):
        ViewSpec(aggregation_seconds=(10, 2))


def test_view_metadata_keeps_window_facts_out_of_series_columns():
    metadata = ViewMetadata(
        start_seconds=60,
        end_seconds=180,
        aggregation_seconds=2,
        normalization_means=np.arange(4.0),
        normalization_scales=np.ones(4),
    )
    assert metadata.aggregation_seconds == 2
    with pytest.raises(ValueError, match="scales"):
        ViewMetadata(0, 1, 1, np.zeros(1), np.zeros(1))


def test_supported_schema_owns_column_semantics():
    assert MARKET_SCHEMA.columns[0] == "bid_price"
    assert MARKET_SCHEMA.index("ask_price") == 4
    assert MARKET_SCHEMA.aggregation_rules["vwap_all"] == "vwap"
    assert MARKET_SCHEMA.forward_fill_indices == list(range(7))
    assert MARKET_SCHEMA.zero_fill_indices == [7, 8]


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


def test_aggregation_matches_quote_trade_semantics():
    values = np.array([
        [10, 10, 11, 9, 12, 3, 4, 1, 1],
        [11, 20, 13, 8, 13, 5, 6, 3, 2],
        [20, 30, 21, 19, 22, 7, 8, 2, 3],
    ], dtype=np.float32)
    result = aggregate(values, 2)
    assert result is not None
    np.testing.assert_allclose(result[0], [11, 17.5, 13, 8, 13, 5, 6, 4, 3])
    np.testing.assert_allclose(result[1], values[2])


def test_normalization_uses_the_established_feature_groups():
    values = np.tile(np.arange(1, 10, dtype=np.float64), (4, 1))
    values += np.arange(4)[:, None]
    normalize(values, build_norm_groups(MARKET_SCHEMA.columns))
    for indices, _ in build_norm_groups(MARKET_SCHEMA.columns):
        assert abs(values[:, indices].mean()) < 1e-12


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


def test_market_session_retains_explicit_boundaries_for_future_framing():
    session = MarketSession(
        "ABC", "2023-01-03", np.array([1, 2]), np.ones((2, 9))
    )
    assert session.date == "2023-01-03"
    assert session.bar_seconds == 1
    assert MarketSession(
        "ABC", "2023-01-03", np.array([1, 61]), np.ones((2, 9)),
        bar_seconds=60,
    ).bar_seconds == 60
    with pytest.raises(ValueError, match="strictly increasing"):
        MarketSession("ABC", "2023-01-03", [2, 1], np.ones((2, 9)))


def test_session_can_be_materialized_at_minute_resolution():
    features = np.tile(np.arange(1, 10, dtype=np.float64), (120, 1))
    session = MarketSession(
        "ABC", "2023-01-03", np.arange(120), features,
    )
    minute = resample_session(session, 60)
    assert minute.bar_seconds == 60
    np.testing.assert_array_equal(minute.timestamps, [0, 60])
    assert minute.features.shape == (2, 9)
    with pytest.raises(ValueError, match="integer multiple"):
        resample_session(minute, 90)
