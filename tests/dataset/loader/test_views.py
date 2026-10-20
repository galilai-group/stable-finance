"""View geometry, metadata, aggregation and normalization."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance.dataset import (
    MARKET_SCHEMA,
    MarketSession,
    ViewMetadata,
    ViewSpec,
    aggregate,
    build_norm_groups,
    normalize,
    resample_session,
)



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
