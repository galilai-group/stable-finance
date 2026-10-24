"""Sessions to model inputs: grid rebuild, views, panels, and training cells."""
from __future__ import annotations

import numpy as np
import pytest
from helpers import IdentityStats as _IdentityStats

from stable_finance.dataset import (
    MARKET_SCHEMA,
    FeatureSchema,
    MarketSession,
    SessionPreprocessor,
    ViewMetadata,
    ViewSpec,
    aggregate,
    build_norm_groups,
    build_session_panel,
    choose_cell_aggregation,
    normalize,
    resample_session,
    sparse_to_dense_grid,
    timeline_bounds_est,
)
from stable_finance.dataset.anchors import SESSION_LEN
from stable_finance.dataset.datasets.market1t.spec import standard_open_est
from stable_finance.dataset.datasets.market1t.views import (
    CellGeometry,
    aggregate_cell,
    anchor_band,
    build_cell_views,
    draw_cell,
    ffill_vwap,
    prior_vwap,
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


def _session():
    n = 300
    values = np.zeros((n, len(MARKET_SCHEMA.columns)), dtype=np.float64)
    values[:, 0] = np.linspace(100, 101, n)
    values[:, 1] = values[:, 0]
    values[:, 2] = values[:, 0] + 0.1
    values[:, 3] = values[:, 0] - 0.1
    values[:, 4] = values[:, 0] + 0.02
    values[:, 5:7] = 100
    values[:, 7:] = 10
    start = standard_open_est("2023-01-03")
    return MarketSession(
        "A", "2023-01-03", np.arange(start, start + n), values,
    )


def test_panel_keeps_metadata_separate_from_view_columns():
    rows = build_session_panel(
        _session(), [150], _IdentityStats(),
        view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)),
        target_types=("return",),
        horizons=(10,),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.view.shape == (4, len(MARKET_SCHEMA.columns))
    assert row.metadata.start_seconds == 147
    assert row.metadata.end_seconds == 151
    assert row.metadata.aggregation_seconds == 1


def test_cell_aggregation_is_reproducible_and_feasible():
    first = choose_cell_aggregation("2023-01-03", 20, 20, 4, choices=(3, 4, 8))
    second = choose_cell_aggregation("2023-01-03", 20, 20, 4, choices=(3, 4, 8))
    assert first == second
    assert first in (3, 4)


def test_panel_carries_the_natural_unit_quote_at_the_decision():
    """The executable price cannot be recovered from the view.

    ``prepare_view`` standardizes each observation against its own window, so
    the bid and ask inside ``row.view`` are in units that differ per row. The
    execution stage charges the spread that was actually quoted, so the quote
    rides alongside the view in natural units instead.
    """
    session = _session()
    rows = build_session_panel(
        session, [150], _IdentityStats(),
        view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)),
        target_types=("return",),
        horizons=(10,),
    )

    bid = session.features[150, MARKET_SCHEMA.index("bid_price")]
    ask = session.features[150, MARKET_SCHEMA.index("ask_price")]
    assert rows[0].quote.tolist() == [bid, ask]
    # The DECISION row, not the window's first or last bar: bid_price rises
    # monotonically here, so reading the wrong end of the view would show up.
    assert rows[0].quote[0] != session.features[147, MARKET_SCHEMA.index("bid_price")]


def test_quote_columns_follow_the_session_schema_rather_than_a_constant():
    """A backend that reorders its columns must move the quote with it.

    Indices read off a module-level constant would silently return two other
    features here, and the execution stage would price every fill against
    whatever happened to sit in those positions.
    """
    reversed_columns = tuple(reversed(MARKET_SCHEMA.columns))
    schema = FeatureSchema(
        columns=reversed_columns,
        forward_fill=MARKET_SCHEMA.forward_fill,
        zero_fill=MARKET_SCHEMA.zero_fill,
        aggregation=tuple(
            (name, MARKET_SCHEMA.aggregation_rules[name]) for name in reversed_columns
        ),
        normalization=MARKET_SCHEMA.normalization,
    )
    original = _session()
    session = MarketSession(
        original.ticker, original.date, original.timestamps,
        original.features[:, ::-1].copy(), schema,
    )
    rows = build_session_panel(
        session, [150], _IdentityStats(),
        view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)),
        target_types=("return",),
        horizons=(10,),
    )

    assert rows[0].quote.tolist() == [
        session.features[150, schema.index("bid_price")],
        session.features[150, schema.index("ask_price")],
    ]
    # And that is the same market the default-ordered session reported.
    assert rows[0].quote.tolist() == [
        original.features[150, MARKET_SCHEMA.index("bid_price")],
        original.features[150, MARKET_SCHEMA.index("ask_price")],
    ]


DATE = "2024-03-05"          # a Tuesday


def _univariate_rows(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return (100 * np.exp(rng.normal(0, 1e-3, n).cumsum()))[:, None]


def test_panel_from_a_univariate_session_has_no_quote():
    start = standard_open_est(DATE)
    session = MarketSession("A", DATE, np.arange(start, start + 400), _univariate_rows(400),
                            FeatureSchema.univariate())
    rows = build_session_panel(session, [150], _IdentityStats(),
                               view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)),
                               target_types=("return",), horizons=(10,))
    assert rows and rows[0].quote is None and rows[0].view.shape == (4, 1)


def _cell_grid(rng, n=5, rows=SESSION_LEN):
    g = np.empty((n, rows, 9), dtype=np.float32)
    for i in range(n):
        mid = 50 + 10 * i + rng.normal(0, 0.02, rows).cumsum()
        g[i, :, 0], g[i, :, 4] = mid - 0.01, mid + 0.01
        vol = rng.poisson(20, rows) * (rng.random(rows) < 0.5)
        g[i, :, 7], g[i, :, 8] = vol, np.minimum(vol, 2)
        g[i, :, 1] = np.where(vol > 0, mid, np.nan)
        idx = np.arange(rows); src = np.where(vol > 0, idx, 0)
        np.maximum.accumulate(src, out=src)
        g[i, :, 1] = np.where(np.isnan(g[i, src, 1]), mid, g[i, src, 1])
        g[i, :, 2], g[i, :, 3] = mid + 0.02, mid - 0.02
        g[i, :, 5], g[i, :, 6] = rng.poisson(100, rows) + 1, rng.poisson(100, rows) + 1
    return g


@pytest.mark.parametrize("factor", [1, 6, 7, 11])
@pytest.mark.parametrize("use_torch", [False, True])
def test_aggregate_cell_matches_the_per_view_aggregate(factor, use_torch):
    if use_torch:
        pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    g = _cell_grid(rng, n=4, rows=factor * 256)
    g[1, 10:40, 7] = 0            # an empty bucket: vwap NaN, sums zero
    stacked = aggregate_cell(np.ascontiguousarray(g.transpose(0, 2, 1)), factor,
                             use_torch=use_torch)
    for k in range(4):
        single = aggregate(g[k].astype(np.float64), factor)
        np.testing.assert_allclose(stacked[k].T, single, rtol=1e-9, atol=0, equal_nan=True)


def test_build_cell_views_matches_the_per_view_pipeline():
    rng = np.random.default_rng(3)
    grid_rows = _cell_grid(rng, n=6)
    groups = build_norm_groups(list(MARKET_SCHEMA.columns))
    geometry = CellGeometry(sequence_length=256, aggregation_seconds=(6, 11))
    draw = draw_cell(np.random.RandomState(5), geometry, SESSION_LEN, 0)
    idx = np.array([4, 0, 2])
    # Zero-volume stretches at the window start exercise the prior lookup.
    grid_rows[idx, draw.start_row:draw.start_row + 40, 7] = 0
    grid_rows[idx, draw.start_row:draw.start_row + 40, 1] = np.nan
    grid = np.ascontiguousarray(grid_rows.transpose(0, 2, 1))     # (N, C, rows)
    views, stats = build_cell_views(grid, idx, draw, groups=groups)
    assert views.shape == (3, 9, 256) and stats.shape == (3, len(groups), 2)
    for k, i in enumerate(idx):
        f = grid_rows[i].astype(np.float64)
        v = aggregate(f[draw.start_row:draw.start_row + draw.window], draw.aggregation)
        ffill_vwap(v, prior_vwap(f, draw.start_row))
        s = []
        normalize(v, groups, stats_out=s)
        np.nan_to_num(v, copy=False, nan=0.0)
        np.testing.assert_allclose(views[k].T, v, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(stats[k], np.asarray(s), rtol=1e-9)


def test_the_draw_is_the_panels_draw():
    geometry = CellGeometry(sequence_length=2048, scale_range=(0.5, 1.0), slack_seconds=960)
    first, last, a_lo, a_hi = anchor_band(geometry, SESSION_LEN, 0)
    assert (first, a_lo, a_hi) == (12300, 6, 11) and last == SESSION_LEN - 1 - 960
    rng = np.random.RandomState(0)
    draws = [draw_cell(rng, geometry, SESSION_LEN, 0) for _ in range(4000)]
    assert all(d is not None for d in draws)
    anchors = np.array([d.anchor_tod for d in draws])
    assert anchors.min() == 12300 and anchors.max() == 22200
    assert all(d.anchor_tod % 300 == 0 for d in draws)
    assert all(d.window == d.aggregation * 2048 and d.start_row == d.anchor_tod - d.window + 1
               for d in draws)
    # 11 s/token needs the last 15 minutes the slack reserves.
    assert max(d.aggregation for d in draws) == 10
    lo, hi = anchors.min(), anchors.max()
    third = (hi - lo) / 3
    assert abs(np.mean(anchors < lo + third) - np.mean(anchors > hi - third)) < 0.06
    # A half day at a late open shifts the band and the resolution together.
    first, last, a_lo, a_hi = anchor_band(CellGeometry(), 12_600, 0)
    assert (a_lo, a_hi, first, last) == (3, 6, 6300, 12_599)
    assert CellGeometry.slack_for([900]) == 960 and CellGeometry.slack_for([300, 900]) == 0
