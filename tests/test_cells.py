"""A cell built K stocks at a time is the per-view pipeline, stacked."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance.dataset.anchors import SESSION_LEN
from stable_finance.dataset.cells import (
    CellGeometry,
    aggregate_cell,
    anchor_band,
    build_cell_views,
    draw_cell,
)
from stable_finance.dataset.transforms import (
    aggregate,
    build_norm_groups,
    ffill_vwap,
    normalize,
    prior_vwap,
)
from stable_finance.dataset.schema import MARKET_SCHEMA


def _grid(rng, n=5, rows=SESSION_LEN):
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
    g = _grid(rng, n=4, rows=factor * 256)
    g[1, 10:40, 7] = 0            # an empty bucket: vwap NaN, sums zero
    stacked = aggregate_cell(np.ascontiguousarray(g.transpose(0, 2, 1)), factor,
                             use_torch=use_torch)
    for k in range(4):
        single = aggregate(g[k].astype(np.float64), factor)
        np.testing.assert_allclose(stacked[k].T, single, rtol=1e-9, atol=0, equal_nan=True)


def test_build_cell_views_matches_the_per_view_pipeline():
    rng = np.random.default_rng(3)
    grid_rows = _grid(rng, n=6)
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
