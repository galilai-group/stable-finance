"""Cross-sectional cells drawn from a day record, built K stocks at a time.

A CELL is K stocks sharing one (date, anchor, resolution) window: the unit a
cross-sectional rank IC is computed over, and therefore the unit a ranking
loss trains on. This module draws cells the way the evaluation panel draws
its decisions -- anchor uniform over the lattice band that admits a full view
at the finest resolution, then a resolution uniform over those that fit
before it (``panels.evenly_spaced_anchors`` / ``choose_cell_aggregation``) --
and turns the K row slices into normalized views in ONE vectorized pass.

The per-view arithmetic is ``transforms.aggregate``, ``transforms.ffill_vwap``
and ``transforms.normalize`` applied along a leading stock axis: the same
schema rules, the same float64 reductions, the same per-view statistics.
tests/test_cells.py holds the K-stack against the per-view loop to float
precision. What changes is the constant factor: one reshape and a handful of
axis reductions for the cell instead of K Python round trips through the
per-view path, on a loader that gets six CPUs per GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stable_finance.dataset.anchors import (
    ANCHOR_STEP,
    RETURN_VWAP_WINDOW,
    SESSION_LEN,
)
from stable_finance.dataset.schema import MARKET_SCHEMA, FeatureSchema
from stable_finance.dataset.transforms import EPS


@dataclass(frozen=True)
class CellGeometry:
    """How a cell's window is drawn; the training-side twin of ``ViewSpec``.

    ``sequence_length`` tokens at ``aggregation_seconds`` (an inclusive band)
    seconds per token, or -- when that is None -- a resolution band derived
    from ``scale_range`` as a fraction of the session, which is how the
    random-resized-crop recipe states it (0.5-1.0 of 23,400 rows over 2,048
    tokens is 6-11 s/token). ``slack_seconds`` is how much session must
    remain after the anchor: the target horizon plus its measurement window
    for a single-horizon label, zero otherwise.
    """

    sequence_length: int = 2048
    scale_range: tuple[float, float] = (0.5, 1.0)
    aggregation_seconds: tuple[int, int] | None = None
    anchor_step: int = ANCHOR_STEP
    slack_seconds: int = 0

    def __post_init__(self) -> None:
        low, high = self.scale_range
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if not 0 < low <= high <= 1:
            raise ValueError("scale_range must satisfy 0 < low <= high <= 1")
        if self.aggregation_seconds is not None:
            a, b = self.aggregation_seconds
            if not 1 <= a <= b:
                raise ValueError("aggregation_seconds must be positive and ordered")
        if self.anchor_step < 1 or self.slack_seconds < 0:
            raise ValueError("anchor_step must be positive and slack non-negative")

    def resolution_band(self, session_rows: int) -> tuple[int, int]:
        if self.aggregation_seconds is not None:
            return int(self.aggregation_seconds[0]), int(self.aggregation_seconds[1])
        low, high = self.scale_range
        a_lo = max(1, round(low * session_rows / self.sequence_length))
        a_hi = max(a_lo, round(high * session_rows / self.sequence_length))
        return a_lo, a_hi

    @staticmethod
    def slack_for(horizons) -> int:
        """The slack a label set pins: one horizon -> h + its VWAP window."""
        unique = {int(h) for h in horizons}
        return (min(unique) + RETURN_VWAP_WINDOW) if len(unique) == 1 else 0


@dataclass(frozen=True)
class CellDraw:
    anchor_tod: int        # seconds past the standard open; on the lattice
    aggregation: int       # seconds per token
    window: int            # rows = aggregation * sequence_length
    start_row: int         # first row of the window in the day's grid


def anchor_band(geometry: CellGeometry, session_rows: int, open_tod: int) -> tuple[int, int, int, int]:
    """``(first, last, a_lo, a_hi)``: the lattice band and resolution band.

    The first anchor is the first lattice point that admits a full view at
    the FINEST resolution (the panel's filter, ``grid >= a_lo * L - 1``); the
    last leaves ``slack_seconds`` before the close and lies inside the
    session.
    """
    a_lo, a_hi = geometry.resolution_band(session_rows)
    grid = geometry.anchor_step
    first = -(-(a_lo * geometry.sequence_length - 1) // grid) * grid
    last = min(open_tod + session_rows - 1, SESSION_LEN - 1 - geometry.slack_seconds)
    return first, last, a_lo, a_hi


def draw_cell(rng: np.random.RandomState, geometry: CellGeometry,
              session_rows: int, open_tod: int) -> CellDraw | None:
    """Anchor first, resolution second -- the panel's draw.

    Drawing the window first and then an anchor it fits before piles cells
    toward the close (every window fits there; only short ones fit at 13:00),
    and a head trained on that mix scored half its panel IC on the months it
    was measured on. Uniform over the band, then uniform over the feasible
    resolutions, matches where the metric looks. Returns None when nothing
    fits, so the caller redraws.
    """
    first, last, a_lo, a_hi = anchor_band(geometry, session_rows, open_tod)
    grid = geometry.anchor_step
    if first > last:
        return None
    anchor = first + grid * int(rng.randint(0, (last - first) // grid + 1))
    L = geometry.sequence_length
    feasible = [g for g in range(a_lo, a_hi + 1)
                if g * L <= anchor + 1 and anchor - open_tod - g * L + 1 >= 0]
    if not feasible:
        return None
    agg = feasible[int(rng.randint(0, len(feasible)))]
    window = agg * L
    return CellDraw(anchor, agg, window, anchor - open_tod - window + 1)


# ── the K-stack of the per-view transforms ───────────────────────────────────


try:  # the fast path; numpy stays the reference and the fallback
    import torch as _torch
except ModuleNotFoundError:  # pragma: no cover - stable-finance has no torch dependency
    _torch = None


def aggregate_cell(x: np.ndarray, factor: int, *, schema: FeatureSchema = MARKET_SCHEMA,
                   use_torch: bool | None = None) -> np.ndarray:
    """``transforms.aggregate`` over a leading stock axis: ``(K, C, W) -> (K, C, W // factor)``.

    Channel-major, the daystore's layout. ``W`` must be a multiple of
    ``factor`` (a cell's window is ``aggregation * sequence_length`` by
    construction), so there is no partial last bucket to special-case.
    ``last``/``max``/``min`` of float32 values are exact in float32; the sums
    run in float64 like the per-view path. The vwap bucket is
    volume-weighted and NaN where no volume printed.

    TORCH WHEN IT IS THERE. A reduction over a trailing axis of 6-11
    elements is where numpy is weakest -- 1.4 ms per column for a 16-stock
    cell however the axes are arranged, measured -- while torch's
    last-dim reductions do the whole cell in under 2 ms on one thread.
    ``use_torch=None`` picks torch if importable; the two paths agree to
    float64 rounding (tests/test_cells.py) and the numpy one is the
    reference.
    """
    K, C, W = x.shape
    if factor < 1 or W % factor:
        raise ValueError(f"window {W} is not a multiple of factor {factor}")
    if C != len(schema.columns):
        raise ValueError("features do not match the supplied schema")
    if factor == 1:
        return np.asarray(x, dtype=np.float64).copy()
    if use_torch is None:
        use_torch = _torch is not None
    if use_torch:
        if _torch is None:
            raise RuntimeError("torch is not installed")
        return _aggregate_cell_torch(np.asarray(x), factor, schema)
    T = W // factor
    part = np.asarray(x).reshape(K, C, T, factor)
    out = np.empty((K, C, T), dtype=np.float64)
    rules = schema.aggregation_rules
    volume_index = schema.index("volume") if "volume" in schema.columns else None
    for index, column in enumerate(schema.columns):
        rule = rules[column]
        values = part[:, index]                                  # (K, T, factor)
        if rule == "last":
            out[:, index] = values[..., -1]
        elif rule == "max":
            out[:, index] = values.max(axis=-1)
        elif rule == "min":
            out[:, index] = values.min(axis=-1)
        elif rule == "sum":
            out[:, index] = values.sum(axis=-1, dtype=np.float64)
        elif rule == "vwap":
            if volume_index is None:
                out[:, index] = values.mean(axis=-1, dtype=np.float64)
                continue
            volume = part[:, volume_index]
            total = volume.sum(axis=-1, dtype=np.float64)
            numerator = np.nansum(np.multiply(values, volume, dtype=np.float64), axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, index] = np.where(total > 0, numerator / total, np.nan)
    return out


def _aggregate_cell_torch(x: np.ndarray, factor: int, schema: FeatureSchema) -> np.ndarray:
    K, C, W = x.shape
    T = W // factor
    part = _torch.from_numpy(x).reshape(K, C, T, factor)
    out = _torch.empty((K, C, T), dtype=_torch.float64)
    rules = schema.aggregation_rules
    volume_index = schema.index("volume") if "volume" in schema.columns else None
    for index, column in enumerate(schema.columns):
        rule = rules[column]
        values = part[:, index]
        if rule == "last":
            out[:, index] = values[..., -1]
        elif rule == "max":
            out[:, index] = values.amax(-1)
        elif rule == "min":
            out[:, index] = values.amin(-1)
        elif rule == "sum":
            out[:, index] = values.sum(-1, dtype=_torch.float64)
        elif rule == "vwap":
            if volume_index is None:
                out[:, index] = values.sum(-1, dtype=_torch.float64) / factor
                continue
            volume = part[:, volume_index]
            total = volume.sum(-1, dtype=_torch.float64)
            products = values.double() * volume.double()
            numerator = _torch.nan_to_num(products, nan=0.0).sum(-1)
            out[:, index] = _torch.where(total > 0, numerator / total, _torch.nan)
    return out.numpy()


def prior_vwaps(grid: np.ndarray, idx: np.ndarray, start_row: int, *,
                schema: FeatureSchema = MARKET_SCHEMA, lookback: int = 3600) -> np.ndarray:
    """``transforms.prior_vwap`` per stock: the last traded VWAP before the window.

    ``grid`` is channel-major ``(N, C, rows)``. Scans a ``lookback`` of rows first (the prior is almost always seconds
    away) and only then the whole prefix, per stock; NaN when nothing traded
    before the window, which is what ``None`` means in the scalar version.
    """
    vwap_index, count_index = schema.index("vwap_all"), schema.index("n")
    idx = np.asarray(idx)
    out = np.full(len(idx), np.nan, dtype=np.float64)
    if start_row <= 0:
        return out
    # One read for the whole cell (a memmap slice costs ~50 us of overhead
    # regardless of size); the full-prefix scan below is the rare fallback.
    lo = max(0, start_row - lookback)
    block = np.asarray(grid[idx][:, [count_index, vwap_index], lo:start_row])   # (K, 2, n)
    valid = (block[:, 0] > 0) & np.isfinite(block[:, 1])
    found = valid.any(axis=1)
    last = block.shape[-1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    out[found] = block[found, 1, last[found]]
    if lo > 0 and not found.all():
        for k in np.flatnonzero(~found):
            counts = np.asarray(grid[idx[k], count_index, :lo])
            vwaps = np.asarray(grid[idx[k], vwap_index, :lo])
            ok = (counts > 0) & np.isfinite(vwaps)
            if ok.any():
                out[k] = float(vwaps[np.flatnonzero(ok)[-1]])
    return out


def ffill_vwap_cell(views: np.ndarray, prior: np.ndarray, *,
                    schema: FeatureSchema = MARKET_SCHEMA) -> None:
    """``transforms.ffill_vwap`` along a leading stock axis, in place."""
    vwap_index = schema.index("vwap_all")
    values = views[:, vwap_index, :]                       # (K, T) view
    K, T = values.shape
    seed = np.isnan(values[:, 0]) & np.isfinite(prior)
    values[seed, 0] = prior[seed]
    missing = np.isnan(values)
    if missing.any():
        source = np.broadcast_to(np.arange(T), (K, T)).copy()
        source[missing] = 0
        np.maximum.accumulate(source, axis=1, out=source)
        values[:] = np.take_along_axis(values, source, axis=1)
    missing = np.isnan(values)
    if missing.any():
        bid, ask = schema.index("bid_price"), schema.index("ask_price")
        mid = (views[:, bid, :] + views[:, ask, :]) / 2
        values[missing] = mid[missing]


def normalize_cell(views: np.ndarray, groups: list[tuple[list[int], bool]]) -> np.ndarray:
    """``transforms.normalize`` per view along a leading stock axis, in place.

    Returns ``(K, n_groups, 2)`` of the (mean, scale) each view was divided
    by -- the natural-unit facts ``ViewMetadata`` carries.
    """
    K = views.shape[0]
    stats = np.empty((K, len(groups), 2), dtype=np.float64)
    for g, (indices, log_transform) in enumerate(groups):
        if log_transform:
            views[:, indices, :] = np.log1p(views[:, indices, :])
        values = views[:, indices, :]                      # (K, g, T)
        mean = values.mean(axis=(1, 2))
        scale = np.sqrt(values.var(axis=(1, 2)) + EPS)
        views[:, indices, :] = (values - mean[:, None, None]) / scale[:, None, None]
        stats[:, g, 0], stats[:, g, 1] = mean, scale
    return stats


def build_cell_views(grid: np.ndarray, idx: np.ndarray, draw: CellDraw, *,
                     groups: list[tuple[list[int], bool]],
                     schema: FeatureSchema = MARKET_SCHEMA,
                     window: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """K normalized views of one cell: ``(K, C, L)`` float64 and ``(K, G, 2)`` stats.

    ``grid`` is the day's channel-major ``(N, C, rows)`` array (a memmap is
    fine: only the K windows and the VWAP look-back are read). Pass ``window``
    when the ``(K, C, W)`` slice is already in hand. NaN left after the VWAP fill --
    a quote column NaN inside a feasible window cannot happen on the dense
    grid -- is zeroed, as every view builder does.
    """
    if window is None:
        window = np.asarray(grid[np.asarray(idx), :, draw.start_row:draw.start_row + draw.window])
    views = aggregate_cell(window, draw.aggregation, schema=schema)
    prior = prior_vwaps(grid, np.asarray(idx), draw.start_row, schema=schema)
    ffill_vwap_cell(views, prior, schema=schema)
    stats = normalize_cell(views, groups)
    if np.isnan(views).any():
        np.nan_to_num(views, copy=False, nan=0.0)
    return views, stats
