"""From stored sessions to model inputs: dense sessions, views, evaluation panels and their cache, training cells.

SESSIONS. Sparse observation reconstruction and session preprocessing.

VIEWS. From dense sessions to model-ready views: aggregation, fills, normalization, geometry.

TRANSFORMS. Reusable transformations from dense sessions to model-ready views.

VIEW CONFIGURATION. Model-independent configuration for extracting a view from a session.

PANELS. Model-neutral evaluation panels -- views beside their labels -- and their disk cache.

PANELS. Model-neutral construction of synchronized cross-sectional panels.

PANEL CACHE. Disk cache for model-neutral synchronized panels.

CELLS. Cross-sectional cells drawn from a day record, built K stocks at a time.

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

PANEL MARKET DATA. Evaluation inputs read from the dataset's stores, independent of any model.

Market data and realized forward returns for a set of day records, on one
shared decision and asset grid, and the panel cache's quotes joined onto
arbitrary rows.

A quote and a realized return are properties of a (date, anchor, ticker) row.
They do not depend on which encoder produced an embedding for that row, which
is what makes this module the join that lets a portfolio be built from
embeddings that were computed for something else entirely: score a sweep once,
then attach the market to it as often as the analysis needs, with no second
forward pass.

:class:`MarketPanel` reads one month of cached panel metadata -- quotes and
raw targets -- WITHOUT touching the cached views, which are the only large
array in a panel. :meth:`MarketPanel.align` then reorders it onto an arbitrary
set of rows, and :func:`to_grid` scatters flat rows onto the
``(decision, asset, horizon)`` rectangle every other stage speaks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fcntl
import hashlib
import json
import shutil
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.anchors import (
    ANCHOR_STEP,
    DEFAULT_ANCHOR_SPEC,
    DEFAULT_TARGET_HORIZONS,
    RETURN_VWAP_WINDOW,
    SESSION_LEN,
    AnchorSpec,
)
from stable_finance.dataset.datasets.market1t.spec import (
    MarketSchedule,
    Timeline,
    as_timeline,
)
from stable_finance.dataset.objective import Objective
from stable_finance.dataset.outcomes import ANCHOR_TARGET_TYPES, anchor_indices
from stable_finance.dataset.schema import (
    FeatureSchema,
    MarketSession,
    MissingRolesError,
    default_schema,
    has_mid,
    mid_series,
)
from stable_finance.evaluation.inputs import to_grid
from stable_finance.dataset.market import MarketData

if TYPE_CHECKING:
    from stable_finance.dataset.datasets.market1t.tables import AnchorTargetStats


# ── sessions ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CacheInfo:
    """Observable state of a :class:`SessionPreprocessor` LRU cache."""

    hits: int
    misses: int
    entries: int
    bytes: int
    max_bytes: int


def sparse_to_dense_grid(
    ts_sec: np.ndarray,
    features: np.ndarray,
    ts_open_sec: int,
    ts_close_sec: int,
    ffill_indices: list[int],
    zerofill_indices: list[int],
    trim_leading_nan: bool = True,
    dtype: np.dtype = np.float64,
    bar_seconds: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map sparse observations onto a canonical grid of ``bar_seconds`` bars.

    An observation lands on the bar whose timestamp it equals; one between
    bars is dropped, so sparse timestamps must already be stamped on the
    grid (as they are when bars and grid share an origin).
    """
    if len(ts_sec) == 0:
        return None
    if features.ndim != 2 or len(features) != len(ts_sec):
        raise ValueError("features must be 2-D with one row per timestamp")
    if ts_close_sec <= ts_open_sec:
        raise ValueError("session close must be after its open")

    canonical = np.arange(ts_open_sec, ts_close_sec, int(bar_seconds), dtype=np.int32)
    n_rows, n_features = len(canonical), features.shape[1]
    dense = np.full((n_rows, n_features), np.nan, dtype=dtype)

    pre_open = np.searchsorted(ts_sec, ts_open_sec)
    if pre_open:
        seed = features[pre_open - 1].astype(dtype)
        dense[0, ffill_indices] = seed[ffill_indices]

    positions = np.searchsorted(canonical, ts_sec)
    clipped = np.minimum(positions, n_rows - 1)
    valid = (positions < n_rows) & (canonical[clipped] == ts_sec)
    dense[positions[valid]] = features[valid].astype(dtype)

    for column in ffill_indices:
        values = dense[:, column]
        missing = np.isnan(values)
        if missing.any():
            source = np.arange(n_rows)
            source[missing] = 0
            np.maximum.accumulate(source, out=source)
            values[:] = values[source]
    for column in zerofill_indices:
        values = dense[:, column]
        values[np.isnan(values)] = 0.0

    if trim_leading_nan:
        complete = ~np.isnan(dense).any(axis=1)
        if not complete.any():
            return None
        first = int(np.argmax(complete))
        canonical, dense = canonical[first:], dense[first:]
    return canonical, dense


class SessionPreprocessor:
    """Turn a raw backend record into a dense :class:`MarketSession`.

    The processor is deliberately independent from MDS. A future LanceDB
    backend only needs to produce the same raw mapping keys.

    ``timeline`` is the dataset's clock: when the market is open and how
    long a row is. ``schedule`` is the older way to name one -- a Market-1T
    holiday schedule (or None) for the Market-1T timeline, or any calendar or
    timeline -- and is used when ``timeline`` is not given.
    """

    def __init__(
        self,
        *,
        schema: FeatureSchema | None = None,
        schedule: MarketSchedule | None = None,
        extended_hours: bool = False,
        zero_features: tuple[str, ...] = (),
        cache_bytes: int = 0,
        bounds: Callable[[str], tuple[int, int]] | None = None,
        timeline: Timeline | None = None,
    ) -> None:
        schema = default_schema(schema)
        if timeline is not None and (schedule is not None or extended_hours):
            raise ValueError("pass a timeline or a Market-1T schedule, not both")
        unknown = set(zero_features) - set(schema.columns)
        if unknown:
            raise ValueError(f"zero_features contains unknown columns: {sorted(unknown)}")
        self.schema = schema
        self.schedule = schedule
        self.extended_hours = extended_hours
        self.timeline = (timeline if timeline is not None
                         else as_timeline(schedule, extended_hours=extended_hours))
        self.bar_seconds = self.timeline.bar_seconds
        self.zero_indices = [schema.index(name) for name in zero_features]
        self.cache_bytes = max(0, int(cache_bytes))
        self.bounds = bounds
        self._cache: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._cached_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_info(self) -> CacheInfo:
        """Return cache telemetry without exposing the mutable cache itself."""
        return CacheInfo(
            hits=self._cache_hits,
            misses=self._cache_misses,
            entries=len(self._cache),
            bytes=self._cached_bytes,
            max_bytes=self.cache_bytes,
        )

    def transform(self, sample: Mapping) -> MarketSession | None:
        ticker, date = str(sample.get("ticker", "")), str(sample.get("date", ""))
        if date and not self.timeline.is_open(date):
            return None
        if "grid_start" in sample:
            return self._from_dense(sample, ticker, date)
        key = (ticker, date)
        cached = self._cache.get(key) if ticker and date else None
        if cached is not None:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            timestamps, stored = cached
            return MarketSession(ticker, date, timestamps, stored.astype(np.float64),
                                 self.schema, self.bar_seconds)

        if self.cache_bytes > 0 and ticker and date:
            self._cache_misses += 1

        start, end = (
            self.bounds(date)
            if self.bounds is not None
            else self.timeline.bounds(date)
        )
        use_cache = self.cache_bytes > 0 and ticker and date
        result = sparse_to_dense_grid(
            np.asarray(sample["ts_interval"]),
            np.asarray(sample["features"]),
            start,
            end,
            self.schema.forward_fill_indices,
            self.schema.zero_fill_indices,
            dtype=np.float32 if use_cache else np.float64,
            bar_seconds=self.bar_seconds,
        )
        if result is None:
            return None
        timestamps, features = result
        if self.zero_indices:
            features[:, self.zero_indices] = 0.0
        if use_cache:
            self._cache[key] = (timestamps, features)
            self._cached_bytes += timestamps.nbytes + features.nbytes
            while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
                _, (old_timestamps, old_features) = self._cache.popitem(last=False)
                self._cached_bytes -= old_timestamps.nbytes + old_features.nbytes
            features = features.astype(np.float64)
        return MarketSession(ticker, date, timestamps, features, self.schema,
                             self.bar_seconds)

    def _from_dense(self, sample: Mapping, ticker: str, date: str) -> MarketSession:
        """Adopt an already-gridded record instead of rebuilding the grid.

        A record written by ``densify`` stores the reconstructed
        session rather than the observed ticks: ``features`` is already
        ``(session_seconds, n_columns)`` with forward- and zero-fill applied,
        and ``grid_start`` is the epoch second of its first row. The timestamps
        are then an arange, which is why they are not stored.

        WHY THE FORMAT EXISTS. Rebuilding the grid is 71% of the CPU a
        dataloader worker spends on a sample (2.89 ms of 4.08 ms; the MDS read
        is 0.25 ms and cropping two global views is 0.95 ms), and it is
        recomputed identically on every visit. Storing the result is also
        SMALLER on the wire despite being 1.74x more raw bytes: a filled grid
        is runs of repeated values, which zstd removes, while the sparse form
        pays for a monotone int32 timestamp column that does not compress.
        Measured at 0.75-0.80x of the sparse record across 2009-06, 2015-06 and
        2023-01 at zstd levels 1, 3 and 9.

        The cache is bypassed deliberately: its whole purpose was to amortize
        the grid rebuild, and there is nothing left to amortize. Holding a
        second copy of data the page cache already has would only take memory
        from the reader.
        """
        if self.extended_hours or self.bounds is not None:
            # A DENSE RECORD CARRIES ONE GRID, and it is the regular session.
            # Silently handing it back to a caller that asked for extended
            # hours (or for custom bounds) would return a 6.5-hour view where a
            # 16-hour one was requested, with nothing in the data to say so --
            # the failure would surface as a quietly different model. Callers
            # needing either must read a sparse dataset, which a densify
            # conversion does not destroy.
            raise ValueError(
                "dense records are the regular session only; "
                "extended_hours and custom bounds require a sparse dataset"
            )
        features = np.asarray(sample["features"])
        if features.ndim == 1:
            features = features.reshape(-1, len(self.schema.columns))
        start = int(np.asarray(sample["grid_start"]).reshape(-1)[0])
        timestamps = start + self.bar_seconds * np.arange(len(features), dtype=np.int64)
        features = features.astype(np.float64)
        if self.zero_indices:
            features[:, self.zero_indices] = 0.0
        return MarketSession(ticker, date, timestamps, features, self.schema,
                             self.bar_seconds)


# ── views ────────────────────────────────────────────────────────────────────

EPS = 1e-5

FEATURE_TYPES: dict[str, tuple[str, ...]] = {
    "price": ("bid_price", "ask_price", "vwap_all", "high", "low", "mid_price"),
    "order_book_size": ("bid_size", "ask_size"),
    "trade_size": ("volume",),
    "trade_count": ("n",),
}
LOG1P_TYPES = {"order_book_size", "trade_size", "trade_count"}


def volume_index(schema: FeatureSchema) -> int | None:
    """The column a ``vwap`` bucket is weighted by, or None for a plain mean.

    The column playing the ``volume`` role.
    """
    return schema.optional_index("volume")


def build_norm_groups(feature_columns: list[str] | tuple[str, ...] | FeatureSchema):
    """Return normalization index groups for a schema or a column selection.

    Pass the schema: its own ``normalization`` is the definition. A list of
    column names is matched against the Market-1T name groups in
    :data:`FEATURE_TYPES`, which is what this function did before schemas
    carried their own groups.
    """
    if isinstance(feature_columns, FeatureSchema):
        return feature_columns.normalization_groups()
    positions = {column: index for index, column in enumerate(feature_columns)}
    groups = []
    for kind, columns in FEATURE_TYPES.items():
        indices = [positions[column] for column in columns if column in positions]
        if indices:
            groups.append((indices, kind in LOG1P_TYPES))
    return groups


def aggregate(
    features: np.ndarray,
    scale_factor: int,
    *,
    offset: int = 0,
    schema: FeatureSchema | None = None,
) -> np.ndarray | None:
    """Aggregate a dense view to ``scale_factor`` bars per row, by its schema's rules."""
    schema = default_schema(schema)
    if scale_factor < 1 or offset < 0:
        raise ValueError("scale_factor must be positive and offset non-negative")
    source = np.asarray(features)[offset:]
    if source.ndim != 2 or source.shape[1] != len(schema.columns):
        raise ValueError("features do not match the supplied schema")
    if scale_factor == 1:
        return source.astype(np.float64, copy=True) if len(source) >= 2 else None
    full_count, remainder = divmod(len(source), scale_factor)
    bucket_count = full_count + bool(remainder)
    if bucket_count < 2:
        return None
    output = np.empty((bucket_count, source.shape[1]), dtype=np.float64)
    rules = schema.aggregation_rules
    weight_index = volume_index(schema)

    def reduce(part: np.ndarray, destination: np.ndarray) -> None:
        for index, column in enumerate(schema.columns):
            rule = rules[column]
            values = part[..., :, index]
            if rule == "last":
                destination[..., index] = values[..., -1]
            elif rule == "first":
                destination[..., index] = values[..., 0]
            elif rule == "mean":
                destination[..., index] = np.nanmean(values, axis=-1)
            elif rule == "max":
                destination[..., index] = values.max(axis=-1)
            elif rule == "min":
                destination[..., index] = values.min(axis=-1)
            elif rule == "sum":
                destination[..., index] = values.sum(axis=-1, dtype=np.float64)
            elif rule == "vwap":
                if weight_index is None:
                    destination[..., index] = values.mean(axis=-1, dtype=np.float64)
                    continue
                volume = part[..., :, weight_index]
                total = volume.sum(axis=-1, dtype=np.float64)
                numerator = np.nansum(
                    np.multiply(values, volume, dtype=np.float64), axis=-1
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    destination[..., index] = np.where(total > 0, numerator / total, np.nan)

    if full_count:
        full = source[: full_count * scale_factor].reshape(
            full_count, scale_factor, source.shape[1]
        )
        reduce(full, output[:full_count])
    if remainder:
        partial = source[full_count * scale_factor:]
        reduce(partial, output[full_count])
    return output


def resample_session(session: MarketSession, bar_seconds: int) -> MarketSession:
    """Convert a dense session to a coarser native bar resolution.

    Only integer coarsening is accepted; silently interpolating between grids
    would change both view geometry and target horizons. The last bucket may
    be partial and is retained using the same schema-aware reducers.
    """
    target = int(bar_seconds)
    if target < session.bar_seconds or target % session.bar_seconds:
        raise ValueError(
            "bar_seconds must be an integer multiple of the source resolution"
        )
    factor = target // session.bar_seconds
    features = aggregate(session.features, factor, schema=session.schema)
    if features is None:
        raise ValueError("session is too short to produce two output bars")
    timestamps = session.timestamps[::factor][:len(features)]
    return MarketSession(
        ticker=session.ticker,
        date=session.date,
        timestamps=timestamps,
        features=features,
        schema=session.schema,
        bar_seconds=target,
    )


def prior_vwap(
    features: np.ndarray,
    window_start_idx: int,
    *,
    schema: FeatureSchema | None = None,
) -> float | None:
    """Return the last valid, traded VWAP preceding a view.

    None when the schema has no ``vwap`` column. A row counts as traded when
    its ``n_trades`` (else its ``volume``) is positive; a schema with neither
    takes any finite VWAP.
    """
    schema = default_schema(schema)
    if window_start_idx <= 0:
        return None
    vwap_index = schema.optional_index("vwap")
    if vwap_index is None:
        return None
    history = features[:window_start_idx]
    valid = np.isfinite(history[:, vwap_index])
    count_index = trade_count_index(schema)
    if count_index is not None:
        valid &= history[:, count_index] > 0
    return float(history[np.flatnonzero(valid)[-1], vwap_index]) if valid.any() else None


def trade_count_index(schema: FeatureSchema) -> int | None:
    """The column that says a bar traded: ``n_trades``, else ``volume``."""
    index = schema.optional_index("n_trades")
    return index if index is not None else schema.optional_index("volume")


def ffill_vwap(
    view: np.ndarray,
    prior: float | None,
    *,
    schema: FeatureSchema | None = None,
) -> None:
    """Fill zero-volume VWAP buckets in place before normalization.

    Forward from the previous bucket, seeded by ``prior``; what is still
    missing takes the midpoint when one can be read. A no-op for a schema
    with no ``vwap`` column.
    """
    schema = default_schema(schema)
    vwap_index = schema.optional_index("vwap")
    if vwap_index is None:
        return
    values = view[:, vwap_index]
    if np.isnan(values[0]) and prior is not None:
        values[0] = prior
    missing = np.isnan(values)
    if missing.any():
        source = np.arange(len(values))
        source[missing] = 0
        np.maximum.accumulate(source, out=source)
        values[:] = values[source]
    missing = np.isnan(values)
    if missing.any() and has_mid(schema):
        values[missing] = mid_series(view[missing], schema)


def normalize(
    features: np.ndarray,
    groups: list[tuple[list[int], bool]],
    stats_out: list[tuple[float, float]] | None = None,
) -> None:
    """Apply per-view grouped normalization in place."""
    for indices, log_transform in groups:
        if log_transform:
            features[:, indices] = np.log1p(features[:, indices])
        values = features[:, indices]
        mean = np.mean(values)
        standard_deviation = np.sqrt(np.var(values) + EPS)
        features[:, indices] = (values - mean) / standard_deviation
        if stats_out is not None:
            stats_out.append((float(mean), float(standard_deviation)))


# ── view configuration ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ViewMetadata:
    """Model-independent facts about a materialized market view.

    Values remain in their natural units. A model package may encode them and
    project them into one or more information tokens, without transporting
    constants as repeated time-series columns.
    """

    start_seconds: float
    end_seconds: float
    aggregation_seconds: float
    normalization_means: NDArray[np.floating]
    normalization_scales: NDArray[np.floating]

    def __post_init__(self) -> None:
        means = np.asarray(self.normalization_means)
        scales = np.asarray(self.normalization_scales)
        if self.end_seconds <= self.start_seconds:
            raise ValueError("view bounds must satisfy start < end")
        if self.aggregation_seconds <= 0:
            raise ValueError("aggregation_seconds must be positive")
        if means.ndim != 1 or scales.shape != means.shape:
            raise ValueError("normalization means and scales must be matching vectors")
        if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
            raise ValueError("normalization scales must be finite and positive")
        object.__setattr__(self, "normalization_means", means)
        object.__setattr__(self, "normalization_scales", scales)


@dataclass(frozen=True)
class ViewSpec:
    """Geometry of a single-session model view.

    These values are hyperparameters, not properties of the dataset. The
    current research recipe uses a 2,048-token view spanning 50–100% of a
    session, but callers must opt into that recipe explicitly through this
    object. ``aggregation_seconds`` takes precedence over ``scale_range`` when
    supplied, matching the established extended-hours behavior.
    """

    sequence_length: int = 2048
    scale_range: tuple[float, float] = (0.5, 1.0)
    aggregation_seconds: tuple[int, int] | None = None
    end_grid_seconds: int | None = None
    min_future_seconds: int = 0

    def __post_init__(self) -> None:
        low, high = self.scale_range
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if not 0 < low <= high <= 1:
            raise ValueError("scale_range must satisfy 0 < low <= high <= 1")
        if self.aggregation_seconds is not None:
            agg_low, agg_high = self.aggregation_seconds
            if not 1 <= agg_low <= agg_high:
                raise ValueError("aggregation_seconds must be positive and ordered")
        if self.end_grid_seconds is not None and self.end_grid_seconds < 1:
            raise ValueError("end_grid_seconds must be positive")
        if self.min_future_seconds < 0:
            raise ValueError("min_future_seconds cannot be negative")


def prepare_view(
    view: np.ndarray,
    normalization_groups: list[tuple[list[int], bool]],
    *,
    start_seconds: float,
    aggregation_seconds: float,
) -> tuple[np.ndarray, ViewMetadata]:
    """Normalize a materialized view and return its metadata separately."""
    statistics: list[tuple[float, float]] = []
    normalize(view, normalization_groups, stats_out=statistics)
    return view, ViewMetadata(
        start_seconds=float(start_seconds),
        end_seconds=float(start_seconds + len(view) * aggregation_seconds),
        aggregation_seconds=float(aggregation_seconds),
        normalization_means=np.asarray(
            [mean for mean, _ in statistics], dtype=np.float64
        ),
        normalization_scales=np.asarray(
            [scale for _, scale in statistics], dtype=np.float64
        ),
    )


# ── panels ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PanelObservation:
    """One model-neutral view and its aligned target metadata."""

    date: str
    ticker: str
    anchor: int
    view: np.ndarray
    metadata: ViewMetadata
    target: np.ndarray
    raw_target: np.ndarray
    quote: np.ndarray | None = None
    """``[best_bid, best_ask]`` in natural units at the decision row.

    The observable market at the instant the decision is made, carried beside
    the view because the view is normalized and its scale is per-observation.
    ``None`` when the producer does not supply it; NaN marks an asset that was
    not quoting, which the execution simulator treats as unfillable. Also
    ``None`` for a schema without both ``bid_price`` and ``ask_price``.
    """


def choose_cell_aggregation(
    date: str,
    anchor: int,
    latest_index: int,
    sequence_length: int = 2048,
    *,
    choices=range(6, 12),
) -> int | None:
    """Choose a reproducible feasible resolution for a decision cell."""
    feasible = [int(value) for value in choices
                if int(value) * sequence_length <= latest_index + 1]
    if not feasible:
        return None
    seed = zlib.crc32(f"{date}@{int(anchor)}".encode())
    return feasible[np.random.RandomState(seed).randint(len(feasible))]


def evenly_spaced_anchors(
    n_per_day: int,
    sequence_length: int = 2048,
    *,
    aggregation_choices=range(6, 12),
    horizons=DEFAULT_TARGET_HORIZONS,
    anchor_spec: AnchorSpec = DEFAULT_ANCHOR_SPEC,
) -> np.ndarray:
    """Choose evenly spaced anchors over the feasible decision band."""
    if n_per_day < 1:
        raise ValueError("n_per_day must be positive")
    choices = tuple(int(value) for value in aggregation_choices)
    requested_horizons = tuple(int(value) for value in horizons)
    grid = anchor_indices(spec=anchor_spec)
    grid = grid[
        (grid >= min(choices) * sequence_length - 1)
        & (grid + min(requested_horizons) <= anchor_spec.session_seconds - 1)
    ]
    if not len(grid):
        raise ValueError("no anchor admits a full view")
    indices = np.linspace(0, len(grid) - 1, n_per_day).round().astype(int)
    return grid[np.unique(indices)]


def build_session_panel(
    session: MarketSession,
    anchors,
    stats: AnchorTargetStats,
    *,
    view_spec: ViewSpec = ViewSpec(aggregation_seconds=(6, 11)),
    target_types=ANCHOR_TARGET_TYPES,
    horizons=DEFAULT_TARGET_HORIZONS,
    target_transform: str = "uniform",
    normalization_groups=None,
    timeline=None,
    objective=None,
) -> list[PanelObservation]:
    """Build requested decisions for one ticker-session.

    Views and natural-unit metadata are returned separately. Model packages
    decide whether and how to encode that metadata into tokens.

    ``anchors`` are seconds past the timeline's origin (Market-1T's 09:30 by
    default). View aggregation counts stored bars, which are seconds only at
    1 Hz. The labels are the ``objective``'s; without one, ``target_types``
    and ``horizons`` define it, and ``stats`` must be a table built for the
    same objective.
    """
    timeline = as_timeline(timeline)
    if timeline.bar_seconds != session.bar_seconds:
        raise ValueError(
            f"the timeline is at {timeline.bar_seconds}s bars, the session at "
            f"{session.bar_seconds}s")
    # The labels are the objective's to compute; the panel only joins them to
    # the views.
    if objective is None:
        objective = Objective(types=tuple(target_types), horizons=tuple(horizons))
    objective = objective.resolve(session.schema)
    target_types, horizons = objective.types, objective.horizons
    table_window = getattr(stats, "window_seconds", None)
    if table_window is not None and table_window != objective.window_seconds:
        raise ValueError(
            f"the target table was built over a {table_window}s window, the labels "
            f"over {objective.window_seconds}s: standardizing one against "
            "the other mixes two different targets")
    bar = session.bar_seconds
    target_types = tuple(target_types)
    horizons = tuple(int(value) for value in horizons)
    groups = (
        build_norm_groups(session.schema)
        if normalization_groups is None else normalization_groups
    )
    choices = view_spec.aggregation_seconds or range(6, 12)
    fixed = choices[0] if choices[0] == choices[1] else None
    features = session.features
    # Read off THIS session's schema rather than a module constant: the
    # execution stage charges the spread that was actually quoted, and a
    # backend that reorders its columns must move these indices with it.
    quote_columns = (
        [session.schema.role_index("bid_price"), session.schema.role_index("ask_price")]
        if session.schema.has("bid_price", "ask_price") else None
    )
    offset = int(session.timestamps[0]) - timeline.origin(session.date)
    output = []
    for anchor_value in anchors:
        anchor = int(anchor_value)
        if (anchor - offset) % bar:
            continue
        row = (anchor - offset) // bar
        if row < 0 or row >= len(features):
            continue
        aggregation = (
            fixed if fixed is not None and fixed * view_spec.sequence_length <= anchor + 1
            else None if fixed is not None
            else choose_cell_aggregation(
                session.date, anchor, anchor, view_spec.sequence_length,
                choices=choices,
            )
        )
        if aggregation is None:
            continue
        window = aggregation * view_spec.sequence_length
        start = row + 1 - window
        if start < 0:
            continue
        view = aggregate(features[start:start + window], aggregation,
                         schema=session.schema)
        if view is None or len(view) != view_spec.sequence_length:
            continue
        raw = objective.raw(features, [row], schema=session.schema, timeline=timeline)[0]
        bundle = stats.transform(
            raw, session.date, anchor, target_types, horizons,
        )
        target = bundle.select(target_transform)
        if not np.isfinite(target).any():
            continue
        ffill_vwap(view, prior_vwap(features, start, schema=session.schema),
                   schema=session.schema)
        view, metadata = prepare_view(
            view, groups,
            start_seconds=offset + start * bar,
            aggregation_seconds=aggregation,
        )
        np.nan_to_num(view, copy=False, nan=0.0)
        output.append(PanelObservation(
            date=session.date,
            ticker=session.ticker,
            anchor=anchor,
            view=view.astype(np.float32),
            metadata=metadata,
            target=target.ravel().astype(np.float32),
            raw_target=raw.ravel().astype(np.float32),
            quote=(features[row, quote_columns].astype(np.float64)
                   if quote_columns is not None else None),
        ))
    return output


def build_sample_panel(
    sample,
    schedule,
    anchors,
    stats: AnchorTargetStats,
    *,
    schema: FeatureSchema | None = None,
    **kwargs,
) -> list[PanelObservation]:
    """Backend-record convenience adapter for :func:`build_session_panel`.

    ``schedule`` is the record's timeline, or a Market-1T holiday schedule.
    """
    timeline = as_timeline(schedule)
    session = SessionPreprocessor(schema=schema, timeline=timeline).transform(sample)
    if session is None:
        return []
    return build_session_panel(session, anchors, stats, timeline=timeline, **kwargs)


def iter_month_panel(
    month_dir,
    stats: AnchorTargetStats,
    schedule,
    anchors,
    batch_size: int,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    **kwargs,
):
    """Yield model-neutral view batches from an MDS month.

    ``schedule`` is the month's holiday schedule (or timeline).
    """
    import tempfile

    from stable_finance.dataset.datasets.market1t.mosaic import (
        open_shard,
        shard_list,
    )
    if batch_size < 1 or num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid batch or shard configuration")
    shards = shard_list(month_dir)[shard_index::num_shards]
    views: list[np.ndarray] = []
    observations: list[PanelObservation] = []
    with tempfile.TemporaryDirectory(prefix="stable_finance_panel_") as scratch:
        for shard in shards:
            reader = open_shard(month_dir, shard, scratch)
            for index in range(shard["samples"]):
                rows = build_sample_panel(
                    reader.get_item(index), schedule, anchors, stats, **kwargs,
                )
                for row in rows:
                    views.append(row.view)
                    observations.append(row)
                    if len(views) == batch_size:
                        yield np.stack(views), observations
                        views, observations = [], []
    if views:
        yield np.stack(views), observations


# ── panel cache ──────────────────────────────────────────────────────────────

# 2 (2026-09-10): panels carry the QUOTE column. The bump is the point: a
# panel built under 1 has no quotes, and a reader that filled the gap with
# NaN would hand the execution stage an unquoted market -- nothing fills,
# every spread charges zero, and a frictionless Sharpe reports as a
# spread-charged one. A miss costs a rebuild; a silent NaN costs a result.
PANEL_CACHE_VERSION = 2


class CacheClaimed(RuntimeError):
    """Another process currently owns this cache build."""


@dataclass(frozen=True)
class CachedPanel:
    views: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    dates: np.ndarray
    anchors: np.ndarray
    tickers: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    aggregations: np.ndarray
    normalization_means: np.ndarray
    normalization_scales: np.ndarray
    #: ``(n_rows, 2)`` bid and ask, or None for a panel whose dataset has no
    #: quotes at all -- recorded as ``"quotes": false`` in its meta.json.
    quotes: np.ndarray | None
    metadata: dict


def cache_key(**identity) -> dict:
    """Return a serializable identity plus a deterministic short hash."""
    value = {"format": PANEL_CACHE_VERSION, **identity}
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["hash"] = hashlib.sha256(blob).hexdigest()[:12]
    return value


class PanelCache:
    """Materialize and stream normalized views with natural-unit metadata."""

    def __init__(self, root: str | Path, *, view_dtype=np.float16):
        self.root = Path(root)
        self.view_dtype = np.dtype(view_dtype)

    def directory(self, month: str, key: dict) -> Path:
        return self.root / key["hash"] / month

    @staticmethod
    def _raw_is_complete(directory: Path, metadata: dict) -> bool:
        expected = (
            int(np.prod(metadata["views_shape"]))
            * np.dtype(metadata["views_dtype"]).itemsize
        )
        try:
            return (directory / "views.raw").stat().st_size == expected
        except OSError:
            return False

    def is_built(self, month: str, key: dict) -> bool:
        directory = self.directory(month, key)
        if not (directory / ".ready").is_file():
            return False
        try:
            metadata = json.loads((directory / "meta.json").read_text())
            return (
                metadata["key"]["hash"] == key["hash"]
                and self._raw_is_complete(directory, metadata)
            )
        except (OSError, ValueError, KeyError):
            return False

    def load(self, month: str, key: dict) -> CachedPanel:
        directory = self.directory(month, key)
        metadata = json.loads((directory / "meta.json").read_text())
        if metadata["key"]["hash"] != key["hash"]:
            raise RuntimeError("panel cache key mismatch")
        shape = tuple(metadata["views_shape"])
        views = np.memmap(
            directory / "views.raw", dtype=np.dtype(metadata["views_dtype"]),
            mode="r", shape=shape,
        )
        names = {
            "targets": "targets.npy", "raw_targets": "raw_targets.npy",
            "dates": "dates.npy", "anchors": "anchors.npy",
            "tickers": "tickers.npy", "starts": "starts.npy",
            "ends": "ends.npy", "aggregations": "aggregations.npy",
            "normalization_means": "normalization_means.npy",
            "normalization_scales": "normalization_scales.npy",
            "quotes": "quotes.npy",
        }
        if metadata.get("quotes", True) is False:
            names.pop("quotes")
        arrays = {
            name: np.load(directory / filename, allow_pickle=False)
            for name, filename in names.items()
        }
        arrays.setdefault("quotes", None)
        if not self._raw_is_complete(directory, metadata):
            raise RuntimeError(f"panel cache at {directory} is truncated")
        if any(len(value) != len(views) for value in arrays.values()
               if value is not None):
            raise RuntimeError(f"panel cache at {directory} has ragged columns")
        return CachedPanel(views=views, metadata=metadata, **arrays)

    def iter_batches(self, month: str, key: dict, batch_size: int):
        panel = self.load(month, key)
        for start in range(0, len(panel.views), batch_size):
            stop = min(start + batch_size, len(panel.views))
            sl = slice(start, stop)
            observations = []
            for index in range(start, stop):
                observations.append(PanelObservation(
                    date=str(panel.dates[index]),
                    ticker=str(panel.tickers[index]),
                    anchor=int(panel.anchors[index]),
                    view=np.asarray(panel.views[index], dtype=np.float32),
                    metadata=ViewMetadata(
                        start_seconds=float(panel.starts[index]),
                        end_seconds=float(panel.ends[index]),
                        aggregation_seconds=float(panel.aggregations[index]),
                        normalization_means=panel.normalization_means[index],
                        normalization_scales=panel.normalization_scales[index],
                    ),
                    target=panel.targets[index],
                    raw_target=panel.raw_targets[index],
                    quote=None if panel.quotes is None else panel.quotes[index],
                ))
            yield np.asarray(panel.views[sl], dtype=np.float32), observations

    def build(self, month: str, key: dict, batches, *, overwrite=False) -> Path:
        directory = self.directory(month, key)
        if self.is_built(month, key) and not overwrite:
            return directory
        directory.parent.mkdir(parents=True, exist_ok=True)
        lock = open(directory.with_suffix(".lock"), "w")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise CacheClaimed(f"{month} is being built elsewhere") from error
            if self.is_built(month, key) and not overwrite:
                return directory
            partial = directory.with_suffix(".partial")
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            columns = {name: [] for name in (
                "targets", "raw_targets", "dates", "anchors", "tickers",
                "starts", "ends", "aggregations", "normalization_means",
                "normalization_scales", "quotes",
            )}
            n_rows, view_shape = 0, None
            with open(partial / "views.raw", "wb", buffering=1 << 22) as output:
                for views, observations in batches:
                    views = np.asarray(views)
                    if len(views) != len(observations):
                        raise ValueError("views and observations are not aligned")
                    if view_shape is None:
                        view_shape = views.shape[1:]
                    elif views.shape[1:] != view_shape:
                        raise ValueError("view shape changed during cache build")
                    output.write(np.ascontiguousarray(
                        views, dtype=self.view_dtype
                    ).tobytes())
                    for row in observations:
                        columns["targets"].append(row.target)
                        columns["raw_targets"].append(row.raw_target)
                        columns["dates"].append(row.date)
                        columns["anchors"].append(row.anchor)
                        columns["tickers"].append(row.ticker)
                        columns["starts"].append(row.metadata.start_seconds)
                        columns["ends"].append(row.metadata.end_seconds)
                        columns["aggregations"].append(row.metadata.aggregation_seconds)
                        columns["normalization_means"].append(row.metadata.normalization_means)
                        columns["normalization_scales"].append(row.metadata.normalization_scales)
                        # None is a dataset with no bid/ask columns at all,
                        # which is NOT a NaN quote: NaN means a name that was
                        # not quoting, and storing a missing column that way
                        # would hand the execution stage an untradable market
                        # where it should refuse. All rows of a panel come
                        # from one schema, so the rows agree; the column is
                        # written only when they have quotes.
                        columns["quotes"].append(
                            None if row.quote is None
                            else np.asarray(row.quote, dtype=np.float64))
                    n_rows += len(views)
            if not n_rows:
                raise RuntimeError("cannot cache an empty panel")
            quoted = {quote is not None for quote in columns["quotes"]}
            if len(quoted) > 1:
                raise ValueError("some observations carry quotes and some do not")
            has_quotes = quoted == {True}
            if not has_quotes:
                columns.pop("quotes")
            for name, values in columns.items():
                np.save(partial / f"{name}.npy", np.asarray(values))
            metadata = {
                "month": month, "key": key, "n_rows": n_rows,
                "views_shape": [n_rows, *view_shape],
                "views_dtype": self.view_dtype.name,
                "quotes": has_quotes,
            }
            (partial / "meta.json").write_text(json.dumps(metadata, indent=2))
            (partial / ".ready").touch()
            if directory.exists():
                shutil.rmtree(directory)
            partial.rename(directory)
            return directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


# ── cells ────────────────────────────────────────────────────────────────────

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
    session_seconds: int = SESSION_LEN

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
    def slack_for(horizons, window: int = RETURN_VWAP_WINDOW) -> int:
        """The slack a label set pins: one horizon -> h + its measurement window."""
        unique = {int(h) for h in horizons}
        return (min(unique) + int(window)) if len(unique) == 1 else 0


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
    last = min(open_tod + session_rows - 1,
               geometry.session_seconds - 1 - geometry.slack_seconds)
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


def aggregate_cell(x: np.ndarray, factor: int, *, schema: FeatureSchema | None = None,
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
    schema = default_schema(schema)
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
    weight_index = volume_index(schema)
    for index, column in enumerate(schema.columns):
        rule = rules[column]
        values = part[:, index]                                  # (K, T, factor)
        if rule == "last":
            out[:, index] = values[..., -1]
        elif rule == "first":
            out[:, index] = values[..., 0]
        elif rule == "mean":
            out[:, index] = np.nanmean(values, axis=-1)
        elif rule == "max":
            out[:, index] = values.max(axis=-1)
        elif rule == "min":
            out[:, index] = values.min(axis=-1)
        elif rule == "sum":
            out[:, index] = values.sum(axis=-1, dtype=np.float64)
        elif rule == "vwap":
            if weight_index is None:
                out[:, index] = values.mean(axis=-1, dtype=np.float64)
                continue
            volume = part[:, weight_index]
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
    weight_index = volume_index(schema)
    for index, column in enumerate(schema.columns):
        rule = rules[column]
        values = part[:, index]
        if rule == "last":
            out[:, index] = values[..., -1]
        elif rule == "first":
            out[:, index] = values[..., 0]
        elif rule == "mean":
            out[:, index] = _torch.nanmean(values.double(), -1)
        elif rule == "max":
            out[:, index] = values.amax(-1)
        elif rule == "min":
            out[:, index] = values.amin(-1)
        elif rule == "sum":
            out[:, index] = values.sum(-1, dtype=_torch.float64)
        elif rule == "vwap":
            if weight_index is None:
                out[:, index] = values.sum(-1, dtype=_torch.float64) / factor
                continue
            volume = part[:, weight_index]
            total = volume.sum(-1, dtype=_torch.float64)
            products = values.double() * volume.double()
            numerator = _torch.nan_to_num(products, nan=0.0).sum(-1)
            out[:, index] = _torch.where(total > 0, numerator / total, _torch.nan)
    return out.numpy()


def prior_vwaps(grid: np.ndarray, idx: np.ndarray, start_row: int, *,
                schema: FeatureSchema | None = None, lookback: int = 3600) -> np.ndarray:
    """``transforms.prior_vwap`` per stock: the last traded VWAP before the window.

    ``grid`` is channel-major ``(N, C, rows)``. Scans a ``lookback`` of rows first (the prior is almost always seconds
    away) and only then the whole prefix, per stock; NaN when nothing traded
    before the window, which is what ``None`` means in the scalar version.
    All NaN for a schema with no ``vwap`` column.
    """
    schema = default_schema(schema)
    vwap_index = schema.optional_index("vwap")
    count_index = trade_count_index(schema)
    idx = np.asarray(idx)
    out = np.full(len(idx), np.nan, dtype=np.float64)
    if start_row <= 0 or vwap_index is None:
        return out
    if count_index is None:
        count_index = vwap_index      # finite VWAP is the only evidence of a trade
    # One read for the whole cell (a memmap slice costs ~50 us of overhead
    # regardless of size); the full-prefix scan below is the rare fallback.
    lo = max(0, start_row - lookback)
    block = np.asarray(grid[idx][:, [count_index, vwap_index], lo:start_row])   # (K, 2, n)
    valid = np.isfinite(block[:, 1])
    if count_index != vwap_index:
        valid &= block[:, 0] > 0
    found = valid.any(axis=1)
    last = block.shape[-1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    out[found] = block[found, 1, last[found]]
    if lo > 0 and not found.all():
        for k in np.flatnonzero(~found):
            counts = np.asarray(grid[idx[k], count_index, :lo])
            vwaps = np.asarray(grid[idx[k], vwap_index, :lo])
            ok = np.isfinite(vwaps) & ((counts > 0) if count_index != vwap_index else True)
            if ok.any():
                out[k] = float(vwaps[np.flatnonzero(ok)[-1]])
    return out


def ffill_vwap_cell(views: np.ndarray, prior: np.ndarray, *,
                    schema: FeatureSchema | None = None) -> None:
    """``transforms.ffill_vwap`` along a leading stock axis, in place."""
    schema = default_schema(schema)
    vwap_index = schema.optional_index("vwap")
    if vwap_index is None:
        return
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
    if missing.any() and has_mid(schema):
        mid_index = schema.optional_index("mid")
        if mid_index is not None:
            mid = views[:, mid_index, :]
        else:
            bid, ask = schema.role_index("bid_price"), schema.role_index("ask_price")
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
                     schema: FeatureSchema | None = None,
                     window: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """K normalized views of one cell: ``(K, C, L)`` float64 and ``(K, G, 2)`` stats.

    ``grid`` is the day's channel-major ``(N, C, rows)`` array (a memmap is
    fine: only the K windows and the VWAP look-back are read). Pass ``window``
    when the ``(K, C, W)`` slice is already in hand. NaN left after the VWAP fill --
    a quote column NaN inside a feasible window cannot happen on the dense
    grid -- is zeroed, as every view builder does.
    """
    schema = default_schema(schema)
    if window is None:
        window = np.asarray(grid[np.asarray(idx), :, draw.start_row:draw.start_row + draw.window])
    views = aggregate_cell(window, draw.aggregation, schema=schema)
    prior = prior_vwaps(grid, np.asarray(idx), draw.start_row, schema=schema)
    ffill_vwap_cell(views, prior, schema=schema)
    stats = normalize_cell(views, groups)
    if np.isnan(views).any():
        np.nan_to_num(views, copy=False, nan=0.0)
    return views, stats


# ── panel market data ────────────────────────────────────────────────────────

def _row_keys(dates: ArrayLike, anchors: ArrayLike,
              tickers: ArrayLike) -> NDArray[np.str_]:
    """The join key, as one string per row.

    A structured dtype or a tuple list would both work; a single string column
    is used because it makes the key visible in any debugger and because the
    dict lookup that follows is the same cost either way.
    """
    date = np.asarray(dates).astype("U10")
    anchor = np.char.mod("%d", np.asarray(anchors).astype(np.int64))
    ticker = np.asarray(tickers).astype("U8")
    return np.char.add(np.char.add(date, np.char.add("|", anchor)),
                       np.char.add("|", ticker))

def find_cached_month(
    root: str | Path, month: str, *, anchors_per_day: int,
    stats_tag: str | None = None,
) -> Path:
    """Locate a ready cache directory for ``month`` at this anchor schedule.

    THE KEY HASH IS DELIBERATELY NOT REQUIRED. A panel's hash covers the view
    construction -- normalization groups, sequence length, aggregation -- none
    of which changes a bid, an ask, or a realized return. Demanding the exact
    hash would force a caller to reconstruct an encoder's view settings just to
    read the market, and would miss a perfectly good quote table sitting under
    a neighbouring key.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"panel cache root {base} does not exist")
    for key_dir in sorted(base.iterdir()):
        directory = key_dir / month
        if not (directory / ".ready").is_file():
            continue
        try:
            key = json.loads((directory / "meta.json").read_text())["key"]
        except (OSError, ValueError, KeyError):
            continue
        if int(key.get("anchors_per_day", -1)) != int(anchors_per_day):
            continue
        if stats_tag is not None and key.get("stats_tag") != stats_tag:
            continue
        # A panel from a dataset without quotes says so in its meta.json; one
        # that should have them and does not is a damaged cache, skipped.
        quotes_absent = json.loads((directory / "meta.json").read_text()).get("quotes") is False
        if quotes_absent or (directory / "quotes.npy").is_file():
            return directory
    raise FileNotFoundError(
        f"no ready panel cache for {month} at {anchors_per_day} anchors/day"
        + (f" with stats {stats_tag!r}" if stats_tag else "")
    )

@dataclass(frozen=True)
class MarketPanel:
    """One month of quotes and realized targets, keyed by row.

    ``quotes`` is None when the panel's dataset has no bid and ask; every
    quote-reading method then raises
    :class:`~stable_finance.dataset.schema.MissingRolesError` rather than
    returning NaN, which would read as names that were not quoting.
    """

    dates: NDArray
    anchors: NDArray[np.int64]
    tickers: NDArray
    quotes: NDArray[np.float64] | None
    raw_targets: NDArray[np.float64]
    month: str

    @classmethod
    def from_cache(
        cls, root: str | Path, month: str, *, anchors_per_day: int,
        stats_tag: str | None = None,
    ) -> "MarketPanel":
        """Read a month's market columns, skipping the cached views entirely."""
        directory = find_cached_month(
            root, month, anchors_per_day=anchors_per_day, stats_tag=stats_tag
        )
        load = lambda name: np.load(directory / name, allow_pickle=False)  # noqa: E731
        quotes = (load("quotes.npy").astype(np.float64)
                  if (directory / "quotes.npy").is_file() else None)
        raw = load("raw_targets.npy").astype(np.float64)
        dates, anchors, tickers = load("dates.npy"), load("anchors.npy"), load("tickers.npy")
        lengths = {len(raw), len(dates), len(anchors), len(tickers)}
        if quotes is not None:
            lengths.add(len(quotes))
        if len(lengths) != 1:
            raise RuntimeError(f"panel cache at {directory} has ragged columns")
        return cls(dates=dates, anchors=anchors.astype(np.int64), tickers=tickers,
                   quotes=quotes, raw_targets=raw, month=month)

    @property
    def has_quotes(self) -> bool:
        return self.quotes is not None

    def _quotes(self) -> NDArray[np.float64]:
        if self.quotes is None:
            raise MissingRolesError(
                f"the {self.month} panel's dataset has no bid_price or ask_price")
        return self.quotes

    @property
    def bid(self) -> NDArray[np.float64]:
        return self._quotes()[:, 0]

    @property
    def ask(self) -> NDArray[np.float64]:
        return self._quotes()[:, 1]

    def half_spread(self) -> NDArray[np.float64]:
        """Quoted half-spread over mid; NaN where the book was not two-sided.

        A crossed book (ask below bid) is treated as unquoted rather than
        clamped to zero, because a negative half-spread would book a REBATE
        for crossing the spread -- turning the most illiquid rows in the
        archive into the most profitable ones.
        """
        bid, ask = self.bid, self.ask
        broken = ~np.isfinite(bid) | ~np.isfinite(ask) | (bid <= 0) | (ask < bid)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(broken, np.nan, (ask - bid) / 2 / ((bid + ask) / 2))

    def align(self, dates: ArrayLike, anchors: ArrayLike,
              tickers: ArrayLike) -> NDArray[np.int64]:
        """Row indices of this panel matching the supplied rows; -1 for misses.

        The caller is expected to CHECK THE MISS RATE rather than trust it. A
        silent miss is the failure mode that matters here: an unmatched row
        carries no quote, so it would be priced as a free trade in a market
        that never existed.
        """
        keys = _row_keys(self.dates, self.anchors, self.tickers)
        if len(np.unique(keys)) != len(keys):
            raise RuntimeError(f"{self.month} panel has duplicate row keys")
        lookup = {key: index for index, key in enumerate(keys.tolist())}
        wanted = _row_keys(dates, anchors, tickers)
        return np.array([lookup.get(key, -1) for key in wanted.tolist()],
                        dtype=np.int64)

    def take(self, index: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """Quotes, half-spread and raw targets for aligned rows; NaN on a miss.

        ``quote`` and ``half_spread`` are absent, not NaN, for a panel without
        quotes.
        """
        idx = np.asarray(index, dtype=np.int64)
        hit = idx >= 0
        safe = np.where(hit, idx, 0)
        blank = lambda a: np.where(hit[:, None], a[safe], np.nan)  # noqa: E731
        out = {"raw_targets": blank(self.raw_targets), "matched": hit}
        if self.has_quotes:
            out["quote"] = blank(self.quotes)
            out["half_spread"] = np.where(hit, self.half_spread()[safe], np.nan)
        return out

    def market(self, index: ArrayLike, decisions: ArrayLike, assets: ArrayLike, *,
               decision_axis: ArrayLike, asset_axis: ArrayLike) -> MarketData:
        """The aligned rows as :class:`MarketData` on a ``(decision, asset)`` grid.

        ``index`` is :meth:`align`'s output for rows labelled by ``decisions``
        and ``assets``; the grid axes are ``decision_axis`` and ``asset_axis``.
        A missed row is NaN, a name not observed. A panel without quotes gives
        market data without bid and ask, which the execution stages refuse.
        """
        if not self.has_quotes:
            return MarketData.empty(decision_axis, asset_axis)
        taken = self.take(index)
        grid = to_grid(taken["quote"], decisions, assets,
                       decision_axis=decision_axis, asset_axis=asset_axis)
        return MarketData({"bid_price": grid[:, :, 0], "ask_price": grid[:, :, 1]},
                          decision_axis, asset_axis)
