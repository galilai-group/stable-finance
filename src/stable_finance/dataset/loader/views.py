"""From dense sessions to model-ready views: aggregation, fills, normalization, geometry.

TRANSFORMS. Reusable transformations from dense sessions to model-ready views.

VIEW CONFIGURATION. Model-independent configuration for extracting a view from a session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stable_finance.dataset.spec.schema import (
    FeatureSchema,
    MarketSession,
    default_schema,
    has_mid,
    mid_series,
)


# ── transforms ───────────────────────────────────────────────────────────────

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
