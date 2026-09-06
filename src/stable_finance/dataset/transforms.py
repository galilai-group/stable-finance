"""Reusable transformations from dense sessions to model-ready views."""

from __future__ import annotations

import numpy as np

from stable_finance.dataset.schema import MARKET_SCHEMA, FeatureSchema

EPS = 1e-5

FEATURE_TYPES: dict[str, tuple[str, ...]] = {
    "price": ("bid_price", "ask_price", "vwap_all", "high", "low", "mid_price"),
    "order_book_size": ("bid_size", "ask_size"),
    "trade_size": ("volume",),
    "trade_count": ("n",),
}
LOG1P_TYPES = {"order_book_size", "trade_size", "trade_count"}


def build_norm_groups(feature_columns: list[str] | tuple[str, ...]):
    """Return normalization index groups for a dataset column selection."""
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
    schema: FeatureSchema = MARKET_SCHEMA,
) -> np.ndarray | None:
    """Aggregate a dense 1 Hz view according to its feature schema."""
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
    volume_index = schema.index("volume") if "volume" in schema.columns else None

    def reduce(part: np.ndarray, destination: np.ndarray) -> None:
        for index, column in enumerate(schema.columns):
            rule = rules[column]
            values = part[..., :, index]
            if rule == "last":
                destination[..., index] = values[..., -1]
            elif rule == "max":
                destination[..., index] = values.max(axis=-1)
            elif rule == "min":
                destination[..., index] = values.min(axis=-1)
            elif rule == "sum":
                destination[..., index] = values.sum(axis=-1, dtype=np.float64)
            elif rule == "vwap":
                if volume_index is None:
                    destination[..., index] = values.mean(axis=-1, dtype=np.float64)
                    continue
                volume = part[..., :, volume_index]
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


def prior_vwap(
    features: np.ndarray,
    window_start_idx: int,
    *,
    schema: FeatureSchema = MARKET_SCHEMA,
) -> float | None:
    """Return the last valid, traded VWAP preceding a view."""
    if window_start_idx <= 0:
        return None
    vwap_index, count_index = schema.index("vwap_all"), schema.index("n")
    history = features[:window_start_idx]
    valid = (history[:, count_index] > 0) & np.isfinite(history[:, vwap_index])
    return float(history[np.flatnonzero(valid)[-1], vwap_index]) if valid.any() else None


def ffill_vwap(
    view: np.ndarray,
    prior: float | None,
    *,
    schema: FeatureSchema = MARKET_SCHEMA,
) -> None:
    """Fill zero-volume VWAP buckets in place before normalization."""
    vwap_index = schema.index("vwap_all")
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
    if missing.any():
        bid, ask = schema.index("bid_price"), schema.index("ask_price")
        values[missing] = (view[missing, bid] + view[missing, ask]) / 2


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
