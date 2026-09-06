"""Sparse observation reconstruction and session preprocessing."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from stable_finance.dataset.calendar import MarketSchedule, timeline_bounds_est
from stable_finance.dataset.schema import MARKET_SCHEMA, FeatureSchema, MarketSession


def sparse_to_dense_grid(
    ts_sec: np.ndarray,
    features: np.ndarray,
    ts_open_sec: int,
    ts_close_sec: int,
    ffill_indices: list[int],
    zerofill_indices: list[int],
    trim_leading_nan: bool = True,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map sparse dataset observations onto a canonical one-second grid."""
    if len(ts_sec) == 0:
        return None
    if features.ndim != 2 or len(features) != len(ts_sec):
        raise ValueError("features must be 2-D with one row per timestamp")
    if ts_close_sec <= ts_open_sec:
        raise ValueError("session close must be after its open")

    canonical = np.arange(ts_open_sec, ts_close_sec, dtype=np.int32)
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
    """

    def __init__(
        self,
        *,
        schema: FeatureSchema = MARKET_SCHEMA,
        schedule: MarketSchedule | None = None,
        extended_hours: bool = False,
        zero_features: tuple[str, ...] = (),
        cache_bytes: int = 0,
    ) -> None:
        unknown = set(zero_features) - set(schema.columns)
        if unknown:
            raise ValueError(f"zero_features contains unknown columns: {sorted(unknown)}")
        self.schema = schema
        self.schedule = schedule
        self.extended_hours = extended_hours
        self.zero_indices = [schema.index(name) for name in zero_features]
        self.cache_bytes = max(0, int(cache_bytes))
        self._cache: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._cached_bytes = 0

    def transform(self, sample: Mapping) -> MarketSession | None:
        ticker, date = str(sample.get("ticker", "")), str(sample.get("date", ""))
        if self.schedule is not None and self.schedule.is_closed(date):
            return None
        key = (ticker, date)
        cached = self._cache.get(key) if ticker and date else None
        if cached is not None:
            self._cache.move_to_end(key)
            timestamps, stored = cached
            return MarketSession(ticker, date, timestamps, stored.astype(np.float64), self.schema)

        start, end = timeline_bounds_est(
            date, schedule=self.schedule, extended_hours=self.extended_hours
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
        return MarketSession(ticker, date, timestamps, features, self.schema)

