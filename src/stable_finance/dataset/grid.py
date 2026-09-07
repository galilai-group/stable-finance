"""Sparse observation reconstruction and session preprocessing."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from stable_finance.dataset.calendar import MarketSchedule, timeline_bounds_est
from stable_finance.dataset.schema import MARKET_SCHEMA, FeatureSchema, MarketSession


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
        bounds: Callable[[str], tuple[int, int]] | None = None,
    ) -> None:
        unknown = set(zero_features) - set(schema.columns)
        if unknown:
            raise ValueError(f"zero_features contains unknown columns: {sorted(unknown)}")
        self.schema = schema
        self.schedule = schedule
        self.extended_hours = extended_hours
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
        if self.schedule is not None and self.schedule.is_closed(date):
            return None
        if "grid_start" in sample:
            return self._from_dense(sample, ticker, date)
        key = (ticker, date)
        cached = self._cache.get(key) if ticker and date else None
        if cached is not None:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            timestamps, stored = cached
            return MarketSession(ticker, date, timestamps, stored.astype(np.float64), self.schema)

        if self.cache_bytes > 0 and ticker and date:
            self._cache_misses += 1

        start, end = (
            self.bounds(date)
            if self.bounds is not None
            else timeline_bounds_est(
                date, schedule=self.schedule, extended_hours=self.extended_hours
            )
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

    def _from_dense(self, sample: Mapping, ticker: str, date: str) -> MarketSession:
        """Adopt an already-gridded record instead of rebuilding the grid.

        A record written by ``write_mds --dense`` stores the reconstructed 1 Hz
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
        timestamps = np.arange(start, start + len(features), dtype=np.int64)
        features = features.astype(np.float64)
        if self.zero_indices:
            features[:, self.zero_indices] = 0.0
        return MarketSession(ticker, date, timestamps, features, self.schema)
