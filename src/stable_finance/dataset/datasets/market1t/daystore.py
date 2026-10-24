"""The day-major store: whole cross-sections by day, their labels, and the market data and returns read from them.

DAYSTORE. Day-major storage of the dense panel: one record per trading day.

WHY A SECOND LAYOUT. The MDS mosaic stores one TICKER-DAY per record, shuffled
within a month. That is the right unit for an i.i.d. sampler and the wrong one
for everything the project now trains: a cross-sectional CELL -- K stocks
sharing one (date, anchor, resolution) window -- is K random records scattered
over ~114 shards, each fetched, decoded and cropped on its own, plus a target
lookup in a month-wide anchor table per stock. On pythia the loader gets six
CPUs per H100 and that per-view work is what caps the GPU at ~60% busy.

Here a trading day is the record. ``features.npy`` is the whole cross-section
of that day as one memmap-able ``(tickers, channels, session_rows)`` float32
array, dense and forward-filled EXACTLY as the dense mosaic stores it (see
``densify.py``; nothing is re-derived from ticks), so a cell is K x C
contiguous runs of one file and the read cost is the bytes of the views and
nothing else. CHANNEL-MAJOR, unlike the mosaic's ``(rows, channels)`` record:
the loader's aggregation reduces along time per channel, and with channels
innermost every such reduction walked the whole window at a stride of nine
(measured 8 ms of a 12.7 ms cell); with rows innermost it is a contiguous
reduction and the ``(C, T)`` view the model wants needs no transpose. The forward targets at every 5-minute anchor -- raw and all four
cross-sectional transforms -- are computed here, once, from the full
cross-section that is in memory at write time, so training never touches the
anchor tables and never recomputes a target. A pass over a training span is
then a shuffle of (day, anchor, resolution, K tickers) draws with no ordering
constraint from the storage, which is what "shuffle 6 or 12 months together"
requires.

THE RECORD, ``{root}/{YYYY}/{MM}/{YYYY-MM-DD}/``:

    meta.json          date, tickers (sorted), rows, open_tod, columns,
                       anchors, types, horizons, dtype, version, schema,
                       origin, timeline and objective
    features.npy       (N, C, rows) float32. Row r is ``open_tod + r *
                       bar_seconds`` seconds past ``origin`` (Market-1T: the
                       standard 09:30 open). Rows before a ticker's
                       first quote are NaN and are never inside a feasible
                       window; ``first_row`` says where each ticker begins.
    features.npy.zst   the same bytes, zstd, for the wire; a staging step
                       decompresses it and keeps the raw one (``--no-raw``
                       keeps only this copy on the archive host)
    first_row.npy      (N,) int32, the trimmed grid's offset per ticker
    targets_raw.npy    (A, N, T, H) float32: anchor_targets per ticker
    targets_zscore.npy, targets_uniform.npy, targets_rank.npy
                       the cross-sectional transforms, cell by cell, with the
                       same thresholds and arithmetic as ``build_targets``
    count.npy          (A, T, H) int32 finite names per cell

    objectives/<key>/  the same target files for another objective (horizons,
                       execution window, price), cached on request; see
                       ``DayRecord.objective_arrays``

The stored ``targets_*`` are ONE objective's labels -- the record's
``default_objective`` -- kept because the loader needs them fast. They are a
cache, not the definition: any other objective is computed from the stored
features on demand, and an objective's key says which definition a cached
array holds.

``{root}/{YYYY}/{MM}/index.json`` lists the month's days with their sizes so a
span can be discovered and budgeted without opening a record.

INVARIANTS, checked by ``verify_day``: every ticker-day of the source month is
present once; ``features[i, :, first_row[i]:].T`` is byte-identical to the
dense mosaic record; ``targets_raw[:, i]`` equals ``anchor_targets`` of that record
on the anchor grid; and the ``uniform`` transform equals what
``AnchorTargetStats`` returns from a table built by ``build_targets`` on the
same month (tests/test_daystore.py).

A record written before the schema, timeline and objective fields existed
reads exactly as it did: the Market-1T schema at 1 Hz, with its stored types
and horizons at one-minute windows.

WHAT IS NOT HERE. Extended hours (the dense mosaic is the regular session
only, and so is this). Risk-factor channels, which stay in
``1Hz_risk_factors``. The industry map, which is monthly and lives beside the
mosaic. Nothing consults the MDS index at read time: a consumer of this
layout does not need ``streaming`` installed.

EVALUATION INPUTS. Evaluation inputs read from the dataset's stores, independent of any model.

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

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.anchors import (
    DEFAULT_ANCHOR_SPEC,
    DEFAULT_TARGET_HORIZONS,
    AnchorSpec,
)
from stable_finance.dataset.datasets.market1t.spec import (
    MarketSchedule,
    Month,
    Timeline,
    as_timeline,
    is_hub,
    resolve_root,
    standard_open_est,
)
from stable_finance.dataset.objective import Objective, cross_section_transforms
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES,
    anchor_targets,
    supported_targets,
)
from stable_finance.dataset.schema import FeatureSchema, default_schema
from stable_finance.dataset.characteristics import months_since_1926, trailing_from_days
from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.returns.contract import ForwardReturns


# ── daystore ─────────────────────────────────────────────────────────────────

FORMAT_VERSION = 1
FEATURES = "features.npy"
OBJECTIVES = "objectives"
FEATURES_ZST = "features.npy.zst"
TRANSFORMS = ("zscore", "uniform", "rank")
# A ticker-day with fewer rows than this is excluded from every cross-section,
# as build_targets excludes it (``len(res[1]) < 10``); its features are kept.
MIN_ROWS = 10


# ── discovery ────────────────────────────────────────────────────────────────


def day_dir(root: str | Path, date: str) -> Path:
    return Path(root) / date[:4] / date[5:7] / date


def month_dir(root: str | Path, month: Month | str) -> Path:
    value = Month.parse(month)
    return Path(root) / f"{value.year:04d}" / f"{value.month:02d}"


def discover_days(root: str | Path, date_start: str, date_end: str) -> list[Path]:
    """Every day directory in ``[date_start, date_end]``, chronological.

    Uses the month ``index.json`` when it exists (the writer's last act, so its
    presence means the month is complete) and falls back to a directory scan
    for a month that is still being written.

    ``root`` may be a Hub spec (see :mod:`stable_finance.dataset.datasets.market1t.spec`). The
    Hub copy ships ``features.npy.zst`` only, so each day in range is then
    decompressed beside it, once.
    """
    hub = is_hub(root)
    start, end = str(date_start)[:10], str(date_end)[:10]
    root = resolve_root(root, start, end)
    out: list[Path] = []
    cursor, last = Month.parse(start[:7]), Month.parse(end[:7])
    while cursor <= last:
        mdir = month_dir(root, cursor)
        index = mdir / "index.json"
        if index.is_file():
            days = [d["date"] for d in json.loads(index.read_text())["days"]]
        elif mdir.is_dir():
            days = sorted(p.name for p in mdir.iterdir()
                          if p.is_dir() and (p / "meta.json").is_file())
        else:
            days = []
        out.extend(mdir / d for d in days if start <= d <= end)
        cursor = cursor.offset(1)
    if not out:
        raise ValueError(
            f"No daystore days under {root} in [{date_start}, {date_end}]; "
            f"expected {root}/YYYY/MM/YYYY-MM-DD/ records")
    if hub:
        for day in out:
            decompress_day(day, remove_zst=False)
    return out


# ── the record ───────────────────────────────────────────────────────────────


class DayRecord:
    """One trading day, opened as memmaps; nothing is read until sliced."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        if meta.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"{self.path}: daystore version {meta.get('version')} != {FORMAT_VERSION}")
        self.meta = meta
        self.date: str = meta["date"]
        self.tickers = np.asarray(meta["tickers"], dtype=object)
        self.rows: int = int(meta["rows"])
        self.open_tod: int = int(meta["open_tod"])
        self.columns: tuple[str, ...] = tuple(meta["columns"])
        self.schema: FeatureSchema = FeatureSchema.from_dict(meta.get("schema", meta["columns"]))
        if self.schema.columns != self.columns:
            raise ValueError(f"{self.path}: schema columns disagree with columns")
        # Records written before timelines were stored are Market-1T at 1 Hz,
        # whose origin is the standard 09:30 open.
        self.timeline: Timeline = (Timeline.from_dict(meta["timeline"]) if "timeline" in meta
                                   else as_timeline(None))
        self.bar_seconds: int = self.timeline.bar_seconds
        self.origin: int = int(meta["origin"]) if "origin" in meta else standard_open_est(self.date)
        self.anchors = np.asarray(meta["anchors"], dtype=np.int64)
        self.types: tuple[str, ...] = tuple(meta["types"])
        self.horizons = np.asarray(meta["horizons"], dtype=np.int64)
        self.first_row = np.load(self.path / "first_row.npy", allow_pickle=False)
        self._ticker_index = {t: i for i, t in enumerate(self.tickers)}
        self._anchor_index = {int(a): i for i, a in enumerate(self.anchors)}
        self._type_index = {t: i for i, t in enumerate(self.types)}
        self._h_index = {int(h): i for i, h in enumerate(self.horizons)}
        self._features = None
        self._targets: dict[str, np.ndarray] = {}
        self._objectives: dict[str, dict[str, np.ndarray]] = {}
        self._default_objective: Objective | None = None

    # -- lazily opened arrays --------------------------------------------------

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            raw = self.path / FEATURES
            if not raw.is_file():
                raise FileNotFoundError(
                    f"{raw} missing: the record holds only {FEATURES_ZST}; run "
                    "decompress_day (staging) before reading it")
            self._features = np.load(raw, mmap_mode="r", allow_pickle=False)
            expected = (self.n_tickers, len(self.columns), self.rows)
            if self._features.shape != expected:
                raise ValueError(
                    f"{raw}: shape {self._features.shape} != {expected}")
        return self._features

    def targets_array(self, transform: str = "raw") -> np.ndarray:
        if transform not in self._targets:
            name = f"targets_{transform}.npy"
            self._targets[transform] = np.load(
                self.path / name, mmap_mode="r", allow_pickle=False)
        return self._targets[transform]

    @property
    def count(self) -> np.ndarray:
        return np.load(self.path / "count.npy", allow_pickle=False)

    # -- reads -------------------------------------------------------------------

    def ticker_index(self, ticker: str) -> int:
        return self._ticker_index[ticker]

    def anchor_row(self, anchor_tod: int) -> int:
        """Row of an anchor given in seconds past the origin."""
        return (int(anchor_tod) - self.open_tod) // self.bar_seconds

    def feasible_from(self, start_row: int) -> np.ndarray:
        """Tickers whose grid begins at or before ``start_row``."""
        return self.first_row <= int(start_row)

    def window(self, idx, start_row: int, length: int) -> np.ndarray:
        """``(K, C, length)`` float32 copy of ``length`` rows for ``idx`` tickers."""
        start_row = int(start_row)
        if start_row < 0 or start_row + int(length) > self.rows:
            raise ValueError(
                f"window [{start_row}, {start_row + length}) outside the "
                f"{self.rows}-row session of {self.date}")
        idx = np.asarray(idx, dtype=np.int64)
        return np.asarray(self.features[idx, :, start_row:start_row + int(length)])

    def targets(self, transform: str, anchor_tod: int, idx, types, horizons, *,
                objective: Objective | None = None) -> np.ndarray:
        """``(K, len(types) * len(horizons))`` float32, type-major like
        :func:`stable_finance.dataset.outcomes.get_target_names`.

        NaN for an anchor off the grid or a (type, horizon) the record does
        not carry -- the loss masks NaN, so an unavailable label costs a row,
        never a crash inside a worker. ``objective`` reads that objective's
        labels (see :meth:`objective_arrays`); the default is the stored one.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if objective is None:
            type_index, h_index = self._type_index, self._h_index
            array = lambda name: self.targets_array(name)  # noqa: E731
        else:
            resolved = objective.resolve(self.schema)
            arrays = self.objective_arrays(resolved)
            type_index = {t: i for i, t in enumerate(resolved.types)}
            h_index = {int(h): i for i, h in enumerate(resolved.horizons)}
            array = arrays.__getitem__
        ti = [type_index.get(str(t), -1) for t in types]
        hi = [h_index.get(int(h), -1) for h in horizons]
        ai = self._anchor_index.get(int(anchor_tod))
        out = np.full((len(idx), len(ti) * len(hi)), np.nan, dtype=np.float32)
        if ai is None or min(ti) < 0 or min(hi) < 0:
            return out
        block = np.asarray(array(transform)[ai, idx])   # (K, T, H)
        out[:] = block[:, ti][:, :, hi].reshape(len(idx), -1)
        return out

    def labels(self, anchor_tod: int, idx, objective: Objective | None = None, *,
               transform: str = "uniform") -> np.ndarray:
        """``(K, n_labels)`` labels of ``idx`` at an anchor, in ``objective.names`` order.

        The standardized forms come from the FULL cross-section of the cell,
        however few names ``idx`` holds: a batch of K is labelled relative to
        every name observed, not ranked among itself.
        """
        resolved = (objective or self.default_objective).resolve(self.schema)
        return self.targets(transform, anchor_tod, idx, resolved.types, resolved.horizons,
                            objective=resolved)

    # -- objectives --------------------------------------------------------------

    @property
    def default_objective(self) -> Objective:
        """The objective the stored ``targets_*`` arrays hold."""
        if self._default_objective is None:
            # A record written before objectives were stored holds Market-1T's
            # standard labels: its types and horizons, one-minute windows.
            stored = (Objective.from_dict(self.meta["objective"]) if "objective" in self.meta
                      else Objective(types=self.types, horizons=tuple(int(h) for h in self.horizons)))
            self._default_objective = stored.resolve(self.schema)
        return self._default_objective

    def objective_dir(self, objective: Objective, cache_root=None) -> Path:
        """Where ``objective``'s labels for this day are cached."""
        key = objective.resolve(self.schema).key
        base = self.path if cache_root is None else day_dir(cache_root, self.date)
        return base / OBJECTIVES / key

    def objective_arrays(self, objective: Objective | None = None, *, persist: bool = False,
                         cache_root=None) -> dict[str, np.ndarray]:
        """``raw``, ``zscore``, ``uniform``, ``rank`` (A, N, T, H) and ``count`` (A, T, H).

        The stored arrays when ``objective`` is the record's own. Otherwise
        from the objective's cache directory when it exists, else computed
        from the stored features -- every name, every anchor -- and kept in
        memory; ``persist`` also writes them under the objective's key (in
        ``cache_root`` when the store itself is read-only).
        """
        resolved = (objective or self.default_objective).resolve(self.schema)
        if resolved.key == self.default_objective.key:
            names = ("raw", *TRANSFORMS)
            out = {name: self.targets_array(name) for name in names}
            out["count"] = self.count
            return out
        if resolved.key in self._objectives:
            if persist:
                self._write_objective(resolved, self._objectives[resolved.key], cache_root)
            return self._objectives[resolved.key]
        directory = self.objective_dir(resolved, cache_root)
        if (directory / "objective.json").is_file():
            out = {name: np.load(directory / f"targets_{name}.npy", mmap_mode="r",
                                 allow_pickle=False) for name in ("raw", *TRANSFORMS)}
            out["count"] = np.load(directory / "count.npy", allow_pickle=False)
        else:
            raw = self._objective_raw(resolved)
            out = {"raw": raw, **resolved.transform(raw)}
            if persist:
                self._write_objective(resolved, out, cache_root)
        self._objectives[resolved.key] = out
        return out

    def cross_section(self, mask, objective: Objective | None = None) -> dict[str, np.ndarray]:
        """The standardized labels over a restricted cross-section (not cached).

        ``mask`` is ``(N,)`` or ``(A, N)``: True for a name in the cross-section,
        such as a screened universe. Ranks are then among those names only.
        """
        resolved = (objective or self.default_objective).resolve(self.schema)
        raw = np.asarray(self.objective_arrays(resolved)["raw"], dtype=np.float32)
        return resolved.transform(raw, mask)

    def _objective_raw(self, objective: Objective) -> np.ndarray:
        anchors_row = (self.anchors - self.open_tod) // self.bar_seconds
        out = np.full((len(self.anchors), self.n_tickers, len(objective.types),
                       len(objective.horizons)), np.nan, dtype=np.float32)
        for i in range(self.n_tickers):
            first = int(self.first_row[i])
            if self.rows - first < MIN_ROWS:
                continue
            trimmed = np.asarray(self.features[i, :, first:], dtype=np.float64).T
            rel = anchors_row - first
            keep = (rel >= 0) & (rel < len(trimmed))
            if keep.any():
                out[keep, i] = objective.raw(trimmed, rel[keep], schema=self.schema,
                                             timeline=self.timeline)
        return out

    def _write_objective(self, objective: Objective, arrays: dict, cache_root) -> Path:
        directory = self.objective_dir(objective, cache_root)
        if (directory / "objective.json").is_file():
            return directory
        partial = directory.with_name(directory.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        for name in ("raw", *TRANSFORMS):
            np.save(partial / f"targets_{name}.npy", np.asarray(arrays[name], dtype=np.float32))
        np.save(partial / "count.npy", np.asarray(arrays["count"], dtype=np.int32))
        (partial / "objective.json").write_text(json.dumps(
            {**objective.to_dict(), "key": objective.key, "names": objective.names}))
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(partial, directory)
        return directory

    def check_chronology(self) -> None:
        """Raise unless this day agrees with its own timeline (:func:`check_chronology`)."""
        check_chronology(self)

    def market_at(self, anchor_tod: int, roles=None) -> np.ndarray:
        """``(len(roles), N)`` float64: each role's value at an anchor's row.

        ``roles`` defaults to every role the schema maps. NaN for a ticker
        whose grid has not begun, and for an anchor outside the session.
        """
        roles = tuple(self.schema.role_map) if roles is None else tuple(roles)
        columns = [self.schema.role_index(role) for role in roles]
        out = np.full((len(roles), self.n_tickers), np.nan)
        row = self.anchor_row(anchor_tod)
        if (int(anchor_tod) - self.open_tod) % self.bar_seconds or not 0 <= row < self.rows:
            return out
        live = self.first_row <= row
        if live.any() and columns:
            block = np.asarray(self.features[:, columns, row], dtype=np.float64)   # (N, R)
            out[:, live] = block[live].T
        return out

    def record(self, i: int) -> dict:
        """The ticker-day as a dense-mosaic record (ticker/date/grid_start/features).

        The one record format: anything that consumes MDS dense records --
        the panel builder, the target-table builder, ``SessionPreprocessor``
        -- reads this dict unchanged.
        """
        i = int(i)
        first = int(self.first_row[i])
        return {
            "ticker": str(self.tickers[i]),
            "date": self.date,
            "grid_start": self.origin + self.open_tod + first * self.bar_seconds,
            "features": np.ascontiguousarray(np.asarray(self.features[i, :, first:]).T),
        }

    def iter_records(self):
        for i in range(self.n_tickers):
            yield self.record(i)


def materialize_objective(root: str | Path, date_start: str, date_end: str,
                          objective: Objective, *, cache_root=None, log=print) -> int:
    """Compute and cache ``objective``'s labels for every day of a span.

    The cache sits in each day record (or under ``cache_root``) keyed by the
    objective, so a loader asking for it later reads arrays instead of
    recomputing. Days already cached are skipped. Returns the days written.
    """
    written = 0
    for path in discover_days(root, date_start, date_end):
        record = DayRecord(path)
        resolved = objective.resolve(record.schema)
        if resolved.key == record.default_objective.key:
            continue
        if (record.objective_dir(resolved, cache_root) / "objective.json").is_file():
            continue
        record.objective_arrays(resolved, persist=True, cache_root=cache_root)
        written += 1
        log(f"{record.date}: cached objective {resolved.key} ({', '.join(resolved.names[:3])}...)")
    return written


def iter_span_records(root: str | Path, date_start: str, date_end: str):
    """Dense-mosaic-shaped records for every ticker-day of a span, by date."""
    for path in discover_days(root, date_start, date_end):
        yield from DayRecord(path).iter_records()


# ── cross-sectional transforms ───────────────────────────────────────────────


# ── building a day ───────────────────────────────────────────────────────────


@dataclass
class DayArrays:
    date: str
    tickers: list[str]
    rows: int
    open_tod: int
    first_row: np.ndarray            # (N,) int32
    features: np.ndarray             # (N, C, rows) float32
    targets: dict[str, np.ndarray]   # raw + transforms, (A, N, T, H) float32
    count: np.ndarray                # (A, T, H) int32
    anchors: np.ndarray              # (A,) int64, seconds past the origin
    types: tuple[str, ...]
    horizons: tuple[int, ...]
    schema: FeatureSchema | None = None   # default_schema(): Market-1T
    origin: int | None = None             # stamp of the row the decision at 0 reads
    timeline: Timeline | None = None      # the clock the rows are stamped on
    objective: Objective | None = None    # the definition of the stored labels

    def __post_init__(self) -> None:
        self.schema = default_schema(self.schema)
        self.timeline = as_timeline(self.timeline)
        if self.origin is None:
            self.origin = standard_open_est(self.date)
        if self.objective is None:
            self.objective = Objective(types=self.types, horizons=self.horizons).resolve(self.schema)

    @property
    def bar_seconds(self) -> int:
        return self.timeline.bar_seconds


def day_targets(features: np.ndarray, first_row: np.ndarray, anchors_row: np.ndarray,
                horizons, types, *, schema: FeatureSchema | None = None,
                spec: AnchorSpec = DEFAULT_ANCHOR_SPEC,
                price: str | None = None) -> np.ndarray:
    """``(A, N, T, H)`` raw targets from row-major ``(N, rows, C)`` features.

    Per-ticker rather than one vectorized pass so that it is literally
    ``anchor_targets`` -- the function the eval panel and the anchor tables
    use -- applied to the record ``densify`` would have handed either of them.
    A ticker shorter than MIN_ROWS is left NaN, as ``build_targets`` skips it.
    """
    N, rows, _ = features.shape
    hs = list(int(h) for h in horizons)
    types = tuple(types)
    out = np.full((len(anchors_row), N, len(types), len(hs)), np.nan, dtype=np.float32)
    for i in range(N):
        first = int(first_row[i])
        if rows - first < MIN_ROWS:
            continue
        trimmed = features[i, first:]
        rel = anchors_row - first
        keep = (rel >= 0) & (rel < len(trimmed))
        if not keep.any():
            continue
        out[keep, i] = anchor_targets(trimmed, hs, rel[keep], types=types,
                                      spec=spec, schema=schema, price=price)
    return out


def default_types(schema: FeatureSchema, price: str | None = None) -> tuple[str, ...]:
    """The anchor target families ``schema`` can form, in the standard order."""
    supported = set(supported_targets(schema, price))
    return tuple(t for t in ANCHOR_TARGET_TYPES if t in supported)


def build_day(records: list[dict], date: str, schedule=None, *,
              horizons=DEFAULT_TARGET_HORIZONS, types=None,
              schema: FeatureSchema | None = None,
              objective: Objective | None = None) -> DayArrays:
    """Assemble one day's arrays from its dense-mosaic records.

    ``schedule`` is the day's timeline, or a Market-1T holiday schedule (None
    is Market-1T without one). The stored labels are ``objective``'s, which
    becomes the record's ``default_objective``; without one, ``types``
    (default: every family the schema supports) and ``horizons`` define it.
    """
    schema = default_schema(schema)
    if not records:
        raise ValueError(f"{date}: no records")
    timeline = as_timeline(schedule)
    if objective is None:
        objective = Objective(types=None if types is None else tuple(types),
                              horizons=tuple(horizons))
    objective = objective.resolve(schema)
    spec = objective.spec(timeline)
    bar = timeline.bar_seconds
    ts_open, ts_close = timeline.bounds(date)
    rows = (ts_close - ts_open) // bar
    origin = timeline.origin(date)
    open_tod = ts_open - origin
    if open_tod % bar:
        raise ValueError(f"{date}: session opens {open_tod}s past the origin, not on a bar")
    C = len(schema.columns)
    records = sorted(records, key=lambda r: str(r["ticker"]))
    tickers = [str(r["ticker"]) for r in records]
    if len(set(tickers)) != len(tickers):
        dup = sorted({t for t in tickers if tickers.count(t) > 1})
        raise ValueError(f"{date}: duplicate tickers {dup[:5]}")
    N = len(tickers)
    features = np.full((N, rows, C), np.nan, dtype=np.float32)
    first_row = np.zeros(N, dtype=np.int32)
    for i, r in enumerate(records):
        if str(r["date"]) != date:
            raise ValueError(f"record {r['ticker']} is dated {r['date']}, not {date}")
        f = np.asarray(r["features"], dtype=np.float32)
        if f.ndim == 1:
            f = f.reshape(-1, C)
        offset = int(np.asarray(r["grid_start"]).reshape(-1)[0]) - ts_open
        first = offset // bar
        if offset % bar or first < 0 or first + len(f) != rows:
            raise ValueError(
                f"{date} {r['ticker']}: grid_start offset {first} with {len(f)} "
                f"rows does not end at the {rows}-row session close")
        features[i, first:] = f
        first_row[i] = first
    anchors_tod = timeline.decisions()
    anchors_row = (anchors_tod - open_tod) // bar
    raw = day_targets(features, first_row, anchors_row, objective.horizons, objective.types,
                      schema=schema, spec=spec, price=objective.price)
    tr = cross_section_transforms(raw)
    count = tr.pop("count")
    features = np.ascontiguousarray(features.transpose(0, 2, 1))   # channel-major
    return DayArrays(
        date=date, tickers=tickers, rows=rows, open_tod=int(open_tod),
        first_row=first_row, features=features,
        targets={"raw": raw, **tr}, count=count, anchors=anchors_tod,
        types=objective.types, horizons=objective.horizons,
        schema=schema, origin=int(origin), timeline=timeline, objective=objective,
    )


def compress_file(src: Path, dst: Path, level: int = 3) -> None:
    import zstandard

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        zstandard.ZstdCompressor(level=level).copy_stream(fin, fout)


def decompress_day(day: str | Path, *, remove_zst: bool = True) -> Path:
    """Materialize ``features.npy`` from its zstd copy (idempotent, atomic)."""
    import zstandard

    day = Path(day)
    raw, zst = day / FEATURES, day / FEATURES_ZST
    meta = json.loads((day / "meta.json").read_text())
    if raw.is_file() and raw.stat().st_size == int(meta["features_bytes"]):
        if remove_zst:
            zst.unlink(missing_ok=True)
        return raw
    if not zst.is_file():
        raise FileNotFoundError(f"{day}: neither {FEATURES} nor {FEATURES_ZST}")
    partial = day / (FEATURES + ".part")
    with open(zst, "rb") as fin, open(partial, "wb") as fout:
        zstandard.ZstdDecompressor().copy_stream(fin, fout)
    if partial.stat().st_size != int(meta["features_bytes"]):
        partial.unlink()
        raise ValueError(f"{day}: decompressed {FEATURES} has the wrong size")
    os.replace(partial, raw)
    if remove_zst:
        zst.unlink(missing_ok=True)
    return raw


def write_day(out_dir: Path, arrays: DayArrays, *, compress: bool = True,
              keep_raw: bool = True) -> dict:
    """Write one record atomically; returns its index entry."""
    out_dir = Path(out_dir)
    if not compress and not keep_raw:
        raise ValueError("a record needs the raw or the compressed features")
    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.save(partial / FEATURES, arrays.features)
    features_bytes = (partial / FEATURES).stat().st_size
    if compress:
        compress_file(partial / FEATURES, partial / FEATURES_ZST)
    if not keep_raw:
        (partial / FEATURES).unlink()
    np.save(partial / "first_row.npy", arrays.first_row.astype(np.int32))
    for name, value in arrays.targets.items():
        np.save(partial / f"targets_{name}.npy", value.astype(np.float32))
    np.save(partial / "count.npy", arrays.count.astype(np.int32))
    meta = {
        "version": FORMAT_VERSION,
        "date": arrays.date,
        "tickers": list(arrays.tickers),
        "rows": int(arrays.rows),
        "open_tod": int(arrays.open_tod),
        "columns": list(arrays.schema.columns),
        "schema": arrays.schema.to_dict(),
        "origin": int(arrays.origin),
        "timeline": arrays.timeline.to_dict(),
        "objective": {**arrays.objective.to_dict(), "key": arrays.objective.key},
        "layout": "ticker,channel,row",
        "dtype": "float32",
        "features_bytes": int(features_bytes),
        "anchors": [int(a) for a in arrays.anchors],
        "types": list(arrays.types),
        "horizons": [int(h) for h in arrays.horizons],
        "target_transforms": ["raw", *TRANSFORMS],
    }
    (partial / "meta.json").write_text(json.dumps(meta))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(partial, out_dir)
    return {"date": arrays.date, "n_tickers": len(arrays.tickers),
            "rows": int(arrays.rows), "features_bytes": int(features_bytes)}


def write_month_index(mdir: Path, entries: list[dict]) -> None:
    entries = sorted(entries, key=lambda e: e["date"])
    (mdir / "index.json").write_text(json.dumps({
        "version": FORMAT_VERSION, "days": entries,
        "n_ticker_days": int(sum(e["n_tickers"] for e in entries)),
        "features_bytes": int(sum(e["features_bytes"] for e in entries)),
    }, indent=1))


# ── conversion from the dense MDS mosaic ─────────────────────────────────────


def _month_records_by_date(month_dir: Path, scratch: Path):
    """``{date: [record, ...]}`` for a dense MDS month, one date at a time.

    The mosaic is shuffled within the month, so a date's records are spread
    over every shard. Every shard is opened (expanded into ``scratch`` when it
    is stored compressed only), the (date -> [(shard, i)]) map is built from
    the ``ticker_date_map.json`` sidecar when present and by one scan
    otherwise, and dates are then materialized one by one so the peak is a
    single day of arrays rather than the month.
    """
    from stable_finance.dataset.datasets.market1t.mosaic import (
        open_shard,
        shard_list,
    )

    shards = shard_list(month_dir)
    readers = [open_shard(month_dir, s, scratch) for s in shards]
    sidecar = month_dir / "ticker_date_map.json"
    n_total = sum(s["samples"] for s in shards)
    dates_flat: list[str] | None = None
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text())
            if len(meta.get("dates", [])) == n_total:
                dates_flat = [str(d) for d in meta["dates"]]
        except (OSError, ValueError):
            dates_flat = None
    by_date: dict[str, list[tuple[int, int]]] = {}
    pos = 0
    for si, s in enumerate(shards):
        for i in range(s["samples"]):
            d = dates_flat[pos] if dates_flat is not None else str(readers[si].get_item(i)["date"])
            by_date.setdefault(d, []).append((si, i))
            pos += 1
    for d in sorted(by_date):
        recs = []
        for si, i in by_date[d]:
            s = readers[si].get_item(i)
            recs.append({
                "ticker": str(s["ticker"]), "date": str(s["date"]),
                "grid_start": int(s["grid_start"]),
                "features": np.asarray(s["features"], dtype=np.float32),
            })
        yield d, recs


class ChronologyError(ValueError):
    """A stored day is not consistent with its own timeline."""


def check_chronology(day, *, n_names: int = 16,
                     rng: np.random.Generator | None = None) -> None:
    """Raise :class:`ChronologyError` unless ``day`` agrees with its timeline.

    ``day`` is a :class:`DayRecord` or :class:`DayArrays`. Beyond the clock
    checks (:meth:`~stable_finance.dataset.datasets.market1t.spec.Timeline.check_day`),
    each sampled name must have no data before its first row: a value there
    would be information from before the name was observed, placed where a
    view could read it.
    """
    problems = day.timeline.check_day(day.date, rows=day.rows, open_tod=day.open_tod,
                                      origin=day.origin, decisions=day.anchors)
    first_row = np.asarray(day.first_row)
    if ((first_row < 0) | (first_row > day.rows)).any():
        problems.append("a first row lies outside the session")
    rng = rng or np.random.default_rng(0)
    n = len(first_row)
    for i in rng.choice(n, size=min(n_names, n), replace=False) if n else []:
        first = int(first_row[i])
        if first and np.isfinite(np.asarray(day.features[int(i), :, :first])).any():
            problems.append(f"{day.tickers[int(i)]} has data before its first row")
    if problems:
        raise ChronologyError(f"{day.date}: " + "; ".join(problems))


def verify_day(arrays: DayArrays, records: list[dict], *, n_check: int = 8,
               rng: np.random.Generator | None = None) -> None:
    """The record round-trips -- features identical, targets the objective's --
    and agrees with its timeline."""
    check_chronology(arrays)
    rng = rng or np.random.default_rng(0)
    index = {str(r["ticker"]): r for r in records}
    picks = rng.choice(len(arrays.tickers), size=min(n_check, len(arrays.tickers)),
                       replace=False)
    ts_open = arrays.origin + arrays.open_tod
    bar = arrays.bar_seconds
    for i in picks:
        i = int(i)
        r = index[arrays.tickers[i]]
        first = int(arrays.first_row[i])
        if int(r["grid_start"]) - ts_open != first * bar:
            raise AssertionError(f"{arrays.date} {arrays.tickers[i]}: first_row")
        back = np.ascontiguousarray(arrays.features[i, :, first:].T)
        np.testing.assert_array_equal(back, np.asarray(r["features"], dtype=np.float32))
        if arrays.rows - first < MIN_ROWS:
            assert not np.isfinite(arrays.targets["raw"][:, i]).any()
            continue
        rel = (arrays.anchors - arrays.open_tod) // bar - first
        keep = (rel >= 0) & (rel < len(back))
        expect = arrays.objective.raw(back, rel[keep], schema=arrays.schema,
                                      timeline=arrays.timeline)
        got = arrays.targets["raw"][keep, i]
        np.testing.assert_allclose(got, expect.astype(np.float32), rtol=1e-6, atol=0,
                                   equal_nan=True)
        assert not np.isfinite(arrays.targets["raw"][~keep, i]).any()


def convert_month(month_dir: Path, out_root: Path, schedule=None, *,
                  compress: bool = True, keep_raw: bool = True,
                  verify_every: int = 1, overwrite: bool = False,
                  horizons=DEFAULT_TARGET_HORIZONS, types=None,
                  objective: Objective | None = None, log=print) -> dict:
    """Write every day of one dense mosaic month. Returns the month index.

    ``schedule`` is the holiday file (or schedule, or timeline) that gives
    short days their early close. Every day is checked
    against the timeline; ``verify_every`` also round-trips every Nth.
    """
    timeline = as_timeline(schedule)
    month_dir, out_root = Path(month_dir), Path(out_root)
    ym = f"{month_dir.parent.name}-{month_dir.name}"
    mdir = month_dir_for(out_root, ym)
    if (mdir / "index.json").is_file() and not overwrite:
        log(f"{ym}: exists, skipping (--overwrite to rebuild)")
        return json.loads((mdir / "index.json").read_text())
    mdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    entries: list[dict] = []
    n_days = n_verified = 0
    scratch_root = tempfile.TemporaryDirectory(prefix=f"daystore_{ym}_", dir=str(out_root))
    try:
        scratch = Path(scratch_root.name)
        for date, records in _month_records_by_date(month_dir, scratch):
            if not timeline.is_open(date):
                log(f"{ym}: {len(records)} records on closed date {date}, dropped")
                continue
            arrays = build_day(records, date, timeline, horizons=horizons, types=types,
                               objective=objective)
            if verify_every and n_days % verify_every == 0:
                verify_day(arrays, records)
                n_verified += 1
            else:
                check_chronology(arrays)
            entries.append(write_day(mdir / date, arrays, compress=compress, keep_raw=keep_raw))
            n_days += 1
    finally:
        scratch_root.cleanup()
    write_month_index(mdir, entries)
    log(f"{ym}: {n_days} days, {sum(e['n_tickers'] for e in entries)} ticker-days, "
        f"{n_verified} verified, {sum(e['features_bytes'] for e in entries) / 1e9:.1f} GB raw "
        f"({time.time() - t0:.0f}s)")
    return json.loads((mdir / "index.json").read_text())


def month_dir_for(root: Path, ym: str) -> Path:
    return month_dir(root, ym)


def _convert_one(args):
    ym, month_dir, out_root, holiday_csv, kw = args
    schedule = MarketSchedule(holiday_csv) if holiday_csv else None
    return convert_month(Path(month_dir), Path(out_root), schedule,
                         log=lambda m: print(m, flush=True), **kw)


def main(argv=None) -> int:
    from stable_finance.dataset.datasets.market1t.spec import month_dirs

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mosaic-dir", default=os.environ.get("MOSAIC_DIR"),
                   required=not os.environ.get("MOSAIC_DIR"),
                   help="dense MDS mosaic root ($MOSAIC_DIR)")
    p.add_argument("--out-dir", default=os.environ.get("DAYSTORE_DIR"),
                   required=not os.environ.get("DAYSTORE_DIR"),
                   help="daystore root ($DAYSTORE_DIR)")
    p.add_argument("--start", required=True, help="first month, YYYY-MM")
    p.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    p.add_argument("--holiday-csv", default=os.environ.get("HOLIDAY_CSV"),
                   help="market_holidays.csv ($HOLIDAY_CSV); short days need it")
    p.add_argument("--workers", type=int, default=4, help="months in parallel")
    p.add_argument("--no-compress", action="store_true", help="skip the .zst copy")
    p.add_argument("--no-raw", action="store_true",
                   help="keep only the .zst copy (archive host without readers)")
    p.add_argument("--verify-every", type=int, default=1,
                   help="round-trip check every Nth day; 0 disables")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_TARGET_HORIZONS),
                   help="target horizons in seconds (-1 measures to the close)")
    p.add_argument("--window", type=int, default=None,
                   help="measurement (execution) window in seconds; default 60")
    p.add_argument("--price", choices=("vwap", "mid", "close"), default=None,
                   help="price method of the return target (default: the schema's richest)")
    args = p.parse_args(argv)

    root, out_root = Path(args.mosaic_dir), Path(args.out_dir)
    months = month_dirs(root, args.start, args.end)
    if not months:
        raise SystemExit(f"no months in [{args.start}, {args.end}] under {root}")
    out_root.mkdir(parents=True, exist_ok=True)
    kw = dict(compress=not args.no_compress, keep_raw=not args.no_raw,
              verify_every=args.verify_every, overwrite=args.overwrite,
              objective=Objective(horizons=tuple(args.horizons), price=args.price,
                                  window_seconds=args.window if args.window is not None else 60))
    jobs = [(ym, str(md), str(out_root), args.holiday_csv, kw) for ym, md in months]
    if args.workers <= 1:
        for job in jobs:
            _convert_one(job)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_convert_one, job): job[0] for job in jobs}
            failed = []
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - report and continue
                    failed.append(futures[fut])
                    print(f"{futures[fut]}: FAILED: {exc!r}", flush=True)
            if failed:
                raise SystemExit(f"{len(failed)} month(s) failed: {failed}")
    return 0


# ── evaluation inputs ────────────────────────────────────────────────────────

def decision_labels(dates: ArrayLike, anchors: ArrayLike) -> NDArray[np.str_]:
    """One label per (date, anchor), the decision identity of a row.

    A decision is a wall-clock instant shared across the cross-section, so
    every stock observed at the same date and anchor belongs to the same
    decision. Sorting these labels sorts chronologically because the date is
    ISO and the anchor is zero-padded.
    """
    anchor = np.char.zfill(np.char.mod("%d", np.asarray(anchors).astype(np.int64)), 4)
    return np.char.add(np.asarray(dates).astype("U10"), np.char.add("|", anchor))

def _rows_reduce(record: DayRecord, columns: list[int], lo: int, hi: int, how: str) -> np.ndarray:
    """``(len(columns), N)`` reduction of the rows ``[lo, hi)`` of each column.

    ``how`` is ``"mean"``, ``"min"`` or ``"max"`` over the finite values; a
    name with none in the range, or a range off the session, is NaN.
    """
    out = np.full((len(columns), record.n_tickers), np.nan)
    lo, hi = max(int(lo), 0), min(int(hi), record.rows)
    if hi <= lo or not columns:
        return out
    block = np.asarray(record.features[:, columns, lo:hi], dtype=np.float64)   # (N, R, rows)
    finite = np.isfinite(block)
    any_finite = finite.any(axis=2)
    if how == "mean":
        value = np.where(finite, block, 0.0).sum(axis=2) / np.maximum(finite.sum(axis=2), 1)
    elif how == "min":
        value = np.where(finite, block, np.inf).min(axis=2)
    else:
        value = np.where(finite, block, -np.inf).max(axis=2)
    out[:] = np.where(any_finite, value, np.nan).T
    return out


def _closes_and_dollars(record: DayRecord) -> tuple[np.ndarray, np.ndarray]:
    """Each name's last price of the day, and its dollar volume over the day."""
    schema = record.schema
    n = record.n_tickers
    if schema.has("bid_price", "ask_price"):
        bid, ask = (np.asarray(record.features[:, schema.role_index(r), :], dtype=np.float64)
                    for r in ("bid_price", "ask_price"))
        price = np.where(np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid),
                         (bid + ask) / 2, np.nan)
    else:
        role = next((r for r in ("mid", "close", "vwap") if schema.has(r)), None)
        price = (np.asarray(record.features[:, schema.role_index(role), :], dtype=np.float64)
                 if role else np.full((n, record.rows), np.nan))
    finite = np.isfinite(price)
    last = np.where(finite.any(axis=1), record.rows - 1 - np.argmax(finite[:, ::-1], axis=1), 0)
    closes = np.where(finite.any(axis=1), price[np.arange(n), last], np.nan)
    dollars = np.full(n, np.nan)
    traded = next((r for r in ("vwap", "close", "mid") if schema.has(r)), None)
    if schema.has("volume") and traded is not None:
        volume = np.asarray(record.features[:, schema.role_index("volume"), :], dtype=np.float64)
        paid = np.asarray(record.features[:, schema.role_index(traded), :], dtype=np.float64)
        dollars = np.nan_to_num(volume * paid, nan=0.0).sum(axis=1)
    return closes, dollars


def market_from_days(days, *, anchors: ArrayLike | None = None,
                     roles: tuple[str, ...] | None = None, lookback: int = 0,
                     delay_seconds: int = 5, depth_seconds: int = 5,
                     window_seconds: int | None = None) -> MarketData:
    """The market a day store recorded, as :class:`MarketData`, by role.

    ``days`` are day-record paths or :class:`~stable_finance.dataset.datasets.market1t.daystore.DayRecord`
    objects, in date order. Each decision is a (date, anchor) with ``anchors``
    in seconds past the origin (default: each day's own anchor grid) and is
    labelled by :func:`decision_labels`; assets are every ticker seen, sorted.
    A column is included for every role the store's schema maps (or just
    ``roles``), so a dataset's columns flow downstream unchanged and a stage
    that needs one it lacks is refused: a univariate store yields ``mid``
    only, which supports frictionless portfolios and nothing priced at the
    touch.

    Values are the decision row's bar. A name before its first quote, or an
    anchor off the session, is NaN.

    The store's rows after the decision supply what the execution models
    need, so every model runs on the store as it is:

    * ``exec_bid_price`` and ``exec_ask_price``: the touch ``delay_seconds``
      after the decision, the quotes an order sent at the decision meets
      (the first row at or after arrival).
    * a book of one level, when the schema maps ``bid_size`` and
      ``ask_size``: that touch, with each side's quoted size averaged over
      the ``depth_seconds`` before arrival, a few seconds of the top of the
      book standing in for the depth the store never saw.
    * ``window_best_buy`` and ``window_best_sell``, with ``window_seconds``:
      the lowest ask and highest bid in the window after the decision, which
      a limit order is filled against.
    * ``adv``, ``idio_vol`` and ``market_vol``, with ``lookback``: from the
      daily dollar volume and closes of the ``lookback`` records before each
      decision's record. The first ``lookback`` records serve only as that
      history and yield no decisions. The volatilities need twenty daily
      returns in the window (see
      :func:`~stable_finance.dataset.characteristics.trailing_from_days`).
    * ``time_trend``, always: the paper's months since June 1926.

    Market cap is not in the store, so ``impact_stock`` needs a
    ``market_cap`` column added by the caller.
    """

    records = [d if isinstance(d, DayRecord) else DayRecord(d) for d in days]
    if not records:
        raise ValueError("no days")
    if lookback < 0 or lookback >= len(records):
        raise ValueError(f"lookback must leave at least one record to decide on; "
                         f"got {lookback} of {len(records)}")
    if delay_seconds < 0 or depth_seconds < 1:
        raise ValueError("delay_seconds must be non-negative and depth_seconds positive")
    if window_seconds is not None and window_seconds < 1:
        raise ValueError("window_seconds must be positive")
    schema = records[0].schema
    for record in records:
        if record.schema != schema:
            raise ValueError(f"{record.date}: schema differs from {records[0].date}")
    wanted = tuple(schema.role_map) if roles is None else tuple(roles)
    schema.require(*wanted, purpose="market_from_days")
    quoted = schema.has("bid_price", "ask_price")
    sized = quoted and schema.has("bid_size", "ask_size")
    assets = np.asarray(sorted({str(t) for r in records for t in r.tickers}))
    asset_of = {ticker: i for i, ticker in enumerate(assets.tolist())}
    quote_columns = [schema.role_index(r) for r in ("bid_price", "ask_price")] if quoted else []
    size_columns = [schema.role_index(r) for r in ("bid_size", "ask_size")] if sized else []

    if lookback:
        closes = np.full((len(records), len(assets)), np.nan)
        dollars = np.full((len(records), len(assets)), np.nan)
        for i, record in enumerate(records):
            columns = [asset_of[str(t)] for t in record.tickers]
            closes[i, columns], dollars[i, columns] = _closes_and_dollars(record)
        adv, idio, mvol = trailing_from_days(closes, dollars, lookback)

    labels, blocks = [], []
    extra = {name: [] for name in ("exec_bid_price", "exec_ask_price", "book_bid_size",
                                   "book_ask_size", "window_best_buy", "window_best_sell",
                                   "adv", "idio_vol", "market_vol", "time_trend")}
    for i, record in enumerate(records[lookback:], start=lookback):
        grid = (record.anchors if anchors is None
                else np.asarray(anchors, dtype=np.int64))
        columns = [asset_of[str(t)] for t in record.tickers]
        bar = record.bar_seconds
        depth_rows = max(1, -(-depth_seconds // bar))
        trend = float(months_since_1926(np.datetime64(record.date, "s"))[()])
        for anchor in grid:
            values = record.market_at(int(anchor), wanted)
            block = np.full((len(wanted), len(assets)), np.nan)
            block[:, columns] = values
            blocks.append(block)
            labels.append((record.date, int(anchor)))
            row = record.anchor_row(int(anchor))
            arrival = -(-(int(anchor) + delay_seconds - record.open_tod) // bar)
            for name in extra:
                extra[name].append(np.full(len(assets), np.nan))
            if quoted and 0 <= arrival < record.rows:
                live = record.first_row <= arrival
                touch = np.asarray(record.features[:, quote_columns, arrival], dtype=np.float64)
                touch[~live] = np.nan
                extra["exec_bid_price"][-1][columns] = touch[:, 0]
                extra["exec_ask_price"][-1][columns] = touch[:, 1]
                if sized:
                    sizes = _rows_reduce(record, size_columns, arrival + 1 - depth_rows,
                                         arrival + 1, "mean")
                    extra["book_bid_size"][-1][columns] = sizes[0]
                    extra["book_ask_size"][-1][columns] = sizes[1]
            if window_seconds is not None and quoted:
                lo, hi = row + 1, row + 1 + -(-window_seconds // bar)
                extra["window_best_buy"][-1][columns] = _rows_reduce(
                    record, quote_columns[1:], lo, hi, "min")[0]
                extra["window_best_sell"][-1][columns] = _rows_reduce(
                    record, quote_columns[:1], lo, hi, "max")[0]
            if lookback:
                extra["adv"][-1][:] = adv[i]
                extra["idio_vol"][-1][:] = idio[i]
                extra["market_vol"][-1][:] = mvol[i]
            extra["time_trend"][-1][:] = trend
    stacked = np.stack(blocks, axis=1)                   # (roles, decisions, assets)
    decisions = decision_labels([d for d, _ in labels], [a for _, a in labels])
    columns = {role: stacked[k] for k, role in enumerate(wanted)}
    columns["time_trend"] = np.stack(extra["time_trend"])
    if quoted:
        columns["exec_bid_price"] = np.stack(extra["exec_bid_price"])
        columns["exec_ask_price"] = np.stack(extra["exec_ask_price"])
    if sized:
        columns["book_bid_price"] = columns["exec_bid_price"][..., None]
        columns["book_ask_price"] = columns["exec_ask_price"][..., None]
        columns["book_bid_size"] = np.stack(extra["book_bid_size"])[..., None]
        columns["book_ask_size"] = np.stack(extra["book_ask_size"])[..., None]
    if window_seconds is not None and quoted:
        columns["window_best_buy"] = np.stack(extra["window_best_buy"])
        columns["window_best_sell"] = np.stack(extra["window_best_sell"])
    if lookback:
        for name in ("adv", "idio_vol", "market_vol"):
            columns[name] = np.stack(extra[name])
    return MarketData(columns, decisions, assets)


def forward_returns_from_days(days, objective=None, *, target_type: str = "return",
                              transform: str = "raw", anchors: ArrayLike | None = None,
                              persist: bool = False, cache_root=None) -> ForwardReturns:
    """One target family of a day store as :class:`ForwardReturns`.

    The grid is :func:`market_from_days`'s -- decisions are (date, anchor)
    labels, assets every ticker sorted -- so the two align without a join.
    The horizon axis is the objective's horizons. ``objective`` defaults to
    each record's stored one; any other (a different horizon set, execution
    window or price) is computed from the stored features, and cached under
    its key when ``persist``. That is the execution-window sweep::

        for window in (60, 300, 900):
            realized = forward_returns_from_days(days, Objective(window_seconds=window))

    ``transform`` picks raw or a cross-sectional form. Horizons that measure
    to the close have no fixed length and are not accepted here.
    """

    records = [d if isinstance(d, DayRecord) else DayRecord(d) for d in days]
    if not records:
        raise ValueError("no days")
    resolved = [(objective or r.default_objective).resolve(r.schema) for r in records]
    keys = {o.key for o in resolved}
    if len(keys) != 1:
        raise ValueError("the days resolve to different objectives; pass one explicitly")
    objective = resolved[0]
    if target_type not in objective.types:
        raise ValueError(f"{target_type!r} is not among the objective's types {objective.types}")
    if any(h <= 0 for h in objective.horizons):
        raise ValueError("ForwardReturns horizons must be fixed positive durations")
    ti = objective.types.index(target_type)
    assets = np.asarray(sorted({str(t) for r in records for t in r.tickers}))
    asset_of = {ticker: i for i, ticker in enumerate(assets.tolist())}
    labels, blocks = [], []
    for record in records:
        arrays = record.objective_arrays(objective, persist=persist, cache_root=cache_root)
        values = np.asarray(arrays[transform], dtype=np.float64)       # (A, N, T, H)
        anchor_index = {int(a): i for i, a in enumerate(record.anchors)}
        grid = record.anchors if anchors is None else np.asarray(anchors, dtype=np.int64)
        columns = [asset_of[str(t)] for t in record.tickers]
        for anchor in grid:
            block = np.full((len(assets), len(objective.horizons)), np.nan)
            row = anchor_index.get(int(anchor))
            if row is not None:
                block[columns] = values[row, :, ti, :]
            blocks.append(block)
            labels.append((record.date, int(anchor)))
    decisions = decision_labels([d for d, _ in labels], [a for _, a in labels])
    return ForwardReturns(np.stack(blocks), decisions, assets, np.asarray(objective.horizons))


if __name__ == "__main__":
    raise SystemExit(main())
