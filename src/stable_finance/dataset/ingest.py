"""Ingest any time series: columns plus roles in, sessions and stores out.

Market-1T has its own reader for Polygon parquet (:mod:`.datasets.market1t.ingest`).
Everything else comes in here, as a table of observations with a time
column, an optional ticker column, and whichever market columns exist::

    schema = FeatureSchema.from_roles(mid="px", volume="qty")
    timeline = Timeline(ContinuousSession("UTC"), bar_seconds=60)
    sessions = sessions_from_frame(frame, schema, timeline, time="ts", stamp="start")

A single price series needs nothing but its values; it is read as the
midpoint::

    sessions = sessions_from_series(timestamps, prices,
                                    timeline=Timeline(FixedSession(), bar_seconds=60),
                                    stamp="start")

The same records feed every layout: :func:`write_mosaic` writes the sparse
MDS mosaic (``densify``, ``build_targets`` and the day-store converter then
read it like any Market-1T month), and :func:`write_daystore` builds the
day-major store straight from sessions without MDS.

EVERY SERIES STATES HOW LONG A ROW IS AND WHICH END OF IT ITS TIMESTAMP
MARKS: the first through its :class:`~stable_finance.dataset.spec.timeline.Timeline`,
the second as ``stamp``. Neither can be defaulted safely: a
minute series read as one-second data has every horizon and window sixty
times too short, and a bar stamped at its start but read as stamped at its
end puts the minute AFTER each timestamp into every view that ends there --
lookahead into the very window a return is measured from. ``stamp="start"``
timestamps are moved to the end of their bar, so stored rows always hold
only what was known by their stamp (:mod:`.spec.timeline`).

TIMESTAMPS. Epoch seconds as integers, or ``datetime64``, or tz-aware
pandas datetimes. Each must fall on its session's bar grid; one that does
not is refused, not dropped or snapped, because snapping silently merges two
observations into one bar and dropping silently loses data. A grid coarser
than declared (every timestamp a multiple of 60 s when 1 s was declared) is
refused too, since it is almost always a mis-declared bar length; pass
``check_bar=False`` for sparse data that happens to land on a coarse grid.

SESSION ASSIGNMENT. An observation belongs to the first open session whose
last bar ends at or after it. Observations before that session's open are
kept: the last one seeds the forward fill at the open, as an overnight quote
seeds Market-1T's first row.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np

from stable_finance.dataset.loader.grid import SessionPreprocessor
from stable_finance.dataset.spec.schema import FeatureSchema, MarketSession
from stable_finance.dataset.spec.timeline import Timeline, as_timeline, to_end_stamps


def epoch_seconds(values) -> np.ndarray:
    """Integer epoch seconds from ints, ``datetime64``, or pandas datetimes."""
    dt_accessor = getattr(values, "dt", None)
    if dt_accessor is not None and getattr(dt_accessor, "tz", None) is not None:
        values = dt_accessor.tz_convert("UTC").dt.tz_localize(None)
    array = np.asarray(values)
    if array.dtype.kind == "M":
        return array.astype("datetime64[s]").astype(np.int64)
    if array.dtype.kind in "iu":
        return array.astype(np.int64)
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)):
            raise ValueError("float timestamps must be whole epoch seconds")
        return array.astype(np.int64)
    raise TypeError(f"cannot read timestamps of dtype {array.dtype}")


def _sessions_covering(timeline: Timeline, first: int, last: int):
    """Open sessions' grids ``(date, first_stamp, end)`` that end after ``first``."""
    start = dt.datetime.fromtimestamp(first, dt.timezone.utc).date() - dt.timedelta(days=1)
    stop = dt.datetime.fromtimestamp(last, dt.timezone.utc).date() + dt.timedelta(days=2)
    out = []
    day = start
    while day <= stop:
        date = day.isoformat()
        if timeline.is_open(date):
            lo, hi = timeline.bounds(date)
            if hi > first:
                out.append((date, int(lo), int(hi)))
        day += dt.timedelta(days=1)
    out.sort(key=lambda item: item[2])
    return out


def _check_bars(ts: np.ndarray, owner: np.ndarray, sessions, bar: int, check_bar: bool) -> None:
    """Every in-session stamp on its session's grid, and the grid not coarser."""
    starts = np.asarray([lo for _, lo, _ in sessions], dtype=np.int64)[owner]
    inside = ts >= starts
    offsets = ts[inside] - starts[inside]
    off_grid = offsets % bar != 0
    if off_grid.any():
        example = int(ts[inside][np.argmax(off_grid)])
        raise ValueError(
            f"{int(off_grid.sum())} timestamps are not on the {bar}-second bar grid "
            f"(e.g. {example}); check the timeline's bar_seconds and the stamp")
    if check_bar and len(offsets) > 1:
        step = math.gcd(*np.unique(offsets // bar).tolist())
        if step > 1:
            raise ValueError(
                f"every timestamp is a multiple of {step * bar}s past the open, but the "
                f"timeline's bar_seconds={bar}: the bars are probably {step * bar}s long "
                "(pass check_bar=False for sparse data on a coarse grid)")


def records_from_arrays(
    timestamps,
    features,
    timeline,
    *,
    stamp: str,
    tickers=None,
    schema: FeatureSchema | None = None,
    check_bar: bool = True,
) -> Iterator[dict]:
    """Sparse ticker-day records (``ticker``, ``date``, ``ts_interval``, ``features``).

    ``features`` is ``(n, C)`` in the schema's column order (or ``(n,)`` for
    one column). ``timeline`` is the dataset's clock (a calendar alone means
    one-second bars) and ``stamp`` which end of its bar each timestamp marks
    (``"start"`` or ``"end"``); the records carry end stamps. ``tickers`` is
    one label per row, or None for a single series named ``"series"``.
    """
    timeline = as_timeline(timeline)
    bar = timeline.bar_seconds
    ts = to_end_stamps(epoch_seconds(timestamps), bar, stamp)
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(ts):
        raise ValueError("features must be (n, C) with one row per timestamp")
    if schema is not None and values.shape[1] != len(schema.columns):
        raise ValueError(f"features have {values.shape[1]} columns, schema {len(schema.columns)}")
    labels = (np.full(len(ts), "series", dtype=object) if tickers is None
              else np.asarray(tickers, dtype=object))
    if len(labels) != len(ts):
        raise ValueError("tickers must have one label per row")
    if not len(ts):
        return
    if ts.max() > np.iinfo(np.int32).max:
        raise ValueError("timestamps past 2038 do not fit the mosaic's int32 seconds")
    sessions = _sessions_covering(timeline, int(ts.min()), int(ts.max()))
    if not sessions:
        return
    # A stamp equal to a grid's end is that session's last bar; the grid's end
    # is exclusive, so the comparison is against end - bar.
    last_bar = np.asarray([hi - bar for _, _, hi in sessions], dtype=np.int64)
    owner = np.searchsorted(last_bar, ts, side="left")
    keep = owner < len(sessions)
    ts, values, labels, owner = ts[keep], values[keep], labels[keep], owner[keep]
    _check_bars(ts, owner, sessions, bar, check_bar)
    order = np.lexsort((ts, owner, labels.astype(str)))
    ts, values, labels, owner = ts[order], values[order], labels[order], owner[order]
    keys = np.char.add(labels.astype(str), np.char.add("\x00", owner.astype(str)))
    boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, len(ts)]):
        date = sessions[int(owner[lo])][0]
        stamps = ts[lo:hi]
        if len(stamps) > 1 and np.any(stamps[1:] == stamps[:-1]):
            raise ValueError(f"{labels[lo]} {date}: duplicate timestamps")
        yield {
            "ticker": str(labels[lo]),
            "date": date,
            "ts_interval": stamps.astype(np.int32),
            "features": values[lo:hi].astype(np.float32),
        }


def records_from_frame(frame: Mapping, schema: FeatureSchema, timeline, *, stamp: str,
                       time: str = "timestamp", ticker: str | None = "ticker",
                       check_bar: bool = True) -> Iterator[dict]:
    """Sparse records from a table: a pandas DataFrame or a mapping of columns.

    The schema's columns are read by name; ``ticker=None`` (or a table with
    no such column) treats the whole table as one series.
    """
    columns = list(frame.keys()) if not hasattr(frame, "columns") else list(frame.columns)
    missing = [c for c in (time, *schema.columns) if c not in columns]
    if missing:
        raise KeyError(f"table has no columns {missing}")
    features = np.column_stack([np.asarray(frame[c], dtype=np.float64) for c in schema.columns])
    labels = frame[ticker] if ticker is not None and ticker in columns else None
    yield from records_from_arrays(frame[time], features, timeline, stamp=stamp, tickers=labels,
                                   schema=schema, check_bar=check_bar)


def _sessions(records: Iterable[dict], schema: FeatureSchema, timeline) -> list[MarketSession]:
    preprocessor = SessionPreprocessor(schema=schema, timeline=as_timeline(timeline))
    out = []
    for record in records:
        session = preprocessor.transform(record)
        if session is not None:
            out.append(session)
    return out


def sessions_from_frame(frame: Mapping, schema: FeatureSchema, timeline, *, stamp: str,
                        time: str = "timestamp", ticker: str | None = "ticker",
                        check_bar: bool = True) -> list[MarketSession]:
    """Dense :class:`MarketSession` objects, one per (ticker, session)."""
    records = records_from_frame(frame, schema, timeline, stamp=stamp, time=time,
                                 ticker=ticker, check_bar=check_bar)
    return _sessions(records, schema, timeline)


def sessions_from_series(timestamps, prices, *, timeline, stamp: str, ticker: str = "series",
                         column: str = "price", check_bar: bool = True) -> list[MarketSession]:
    """Sessions from one price series, read as the midpoint."""
    schema = FeatureSchema.univariate(column)
    labels = np.full(len(np.asarray(prices)), ticker, dtype=object)
    records = records_from_arrays(timestamps, np.asarray(prices, dtype=np.float64), timeline,
                                  stamp=stamp, tickers=labels, schema=schema,
                                  check_bar=check_bar)
    return _sessions(records, schema, timeline)


def dense_record(session: MarketSession) -> dict:
    """A session as a dense-mosaic record, the form the day store is built from."""
    return {
        "ticker": str(session.ticker),
        "date": str(session.date),
        "grid_start": int(session.timestamps[0]),
        "features": np.asarray(session.features, dtype=np.float32),
    }


def write_mosaic(records: Iterable[dict], root: str | Path, schema: FeatureSchema, timeline, *,
                 dense: bool = False) -> dict[str, int]:
    """Write records as a mosaic, one ``root/YYYY/MM`` directory per month.

    Needs the ``mds`` extra. Returns records written per month.
    """
    from stable_finance.dataset.storage.mosaic import write_records

    timeline = as_timeline(timeline)
    by_month: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_month[str(record["date"])[:7]].append(record)
    written = {}
    for month in sorted(by_month):
        year, mm = month.split("-")
        written[month] = write_records(by_month[month], Path(root) / year / mm, schema,
                                       timeline=timeline, dense=dense)
    return written


def write_daystore(sessions: Iterable[MarketSession], root: str | Path, timeline, *,
                   objective=None, compress: bool = False) -> list[Path]:
    """Build the day-major store directly from sessions, without MDS.

    Every session must share one schema and the timeline's bar length.
    ``objective`` defines the stored labels (default: every family the schema
    supports at its richest price). ``compress`` adds the zstd copy (needs
    ``zstandard``).
    """
    from stable_finance.dataset.storage.daystore import (
        build_day,
        check_chronology,
        month_dir,
        write_day,
        write_month_index,
    )
    from stable_finance.dataset.storage.meta import write_store_meta

    timeline = as_timeline(timeline)
    by_date: dict[str, list[MarketSession]] = defaultdict(list)
    for session in sessions:
        by_date[session.date].append(session)
    if not by_date:
        raise ValueError("no sessions to write")
    schema = next(iter(by_date.values()))[0].schema
    for group in by_date.values():
        for session in group:
            if session.schema != schema or session.bar_seconds != timeline.bar_seconds:
                raise ValueError("sessions disagree on schema, or with the timeline's bar length")
    entries: dict[str, list[dict]] = defaultdict(list)
    written = []
    for date in sorted(by_date):
        arrays = build_day([dense_record(s) for s in by_date[date]], date, timeline,
                           schema=schema, objective=objective)
        check_chronology(arrays)
        mdir = month_dir(root, date[:7])
        mdir.mkdir(parents=True, exist_ok=True)
        entries[date[:7]].append(write_day(mdir / date, arrays, compress=compress))
        written.append(mdir / date)
    for month, month_entries in entries.items():
        mdir = month_dir(root, month)
        write_store_meta(mdir, schema, timeline=timeline)
        write_month_index(mdir, month_entries)
    return written
