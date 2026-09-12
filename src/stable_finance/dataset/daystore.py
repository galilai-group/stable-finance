"""Day-major storage of the dense 1 Hz panel: one record per trading day.

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
                       anchors, types, horizons, dtype, version
    features.npy       (N, C, rows) float32. Row r is second ``open_tod + r``
                       past the standard 09:30 open. Rows before a ticker's
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

``{root}/{YYYY}/{MM}/index.json`` lists the month's days with their sizes so a
span can be discovered and budgeted without opening a record.

INVARIANTS, checked by ``verify_day``: every ticker-day of the source month is
present once; ``features[i, :, first_row[i]:].T`` is byte-identical to the
dense mosaic record; ``targets_raw[:, i]`` equals ``anchor_targets`` of that record
on the anchor grid; and the ``uniform`` transform equals what
``AnchorTargetStats`` returns from a table built by ``build_targets`` on the
same month (tests/test_daystore.py).

WHAT IS NOT HERE. Extended hours (the dense mosaic is the regular session
only, and so is this). Risk-factor channels, which stay in
``1Hz_risk_factors``. The industry map, which is monthly and lives beside the
mosaic. Nothing consults the MDS index at read time: a consumer of this
layout does not need ``streaming`` installed.
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

from stable_finance.dataset.anchors import DEFAULT_TARGET_HORIZONS
from stable_finance.dataset.calendar import (
    MarketSchedule,
    standard_open_est,
    timeline_bounds_est,
)
from stable_finance.dataset.months import Month
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES,
    anchor_indices,
    anchor_targets,
)
from stable_finance.dataset.schema import MARKET_SCHEMA
from stable_finance.dataset.targets import MIN_NAMES

FORMAT_VERSION = 1
FEATURES = "features.npy"
FEATURES_ZST = "features.npy.zst"
TRANSFORMS = ("zscore", "uniform", "rank")
# Empirical-CDF resolution of the rank target; the same grid build_targets
# uses (N_QUANTILES there), so the two transforms agree to float precision.
N_QUANTILES = 64
QUANTILE_LEVELS = np.linspace(0.0, 1.0, N_QUANTILES)
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
    """
    root = Path(root)
    start, end = str(date_start)[:10], str(date_end)[:10]
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
        """Row of a wall-clock anchor (seconds past the standard open)."""
        return int(anchor_tod) - self.open_tod

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

    def targets(self, transform: str, anchor_tod: int, idx, types, horizons) -> np.ndarray:
        """``(K, len(types) * len(horizons))`` float32, type-major like
        :func:`stable_finance.dataset.outcomes.get_target_names`.

        NaN for an anchor off the grid or a (type, horizon) the record does
        not carry -- the loss masks NaN, so an unavailable label costs a row,
        never a crash inside a worker.
        """
        idx = np.asarray(idx, dtype=np.int64)
        ti = [self._type_index.get(str(t), -1) for t in types]
        hi = [self._h_index.get(int(h), -1) for h in horizons]
        ai = self._anchor_index.get(int(anchor_tod))
        out = np.full((len(idx), len(ti) * len(hi)), np.nan, dtype=np.float32)
        if ai is None or min(ti) < 0 or min(hi) < 0:
            return out
        block = np.asarray(self.targets_array(transform)[ai, idx])   # (K, T, H)
        out[:] = block[:, ti][:, :, hi].reshape(len(idx), -1)
        return out

    def record(self, i: int) -> dict:
        """The ticker-day as a dense-mosaic record (ticker/date/grid_start/features).

        The compatibility surface: anything that consumes MDS dense records --
        the panel builder, the target-table builder, ``SessionPreprocessor``
        -- reads this dict unchanged.
        """
        i = int(i)
        first = int(self.first_row[i])
        return {
            "ticker": str(self.tickers[i]),
            "date": self.date,
            "grid_start": standard_open_est(self.date) + self.open_tod + first,
            "features": np.ascontiguousarray(np.asarray(self.features[i, :, first:]).T),
        }

    def iter_records(self):
        for i in range(self.n_tickers):
            yield self.record(i)


def iter_span_records(root: str | Path, date_start: str, date_end: str):
    """Dense-mosaic-shaped records for every ticker-day of a span, by date."""
    for path in discover_days(root, date_start, date_end):
        yield from DayRecord(path).iter_records()


# ── cross-sectional transforms ───────────────────────────────────────────────


def _normal_ppf(u: np.ndarray) -> np.ndarray:
    """Vectorized inverse standard-normal CDF (Acklam), the scalar routine in
    ``targets._normal_ppf`` applied elementwise; SciPy is not a dependency."""
    u = np.asarray(u, dtype=np.float64)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398,
         2.445134137142996, 3.754408661907416]
    out = np.full(u.shape, np.nan)
    lo, hi = u < 0.02425, u > 0.97575
    mid = ~(lo | hi) & np.isfinite(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.sqrt(-2 * np.log(u[lo]))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = np.sqrt(-2 * np.log(1 - u[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = u[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


def cross_section_transforms(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Every standard target transform of a day, from its full cross-section.

    ``raw`` is ``(A, N, T, H)``. Returns ``zscore``/``uniform``/``rank`` of the
    same shape and ``count`` of ``(A, T, H)``. The arithmetic follows the
    table path exactly: population moments finalized to float32 and masked
    below MIN_NAMES (``targets.finalize``); the uniform score is the average
    rank among the float32 order statistics over ``n + 1``
    (``AnchorTargetStats.uniform_score``); the Gaussian rank interpolates the
    64-level empirical quantiles and clips to half a level from either end
    (``rank_score``). Only the summation order differs, which is inside float32
    rounding of the stored moments.
    """
    raw = np.asarray(raw)
    if raw.ndim != 4:
        raise ValueError(f"raw targets must be (A, N, T, H); got {raw.shape}")
    A, N, T, H = raw.shape
    # (cells, N): one row per (anchor, type, horizon) cross-section.
    cells = np.ascontiguousarray(
        raw.astype(np.float32).transpose(0, 2, 3, 1).reshape(-1, N))
    finite = np.isfinite(cells)
    count = finite.sum(axis=1)
    n = count.astype(np.float64)
    v = np.where(finite, cells, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = v.sum(axis=1) / n
        variance = (v * v).sum(axis=1) / n - mean * mean
        sigma = np.sqrt(np.maximum(variance, 0.0))
    thin = count < MIN_NAMES
    mean32 = np.where(thin, np.nan, mean).astype(np.float32)
    sigma32 = np.where(thin | (sigma <= 0), np.nan, sigma).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (cells.astype(np.float64) - mean32[:, None]) / sigma32[:, None]
    z = np.where(np.isfinite(sigma32)[:, None] & finite, z, np.nan)

    uniform = np.full(cells.shape, np.nan, dtype=np.float64)
    rank = np.full(cells.shape, np.nan, dtype=np.float64)
    levels = QUANTILE_LEVELS
    bound = 0.5 / (len(levels) * 2)
    for c in range(len(cells)):
        ok = finite[c]
        m = int(count[c])
        if m == 0:
            continue
        vals = cells[c, ok]                      # float32, the table's dtype
        if m >= MIN_NAMES:
            grid = np.sort(vals)
            left = np.searchsorted(grid, vals, side="left")
            right = np.searchsorted(grid, vals, side="right")
            uniform[c, ok] = (left + right + 1.0) / (2.0 * (m + 1.0))
        q = np.nanquantile(vals, levels).astype(np.float32)
        if np.isfinite(q).all():
            p = np.interp(vals.astype(np.float64), q.astype(np.float64), levels)
            rank[c, ok] = _normal_ppf(np.clip(p, bound, 1.0 - bound))

    def back(x):
        return np.ascontiguousarray(
            x.reshape(A, T, H, N).transpose(0, 3, 1, 2)).astype(np.float32)

    return {
        "zscore": back(z), "uniform": back(uniform), "rank": back(rank),
        "count": count.reshape(A, T, H).astype(np.int32),
    }


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
    anchors: np.ndarray              # (A,) int64, seconds past the standard open
    types: tuple[str, ...]
    horizons: tuple[int, ...]


def day_targets(features: np.ndarray, first_row: np.ndarray, anchors_row: np.ndarray,
                horizons, types) -> np.ndarray:
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
        out[keep, i] = anchor_targets(trimmed, hs, rel[keep], types=types)
    return out


def build_day(records: list[dict], date: str, schedule: MarketSchedule | None, *,
              horizons=DEFAULT_TARGET_HORIZONS, types=ANCHOR_TARGET_TYPES) -> DayArrays:
    """Assemble one day's arrays from its dense-mosaic records."""
    if not records:
        raise ValueError(f"{date}: no records")
    ts_open, ts_close = timeline_bounds_est(date, schedule=schedule)
    rows = ts_close - ts_open
    open_tod = ts_open - standard_open_est(date)
    C = len(MARKET_SCHEMA.columns)
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
        first = int(np.asarray(r["grid_start"]).reshape(-1)[0]) - ts_open
        if first < 0 or first + len(f) != rows:
            raise ValueError(
                f"{date} {r['ticker']}: grid_start offset {first} with {len(f)} "
                f"rows does not end at the {rows}-row session close")
        features[i, first:] = f
        first_row[i] = first
    anchors_tod = anchor_indices()
    anchors_row = anchors_tod - open_tod
    raw = day_targets(features, first_row, anchors_row, horizons, types)
    tr = cross_section_transforms(raw)
    count = tr.pop("count")
    features = np.ascontiguousarray(features.transpose(0, 2, 1))   # channel-major
    return DayArrays(
        date=date, tickers=tickers, rows=rows, open_tod=int(open_tod),
        first_row=first_row, features=features,
        targets={"raw": raw, **tr}, count=count, anchors=anchors_tod,
        types=tuple(types), horizons=tuple(int(h) for h in horizons),
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
        "columns": list(MARKET_SCHEMA.columns),
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
    from stable_finance.dataset.mds import open_shard, shard_list

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


def verify_day(arrays: DayArrays, records: list[dict], *, n_check: int = 8,
               rng: np.random.Generator | None = None) -> None:
    """The record round-trips: features identical, targets are anchor_targets."""
    rng = rng or np.random.default_rng(0)
    index = {str(r["ticker"]): r for r in records}
    picks = rng.choice(len(arrays.tickers), size=min(n_check, len(arrays.tickers)),
                       replace=False)
    ts_open = standard_open_est(arrays.date) + arrays.open_tod
    for i in picks:
        i = int(i)
        r = index[arrays.tickers[i]]
        first = int(arrays.first_row[i])
        if int(r["grid_start"]) - ts_open != first:
            raise AssertionError(f"{arrays.date} {arrays.tickers[i]}: first_row")
        back = np.ascontiguousarray(arrays.features[i, :, first:].T)
        np.testing.assert_array_equal(back, np.asarray(r["features"], dtype=np.float32))
        if arrays.rows - first < MIN_ROWS:
            assert not np.isfinite(arrays.targets["raw"][:, i]).any()
            continue
        rel = arrays.anchors - arrays.open_tod - first
        keep = (rel >= 0) & (rel < len(back))
        expect = anchor_targets(back, list(arrays.horizons), rel[keep], types=arrays.types)
        got = arrays.targets["raw"][keep, i]
        np.testing.assert_allclose(got, expect.astype(np.float32), rtol=1e-6, atol=0,
                                   equal_nan=True)
        assert not np.isfinite(arrays.targets["raw"][~keep, i]).any()


def convert_month(month_dir: Path, out_root: Path, schedule: MarketSchedule | None, *,
                  compress: bool = True, keep_raw: bool = True,
                  verify_every: int = 1, overwrite: bool = False,
                  horizons=DEFAULT_TARGET_HORIZONS, types=ANCHOR_TARGET_TYPES,
                  log=print) -> dict:
    """Write every day of one dense mosaic month. Returns the month index."""
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
            if schedule is not None and schedule.is_closed(date):
                log(f"{ym}: {len(records)} records on closed date {date}, dropped")
                continue
            arrays = build_day(records, date, schedule, horizons=horizons, types=types)
            if verify_every and n_days % verify_every == 0:
                verify_day(arrays, records)
                n_verified += 1
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
    from stable_finance.dataset.densify import month_dirs

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
    args = p.parse_args(argv)

    root, out_root = Path(args.mosaic_dir), Path(args.out_dir)
    months = month_dirs(root, args.start, args.end)
    if not months:
        raise SystemExit(f"no months in [{args.start}, {args.end}] under {root}")
    out_root.mkdir(parents=True, exist_ok=True)
    kw = dict(compress=not args.no_compress, keep_raw=not args.no_raw,
              verify_every=args.verify_every, overwrite=args.overwrite)
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


if __name__ == "__main__":
    raise SystemExit(main())
