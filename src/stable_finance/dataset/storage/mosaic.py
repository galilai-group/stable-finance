"""The ticker-day MDS mosaic: its records, reader and writer, and the conversions between its forms.

MOSAIC. MosaicML Streaming backend: the ticker-day mosaic, for any dataset.

One MDS sample is one (ticker, date). Two record layouts exist:

``MDS_COLUMNS`` (sparse)
    ``ts_interval`` epoch seconds of each observation and ``features`` the
    observed rows, ``(n_obs, n_columns)``. The grid is rebuilt at read time.
``DENSE_MDS_COLUMNS`` (dense)
    ``grid_start`` the epoch second of the first row and ``features`` the
    reconstructed session, forward- and zero-filled.

Each month directory also carries ``dataset.json`` (:mod:`.meta`) naming the
schema, calendar and bar spacing, so a reader needs no dataset constant.

DENSIFY. Rewrite a sparse tick-level MDS dataset as reconstructed dense sessions.

WHY. Rebuilding the grid from ticks is 71% of the CPU a dataloader worker
spends on a sample -- 2.89 ms of 4.08 ms, against 0.25 ms for the MDS read and
0.95 ms for cropping two global views -- and it is recomputed identically every
time a ticker-day is visited. Storing the reconstruction removes that work
outright, and removes the reason the per-worker grid cache exists.

IT IS ALSO SMALLER ON THE WIRE, which is the counterintuitive part. A filled
grid is 1.74x more raw bytes than the ticks it came from, but it is runs of
repeated values, which zstd removes; the sparse form pays for a monotone int32
timestamp column that does not compress. Measured at 0.75-0.80x of the sparse
record across 2009-06, 2015-06 and 2023-01 at zstd levels 1, 3 and 9. The cost
lands on NODE-LOCAL disk instead, where shards are materialized raw: 1.74x.

REGULAR SESSION ONLY, deliberately. Extended-hours bounds compress to almost
the same size (0.83x vs 0.80x -- pre- and post-market are nearly pure fill
runs) but take node-local raw storage to 3.37x, and only 3.6% of stored ticks
fall outside the regular session. Callers that need extended hours read the
sparse dataset, which is not deleted by this conversion.

This is a CONVERSION, not a re-ingest: it reads the existing MDS rather than
the source parquet, so it costs one densify per ticker-day and no parsing. The
output is byte-identical to what ``SessionPreprocessor`` would have built at
read time, which is the property ``verify`` checks.

RESHARD. Rewrite a dense mosaic month with its samples ordered by DATE.

WHY THIS EXISTS. The writer shuffles: ``write_mds`` accumulates
``BUFFER_FLUSH_PARTITIONS`` partitions and shuffles before flushing, so a
79-sample shard holds 17-19 distinct dates and roughly four tickers per date.
That is the right layout for an i.i.d. sampler and the wrong one for a
CROSS-SECTIONAL objective.

The supervised head is trained by ``SupervisedModel._within_cell_loss``, which
ranks only among samples sharing a ``(date, anchor)`` -- one cross-section, the
unit the reported rank IC is computed over. Cross-cell pairs are excluded, not
downweighted, because comparing a stock at 13:00 Tuesday against one at 15:30
Thursday is a different question from the one the metric asks.

With the shuffled layout those pairs are scarce: a random batch of 256 rows
yields ~72 usable pairs, so most of the batch produces no gradient.

A CORRECTION IS RECORDED HERE ON PURPOSE, because an earlier version of this
file drew the wrong conclusion from the right measurement. It claimed the
head's rank-IC deficit against a ridge probe on the SAME frozen embeddings
(+0.0767 vs +0.1418) WAS the pair count, on the strength of an offline refit
at ~48 pairs reproducing the shipped head to within 0.0005. That agreement was
a coincidence, and it was persuasive enough to stop the investigation.

The actual cause was that ``_within_cell_loss`` had never executed.
``collate_bucketed`` dropped ``xs_cell``, so ``bucket.get("xs_cell")`` was
None on every step and every supervised head trained on the FLAT cross-cell
surrogate instead. Fixed in market-jepa 00fbb35; collating the key raised
pairs from ~48 to 71.5 on this same unchanged mosaic.

So date ordering is a pair-SUPPLY optimization and nothing more. It does not
explain the historical head-probe gap and must not be quoted as though it
does. Whether more pairs buys accuracy on top of the fix is an open question
this module exists to let us measure, not one it has answered.

WHAT ORDERING BUYS. A ticker-day record holds the whole session, so the anchor
is a free choice at crop time -- a cross-section never has to be materialized
or duplicated, it only needs enough same-date tickers resident in one batch.
Ordering by date makes a batch date-local by sequential read, which lifts the
expected pairs per 256-row batch from ~48 to ~907 even with every sample
drawing its own anchor, and to ~32,640 if the batch shares one.

IT IS A PURE RE-LAYOUT. Every record is copied through byte-identical --
same ticker, date, grid_start and features -- and the count is asserted. No
session is rebuilt, nothing is re-derived from ticks, and the source month is
never modified. That is what makes it safe to run against a mosaic other
results already depend on: point a consumer at the new directory or the old
one and it reads the same data.

THE COST, STATED. Date-local batches are correlated -- same regime, same news,
same market-wide shock. The ranking loss wants that correlation and the
backbone's own learning may not, so a sampler should draw SEVERAL dates per
batch rather than one.

WHICH IS WHY THE SHUFFLE KNOBS MATTER MORE THAN THE ORDERING. Measured with
MosaicML streaming on a resharded month: ``py1s`` is useless here (22.3
dates/batch, i.e. it undoes the layout), while ``py1br`` with
``shuffle_block_size=128`` gives 5.7 dates and 426.7 pairs per batch -- 5.8x
the supply at a date count that still mixes regimes. Ordering alone, with
shuffling off, reaches 1261 pairs at 1.3 dates/batch, which is the degenerate
end of the trade and not recommended.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from stable_finance.dataset.datasets.market1t.spec import MarketSchedule
from stable_finance.dataset.loader.grid import SessionPreprocessor
from stable_finance.dataset.storage.locations import (
    Month,
    infer_period_frequency,
    month_dirs,
    period_directories,
    validate_period_alignment,
)
from stable_finance.dataset.storage.meta import (
    DENSE_MDS_COLUMNS,
    MDS_COLUMNS,
    STORE_META,
    read_store_meta,
    require_holiday_file,
    write_store_meta,
)


# ── mosaic ───────────────────────────────────────────────────────────────────

try:
    from streaming import Stream, StreamingDataset
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "MDS support requires `uv sync --extra mds`"
    ) from error


def flush_observations(buffer: list[dict], writer) -> int:
    """Shuffle and write buffered observation dicts to MDS. Returns count written."""
    if not buffer:
        return 0
    random.shuffle(buffer)
    for obs in buffer:
        writer.write(obs)
    return len(buffer)


def write_records(records: Iterable[dict], out_dir: str | Path, schema, *,
                  timeline, dense: bool = False,
                  shuffle_buffer: int = 20_000, compression: str | None = "zstd") -> int:
    """Write sparse (or ``dense``) ticker-day records as one mosaic directory.

    Records are shuffled through a buffer of ``shuffle_buffer`` before they
    are written, as the Market-1T writer does, so an i.i.d. sampler reading
    the directory sees mixed dates and tickers. ``dataset.json`` is written
    last: MDSWriter empties the directory when it opens it.
    """
    from streaming import MDSWriter

    out_dir = Path(out_dir)
    width = len(schema.columns)
    total = 0
    buffer: list[dict] = []
    with MDSWriter(out=str(out_dir), columns=DENSE_MDS_COLUMNS if dense else MDS_COLUMNS,
                   compression=compression, exist_ok=True) as writer:
        for record in records:
            features = np.asarray(record["features"], dtype=np.float32)
            if features.ndim != 2 or features.shape[1] != width:
                raise ValueError(
                    f"{record.get('ticker')} {record.get('date')}: features "
                    f"{features.shape} do not match the {width}-column schema")
            row = {"ticker": str(record["ticker"]), "date": str(record["date"]),
                   "features": features}
            if dense:
                row["grid_start"] = int(record["grid_start"])
            else:
                row["ts_interval"] = np.asarray(record["ts_interval"], dtype=np.int32)
            buffer.append(row)
            if len(buffer) >= shuffle_buffer:
                total += flush_observations(buffer, writer)
                buffer = []
        total += flush_observations(buffer, writer)
    write_store_meta(out_dir, schema, timeline=timeline)
    return total


def discover_streams(*args, **kwargs):
    """Build MosaicML streams for :func:`period_directories`."""
    try:
        directories = period_directories(*args, **kwargs)
    except ValueError as error:
        message = str(error)
        if message.startswith("No dataset period directories"):
            message = message.replace(
                "No dataset period directories", "No MDS period directories", 1
            )
            raise ValueError(message) from error
        raise
    return [Stream(local=str(path)) for path in directories]


def shard_list(month_dir: str | Path) -> list[dict]:
    with open(Path(month_dir) / "index.json") as file:
        return json.load(file)["shards"]


def open_shard(month_dir: str | Path, shard_meta: dict, tmpdir: str | Path):
    """Read an MDS shard, expanding compressed data into caller-owned scratch."""
    from streaming.base.format import reader_from_json
    import zstandard

    month_dir, tmpdir = Path(month_dir), Path(tmpdir)
    raw = shard_meta["raw_data"]["basename"]
    if (month_dir / raw).is_file():
        return reader_from_json(str(month_dir), None, shard_meta)
    compressed = shard_meta.get("zip_data")
    if not compressed:
        raise FileNotFoundError(f"{month_dir / raw} missing and not compressed")
    destination = tmpdir / raw
    if not destination.is_file():
        partial = destination.with_suffix(destination.suffix + ".part")
        with open(month_dir / compressed["basename"], "rb") as source, open(
            partial, "wb"
        ) as output:
            zstandard.ZstdDecompressor().copy_stream(source, output)
        partial.replace(destination)
    local = dict(shard_meta)
    local["compression"] = None
    local["zip_data"] = None
    shutil.copy(month_dir / "index.json", tmpdir / "index.json")
    return reader_from_json(str(tmpdir), None, local)


def configure_epoch_size(dataset, total_steps: int, batch_size: int) -> None:
    """Resize a MosaicML dataset epoch to cover a complete training run.

    MosaicML otherwise recycles samples at its epoch boundary during a
    step-based run. The ten-batch margin keeps dataloader prefetch from
    crossing that boundary. This mutates the supplied dataset in place.
    """
    if total_steps < 1 or batch_size < 1:
        raise ValueError("total_steps and batch_size must be positive")
    epoch_samples = (total_steps + 10) * batch_size
    for stream in dataset.streams:
        for attribute in ("proportion", "repeat", "choose"):
            if hasattr(stream, attribute):
                delattr(stream, attribute)
    dataset.epoch_size = Stream.apply_weights(
        dataset.streams,
        dataset.samples_per_stream,
        epoch_samples,
        dataset.shuffle_seed,
    )
    dataset.length = math.ceil(
        dataset.epoch_size / dataset._parallel_rank_world.num_ranks
    )


class StreamingMarketDataset(StreamingDataset):
    """MDS sampling plus backend-neutral dense-session preprocessing.

    :meth:`from_month` builds its preprocessor from the month's
    ``dataset.json`` when the caller supplies none.
    """

    def __init__(self, *, preprocessor: SessionPreprocessor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = preprocessor or SessionPreprocessor()

    @classmethod
    def from_month(cls, mosaic_dir: str | Path, month: Month | str, **kwargs):
        """Construct a dataset over one probe-fit or evaluation month."""
        value = Month.parse(month)
        streams = discover_streams(mosaic_dir, value.first_day, value.last_day)
        if kwargs.get("preprocessor") is None and streams:
            meta = read_store_meta(streams[0].local)
            kwargs["preprocessor"] = SessionPreprocessor(schema=meta.schema,
                                                         timeline=meta.timeline())
        return cls(streams=streams, **kwargs)

    def raw_sample(self, index: int) -> dict:
        """Expose the storage record for specialized research adapters."""
        return super().__getitem__(index)

    def __getitem__(self, index: int):
        return self.preprocessor.transform(self.raw_sample(index))


# ── densify ──────────────────────────────────────────────────────────────────

def _samples(month_dir: Path, scratch: Path):
    """Iterate a month's records shard by shard, raw or compressed.

    ``open_shard`` is what makes this safe on the real archive: 49 of 208
    month directories are PARTIALLY materialized, holding fewer raw .mds than
    .mds.zstd, and anything that reads the raw file directly (LocalDataset,
    MDSReader on the month dir) dies on the first shard that was never
    decompressed. open_shard expands those into caller-owned scratch instead.

    The scratch directory must live somewhere with room for one decompressed
    shard at a time -- 64 MB -- and NOT on a small shared /tmp.
    """
    import json


    shards = json.loads((month_dir / "index.json").read_text())["shards"]
    for shard in shards:
        reader = open_shard(month_dir, shard, scratch)
        for i in range(shard["samples"]):
            yield reader[i]
        for stale in scratch.glob("*.mds"):
            stale.unlink(missing_ok=True)


def month_timeline(month_dir: Path, schedule=None):
    """The timeline a month was written on.

    A month records its own timeline. A Market-1T month takes ``schedule``
    (its holiday file, or a whole timeline) for its short days, since those
    live in a file the store does not carry.
    """
    return read_store_meta(month_dir).timeline(schedule)


def densify_month(month_dir: Path, out_dir: Path, schedule=None,
                  compression: str = "zstd", verify_every: int = 200) -> tuple[int, int, int]:
    """Write one month's dense shards. Returns (written, skipped, verified).

    The schema and timeline come from the month's ``dataset.json`` and are
    copied to the output; ``schedule`` is a Market-1T month's holiday file.

    VERIFICATION HAPPENS HERE, not on a re-read, and it checks the invariant
    that matters: a dense record must reproduce the session a reader would
    have built from the sparse one. Both directions are exercised in memory --
    the array about to be written is compared against a fresh
    ``SessionPreprocessor`` build, and the record dict is then passed back
    through ``transform`` to confirm the dense read path returns the same
    session. Re-reading the shard would test MosaicML's serializer instead,
    which is not what is at risk, and cannot be done in place anyway: a
    freshly written compressed shard has to be materialized before MDSReader
    will open it.
    """
    import tempfile

    from streaming import MDSWriter

    meta = read_store_meta(month_dir)
    timeline = meta.timeline(schedule)
    # cache_bytes=0: every ticker-day is visited exactly once here, so a cache
    # would only cost memory.
    pre = SessionPreprocessor(schema=meta.schema, timeline=timeline, cache_bytes=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = verified = 0
    # Scratch beside the output, not in the system temp: shard expansion needs
    # room and /tmp is commonly a small shared volume.
    scratch_root = tempfile.TemporaryDirectory(prefix="densify_", dir=str(out_dir.parent))
    scratch = Path(scratch_root.name)
    with MDSWriter(out=str(out_dir), columns=DENSE_MDS_COLUMNS,
                   compression=compression, exist_ok=True) as writer:
        for sample in _samples(month_dir, scratch):
            session = pre.transform(sample)
            if session is None:
                # A closed date, or ticks that never reach the session. The
                # sparse record can hold either; the dense one cannot
                # represent them, and the reader dropped them anyway.
                skipped += 1
                continue
            record = {
                "ticker": str(session.ticker),
                "date": str(session.date),
                "grid_start": int(session.timestamps[0]),
                "features": session.features.astype(np.float32),
            }
            if verify_every and written % verify_every == 0:
                back = pre.transform(record)
                assert back is not None, (record["ticker"], record["date"])
                np.testing.assert_array_equal(
                    back.features.astype(np.float32), record["features"])
                np.testing.assert_array_equal(back.timestamps, session.timestamps)
                verified += 1
            writer.write(record)
            written += 1
    scratch_root.cleanup()
    # After the writer: MDSWriter empties its directory when it opens it.
    write_store_meta(out_dir, meta.schema, timeline=timeline)
    for sidecar in ("ticker_date_map.json",):
        src = month_dir / sidecar
        if src.is_file():
            (out_dir / sidecar).write_bytes(src.read_bytes())
    return written, skipped, verified


def densify_main(argv=None, *, default_mosaic_dir=None, default_holiday_csv=None):
    p = argparse.ArgumentParser(description=__doc__)
    default_mosaic_dir = default_mosaic_dir or os.environ.get("MOSAIC_DIR")
    default_holiday_csv = default_holiday_csv or os.environ.get("HOLIDAY_CSV")
    p.add_argument("--mosaic-dir", default=default_mosaic_dir,
                   required=default_mosaic_dir is None, help="sparse mosaic root ($MOSAIC_DIR)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", required=True, help="first month, YYYY-MM")
    p.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    p.add_argument("--holiday-csv", default=default_holiday_csv,
                   help="Market-1T holiday file ($HOLIDAY_CSV); required for a "
                        "Market-1T store, unused by one that records its own calendar")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify-every", type=int, default=200,
                   help="check every Nth converted ticker-day; 0 disables")
    args = p.parse_args(argv)

    schedule = MarketSchedule(args.holiday_csv) if args.holiday_csv else None
    root, out_root = Path(args.mosaic_dir), Path(args.out_dir)
    months = month_dirs(root, args.start, args.end)
    if not months:
        raise SystemExit(f"no months in [{args.start}, {args.end}] under {root}")
    require_holiday_file(months, args.holiday_csv)

    import time
    for ym, month_dir in months:
        year, mm = ym.split("-")
        out_dir = out_root / year / mm
        if (out_dir / "index.json").is_file() and not args.overwrite:
            print(f"{ym}: exists, skipping (--overwrite to rebuild)", flush=True)
            continue
        t0 = time.time()
        written, skipped, verified = densify_month(
            month_dir, out_dir, schedule, verify_every=args.verify_every)
        print(f"{ym}: {written} sessions ({skipped} skipped), "
              f"{verified} verified identical  ({time.time() - t0:.0f}s)", flush=True)


# ── reshard ──────────────────────────────────────────────────────────────────

# A month is ~11 GiB dense. Loading one is the simple, single-read path; a
# larger month is refused rather than silently thrashing.
DEFAULT_MAX_GIB = 48.0


def month_bytes(month_dir: Path) -> int:
    """On-disk size of a month, from its index rather than a directory walk."""
    idx = json.loads((Path(month_dir) / "index.json").read_text())
    total = 0
    for s in idx.get("shards", []):
        total += s["raw_data"]["bytes"]
        zd = s.get("zip_data")
        if zd:
            total += zd["bytes"]
    return total


def read_month(month_dir: Path, scratch_dir: Path | str) -> list[dict]:
    """Every record of a dense month, in stored order.

    ``scratch_dir`` MUST be on a volume with room, and one shard is expanded at
    a time. ``open_shard`` decompresses into scratch, so a month-long
    TemporaryDirectory accumulates the whole month UNCOMPRESSED -- ~11 GiB
    against a /tmp that is commonly a small shared volume (9.8 GiB on bll01,
    which is exactly how this failed the first time). densify.py already
    carries the same warning; this is the second place to learn it.
    """
    month_dir = Path(month_dir)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for meta in shard_list(month_dir):
        # Per-shard, so only one expansion is on disk at any moment.
        with tempfile.TemporaryDirectory(prefix="reshard_read_",
                                         dir=str(scratch_dir)) as td:
            reader = open_shard(month_dir, meta, td)
            for i in range(meta["samples"]):
                s = reader.get_item(i)
                out.append({
                    "ticker": str(s["ticker"]),
                    "date": str(s["date"]),
                    "grid_start": int(s["grid_start"]),
                    "features": np.asarray(s["features"], dtype=np.float32),
                })
    return out


def reshard_month(src: Path, dst: Path, *, compression: str = "zstd",
                  size_limit: int | None = None,
                  max_gib: float = DEFAULT_MAX_GIB,
                  verbose: bool = True) -> dict:
    """Copy ``src`` to ``dst`` with records sorted by ``(date, ticker)``.

    Returns a summary dict. ``dst`` must not already hold an index.
    """
    from streaming import MDSWriter

    src, dst = Path(src), Path(dst)
    if (dst / "index.json").exists():
        raise FileExistsError(f"{dst} already holds an MDS index; refusing")
    gib = month_bytes(src) / 2 ** 30
    if gib > max_gib:
        raise MemoryError(
            f"{src} is {gib:.1f} GiB, over max_gib={max_gib}. This path reads "
            "the month into memory; raise the cap only if the box has room.")

    records = read_month(src, scratch_dir=dst.parent)
    n_in = len(records)
    if not n_in:
        raise ValueError(f"{src} holds no samples")
    # (date, ticker) is a total order on a dense month: one record per pair.
    records.sort(key=lambda r: (r["date"], r["ticker"]))

    dst.mkdir(parents=True, exist_ok=True)
    kw = dict(out=str(dst), columns=DENSE_MDS_COLUMNS,
              compression=compression, exist_ok=True)
    if size_limit is not None:
        kw["size_limit"] = int(size_limit)
    with MDSWriter(**kw) as w:
        for r in records:
            w.write(r)
    # After the writer: MDSWriter empties its directory when it opens it.
    if (src / STORE_META).is_file():
        (dst / STORE_META).write_bytes((src / STORE_META).read_bytes())

    dates = sorted({r["date"] for r in records})
    per_date = [sum(1 for r in records if r["date"] == d) for d in dates]
    summary = {
        "src": str(src), "dst": str(dst), "samples": n_in,
        "dates": len(dates),
        "tickers_per_date_median": int(np.median(per_date)),
        "gib_in": round(gib, 2),
    }
    if verbose:
        print(f"  {src.parent.name}/{src.name}: {n_in} samples, "
              f"{len(dates)} dates, median {summary['tickers_per_date_median']} "
              f"tickers/date -> {dst}")
    return summary


def verify_reshard(src: Path, dst: Path) -> dict:
    """Assert the copy is a permutation of the source, and report date locality.

    Content equality is checked on the (ticker, date) key set and on a sampled
    set of feature arrays -- a full array compare would re-read both months a
    second time for a property the writer cannot violate one record at a time.
    """
    dst = Path(dst)
    a = read_month(Path(src), scratch_dir=dst.parent)
    b = read_month(dst, scratch_dir=dst.parent)
    if len(a) != len(b):
        raise AssertionError(f"sample count {len(a)} -> {len(b)}")
    ka = sorted((r["date"], r["ticker"]) for r in a)
    kb = sorted((r["date"], r["ticker"]) for r in b)
    if ka != kb:
        raise AssertionError("(date, ticker) key sets differ")
    by_key = {(r["date"], r["ticker"]): r for r in a}
    rng = np.random.default_rng(0)
    for i in rng.choice(len(b), size=min(64, len(b)), replace=False):
        r = b[int(i)]
        o = by_key[(r["date"], r["ticker"])]
        if r["grid_start"] != o["grid_start"]:
            raise AssertionError(f"grid_start differs for {r['ticker']} {r['date']}")
        if not np.array_equal(r["features"], o["features"]):
            raise AssertionError(f"features differ for {r['ticker']} {r['date']}")

    def locality(recs, window=79):
        """Mean distinct dates in a sliding window the size of one shard."""
        d = [r["date"] for r in recs]
        return float(np.mean([len(set(d[i:i + window]))
                              for i in range(0, len(d) - window, window)]))
    return {"samples": len(a),
            "dates_per_shard_before": round(locality(a), 1),
            "dates_per_shard_after": round(locality(b), 1)}


def reshard_main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--src-root", required=True,
                   help="mosaic root holding <YYYY>/<MM>/ month dirs")
    p.add_argument("--dst-root", required=True)
    p.add_argument("--months", nargs="+", required=True, help="YYYY-MM ...")
    p.add_argument("--size-limit", type=int, default=None)
    p.add_argument("--max-gib", type=float, default=DEFAULT_MAX_GIB)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args(argv)

    src_root, dst_root = Path(a.src_root), Path(a.dst_root)
    for ym in a.months:
        y, m = ym.split("-")
        src, dst = src_root / y / m, dst_root / y / m
        if not (src / "index.json").is_file():
            raise SystemExit(f"no month at {src}")
        reshard_month(src, dst, size_limit=a.size_limit, max_gib=a.max_gib)
        if a.verify:
            print(f"    verify: {verify_reshard(src, dst)}")
    return 0
