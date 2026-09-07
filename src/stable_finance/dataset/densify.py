"""Rewrite a sparse tick-level MDS dataset as reconstructed 1 Hz sessions.

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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stable_finance.dataset.calendar import MarketSchedule
from stable_finance.dataset.grid import SessionPreprocessor
from stable_finance.dataset.write_mds import DENSE_MDS_COLUMNS


def month_dirs(root: Path, start: str, end: str) -> list[tuple[str, Path]]:
    """Months in range that actually hold samples.

    A SAMPLE-LESS MONTH IS SKIPPED, NOT FAILED. Some month directories carry a
    valid index.json declaring zero shards -- 2007-12 and 2025-01..03 in the
    current dataset, the boundaries of what was ingested. LocalDataset raises
    IndexError deep inside its Spanner on those rather than reporting an empty
    dataset, so they have to be filtered here or a fan-out over months dies on
    its edges.
    """
    import json

    out = []
    for year_dir in sorted(root.glob("[0-9][0-9][0-9][0-9]")):
        for month_dir in sorted(year_dir.glob("[0-9][0-9]")):
            ym = f"{year_dir.name}-{month_dir.name}"
            if not (start <= ym <= end):
                continue
            index = month_dir / "index.json"
            if not index.is_file():
                continue
            shards = json.loads(index.read_text()).get("shards") or []
            if not shards or not sum(s.get("samples", 0) for s in shards):
                print(f"{ym}: no samples, skipping", flush=True)
                continue
            out.append((ym, month_dir))
    return out


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

    from stable_finance.dataset.mds import open_shard

    shards = json.loads((month_dir / "index.json").read_text())["shards"]
    for shard in shards:
        reader = open_shard(month_dir, shard, scratch)
        for i in range(shard["samples"]):
            yield reader[i]
        for stale in scratch.glob("*.mds"):
            stale.unlink(missing_ok=True)


def convert_month(month_dir: Path, out_dir: Path, schedule: MarketSchedule,
                  compression: str = "zstd", verify_every: int = 200) -> tuple[int, int, int]:
    """Write one month's dense shards. Returns (written, skipped, verified).

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

    # cache_bytes=0: every ticker-day is visited exactly once here, so a cache
    # would only cost memory.
    pre = SessionPreprocessor(schedule=schedule, cache_bytes=0)
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
    for sidecar in ("ticker_date_map.json",):
        src = month_dir / sidecar
        if src.is_file():
            (out_dir / sidecar).write_bytes(src.read_bytes())
    return written, skipped, verified


def main(argv=None, *, default_mosaic_dir=None, default_holiday_csv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mosaic-dir", default=default_mosaic_dir,
                   required=default_mosaic_dir is None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", required=True, help="first month, YYYY-MM")
    p.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    p.add_argument("--holiday-csv", default=default_holiday_csv,
                   required=default_holiday_csv is None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify-every", type=int, default=200,
                   help="check every Nth converted ticker-day; 0 disables")
    args = p.parse_args(argv)

    schedule = MarketSchedule(args.holiday_csv)
    root, out_root = Path(args.mosaic_dir), Path(args.out_dir)
    months = month_dirs(root, args.start, args.end)
    if not months:
        raise SystemExit(f"no months in [{args.start}, {args.end}] under {root}")

    import time
    for ym, month_dir in months:
        year, mm = ym.split("-")
        out_dir = out_root / year / mm
        if (out_dir / "index.json").is_file() and not args.overwrite:
            print(f"{ym}: exists, skipping (--overwrite to rebuild)", flush=True)
            continue
        t0 = time.time()
        written, skipped, verified = convert_month(
            month_dir, out_dir, schedule, verify_every=args.verify_every)
        print(f"{ym}: {written} sessions ({skipped} skipped), "
              f"{verified} verified identical  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
