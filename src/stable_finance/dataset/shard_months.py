"""Convert every month of raw Polygon snapshots to MDS, resumably and in parallel.

The month-level driver over ``write_mds``: discovers every ``YYYY/MM``
directory under the raw root (or takes an explicit list / range), skips
months whose output already holds an index (so a re-run after a crash only
does the remainder; ``--force`` re-shards them), runs ``GROUP_SIZE`` months
at a time as separate processes, and keeps one log per month.

Usage:
    python -m stable_finance.dataset.shard_months                      # all months
    python -m stable_finance.dataset.shard_months 2020-07 2021-08      # just these
    python -m stable_finance.dataset.shard_months --start 2015-01 --end 2019-12
    python -m stable_finance.dataset.shard_months --group-size 4 --force

Paths come from the flags or the environment: RAW_DATA_DIR, MOSAIC_DIR,
METADATA_PATH, HOLIDAY_CSV. The output is the SPARSE tick layout;
``densify`` turns it into the dense sessions training reads, and
``daystore`` regroups those by day.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def discover_months(raw_root: Path, start: str | None, end: str | None) -> list[str]:
    months = []
    for year_dir in sorted(raw_root.glob("[0-9][0-9][0-9][0-9]")):
        for month_dir in sorted(year_dir.glob("[0-9][0-9]")):
            if not month_dir.is_dir():
                continue
            ym = f"{year_dir.name}-{month_dir.name}"
            if start and ym < start:
                continue
            if end and ym > end:
                continue
            months.append(ym)
    return months


def already_sharded(mosaic_dir: Path, ym: str) -> bool:
    out = mosaic_dir / ym[:4] / ym[5:7]
    return out.is_dir() and any(out.iterdir())


def shard_command(ym: str, raw_root: Path, mosaic_dir: Path, metadata: str | None,
                  holiday_csv: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "stable_finance.dataset.write_mds",
           "--start", ym, "--end", ym, "--base-path", str(raw_root),
           "--output", str(mosaic_dir), "--cal-freq", "mnth", "--n-writers", "1"]
    if metadata:
        cmd += ["--metadata", metadata]
    if holiday_csv:
        cmd += ["--holiday-csv", holiday_csv]
    return cmd


def run(months: list[str], *, raw_root: Path, mosaic_dir: Path, log_dir: Path,
        metadata: str | None, holiday_csv: str | None, group_size: int,
        force: bool, log=print) -> list[str]:
    """Shard ``months`` in groups; returns the months that failed."""
    log_dir.mkdir(parents=True, exist_ok=True)
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    pending = [ym for ym in months if force or not already_sharded(mosaic_dir, ym)]
    skipped = [ym for ym in months if ym not in pending]
    if skipped:
        log(f"Skipping {len(skipped)} already-sharded months: {' '.join(skipped)}")
    if not pending:
        log(f"Nothing to do -- all {len(months)} months already present in {mosaic_dir}.")
        return []
    log(f"Sharding {len(pending)} months in groups of {group_size}. Logs: {log_dir}")
    failed: list[str] = []
    for i in range(0, len(pending), group_size):
        group = pending[i:i + group_size]
        log(f"=== Group {i // group_size + 1}: {' '.join(group)} ===")
        procs = []
        for ym in group:
            logf = open(log_dir / f"{ym}.log", "w")
            procs.append((ym, logf, subprocess.Popen(
                shard_command(ym, raw_root, mosaic_dir, metadata, holiday_csv),
                stdout=logf, stderr=subprocess.STDOUT)))
        for ym, logf, proc in procs:
            rc = proc.wait()
            logf.close()
            if rc == 0:
                log(f"  [{ym}] OK")
            else:
                log(f"  [{ym}] FAILED (exit {rc}, see {log_dir / (ym + '.log')})")
                failed.append(ym)
    if failed:
        log(f"Completed with {len(failed)} failures: {' '.join(failed)}")
    else:
        log(f"All {len(pending)} months sharded successfully.")
    return failed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("months", nargs="*", help="YYYY-MM ...; default: discover under --raw-dir")
    p.add_argument("--raw-dir", default=os.environ.get("RAW_DATA_DIR"),
                   required=not os.environ.get("RAW_DATA_DIR"))
    p.add_argument("--mosaic-dir", default=os.environ.get("MOSAIC_DIR"),
                   required=not os.environ.get("MOSAIC_DIR"))
    p.add_argument("--metadata", default=os.environ.get("METADATA_PATH"))
    p.add_argument("--holiday-csv", default=os.environ.get("HOLIDAY_CSV"))
    p.add_argument("--log-dir", default=None, help="default: <mosaic-dir>/.shard_logs")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--group-size", type=int, default=int(os.environ.get("GROUP_SIZE", "8")))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    raw_root, mosaic_dir = Path(args.raw_dir), Path(args.mosaic_dir)
    months = list(args.months) or discover_months(raw_root, args.start, args.end)
    if not months:
        raise SystemExit(f"No months to shard under {raw_root}.")
    failed = run(months, raw_root=raw_root, mosaic_dir=mosaic_dir,
                 log_dir=Path(args.log_dir) if args.log_dir else mosaic_dir / ".shard_logs",
                 metadata=args.metadata, holiday_csv=args.holiday_csv,
                 group_size=args.group_size, force=args.force)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
