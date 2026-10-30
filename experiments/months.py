"""Build the v2 month cache for every month of the store, in order, on a small disk.

The day store holds eighteen months (2019-07 to 2020-12), about 1.7 GB of
download each; all of them would not fit beside everything else on a laptop
or this container. What the experiments need from a month is its v2 cache
(:mod:`experiments.month_cache_v2`, about 200 MB), so this walks the months in
date order and, for each: downloads it, builds its cache a day at a time (one
unpacked day of scratch, about 0.6 GB, deleted after the day), and then
deletes that month's download and its per-day files. Peak disk is one month's
download plus one unpacked day, about 2.5 GB, plus the finished caches.

SAFE TO KILL AND RESTART, by construction rather than by care:

* a month whose cache file exists is skipped (and any download it left behind
  is deleted), so a rerun picks up at the first unfinished month;
* inside a month, each day's file is written atomically and skipped when
  present, the download resumes where it stopped, and the unpack is atomic
  (``decompress_day``), so a restart loses at most the day in progress;
* the month file is written atomically and the download deleted only after;
* a lock file stops a second copy from running against the same cache, and
  dies with its process;
* the list of months is cached, so a restart without network can go on with
  months already downloaded.

Progress goes to stdout and to ``months.log`` beside the caches.

    uv run --extra hf --extra mds --extra experiments python -m experiments.months --list
    uv run --extra hf --extra mds --extra experiments python -m experiments.months \\
        --keep-raw 2020-07 2020-08

``--keep-raw`` keeps the downloads of the months named (here the two whose
one-second data the current experiments and smoke tests unpack); ``--v1``
also writes each month's v1 file, cut from the v2 one, so specs on
``"cache": "v1"`` run on every month too.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

from experiments import month_cache_v2 as v2
from experiments.month_cache import ROOT
from stable_finance.dataset.datasets.market1t.spec import cache_root, parse_hub

#: Free disk, in GB, a month needs before it starts: its download (about 1.7),
#: one unpacked day (0.6), its cache (0.2), and a margin.
MIN_FREE_GB = 4.0


class Log:
    """Timestamped lines to stdout and to a log file beside the caches."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.path, "a") as handle:
            handle.write(line + "\n")


def _store_local() -> tuple[Path, str]:
    """The store's local download directory and its subdirectory, as ``resolve_root`` names them."""
    _, repo_id, revision, subdir = parse_hub(ROOT)
    local = cache_root() / repo_id.replace("/", "--")
    if revision:
        local = local.with_name(f"{local.name}@{revision}")
    return local, subdir


def raw_month_dirs(month: str) -> list[Path]:
    """Where a month's download lives: its day records and the Hub's metadata for them."""
    local, subdir = _store_local()
    year, mm = month[:4], month[5:7]
    return [local / subdir / year / mm,
            local / ".cache" / "huggingface" / "download" / subdir / year / mm]


def store_months(log) -> list[str]:
    """Every month of the store, from the month index files the Hub repo lists.

    A month is listed when its ``index.json`` is (the writer's last act, so a
    month still being uploaded is not). The list is cached; without network,
    the cached list is used, else the months downloaded so far.
    """
    listing = v2.cache_dir() / "store_months.json"
    repo_type, repo_id, revision, subdir = parse_hub(ROOT)
    pattern = re.compile(rf"^{re.escape(subdir)}/(\d{{4}})/(\d{{2}})/index\.json$")
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id, repo_type=repo_type, revision=revision)
        months = sorted(f"{m[1]}-{m[2]}" for f in files if (m := pattern.match(f)))
        listing.write_text(json.dumps(months))
        return months
    except (OSError, ImportError) as error:
        if listing.is_file():
            log(f"could not list the Hub ({error!r}); using the cached list")
            return json.loads(listing.read_text())
        local, _ = _store_local()
        log(f"could not list the Hub ({error!r}); using the months downloaded so far")
        return sorted(f"{p.parent.parent.name}-{p.parent.name}"
                      for p in (local / subdir).glob("[0-9]" * 4 + "/[0-9][0-9]/index.json"))


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def clean(month: str, *, keep_raw: bool, keep_days: bool) -> int:
    """Delete a finished month's per-day files and (unless kept) its download; bytes freed."""
    freed = 0
    if not keep_days:
        for day in (v2.cache_dir() / "days-v2").glob(f"{month}-*.npz*"):
            freed += day.stat().st_size
            day.unlink()
    if not keep_raw:
        root = cache_root().resolve()
        for path in raw_month_dirs(month):
            if path.exists():
                # Never delete outside the cache, whatever the store's name resolves to.
                if root not in path.resolve().parents:
                    raise RuntimeError(f"refusing to delete {path}: not under {root}")
                freed += _size(path)
                shutil.rmtree(path)
    return freed


def status(months: list[str]) -> list[dict]:
    out = []
    for month in months:
        target = v2.month_path(month)
        out.append({"month": month, "v2": target.is_file(),
                    "v2_mb": round(_size(target) / 1e6) if target.is_file() else None,
                    "v1": (v2.cache_dir() / f"market1t-{month}.npz").is_file(),
                    "raw_gb": round(_size(raw_month_dirs(month)[0]) / 1e9, 2),
                    "days_v2": len(list((v2.cache_dir() / "days-v2").glob(f"{month}-*.npz")))})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="0000-00", help="first month, YYYY-MM")
    parser.add_argument("--end", default="9999-99", help="last month, YYYY-MM")
    parser.add_argument("--list", action="store_true", help="print each month's state and stop")
    parser.add_argument("--keep-raw", nargs="*", default=(), metavar="YYYY-MM",
                        help="months whose download is kept")
    parser.add_argument("--no-delete", action="store_true", help="keep every download")
    parser.add_argument("--keep-days", action="store_true", help="keep the per-day v2 files")
    parser.add_argument("--v1", action="store_true", help="also write each month's v1 file")
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                        help="stop before a month when less disk than this is free")
    args = parser.parse_args(argv)

    out = v2.cache_dir()
    log = Log(out / "months.log")
    months = [m for m in store_months(log) if args.start <= m <= args.end]
    if args.list:
        for row in status(months):
            print(json.dumps(row))
        return 0

    with open(out / "months.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"another experiments.months holds {out / 'months.lock'}; stopping")
            return 2
        return build_all(months, args, log)


def build_all(months: list[str], args, log) -> int:
    """Build, clean and log each month in order; 0 unless a month failed."""
    out = v2.cache_dir()
    log(f"{len(months)} months: {months[0] if months else '-'} .. {months[-1] if months else '-'} "
        f"-> {out}")
    built, failed, seconds = [], [], []
    for i, month in enumerate(months):
        keep_raw = args.no_delete or month in args.keep_raw
        target = v2.month_path(month)
        if target.is_file():
            if args.v1:
                v2.write_v1(month)
            freed = clean(month, keep_raw=keep_raw, keep_days=args.keep_days)
            log(f"{month} ({i + 1}/{len(months)}): cached, skipped"
                + (f"; deleted {freed / 1e9:.2f} GB left behind" if freed else ""))
            continue
        free = shutil.disk_usage(out).free / 1e9
        if free < args.min_free_gb:
            log(f"{month}: only {free:.1f} GB free, below --min-free-gb {args.min_free_gb}; stopping")
            return 2
        log(f"{month} ({i + 1}/{len(months)}): building ({free:.1f} GB free)")
        t0 = time.time()
        try:
            v2.build_month(month, log=lambda message: log(f"  {message}"))
            if args.v1:
                v2.write_v1(month)
        except (OSError, ValueError, RuntimeError):
            # One bad month must not cost the rest of an unattended run; it is
            # reported at the end, and a rerun retries it.
            log(f"{month}: FAILED\n{traceback.format_exc()}")
            failed.append(month)
            continue
        freed = clean(month, keep_raw=keep_raw, keep_days=args.keep_days)
        seconds.append(time.time() - t0)
        built.append(month)
        left = sum(1 for m in months[i + 1:] if not v2.month_path(m).is_file())
        eta = sum(seconds) / len(seconds) * left / 3600
        log(f"{month}: built in {seconds[-1] / 60:.1f} min, {_size(target) / 1e6:.0f} MB; "
            f"freed {freed / 1e9:.2f} GB{' (download kept)' if keep_raw else ''}; "
            f"{shutil.disk_usage(out).free / 1e9:.1f} GB free; {left} to go, about {eta:.1f} h")
    log(f"done: built {built or 'none'}" + (f"; FAILED {failed} (rerun to retry)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
