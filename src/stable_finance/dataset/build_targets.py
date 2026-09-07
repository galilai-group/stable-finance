"""Build monthly cross-sectional anchor tables and empirical target transforms.

One pass over a month's mosaic shards folds every (ticker, date) into running
moments of the raw forward targets on the 5-minute anchor grid, then writes
moments and order statistics per (date, anchor, target_type, horizon). Training and
eval both look the target up here, so the cross-section a sample is
standardized against is identical no matter which pipeline reads it.

The table is the ONLY place the cross-section exists: a streaming sample sees
one stock, so ``mu``/``sigma`` across ~700 names at a shared instant can never
be assembled at training time.

Cost is ~1 decode per ticker-day and the arithmetic is vectorized over all
(anchor, type, horizon) cells at once. The exact uniform target stores the
finite per-cell order statistic, so its table is materially larger than the
moment-only format.

Usage:
    python -m stable_finance.dataset.build_targets \
        --out-dir /data/lab/market-jepa-mosaic/xs_anchor_stats \
        --start 2023-01 --end 2023-02
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_finance.dataset import (
    DEFAULT_TARGET_HORIZONS,
    MARKET_SCHEMA,
    MarketSchedule,
    sparse_to_dense_grid,
    standard_open_est,
    timeline_bounds_est,
)
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES as TARGET_TYPES,
    anchor_indices,
    anchor_targets,
)
from stable_finance.dataset.targets import (
    TARGET_TRANSFORMS,
    accumulate,
    build_cross_section_metadata,
)

# Empirical-CDF resolution for the RANK target. 64 levels against ~526
# names per cell is ~1.6% — finer than the label noise, and 7.9 MB per
# month compressed rather than storing the whole panel.
N_QUANTILES = 64
QUANTILE_LEVELS = np.linspace(0.0, 1.0, N_QUANTILES)

FFILL_IDX = MARKET_SCHEMA.forward_fill_indices
ZEROFILL_IDX = MARKET_SCHEMA.zero_fill_indices


def month_dirs(mosaic_dir: Path, start: str, end: str) -> list[tuple[str, Path]]:
    """``(YYYY-MM, dir)`` for every month in [start, end], inclusive.

    The mosaic is laid out ``{root}/{year}/{MM}/`` (see ``discover_streams``),
    so the month label has to be reassembled from the two path components.
    """
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        d = mosaic_dir / str(y) / f"{m:02d}"
        if (d / "index.json").is_file():
            out.append((f"{y}-{m:02d}", d))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _shard_partial(month_dir: str, shard_meta: dict, holiday_csv: str,
                   target_types: tuple[str, ...], horizons: tuple[int, ...],
                   keep_samples: bool):
    """Fold one shard's ticker-days into per-date moment arrays.

    Returns ``{date: (sums, sqs, cnts)}``, each (A, T, H). Runs in a worker
    process, so the schedule is rebuilt here rather than pickled.
    """
    import tempfile

    from stable_finance.dataset.mds import open_shard

    schedule = MarketSchedule(holiday_csv)
    anchors_tod = anchor_indices()
    hs = list(horizons)
    shape = (len(anchors_tod), len(target_types), len(hs))

    out: dict[str, tuple] = {}
    with tempfile.TemporaryDirectory(prefix="xs_shard_") as tmpdir:
        reader = open_shard(month_dir, shard_meta, tmpdir)
        _fold_shard(
            reader, shard_meta["samples"], schedule, anchors_tod, hs,
            target_types, shape, out, keep_samples,
        )
    return out


def _fold_shard(reader, n_samples, schedule, anchors_tod, hs, target_types,
                shape, out, keep_samples):
    """Accumulate every ticker-day in one shard into ``out``, keyed by date."""
    for i in range(n_samples):
        sample = reader.get_item(i)
        date_str = str(sample["date"])
        try:
            ts_open, ts_close = timeline_bounds_est(date_str, schedule=schedule)
        except ValueError:
            continue  # market closed — a stray sample on a holiday
        res = sparse_to_dense_grid(
            ts_sec=sample["ts_interval"],
            features=sample["features"],
            ts_open_sec=ts_open,
            ts_close_sec=ts_close,
            ffill_indices=FFILL_IDX,
            zerofill_indices=ZEROFILL_IDX,
            trim_leading_nan=True,
        )
        if res is None or len(res[1]) < 10:
            continue
        canonical_sec, features = res

        # Anchors are wall-clock instants measured from the standard 09:30
        # open (the dataset's meta.tod_sec convention). A ticker that starts
        # quoting late has its grid trimmed, so its row 0 is NOT 09:30 and the
        # anchor must be shifted by that offset before indexing.
        tod_offset = int(canonical_sec[0]) - standard_open_est(date_str)
        rows = anchors_tod - tod_offset
        keep = (rows >= 0) & (rows < len(features))
        if not keep.any():
            continue

        y = np.full(shape, np.nan)
        y[keep] = anchor_targets(
            features, hs, rows[keep], types=target_types,
        )

        if date_str not in out:
            out[date_str] = (
                np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int64),
                [],
            )
        accumulate(*out[date_str][:3], y)
        # Retain the raw value so the cell's empirical CDF can be quantized
        # later. (mu, sigma) is not enough for a RANK target: assuming the
        # cross-section is Gaussian makes Phi^-1(Phi(z)) collapse back to z,
        # so the whole point is the shape the two moments throw away.
        # ~600 ticker-days x (A,T,H) float32 is ~34 MB per date — cheap.
        if keep_samples:
            out[date_str][3].append(y.astype(np.float32))


def build_month(
    ym: str,
    month_dir: Path,
    out_dir: Path,
    holiday_csv: str,
    workers: int,
    *,
    target_types: tuple[str, ...] = tuple(ANCHOR_TARGET_TYPES),
    horizons: tuple[int, ...] = DEFAULT_TARGET_HORIZONS,
    transforms: tuple[str, ...] = TARGET_TRANSFORMS,
) -> Path:
    """Build one month, calculating only requested families and horizons."""
    from joblib import Parallel, delayed

    unknown_types = set(target_types) - set(ANCHOR_TARGET_TYPES)
    unknown_transforms = set(transforms) - set(TARGET_TRANSFORMS)
    if unknown_types:
        raise ValueError(f"unknown target types: {sorted(unknown_types)}")
    if unknown_transforms:
        raise ValueError(f"unknown target transforms: {sorted(unknown_transforms)}")
    if not target_types or not horizons:
        raise ValueError("target_types and horizons cannot be empty")
    keep_samples = bool({"uniform", "rank"} & set(transforms))
    with open(month_dir / "index.json") as f:
        shards = json.load(f)["shards"]

    parts = Parallel(n_jobs=workers, backend="loky")(
        delayed(_shard_partial)(
            str(month_dir), s, holiday_csv, target_types, horizons, keep_samples,
        )
        for s in shards
    )

    anchors_tod = anchor_indices()
    hs = list(horizons)
    shape = (len(anchors_tod), len(target_types), len(hs))
    merged: dict[str, tuple] = {}
    for part in parts:
        for date_str, (s, q, c, vals) in part.items():
            if date_str not in merged:
                merged[date_str] = (
                    np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int64),
                    [],
                )
            m = merged[date_str]
            m[0][:] += s
            m[1][:] += q
            m[2][:] += c
            m[3].extend(vals)

    dates = sorted(merged)
    D = len(dates)
    sums = np.stack([merged[d][0] for d in dates]) if D else np.zeros((0, *shape))
    sqs = np.stack([merged[d][1] for d in dates]) if D else np.zeros((0, *shape))
    cnts = (
        np.stack([merged[d][2] for d in dates]) if D
        else np.zeros((0, *shape), dtype=np.int64)
    )
    table = build_cross_section_metadata(
        sums, sqs, cnts,
        ([np.stack(merged[d][3]) for d in dates] if keep_samples else None),
        QUANTILE_LEVELS,
        include_quantiles="rank" in transforms,
        include_sorted="uniform" in transforms,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ym}.npz"
    np.savez_compressed(
        path,
        dates=np.array(dates),
        anchors=anchors_tod.astype(np.int32),
        types=np.array(target_types),
        horizons=np.array(hs, dtype=np.int32),
        **table,
    )
    usable = int(np.isfinite(table["sigma"]).sum())
    print(
        f"{ym}: {D} dates, "
        f"{usable}/{table['sigma'].size} usable cells "
        f"({100.0 * usable / max(1, table['sigma'].size):.1f}%), "
        f"median names/cell {int(np.median(cnts[cnts > 0])) if cnts.any() else 0} "
        f"-> {path}",
        flush=True,
    )
    return path


def main(argv=None, *, default_mosaic_dir=None, default_holiday_csv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mosaic-dir", default=default_mosaic_dir,
                   required=default_mosaic_dir is None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", required=True, help="first month, YYYY-MM")
    p.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    p.add_argument("--holiday-csv", default=default_holiday_csv,
                   required=default_holiday_csv is None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target-types", nargs="+", choices=ANCHOR_TARGET_TYPES,
                   default=list(ANCHOR_TARGET_TYPES))
    p.add_argument("--horizons", nargs="+", type=int,
                   default=list(DEFAULT_TARGET_HORIZONS))
    p.add_argument("--transforms", nargs="+", choices=TARGET_TRANSFORMS,
                   default=list(TARGET_TRANSFORMS))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    months = month_dirs(Path(args.mosaic_dir), args.start, args.end)
    if not months:
        raise SystemExit(f"No months in [{args.start}, {args.end}] under {args.mosaic_dir}")

    for ym, md in months:
        if (out_dir / f"{ym}.npz").is_file() and not args.overwrite:
            print(f"{ym}: exists, skipping (--overwrite to rebuild)", flush=True)
            continue
        t0 = time.time()
        build_month(
            ym, md, out_dir, args.holiday_csv, args.workers,
            target_types=tuple(args.target_types),
            horizons=tuple(args.horizons),
            transforms=tuple(args.transforms),
        )
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
