"""Convert Polygon snapshot parquet data to MosaicML Streaming (MDS) format.

Each MDS sample is one date-ticker observation: all sparse rows for a given
ticker on a given date, stored as variable-length ndarray columns. Output is
organized by calendar period (month or quarter) with a strong shuffle within
each period.  Output directory structure: ``{out_base}/{year}/{MM}/``.

Usage:
    # Normal MDS conversion:
    python -m stable_finance.dataset.write_mds --start 2023-01 --end 2023-02 \
        --cal-freq mnth --metadata

    # Risk factor extraction (separate invocation):
    python -m stable_finance.dataset.write_mds --start 2023-01 --end 2023-12 \
        --risk-factors IWM

"""

import argparse
import datetime
import json
import os
import random
from calendar import monthrange
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from streaming import MDSWriter
from tqdm import tqdm

from stable_finance.dataset import (
    MARKET_SCHEMA,
    MarketSchedule,
    sparse_to_dense_grid,
    timeline_bounds_est,
)

FEATURE_COLUMNS = list(MARKET_SCHEMA.columns)

TARGET_COLUMNS = ["ticker", "ts_interval"] + FEATURE_COLUMNS

MDS_COLUMNS = {
    "ticker": "str",
    "date": "str",
    "ts_interval": "ndarray:int32",
    "features": "ndarray:float32",
}

# Buffer flush threshold: number of partitions to accumulate before shuffling and writing
BUFFER_FLUSH_PARTITIONS = 200

DEFAULT_METADATA_PATH = os.environ.get("METADATA_PATH")
DEFAULT_RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
DEFAULT_MOSAIC_DIR = os.environ.get("MOSAIC_DIR", None)
DEFAULT_RISK_FACTOR_DIR = os.environ.get("RISK_FACTOR_DIR", None)


def discover_trading_days(base_path: Path, start: str, end: str) -> list[tuple[str, Path]]:
    """Find all trading day directories within a date range.

    Args:
        base_path: Root data path (e.g. .../1Hz)
        start: Start month 'YYYY-MM' (inclusive)
        end: End month 'YYYY-MM' (inclusive)

    Returns:
        List of (date_str, day_dir_path) sorted chronologically.
    """
    start_y, start_m = int(start[:4]), int(start[5:7])
    end_y, end_m = int(end[:4]), int(end[5:7])

    results = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        month_dir = base_path / str(y) / f"{m:02d}"
        if month_dir.exists():
            results.extend(
                (d.stem, d) for d in sorted(month_dir.iterdir())
            )
        m += 1
        if m > 12:
            m = 1
            y += 1
    return results


def load_metadata_filter(
    metadata_path: Path, start: str, end: str
) -> dict[tuple[str, int], set[str]]:
    """Load metadata.parquet and build a filter lookup.

    Returns:
        Dict mapping (date_str, partition_id) -> set of valid tickers.
    """
    df = pd.read_parquet(metadata_path, columns=["date", "ticker", "partition"])

    # Filter to the requested date range (start/end are YYYY-MM strings)
    start_date = datetime.date(int(start[:4]), int(start[5:7]), 1)
    end_y, end_m = int(end[:4]), int(end[5:7])
    end_date = datetime.date(end_y, end_m, monthrange(end_y, end_m)[1])

    df["date"] = pd.to_datetime(df["date"]).dt.date
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df[mask]

    lookup: dict[tuple[str, int], set[str]] = defaultdict(set)
    for date, ticker, partition in zip(df["date"], df["ticker"], df["partition"]):
        lookup[(str(date), int(partition))].add(ticker)
    return dict(lookup)


def get_period_key(date: datetime.date, cal_freq: str) -> tuple[int, str]:
    """Return ``(year, MM)`` for a given date and frequency.

    For ``"mnth"``: the date's own month.
    For ``"qtr"``: the quarter-end month (03, 06, 09, 12).
    """
    if cal_freq == "mnth":
        return (date.year, f"{date.month:02d}")
    elif cal_freq == "qtr":
        qtr_month = ((date.month - 1) // 3 + 1) * 3
        return (date.year, f"{qtr_month:02d}")
    raise ValueError(f"Unknown cal_freq: {cal_freq}")


def enumerate_periods(start: str, end: str, cal_freq: str) -> list[tuple[int, str]]:
    """Return ordered ``(year, MM)`` period keys for the given range.

    Pure date arithmetic — no filesystem access.
    """
    start_y, start_m = int(start[:4]), int(start[5:7])
    end_y, end_m = int(end[:4]), int(end[5:7])

    keys: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        key = get_period_key(datetime.date(y, m, 1), cal_freq)
        if key not in seen:
            seen.add(key)
            keys.append(key)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return keys


def _period_discovery_range(year: int, mm: str, cal_freq: str) -> tuple[str, str]:
    """Return ``(start_YYYY-MM, end_YYYY-MM)`` for :func:`discover_trading_days`."""
    if cal_freq == "mnth":
        ym = f"{year}-{mm}"
        return (ym, ym)
    # qtr: quarter spans 3 months ending at mm
    end_month = int(mm)
    start_month = end_month - 2
    return (f"{year}-{start_month:02d}", f"{year}-{mm}")


def discover_and_convert_period(
    base_path: Path,
    out_base: Path,
    period_key: tuple[int, str],
    cal_freq: str,
    ticker_filter: dict[tuple[str, int], set[str]] | None = None,
    pbar_position: int = 0,
    schedule: MarketSchedule | None = None,
) -> tuple[int, int, int]:
    """Discover trading days, scan partitions, and convert one period.

    Returns:
        ``(n_trading_days, n_partitions, n_observations)``
    """
    year, mm = period_key
    disc_start, disc_end = _period_discovery_range(year, mm, cal_freq)
    trading_days = discover_trading_days(base_path, disc_start, disc_end)

    if schedule is not None:
        trading_days = [(d, p) for d, p in trading_days if not schedule.is_closed(d)]

    # Scan partitions for this period's trading days
    entries: list[tuple[str, int, Path]] = []
    for date_str, day_dir in trading_days:
        for part_dir in sorted(day_dir.iterdir()):
            partition_id = int(part_dir.name.split("=", 1)[1])
            entries.append((date_str, partition_id, part_dir / "0.parquet"))
    random.shuffle(entries)

    n_obs = convert_period(out_base, period_key, entries, ticker_filter, pbar_position)
    return (len(trading_days), len(entries), n_obs)


def flush_observations(buffer: list[dict], writer: MDSWriter) -> int:
    """Shuffle and write buffered observation dicts to MDS. Returns count written."""
    if not buffer:
        return 0
    random.shuffle(buffer)
    for obs in buffer:
        writer.write(obs)
    return len(buffer)


def convert_period(
    out_base: Path,
    period_key: tuple[int, str],
    entries: list[tuple[str, int, Path]],
    ticker_filter: dict[tuple[str, int], set[str]] | None = None,
    pbar_position: int = 0,
) -> int:
    """Convert one period of data to MDS format.

    Each partition file is read, optionally filtered by metadata, grouped by
    ticker, and each ticker group becomes one MDS observation with
    variable-length ndarray columns.
    """
    if not entries:
        return 0

    year, mm = period_key
    out_dir = out_base / str(year) / mm
    out_dir.mkdir(parents=True, exist_ok=True)

    total_obs = 0
    buffer: list[dict] = []
    partitions_in_buffer = 0
    label = f"{year}/{mm}"

    with MDSWriter(
        out=str(out_dir),
        columns=MDS_COLUMNS,
        compression="zstd",
        exist_ok=True,
    ) as writer:
        pbar = tqdm(
            total=len(entries),
            desc=label,
            unit="part",
            position=pbar_position,
            leave=True,
        )

        for date_str, partition_id, path in entries:
            if ticker_filter is not None:
                valid_tickers = ticker_filter.get((date_str, partition_id), set())
                if not valid_tickers:
                    pbar.update(1)
                    continue
                df = pd.read_parquet(path, columns=TARGET_COLUMNS)
                df = df[df["ticker"].isin(valid_tickers)]
            else:
                df = pd.read_parquet(path, columns=TARGET_COLUMNS)

            for ticker, group in df.groupby("ticker", sort=False):
                group = group.sort_values("ts_interval")

                # Convert ts_interval from nanoseconds to seconds as int32
                ts_sec = (group["ts_interval"].to_numpy() / 1e9).astype(np.int64)
                assert ts_sec.max() <= np.iinfo(np.int32).max, (
                    f"ts overflow: {ts_sec.max()} > {np.iinfo(np.int32).max} "
                    f"for {ticker} on {date_str}"
                )
                ts_sec = ts_sec.astype(np.int32)

                # Build features array (N, 9) as float32
                features = group[FEATURE_COLUMNS].to_numpy().astype(np.float32)
                assert not np.any(np.isinf(features)), (
                    f"inf in features for {ticker} on {date_str}"
                )

                buffer.append({
                    "ticker": str(ticker),
                    "date": date_str,
                    "ts_interval": ts_sec,
                    "features": features,
                })

            partitions_in_buffer += 1
            pbar.update(1)
            pbar.set_postfix(obs=f"{total_obs:,}", buf=partitions_in_buffer)

            if partitions_in_buffer >= BUFFER_FLUSH_PARTITIONS:
                total_obs += flush_observations(buffer, writer)
                pbar.set_postfix(obs=f"{total_obs:,}")
                buffer = []
                partitions_in_buffer = 0

        # Flush remainder
        if buffer:
            total_obs += flush_observations(buffer, writer)
            pbar.set_postfix(obs=f"{total_obs:,}", flush="final")

        pbar.close()

    return total_obs


# ---------------------------------------------------------------------------
# Risk factor extraction
# ---------------------------------------------------------------------------

# Column indices for fill logic (same as streaming_dataset.py)
_FFILL_INDICES = MARKET_SCHEMA.forward_fill_indices
_ZEROFILL_INDICES = MARKET_SCHEMA.zero_fill_indices


def discover_ticker_partitions(
    day_dir: Path, tickers: list[str]
) -> dict[str, int]:
    """Scan partitions of one day to find which partition holds each ticker.

    Returns:
        {ticker: partition_id}. Raises if any ticker not found.
    """
    remaining = set(tickers)
    result: dict[str, int] = {}

    for part_dir in day_dir.iterdir():
        if not remaining:
            break
        if not part_dir.is_dir() or not part_dir.name.startswith("partition="):
            continue
        parquet_file = part_dir / "0.parquet"
        if not parquet_file.exists():
            continue
        partition_id = int(part_dir.name.split("=", 1)[1])
        df = pd.read_parquet(parquet_file, columns=["ticker"])
        found = remaining & set(df["ticker"].unique())
        for t in found:
            result[t] = partition_id
        remaining -= found

    if remaining:
        raise ValueError(
            f"Tickers not found in {day_dir}: {remaining}. "
            f"Available partitions were scanned but these tickers are missing."
        )
    return result


def convert_risk_factors(
    base_path: Path, out_dir: Path, tickers: list[str],
    start: str, end: str,
) -> None:
    """Extract risk factor data as dense 1Hz numpy arrays.

    For each ticker, produces:
        {out_dir}/{ticker}/features.npy  — (n_dates, 23400, 9) float32
        {out_dir}/{ticker}/meta.json     — ordered dates, shape, column names
    """
    trading_days = discover_trading_days(base_path, start, end)
    if not trading_days:
        raise ValueError(f"No trading days found in {base_path} for {start}–{end}")

    print(f"Risk factor extraction: {len(trading_days)} trading days, tickers={tickers}")

    # Discover partitions from the LAST (most recent) trading day
    last_date_str, last_day_dir = trading_days[-1]
    print(f"Discovering partitions from {last_date_str}...")
    ticker_partitions = discover_ticker_partitions(last_day_dir, tickers)
    for t, p in ticker_partitions.items():
        print(f"  {t} → partition={p}")

    # Pre-compute grid size from any date
    ts_open, ts_close = timeline_bounds_est(trading_days[0][0])
    grid_size = ts_close - ts_open  # 23400 for standard 6.5h trading day

    # Extract each ticker
    for ticker in tickers:
        partition_id = ticker_partitions[ticker]
        ticker_dir = out_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        dates: list[str] = []
        day_arrays: list[np.ndarray] = []

        for date_str, day_dir in tqdm(trading_days, desc=f"Extracting {ticker}"):
            parquet_file = day_dir / f"partition={partition_id}" / "0.parquet"
            if not parquet_file.exists():
                print(f"  WARNING: {parquet_file} missing, skipping {date_str}")
                continue

            df = pd.read_parquet(parquet_file, columns=TARGET_COLUMNS)
            df = df[df["ticker"] == ticker]

            if df.empty:
                print(f"  WARNING: {ticker} not found in partition={partition_id} on {date_str}")
                continue

            df = df.sort_values("ts_interval")

            # Convert ts_interval from nanoseconds to seconds
            ts_sec = (df["ts_interval"].to_numpy() / 1e9).astype(np.int64)
            assert ts_sec.max() <= np.iinfo(np.int32).max
            ts_sec = ts_sec.astype(np.int32)

            features = df[FEATURE_COLUMNS].to_numpy().astype(np.float32)

            ts_open_sec, ts_close_sec = timeline_bounds_est(date_str)
            result = sparse_to_dense_grid(
                ts_sec=ts_sec,
                features=features,
                ts_open_sec=ts_open_sec,
                ts_close_sec=ts_close_sec,
                ffill_indices=_FFILL_INDICES,
                zerofill_indices=_ZEROFILL_INDICES,
                trim_leading_nan=False,  # Keep full grid for alignment
            )

            if result is None:
                print(f"  WARNING: all-NaN for {ticker} on {date_str}")
                continue

            _, dense = result
            assert dense.shape == (grid_size, len(FEATURE_COLUMNS)), (
                f"Shape mismatch: {dense.shape} != ({grid_size}, {len(FEATURE_COLUMNS)}) "
                f"for {ticker} on {date_str}"
            )

            n_nan = np.isnan(dense).any(axis=1).sum()
            if n_nan > 0:
                print(f"  WARNING: {n_nan} NaN rows remain for {ticker} on {date_str}")
                np.nan_to_num(dense, copy=False, nan=0.0)

            dates.append(date_str)
            day_arrays.append(dense.astype(np.float32))

        if not day_arrays:
            raise ValueError(f"No valid data extracted for {ticker}")

        # Stack to (n_dates, grid_size, n_features) and save
        stacked = np.stack(day_arrays, axis=0)
        npy_path = ticker_dir / "features.npy"
        np.save(npy_path, stacked)

        meta = {
            "dates": dates,
            "shape": list(stacked.shape),
            "feature_columns": FEATURE_COLUMNS,
        }
        with open(ticker_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {ticker}: {stacked.shape} saved to {ticker_dir}")


def main(argv=None, *, default_base_path=None, default_metadata_path=None,
         default_output=None, default_risk_factor_dir=None):
    parser = argparse.ArgumentParser(
        description="Convert polygon snapshot data to MosaicML Streaming format."
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start month inclusive (YYYY-MM), e.g. 2023-01",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End month inclusive (YYYY-MM), e.g. 2023-04",
    )
    parser.add_argument(
        "--base-path",
        default=default_base_path or DEFAULT_RAW_DATA_DIR,
        required=not (default_base_path or DEFAULT_RAW_DATA_DIR),
        help="Base path to Polygon snapshot data",
    )
    parser.add_argument(
        "--n-writers",
        type=int,
        default=os.cpu_count(),
        help="Number of periods to process in parallel (default: all cores)",
    )
    parser.add_argument(
        "--metadata",
        nargs="?",
        const=default_metadata_path or DEFAULT_METADATA_PATH,
        default=None,
        help="Path to metadata.parquet for filtering. Flag without value uses default. Omit to disable filtering.",
    )
    parser.add_argument(
        "--cal-freq",
        choices=["mnth", "qtr"],
        default="mnth",
        help="Output grouping frequency (default: mnth).",
    )
    parser.add_argument(
        "--risk-factors",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="Extract risk factor data instead of MDS conversion. "
             "Provide one or more tickers (e.g. IWM SPY).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output directory (default: derived from base-path).",
    )
    parser.add_argument(
        "--holiday-csv",
        default=None,
        help="Path to market_holidays.csv. Closed days will be excluded from MDS conversion.",
    )
    args = parser.parse_args(argv)

    base_path = Path(args.base_path)

    # Risk factor extraction mode
    if args.risk_factors:
        rf_out = Path(args.output) if args.output else (
            Path(default_risk_factor_dir or DEFAULT_RISK_FACTOR_DIR)
            if (default_risk_factor_dir or DEFAULT_RISK_FACTOR_DIR)
            else base_path.parent / (base_path.name + "_risk_factors")
        )
        print(f"Risk factor extraction mode")
        print(f"Input:  {base_path}")
        print(f"Output: {rf_out}")
        convert_risk_factors(base_path, rf_out, args.risk_factors, args.start, args.end)
        print("\nRisk factor extraction done.")
        return

    out_base = Path(args.output) if args.output else (
        Path(default_output or DEFAULT_MOSAIC_DIR)
        if (default_output or DEFAULT_MOSAIC_DIR)
        else base_path.parent / (base_path.name + "_mosaic_" + args.cal_freq)
    )

    schedule = MarketSchedule(args.holiday_csv) if args.holiday_csv else None

    # Load metadata filter if requested
    ticker_filter = None
    if args.metadata is not None:
        metadata_path = Path(args.metadata)
        print(f"Loading metadata filter from {metadata_path}")
        ticker_filter = load_metadata_filter(metadata_path, args.start, args.end)
        print(f"  {len(ticker_filter)} (date, partition) pairs in filter")

    # Enumerate periods (instant — pure date arithmetic, no I/O)
    period_keys = enumerate_periods(args.start, args.end, args.cal_freq)

    print(f"Converting {args.start} to {args.end} (freq={args.cal_freq})")
    print(f"Input:  {base_path}")
    print(f"Output: {out_base}")
    print(f"Metadata filter: {'enabled' if ticker_filter is not None else 'disabled'}")
    print(f"Periods: {len(period_keys)}")
    print(f"Parallel writers: {args.n_writers}")
    print(f"Buffer flush every {BUFFER_FLUSH_PARTITIONS} partitions (~20K observations)")
    for year, mm in period_keys:
        print(f"  {year}/{mm}")
    print()

    # Each worker discovers its own trading days, scans partitions, and converts.
    if args.n_writers == 1:
        for period_key in period_keys:
            n_days, n_parts, n_obs = discover_and_convert_period(
                base_path, out_base, period_key, args.cal_freq, ticker_filter,
                pbar_position=0, schedule=schedule,
            )
            label = f"{period_key[0]}/{period_key[1]}"
            tqdm.write(f"{label} complete: {n_days} days, {n_parts:,} partitions, {n_obs:,} observations")
    else:
        with ProcessPoolExecutor(max_workers=args.n_writers) as pool:
            futures = {}
            for i, period_key in enumerate(period_keys):
                fut = pool.submit(
                    discover_and_convert_period,
                    base_path, out_base, period_key, args.cal_freq, ticker_filter, i,
                    schedule,
                )
                futures[fut] = period_key

            for fut in as_completed(futures):
                period_key = futures[fut]
                label = f"{period_key[0]}/{period_key[1]}"
                try:
                    n_days, n_parts, n_obs = fut.result()
                    tqdm.write(f"{label} complete: {n_days} days, {n_parts:,} partitions, {n_obs:,} observations")
                except Exception as e:
                    tqdm.write(f"{label} FAILED: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
