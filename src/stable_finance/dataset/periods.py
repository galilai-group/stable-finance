"""Discovery of the dataset's month- and quarter-partitioned storage."""

from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path

from stable_finance.dataset.months import Month


def infer_period_frequency(dataset_root: str | Path) -> str:
    return "qtr" if "qtr" in Path(dataset_root).name else "mnth"


def validate_period_alignment(start: dt.date, end: dt.date, frequency: str) -> None:
    if start.day != 1:
        raise ValueError(f"date_start ({start}) must be the 1st of a month.")
    last = calendar.monthrange(end.year, end.month)[1]
    if end.day != last:
        raise ValueError(
            f"date_end ({end}) must be the last day of a month "
            f"(expected {end.year}-{end.month:02d}-{last:02d})."
        )
    quarter_starts, quarter_ends = {1, 4, 7, 10}, {3, 6, 9, 12}
    if frequency == "qtr" and start.month not in quarter_starts:
        nearest = min(quarter_starts, key=lambda month: abs(month - start.month))
        raise ValueError(
            f"date_start month ({start.month:02d}) is not quarter-aligned. "
            f"Nearest valid start month: {nearest:02d}."
        )
    if frequency == "qtr" and end.month not in quarter_ends:
        nearest = min(quarter_ends, key=lambda month: abs(month - end.month))
        raise ValueError(
            f"date_end month ({end.month:02d}) is not quarter-aligned. "
            f"Nearest valid end month: {nearest:02d}."
        )


def period_directories(
    dataset_root: str | Path,
    date_start: str | dt.date,
    date_end: str | dt.date,
    *,
    allow_unaligned_dates: bool = False,
) -> list[Path]:
    """Return existing dataset period directories in chronological order."""
    start = dt.date.fromisoformat(str(date_start))
    end = dt.date.fromisoformat(str(date_end))
    frequency = infer_period_frequency(dataset_root)
    if not allow_unaligned_dates:
        validate_period_alignment(start, end, frequency)
    root = Path(dataset_root)
    found = []
    cursor, last = Month(start.year, start.month), Month(end.year, end.month)
    while cursor <= last:
        if frequency == "mnth" or cursor.month in {3, 6, 9, 12}:
            candidate = root / str(cursor.year) / f"{cursor.month:02d}"
            if candidate.is_dir():
                found.append(candidate)
        cursor = cursor.offset(1)
    if not found:
        raise ValueError(
            f"No dataset period directories found in {dataset_root} for date "
            f"range [{date_start}, {date_end}]. Expected "
            f"{dataset_root}/{{year}}/{{MM}}/ subdirectories."
        )
    return found

