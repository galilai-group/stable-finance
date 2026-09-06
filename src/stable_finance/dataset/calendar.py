"""Exchange schedule and single-session timestamp bounds."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
EXTENDED_OPEN_HHMM = "04:00"
EXTENDED_POST_CLOSE_HOURS = 4


class MarketSchedule:
    """Holiday and short-day schedule parsed from the dataset CSV format."""

    def __init__(self, csv_path: str | Path) -> None:
        self._closed_dates: set[str] = set()
        self._short_days: dict[str, tuple[str, str]] = {}
        with open(csv_path, newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                date, status, start, end, *_ = row
                status = status.strip().lower()
                if status == "closed":
                    self._closed_dates.add(date)
                elif status == "short day":
                    end = end.strip()
                    if not end:
                        raise ValueError(f"Short day {date} has no end_time")
                    self._short_days[date] = (start.strip() or "09:30", end)
        for date in self._closed_dates:
            self._short_days.pop(date, None)

    def is_closed(self, date: str) -> bool:
        return date in self._closed_dates

    def market_hours(self, date: str) -> tuple[str, str]:
        if self.is_closed(date):
            raise ValueError(f"Market is closed on {date}")
        return self._short_days.get(date, ("09:30", "16:00"))


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def timeline_bounds_est(
    date: str,
    schedule: MarketSchedule | None = None,
    extended_hours: bool = False,
) -> tuple[int, int]:
    """Return the inclusive-open, exclusive-close epoch seconds for a session."""
    if schedule is not None and schedule.is_closed(date):
        raise ValueError(f"Market is closed on {date}")
    localized = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ET)
    open_text, close_text = (
        schedule.market_hours(date) if schedule is not None else ("09:30", "16:00")
    )
    open_hour, open_minute = _parse_hhmm(open_text)
    close_hour, close_minute = _parse_hhmm(close_text)
    if extended_hours:
        open_hour, open_minute = _parse_hhmm(EXTENDED_OPEN_HHMM)
        close_hour += EXTENDED_POST_CLOSE_HOURS
    start = int(
        (localized + dt.timedelta(hours=open_hour, minutes=open_minute)).timestamp()
    ) + 1
    end = int(
        (localized + dt.timedelta(hours=close_hour, minutes=close_minute)).timestamp()
    ) + 1
    return start, end


def standard_open_est(date: str) -> int:
    """Return the standard 09:30 ET grid origin, including the dataset offset."""
    localized = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ET)
    return int((localized + dt.timedelta(hours=9, minutes=30)).timestamp()) + 1
