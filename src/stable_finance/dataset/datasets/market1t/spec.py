"""Market-1T, described: its columns and roles, and its exchange calendar.

SCHEMA. Column semantics of Market-1T, the Polygon-derived 1 Hz US-equity dataset.

CALENDAR. Market-1T's exchange calendar: NYSE hours, a holiday file, and its grid offset.

Market-1T stamps each one-second bar at its END, so the first regular-session
row is 09:30:01 ET and every bound below carries ``+ 1``. The holiday CSV
lists closed days and short days with their early close.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

from stable_finance.dataset.spec.schema import FeatureSchema
from stable_finance.dataset.spec.timeline import _parse_hhmm


# ── schema ───────────────────────────────────────────────────────────────────

MARKET_SCHEMA = FeatureSchema(
    columns=(
        "bid_price", "vwap_all", "high", "low", "ask_price",
        "bid_size", "ask_size", "volume", "n",
    ),
    forward_fill=frozenset({
        "bid_price", "vwap_all", "high", "low", "ask_price",
        "bid_size", "ask_size",
    }),
    zero_fill=frozenset({"volume", "n"}),
    aggregation=(
        ("bid_price", "last"), ("vwap_all", "vwap"), ("high", "max"),
        ("low", "min"), ("ask_price", "last"), ("bid_size", "last"),
        ("ask_size", "last"), ("volume", "sum"), ("n", "sum"),
    ),
    normalization=(
        ("price", ("bid_price", "ask_price", "vwap_all", "high", "low"), False),
        ("order_book_size", ("bid_size", "ask_size"), True),
        ("trade_size", ("volume",), True),
        ("trade_count", ("n",), True),
    ),
    roles=(
        ("bid_price", "bid_price"), ("ask_price", "ask_price"),
        ("vwap", "vwap_all"), ("high", "high"), ("low", "low"),
        ("bid_size", "bid_size"), ("ask_size", "ask_size"),
        ("volume", "volume"), ("n_trades", "n"),
    ),
)


# ── calendar ─────────────────────────────────────────────────────────────────

EASTERN_TIME = ZoneInfo("America/New_York")
EXTENDED_OPEN_HHMM = "04:00"
EXTENDED_POST_CLOSE_HOURS = 4


class MarketSchedule:
    """Holiday and short-day schedule parsed from the dataset CSV format."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = str(csv_path)
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


def timeline_bounds_est(
    date: str,
    schedule: MarketSchedule | None = None,
    extended_hours: bool = False,
) -> tuple[int, int]:
    """Return the inclusive-open, exclusive-close epoch seconds for a session."""
    if schedule is not None and schedule.is_closed(date):
        raise ValueError(f"Market is closed on {date}")
    localized = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=EASTERN_TIME)
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
    localized = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=EASTERN_TIME)
    return int((localized + dt.timedelta(hours=9, minutes=30)).timestamp()) + 1


class Market1TCalendar:
    """:class:`~stable_finance.dataset.spec.timeline.SessionCalendar` for Market-1T.

    Without a schedule every date is open, which is how the historical
    ``schedule=None`` call sites behaved: closed days are then simply absent
    from the data rather than refused.

    Bounds and origin are WALL CLOCK, as the protocol states them. The grid
    helpers add one bar, which at Market-1T's one-second bars reproduces
    :func:`timeline_bounds_est` and :func:`standard_open_est` exactly: those
    two functions were always the end-stamped one-second grid.
    """

    def __init__(self, schedule: MarketSchedule | None = None, *,
                 extended_hours: bool = False) -> None:
        self.schedule = schedule
        self.extended_hours = bool(extended_hours)

    def is_open(self, date: str) -> bool:
        return self.schedule is None or not self.schedule.is_closed(date)

    def bounds(self, date: str) -> tuple[int, int]:
        start, end = timeline_bounds_est(date, schedule=self.schedule,
                                         extended_hours=self.extended_hours)
        return start - 1, end - 1

    def origin(self, date: str) -> int:
        return standard_open_est(date) - 1

    def to_dict(self) -> dict:
        return {"kind": "market1t"}

    def __eq__(self, other) -> bool:
        return (isinstance(other, Market1TCalendar)
                and self.extended_hours == other.extended_hours
                and getattr(self.schedule, "csv_path", None) == getattr(other.schedule, "csv_path", None))

    def __hash__(self) -> int:
        return hash((getattr(self.schedule, "csv_path", None), self.extended_hours))

    def __reduce__(self):
        # Picklable for process pools: the schedule is re-read from its file.
        path = self.schedule.csv_path if self.schedule is not None else None
        return (_rebuild, (path, self.extended_hours))

    def __repr__(self) -> str:
        path = self.schedule.csv_path if self.schedule is not None else None
        return f"Market1TCalendar(schedule={path!r}, extended_hours={self.extended_hours})"


def _rebuild(path, extended_hours):
    return Market1TCalendar(MarketSchedule(path) if path else None,
                            extended_hours=extended_hours)


def market1t_timeline(schedule: MarketSchedule | str | None = None, *, extended_hours: bool = False,
                      post_close_seconds: int = 0):
    """Market-1T's clock: the NYSE calendar, one-second bars, a decision every 5 minutes.

    ``schedule`` is the holiday file (or its path); without one every weekday
    is a full session. ``post_close_seconds`` keeps a tail past the close for
    labels measured into the auction.
    """
    from stable_finance.dataset.spec.timeline import Timeline

    if isinstance(schedule, (str, bytes)) or hasattr(schedule, "__fspath__"):
        schedule = MarketSchedule(schedule)
    return Timeline(Market1TCalendar(schedule, extended_hours=extended_hours),
                    post_close_seconds=post_close_seconds)

