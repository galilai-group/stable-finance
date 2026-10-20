"""Market-1T, described: its columns, its exchange calendar and clock, and where its stores live.

SPEC. Market-1T, described: its columns and roles, and its exchange calendar.

SCHEMA. Column semantics of Market-1T, the Polygon-derived 1 Hz US-equity dataset.

CALENDAR. Market-1T's exchange calendar: NYSE hours, a holiday file, and its grid offset.

Market-1T stamps each one-second bar at its END, so the first regular-session
row is 09:30:01 ET and every bound below carries ``+ 1``. The holiday CSV
lists closed days and short days with their early close.

STORE LOCATIONS. Where a store's months are: month identifiers, period directories, local or Hub roots.

MONTHS. Calendar-month identifiers used by fitting and evaluation APIs.

PERIODS. Discovery of the dataset's month- and quarter-partitioned storage.

SOURCES. Where a store's bytes come from: a local directory or a Hugging Face repo.

Every store is laid out as ``<root>/YYYY/MM/...``, so a root is all a reader
needs. A root is either a local path, returned untouched, or a Hub spec::

    hf://datasets/<org>/<repo>[@<revision>]/<subdir>

which is materialized under a local cache and returned as an ordinary path.
Only the months a caller asks for are downloaded, so pointing a one-month eval
at a multi-month repo fetches one month.

The cache is ``$STABLE_FINANCE_CACHE`` (default ``~/.cache/stable-finance``),
one real directory per repo rather than the Hub's symlinked blob cache: the
day store decompresses ``features.npy`` in place beside its ``.zst``, and that
needs a directory the reader owns. Downloads are resumable and a second call
for the same months is a metadata check.

Keep ``<subdir>`` named like the local store (``1Hz_mosaic_mnth``,
``1Hz_daystore``, ...): :func:`infer_period_frequency` reads the frequency off
the root's name, and the resolved path ends in ``<subdir>``.

MONTH DIRECTORIES. Months of a store that hold samples.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np

from stable_finance.dataset.anchors import (
    DEFAULT_ANCHOR_SPEC,
    RETURN_VWAP_WINDOW,
    AnchorSpec,
)
from stable_finance.dataset.schema import FeatureSchema


# ── spec ─────────────────────────────────────────────────────────────────────

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
    """:class:`~stable_finance.dataset.datasets.market1t.spec.SessionCalendar` for Market-1T.

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


# ── timeline ─────────────────────────────────────────────────────────────────

# The calendar interface a timeline reads. Market1TCalendar implements it.
class SessionCalendar(Protocol):
    def is_open(self, date: str) -> bool: ...

    def bounds(self, date: str) -> tuple[int, int]: ...

    def origin(self, date: str) -> int: ...


@dataclass(frozen=True)
class Timeline:
    """A dataset's clock. See the module docstring.

    ``session_seconds`` defaults to the regular 6.5-hour session. Every
    duration must be a whole number of bars.
    """

    calendar: SessionCalendar
    bar_seconds: int = 1
    step_seconds: int = 300
    session_seconds: int | None = None
    post_close_seconds: int = 0

    def __post_init__(self) -> None:
        if self.session_seconds is None:
            object.__setattr__(self, "session_seconds", DEFAULT_ANCHOR_SPEC.session_seconds)
        self.anchor_spec()                    # validates every duration

    # -- the calendar, on the row grid -----------------------------------------

    def is_open(self, date: str) -> bool:
        return self.calendar.is_open(date)

    def bounds(self, date: str) -> tuple[int, int]:
        """Stamps of a session's rows: ``[open + bar, close + bar)``."""
        open_, close = self.calendar.bounds(date)
        return int(open_) + self.bar_seconds, int(close) + self.bar_seconds

    def origin(self, date: str) -> int:
        """Stamp of the row the decision at offset 0 reads."""
        return int(self.calendar.origin(date)) + self.bar_seconds

    # -- the decision grid -------------------------------------------------------

    def decisions(self) -> np.ndarray:
        """Decision offsets in seconds past the origin."""
        return np.arange(0, self.session_seconds, self.step_seconds, dtype=np.int64)

    def rows(self, seconds: int) -> int:
        """A duration in seconds as a number of rows."""
        return self.anchor_spec().rows(seconds)

    def anchor_spec(self, window_seconds: int = RETURN_VWAP_WINDOW) -> AnchorSpec:
        """This clock with a label's measurement window, as an :class:`AnchorSpec`."""
        return AnchorSpec(session_seconds=self.session_seconds, step_seconds=self.step_seconds,
                          measurement_window_seconds=int(window_seconds),
                          post_close_seconds=self.post_close_seconds,
                          bar_seconds=self.bar_seconds)

    # -- description -------------------------------------------------------------

    def to_dict(self) -> dict:
        to_dict = getattr(self.calendar, "to_dict", None)
        return {"calendar": to_dict() if callable(to_dict) else None,
                "bar_seconds": self.bar_seconds, "step_seconds": self.step_seconds,
                "session_seconds": self.session_seconds,
                "post_close_seconds": self.post_close_seconds}

    @classmethod
    def from_dict(cls, value: dict, *, calendar=None) -> "Timeline":
        """Inverse of :meth:`to_dict`; ``calendar`` supplies the holiday file.

        The calendar is recorded by kind only -- its holiday file is a path on
        the writer's machine -- so a reader passes its own to keep short days.
        """
        return cls(calendar=calendar if calendar is not None else Market1TCalendar(),
                   bar_seconds=int(value.get("bar_seconds", 1)),
                   step_seconds=int(value.get("step_seconds", 300)),
                   session_seconds=value.get("session_seconds"),
                   post_close_seconds=int(value.get("post_close_seconds", 0)))

    def check_day(self, date: str, *, rows: int, open_tod: int, origin: int,
                  decisions) -> list[str]:
        """What is inconsistent between a stored day and this clock (empty: nothing).

        A day is consistent when its date is a session, its rows are exactly
        the session's bars stamped from the open, its origin is this clock's,
        and its decisions are exactly the clock's grid -- which also makes
        every name in the day decide at the same instants.
        """
        problems = []
        if not self.is_open(date):
            return [f"{date} is not a session on this timeline"]
        lo, hi = self.bounds(date)
        if rows != (hi - lo) // self.bar_seconds:
            problems.append(f"{rows} rows, but the session has {(hi - lo) // self.bar_seconds} bars")
        if origin != self.origin(date):
            problems.append(f"origin {origin}, but the clock's is {self.origin(date)}")
        if open_tod != lo - self.origin(date):
            problems.append(f"first row {open_tod}s past the origin, but the session's "
                            f"is {lo - self.origin(date)}s")
        expected = self.decisions()
        decisions = np.asarray(decisions, dtype=np.int64)
        if decisions.shape != expected.shape or (decisions != expected).any():
            problems.append("decisions are not the clock's decision grid")
        return problems

    def is_market1t(self) -> bool:
        """Does this clock run on Market-1T's calendar (whose short days need its file)?"""

        return isinstance(self.calendar, Market1TCalendar)

    def same_clock(self, other: "Timeline", dates) -> bool:
        """Do two timelines stamp the same rows and decisions on ``dates``?

        The condition for combining two datasets -- a risk factor beside the
        main store, say -- row for row and decision for decision.
        """
        if (self.bar_seconds, self.step_seconds, self.session_seconds) != (
                other.bar_seconds, other.step_seconds, other.session_seconds):
            return False
        for date in dates:
            if self.is_open(date) != other.is_open(date):
                return False
            if self.is_open(date) and (self.bounds(date) != other.bounds(date)
                                       or self.origin(date) != other.origin(date)):
                return False
        return True


def market1t_timeline(schedule: MarketSchedule | str | None = None, *, extended_hours: bool = False,
                      post_close_seconds: int = 0):
    """Market-1T's clock: the NYSE calendar, one-second bars, a decision every 5 minutes.

    ``schedule`` is the holiday file (or its path); without one every weekday
    is a full session. ``post_close_seconds`` keeps a tail past the close for
    labels measured into the auction.
    """
    if isinstance(schedule, (str, bytes)) or hasattr(schedule, "__fspath__"):
        schedule = MarketSchedule(schedule)
    return Timeline(Market1TCalendar(schedule, extended_hours=extended_hours),
                    post_close_seconds=post_close_seconds)


def as_timeline(value=None, *, extended_hours: bool = False) -> Timeline:
    """The timeline meant by what a caller passed.

    A :class:`Timeline` is itself. None, a ``MarketSchedule``, or a holiday
    file's path -- what every ``schedule`` argument has always meant -- is
    Market-1T's timeline with that schedule.
    """
    if isinstance(value, Timeline):
        if extended_hours:
            raise ValueError("extended_hours belongs to the calendar; build the timeline with it")
        return value
    if value is None or isinstance(value, (MarketSchedule, str)) or hasattr(value, "__fspath__"):
        return market1t_timeline(value, extended_hours=extended_hours)
    raise TypeError(f"not a timeline or Market-1T schedule: {value!r}")


# ── store locations ──────────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class Month:
    """A validated calendar month with inclusive date bounds."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if self.year < 1 or not 1 <= self.month <= 12:
            raise ValueError(f"invalid calendar month: {self.year}-{self.month}")

    @classmethod
    def parse(cls, value: "Month | str") -> "Month":
        if isinstance(value, cls):
            return value
        try:
            year, month = (int(part) for part in str(value)[:7].split("-"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"month must be YYYY-MM, got {value!r}") from error
        return cls(year, month)

    @property
    def first_day(self) -> dt.date:
        return dt.date(self.year, self.month, 1)

    @property
    def last_day(self) -> dt.date:
        return dt.date(
            self.year, self.month, calendar.monthrange(self.year, self.month)[1]
        )

    def offset(self, count: int) -> "Month":
        ordinal = self.year * 12 + self.month - 1 + count
        year, zero_based_month = divmod(ordinal, 12)
        return Month(year, zero_based_month + 1)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def next_month(value: Month | str) -> str:
    """Return the calendar month after ``value`` as ``YYYY-MM``."""
    return str(Month.parse(value).offset(1))


# ── periods ──────────────────────────────────────────────────────────────────

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
    """Return existing dataset period directories in chronological order.

    ``dataset_root`` may be a Hub spec (see :mod:`stable_finance.dataset.datasets.market1t.spec`);
    the months in range are then downloaded first and local paths returned.
    """
    start = dt.date.fromisoformat(str(date_start))
    end = dt.date.fromisoformat(str(date_end))
    frequency = infer_period_frequency(dataset_root)
    if not allow_unaligned_dates:
        validate_period_alignment(start, end, frequency)
    root = resolve_root(dataset_root, start.isoformat(), end.isoformat())
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


# ── sources ──────────────────────────────────────────────────────────────────

HF_SCHEME = "hf://"


def is_hub(root: str | Path) -> bool:
    return str(root).startswith(HF_SCHEME)


def parse_hub(spec: str) -> tuple[str, str, str | None, str]:
    """``hf://<type>s/<org>/<repo>[@rev]/<subdir>`` -> (repo_type, repo_id, revision, subdir)."""
    parts = spec[len(HF_SCHEME):].strip("/").split("/")
    if len(parts) < 3 or parts[0] not in ("datasets", "models"):
        raise ValueError(
            f"{spec!r}: expected hf://datasets/<org>/<repo>[@revision]/<subdir>")
    repo_type = parts[0][:-1]
    repo, _, revision = parts[2].partition("@")
    return repo_type, f"{parts[1]}/{repo}", revision or None, "/".join(parts[3:])


def cache_root() -> Path:
    return Path(os.environ.get("STABLE_FINANCE_CACHE",
                               Path.home() / ".cache" / "stable-finance"))


def resolve_root(root: str | Path, date_start: str | None = None,
                 date_end: str | None = None) -> Path:
    """A local directory holding ``root``'s months in ``[date_start, date_end]``.

    A local ``root`` is returned as is -- nothing is checked or copied, so
    existing configs behave exactly as before. A Hub spec downloads the months
    in range (every month when no range is given) and returns the local copy.
    """
    if not is_hub(root):
        return Path(root)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Reading a store from the Hub requires `uv sync --extra hf`") from error

    repo_type, repo_id, revision, subdir = parse_hub(str(root))
    prefix = f"{subdir}/" if subdir else ""
    if date_start is not None and date_end is not None:
        cursor, last = Month.parse(str(date_start)[:7]), Month.parse(str(date_end)[:7])
        patterns = []
        while cursor <= last:
            patterns.append(f"{prefix}{cursor.year:04d}/{cursor.month:02d}/*")
            cursor = cursor.offset(1)
    else:
        patterns = [f"{prefix}*"]

    local = cache_root() / repo_id.replace("/", "--")
    if revision:
        local = local.with_name(f"{local.name}@{revision}")
    snapshot_download(repo_id, repo_type=repo_type, revision=revision,
                      allow_patterns=patterns, local_dir=local)
    return local / subdir if subdir else local


# ── month directories ────────────────────────────────────────────────────────

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
