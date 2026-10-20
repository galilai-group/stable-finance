"""A dataset's clock: when its market is open, how long a row is, when decisions fall.

A :class:`Timeline` is the one description of time a dataset has. Every
builder -- ingestion, grids, the stores, panels, cells, labels -- reads time
from it rather than from separate calendar, bar-length and grid arguments,
so they cannot disagree. A timeline holds:

``calendar``
    When the market is open, as wall-clock facts per date (a
    :class:`SessionCalendar`)::

        is_open(date)  -> bool
        bounds(date)   -> (open, close)   # epoch seconds of the session
        origin(date)   -> int             # the instant decisions count from

    ``origin`` and ``bounds()[0]`` differ only on days that open late;
    Market-1T counts decisions from the standard 09:30 open, so a 13:00
    decision is the same instant on a short day as on a full one.
    :class:`FixedSession` is a daily session at fixed local times,
    :class:`ContinuousSession` a market that never closes. Concrete
    calendars, such as Market-1T's NYSE calendar with its holiday file, live
    with their dataset in :mod:`stable_finance.dataset.datasets`.
``bar_seconds``
    How long a stored row is.
``step_seconds`` and ``session_seconds``
    The decision grid: a decision every ``step_seconds`` from the origin,
    over ``session_seconds``.
``post_close_seconds``
    A tail kept past the close, so a label can be measured into the auction.

ROWS ARE STAMPED AT THE END OF THEIR BAR. A stored row holds what was known
by its timestamp and nothing after it, so a view ending at a row can never see
the bar that follows. A session's rows are stamped ``open + bar, open + 2 *
bar, ..., close`` (:meth:`Timeline.bounds`), and the decision at offset ``a``
reads the row stamped ``origin + a + bar`` (:meth:`Timeline.origin` is the
stamp of offset 0's row). Data stamped at the START of each bar, as most bar
vendors stamp it, is moved onto this convention at ingestion
(:func:`to_end_stamps`). Market-1T's long-standing "+ 1" is this rule at
one-second bars.

How long a label measures -- the window at each end of a return -- is not
part of the clock: it belongs to the objective
(:class:`~stable_finance.dataset.labels.objective.Objective`).
:class:`AnchorSpec` is the older form of the same geometry plus that window,
kept because the target functions and existing callers take it;
:meth:`Timeline.anchor_spec` builds one.
"""


from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import numpy as np


# ── sessions ─────────────────────────────────────────────────────────────────

@runtime_checkable
class SessionCalendar(Protocol):
    def is_open(self, date: str) -> bool: ...

    def bounds(self, date: str) -> tuple[int, int]: ...

    def origin(self, date: str) -> int: ...


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def _local_epoch(date: str, hhmm: str, zone: ZoneInfo) -> int:
    hour, minute = _parse_hhmm(hhmm)
    day = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=zone)
    return int((day + dt.timedelta(hours=hour, minutes=minute)).timestamp())


#: How input timestamps label their bars.
STAMPS = ("end", "start")


def to_end_stamps(timestamps, bar_seconds: int, stamp: str):
    """Input timestamps restated as the END of each bar."""
    if stamp not in STAMPS:
        raise ValueError(f"stamp must be one of {STAMPS}, got {stamp!r}")
    return timestamps + int(bar_seconds) if stamp == "start" else timestamps


@dataclass(frozen=True)
class FixedSession:
    """A daily session at fixed local wall-clock times.

    ``closed_dates`` lists dates with no session; ``weekdays_only`` closes
    Saturdays and Sundays. ``close`` may be ``"24:00"``. ``offset_seconds``
    shifts the session's wall clock, for a venue whose session is not on the
    minute; how bars are stamped is a separate fact, given at ingestion.
    """

    open: str = "09:30"
    close: str = "16:00"
    timezone: str = "America/New_York"
    closed_dates: frozenset[str] = frozenset()
    weekdays_only: bool = True
    offset_seconds: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "closed_dates", frozenset(self.closed_dates))
        if _parse_hhmm(self.close) <= _parse_hhmm(self.open):
            raise ValueError("close must be after open")
        ZoneInfo(self.timezone)

    def is_open(self, date: str) -> bool:
        if date in self.closed_dates:
            return False
        if self.weekdays_only:
            return dt.date.fromisoformat(date).weekday() < 5
        return True

    def bounds(self, date: str) -> tuple[int, int]:
        if not self.is_open(date):
            raise ValueError(f"no session on {date}")
        zone = ZoneInfo(self.timezone)
        return (_local_epoch(date, self.open, zone) + self.offset_seconds,
                _local_epoch(date, self.close, zone) + self.offset_seconds)

    def origin(self, date: str) -> int:
        zone = ZoneInfo(self.timezone)
        return _local_epoch(date, self.open, zone) + self.offset_seconds

    @property
    def session_seconds(self) -> int:
        (oh, om), (ch, cm) = _parse_hhmm(self.open), _parse_hhmm(self.close)
        return (ch - oh) * 3600 + (cm - om) * 60

    def to_dict(self) -> dict:
        return {"kind": "fixed", "open": self.open, "close": self.close,
                "timezone": self.timezone, "closed_dates": sorted(self.closed_dates),
                "weekdays_only": self.weekdays_only,
                "offset_seconds": self.offset_seconds}


def ContinuousSession(timezone: str = "UTC", *, closed_dates=(),
                      offset_seconds: int = 0) -> FixedSession:
    """A market that trades around the clock, cut into local calendar days."""
    return FixedSession(open="00:00", close="24:00", timezone=timezone,
                        closed_dates=frozenset(closed_dates), weekdays_only=False,
                        offset_seconds=offset_seconds)


def _calendar_from_dict(value: dict) -> SessionCalendar:
    kind = value.get("kind")
    if kind == "fixed":
        return FixedSession(open=value["open"], close=value["close"],
                            timezone=value["timezone"],
                            closed_dates=frozenset(value.get("closed_dates", ())),
                            weekdays_only=bool(value.get("weekdays_only", True)),
                            offset_seconds=int(value.get("offset_seconds", 0)))
    if kind == "market1t":
        from stable_finance.dataset.datasets.market1t.spec import Market1TCalendar

        return Market1TCalendar()
    raise ValueError(f"unknown calendar kind {kind!r}")


# ── decision grid ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnchorSpec:
    """Geometry shared by target tables, training crops, and evaluation.

    These defaults describe Market-1T's regular-session recipe. Keeping them
    in a value object lets callers inject a different grid without treating
    dataset geometry as model configuration.

    Every field is in SECONDS. ``bar_seconds`` is the spacing of the stored
    rows, so the row counts the target builders index with are the ``*_rows``
    properties: at 1 Hz they equal the seconds, at one-minute bars they are a
    sixtieth. Every duration must be a whole number of bars.
    """

    session_seconds: int = 23_400
    step_seconds: int = 300
    measurement_window_seconds: int = 60
    post_close_seconds: int = 0
    bar_seconds: int = 1

    def __post_init__(self) -> None:
        if self.session_seconds < 1:
            raise ValueError("session_seconds must be positive")
        if self.step_seconds < 1:
            raise ValueError("step_seconds must be positive")
        if self.step_seconds > self.session_seconds:
            raise ValueError("step_seconds cannot exceed session_seconds")
        if self.measurement_window_seconds < 1:
            raise ValueError("measurement_window_seconds must be positive")
        if self.post_close_seconds < 0:
            raise ValueError("post_close_seconds cannot be negative")
        if self.bar_seconds < 1:
            raise ValueError("bar_seconds must be positive")
        for name in ("session_seconds", "step_seconds",
                     "measurement_window_seconds", "post_close_seconds"):
            if getattr(self, name) % self.bar_seconds:
                raise ValueError(
                    f"{name}={getattr(self, name)} is not a whole number of "
                    f"{self.bar_seconds}-second bars")

    def rows(self, seconds: int) -> int:
        """A duration in seconds as a number of stored rows."""
        seconds = int(seconds)
        if seconds % self.bar_seconds:
            raise ValueError(
                f"{seconds}s is not a whole number of {self.bar_seconds}-second bars")
        return seconds // self.bar_seconds

    @property
    def session_rows(self) -> int:
        return self.session_seconds // self.bar_seconds

    @property
    def step_rows(self) -> int:
        return self.step_seconds // self.bar_seconds

    @property
    def window_rows(self) -> int:
        return self.measurement_window_seconds // self.bar_seconds

    @property
    def anchors_per_session(self) -> int:
        return self.session_seconds // self.step_seconds

    @property
    def close_index(self) -> int:
        """Row of the closing auction: the first second after the session.

        Rows 0..session_seconds-1 are the regular session, so the auction
        prints at ``session_seconds`` itself. It is not part of the session
        and never enters a view. A row index, like every index the target
        builders take.
        """
        return self.session_rows

    @property
    def rows_retained(self) -> int:
        """Rows a stored session holds, regular session plus the tail."""
        return (self.session_seconds + self.post_close_seconds) // self.bar_seconds

    @property
    def measures_close(self) -> bool:
        """Is enough of the tail retained to measure the auction window?

        The auction is read as a VWAP over ``[close_index, +window)`` like
        every other forward measurement, so the tail has to cover the window
        or the target cannot be formed.
        """
        return self.post_close_seconds >= self.measurement_window_seconds


DEFAULT_ANCHOR_SPEC = AnchorSpec()


# Named compatibility constants for concise numerical code and downstream
# scripts. New APIs should accept an AnchorSpec where geometry is configurable.
SESSION_LEN = DEFAULT_ANCHOR_SPEC.session_seconds
ANCHOR_STEP = DEFAULT_ANCHOR_SPEC.step_seconds
RETURN_VWAP_WINDOW = DEFAULT_ANCHOR_SPEC.measurement_window_seconds
ANCHORS_PER_DAY = DEFAULT_ANCHOR_SPEC.anchors_per_session
DEFAULT_TARGET_HORIZONS = (300, 600, 900, 1800, 3600, 7200)

#: Horizon sentinel meaning "measure to the closing auction" rather than to a
#: fixed offset. A fixed horizon asks the same question at every anchor and so
#: answers a different one as the session runs out: from 15:00 a two-hour
#: return is unmeasurable, and the anchors where it IS measurable are exactly
#: the early ones, which silently turns a horizon study into a time-of-day
#: study. Measuring to the close instead asks one question -- what happens
#: between here and the auction -- that every anchor can answer, with a
#: holding period that shortens through the day as a real position's does.
#:
#: It travels inside the ordinary ``horizons`` tuple so that target tables,
#: the day store and the panel cache carry it as one more column without any
#: of them learning about it. Negative because no real offset can collide.
TO_CLOSE = -1

#: The recipe that can measure it: the stored session keeps one measurement
#: window past the close. The regular session is unchanged at 23,400 rows, so
#: anchors, crops and views are bit-identical to the default spec -- the tail
#: exists only for the target builder to read.
CLOSING_AUCTION_SPEC = AnchorSpec(post_close_seconds=RETURN_VWAP_WINDOW)


# ── the timeline ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Timeline:
    """A dataset's clock. See the module docstring.

    ``session_seconds`` defaults to the calendar's own session length when it
    states one (:attr:`FixedSession.session_seconds`), else Market-1T's
    regular 6.5 hours. Every duration must be a whole number of bars.
    """

    calendar: SessionCalendar
    bar_seconds: int = 1
    step_seconds: int = 300
    session_seconds: int | None = None
    post_close_seconds: int = 0

    def __post_init__(self) -> None:
        if self.session_seconds is None:
            object.__setattr__(self, "session_seconds", int(getattr(
                self.calendar, "session_seconds", DEFAULT_ANCHOR_SPEC.session_seconds)))
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
    def from_dict(cls, value: dict, *, calendar: SessionCalendar | None = None) -> "Timeline":
        """Inverse of :meth:`to_dict`. ``calendar`` replaces the recorded one.

        A Market-1T calendar is recorded by kind only -- its holiday file is a
        path on the writer's machine -- so a reader passes its own to keep
        short days.
        """
        if calendar is None:
            recorded = value.get("calendar")
            calendar = _calendar_from_dict(recorded) if recorded else _market1t_calendar()
        return cls(calendar=calendar, bar_seconds=int(value.get("bar_seconds", 1)),
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

    def with_calendar(self, calendar: SessionCalendar) -> "Timeline":
        return dataclasses.replace(self, calendar=calendar)

    @property
    def is_market1t(self) -> bool:
        """Does this clock run on Market-1T's calendar (whose short days need its file)?"""
        from stable_finance.dataset.datasets.market1t.spec import Market1TCalendar

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


def _market1t_calendar(schedule=None, *, extended_hours: bool = False):
    from stable_finance.dataset.datasets.market1t.spec import Market1TCalendar

    return Market1TCalendar(schedule, extended_hours=extended_hours)


def as_timeline(value=None, *, extended_hours: bool = False) -> Timeline:
    """The timeline meant by what a caller passed.

    A :class:`Timeline` is itself. A :class:`SessionCalendar` is that
    calendar at one-second bars on the standard decision grid. None and a
    Market-1T holiday ``MarketSchedule`` -- what every ``schedule`` argument
    meant before timelines existed -- are the Market-1T timeline.
    """
    from stable_finance.dataset.datasets.market1t.spec import MarketSchedule

    if isinstance(value, Timeline):
        if extended_hours:
            raise ValueError("extended_hours belongs to the Market-1T calendar; build its timeline")
        return value
    if value is None or isinstance(value, MarketSchedule):
        return Timeline(_market1t_calendar(value, extended_hours=extended_hours))
    if isinstance(value, SessionCalendar):
        if extended_hours:
            raise ValueError("extended_hours applies to the Market-1T calendar only")
        return Timeline(value)
    raise TypeError(f"not a timeline, session calendar or Market-1T schedule: {value!r}")
