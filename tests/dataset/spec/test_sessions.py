"""Session calendars: fixed daily sessions, continuous markets, Market-1T's."""
from __future__ import annotations

import pytest

from stable_finance.dataset import ContinuousSession, FixedSession, Market1TCalendar



DATE = "2024-03-05"          # a Tuesday


def test_fixed_session_bounds_and_closures():
    cal = FixedSession("09:30", "16:00", "America/New_York", closed_dates={"2024-03-06"})
    o, c = cal.bounds(DATE)
    assert c - o == 23_400 == cal.session_seconds
    assert cal.origin(DATE) == o
    assert not cal.is_open("2024-03-06") and not cal.is_open("2024-03-09")   # holiday, Saturday
    with pytest.raises(ValueError):
        cal.bounds("2024-03-09")
    # Market-1T is the same wall-clock session; its "+ 1" is the end-stamped
    # one-second grid, which the timeline derives rather than hard-codes.
    from stable_finance.dataset import timeline_bounds_est, standard_open_est
    from stable_finance.dataset import Timeline

    assert Market1TCalendar().bounds(DATE) == (o, c)
    assert Timeline(Market1TCalendar()).bounds(DATE) == timeline_bounds_est(DATE)
    assert Timeline(Market1TCalendar()).origin(DATE) == standard_open_est(DATE)
    assert Timeline(cal, bar_seconds=60).bounds(DATE) == (o + 60, c + 60)


def test_continuous_session_is_every_calendar_day():
    cal = ContinuousSession("UTC")
    o, c = cal.bounds("2024-03-09")
    assert cal.is_open("2024-03-09") and c - o == 86_400 == cal.session_seconds
    assert o % 86_400 == 0
