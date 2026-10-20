"""One clock per dataset, and the check that a stored day keeps to it."""
from __future__ import annotations

import json
import pickle
import shutil

import numpy as np
import pytest

from stable_finance.dataset import (
    ContinuousSession,
    FixedSession,
    MarketSchedule,
    Timeline,
    as_timeline,
    market1t_timeline,
    sessions_from_series,
    standard_open_est,
    timeline_bounds_est,
    write_daystore,
)
from stable_finance.dataset.storage.daystore import ChronologyError, DayRecord

DATE = "2024-03-05"


def test_market1t_timeline_is_the_published_grid():
    clock = market1t_timeline()
    assert clock.bounds(DATE) == timeline_bounds_est(DATE)
    assert clock.origin(DATE) == standard_open_est(DATE)
    assert clock.decisions().tolist() == list(range(0, 23_400, 300))
    assert as_timeline(None) == clock and as_timeline(clock) is clock


def test_a_schedule_or_a_calendar_names_a_timeline(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("date,status,start_time,end_time\n2023-11-24,short day,09:30,13:00\n")
    assert as_timeline(MarketSchedule(csv)).bounds("2023-11-24") == timeline_bounds_est(
        "2023-11-24", schedule=MarketSchedule(csv))
    assert market1t_timeline(str(csv)) == as_timeline(MarketSchedule(csv))
    fixed = as_timeline(FixedSession())
    assert fixed.bar_seconds == 1 and fixed.session_seconds == 23_400
    with pytest.raises(TypeError):
        as_timeline("not a calendar")


def test_a_timeline_round_trips_and_pickles(tmp_path):
    clock = Timeline(ContinuousSession("UTC"), bar_seconds=60, step_seconds=900)
    assert clock.session_seconds == 86_400
    assert Timeline.from_dict(json.loads(json.dumps(clock.to_dict()))) == clock
    csv = tmp_path / "h.csv"
    csv.write_text("date,status,start_time,end_time\n")
    for value in (clock, market1t_timeline(str(csv))):
        assert pickle.loads(pickle.dumps(value)) == value      # worker processes get it whole


def test_every_duration_must_be_whole_bars():
    with pytest.raises(ValueError, match="whole number"):
        Timeline(FixedSession(), bar_seconds=60, step_seconds=90)


def test_two_datasets_share_a_clock_only_when_rows_and_decisions_agree():
    a = Timeline(FixedSession(), bar_seconds=60)
    assert a.same_clock(Timeline(FixedSession(), bar_seconds=60), [DATE, "2024-03-09"])
    assert not a.same_clock(Timeline(FixedSession(), bar_seconds=1), [DATE])
    assert not a.same_clock(Timeline(FixedSession(close="13:00"), bar_seconds=60,
                                     session_seconds=23_400), [DATE])
    # Market-1T on an ordinary day is the 09:30-16:00 New York session.
    assert market1t_timeline().same_clock(as_timeline(FixedSession()), [DATE])


@pytest.fixture
def day(tmp_path):
    clock = Timeline(FixedSession(), bar_seconds=60)
    o, _ = FixedSession().bounds(DATE)
    rng = np.random.default_rng(1)
    sessions = []
    for k in range(6):
        late = 30 if k == 0 else 0
        px = 50 + k + rng.normal(0, 0.02, 390 - late).cumsum()
        sessions += sessions_from_series(o + 60 * np.arange(late, 390), px, timeline=clock,
                                         ticker=f"S{k}", stamp="start")
    (path,) = write_daystore(sessions, tmp_path / "days", clock)
    return path


def test_a_stored_day_keeps_to_its_clock(day):
    record = DayRecord(day)
    record.check_chronology()
    assert record.timeline == Timeline(FixedSession(), bar_seconds=60)


def _tamper(day, tmp_path, change):
    copy = tmp_path / "copy" / day.name
    shutil.copytree(day, copy)
    meta = json.loads((copy / "meta.json").read_text())
    change(copy, meta)
    (copy / "meta.json").write_text(json.dumps(meta))
    return DayRecord(copy)


@pytest.mark.parametrize("change, message", [
    (lambda d, m: m.update(anchors=m["anchors"][1:]), "decision grid"),
    (lambda d, m: m.update(origin=m["origin"] + 60), "origin"),
    (lambda d, m: m.update(date="2024-03-09"), "not a session"),
    (lambda d, m: m["timeline"].update(step_seconds=600), "decision grid"),
])
def test_an_inconsistent_day_is_refused(day, tmp_path, change, message):
    with pytest.raises(ChronologyError, match=message):
        _tamper(day, tmp_path, change).check_chronology()


def test_data_before_a_names_first_row_is_refused(day, tmp_path):
    def leak(copy, meta):
        features = np.load(copy / "features.npy")
        first = np.load(copy / "first_row.npy")
        late = int(np.argmax(first))
        features[late, :, 0] = 1.0                     # a value before it was observed
        np.save(copy / "features.npy", features)

    with pytest.raises(ChronologyError, match="before its first row"):
        _tamper(day, tmp_path, leak).check_chronology()
