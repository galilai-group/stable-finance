"""Market-1T's calendar, its clock, chronology checks, and where its stores live."""
from __future__ import annotations

import datetime as dt
import fnmatch
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streaming")   # the MDS extra (`uv sync --extra mds`)
from helpers import market1t_store
from test_daystore import DATES, mosaic, store  # noqa: F401  (fixtures)

from stable_finance.dataset import (
    MarketSchedule,
    Month,
    Timeline,
    as_timeline,
    infer_period_frequency,
    market1t_timeline,
    next_month,
    period_directories,
    standard_open_est,
    timeline_bounds_est,
)
from stable_finance.dataset.datasets.market1t.daystore import (
    FEATURES,
    ChronologyError,
    DayRecord,
    discover_days,
)
from stable_finance.dataset.datasets.market1t.spec import parse_hub, resolve_root



@pytest.fixture
def schedule(tmp_path):
    path = tmp_path / "market_holidays.csv"
    path.write_text(
        "date,status,start_time,end_time,holiday_name\n"
        "2002-09-11,short day,11:00,16:00,Late open\n"
        "2023-11-24,short day,09:30,13:00,Early close\n"
        "2023-12-25,closed,,,Christmas\n"
    )
    return MarketSchedule(path)


def test_schedule_handles_normal_short_closed_and_late_open_days(schedule):
    assert schedule.market_hours("2023-06-15") == ("09:30", "16:00")
    assert schedule.market_hours("2023-11-24") == ("09:30", "13:00")
    assert schedule.market_hours("2002-09-11") == ("11:00", "16:00")
    assert schedule.is_closed("2023-12-25")
    with pytest.raises(ValueError, match="closed"):
        schedule.market_hours("2023-12-25")


def test_schedule_rejects_short_day_without_a_close(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "date,status,start_time,end_time,holiday_name\n"
        "2099-01-02,short day,09:30,,Bad row\n"
    )
    with pytest.raises(ValueError, match="no end_time"):
        MarketSchedule(path)


def test_timeline_respects_schedule_and_standard_open(schedule):
    late_open, late_close = timeline_bounds_est("2002-09-11", schedule=schedule)
    assert late_close - late_open == 5 * 3600
    assert late_open - standard_open_est("2002-09-11") == 90 * 60

    short_open, short_close = timeline_bounds_est("2023-11-24", schedule=schedule)
    assert short_close - short_open == 3.5 * 3600
    with pytest.raises(ValueError, match="closed"):
        timeline_bounds_est("2023-12-25", schedule=schedule)


def test_extended_hours_follow_the_scheduled_close(schedule):
    start, end = timeline_bounds_est(
        "2023-11-24", schedule=schedule, extended_hours=True
    )
    assert end - start == 13 * 3600  # 04:00 through 17:00 ET

    start, end = timeline_bounds_est("2023-06-15", extended_hours=True)
    assert end - start == 16 * 3600  # 04:00 through 20:00 ET


@pytest.mark.parametrize(
    ("date", "open_utc", "close_utc"),
    [
        ("2023-03-10", "14:30:01", "21:00:01"),
        ("2023-03-13", "13:30:01", "20:00:01"),
        ("2023-11-03", "13:30:01", "20:00:01"),
        ("2023-11-06", "14:30:01", "21:00:01"),
    ],
)
def test_session_bounds_use_the_correct_dst_offset(date, open_utc, close_utc):
    """Pin absolute UTC times, not only the always-6.5-hour session length."""
    start, end = timeline_bounds_est(date)
    as_utc = lambda value: dt.datetime.fromtimestamp(
        value, dt.timezone.utc
    ).strftime("%H:%M:%S")
    assert as_utc(start) == open_utc
    assert as_utc(end) == close_utc
    assert standard_open_est(date) == start


DATE = "2023-01-03"


@pytest.fixture
def holidays(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("date,status,start_time,end_time\n2023-11-24,short day,09:30,13:00\n"
                   "2023-12-25,closed,,\n")
    return csv


def test_the_timeline_is_the_published_grid():
    clock = market1t_timeline()
    assert clock.bounds(DATE) == timeline_bounds_est(DATE)
    assert clock.origin(DATE) == standard_open_est(DATE)
    assert clock.decisions().tolist() == list(range(0, 23_400, 300))
    assert as_timeline(None) == clock and as_timeline(clock) is clock


def test_a_schedule_or_its_path_names_the_timeline(holidays):
    schedule = MarketSchedule(holidays)
    assert as_timeline(schedule).bounds("2023-11-24") == timeline_bounds_est(
        "2023-11-24", schedule=schedule)
    assert market1t_timeline(str(holidays)) == as_timeline(schedule)
    assert not as_timeline(schedule).is_open("2023-12-25")
    with pytest.raises(TypeError):
        as_timeline(3.5)


def test_a_timeline_round_trips_and_pickles(holidays):
    clock = market1t_timeline(str(holidays), post_close_seconds=60)
    assert Timeline.from_dict(json.loads(json.dumps(clock.to_dict())),
                              calendar=clock.calendar) == clock
    assert pickle.loads(pickle.dumps(clock)) == clock     # worker processes get it whole


def test_every_duration_must_be_whole_bars():
    with pytest.raises(ValueError, match="whole number"):
        Timeline(market1t_timeline().calendar, bar_seconds=60, step_seconds=90)


def test_two_clocks_agree_only_where_their_sessions_do(holidays):
    plain, scheduled = market1t_timeline(), market1t_timeline(str(holidays))
    assert plain.same_clock(scheduled, [DATE])
    assert not plain.same_clock(scheduled, ["2023-11-24"])     # a short day only one knows


@pytest.fixture
def day(tmp_path):
    (path,) = market1t_store(tmp_path / "days", [DATE], n_tickers=6, late=("T00",))
    return path


def test_a_stored_day_keeps_to_its_clock(day):
    record = DayRecord(day)
    record.check_chronology()
    assert record.timeline == market1t_timeline()


def _tamper(day, tmp_path, change):
    copy = tmp_path / "copy" / day.name
    shutil.copytree(day, copy)
    meta = json.loads((copy / "meta.json").read_text())
    change(copy, meta)
    (copy / "meta.json").write_text(json.dumps(meta))
    return DayRecord(copy)


@pytest.mark.parametrize("change, message", [
    (lambda d, m: m.update(anchors=m["anchors"][1:]), "decision grid"),
    (lambda d, m: m.update(origin=m["origin"] + 1), "origin"),
    (lambda d, m: m.update(date="2023-01-07"), "origin"),   # a day moved to another date
    (lambda d, m: m["timeline"].update(step_seconds=600), "decision grid"),
])
def test_an_inconsistent_day_is_refused(day, tmp_path, change, message):
    with pytest.raises(ChronologyError, match=message):
        _tamper(day, tmp_path, change).check_chronology()


def test_data_before_a_names_first_row_is_refused(day, tmp_path):
    def leak(copy, meta):
        features = np.load(copy / "features.npy")
        features[0, :, 0] = 1.0                        # T00 starts late; row 0 predates it
        np.save(copy / "features.npy", features)

    with pytest.raises(ChronologyError, match="before its first row"):
        _tamper(day, tmp_path, leak).check_chronology()


REPO = "hf://datasets/fin-ai-lab/Market-1T-test"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub: ``snapshot_download`` copies the matching files of ``remote``."""
    import huggingface_hub

    remote = tmp_path / "remote"
    calls = []

    def snapshot_download(repo_id, *, repo_type, revision, allow_patterns, local_dir):
        calls.append({"repo_id": repo_id, "revision": revision, "patterns": allow_patterns})
        for src in remote.rglob("*"):
            rel = src.relative_to(remote).as_posix()
            if src.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                dst = Path(local_dir) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setenv("STABLE_FINANCE_CACHE", str(tmp_path / "cache"))
    return remote, calls


def test_a_local_root_is_returned_untouched(tmp_path):
    assert resolve_root(tmp_path / "anything", "2023-01-01", "2023-01-31") == tmp_path / "anything"


def test_hub_specs_parse():
    assert parse_hub(f"{REPO}/1Hz_daystore") == (
        "dataset", "fin-ai-lab/Market-1T-test", None, "1Hz_daystore")
    assert parse_hub(f"{REPO}@v1/a/b") == ("dataset", "fin-ai-lab/Market-1T-test", "v1", "a/b")
    with pytest.raises(ValueError):
        parse_hub("hf://fin-ai-lab/Market-1T-test")


def test_only_the_months_in_range_are_downloaded(hub):
    remote, calls = hub
    for month in ("2023/01", "2023/02", "2023/03"):
        (remote / "1Hz_mosaic_mnth" / month).mkdir(parents=True)
        (remote / "1Hz_mosaic_mnth" / month / "index.json").write_text("{}")
    dirs = period_directories(f"{REPO}/1Hz_mosaic_mnth", "2023-01-01", "2023-02-28")
    assert [d.relative_to(d.parents[1]).as_posix() for d in dirs] == ["2023/01", "2023/02"]
    assert all(d.is_dir() and d.name != "03" for d in dirs)
    assert not (dirs[0].parents[1] / "2023" / "03").exists()
    assert calls[0]["patterns"] == ["1Hz_mosaic_mnth/2023/01/*", "1Hz_mosaic_mnth/2023/02/*"]


def test_a_hub_day_store_reads_like_the_local_one(hub, store):  # noqa: F811
    remote, _ = hub
    shutil.copytree(store, remote / "1Hz_daystore")
    for raw in (remote / "1Hz_daystore").rglob(FEATURES):
        raw.unlink()                                  # the Hub ships the .zst only
    days = discover_days(f"{REPO}/1Hz_daystore", "2023-01-01", "2023-01-31")
    assert [d.name for d in days] == list(DATES)
    for day in days:
        np.testing.assert_array_equal(
            DayRecord(day).features, DayRecord(store / "2023" / "01" / day.name).features)
    # Idempotent: a second discovery neither re-decompresses nor fails.
    before = {d: (d / FEATURES).stat().st_mtime_ns for d in days}
    discover_days(f"{REPO}/1Hz_daystore", "2023-01-01", "2023-01-31")
    assert {d: (d / FEATURES).stat().st_mtime_ns for d in days} == before
    assert json.loads((days[0].parent / "index.json").read_text())["days"]


def test_month_is_a_first_class_fit_period():
    month = Month.parse("2020-12")
    assert month.first_day == dt.date(2020, 12, 1)
    assert month.last_day == dt.date(2020, 12, 31)
    assert next_month(month) == "2021-01"


def test_partition_discovery_is_storage_backend_independent(tmp_path):
    root = tmp_path / "1Hz_mosaic_mnth"
    (root / "2020" / "01").mkdir(parents=True)
    (root / "2020" / "03").mkdir(parents=True)
    assert infer_period_frequency(root) == "mnth"
    assert period_directories(root, "2020-01-01", "2020-03-31") == [
        root / "2020" / "01",
        root / "2020" / "03",
    ]
