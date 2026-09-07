from __future__ import annotations

import pytest

from stable_finance.dataset import MarketSchedule, standard_open_est, timeline_bounds_est


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
