"""Any series in: bar length and stamp declared, sessions out, stores written."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance.dataset import (
    AnchorSpec,
    ContinuousSession,
    FeatureSchema,
    FixedSession,
    Objective,
    Timeline,
    anchor_targets,
    sessions_from_frame,
    sessions_from_series,
    supported_targets,
    write_daystore,
)
from stable_finance.dataset.ingest import records_from_arrays
from stable_finance.dataset.storage.daystore import DayRecord
from stable_finance.dataset.storage.meta import read_store_meta



DATE = "2024-03-05"          # a Tuesday


def test_series_ingests_onto_a_minute_grid_with_overnight_seed():
    cal = FixedSession("09:30", "16:00", "America/New_York")
    o, c = cal.bounds(DATE)
    ts = np.r_[o - 3600, o + 60 * np.arange(1, 390)]   # an overnight bar, then minutes
    px = np.r_[99.0, 100 + 0.01 * np.arange(389)]
    (session,) = sessions_from_series(ts, px, timeline=Timeline(cal, bar_seconds=60), stamp="start")
    assert session.date == DATE and session.bar_seconds == 60
    # 390 rows stamped at their bar's end: 09:31 ... 16:00.
    assert len(session.features) == 390 and session.timestamps[0] == o + 60
    assert session.timestamps[-1] == c
    assert session.features[0, 0] == 99.0               # seeded at the open
    assert session.features[1, 0] == 100.0


def test_records_belong_to_the_next_session_to_close():
    cal = FixedSession("09:30", "16:00", "America/New_York")
    o, c = cal.bounds(DATE)
    ts = np.asarray([o, c - 1, c + 10])                # the last one is after the close
    recs = list(records_from_arrays(ts, [1.0, 2.0, 3.0], Timeline(cal, bar_seconds=1), stamp="end",
                                    schema=FeatureSchema.univariate()))
    assert [(r["date"], len(r["ts_interval"])) for r in recs] == [
        ("2024-03-05", 2), ("2024-03-06", 1)]


def test_start_stamps_are_moved_to_the_end_of_their_bar():
    cal = FixedSession()
    o, _ = cal.bounds(DATE)
    px = 100 + np.arange(390.0)
    (start,) = sessions_from_series(o + 60 * np.arange(390), px, timeline=Timeline(cal, bar_seconds=60), stamp="start")
    (end,) = sessions_from_series(o + 60 * np.arange(1, 391), px, timeline=Timeline(cal, bar_seconds=60), stamp="end")
    np.testing.assert_array_equal(start.timestamps, end.timestamps)
    np.testing.assert_array_equal(start.features, end.features)


def test_a_view_never_holds_the_bar_after_its_decision():
    """Anchor a decides at the end of the bar starting a seconds after the open.

    The view ends at that bar. Read start stamps as end stamps and the view
    would hold the next minute: the first minute of the return's own window.
    """
    cal = FixedSession()
    o, _ = cal.bounds(DATE)
    px = 100 + np.arange(390.0)                        # bar k starts at o + 60 k
    (session,) = sessions_from_series(o + 60 * np.arange(390), px, timeline=Timeline(cal, bar_seconds=60), stamp="start")

    anchor = 600
    row = (anchor - (int(session.timestamps[0]) - Timeline(cal, bar_seconds=60).origin(DATE))) // 60
    assert session.features[row, 0] == px[anchor // 60]          # the bar 09:40-09:41
    assert session.timestamps[row] == o + anchor + 60            # known at 09:41


def test_a_misdeclared_bar_or_an_off_grid_stamp_is_refused():
    cal = FixedSession()
    o, _ = cal.bounds(DATE)
    minutes = o + 60 * np.arange(390)
    with pytest.raises(ValueError, match="probably 60s long"):
        sessions_from_series(minutes, np.ones(390), timeline=Timeline(cal, bar_seconds=1), stamp="start")
    assert sessions_from_series(minutes, np.ones(390), timeline=Timeline(cal, bar_seconds=1),
                                stamp="start", check_bar=False)
    with pytest.raises(ValueError, match="not on the 60-second bar grid"):
        sessions_from_series(minutes + 30, np.ones(390), timeline=Timeline(cal, bar_seconds=60),
                             stamp="start")
    with pytest.raises(ValueError, match="stamp must be"):
        sessions_from_series(minutes, np.ones(390), timeline=Timeline(cal, bar_seconds=60),
                             stamp="middle")
    with pytest.raises(TypeError):
        sessions_from_series(minutes, np.ones(390), timeline=cal)       # stamp is required


def test_frame_with_tickers_and_partial_columns():
    pd = pytest.importorskip("pandas")
    cal = ContinuousSession("UTC")
    o, _ = cal.bounds(DATE)
    n = 3600
    frame = pd.DataFrame({
        "ts": pd.to_datetime(np.r_[o + np.arange(n), o + np.arange(n)], unit="s", utc=True),
        "sym": ["X"] * n + ["Y"] * n,
        "close": np.r_[np.linspace(10, 11, n), np.linspace(20, 19, n)],
        "volume": np.ones(2 * n),
    })
    schema = FeatureSchema.from_roles(close="close", volume="volume")
    sessions = sessions_from_frame(frame, schema, Timeline(cal, bar_seconds=1), time="ts", ticker="sym",
                                   stamp="start")
    assert sorted(s.ticker for s in sessions) == ["X", "Y"]
    assert all(len(s.features) == 86_400 for s in sessions)
    assert supported_targets(schema) == ("return",)


def test_daystore_from_any_dataset_round_trips(tmp_path):
    cal = FixedSession("09:30", "16:00", "America/New_York")
    o, _ = cal.bounds(DATE)
    rng = np.random.default_rng(11)
    sessions = []
    for k, ticker in enumerate("ABCDEFGHIJ"):
        px = 50 + k + rng.normal(0, 0.02, 390).cumsum()
        sessions += sessions_from_series(o + 60 * np.arange(390), px, timeline=Timeline(cal, bar_seconds=60),
                                         ticker=ticker, stamp="start")
    spec = AnchorSpec(session_seconds=23_400, bar_seconds=60)
    (day,) = write_daystore(sessions, tmp_path, Timeline(cal, bar_seconds=60),
                             objective=Objective(horizons=(300, 1800)))

    record = DayRecord(day)
    assert record.schema == FeatureSchema.univariate()
    assert record.bar_seconds == 60 and record.origin == o + 60 and record.default_objective.price == "mid"
    assert record.types == ("return", "volatility_change")
    assert record.rows == 390 and record.anchor_row(600) == 10
    back = record.record(int(record.ticker_index("C")))
    np.testing.assert_array_equal(back["features"], sessions[2].features.astype(np.float32))
    assert back["grid_start"] == o + 60
    raw = record.targets("raw", 600, [2], ["return"], [300])
    expect = anchor_targets(sessions[2].features, [300], np.asarray([10]),
                            types=("return",), spec=spec, schema=record.schema)
    np.testing.assert_allclose(raw[0, 0], expect[0, 0, 0], rtol=1e-6)
    meta = read_store_meta(day.parent)
    assert meta.schema == record.schema and meta.bar_seconds == 60
    assert meta.timeline().calendar == cal and meta.bar_seconds == 60
