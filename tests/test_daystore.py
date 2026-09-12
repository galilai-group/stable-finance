"""The day-major store is the dense mosaic regrouped, with its targets built in."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("streaming")

from stable_finance.dataset import AnchorTargetStats, get_target_names
from stable_finance.dataset.calendar import standard_open_est, timeline_bounds_est
from stable_finance.dataset.daystore import (
    DayRecord,
    convert_month,
    cross_section_transforms,
    discover_days,
    iter_span_records,
)
from stable_finance.dataset.outcomes import anchor_indices, anchor_targets
from stable_finance.dataset.write_mds import DENSE_MDS_COLUMNS, MDS_COLUMNS

DATES = ("2023-01-03", "2023-01-04")
TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
ROWS = 23_400


def _session(rng, n_rows, seed_price):
    """A plausible dense session: positive quotes, bid < ask, integer sizes."""
    steps = rng.normal(0, 0.01, n_rows).cumsum()
    mid = seed_price * np.exp(steps)
    half = 0.01 + 0.002 * rng.random(n_rows)
    f = np.empty((n_rows, 9), dtype=np.float32)
    f[:, 0] = mid - half                                   # bid
    f[:, 4] = mid + half                                   # ask
    f[:, 7] = rng.poisson(30, n_rows) * (rng.random(n_rows) < 0.6)   # volume
    f[:, 8] = np.minimum(f[:, 7], rng.poisson(3, n_rows))  # n
    f[:, 1] = np.where(f[:, 7] > 0, mid + rng.normal(0, 0.003, n_rows), np.nan)
    # The dense mosaic forward-fills vwap over zero-volume seconds.
    idx = np.arange(n_rows); src = np.where(np.isnan(f[:, 1]), 0, idx)
    np.maximum.accumulate(src, out=src); f[:, 1] = f[src, 1]
    if np.isnan(f[0, 1]):
        f[:, 1] = np.where(np.isnan(f[:, 1]), mid, f[:, 1])
    f[:, 2] = np.maximum(f[:, 1], mid) + 0.001               # high
    f[:, 3] = np.minimum(f[:, 1], mid) - 0.001               # low
    f[:, 5] = rng.poisson(200, n_rows) + 1                   # bid_size
    f[:, 6] = rng.poisson(200, n_rows) + 1                   # ask_size
    return f


@pytest.fixture(scope="module")
def mosaic(tmp_path_factory):
    """A shuffled dense month plus its sparse twin (for the anchor tables)."""
    from streaming import MDSWriter

    root = tmp_path_factory.mktemp("mosaic")
    dense, sparse = root / "dense" / "2023" / "01", root / "sparse" / "2023" / "01"
    rng = np.random.default_rng(7)
    records = []
    for d, date in enumerate(DATES):
        ts_open, ts_close = timeline_bounds_est(date)
        for t, ticker in enumerate(TICKERS):
            # One late starter (a trimmed grid) and one stub too short to
            # enter any cross-section.
            first = 5_000 if ticker == "CCC" else (ROWS - 4 if ticker == "FFF" else 0)
            f = _session(rng, ROWS - first, 20.0 + 15 * t + d)
            records.append({"ticker": ticker, "date": date,
                            "grid_start": ts_open + first, "features": f})
    order = rng.permutation(len(records))
    with MDSWriter(out=str(dense), columns=DENSE_MDS_COLUMNS, compression="zstd",
                   size_limit=1 << 22) as w:
        for i in order:
            w.write(records[i])
    with MDSWriter(out=str(sparse), columns=MDS_COLUMNS, compression="zstd",
                   size_limit=1 << 22) as w:
        for i in order:
            r = records[i]
            w.write({"ticker": r["ticker"], "date": r["date"],
                     "ts_interval": (r["grid_start"] + np.arange(len(r["features"]))).astype(np.int32),
                     "features": r["features"]})
    return {"dense": dense, "sparse": sparse, "records": records, "root": root}


@pytest.fixture(scope="module")
def store(mosaic, tmp_path_factory):
    out = tmp_path_factory.mktemp("daystore")
    convert_month(mosaic["dense"], out, None, verify_every=1, log=lambda m: None)
    return out


def test_every_ticker_day_is_present_once_and_byte_identical(mosaic, store):
    days = discover_days(store, "2023-01-01", "2023-01-31")
    assert [d.name for d in days] == list(DATES)
    seen = set()
    by_key = {(r["ticker"], r["date"]): r for r in mosaic["records"]}
    for rec in iter_span_records(store, "2023-01-01", "2023-01-31"):
        key = (rec["ticker"], rec["date"])
        assert key not in seen
        seen.add(key)
        src = by_key[key]
        assert rec["grid_start"] == src["grid_start"]
        np.testing.assert_array_equal(rec["features"], src["features"])
    assert len(seen) == len(by_key)
    index = json.loads((store / "2023" / "01" / "index.json").read_text())
    assert index["n_ticker_days"] == len(by_key)


def test_the_grid_is_nan_before_the_first_quote_and_dense_after(store):
    day = DayRecord(store / "2023" / "01" / DATES[0])
    c = day.ticker_index("CCC")
    assert day.first_row[c] == 5_000 and day.rows == ROWS and day.open_tod == 0
    assert day.features.shape == (len(TICKERS), 9, ROWS)
    assert np.isnan(day.features[c, :, :5_000]).all()
    assert np.isfinite(day.features[c, :, 5_000:]).all()
    assert day.feasible_from(4_999).tolist() == [t not in ("CCC", "FFF") for t in TICKERS]
    w = day.window([0, c], 6_000, 100)
    assert w.shape == (2, 9, 100) and w.dtype == np.float32
    np.testing.assert_array_equal(w[1], day.record(c)["features"][1_000:1_100].T)
    with pytest.raises(ValueError):
        day.window([0], ROWS - 10, 100)


def test_raw_targets_are_anchor_targets_of_each_record(store, mosaic):
    day = DayRecord(store / "2023" / "01" / DATES[1])
    raw = day.targets_array("raw")
    assert raw.shape == (len(anchor_indices()), len(TICKERS), 3, 6)
    for i, ticker in enumerate(TICKERS):
        rec = day.record(i)
        first = int(day.first_row[i])
        if ROWS - first < 10:
            assert not np.isfinite(raw[:, i]).any()   # FFF: too short to count
            continue
        rel = day.anchors - first
        keep = (rel >= 0) & (rel < ROWS - first)
        expect = anchor_targets(rec["features"], day.horizons.tolist(), rel[keep],
                                types=day.types)
        np.testing.assert_allclose(raw[keep, i], expect.astype(np.float32),
                                   rtol=1e-6, equal_nan=True)
        assert not np.isfinite(raw[~keep, i]).any()
    # Type-major selection, in get_target_names order.
    names = get_target_names([900, 300], ["spread_change", "return"])
    got = day.targets("raw", int(day.anchors[50]), [0, 1], ["spread_change", "return"], [900, 300])
    assert got.shape == (2, 4) and names[2] == "return_900"
    np.testing.assert_array_equal(got[:, 2], raw[50, [0, 1], 0, 2])
    assert np.isnan(day.targets("raw", 17, [0], ["return"], [900])).all()   # off-lattice


def test_transforms_agree_with_the_anchor_tables(store, mosaic, tmp_path):
    """The uniform/zscore/rank stored here == what AnchorTargetStats returns
    from a build_targets table on the same month, per (ticker, anchor)."""
    from stable_finance.dataset.build_targets import build_month

    holidays = tmp_path / "holidays.csv"
    holidays.write_text("date,status,start_time,end_time\n")
    table = build_month("2023-01", mosaic["sparse"], tmp_path / "xs", str(holidays), 1)
    stats = AnchorTargetStats(table)
    for date in DATES:
        day = DayRecord(store / "2023" / "01" / date)
        raw = day.targets_array("raw")
        for name in ("uniform", "zscore", "rank"):
            stored = day.targets_array(name)
            for i in range(len(TICKERS)):
                for a in range(0, len(day.anchors), 9):
                    y = raw[a, i]
                    if not np.isfinite(y).any():
                        continue
                    expect = stats.transform(y.astype(np.float64), date, int(day.anchors[a]),
                                             day.types, day.horizons.tolist()).select(name)
                    np.testing.assert_allclose(stored[a, i], expect, rtol=1e-5, atol=1e-6,
                                               equal_nan=True, err_msg=f"{name} {date} {i} {a}")
    # Five names per cell is under MIN_NAMES, so uniform/zscore are NaN
    # everywhere in this tiny month while rank (no threshold) is not.
    day = DayRecord(store / "2023" / "01" / DATES[0])
    assert day.count.max() == 5
    assert not np.isfinite(day.targets_array("uniform")).any()
    assert np.isfinite(day.targets_array("rank")).any()


def test_transforms_on_a_wide_cross_section():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(2, 40, 1, 1)).astype(np.float32)
    raw[0, :3] = np.nan
    tr = cross_section_transforms(raw)
    assert tr["count"].tolist() == [[[37]], [[40]]]
    u = tr["uniform"][1, :, 0, 0]
    order = np.argsort(raw[1, :, 0, 0])
    np.testing.assert_allclose(u[order], (np.arange(40) + 1) / 41, rtol=1e-6)
    z = tr["zscore"][1, :, 0, 0]
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-5
    assert np.isnan(tr["uniform"][0, :3, 0, 0]).all() and np.isfinite(tr["uniform"][0, 3:, 0, 0]).all()
    r = tr["rank"][1, :, 0, 0]
    assert np.all(np.diff(r[order]) >= 0) and abs(r).max() < 3


def test_discovery_falls_back_to_a_scan_while_a_month_is_being_written(store, tmp_path):
    partial = tmp_path / "partial"
    (partial / "2023" / "01").mkdir(parents=True)
    import shutil
    shutil.copytree(store / "2023" / "01" / DATES[1], partial / "2023" / "01" / DATES[1])
    assert [d.name for d in discover_days(partial, "2023-01-01", "2023-01-31")] == [DATES[1]]
    with pytest.raises(ValueError):
        discover_days(partial, "2022-01-01", "2022-12-31")
