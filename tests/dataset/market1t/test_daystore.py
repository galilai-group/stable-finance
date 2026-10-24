"""The day store: Market-1T's cross-sections by day, their labels and objective caches."""
from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import IdentityStats as _IdentityStats, market1t_store

from stable_finance import forward_returns_from_days, market_from_days
from stable_finance.dataset import (
    TO_CLOSE,
    AnchorSpec,
    AnchorTargetStats,
    Objective,
    SessionPreprocessor,
    ViewSpec,
    anchor_targets,
    build_session_panel,
    get_target_names,
)
from stable_finance.dataset.datasets.market1t.daystore import (
    DayRecord,
    convert_month,
    discover_days,
    iter_span_records,
    materialize_objective,
)
from stable_finance.dataset.datasets.market1t.mosaic import DENSE_MDS_COLUMNS, MDS_COLUMNS
from stable_finance.dataset.datasets.market1t.spec import timeline_bounds_est
from stable_finance.dataset.objective import cross_section_transforms
from stable_finance.dataset.outcomes import anchor_indices



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
    pytest.importorskip("streaming")
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
    from stable_finance.dataset.datasets.market1t.tables import build_month

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


def test_stored_targets_are_the_default_objective(store):
    from stable_finance.dataset import DEFAULT_TARGET_HORIZONS, Objective

    day = DayRecord(store / "2023" / "01" / DATES[1])
    default = day.default_objective
    assert default.types == ("return", "volatility_change", "spread_change")
    assert default.horizons == DEFAULT_TARGET_HORIZONS
    assert default.window_seconds == 60 and default.price == "vwap"
    assert Objective().resolve(day.schema).key == default.key
    arrays = day.objective_arrays()
    np.testing.assert_array_equal(arrays["raw"], day.targets_array("raw"))


def test_a_recomputed_objective_reproduces_the_stored_targets_exactly(store):
    """A different key forces recomputation from the stored features."""
    from stable_finance.dataset import Objective

    day = DayRecord(store / "2023" / "01" / DATES[1])
    same_labels = Objective(min_names=19)            # same raw definition, different key
    assert same_labels.resolve(day.schema).key != day.default_objective.key
    raw = day.objective_arrays(same_labels)["raw"]
    np.testing.assert_array_equal(raw, day.targets_array("raw"))


def test_a_record_written_before_objectives_reads_the_same(store, tmp_path):
    import shutil

    source = store / "2023" / "01" / DATES[0]
    copy = tmp_path / "2023" / "01" / DATES[0]
    shutil.copytree(source, copy)
    meta = json.loads((copy / "meta.json").read_text())
    for key in ("schema", "origin", "timeline", "objective"):
        meta.pop(key, None)
    (copy / "meta.json").write_text(json.dumps(meta))
    old, new = DayRecord(copy), DayRecord(source)
    assert old.default_objective.key == new.default_objective.key
    assert old.origin == new.origin and old.schema == new.schema
    for i in range(old.n_tickers):
        assert old.record(i)["grid_start"] == new.record(i)["grid_start"]


DATE = "2023-01-03"


@pytest.fixture(scope="module")
def days(tmp_path_factory):
    """One Market-1T day for 24 names (a cell needs MIN_NAMES = 20)."""
    root = tmp_path_factory.mktemp("days")
    return {"root": root, "paths": market1t_store(root, [DATE], n_tickers=24)}


def test_another_window_is_computed_from_the_stored_series(days):
    record = DayRecord(days["paths"][0])
    five = Objective(horizons=(900,), window_seconds=300)
    raw = record.objective_arrays(five)["raw"]
    features = record.record(3)["features"].astype(np.float64)
    rows = (record.anchors - record.open_tod)[:60]
    expect = anchor_targets(features, [900], rows, types=record.types,
                            spec=AnchorSpec(measurement_window_seconds=300), schema=record.schema)
    np.testing.assert_allclose(raw[:60, 3], expect, rtol=1e-6, equal_nan=True)
    stored = record.targets_array("raw")
    assert not np.allclose(raw[:60, 3, 0, 0], stored[:60, 3, 0, 2], equal_nan=True)


def test_an_objective_is_cached_under_its_key_and_read_back(days, tmp_path):
    record = DayRecord(days["paths"][0])
    five = Objective(window_seconds=300)
    computed = record.objective_arrays(five, persist=True, cache_root=tmp_path)
    directory = record.objective_dir(five, tmp_path)
    assert (directory / "objective.json").is_file()
    assert directory.name == five.resolve(record.schema).key
    fresh = DayRecord(days["paths"][0]).objective_arrays(five, cache_root=tmp_path)
    assert isinstance(fresh["raw"], np.memmap)                # read, not recomputed
    np.testing.assert_array_equal(fresh["uniform"], computed["uniform"])
    assert materialize_objective(days["root"], DATE, DATE, five, cache_root=tmp_path,
                                 log=lambda *_: None) == 0           # already there
    assert materialize_objective(days["root"], DATE, DATE, Objective(window_seconds=120),
                                 cache_root=tmp_path, log=lambda *_: None) == 1


def test_a_batch_is_labelled_against_the_whole_cross_section(days):
    record = DayRecord(days["paths"][0])
    anchor = int(record.anchors[30])
    batch = np.arange(4)
    labels = record.labels(anchor, batch)                     # uniform, default objective
    full = record.targets_array("uniform")[30][:, 0, 0]          # every name, return_300
    np.testing.assert_array_equal(labels[:, 0], full[batch])
    raw = record.targets_array("raw")[30][batch, 0, 0]
    within_batch = (np.argsort(np.argsort(raw)) + 1) / (len(batch) + 1)
    assert not np.allclose(labels[:, 0], within_batch)


def test_a_restricted_cross_section_ranks_only_its_members(days):
    record = DayRecord(days["paths"][0])
    keep = np.zeros(record.n_tickers, bool)
    keep[:21] = True
    masked = record.cross_section(keep)
    assert np.isnan(masked["uniform"][:, ~keep]).all()
    cell = masked["uniform"][30, keep, 0, 0]
    assert np.isfinite(cell).all()
    np.testing.assert_allclose(np.sort(cell), np.arange(1, 22) / 22, rtol=1e-6)
    thin = np.zeros(record.n_tickers, bool)
    thin[:5] = True                                          # below MIN_NAMES
    assert np.isnan(record.cross_section(thin)["uniform"]).all()


def test_forward_returns_sweep_the_execution_window_on_the_market_grid(days):
    paths = days["paths"]
    stored = forward_returns_from_days(paths)
    market = market_from_days(paths)
    assert np.array_equal(stored.decisions, market.decisions)
    assert np.array_equal(stored.assets, market.assets)
    assert stored.horizons.tolist() == [300, 600, 900, 1800, 3600, 7200]
    record = DayRecord(paths[0])
    i = record.ticker_index("T05")
    np.testing.assert_array_equal(stored.values[30, 5], record.targets_array("raw")[30, i, 0])
    windows = {w: forward_returns_from_days(paths, Objective(horizons=(900,), window_seconds=w))
               for w in (60, 300, 900)}
    np.testing.assert_array_equal(windows[60].values[:, :, 0], stored.values[:, :, 2])
    assert not np.allclose(windows[60].values, windows[900].values, equal_nan=True)
    with pytest.raises(ValueError, match="fixed positive"):
        forward_returns_from_days(paths, Objective(horizons=(TO_CLOSE,)))


def test_a_store_can_be_written_with_any_default_objective(days, tmp_path):
    five = Objective(horizons=(600,), window_seconds=300)
    (path,) = market1t_store(tmp_path, [DATE], n_tickers=24, objective=five)
    record = DayRecord(path)
    assert record.default_objective.key == five.resolve(record.schema).key
    assert record.default_objective.window_seconds == 300
    np.testing.assert_array_equal(
        record.targets_array("raw"),
        DayRecord(days["paths"][0]).objective_arrays(five)["raw"])


def test_the_panel_builder_takes_an_objective_and_checks_the_table_window(days):
    record = DayRecord(days["paths"][0])
    session = SessionPreprocessor().transform(record.record(0))
    two = Objective(types=("return",), horizons=(600,), window_seconds=120)
    view = ViewSpec(sequence_length=4, aggregation_seconds=(1, 1))
    rows = build_session_panel(session, [3600], _IdentityStats(), objective=two, view_spec=view)
    expect = anchor_targets(session.features, [600], np.asarray([3600]), types=("return",),
                            spec=AnchorSpec(measurement_window_seconds=120),
                            schema=session.schema)
    np.testing.assert_allclose(rows[0].raw_target, expect.ravel(), rtol=1e-6)

    class OneMinuteTable(_IdentityStats):
        window_seconds = 60

    with pytest.raises(ValueError, match="60s window"):
        build_session_panel(session, [3600], OneMinuteTable(), objective=two, view_spec=view)


def test_market_from_days_reads_what_every_execution_model_needs(tmp_path):
    paths = market1t_store(tmp_path, ["2023-01-03", "2023-01-04"], n_tickers=4)
    market = market_from_days(paths, lookback=1, delay_seconds=5, depth_seconds=3,
                              window_seconds=300)
    first, second = DayRecord(paths[0]), DayRecord(paths[1])
    assert len(market.decisions) == len(second.anchors)       # the first day is history
    assert market.level == "book"
    k = 10
    row = second.anchor_row(int(second.anchors[k]))
    bid, ask, bsz, asz, vol, vwap = (second.schema.role_index(r) for r in (
        "bid_price", "ask_price", "bid_size", "ask_size", "volume", "vwap"))
    f = np.asarray(second.features, dtype=np.float64)
    np.testing.assert_allclose(market.get("bid_price")[k], f[:, bid, row])
    np.testing.assert_allclose(market.get("exec_ask_price")[k], f[:, ask, row + 5])
    np.testing.assert_allclose(market.get("book_ask_price")[k, :, 0], f[:, ask, row + 5])
    np.testing.assert_allclose(market.get("book_bid_size")[k, :, 0],
                               f[:, bsz, row + 3:row + 6].mean(axis=1))
    np.testing.assert_allclose(market.get("window_best_buy")[k],
                               f[:, ask, row + 1:row + 301].min(axis=1))
    np.testing.assert_allclose(market.get("window_best_sell")[k],
                               f[:, bid, row + 1:row + 301].max(axis=1))
    g = np.asarray(first.features, dtype=np.float64)
    np.testing.assert_allclose(market.get("adv")[k],
                               np.nan_to_num(g[:, vol] * g[:, vwap]).sum(axis=1), rtol=1e-6)
    assert np.isnan(market.get("idio_vol")).all()             # one day is not twenty
    assert market.get("time_trend")[k, 0] == (2023 - 1926) * 12 - 5 + 1
    # The arrival quote is a different number from the decision's.
    assert not np.allclose(market.get("exec_bid_price"), market.get("bid_price"), equal_nan=True)
    # Without history there is no dollar volume to size against, and no decisions are lost.
    plain = market_from_days(paths)
    assert len(plain.decisions) == 2 * len(second.anchors) and not plain.has("adv")
    with pytest.raises(ValueError, match="lookback"):
        market_from_days(paths, lookback=2)
