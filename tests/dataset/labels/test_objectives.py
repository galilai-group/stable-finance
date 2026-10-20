"""Objectives are separate from the data and the loader, and caches are keyed by them."""
from __future__ import annotations

import numpy as np
import pytest
from helpers import IdentityStats as _IdentityStats

from stable_finance import forward_returns_from_days, market_from_days
from stable_finance.dataset import (
    MARKET_SCHEMA,
    TO_CLOSE,
    AnchorSpec,
    FeatureSchema,
    FixedSession,
    Objective,
    Timeline,
    ViewSpec,
    anchor_targets,
    build_session_panel,
    sessions_from_series,
    write_daystore,
)
from stable_finance.dataset.storage.daystore import DayRecord, materialize_objective

DATE = "2024-03-05"
N_NAMES = 24                                   # a cell needs MIN_NAMES = 20
TIMELINE = Timeline(FixedSession(), bar_seconds=60)


@pytest.fixture(scope="module")
def days(tmp_path_factory):
    """One univariate minute-bar day for 24 names, stored with the default objective."""
    root = tmp_path_factory.mktemp("days")
    cal = FixedSession()
    o, _ = cal.bounds(DATE)
    rng = np.random.default_rng(0)
    sessions = []
    for k in range(N_NAMES):
        px = 20 + k + rng.normal(0, 0.02, 390).cumsum()
        sessions += sessions_from_series(o + 60 * np.arange(390), px, timeline=Timeline(cal, bar_seconds=60),
                                         ticker=f"S{k:02d}", stamp="start")
    paths = write_daystore(sessions, root, TIMELINE,
                           objective=Objective(horizons=(300, 900, 1800)))
    return {"root": root, "paths": paths, "sessions": sessions}


# ── the definition ───────────────────────────────────────────────────────────


def test_an_objective_resolves_against_the_schema_and_keys_its_definition():
    univariate = FeatureSchema.univariate()
    resolved = Objective().resolve(univariate)
    assert resolved.types == ("return", "volatility_change") and resolved.price == "mid"
    assert resolved.key == Objective().resolve(univariate).key           # stable
    assert resolved.key != Objective(window_seconds=300).resolve(univariate).key
    assert resolved.key != Objective().resolve(MARKET_SCHEMA).key        # vwap, three families
    with pytest.raises(ValueError, match="resolve"):
        Objective().key
    with pytest.raises(Exception, match="bid_price"):
        Objective(types=("spread_change",)).resolve(univariate)


def test_names_carry_the_window_and_lookahead_is_horizon_plus_window():
    five = Objective(types=("return",), horizons=(900,), window_seconds=300)
    assert five.resolve(MARKET_SCHEMA).names == ["return_900@300"]
    assert Objective(types=("return",), horizons=(900,)).resolve(MARKET_SCHEMA).names == [
        "return_900"]
    assert five.lookahead_seconds == 1200
    assert Objective(horizons=(300, TO_CLOSE)).lookahead_seconds is None
    with pytest.raises(ValueError):
        Objective(horizons=(0,))
    with pytest.raises(ValueError):
        Objective(cross_section="batch")


# ── labels from the store ────────────────────────────────────────────────────


def test_another_window_is_computed_from_the_stored_series(days):
    record = DayRecord(days["paths"][0])
    five = Objective(horizons=(900,), window_seconds=300)
    raw = record.objective_arrays(five)["raw"]
    session = days["sessions"][3]
    i = record.ticker_index(session.ticker)
    rows = (record.anchors - record.open_tod) // 60
    expect = anchor_targets(session.features, [900], rows[:60], types=("return", "volatility_change"),
                            spec=AnchorSpec(bar_seconds=60, measurement_window_seconds=300),
                            schema=record.schema, price="mid")
    np.testing.assert_allclose(raw[:60, i], expect, rtol=1e-6, equal_nan=True)
    stored = record.targets_array("raw")
    assert not np.allclose(raw[:60, i, 0, 0], stored[:60, i, 0, 1], equal_nan=True)


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
    assert stored.horizons.tolist() == [300, 900, 1800]
    record = DayRecord(paths[0])
    i = record.ticker_index("S05")
    np.testing.assert_array_equal(stored.values[30, 5], record.targets_array("raw")[30, i, 0])
    windows = {w: forward_returns_from_days(paths, Objective(horizons=(900,), window_seconds=w))
               for w in (60, 300, 900)}
    np.testing.assert_array_equal(windows[60].values[:, :, 0], stored.values[:, :, 1])
    assert not np.allclose(windows[60].values, windows[900].values, equal_nan=True)
    with pytest.raises(ValueError, match="fixed positive"):
        forward_returns_from_days(paths, Objective(horizons=(TO_CLOSE,)))


def test_a_store_can_be_written_with_any_default_objective(days, tmp_path):
    five = Objective(horizons=(600,), window_seconds=300)
    (path,) = write_daystore(days["sessions"], tmp_path, TIMELINE,
                             objective=five)
    record = DayRecord(path)
    assert record.default_objective.key == five.resolve(record.schema).key
    assert record.default_objective.window_seconds == 300
    np.testing.assert_array_equal(
        record.targets_array("raw"),
        DayRecord(days["paths"][0]).objective_arrays(five)["raw"])


def test_the_panel_builder_takes_an_objective_and_checks_the_table_window(days):
    cal = FixedSession()
    session = days["sessions"][0]
    two = Objective(types=("return",), horizons=(600,), window_seconds=120)
    rows = build_session_panel(session, [3600], _IdentityStats(), timeline=TIMELINE, objective=two,
                               view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)))
    row = (3600 - (int(session.timestamps[0]) - (cal.origin(DATE) + 60))) // 60
    expect = anchor_targets(session.features, [600], np.asarray([row]), types=("return",),
                            spec=AnchorSpec(bar_seconds=60, measurement_window_seconds=120),
                            schema=session.schema, price="mid")
    np.testing.assert_allclose(rows[0].raw_target, expect.ravel(), rtol=1e-6)

    class OneMinuteTable(_IdentityStats):
        window_seconds = 60

    with pytest.raises(ValueError, match="60s window"):
        build_session_panel(session, [3600], OneMinuteTable(), timeline=TIMELINE, objective=two,
                            view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)))
