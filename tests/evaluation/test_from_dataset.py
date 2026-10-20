import json

import numpy as np
import pytest

from stable_finance import MarketPanel, decision_labels, find_cached_month, to_grid


def write_cache(root, month, *, anchors_per_day, stats_tag="stats", ready=True,
                n_rows=6, quotes=True):
    directory = root / f"key{anchors_per_day}{stats_tag}" / month
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(
        {"month": month, "key": {"anchors_per_day": anchors_per_day,
                                 "stats_tag": stats_tag, "hash": "abc"}}))
    dates = np.array([f"{month}-0{1 + i // 3}" for i in range(n_rows)], dtype="U10")
    anchors = np.array([i % 3 for i in range(n_rows)], dtype=np.int64)
    tickers = np.array([f"T{i % 2}" for i in range(n_rows)], dtype="U5")
    np.save(directory / "dates.npy", dates)
    np.save(directory / "anchors.npy", anchors)
    np.save(directory / "tickers.npy", tickers)
    np.save(directory / "raw_targets.npy", np.arange(n_rows * 2, dtype=float).reshape(n_rows, 2))
    if quotes:
        book = np.column_stack([np.full(n_rows, 99.0), np.full(n_rows, 101.0)])
        np.save(directory / "quotes.npy", book)
    if ready:
        (directory / ".ready").touch()
    return directory


def test_finds_a_ready_month_by_anchor_schedule(tmp_path):
    write_cache(tmp_path, "2008-08", anchors_per_day=8)
    write_cache(tmp_path, "2008-08", anchors_per_day=36)
    found = find_cached_month(tmp_path, "2008-08", anchors_per_day=36)
    assert json.loads((found / "meta.json").read_text())["key"]["anchors_per_day"] == 36


def test_an_unready_month_is_not_found(tmp_path):
    write_cache(tmp_path, "2008-08", anchors_per_day=8, ready=False)
    with pytest.raises(FileNotFoundError, match="no ready panel cache"):
        find_cached_month(tmp_path, "2008-08", anchors_per_day=8)


def test_a_month_without_quotes_is_not_found(tmp_path):
    """A cache predating the quote column would price every trade as free."""
    write_cache(tmp_path, "2008-08", anchors_per_day=8, quotes=False)
    with pytest.raises(FileNotFoundError):
        find_cached_month(tmp_path, "2008-08", anchors_per_day=8)


def test_the_stats_tag_is_honoured_when_supplied(tmp_path):
    write_cache(tmp_path, "2008-08", anchors_per_day=8, stats_tag="other")
    with pytest.raises(FileNotFoundError, match="stats"):
        find_cached_month(tmp_path, "2008-08", anchors_per_day=8, stats_tag="wanted")


def test_half_spread_is_a_fraction_of_mid(tmp_path):
    write_cache(tmp_path, "2008-08", anchors_per_day=8)
    panel = MarketPanel.from_cache(tmp_path, "2008-08", anchors_per_day=8)
    np.testing.assert_allclose(panel.half_spread(), np.full(6, 1.0 / 100.0))


def test_a_crossed_book_is_unquoted_rather_than_a_rebate(tmp_path):
    directory = write_cache(tmp_path, "2008-08", anchors_per_day=8)
    book = np.column_stack([np.full(6, 101.0), np.full(6, 99.0)])
    np.save(directory / "quotes.npy", book)
    panel = MarketPanel.from_cache(tmp_path, "2008-08", anchors_per_day=8)
    assert np.isnan(panel.half_spread()).all()


def test_align_matches_rows_and_reports_misses(tmp_path):
    write_cache(tmp_path, "2008-08", anchors_per_day=8)
    panel = MarketPanel.from_cache(tmp_path, "2008-08", anchors_per_day=8)
    index = panel.align(
        np.array(["2008-08-01", "2008-08-01", "1999-01-01"]),
        np.array([0, 1, 0]),
        np.array(["T0", "T1", "T0"]),
    )
    np.testing.assert_array_equal(index, [0, 1, -1])
    taken = panel.take(index)
    np.testing.assert_array_equal(taken["matched"], [True, True, False])
    assert np.isnan(taken["half_spread"][2])
    assert np.isnan(taken["raw_targets"][2]).all()


def test_align_is_order_independent(tmp_path):
    write_cache(tmp_path, "2008-08", anchors_per_day=8)
    panel = MarketPanel.from_cache(tmp_path, "2008-08", anchors_per_day=8)
    shuffled = np.array([4, 0, 2, 1])
    index = panel.align(panel.dates[shuffled], panel.anchors[shuffled],
                        panel.tickers[shuffled])
    np.testing.assert_array_equal(index, shuffled)


def test_duplicate_row_keys_are_rejected(tmp_path):
    directory = write_cache(tmp_path, "2008-08", anchors_per_day=8)
    np.save(directory / "tickers.npy", np.full(6, "T0", dtype="U5"))
    np.save(directory / "anchors.npy", np.zeros(6, dtype=np.int64))
    np.save(directory / "dates.npy", np.full(6, "2008-08-01", dtype="U10"))
    panel = MarketPanel.from_cache(tmp_path, "2008-08", anchors_per_day=8)
    with pytest.raises(RuntimeError, match="duplicate row keys"):
        panel.align(panel.dates, panel.anchors, panel.tickers)


def test_ragged_columns_are_rejected(tmp_path):
    directory = write_cache(tmp_path, "2008-08", anchors_per_day=8)
    np.save(directory / "quotes.npy", np.ones((3, 2)))
    with pytest.raises(RuntimeError, match="ragged columns"):
        MarketPanel.from_cache(tmp_path, "2008-08", anchors_per_day=8)


def test_decision_labels_sort_chronologically():
    labels = decision_labels(
        np.array(["2008-08-02", "2008-08-01", "2008-08-01"]), np.array([0, 10, 2])
    )
    assert sorted(labels.tolist()) == ["2008-08-01|0002", "2008-08-01|0010",
                                       "2008-08-02|0000"]


def test_to_grid_scatters_rows_and_leaves_gaps_missing():
    grid = to_grid(
        np.array([1.0, 2.0]), np.array(["d0", "d1"]), np.array(["A", "A"]),
        decision_axis=np.array(["d0", "d1"]), asset_axis=np.array(["A", "B"]),
    )
    assert grid.shape == (2, 2, 1)
    np.testing.assert_array_equal(grid[:, 0, 0], [1.0, 2.0])
    assert np.isnan(grid[:, 1, 0]).all()


def test_to_grid_drops_rows_outside_the_requested_axes():
    grid = to_grid(
        np.array([1.0, 9.0]), np.array(["d0", "elsewhere"]), np.array(["A", "A"]),
        decision_axis=np.array(["d0"]), asset_axis=np.array(["A"]),
    )
    assert grid.shape == (1, 1, 1) and grid[0, 0, 0] == 1.0
