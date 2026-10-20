"""A dataset that is not Market-1T through the whole MDS pipeline."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from stable_finance.dataset import ContinuousSession, FeatureSchema, Timeline
from stable_finance.dataset.storage.daystore import DayRecord
from stable_finance.dataset.storage.meta import read_store_meta



def test_any_dataset_runs_the_whole_mds_pipeline(tmp_path):
    """Sparse mosaic -> densify -> day store and target tables, with no Market-1T."""
    pytest.importorskip("streaming")
    pytest.importorskip("joblib")
    from stable_finance.dataset import AnchorTargetStats, write_mosaic
    from stable_finance.dataset.labels.tables import build_month
    from stable_finance.dataset.ingest import records_from_frame
    from stable_finance.dataset.storage import daystore, mosaic

    cal = ContinuousSession("UTC")
    schema = FeatureSchema.from_roles(close="c", volume="v")
    rng = np.random.default_rng(2)
    rows = {"t": [], "sym": [], "c": [], "v": []}
    for date in ("2024-03-09", "2024-03-10"):                # a weekend: continuous trades it
        o, _ = cal.bounds(date)
        for k in range(24):                                  # a cell needs MIN_NAMES = 20
            n = 1440                                         # one print a minute
            rows["t"].append(o + 60 * np.arange(n))
            rows["sym"].append(np.full(n, f"S{k:02d}"))
            rows["c"].append(10 + k + rng.normal(0, 0.01, n).cumsum())
            rows["v"].append(rng.poisson(3, n).astype(float))
    frame = {key: np.concatenate(value) for key, value in rows.items()}
    sparse_root = tmp_path / "sparse"
    written = write_mosaic(records_from_frame(frame, schema, Timeline(cal, bar_seconds=60), time="t", ticker="sym",
                                              stamp="start"),
                           sparse_root, schema, Timeline(cal, bar_seconds=60))
    assert written == {"2024-03": 48}

    month = sparse_root / "2024" / "03"
    dense_month = tmp_path / "dense" / "2024" / "03"
    n_written, skipped, _ = mosaic.densify_month(month, dense_month, verify_every=1)
    assert (n_written, skipped) == (48, 0)
    assert read_store_meta(dense_month).schema == schema

    index = daystore.convert_month(dense_month, tmp_path / "days", horizons=(300, 3600),
                                   log=lambda *_: None)
    assert [d["date"] for d in index["days"]] == ["2024-03-09", "2024-03-10"]
    record = DayRecord(tmp_path / "days" / "2024" / "03" / "2024-03-09")
    assert record.schema == schema and record.default_objective.price == "close"
    assert record.types == ("return",) and record.rows == 1440 and record.bar_seconds == 60

    table = build_month("2024-03", month, tmp_path / "tables", None, 1, horizons=(300, 3600))
    stats = AnchorTargetStats(table)
    assert stats.window_seconds == 60 and stats.price == "close"
    # Both paths saw the same cross-sections: identical name counts per
    # (anchor, type, horizon) cell, and the table's mean is the day's raw mean.
    d = stats.dates.index("2024-03-09")
    np.testing.assert_array_equal(stats.count[d], record.count)
    raw = np.asarray(record.targets_array("raw"), dtype=np.float64)       # (A, N, T, H)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the anchors with no target
        mean = np.nanmean(raw, axis=1)
    usable = np.isfinite(stats.mu[d])
    assert usable.any()
    np.testing.assert_allclose(stats.mu[d][usable], mean[usable], rtol=1e-5, atol=1e-9)
