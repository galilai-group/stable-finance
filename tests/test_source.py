"""A store root may be a local directory or a Hugging Face repo."""
from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from stable_finance.dataset.daystore import FEATURES, DayRecord, discover_days
from stable_finance.dataset.periods import period_directories
from stable_finance.dataset.source import parse_hub, resolve_root
from test_daystore import DATES, mosaic, store  # noqa: F401  (fixtures)

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
