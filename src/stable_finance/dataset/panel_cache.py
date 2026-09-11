"""Disk cache for model-neutral synchronized panels."""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stable_finance.dataset.panels import PanelObservation
from stable_finance.dataset.views import ViewMetadata


# 2 (2026-09-10): panels carry the QUOTE column. The bump is the point: a
# panel built under 1 has no quotes, and a reader that filled the gap with
# NaN would hand the execution stage an unquoted market -- nothing fills,
# every spread charges zero, and a frictionless Sharpe reports as a
# spread-charged one. A miss costs a rebuild; a silent NaN costs a result.
FORMAT_VERSION = 2


class CacheClaimed(RuntimeError):
    """Another process currently owns this cache build."""


@dataclass(frozen=True)
class CachedPanel:
    views: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    dates: np.ndarray
    anchors: np.ndarray
    tickers: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    aggregations: np.ndarray
    normalization_means: np.ndarray
    normalization_scales: np.ndarray
    quotes: np.ndarray
    metadata: dict


def cache_key(**identity) -> dict:
    """Return a serializable identity plus a deterministic short hash."""
    value = {"format": FORMAT_VERSION, **identity}
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["hash"] = hashlib.sha256(blob).hexdigest()[:12]
    return value


class PanelCache:
    """Materialize and stream normalized views with natural-unit metadata."""

    def __init__(self, root: str | Path, *, view_dtype=np.float16):
        self.root = Path(root)
        self.view_dtype = np.dtype(view_dtype)

    def directory(self, month: str, key: dict) -> Path:
        return self.root / key["hash"] / month

    @staticmethod
    def _raw_is_complete(directory: Path, metadata: dict) -> bool:
        expected = (
            int(np.prod(metadata["views_shape"]))
            * np.dtype(metadata["views_dtype"]).itemsize
        )
        try:
            return (directory / "views.raw").stat().st_size == expected
        except OSError:
            return False

    def is_built(self, month: str, key: dict) -> bool:
        directory = self.directory(month, key)
        if not (directory / ".ready").is_file():
            return False
        try:
            metadata = json.loads((directory / "meta.json").read_text())
            return (
                metadata["key"]["hash"] == key["hash"]
                and self._raw_is_complete(directory, metadata)
            )
        except (OSError, ValueError, KeyError):
            return False

    def load(self, month: str, key: dict) -> CachedPanel:
        directory = self.directory(month, key)
        metadata = json.loads((directory / "meta.json").read_text())
        if metadata["key"]["hash"] != key["hash"]:
            raise RuntimeError("panel cache key mismatch")
        shape = tuple(metadata["views_shape"])
        views = np.memmap(
            directory / "views.raw", dtype=np.dtype(metadata["views_dtype"]),
            mode="r", shape=shape,
        )
        names = {
            "targets": "targets.npy", "raw_targets": "raw_targets.npy",
            "dates": "dates.npy", "anchors": "anchors.npy",
            "tickers": "tickers.npy", "starts": "starts.npy",
            "ends": "ends.npy", "aggregations": "aggregations.npy",
            "normalization_means": "normalization_means.npy",
            "normalization_scales": "normalization_scales.npy",
            "quotes": "quotes.npy",
        }
        arrays = {
            name: np.load(directory / filename, allow_pickle=False)
            for name, filename in names.items()
        }
        if not self._raw_is_complete(directory, metadata):
            raise RuntimeError(f"panel cache at {directory} is truncated")
        if any(len(value) != len(views) for value in arrays.values()):
            raise RuntimeError(f"panel cache at {directory} has ragged columns")
        return CachedPanel(views=views, metadata=metadata, **arrays)

    def iter_batches(self, month: str, key: dict, batch_size: int):
        panel = self.load(month, key)
        for start in range(0, len(panel.views), batch_size):
            stop = min(start + batch_size, len(panel.views))
            sl = slice(start, stop)
            observations = []
            for index in range(start, stop):
                observations.append(PanelObservation(
                    date=str(panel.dates[index]),
                    ticker=str(panel.tickers[index]),
                    anchor=int(panel.anchors[index]),
                    view=np.asarray(panel.views[index], dtype=np.float32),
                    metadata=ViewMetadata(
                        start_seconds=float(panel.starts[index]),
                        end_seconds=float(panel.ends[index]),
                        aggregation_seconds=float(panel.aggregations[index]),
                        normalization_means=panel.normalization_means[index],
                        normalization_scales=panel.normalization_scales[index],
                    ),
                    target=panel.targets[index],
                    raw_target=panel.raw_targets[index],
                    quote=panel.quotes[index],
                ))
            yield np.asarray(panel.views[sl], dtype=np.float32), observations

    def build(self, month: str, key: dict, batches, *, overwrite=False) -> Path:
        directory = self.directory(month, key)
        if self.is_built(month, key) and not overwrite:
            return directory
        directory.parent.mkdir(parents=True, exist_ok=True)
        lock = open(directory.with_suffix(".lock"), "w")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise CacheClaimed(f"{month} is being built elsewhere") from error
            if self.is_built(month, key) and not overwrite:
                return directory
            partial = directory.with_suffix(".partial")
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            columns = {name: [] for name in (
                "targets", "raw_targets", "dates", "anchors", "tickers",
                "starts", "ends", "aggregations", "normalization_means",
                "normalization_scales", "quotes",
            )}
            n_rows, view_shape = 0, None
            with open(partial / "views.raw", "wb", buffering=1 << 22) as output:
                for views, observations in batches:
                    views = np.asarray(views)
                    if len(views) != len(observations):
                        raise ValueError("views and observations are not aligned")
                    if view_shape is None:
                        view_shape = views.shape[1:]
                    elif views.shape[1:] != view_shape:
                        raise ValueError("view shape changed during cache build")
                    output.write(np.ascontiguousarray(
                        views, dtype=self.view_dtype
                    ).tobytes())
                    for row in observations:
                        columns["targets"].append(row.target)
                        columns["raw_targets"].append(row.raw_target)
                        columns["dates"].append(row.date)
                        columns["anchors"].append(row.anchor)
                        columns["tickers"].append(row.ticker)
                        columns["starts"].append(row.metadata.start_seconds)
                        columns["ends"].append(row.metadata.end_seconds)
                        columns["aggregations"].append(row.metadata.aggregation_seconds)
                        columns["normalization_means"].append(row.metadata.normalization_means)
                        columns["normalization_scales"].append(row.metadata.normalization_scales)
                        # An observation whose backend carried no bid/ask is
                        # stored as NaN, which is what an absent quote MEANS
                        # downstream -- distinct from the column being missing
                        # entirely, which is what version 1 could not express.
                        columns["quotes"].append(
                            np.full(2, np.nan, dtype=np.float64)
                            if row.quote is None
                            else np.asarray(row.quote, dtype=np.float64))
                    n_rows += len(views)
            if not n_rows:
                raise RuntimeError("cannot cache an empty panel")
            for name, values in columns.items():
                np.save(partial / f"{name}.npy", np.asarray(values))
            metadata = {
                "month": month, "key": key, "n_rows": n_rows,
                "views_shape": [n_rows, *view_shape],
                "views_dtype": self.view_dtype.name,
            }
            (partial / "meta.json").write_text(json.dumps(metadata, indent=2))
            (partial / ".ready").touch()
            if directory.exists():
                shutil.rmtree(directory)
            partial.rename(directory)
            return directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
