"""MosaicML Streaming backend for the supported market dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from stable_finance.dataset.grid import SessionPreprocessor
from stable_finance.dataset.months import Month
from stable_finance.dataset.periods import (
    infer_period_frequency,
    period_directories,
    validate_period_alignment,
)

try:
    from streaming import Stream, StreamingDataset
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "MDS support requires `uv sync --extra mds`"
    ) from error


def discover_streams(*args, **kwargs):
    """Build MosaicML streams for :func:`period_directories`."""
    return [Stream(local=str(path)) for path in period_directories(*args, **kwargs)]


def shard_list(month_dir: str | Path) -> list[dict]:
    with open(Path(month_dir) / "index.json") as file:
        return json.load(file)["shards"]


def open_shard(month_dir: str | Path, shard_meta: dict, tmpdir: str | Path):
    """Read an MDS shard, expanding compressed data into caller-owned scratch."""
    from streaming.base.format import reader_from_json
    import zstandard

    month_dir, tmpdir = Path(month_dir), Path(tmpdir)
    raw = shard_meta["raw_data"]["basename"]
    if (month_dir / raw).is_file():
        return reader_from_json(str(month_dir), None, shard_meta)
    compressed = shard_meta.get("zip_data")
    if not compressed:
        raise FileNotFoundError(f"{month_dir / raw} missing and not compressed")
    destination = tmpdir / raw
    if not destination.is_file():
        partial = destination.with_suffix(destination.suffix + ".part")
        with open(month_dir / compressed["basename"], "rb") as source, open(
            partial, "wb"
        ) as output:
            zstandard.ZstdDecompressor().copy_stream(source, output)
        partial.replace(destination)
    local = dict(shard_meta)
    local["compression"] = None
    local["zip_data"] = None
    shutil.copy(month_dir / "index.json", tmpdir / "index.json")
    return reader_from_json(str(tmpdir), None, local)


class StreamingMarketDataset(StreamingDataset):
    """MDS sampling plus backend-neutral dense-session preprocessing."""

    def __init__(self, *, preprocessor: SessionPreprocessor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = preprocessor or SessionPreprocessor()

    @classmethod
    def from_month(cls, mosaic_dir: str | Path, month: Month | str, **kwargs):
        """Construct a dataset over one probe-fit or evaluation month."""
        value = Month.parse(month)
        return cls(
            streams=discover_streams(mosaic_dir, value.first_day, value.last_day),
            **kwargs,
        )

    def raw_sample(self, index: int) -> dict:
        """Expose the storage record for specialized research adapters."""
        return super().__getitem__(index)

    def __getitem__(self, index: int):
        return self.preprocessor.transform(self.raw_sample(index))
