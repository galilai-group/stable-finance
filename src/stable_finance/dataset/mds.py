"""MosaicML Streaming backend for the supported market dataset."""

from __future__ import annotations

import json
import math
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
    try:
        directories = period_directories(*args, **kwargs)
    except ValueError as error:
        message = str(error)
        if message.startswith("No dataset period directories"):
            message = message.replace(
                "No dataset period directories", "No MDS period directories", 1
            )
            raise ValueError(message) from error
        raise
    return [Stream(local=str(path)) for path in directories]


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


def configure_epoch_size(dataset, total_steps: int, batch_size: int) -> None:
    """Resize a MosaicML dataset epoch to cover a complete training run.

    MosaicML otherwise recycles samples at its epoch boundary during a
    step-based run. The ten-batch margin keeps dataloader prefetch from
    crossing that boundary. This mutates the supplied dataset in place.
    """
    if total_steps < 1 or batch_size < 1:
        raise ValueError("total_steps and batch_size must be positive")
    epoch_samples = (total_steps + 10) * batch_size
    for stream in dataset.streams:
        for attribute in ("proportion", "repeat", "choose"):
            if hasattr(stream, attribute):
                delattr(stream, attribute)
    dataset.epoch_size = Stream.apply_weights(
        dataset.streams,
        dataset.samples_per_stream,
        epoch_samples,
        dataset.shuffle_seed,
    )
    dataset.length = math.ceil(
        dataset.epoch_size / dataset._parallel_rank_world.num_ranks
    )


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
