"""Model-neutral evaluation panels -- views beside their labels -- and their disk cache.

PANELS. Model-neutral construction of synchronized cross-sectional panels.

PANEL CACHE. Disk cache for model-neutral synchronized panels.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stable_finance.dataset.labels.families import ANCHOR_TARGET_TYPES, anchor_indices
from stable_finance.dataset.labels.objective import Objective
from stable_finance.dataset.labels.tables import AnchorTargetStats
from stable_finance.dataset.loader.grid import SessionPreprocessor
from stable_finance.dataset.loader.views import (
    ViewMetadata,
    ViewSpec,
    aggregate,
    build_norm_groups,
    ffill_vwap,
    prepare_view,
    prior_vwap,
)
from stable_finance.dataset.spec.schema import FeatureSchema, MarketSession
from stable_finance.dataset.spec.timeline import (
    DEFAULT_ANCHOR_SPEC,
    DEFAULT_TARGET_HORIZONS,
    AnchorSpec,
    as_timeline,
)


# ── panels ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PanelObservation:
    """One model-neutral view and its aligned target metadata."""

    date: str
    ticker: str
    anchor: int
    view: np.ndarray
    metadata: ViewMetadata
    target: np.ndarray
    raw_target: np.ndarray
    quote: np.ndarray | None = None
    """``[best_bid, best_ask]`` in natural units at the decision row.

    The observable market at the instant the decision is made, carried beside
    the view because the view is normalized and its scale is per-observation.
    ``None`` when the producer does not supply it; NaN marks an asset that was
    not quoting, which the execution simulator treats as unfillable. Also
    ``None`` for a schema without both ``bid_price`` and ``ask_price``.
    """


def choose_cell_aggregation(
    date: str,
    anchor: int,
    latest_index: int,
    sequence_length: int = 2048,
    *,
    choices=range(6, 12),
) -> int | None:
    """Choose a reproducible feasible resolution for a decision cell."""
    feasible = [int(value) for value in choices
                if int(value) * sequence_length <= latest_index + 1]
    if not feasible:
        return None
    seed = zlib.crc32(f"{date}@{int(anchor)}".encode())
    return feasible[np.random.RandomState(seed).randint(len(feasible))]


def evenly_spaced_anchors(
    n_per_day: int,
    sequence_length: int = 2048,
    *,
    aggregation_choices=range(6, 12),
    horizons=DEFAULT_TARGET_HORIZONS,
    anchor_spec: AnchorSpec = DEFAULT_ANCHOR_SPEC,
) -> np.ndarray:
    """Choose evenly spaced anchors over the feasible decision band."""
    if n_per_day < 1:
        raise ValueError("n_per_day must be positive")
    choices = tuple(int(value) for value in aggregation_choices)
    requested_horizons = tuple(int(value) for value in horizons)
    grid = anchor_indices(spec=anchor_spec)
    grid = grid[
        (grid >= min(choices) * sequence_length - 1)
        & (grid + min(requested_horizons) <= anchor_spec.session_seconds - 1)
    ]
    if not len(grid):
        raise ValueError("no anchor admits a full view")
    indices = np.linspace(0, len(grid) - 1, n_per_day).round().astype(int)
    return grid[np.unique(indices)]


def build_session_panel(
    session: MarketSession,
    anchors,
    stats: AnchorTargetStats,
    *,
    view_spec: ViewSpec = ViewSpec(aggregation_seconds=(6, 11)),
    target_types=ANCHOR_TARGET_TYPES,
    horizons=DEFAULT_TARGET_HORIZONS,
    target_transform: str = "uniform",
    normalization_groups=None,
    timeline=None,
    objective=None,
) -> list[PanelObservation]:
    """Build requested decisions for one ticker-session.

    Views and natural-unit metadata are returned separately. Model packages
    decide whether and how to encode that metadata into tokens.

    ``anchors`` are seconds past the timeline's origin (Market-1T's 09:30 by
    default). View aggregation counts stored bars, which are seconds only at
    1 Hz. The labels are the ``objective``'s; without one, ``target_types``
    and ``horizons`` define it, and ``stats`` must be a table built for the
    same objective.
    """
    timeline = as_timeline(timeline)
    if timeline.bar_seconds != session.bar_seconds:
        raise ValueError(
            f"the timeline is at {timeline.bar_seconds}s bars, the session at "
            f"{session.bar_seconds}s")
    # The labels are the objective's to compute; the panel only joins them to
    # the views.
    if objective is None:
        objective = Objective(types=tuple(target_types), horizons=tuple(horizons))
    objective = objective.resolve(session.schema)
    target_types, horizons = objective.types, objective.horizons
    table_window = getattr(stats, "window_seconds", None)
    if table_window is not None and table_window != objective.window_seconds:
        raise ValueError(
            f"the target table was built over a {table_window}s window, the labels "
            f"over {objective.window_seconds}s: standardizing one against "
            "the other mixes two different targets")
    bar = session.bar_seconds
    target_types = tuple(target_types)
    horizons = tuple(int(value) for value in horizons)
    groups = (
        build_norm_groups(session.schema)
        if normalization_groups is None else normalization_groups
    )
    choices = view_spec.aggregation_seconds or range(6, 12)
    fixed = choices[0] if choices[0] == choices[1] else None
    features = session.features
    # Read off THIS session's schema rather than a module constant: the
    # execution stage charges the spread that was actually quoted, and a
    # backend that reorders its columns must move these indices with it.
    quote_columns = (
        [session.schema.role_index("bid_price"), session.schema.role_index("ask_price")]
        if session.schema.has("bid_price", "ask_price") else None
    )
    offset = int(session.timestamps[0]) - timeline.origin(session.date)
    output = []
    for anchor_value in anchors:
        anchor = int(anchor_value)
        if (anchor - offset) % bar:
            continue
        row = (anchor - offset) // bar
        if row < 0 or row >= len(features):
            continue
        aggregation = (
            fixed if fixed is not None and fixed * view_spec.sequence_length <= anchor + 1
            else None if fixed is not None
            else choose_cell_aggregation(
                session.date, anchor, anchor, view_spec.sequence_length,
                choices=choices,
            )
        )
        if aggregation is None:
            continue
        window = aggregation * view_spec.sequence_length
        start = row + 1 - window
        if start < 0:
            continue
        view = aggregate(features[start:start + window], aggregation,
                         schema=session.schema)
        if view is None or len(view) != view_spec.sequence_length:
            continue
        raw = objective.raw(features, [row], schema=session.schema, timeline=timeline)[0]
        bundle = stats.transform(
            raw, session.date, anchor, target_types, horizons,
        )
        target = bundle.select(target_transform)
        if not np.isfinite(target).any():
            continue
        ffill_vwap(view, prior_vwap(features, start, schema=session.schema),
                   schema=session.schema)
        view, metadata = prepare_view(
            view, groups,
            start_seconds=offset + start * bar,
            aggregation_seconds=aggregation,
        )
        np.nan_to_num(view, copy=False, nan=0.0)
        output.append(PanelObservation(
            date=session.date,
            ticker=session.ticker,
            anchor=anchor,
            view=view.astype(np.float32),
            metadata=metadata,
            target=target.ravel().astype(np.float32),
            raw_target=raw.ravel().astype(np.float32),
            quote=(features[row, quote_columns].astype(np.float64)
                   if quote_columns is not None else None),
        ))
    return output


def build_sample_panel(
    sample,
    schedule,
    anchors,
    stats: AnchorTargetStats,
    *,
    schema: FeatureSchema | None = None,
    **kwargs,
) -> list[PanelObservation]:
    """Backend-record convenience adapter for :func:`build_session_panel`.

    ``schedule`` is the record's timeline, or a Market-1T holiday schedule.
    """
    timeline = as_timeline(schedule)
    session = SessionPreprocessor(schema=schema, timeline=timeline).transform(sample)
    if session is None:
        return []
    return build_session_panel(session, anchors, stats, timeline=timeline, **kwargs)


def iter_month_panel(
    month_dir,
    stats: AnchorTargetStats,
    schedule,
    anchors,
    batch_size: int,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    **kwargs,
):
    """Yield model-neutral view batches from an MDS month.

    The month's schema and timeline come from its ``dataset.json``;
    ``schedule`` is a Market-1T month's holiday schedule.
    """
    import tempfile

    from stable_finance.dataset.storage.mosaic import (
        open_shard,
        shard_list,
    )
    from stable_finance.dataset.storage.meta import read_store_meta

    meta = read_store_meta(month_dir)
    kwargs.setdefault("schema", meta.schema)
    schedule = meta.timeline(schedule)
    if batch_size < 1 or num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid batch or shard configuration")
    shards = shard_list(month_dir)[shard_index::num_shards]
    views: list[np.ndarray] = []
    observations: list[PanelObservation] = []
    with tempfile.TemporaryDirectory(prefix="stable_finance_panel_") as scratch:
        for shard in shards:
            reader = open_shard(month_dir, shard, scratch)
            for index in range(shard["samples"]):
                rows = build_sample_panel(
                    reader.get_item(index), schedule, anchors, stats, **kwargs,
                )
                for row in rows:
                    views.append(row.view)
                    observations.append(row)
                    if len(views) == batch_size:
                        yield np.stack(views), observations
                        views, observations = [], []
    if views:
        yield np.stack(views), observations


# ── panel cache ──────────────────────────────────────────────────────────────

# 2 (2026-09-10): panels carry the QUOTE column. The bump is the point: a
# panel built under 1 has no quotes, and a reader that filled the gap with
# NaN would hand the execution stage an unquoted market -- nothing fills,
# every spread charges zero, and a frictionless Sharpe reports as a
# spread-charged one. A miss costs a rebuild; a silent NaN costs a result.
PANEL_CACHE_VERSION = 2


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
    #: ``(n_rows, 2)`` bid and ask, or None for a panel whose dataset has no
    #: quotes at all -- recorded as ``"quotes": false`` in its meta.json.
    quotes: np.ndarray | None
    metadata: dict


def cache_key(**identity) -> dict:
    """Return a serializable identity plus a deterministic short hash."""
    value = {"format": PANEL_CACHE_VERSION, **identity}
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
        if metadata.get("quotes", True) is False:
            names.pop("quotes")
        arrays = {
            name: np.load(directory / filename, allow_pickle=False)
            for name, filename in names.items()
        }
        arrays.setdefault("quotes", None)
        if not self._raw_is_complete(directory, metadata):
            raise RuntimeError(f"panel cache at {directory} is truncated")
        if any(len(value) != len(views) for value in arrays.values()
               if value is not None):
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
                    quote=None if panel.quotes is None else panel.quotes[index],
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
                        # None is a dataset with no bid/ask columns at all,
                        # which is NOT a NaN quote: NaN means a name that was
                        # not quoting, and storing a missing column that way
                        # would hand the execution stage an untradable market
                        # where it should refuse. All rows of a panel come
                        # from one schema, so the rows agree; the column is
                        # written only when they have quotes.
                        columns["quotes"].append(
                            None if row.quote is None
                            else np.asarray(row.quote, dtype=np.float64))
                    n_rows += len(views)
            if not n_rows:
                raise RuntimeError("cannot cache an empty panel")
            quoted = {quote is not None for quote in columns["quotes"]}
            if len(quoted) > 1:
                raise ValueError("some observations carry quotes and some do not")
            has_quotes = quoted == {True}
            if not has_quotes:
                columns.pop("quotes")
            for name, values in columns.items():
                np.save(partial / f"{name}.npy", np.asarray(values))
            metadata = {
                "month": month, "key": key, "n_rows": n_rows,
                "views_shape": [n_rows, *view_shape],
                "views_dtype": self.view_dtype.name,
                "quotes": has_quotes,
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
