"""Model-neutral construction of synchronized cross-sectional panels."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from stable_finance.dataset.anchors import (
    DEFAULT_TARGET_HORIZONS,
    SESSION_LEN,
    AnchorSpec,
    DEFAULT_ANCHOR_SPEC,
)
from stable_finance.dataset.calendar import MarketSchedule, standard_open_est
from stable_finance.dataset.grid import SessionPreprocessor
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES,
    anchor_indices,
    anchor_targets,
)
from stable_finance.dataset.schema import MARKET_SCHEMA, MarketSession
from stable_finance.dataset.targets import AnchorTargetStats
from stable_finance.dataset.transforms import (
    aggregate,
    build_norm_groups,
    ffill_vwap,
    prior_vwap,
)
from stable_finance.dataset.views import ViewMetadata, ViewSpec, prepare_view


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
    not quoting, which the execution simulator treats as unfillable.
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
        & (grid + min(requested_horizons) <= SESSION_LEN - 1)
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
) -> list[PanelObservation]:
    """Build requested decisions for one ticker-session.

    Views and natural-unit metadata are returned separately. Model packages
    decide whether and how to encode that metadata into tokens.
    """
    target_types = tuple(target_types)
    horizons = tuple(int(value) for value in horizons)
    groups = (
        build_norm_groups(list(session.schema.columns))
        if normalization_groups is None else normalization_groups
    )
    choices = view_spec.aggregation_seconds or range(6, 12)
    fixed = choices[0] if choices[0] == choices[1] else None
    features = session.features
    # Read off THIS session's schema rather than a module constant: the
    # execution stage charges the spread that was actually quoted, and a
    # backend that reorders its columns must move these indices with it.
    quote_columns = [
        session.schema.index("bid_price"), session.schema.index("ask_price"),
    ]
    offset = int(session.timestamps[0]) - standard_open_est(session.date)
    output = []
    for anchor_value in anchors:
        anchor = int(anchor_value)
        row = anchor - offset
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
        view = aggregate(features[start:start + window], aggregation)
        if view is None or len(view) != view_spec.sequence_length:
            continue
        raw = anchor_targets(
            features, horizons, np.asarray([row]), types=target_types,
        )[0]
        bundle = stats.transform(
            raw, session.date, anchor, target_types, horizons,
        )
        target = bundle.select(target_transform)
        if not np.isfinite(target).any():
            continue
        ffill_vwap(view, prior_vwap(features, start), schema=session.schema)
        view, metadata = prepare_view(
            view, groups,
            start_seconds=offset + start,
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
            quote=features[row, quote_columns].astype(np.float64),
        ))
    return output


def build_sample_panel(
    sample,
    schedule: MarketSchedule,
    anchors,
    stats: AnchorTargetStats,
    **kwargs,
) -> list[PanelObservation]:
    """Backend-record convenience adapter for :func:`build_session_panel`."""
    session = SessionPreprocessor(schedule=schedule).transform(sample)
    if session is None:
        return []
    return build_session_panel(session, anchors, stats, **kwargs)


def iter_month_panel(
    month_dir,
    stats: AnchorTargetStats,
    schedule: MarketSchedule,
    anchors,
    batch_size: int,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    **kwargs,
):
    """Yield model-neutral view batches from an MDS month."""
    import tempfile

    from stable_finance.dataset.mds import open_shard, shard_list

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
