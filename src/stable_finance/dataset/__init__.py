"""Market time series in; sessions, views and labels out.

Generic, for any dataset:

``schema``     what the columns mean by role, and the price methods
``anchors``    where decisions fall and how long a label measures (``AnchorSpec``)
``outcomes``   the target families: returns, spread and volatility changes, by role
``objective``  one label definition with a cache key, and its cross-sectional forms

A dataset with no machinery of its own needs nothing more: pass its table to
:func:`stable_finance.evaluation.inputs.evaluation_inputs` for evaluation.

Datasets with their own pipeline live in :mod:`stable_finance.dataset.datasets`;
Market-1T's is :mod:`stable_finance.dataset.datasets.market1t`, and it is the
default wherever a caller names nothing else. Everything public is re-exported
here.
"""

from stable_finance.dataset.anchors import (
    DEFAULT_ANCHOR_SPEC,
    DEFAULT_TARGET_HORIZONS,
    TO_CLOSE,
    AnchorSpec,
)
from stable_finance.dataset.objective import (
    CROSS_SECTIONS,
    Objective,
)
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES,
    PAIR_TARGET_TYPES,
    anchor_indices,
    anchor_offsets,
    anchor_targets,
    check_targets,
    compute_pair_targets,
    forward_price,
    forward_vwap,
    get_target_names,
    supported_targets,
    window_realized_volatility,
    window_spread,
)
from stable_finance.dataset.schema import (
    PRICE_METHODS,
    ROLES,
    FeatureSchema,
    MarketSession,
    MissingRolesError,
    default_price,
    default_schema,
    mid_series,
)
from stable_finance.dataset.datasets.market1t.daystore import (
    ChronologyError,
    DayRecord,
    check_chronology,
    forward_returns_from_days,
    market_from_days,
)
from stable_finance.dataset.datasets.market1t.spec import (
    EASTERN_TIME,
    MARKET_SCHEMA,
    Market1TCalendar,
    MarketSchedule,
    Month,
    SessionCalendar,
    Timeline,
    as_timeline,
    infer_period_frequency,
    market1t_timeline,
    next_month,
    period_directories,
    standard_open_est,
    timeline_bounds_est,
    validate_period_alignment,
)
from stable_finance.dataset.datasets.market1t.tables import (
    TARGET_TRANSFORMS,
    AnchorTargetStats,
    CrossSectionalTargetBundle,
    build_cross_section_metadata,
)
from stable_finance.dataset.datasets.market1t.views import (
    CacheClaimed,
    CachedPanel,
    CacheInfo,
    PanelCache,
    PanelObservation,
    SessionPreprocessor,
    ViewMetadata,
    ViewSpec,
    aggregate,
    build_norm_groups,
    build_sample_panel,
    build_session_panel,
    choose_cell_aggregation,
    evenly_spaced_anchors,
    ffill_vwap,
    iter_month_panel,
    normalize,
    prepare_view,
    prior_vwap,
    resample_session,
    sparse_to_dense_grid,
)

from stable_finance.dataset.datasets.market1t.views import cache_key as panel_cache_key

__all__ = [
    "panel_cache_key",
    "ANCHOR_TARGET_TYPES",
    "CROSS_SECTIONS",
    "DEFAULT_ANCHOR_SPEC",
    "DEFAULT_TARGET_HORIZONS",
    "EASTERN_TIME",
    "MARKET_SCHEMA",
    "PAIR_TARGET_TYPES",
    "PRICE_METHODS",
    "ROLES",
    "TARGET_TRANSFORMS",
    "TO_CLOSE",
    "AnchorSpec",
    "AnchorTargetStats",
    "CacheClaimed",
    "CachedPanel",
    "CacheInfo",
    "ChronologyError",
    "CrossSectionalTargetBundle",
    "DayRecord",
    "FeatureSchema",
    "Market1TCalendar",
    "MarketSchedule",
    "MarketSession",
    "MissingRolesError",
    "Month",
    "Objective",
    "PanelCache",
    "PanelObservation",
    "SessionCalendar",
    "SessionPreprocessor",
    "Timeline",
    "ViewMetadata",
    "ViewSpec",
    "aggregate",
    "anchor_indices",
    "anchor_offsets",
    "anchor_targets",
    "as_timeline",
    "build_cross_section_metadata",
    "build_norm_groups",
    "build_sample_panel",
    "build_session_panel",
    "check_chronology",
    "check_targets",
    "choose_cell_aggregation",
    "compute_pair_targets",
    "default_price",
    "default_schema",
    "evenly_spaced_anchors",
    "ffill_vwap",
    "forward_price",
    "forward_returns_from_days",
    "forward_vwap",
    "get_target_names",
    "infer_period_frequency",
    "iter_month_panel",
    "market1t_timeline",
    "market_from_days",
    "mid_series",
    "next_month",
    "normalize",
    "period_directories",
    "prepare_view",
    "prior_vwap",
    "resample_session",
    "sparse_to_dense_grid",
    "standard_open_est",
    "supported_targets",
    "timeline_bounds_est",
    "validate_period_alignment",
    "window_realized_volatility",
    "window_spread",
]
