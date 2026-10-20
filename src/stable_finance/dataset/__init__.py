"""Market time series in; sessions, views and labels out. Grouped by job:

``spec``      what a dataset is: columns by role and price methods (``schema``),
              sessions and the decision grid (``calendar``)
``storage``   where it lives: store locations, the ``dataset.json`` description,
              the MDS ``mosaic`` and its conversions, the ``daystore``
``loader``    what a model sees: dense sessions (``grid``), ``views``,
              evaluation ``panels`` and their cache, training ``cells``
``labels``    what a model is scored on: target ``families``, the ``objective``
              and its cross-sectional transforms, cross-section ``tables``
``ingest``    any table or array in, with its bar length and stamp declared
``datasets``  concrete datasets, one package each; ``datasets.market1t`` holds
              Market-1T's spec, Polygon ingestion and industry map

Nothing downstream assumes a dataset: a schema says what each column means,
a calendar says when the market is open, and a store records both. Market-1T
is the default wherever a caller or a store names nothing else, through
:func:`~stable_finance.dataset.spec.schema.default_schema`. Everything public
is re-exported here.
"""

from stable_finance.dataset.datasets.market1t.spec import (
    EASTERN_TIME,
    MARKET_SCHEMA,
    Market1TCalendar,
    MarketSchedule,
    market1t_timeline,
    standard_open_est,
    timeline_bounds_est,
)
from stable_finance.dataset.ingest import (
    records_from_arrays,
    records_from_frame,
    sessions_from_frame,
    sessions_from_series,
    write_daystore,
    write_mosaic,
)
from stable_finance.dataset.labels.families import (
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
from stable_finance.dataset.labels.objective import CROSS_SECTIONS, Objective
from stable_finance.dataset.labels.tables import (
    TARGET_TRANSFORMS,
    AnchorTargetStats,
    CrossSectionalTargetBundle,
    build_cross_section_metadata,
)
from stable_finance.dataset.loader.grid import (
    CacheInfo,
    SessionPreprocessor,
    sparse_to_dense_grid,
)
from stable_finance.dataset.loader.panels import (
    CacheClaimed,
    CachedPanel,
    PanelCache,
    PanelObservation,
    build_sample_panel,
    build_session_panel,
    cache_key as panel_cache_key,
    choose_cell_aggregation,
    evenly_spaced_anchors,
    iter_month_panel,
)
from stable_finance.dataset.loader.views import (
    ViewMetadata,
    ViewSpec,
    aggregate,
    build_norm_groups,
    ffill_vwap,
    normalize,
    prepare_view,
    prior_vwap,
    resample_session,
)
from stable_finance.dataset.spec.schema import (
    PRICE_METHODS,
    ROLES,
    FeatureSchema,
    MarketSession,
    MissingRolesError,
    default_price,
    default_schema,
    mid_series,
)
from stable_finance.dataset.spec.timeline import (
    DEFAULT_ANCHOR_SPEC,
    DEFAULT_TARGET_HORIZONS,
    STAMPS,
    TO_CLOSE,
    AnchorSpec,
    ContinuousSession,
    FixedSession,
    SessionCalendar,
    Timeline,
    as_timeline,
)
from stable_finance.dataset.storage.locations import (
    Month,
    infer_period_frequency,
    next_month,
    period_directories,
    validate_period_alignment,
)

__all__ = [
    "Timeline",
    "as_timeline",
    "market1t_timeline",
    "default_schema",
    "CROSS_SECTIONS",
    "STAMPS",
    "TO_CLOSE",
    "Objective",
    "ContinuousSession",
    "FixedSession",
    "Market1TCalendar",
    "MissingRolesError",
    "PRICE_METHODS",
    "ROLES",
    "SessionCalendar",
    "anchor_offsets",
    "check_targets",
    "default_price",
    "forward_price",
    "mid_series",
    "records_from_arrays",
    "records_from_frame",
    "sessions_from_frame",
    "sessions_from_series",
    "supported_targets",
    "write_daystore",
    "write_mosaic",
    "FeatureSchema",
    "AnchorSpec",
    "DEFAULT_ANCHOR_SPEC",
    "DEFAULT_TARGET_HORIZONS",
    "EASTERN_TIME",
    "MARKET_SCHEMA",
    "MarketSchedule",
    "MarketSession",
    "Month",
    "SessionPreprocessor",
    "CacheInfo",
    "ViewSpec",
    "ViewMetadata",
    "PanelObservation",
    "PanelCache",
    "CachedPanel",
    "CacheClaimed",
    "AnchorTargetStats",
    "CrossSectionalTargetBundle",
    "TARGET_TRANSFORMS",
    "ANCHOR_TARGET_TYPES",
    "PAIR_TARGET_TYPES",
    "anchor_indices",
    "anchor_targets",
    "compute_pair_targets",
    "build_cross_section_metadata",
    "build_sample_panel",
    "build_session_panel",
    "choose_cell_aggregation",
    "evenly_spaced_anchors",
    "iter_month_panel",
    "panel_cache_key",
    "aggregate",
    "build_norm_groups",
    "ffill_vwap",
    "forward_vwap",
    "get_target_names",
    "infer_period_frequency",
    "next_month",
    "normalize",
    "period_directories",
    "prior_vwap",
    "prepare_view",
    "resample_session",
    "sparse_to_dense_grid",
    "standard_open_est",
    "timeline_bounds_est",
    "validate_period_alignment",
    "window_realized_volatility",
    "window_spread",
]
