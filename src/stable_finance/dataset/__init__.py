"""Dataset-native schemas and preprocessing.

The modules in this package describe the supported US-equity dataset without
coupling it to a particular model. Storage backends live beside these types;
MDS is the v1 backend and can later be joined by LanceDB without changing the
session or view contracts.
"""

from stable_finance.dataset.calendar import (
    EASTERN_TIME,
    MarketSchedule,
    standard_open_est,
    timeline_bounds_est,
)
from stable_finance.dataset.anchors import AnchorSpec, DEFAULT_ANCHOR_SPEC
from stable_finance.dataset.grid import SessionPreprocessor, sparse_to_dense_grid
from stable_finance.dataset.months import Month, next_month
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES,
    PAIR_TARGET_TYPES,
    anchor_indices,
    anchor_targets,
    compute_pair_targets,
    forward_vwap,
    get_target_names,
    window_realized_volatility,
    window_spread,
)
from stable_finance.dataset.periods import (
    infer_period_frequency,
    period_directories,
    validate_period_alignment,
)
from stable_finance.dataset.schema import (
    MARKET_SCHEMA,
    FeatureSchema,
    MarketSession,
)
from stable_finance.dataset.transforms import (
    aggregate,
    build_norm_groups,
    ffill_vwap,
    normalize,
    prior_vwap,
)
from stable_finance.dataset.targets import (
    AnchorTargetStats,
    CrossSectionalTargetBundle,
    TARGET_TRANSFORMS,
    build_cross_section_metadata,
)
from stable_finance.dataset.views import ViewSpec

__all__ = [
    "FeatureSchema",
    "AnchorSpec",
    "DEFAULT_ANCHOR_SPEC",
    "EASTERN_TIME",
    "MARKET_SCHEMA",
    "MarketSchedule",
    "MarketSession",
    "Month",
    "SessionPreprocessor",
    "ViewSpec",
    "AnchorTargetStats",
    "CrossSectionalTargetBundle",
    "TARGET_TRANSFORMS",
    "ANCHOR_TARGET_TYPES",
    "PAIR_TARGET_TYPES",
    "anchor_indices",
    "anchor_targets",
    "compute_pair_targets",
    "build_cross_section_metadata",
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
    "sparse_to_dense_grid",
    "standard_open_est",
    "timeline_bounds_est",
    "validate_period_alignment",
    "window_realized_volatility",
    "window_spread",
]
