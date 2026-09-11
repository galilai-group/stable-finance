"""Evaluation building blocks for financial models and portfolios."""

from stable_finance.allocation import (
    MeanVarianceAllocator,
    allocate,
    estimate_covariance,
)
from stable_finance.data import (
    Embeddings,
    Fills,
    ForwardReturns,
    Orders,
    PortfolioWeights,
)
from stable_finance.dataset.targets import (
    AnchorTargetStats,
    CrossSectionalTargetBundle,
    TARGET_TRANSFORMS,
)
from stable_finance.execution import (
    evaluate_fills,
    orders_from_weights,
    positions_from_fills,
    simulate_fills,
)
from stable_finance.metrics import (
    InformationCoefficient,
    cross_spread_sharpe,
    evaluate_forward_returns,
    grouped_rank_ic,
    grouped_rank_ic_by_label,
    mid_price_sharpe,
    sharpe_ratio,
    paired_difference,
    pooled_estimates,
)
from stable_finance.portfolio import BacktestResult, evaluate_weights
from stable_finance.probes import ColumnwiseRidge, MonthlyProbeData, RidgeProbe, fit

__all__ = [
    "BacktestResult",
    "AnchorTargetStats",
    "CrossSectionalTargetBundle",
    "ColumnwiseRidge",
    "Embeddings",
    "Fills",
    "ForwardReturns",
    "InformationCoefficient",
    "MeanVarianceAllocator",
    "MonthlyProbeData",
    "Orders",
    "PortfolioWeights",
    "RidgeProbe",
    "TARGET_TRANSFORMS",
    "allocate",
    "cross_spread_sharpe",
    "estimate_covariance",
    "evaluate_fills",
    "evaluate_forward_returns",
    "evaluate_weights",
    "fit",
    "grouped_rank_ic",
    "grouped_rank_ic_by_label",
    "mid_price_sharpe",
    "orders_from_weights",
    "positions_from_fills",
    "sharpe_ratio",
    "simulate_fills",
    "paired_difference",
    "pooled_estimates",
]
