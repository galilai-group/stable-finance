"""Evaluation building blocks for financial models and portfolios."""

from stable_finance.data import Embeddings, ForwardReturns, PortfolioWeights
from stable_finance.dataset.targets import (
    AnchorTargetStats,
    CrossSectionalTargetBundle,
    TARGET_TRANSFORMS,
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
    "ForwardReturns",
    "InformationCoefficient",
    "MonthlyProbeData",
    "PortfolioWeights",
    "RidgeProbe",
    "TARGET_TRANSFORMS",
    "cross_spread_sharpe",
    "evaluate_forward_returns",
    "evaluate_weights",
    "fit",
    "grouped_rank_ic",
    "grouped_rank_ic_by_label",
    "mid_price_sharpe",
    "sharpe_ratio",
    "paired_difference",
    "pooled_estimates",
]
