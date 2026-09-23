"""Evaluation building blocks for financial models and portfolios."""

from stable_finance.backtest import BacktestReport, run_backtest, sweep_configs
from stable_finance.config import (
    ANNUALIZATIONS,
    BacktestConfig,
    CONFIG_SPACE,
    DEFAULT_BACKTEST,
    REBALANCE_RULES,
    enumerate_configs,
)
from stable_finance.costs import COST_MODELS, trading_cost
from stable_finance.risk import (
    RISK_MODELS,
    diagonal_covariance,
    embedding_covariance,
    estimate_risk,
    ledoit_wolf_covariance,
)
from stable_finance.selection import (
    SELECTION_RULES, persistent_selection, select_book, tail_selection,
    threshold_selection,
)
from stable_finance.universe import (
    UNIVERSE_SCREENS,
    apply_screen,
    liquidity_screen,
    presence_screen,
    tradable,
)
from stable_finance.weighting import WEIGHTING_RULES, weight_book
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
from stable_finance.market import (
    MarketPanel,
    decision_labels,
    find_cached_month,
    to_grid,
)
from stable_finance.metrics import (
    InformationCoefficient,
    cross_spread_sharpe,
    edge_to_cost,
    evaluate_forward_returns,
    grouped_rank_ic,
    grouped_rank_ic_by_label,
    mid_price_sharpe,
    sharpe_ratio,
    paired_difference,
    pooled_estimates,
)
from stable_finance.portfolio import BacktestResult, evaluate_weights
from stable_finance.streaming import StreamingRidge
from stable_finance.probes import ColumnwiseRidge, MonthlyProbeData, RidgeProbe, fit

__all__ = [
    "ANNUALIZATIONS",
    "AnchorTargetStats",
    "BacktestConfig",
    "BacktestReport",
    "BacktestResult",
    "CONFIG_SPACE",
    "COST_MODELS",
    "ColumnwiseRidge",
    "CrossSectionalTargetBundle",
    "DEFAULT_BACKTEST",
    "Embeddings",
    "Fills",
    "ForwardReturns",
    "InformationCoefficient",
    "MarketPanel",
    "MeanVarianceAllocator",
    "MonthlyProbeData",
    "Orders",
    "PortfolioWeights",
    "REBALANCE_RULES",
    "RISK_MODELS",
    "RidgeProbe",
    "SELECTION_RULES",
    "StreamingRidge",
    "TARGET_TRANSFORMS",
    "UNIVERSE_SCREENS",
    "WEIGHTING_RULES",
    "allocate",
    "apply_screen",
    "cross_spread_sharpe",
    "decision_labels",
    "diagonal_covariance",
    "edge_to_cost",
    "embedding_covariance",
    "enumerate_configs",
    "estimate_covariance",
    "estimate_risk",
    "evaluate_fills",
    "evaluate_forward_returns",
    "evaluate_weights",
    "find_cached_month",
    "fit",
    "grouped_rank_ic",
    "grouped_rank_ic_by_label",
    "ledoit_wolf_covariance",
    "liquidity_screen",
    "mid_price_sharpe",
    "orders_from_weights",
    "paired_difference",
    "persistent_selection",
    "pooled_estimates",
    "positions_from_fills",
    "presence_screen",
    "run_backtest",
    "select_book",
    "sharpe_ratio",
    "simulate_fills",
    "sweep_configs",
    "tail_selection",
    "threshold_selection",
    "to_grid",
    "tradable",
    "trading_cost",
    "weight_book",
]
