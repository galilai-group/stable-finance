"""Stage 3, forward returns to weights: universe, selection, risk, sizing, costs, the backtest harness, and portfolio performance."""

from stable_finance.evaluation.portfolio.allocation import (
    MeanVarianceAllocator,
    allocate,
    estimate_covariance,
)
from stable_finance.evaluation.portfolio.backtest import (
    BacktestReport,
    PnLSeries,
    run_backtest,
    sweep_configs,
)
from stable_finance.evaluation.portfolio.config import (
    ALLOCATORS,
    ANNUALIZATIONS,
    AXIS_REQUIRES,
    CONFIG_SPACE,
    DEFAULT_BACKTEST,
    EFQ_GRID,
    REBALANCE_RULES,
    BacktestConfig,
    check_config,
    enumerate_configs,
)
from stable_finance.evaluation.portfolio.contract import (
    PortfolioWeights,
)
from stable_finance.evaluation.portfolio.performance import (
    BacktestResult,
    ExecutionQualityCurve,
    cross_spread_sharpe,
    edge_to_cost,
    evaluate_weights,
    mid_price_sharpe,
    sharpe_ratio,
    sharpe_vs_execution_quality,
)
from stable_finance.evaluation.portfolio.risk import (
    RISK_MODEL_REQUIRES,
    RISK_MODELS,
    diagonal_covariance,
    embedding_covariance,
    estimate_risk,
    ledoit_wolf_covariance,
)
from stable_finance.evaluation.portfolio.selection import (
    SELECTION_REQUIRES,
    SELECTION_RULES,
    persistent_selection,
    select_book,
    tail_selection,
    threshold_selection,
)
from stable_finance.evaluation.portfolio.universe import (
    UNIVERSE_REQUIRES,
    UNIVERSE_SCREENS,
    apply_screen,
    liquidity_screen,
    presence_screen,
    tradable,
)
from stable_finance.evaluation.portfolio.weighting import (
    WEIGHTING_REQUIRES,
    WEIGHTING_RULES,
    weight_book,
)

__all__ = [
    "ALLOCATORS",
    "ANNUALIZATIONS",
    "AXIS_REQUIRES",
    "BacktestConfig",
    "BacktestReport",
    "BacktestResult",
    "CONFIG_SPACE",
    "DEFAULT_BACKTEST",
    "EFQ_GRID",
    "ExecutionQualityCurve",
    "MeanVarianceAllocator",
    "PnLSeries",
    "PortfolioWeights",
    "REBALANCE_RULES",
    "RISK_MODELS",
    "RISK_MODEL_REQUIRES",
    "SELECTION_REQUIRES",
    "SELECTION_RULES",
    "UNIVERSE_REQUIRES",
    "UNIVERSE_SCREENS",
    "WEIGHTING_REQUIRES",
    "WEIGHTING_RULES",
    "allocate",
    "apply_screen",
    "check_config",
    "cross_spread_sharpe",
    "diagonal_covariance",
    "edge_to_cost",
    "embedding_covariance",
    "enumerate_configs",
    "estimate_covariance",
    "estimate_risk",
    "evaluate_weights",
    "ledoit_wolf_covariance",
    "liquidity_screen",
    "mid_price_sharpe",
    "persistent_selection",
    "presence_screen",
    "run_backtest",
    "select_book",
    "sharpe_ratio",
    "sharpe_vs_execution_quality",
    "sweep_configs",
    "tail_selection",
    "threshold_selection",
    "tradable",
    "weight_book",
]
