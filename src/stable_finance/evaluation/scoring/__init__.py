"""Stage 5, how good each stage's output is: the IC of a forecast, the P&L and Sharpe of a portfolio, and execution-quality curves."""

from stable_finance.evaluation.scoring.ic import (
    InformationCoefficient,
    evaluate_forward_returns,
    grouped_rank_ic,
    grouped_rank_ic_by_label,
    paired_difference,
    pooled_estimates,
)
from stable_finance.evaluation.scoring.pnl import (
    BacktestResult,
    evaluate_fills,
    evaluate_weights,
    positions_from_fills,
)
from stable_finance.evaluation.scoring.sharpe import (
    ExecutionQualityCurve,
    breakeven_efq,
    cross_spread_sharpe,
    mid_price_sharpe,
    sharpe_ratio,
    sharpe_vs_execution_quality,
)

__all__ = [
    "BacktestResult",
    "ExecutionQualityCurve",
    "InformationCoefficient",
    "breakeven_efq",
    "cross_spread_sharpe",
    "evaluate_fills",
    "evaluate_forward_returns",
    "evaluate_weights",
    "grouped_rank_ic",
    "grouped_rank_ic_by_label",
    "mid_price_sharpe",
    "paired_difference",
    "pooled_estimates",
    "positions_from_fills",
    "sharpe_ratio",
    "sharpe_vs_execution_quality",
]
