"""Evaluation building blocks for financial models and portfolios."""

from stable_finance.data import ForwardReturns, PortfolioWeights
from stable_finance.metrics import (
    InformationCoefficient,
    evaluate_forward_returns,
    grouped_rank_ic,
    sharpe_ratio,
)
from stable_finance.portfolio import BacktestResult, evaluate_weights

__all__ = [
    "BacktestResult",
    "ForwardReturns",
    "InformationCoefficient",
    "PortfolioWeights",
    "evaluate_forward_returns",
    "evaluate_weights",
    "grouped_rank_ic",
    "sharpe_ratio",
]
