"""Stage 4, target to trades: the trading policies. How far to move given costs, and whether a trade is worth making; hand-written rules now, with ``rl`` reserved for learned policies."""

from stable_finance.evaluation.trades.rebalance import (
    MeanVarianceAllocator,
    allocate,
    rebalance,
)
from stable_finance.evaluation.trades.rules import (
    edge_to_cost,
    persistent_selection,
    threshold_selection,
)

__all__ = [
    "MeanVarianceAllocator",
    "allocate",
    "edge_to_cost",
    "persistent_selection",
    "rebalance",
    "threshold_selection",
]
