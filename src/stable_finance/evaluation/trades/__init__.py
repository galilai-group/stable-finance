"""Stage 4, weights to trades: orders, the realized position path, and its score."""

from stable_finance.evaluation.trades.contract import (
    Fills,
    Orders,
)
from stable_finance.evaluation.trades.execution import (
    evaluate_fills,
    orders_from_weights,
    positions_from_fills,
)

__all__ = [
    "Fills",
    "Orders",
    "evaluate_fills",
    "orders_from_weights",
    "positions_from_fills",
]
