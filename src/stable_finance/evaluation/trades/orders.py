"""Stage 4, weights to trades: orders, the realized position path, and its score.

Weights say what the portfolio should hold; orders say what must be traded to
get there; fills say what actually traded and at which price. Keeping the
three apart lets the realized position path diverge from the target whenever
an order cannot be executed, which is exactly what a Sharpe marked on target
weights hides. How an order fills -- at the touch, or at the return's own
benchmark -- is the execution model's rule
(:func:`stable_finance.evaluation.trades.fills.simulate_fills`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.trades.contract import Orders


def _initial_positions(initial: ArrayLike | None, shape: tuple[int, int]) -> np.ndarray:
    if initial is None:
        return np.zeros(shape, dtype=np.float64)
    array = np.asarray(initial, dtype=np.float64)
    try:
        array = np.broadcast_to(array, shape).astype(np.float64)
    except ValueError as error:
        raise ValueError(
            f"initial must broadcast to {shape}, got shape {array.shape}"
        ) from error
    if not np.isfinite(array).all():
        raise ValueError("initial positions must be finite")
    return array


def orders_from_weights(
    weights: PortfolioWeights, *, initial: ArrayLike | None = None
) -> Orders:
    """Orders that move the portfolio along the target weight path.

    Each decision's order is the target weight minus the previous decision's
    target weight; the first decision trades from ``initial`` (default cash),
    which broadcasts to ``(n_assets, n_horizons)``. A NaN target is treated as
    zero exposure. Orders are generated from targets, not fills, so a missed
    fill is not retried at the next decision; feed realized positions back
    through ``initial`` to do that.
    """
    target = np.nan_to_num(weights.values, nan=0.0)
    n_decisions, n_assets, n_horizons = target.shape
    start = _initial_positions(initial, (n_assets, n_horizons))
    previous = np.concatenate([start[None], target[:-1]], axis=0)
    return Orders(
        target - previous, weights.decisions, weights.assets, weights.horizons
    )
