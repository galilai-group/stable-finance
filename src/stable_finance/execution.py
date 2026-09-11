"""Orders, fills, and a bid/ask execution simulator.

Weights say what the portfolio should hold; orders say what must be traded to
get there; fills say what actually traded and at which price. Keeping the
three apart lets the realized position path diverge from the target whenever
an order cannot be executed, which is exactly what a Sharpe marked on target
weights hides.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.data import (
    Fills,
    ForwardReturns,
    Orders,
    PortfolioWeights,
    require_aligned,
)
from stable_finance.metrics import _validated_quotes, sharpe_ratio
from stable_finance.portfolio import BacktestResult


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


def simulate_fills(orders: Orders, best_bid: ArrayLike, best_ask: ArrayLike) -> Fills:
    """Fill every order in full at the touch, or not at all.

    Buys execute at the best ask and sells at the best bid. Quotes have shape
    ``(n_decisions, n_assets)`` and apply to every horizon. An order whose
    quote is missing or non-positive is left unfilled. Depth, partial fills,
    market impact, and commissions are outside this simulator.
    """
    bid, ask = _validated_quotes(best_bid, best_ask)
    expected = orders.values.shape[:2]
    if bid.shape != expected:
        raise ValueError(f"best_bid and best_ask must have shape {expected}")
    quoted = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
    requested = np.nan_to_num(orders.values, nan=0.0)
    fillable = (requested != 0) & quoted[:, :, None]
    quantity = np.where(fillable, requested, 0.0)
    touch = np.where(requested > 0, ask[:, :, None], bid[:, :, None])
    price = np.where(fillable, touch, np.nan)
    mid = np.where(fillable, ((bid + ask) / 2)[:, :, None], np.nan)
    return Fills(quantity, price, mid, orders.decisions, orders.assets, orders.horizons)


def positions_from_fills(
    fills: Fills, *, initial: ArrayLike | None = None
) -> PortfolioWeights:
    """Weights actually held after each decision's fills."""
    n_decisions, n_assets, n_horizons = fills.quantity.shape
    start = _initial_positions(initial, (n_assets, n_horizons))
    positions = start[None] + np.cumsum(fills.quantity, axis=0)
    return PortfolioWeights(positions, fills.decisions, fills.assets, fills.horizons)


def evaluate_fills(
    fills: Fills,
    realized_mid_returns: ForwardReturns,
    *,
    initial: ArrayLike | None = None,
    periods_per_year: float = 252.0,
) -> tuple[BacktestResult, ...]:
    """Mark the realized position path at mid and charge each fill's cost.

    Gross return is the position held after the decision's fills times the
    mid-to-mid forward return. Execution cost is the filled quantity times the
    fill price's distance from mid, so buys above mid and sells below mid both
    cost money. One result is returned per horizon.
    """
    require_aligned(fills, realized_mid_returns)
    positions = positions_from_fills(fills, initial=initial).values
    traded = fills.quantity != 0
    slippage = np.where(
        traded, (fills.price - fills.mid_price) / fills.mid_price, 0.0
    )
    per_asset_cost = fills.quantity * slippage
    results = []
    for horizon in range(len(fills.horizons)):
        position = positions[:, :, horizon]
        outcome = realized_mid_returns.values[:, :, horizon]
        valid = np.isfinite(outcome)
        gross = np.where(valid, position * outcome, 0.0).sum(axis=1)
        gross[~valid.any(axis=1)] = np.nan
        costs = per_asset_cost[:, :, horizon].sum(axis=1)
        net = gross - costs
        results.append(BacktestResult(
            gross_returns=gross,
            costs=costs,
            net_returns=net,
            sharpe=sharpe_ratio(net, periods_per_year=periods_per_year),
        ))
    return tuple(results)
