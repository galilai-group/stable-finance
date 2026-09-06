"""Evaluate portfolio weights using observable bid/ask trading costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.data import ForwardReturns, PortfolioWeights, require_aligned
from stable_finance.metrics import sharpe_ratio


@dataclass(frozen=True)
class BacktestResult:
    """One periodic return series and its summary statistics."""

    gross_returns: NDArray[np.float64]
    costs: NDArray[np.float64]
    net_returns: NDArray[np.float64]
    sharpe: float


def evaluate_weights(
    weights: PortfolioWeights,
    realized: ForwardReturns,
    *,
    half_spread: ArrayLike | None = None,
    periods_per_year: float = 252.0,
) -> tuple[BacktestResult, ...]:
    """Evaluate weights independently at every horizon.

    ``half_spread`` is the entry half-spread divided by mid price and must have
    shape ``(n_decisions, n_assets)``. Turnover is ``abs(w_t - w_{t-1})``;
    the first portfolio is traded from cash. This models crossing the quoted
    spread and deliberately excludes market impact and commissions.
    """
    require_aligned(weights, realized)
    n_decisions, n_assets, n_horizons = weights.values.shape
    if half_spread is None:
        spread = np.zeros((n_decisions, n_assets), dtype=np.float64)
    else:
        spread = np.asarray(half_spread, dtype=np.float64)
        if spread.shape != (n_decisions, n_assets):
            raise ValueError(
                "half_spread must have shape "
                f"{(n_decisions, n_assets)}, got {spread.shape}"
            )
        if np.any(np.isfinite(spread) & (spread < 0)):
            raise ValueError("half_spread cannot be negative")

    results = []
    for horizon in range(n_horizons):
        weight = weights.values[:, :, horizon]
        outcome = realized.values[:, :, horizon]
        valid = np.isfinite(weight) & np.isfinite(outcome)
        gross = np.where(valid, weight * outcome, 0.0).sum(axis=1)
        gross[~valid.any(axis=1)] = np.nan

        previous = np.vstack([np.zeros((1, n_assets)), weight[:-1]])
        turnover = np.abs(weight - previous)
        costs = np.where(np.isfinite(turnover) & np.isfinite(spread),
                         turnover * spread, 0.0).sum(axis=1)
        costs[~np.isfinite(weight).any(axis=1)] = np.nan
        net = gross - costs
        results.append(BacktestResult(
            gross_returns=gross,
            costs=costs,
            net_returns=net,
            sharpe=sharpe_ratio(net, periods_per_year=periods_per_year),
        ))
    return tuple(results)

