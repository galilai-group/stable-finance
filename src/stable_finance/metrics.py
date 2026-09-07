"""Stateless metrics used by forecasts and portfolio backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.data import (
    ForwardReturns,
    PortfolioWeights,
    require_aligned,
)


@dataclass(frozen=True)
class InformationCoefficient:
    """Mean cross-sectional rank IC and its decision-level standard error."""

    mean: float
    standard_error: float
    observations: int


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average-tie ranks without requiring scipy."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = starts.cumsum()
    bounds = np.r_[np.flatnonzero(starts), len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = 0.5 * (bounds[dense] + bounds[dense - 1] + 1)
    return ranks


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation, used on ranks to obtain Spearman correlation."""
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else np.nan


def grouped_rank_ic(predicted: ArrayLike, realized: ArrayLike, *,
                    min_assets: int = 20) -> InformationCoefficient:
    """Average Spearman correlation across decision-time cross-sections.

    Args:
        predicted: ``(n_decisions, n_assets)`` scores or forward returns.
        realized: Array of identical shape containing realized returns.
        min_assets: Minimum finite pairs needed to score a decision.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != realized.shape:
        raise ValueError("predicted and realized must be equally shaped 2-D arrays")
    if min_assets < 2:
        raise ValueError("min_assets must be at least 2")

    values: list[float] = []
    for prediction, outcome in zip(predicted, realized, strict=True):
        valid = np.isfinite(prediction) & np.isfinite(outcome)
        if valid.sum() < min_assets:
            continue
        value = _pearson_correlation(
            _rankdata(prediction[valid]), _rankdata(outcome[valid])
        )
        if np.isfinite(value):
            values.append(value)

    if not values:
        return InformationCoefficient(np.nan, np.nan, 0)
    array = np.asarray(values)
    standard_error = (
        float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else np.nan
    )
    return InformationCoefficient(float(array.mean()), standard_error, len(array))


def grouped_rank_ic_by_label(
    predicted: ArrayLike,
    realized: ArrayLike,
    groups: ArrayLike,
    *,
    min_assets: int = 20,
) -> InformationCoefficient:
    """Average Spearman correlation over a flat, explicitly grouped panel.

    This is the injectable counterpart to :func:`grouped_rank_ic`: callers
    that assemble sparse or cached panels need not first densify them into a
    decision-by-asset rectangle.
    """
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    realized = np.asarray(realized, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    if predicted.shape != realized.shape or predicted.shape != groups.shape:
        raise ValueError("predicted, realized, and groups must have equal length")
    if min_assets < 2:
        raise ValueError("min_assets must be at least 2")
    valid = np.isfinite(predicted) & np.isfinite(realized)
    predicted, realized, groups = predicted[valid], realized[valid], groups[valid]
    if len(predicted) == 0:
        return InformationCoefficient(np.nan, np.nan, 0)
    order = np.argsort(groups, kind="stable")
    predicted, realized, groups = predicted[order], realized[order], groups[order]
    bounds = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])
    values = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        if high - low < min_assets:
            continue
        value = _pearson_correlation(
            _rankdata(predicted[low:high]), _rankdata(realized[low:high])
        )
        if np.isfinite(value):
            values.append(value)
    if not values:
        return InformationCoefficient(np.nan, np.nan, 0)
    array = np.asarray(values)
    standard_error = (
        float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else np.nan
    )
    return InformationCoefficient(float(array.mean()), standard_error, len(array))


def pooled_estimates(values: ArrayLike) -> InformationCoefficient:
    """Pool estimates with each supplied value as one independent unit."""
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return InformationCoefficient(np.nan, np.nan, 0)
    standard_error = (
        float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else np.nan
    )
    return InformationCoefficient(float(array.mean()), standard_error, len(array))


def paired_difference(
    estimates: ArrayLike, baselines: ArrayLike
) -> InformationCoefficient:
    """Pool paired estimate-minus-baseline differences."""
    estimates = np.asarray(estimates, dtype=np.float64)
    baselines = np.asarray(baselines, dtype=np.float64)
    if estimates.shape != baselines.shape:
        raise ValueError("estimates and baselines must have matching shapes")
    return pooled_estimates(estimates - baselines)


def evaluate_forward_returns(
    predicted: ForwardReturns,
    realized: ForwardReturns,
    *,
    min_assets: int = 20,
) -> tuple[InformationCoefficient, ...]:
    """Compute one cross-sectional rank IC result per forecast horizon."""
    require_aligned(predicted, realized)
    return tuple(
        grouped_rank_ic(
            predicted.values[:, :, horizon],
            realized.values[:, :, horizon],
            min_assets=min_assets,
        )
        for horizon in range(len(predicted.horizons))
    )


def sharpe_ratio(returns: ArrayLike, *, periods_per_year: float = 252.0) -> float:
    """Annualized arithmetic Sharpe ratio of a periodic return series."""
    values = np.asarray(returns, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    volatility = values.std(ddof=1)
    if volatility <= 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * values.mean() / volatility)


def _portfolio_returns_at_mid(
    weights: PortfolioWeights,
    realized_mid_returns: ForwardReturns,
) -> tuple[np.ndarray, ...]:
    """Return portfolio P&L marked at the mid price for every horizon."""
    require_aligned(weights, realized_mid_returns)
    results = []
    for horizon in range(len(weights.horizons)):
        weight = weights.values[:, :, horizon]
        outcome = realized_mid_returns.values[:, :, horizon]
        valid = np.isfinite(weight) & np.isfinite(outcome)
        returns = np.where(valid, weight * outcome, 0.0).sum(axis=1)
        returns[~valid.any(axis=1)] = np.nan
        results.append(returns)
    return tuple(results)


def mid_price_sharpe(
    weights: PortfolioWeights,
    realized_mid_returns: ForwardReturns,
    *,
    periods_per_year: float = 252.0,
) -> tuple[float, ...]:
    """Annualized Sharpe at the mid price, independently per horizon.

    ``realized_mid_returns`` must contain mid-to-mid forward returns. This is
    the frictionless result: no spread, impact, or commission is charged.
    """
    return tuple(
        sharpe_ratio(returns, periods_per_year=periods_per_year)
        for returns in _portfolio_returns_at_mid(weights, realized_mid_returns)
    )


def _validated_quotes(
    best_bid: ArrayLike,
    best_ask: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equally shaped quote arrays after validating finite markets."""
    bid = np.asarray(best_bid, dtype=np.float64)
    ask = np.asarray(best_ask, dtype=np.float64)
    if bid.ndim != 2 or bid.shape != ask.shape:
        raise ValueError("best_bid and best_ask must be equally shaped 2-D arrays")
    crossed = np.isfinite(bid) & np.isfinite(ask) & (ask < bid)
    if crossed.any():
        raise ValueError("best_ask cannot be below best_bid")
    return bid, ask


def cross_spread_sharpe(
    weights: PortfolioWeights,
    realized_mid_returns: ForwardReturns,
    best_bid: ArrayLike,
    best_ask: ArrayLike,
    *,
    periods_per_year: float = 252.0,
) -> tuple[float, ...]:
    """Annualized Sharpe after crossing the quoted spread on every trade.

    Bid and ask must have shape ``(n_decisions, n_assets)`` and represent the
    observable quote at each rebalance. A positive weight change buys at the
    best ask; a negative weight change sells at the best bid. The execution
    price is compared with that quote's midpoint and charged against mid-marked
    P&L. The first portfolio is entered from cash. Market impact, quote depth,
    partial fills, and commissions remain outside this metric.
    """
    gross_returns = _portfolio_returns_at_mid(weights, realized_mid_returns)
    expected = weights.values.shape[:2]
    bid, ask = _validated_quotes(best_bid, best_ask)
    if bid.shape != expected:
        raise ValueError(f"best_bid and best_ask must have shape {expected}")

    sharpes = []
    for horizon, gross in enumerate(gross_returns):
        weight = weights.values[:, :, horizon]
        previous = np.vstack([np.zeros((1, expected[1])), weight[:-1]])
        trade = weight - previous
        traded = np.isfinite(trade) & (trade != 0)
        valid_quote = (
            np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
        )
        missing_execution_quote = traded & ~valid_quote

        midpoint = (bid + ask) / 2
        execution_price = np.where(trade > 0, ask, bid)
        # Signed trade makes both directions a positive cost: buys execute
        # above mid and sells (negative trade) execute below mid.
        execution_cost = trade * (execution_price - midpoint) / midpoint
        costs = np.where(traded & valid_quote, execution_cost, 0.0).sum(axis=1)
        net = gross - costs
        net[missing_execution_quote.any(axis=1)] = np.nan
        sharpes.append(sharpe_ratio(net, periods_per_year=periods_per_year))
    return tuple(sharpes)
