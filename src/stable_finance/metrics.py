"""Stateless metrics used by forecasts and portfolio backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.data import ForwardReturns, require_aligned


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


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
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
        value = _correlation(_rankdata(prediction[valid]), _rankdata(outcome[valid]))
        if np.isfinite(value):
            values.append(value)

    if not values:
        return InformationCoefficient(np.nan, np.nan, 0)
    array = np.asarray(values)
    standard_error = (
        float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else np.nan
    )
    return InformationCoefficient(float(array.mean()), standard_error, len(array))


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
