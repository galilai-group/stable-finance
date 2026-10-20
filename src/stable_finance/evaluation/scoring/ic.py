"""How good a return forecast is: the cross-sectional rank information coefficient (IC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.evaluation._grid import _rankdata, require_aligned
from stable_finance.evaluation.market_data import requires
from stable_finance.evaluation.returns.contract import ForwardReturns


@dataclass(frozen=True)
class InformationCoefficient:
    """Mean cross-sectional rank IC and its decision-level standard error."""

    mean: float
    standard_error: float
    observations: int


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


@requires()
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
