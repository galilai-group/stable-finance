"""How good a return forecast is: the cross-sectional rank information coefficient (IC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.evaluation._grid import _rankdata, require_aligned
from stable_finance.dataset.market import requires
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.scoring.stats import _bartlett_long_run_variance


@dataclass(frozen=True)
class InformationCoefficient:
    """Mean cross-sectional rank IC and its decision-level standard error.

    The standard error treats the decisions' ICs as independent unless it was
    asked to allow for their overlap (``lags`` in :func:`grouped_rank_ic`):
    forward returns longer than the decision spacing share their later
    stretches, so neighbouring ICs are correlated and the iid figure is too
    small.
    """

    mean: float
    standard_error: float
    observations: int


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation, used on ranks to obtain Spearman correlation."""
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else np.nan


def _newey_west_standard_error(values: np.ndarray, rows: np.ndarray, n_rows: int,
                               lags: int) -> float:
    """Standard error of the mean of a series observed at ``rows`` of ``n_rows``.

    Newey--West (Bartlett kernel) with lags counted in rows, so two values
    are ``l`` apart when their rows are, whatever was skipped between them:
    the series is demeaned and zero-filled where it was not observed, which
    drops exactly the pairs with a missing side. Scaled by ``1 / (n - 1)``,
    so ``lags=0`` is the iid ``std(ddof=1) / sqrt(n)``.
    """
    n = len(values)
    if n < 2:
        return np.nan
    series = np.zeros(n_rows)
    series[rows] = values - values.mean()
    long_run = _bartlett_long_run_variance(series, lags) * n_rows / (n - 1)
    return float(np.sqrt(max(long_run, 0.0) / n))


def grouped_rank_ic(predicted: ArrayLike, realized: ArrayLike, *,
                    min_assets: int = 20, lags: int | None = None) -> InformationCoefficient:
    """Average Spearman correlation across decision-time cross-sections.

    Args:
        predicted: ``(n_decisions, n_assets)`` scores or forward returns.
        realized: Array of identical shape containing realized returns.
        min_assets: Minimum finite pairs needed to score a decision.
        lags: Newey--West lags for the standard error, in decisions. None
            (default) treats the decisions' ICs as independent. A forward
            return spanning ``k`` decisions overlaps the next ``k - 1``
            decisions' returns, so their ICs are correlated out to lag
            ``k - 1`` and ``lags`` should be at least that (the Bartlett
            weights shrink the longer lags, so a little more is safer).
            Lags count rows of the panel: a decision too sparse to score
            leaves a gap rather than closing up. ``lags=0`` is the iid
            figure. The mean is the same either way.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != realized.shape:
        raise ValueError("predicted and realized must be equally shaped 2-D arrays")
    if min_assets < 2:
        raise ValueError("min_assets must be at least 2")
    if lags is not None and (isinstance(lags, bool) or int(lags) != lags or lags < 0):
        raise ValueError("lags must be a non-negative integer, or None")

    values: list[float] = []
    rows: list[int] = []
    for row, (prediction, outcome) in enumerate(zip(predicted, realized, strict=True)):
        valid = np.isfinite(prediction) & np.isfinite(outcome)
        if valid.sum() < min_assets:
            continue
        value = _pearson_correlation(
            _rankdata(prediction[valid]), _rankdata(outcome[valid])
        )
        if np.isfinite(value):
            values.append(value)
            rows.append(row)

    if not values:
        return InformationCoefficient(np.nan, np.nan, 0)
    array = np.asarray(values)
    if lags is not None:
        standard_error = _newey_west_standard_error(array, np.asarray(rows),
                                                    len(predicted), int(lags))
    else:
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
