"""Mean-variance allocation from return forecasts to portfolio weights.

Given a forecast ``mu`` for every asset, a covariance ``Sigma``, and the
portfolio ``w_prev`` currently held, each rebalance chooses the weights that
maximise

    mu' w - (risk_aversion / 2) w' Sigma w - sum_i cost_i |w_i - w_prev_i|

The last term is a proportional transaction cost: ``cost_i`` is the cost per
unit of weight traded in asset ``i``, for instance the quoted half-spread as a
fraction of mid. It is what makes the optimal portfolio depend on the current
one; without it every rebalance would jump straight to ``Sigma^-1 mu /
risk_aversion``. The objective is convex and is solved by accelerated
proximal gradient descent, so no additional dependency is needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from stable_finance.data import ForwardReturns, PortfolioWeights


def estimate_covariance(
    returns: ForwardReturns,
    *,
    shrinkage: float = 0.1,
    min_samples: int = 2,
) -> NDArray[np.float64]:
    """Per-horizon asset covariance estimated from a realized-return history.

    Decisions are the samples. Each pair of assets is estimated over the
    decisions where both are finite, the result is shrunk toward its diagonal
    by ``shrinkage``, and negative eigenvalues are clipped so the allocation
    objective is convex. The returned array has shape
    ``(n_horizons, n_assets, n_assets)``.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    n_decisions, n_assets, n_horizons = returns.values.shape
    covariances = np.empty((n_horizons, n_assets, n_assets), dtype=np.float64)
    for horizon in range(n_horizons):
        x = returns.values[:, :, horizon]
        finite = np.isfinite(x)
        counts = finite.sum(axis=0)
        if (counts < min_samples).any():
            asset = returns.assets[int(np.argmin(counts))]
            raise ValueError(
                f"asset {asset!r} has {int(counts.min())} finite returns at "
                f"horizon {returns.horizons[horizon]!r}; need {min_samples}"
            )
        mean = np.where(finite, x, 0.0).sum(axis=0) / counts
        centered = np.where(finite, x - mean, 0.0)
        pairs = finite.T.astype(np.float64) @ finite.astype(np.float64)
        sample = (centered.T @ centered) / np.maximum(pairs - 1, 1)
        sample[pairs < 2] = 0.0
        diagonal = np.diag(np.diag(sample))
        shrunk = (1 - shrinkage) * sample + shrinkage * diagonal
        eigenvalues, eigenvectors = np.linalg.eigh((shrunk + shrunk.T) / 2)
        if eigenvalues.max() <= 0:
            raise ValueError(
                f"returns at horizon {returns.horizons[horizon]!r} have no variance"
            )
        # A tiny floor keeps the risk term strictly convex so that assets with
        # (numerically) zero variance cannot receive unbounded weight.
        eigenvalues = np.maximum(eigenvalues, 1e-8 * eigenvalues.max())
        covariances[horizon] = (eigenvectors * eigenvalues) @ eigenvectors.T
    return covariances


def _soft_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _rebalance(
    mu: np.ndarray,
    sigma: np.ndarray,
    previous: np.ndarray,
    cost: np.ndarray,
    *,
    risk_aversion: float,
    step: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Solve one rebalance with FISTA; the cost term is handled by its prox."""
    weights = previous.copy()
    momentum = previous.copy()
    t = 1.0
    for _ in range(max_iter):
        gradient = risk_aversion * (sigma @ momentum) - mu
        proposal = momentum - step * gradient
        # Proximal step for cost * |w - previous|: shrink the trade toward
        # zero. An infinite cost freezes the asset at its previous weight.
        updated = previous + _soft_threshold(proposal - previous, step * cost)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = updated + ((t - 1.0) / t_next) * (updated - weights)
        converged = np.max(np.abs(updated - weights)) < tol
        weights, t = updated, t_next
        if converged:
            break
    return weights


def _broadcast_cost(
    cost: ArrayLike | None, shape: tuple[int, int]
) -> np.ndarray:
    if cost is None:
        return np.zeros(shape, dtype=np.float64)
    array = np.asarray(cost, dtype=np.float64)
    try:
        array = np.broadcast_to(array, shape).astype(np.float64)
    except ValueError as error:
        raise ValueError(
            f"cost must broadcast to {shape}, got shape {array.shape}"
        ) from error
    if np.any(np.isfinite(array) & (array < 0)):
        raise ValueError("cost cannot be negative")
    # A missing cost means the asset cannot be priced, so it cannot be traded.
    return np.where(np.isfinite(array), array, np.inf)


def _broadcast_initial(
    initial: ArrayLike | None, shape: tuple[int, int]
) -> np.ndarray:
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
        raise ValueError("initial weights must be finite")
    return array


def allocate(
    forecast: ForwardReturns,
    covariance: ArrayLike,
    *,
    risk_aversion: float = 1.0,
    cost: ArrayLike | None = None,
    initial: ArrayLike | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> PortfolioWeights:
    """Turn forecasts into weights given a covariance for every horizon.

    Decisions are rebalanced in order, each starting from the weights chosen
    at the previous decision, so the whole path reflects the transaction-cost
    trade-off. ``covariance`` has shape ``(n_horizons, n_assets, n_assets)``.
    ``cost`` broadcasts to ``(n_decisions, n_assets)``; NaN marks an asset that
    cannot be traded at that decision and holds its previous weight.
    ``initial`` broadcasts to ``(n_assets, n_horizons)`` and defaults to cash.
    A missing forecast is treated as a zero expected return: the asset stays
    in the problem and is traded only as far as risk and cost justify.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    n_decisions, n_assets, n_horizons = forecast.values.shape
    sigma = np.asarray(covariance, dtype=np.float64)
    if sigma.shape != (n_horizons, n_assets, n_assets):
        raise ValueError(
            f"covariance must have shape {(n_horizons, n_assets, n_assets)}, "
            f"got {sigma.shape}"
        )
    if not np.isfinite(sigma).all():
        raise ValueError("covariance must be finite")
    costs = _broadcast_cost(cost, (n_decisions, n_assets))
    previous = _broadcast_initial(initial, (n_assets, n_horizons))

    weights = np.empty_like(forecast.values)
    for horizon in range(n_horizons):
        largest = float(np.linalg.eigvalsh(sigma[horizon]).max())
        if largest <= 0:
            raise ValueError("covariance must have positive variance")
        step = 1.0 / (risk_aversion * largest)
        held = previous[:, horizon]
        for decision in range(n_decisions):
            mu = np.nan_to_num(forecast.values[decision, :, horizon], nan=0.0)
            held = _rebalance(
                mu, sigma[horizon], held, costs[decision],
                risk_aversion=risk_aversion, step=step,
                max_iter=max_iter, tol=tol,
            )
            weights[decision, :, horizon] = held
    return PortfolioWeights(
        weights, forecast.decisions, forecast.assets, forecast.horizons
    )


class MeanVarianceAllocator(BaseEstimator):
    """Forecast-to-weights adapter with a proportional transaction cost.

    ``fit`` estimates one asset covariance per horizon from a realized-return
    history; ``predict`` solves the cost-aware mean-variance problem at every
    decision of a forecast panel, starting from the current portfolio. Bring
    your own covariance through :func:`allocate` to skip fitting.
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        *,
        shrinkage: float = 0.1,
        min_samples: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.shrinkage = shrinkage
        self.min_samples = min_samples
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, returns: ForwardReturns, y: None = None) -> "MeanVarianceAllocator":
        """Estimate the per-horizon covariance from realized returns."""
        self.covariance_ = estimate_covariance(
            returns, shrinkage=self.shrinkage, min_samples=self.min_samples
        )
        self.assets_ = returns.assets.copy()
        self.horizons_ = returns.horizons.copy()
        return self

    def predict(
        self,
        forecast: ForwardReturns,
        *,
        cost: ArrayLike | None = None,
        initial: ArrayLike | None = None,
    ) -> PortfolioWeights:
        """Allocate every decision of ``forecast``; see :func:`allocate`."""
        if not hasattr(self, "covariance_"):
            raise NotFittedError("fit the MeanVarianceAllocator before calling predict")
        if not np.array_equal(forecast.assets, self.assets_):
            raise ValueError("assets are not aligned with the fitted covariance")
        if not np.array_equal(forecast.horizons, self.horizons_):
            raise ValueError("horizons are not aligned with the fitted covariance")
        return allocate(
            forecast,
            self.covariance_,
            risk_aversion=self.risk_aversion,
            cost=cost,
            initial=initial,
            max_iter=self.max_iter,
            tol=self.tol,
        )
