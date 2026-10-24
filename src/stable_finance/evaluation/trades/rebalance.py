"""How far to move from the portfolio held toward the target, given what trading costs.

The portfolio stage solves for the holding it would want with no frictions:
for mean-variance, ``w* = Sigma^-1 mu / risk_aversion``. Trading there costs
money, so this stage decides how much of the move is worth making. Each
rebalance chooses the weights that minimise

    (risk_aversion / 2) (w - w*)' Sigma (w - w*) + sum_i cost_i |w_i - w_prev_i|

-- tracking error against the target, under the target's own risk model,
against the cost of the trade from ``w_prev``. ``cost_i`` is the cost per
unit of weight traded, for instance the quoted half-spread as a fraction of
mid. Without it every rebalance jumps straight to the target.

This is the cost-aware mean-variance problem, not an approximation of it:

    mu' w - (risk_aversion / 2) w' Sigma w
        = -(risk_aversion / 2) (w - w*)' Sigma (w - w*) + const

so :func:`rebalance` toward the mean-variance target gives the weights
:func:`allocate` gets by solving the whole objective at once. ``allocate`` and
:class:`MeanVarianceAllocator` remain for callers who hold a forecast rather
than a target. The problem is convex and is solved by accelerated proximal
gradient descent, whose proximal step is the trade rule itself: move toward
the target unless the cost of the move exceeds what it buys.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from stable_finance.dataset.market import MarketData, market_quantity, requires
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.risk import estimate_covariance
from stable_finance.evaluation.returns.contract import ForwardReturns


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


def _track(signal: np.ndarray, covariance: ArrayLike, *, risk_aversion: float, cost,
           initial, max_iter: int, tol: float) -> np.ndarray:
    """Every decision's rebalance, in order, from ``signal = risk_aversion * Sigma @ w*``.

    ``signal`` is ``(n_decisions, n_assets, n_horizons)``: the forecast itself
    for :func:`allocate`, and the target's implied forecast for
    :func:`rebalance`. Decisions are solved in order, each from the weights
    chosen at the previous one.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    n_decisions, n_assets, n_horizons = signal.shape
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
    weights = np.empty_like(signal)
    for horizon in range(n_horizons):
        largest = float(np.linalg.eigvalsh(sigma[horizon]).max())
        if largest <= 0:
            raise ValueError("covariance must have positive variance")
        step = 1.0 / (risk_aversion * largest)
        held = previous[:, horizon]
        for decision in range(n_decisions):
            held = _rebalance(
                signal[decision, :, horizon], sigma[horizon], held, costs[decision],
                risk_aversion=risk_aversion, step=step, max_iter=max_iter, tol=tol,
            )
            weights[decision, :, horizon] = held
    return weights


@requires("half_spread")
def rebalance(
    target: PortfolioWeights,
    covariance: ArrayLike,
    *,
    risk_aversion: float = 1.0,
    cost: ArrayLike | MarketData | None = None,
    initial: ArrayLike | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> PortfolioWeights:
    """The weights actually held, moving toward ``target`` only as far as cost allows.

    ``target`` is the frictionless holding at each decision, such as
    :func:`~stable_finance.evaluation.portfolio.construction.mean_variance_target`.
    ``covariance`` (``(n_horizons, n_assets, n_assets)``) prices the tracking
    error and ``risk_aversion`` its weight against cost. ``cost`` broadcasts to
    ``(n_decisions, n_assets)``; NaN marks an asset that cannot be traded at
    that decision and holds its previous weight. ``initial`` broadcasts to
    ``(n_assets, n_horizons)`` and defaults to cash. A missing target is a
    target of zero. ``cost=None`` moves straight to the target.
    """
    cost = market_quantity(cost, "half_spread", stage=rebalance, like=target)
    sigma = np.asarray(covariance, dtype=np.float64)
    goal = np.nan_to_num(target.values, nan=0.0)
    if sigma.shape[0] != goal.shape[2]:
        raise ValueError("covariance must have one matrix per horizon")
    signal = risk_aversion * np.einsum("hij,djh->dih", sigma, goal)
    weights = _track(signal, sigma, risk_aversion=risk_aversion, cost=cost, initial=initial,
                     max_iter=max_iter, tol=tol)
    return PortfolioWeights(weights, target.decisions, target.assets, target.horizons)


@requires("half_spread")
def allocate(
    forecast: ForwardReturns,
    covariance: ArrayLike,
    *,
    risk_aversion: float = 1.0,
    cost: ArrayLike | MarketData | None = None,
    initial: ArrayLike | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> PortfolioWeights:
    """Forecasts to held weights in one solve: the mean-variance target, rebalanced toward.

    Equivalent to ``rebalance(mean_variance_target(forecast, ...), ...)``,
    solved from the forecast directly. Decisions are rebalanced in order, each
    starting from the weights chosen at the previous decision, so the whole
    path reflects the transaction-cost trade-off. ``covariance`` has shape
    ``(n_horizons, n_assets, n_assets)``. ``cost`` broadcasts to
    ``(n_decisions, n_assets)``; NaN marks an asset that cannot be traded at
    that decision and holds its previous weight. ``initial`` broadcasts to
    ``(n_assets, n_horizons)`` and defaults to cash. A missing forecast is
    treated as a zero expected return: the asset stays in the problem and is
    traded only as far as risk and cost justify.

    ``cost=None`` allocates frictionlessly. :class:`MarketData` in its place
    charges the quoted half-spread (EFQ 1) and must carry one; scale it
    yourself, or pass ``trading_cost(...)``, for any other execution model.
    """
    cost = market_quantity(cost, "half_spread", stage=allocate, like=forecast)
    signal = np.nan_to_num(forecast.values, nan=0.0)
    weights = _track(signal, covariance, risk_aversion=risk_aversion, cost=cost,
                     initial=initial, max_iter=max_iter, tol=tol)
    return PortfolioWeights(weights, forecast.decisions, forecast.assets, forecast.horizons)


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
        cost: ArrayLike | MarketData | None = None,
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
