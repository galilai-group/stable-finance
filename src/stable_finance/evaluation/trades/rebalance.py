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
than a target. Without factor limits, accelerated proximal gradient descent
solves the convex problem: move toward the target unless the cost exceeds
what it buys. Optional factor limits constrain that same objective through a
bounded dual solve, with feasibility and a primal-dual gap check.

``risk_aversion`` may be one scalar or one value per decision. A per-decision
``lambda_t`` lets a caller calibrate the trade-off from data available at each
decision instead of from the whole panel; each decision's solve then uses its
own ``lambda_t``, step size included, and a scalar is the constant special case.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from stable_finance.dataset.market import MarketData, market_quantity, requires
from stable_finance.evaluation.portfolio.construction import _per_decision_risk_aversion
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



def _factor_limits(loadings, lower, upper, n_assets):
    """Static asset-by-factor loadings and explicit signed exposure bounds."""
    if loadings is None:
        if lower is not None or upper is not None:
            raise ValueError("factor bounds require factor_loadings")
        return None
    raw = np.asarray(loadings)
    if (raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[0] != n_assets
            or raw.shape[1] == 0 or not np.isfinite(raw).all()):
        raise ValueError("factor_loadings must be finite with shape (n_assets, n_factors)")
    bounds = []
    for value, name in ((lower, "factor_lower"), (upper, "factor_upper")):
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or array.shape not in ((), (raw.shape[1],)):
            raise ValueError(f"{name} must be a scalar or one value per factor")
        bounds.append(np.broadcast_to(np.asarray(array, dtype=float), (raw.shape[1],)))
    lo, hi = bounds
    if (np.isnan(lo).any() or np.isnan(hi).any() or np.isposinf(lo).any()
            or np.isneginf(hi).any() or np.any(lo > hi)):
        raise ValueError("factor bounds must be ordered; only outward infinities are allowed")
    return np.asarray(raw, dtype=float), lo, hi


def _factor_rebalance(mu, sigma, previous, cost, factors, *, risk_aversion, max_iter, tol):
    """Solve the same cost-aware objective with linear exposure limits.

    The smooth dual uses bounded L1 subgradients and nonnegative factor
    multipliers. Its value gives a lower bound on the primal objective;
    primal feasibility and the dual gap independently verify the solution.
    """
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import linprog, minimize

    loadings, lower, upper = factors
    factor_scale = np.max(np.abs(loadings), axis=0)
    factor_scale = np.where(factor_scale > 0., factor_scale, 1.)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_lower, normalized_upper = lower/factor_scale, upper/factor_scale
    if (np.any(np.isfinite(lower) & ~np.isfinite(normalized_lower))
            or np.any(np.isfinite(upper) & ~np.isfinite(normalized_upper))):
        raise ValueError("factor bounds exceed finite range after unit normalization")
    loadings, lower, upper = loadings/factor_scale, normalized_lower, normalized_upper
    check_tol = 1e-8
    gap_tol = max(1e-10, min(10*tol, 1e-8))
    free = np.isfinite(cost)
    held = previous.copy()

    def violation(weights):
        exposure = loadings.T @ weights
        if not np.isfinite(exposure).all():
            raise RuntimeError("factor exposure exceeds finite range")
        return max(0., float(np.max(lower-exposure)), float(np.max(exposure-upper)))

    if not free.any():
        if violation(held) > check_tol:
            raise ValueError("factor bounds are infeasible with all positions frozen")
        return held
    matrix_scale = max(float(np.abs(sigma).max()), np.finfo(float).tiny)
    if np.max(np.abs(sigma-sigma.T)) > 1e-10*matrix_scale:
        raise ValueError("factor-constrained covariance must be symmetric")
    symmetric = sigma/2+sigma.T/2
    hessian = risk_aversion*symmetric[np.ix_(free, free)]
    linear = risk_aversion*symmetric[np.ix_(free, ~free)] @ previous[~free]-mu[free]
    paid, start = cost[free], previous[free]
    rows = loadings[free].T
    fixed = loadings[~free].T @ previous[~free]
    finite_upper, finite_lower = np.isfinite(upper), np.isfinite(lower)
    matrix = np.vstack([rows[finite_upper], -rows[finite_lower]])
    budget = np.r_[upper[finite_upper]-fixed[finite_upper],
                   fixed[finite_lower]-lower[finite_lower]]
    # A change of factor units must not change the optimized portfolio.
    row_scale = np.max(np.abs(matrix), axis=1)
    row_scale = np.where(row_scale > 0., row_scale, 1.)
    matrix, budget = matrix/row_scale[:, None], budget/row_scale
    if not all(np.isfinite(value).all() for value in (hessian, linear, matrix, budget)):
        raise ValueError("factor-constrained inputs exceed finite numerical range")
    feasible = linprog(np.zeros(len(start)), A_ub=matrix if len(matrix) else None,
        b_ub=budget if len(matrix) else None, bounds=[(None, None)]*len(start),
        method="highs", options=dict(primal_feasibility_tolerance=1e-9,
                                      dual_feasibility_tolerance=1e-9))
    if feasible.status == 2:
        raise ValueError("factor bounds are infeasible with the frozen positions")
    if not feasible.success:
        raise RuntimeError(f"factor feasibility check failed: {feasible.message}")
    scale = max(float(np.abs(hessian).max()), float(np.abs(linear).max()),
                float(paid.max()), np.finfo(float).tiny)
    hessian, linear, paid = hessian/scale, linear/scale, paid/scale
    try:
        decomposition = cho_factor(hessian, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("factor-constrained covariance must be positive definite on tradable assets") from error
    n = len(start)

    def dual(parameters):
        subgradient, multipliers = parameters[:n], parameters[n:]
        residual = linear+subgradient+matrix.T @ multipliers
        weights = -cho_solve(decomposition, residual, check_finite=False)
        value = .5*residual @ (-weights)+subgradient @ start+multipliers @ budget
        gradient = np.r_[start-weights, budget-matrix @ weights]
        return float(value), gradient

    solved = minimize(dual, np.zeros(n+len(matrix)), jac=True, method="L-BFGS-B",
        bounds=[(-value, value) for value in paid]+[(0., None)]*len(matrix),
        options=dict(maxiter=int(max_iter), ftol=0.,
                     gtol=min(tol, 1e-12), maxls=50))
    parameters = np.asarray(solved.x)
    if parameters.shape != (n+len(matrix),) or not np.isfinite(parameters).all():
        raise RuntimeError("factor-constrained optimizer returned invalid multipliers")
    # A feasible dual point always gives a valid lower bound, including when
    # the optimizer's returned coordinates differ from a box edge by roundoff.
    parameters = np.r_[np.clip(parameters[:n], -paid, paid), np.maximum(parameters[n:], 0.)]
    # L-BFGS can stall when another decrease is below objective roundoff.
    # One quadratic Newton step on interior dual coordinates restores their
    # stationarity; keep it only when the same dual box remains feasible.
    box_lower, box_upper = np.r_[-paid, np.zeros(len(matrix))], np.r_[paid, np.full(len(matrix), np.inf)]
    interior = (parameters > box_lower+1e-10) & (parameters < box_upper-1e-10)
    if interior.any():
        directions = np.column_stack([np.eye(n), matrix.T])[:, interior]
        curvature = directions.T @ cho_solve(decomposition, directions, check_finite=False)
        adjustment = np.linalg.lstsq(curvature, -dual(parameters)[1][interior], rcond=1e-12)[0]
        trial = parameters.copy()
        trial[interior] += adjustment
        if np.all(trial >= box_lower) and np.all(trial <= box_upper):
            parameters = trial
    residual = linear+parameters[:n]+matrix.T @ parameters[n:]
    candidate = -cho_solve(decomposition, residual, check_finite=False)
    held[free] = candidate
    primal = .5*candidate @ hessian @ candidate+linear @ candidate+paid @ np.abs(candidate-start)
    negative_dual, _ = dual(parameters)
    gap = primal+negative_dual
    if (not np.isfinite(gap) or violation(held) > check_tol
            or gap < -1e-8*max(1., abs(primal), abs(negative_dual))
            or gap > gap_tol*max(1., abs(primal), abs(negative_dual))):
        raise RuntimeError(f"factor-constrained optimizer could not be verified: {solved.message}")
    return held


def _track(signal: np.ndarray, covariance: ArrayLike, *, risk_aversion: float | ArrayLike,
           cost, initial, max_iter: int, tol: float, factor_loadings=None,
           factor_lower=None, factor_upper=None) -> np.ndarray:
    """Every decision's rebalance, in order, from ``signal = lambda_t * Sigma @ w*``.

    ``signal`` is ``(n_decisions, n_assets, n_horizons)``: the forecast itself
    for :func:`allocate`, and the target's implied forecast for
    :func:`rebalance`. Decisions are solved in order, each from the weights
    chosen at the previous one. ``risk_aversion`` is a scalar or one
    ``lambda_t`` per decision; the proximal step ``1 / (lambda_t * L)``, with
    ``L`` the covariance's largest eigenvalue, is that decision's own, since
    the gradient's Lipschitz constant scales with ``lambda_t``.
    """
    n_decisions, n_assets, n_horizons = signal.shape
    aversion = _per_decision_risk_aversion(risk_aversion, n_decisions)
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
    factors = _factor_limits(factor_loadings, factor_lower, factor_upper, n_assets)
    if factors is not None:
        if (isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer))
                or max_iter < 1 or not np.isfinite(tol) or tol <= 0):
            raise ValueError("factor-constrained max_iter must be positive and tol finite and positive")
    weights = np.empty_like(signal)
    for horizon in range(n_horizons):
        if factors is None:
            largest = float(np.linalg.eigvalsh(sigma[horizon]).max())
            if largest <= 0:
                raise ValueError("covariance must have positive variance")
        held = previous[:, horizon]
        for decision in range(n_decisions):
            lam = float(aversion[decision])
            if factors is None:
                held = _rebalance(
                    signal[decision, :, horizon], sigma[horizon], held, costs[decision],
                    risk_aversion=lam, step=1.0 / (lam * largest),
                    max_iter=max_iter, tol=tol,
                )
            else:
                try:
                    held = _factor_rebalance(signal[decision, :, horizon], sigma[horizon],
                        held, costs[decision], factors, risk_aversion=lam,
                        max_iter=max_iter, tol=tol)
                except (ValueError, RuntimeError) as error:
                    raise type(error)(f"decision {decision}, horizon {horizon}: {error}") from error
            weights[decision, :, horizon] = held
    return weights


@requires("half_spread")
def rebalance(
    target: PortfolioWeights,
    covariance: ArrayLike,
    *,
    risk_aversion: float | ArrayLike = 1.0,
    cost: ArrayLike | MarketData | None = None,
    initial: ArrayLike | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    factor_loadings: ArrayLike | None = None,
    factor_lower: ArrayLike | None = None,
    factor_upper: ArrayLike | None = None,
) -> PortfolioWeights:
    """The weights actually held, moving toward ``target`` only as far as cost allows.

    ``target`` is the frictionless holding at each decision, such as
    :func:`~stable_finance.evaluation.portfolio.construction.mean_variance_target`.
    ``covariance`` (``(n_horizons, n_assets, n_assets)``) prices the tracking
    error and ``risk_aversion`` its weight against cost: one positive scalar,
    or a 1-D array with one ``lambda_t`` per decision, which should match the
    one the target was built with. ``cost`` broadcasts to
    ``(n_decisions, n_assets)``; NaN marks an asset that cannot be traded at
    that decision and holds its previous weight. ``initial`` broadcasts to
    ``(n_assets, n_horizons)`` and defaults to cash. A missing target is a
    target of zero. ``cost=None`` moves straight to an unconstrained target.
    Optional factor bounds use the same arguments as :func:`allocate`. They
    constrain the modeled weight path, not the execution fills.
    """
    cost = market_quantity(cost, "half_spread", stage=rebalance, like=target)
    sigma = np.asarray(covariance, dtype=np.float64)
    goal = np.nan_to_num(target.values, nan=0.0)
    if sigma.shape[0] != goal.shape[2]:
        raise ValueError("covariance must have one matrix per horizon")
    aversion = _per_decision_risk_aversion(risk_aversion, goal.shape[0])
    signal = aversion[:, None, None] * np.einsum("hij,djh->dih", sigma, goal)
    weights = _track(signal, sigma, risk_aversion=aversion, cost=cost, initial=initial,
                     max_iter=max_iter, tol=tol, factor_loadings=factor_loadings,
                     factor_lower=factor_lower, factor_upper=factor_upper)
    return PortfolioWeights(weights, target.decisions, target.assets, target.horizons)


@requires("half_spread")
def allocate(
    forecast: ForwardReturns,
    covariance: ArrayLike,
    *,
    risk_aversion: float | ArrayLike = 1.0,
    cost: ArrayLike | MarketData | None = None,
    initial: ArrayLike | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    factor_loadings: ArrayLike | None = None,
    factor_lower: ArrayLike | None = None,
    factor_upper: ArrayLike | None = None,
) -> PortfolioWeights:
    """Forecasts to held weights in one solve: the mean-variance target, rebalanced toward.

    Equivalent to ``rebalance(mean_variance_target(forecast, ...), ...)``,
    solved from the forecast directly. Decisions are rebalanced in order, each
    starting from the weights chosen at the previous decision, so the whole
    path reflects the transaction-cost trade-off. ``covariance`` has shape
    ``(n_horizons, n_assets, n_assets)``. ``risk_aversion`` is one positive
    scalar or a 1-D array with one ``lambda_t`` per decision. ``cost`` broadcasts to
    ``(n_decisions, n_assets)``; NaN marks an asset that cannot be traded at
    that decision and holds its previous weight. ``initial`` broadcasts to
    ``(n_assets, n_horizons)`` and defaults to cash. A missing forecast is
    treated as a zero expected return: the asset stays in the problem and is
    traded only as far as risk and cost justify.

    Optional static ``factor_loadings (n_assets, n_factors)`` impose
    ``factor_lower <= factor_loadings.T @ weights <= factor_upper`` at every
    decision and horizon. Both bounds are explicit scalars or factor vectors;
    outward infinities are allowed. Loadings must be finite and known when
    used. Frozen names retain their exposure. Infeasibility raises ValueError;
    a numerical failure raises RuntimeError, with decision/horizon indices.
    This path requires positive definite covariance on tradable assets, uses
    SciPy's bounded dual solve, and verifies feasibility and the dual gap.
    Feasibility tolerance is 1e-8 after dividing each factor column and its
    bounds by that column's largest absolute loading (one for a zero column).
    This fixes tolerance in weight units independently of the factor's units
    and of the optimizer's ``tol``. Omitted factors retain the existing FISTA
    path and defaults. No factor, gross, net or name limits are added implicitly.

    ``cost=None`` allocates frictionlessly. :class:`MarketData` in its place
    charges the quoted half-spread (EFQ 1) and must carry one; scale it
    yourself, or pass ``trading_cost(...)``, for any other execution model.
    """
    cost = market_quantity(cost, "half_spread", stage=allocate, like=forecast)
    signal = np.nan_to_num(forecast.values, nan=0.0)
    weights = _track(signal, covariance, risk_aversion=risk_aversion, cost=cost,
                     initial=initial, max_iter=max_iter, tol=tol,
                     factor_loadings=factor_loadings, factor_lower=factor_lower,
                     factor_upper=factor_upper)
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
        factor_loadings: ArrayLike | None = None,
        factor_lower: ArrayLike | None = None,
        factor_upper: ArrayLike | None = None,
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
            factor_loadings=factor_loadings,
            factor_lower=factor_lower,
            factor_upper=factor_upper,
        )
