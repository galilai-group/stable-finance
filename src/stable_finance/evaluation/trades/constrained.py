"""One current-state target-tracking decision with explicit position limits.

The caller supplies one combined target, current actual holdings and pending
orders marked at the same current NAV/prices. This module estimates no signal,
risk or cost and simulates no fills. Costs influence the decision; the execution
ledger charges observed fills separately. A constrained multi-horizon target
does not identify its original alpha objective, so this is explicitly a common-
interval tracking policy rather than an equivalent joint return-risk solve.

Only NumPy is imported at module import time. The optional numerical backend is
loaded when a binding solve is required. This file is also directly loadable in
an isolated execution process without importing another package implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["RebalanceConstraints", "ConstrainedRebalanceResult", "solve_constrained_rebalance"]


def _nonnegative(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative scalar")


@dataclass(frozen=True)
class RebalanceConstraints:
    """Limits on planned committed weights, in current portfolio-equity units."""

    gross_limit: float = 1.0
    max_weight: float = 0.1
    dollar_neutral: bool = True
    max_abs_beta: float | None = None

    def __post_init__(self):
        _nonnegative(self.gross_limit, "gross_limit")
        _nonnegative(self.max_weight, "max_weight")
        if not isinstance(self.dollar_neutral, (bool, np.bool_)):
            raise ValueError("dollar_neutral must be boolean")
        if self.max_abs_beta is not None:
            _nonnegative(self.max_abs_beta, "max_abs_beta")


@dataclass(frozen=True)
class ConstrainedRebalanceResult:
    """Certified plan, or an explicitly infeasible frozen-holdings problem.

    ``tracking_risk`` is ``(weights-target)' covariance (weights-target)``.
    ``objective`` is ``risk_aversion/2 * tracking_risk + estimated_cost``.
    Turnover and estimated cost concern NEW orders relative to actual plus
    pending holdings. They are not filled turnover or realized cash charges.
    Gap is in original objective return units; weights are copied read-only.
    An infeasible result has no weights or numerical objective/certificate.
    """

    status: str
    weights: NDArray[np.float64] | None
    objective: float | None
    tracking_risk: float | None
    estimated_cost: float | None
    turnover: float | None
    optimality_gap: float | None
    max_constraint_violation: float | None
    beta_exposure: float | None
    message: str
    factor_exposures: NDArray[np.float64] | None = None
    factor_names: tuple[str, ...] | None = None
    factor_lower_slack: NDArray[np.float64] | None = None
    factor_upper_slack: NDArray[np.float64] | None = None


def _vector(value, n, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (n,):
        raise ValueError(f"{name} must be a finite asset vector of shape {(n,)}")
    array = np.asarray(raw, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite asset vector of shape {(n,)}")
    return array


def _factor_arrays(*args, **kwargs):
    """Load optional factor validation without changing legacy imports."""
    try:
        from stable_finance.evaluation.portfolio.factors import factor_constraint_arrays
    except ModuleNotFoundError as error:
        if error.name is not None and not error.name.startswith("stable_finance"):
            raise
        # The preserved replay runner loads this module under an old namespace.
        # Resolve the adjacent standalone contract only on the new factor path.
        import importlib.util
        from pathlib import Path
        import sys

        path = Path(__file__).resolve().parents[1] / "portfolio" / "factors.py"
        key = "_stable_finance_factor_contract_" + str(abs(hash(str(path))))
        module = sys.modules.get(key)
        if module is None:
            spec = importlib.util.spec_from_file_location(key, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        factor_constraint_arrays = module.factor_constraint_arrays
    return factor_constraint_arrays(*args, **kwargs)


def _readonly(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def solve_constrained_rebalance(
    target: ArrayLike,
    covariance: ArrayLike,
    *,
    previous: ArrayLike,
    pending: ArrayLike | None = None,
    cost: ArrayLike = 0.0,
    risk_aversion: float = 5.0,
    constraints: RebalanceConstraints = RebalanceConstraints(),
    beta: ArrayLike | None = None,
    tradable: ArrayLike | None = None,
    factor_loadings: ArrayLike | None = None,
    factor_lower: ArrayLike | None = None,
    factor_upper: ArrayLike | None = None,
    factor_names: tuple[str, ...] | None = None,
) -> ConstrainedRebalanceResult:
    """Minimize tracking risk plus one-way costs, preserving position limits.

    With ``committed = previous + pending``, solve

        min_w gamma/2 * (w-target)' Sigma (w-target)
              + sum_i cost_i * abs(w_i-committed_i).

    The decision's supplied risk/cost/beta estimates must be known at that
    decision and describe the SAME interval. Pending orders are irrevocable,
    not fills: their committed cost is sunk and only additional orders are
    penalized. ``tradable=False`` freezes a coordinate at its committed value.
    A NaN or positive-infinite cost also freezes that coordinate. Missing cost
    is never treated as a free trade. Negative costs, including minus infinity,
    are rejected. Beta and ``max_abs_beta`` must be supplied together, with
    finite beta for every coordinate including frozen holdings.
    Optional ``factor_loadings`` has shape ``(N,F)``; scalar or ``(F,)``
    ``factor_lower`` and ``factor_upper`` bound ``loadings.T @ weights``.
    Generic factor limits intersect with any supplied legacy beta limit.

    Constraints apply to the resulting committed plan. Unknown arrival prices,
    partial fills and subsequent NAV/price drift can change actual exposure.
    Frozen holdings can make the problem infeasible; then ``status`` is
    ``infeasible`` and no trade is silently invented. Backend failures or failed
    independent certificates raise RuntimeError instead. There is no general
    clipping, projection, gross restoration or silent constraint relaxation.
    """
    raw = np.asarray(target)
    if raw.ndim != 1 or not raw.size:
        raise ValueError("target must be a non-empty finite asset vector")
    n = raw.size
    goal = _vector(target, n, "target")
    actual = _vector(previous, n, "previous")
    outstanding = np.zeros(n) if pending is None else _vector(pending, n, "pending")
    with np.errstate(over="ignore", invalid="ignore"):
        committed = actual + outstanding
    if not np.isfinite(committed).all():
        raise ValueError("previous plus pending must be finite")
    _nonnegative(risk_aversion, "risk_aversion")
    if risk_aversion == 0:
        raise ValueError("risk_aversion must be positive")
    if not isinstance(constraints, RebalanceConstraints):
        raise TypeError("constraints must be RebalanceConstraints")
    raw_sigma = np.asarray(covariance)
    if raw_sigma.dtype.kind not in "iuf" or raw_sigma.shape != (n, n):
        raise ValueError(f"covariance must be finite with shape {(n, n)}")
    sigma = np.asarray(raw_sigma, dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError("covariance must be finite")
    matrix_scale = max(float(np.abs(sigma).max()), np.finfo(float).tiny)
    if np.max(np.abs(sigma-sigma.T)) > 1e-10*matrix_scale:
        raise ValueError("covariance must be symmetric")
    sigma = (sigma+sigma.T)/2
    if np.linalg.eigvalsh(sigma)[0] < -1e-10*matrix_scale:
        raise ValueError("covariance must be positive semidefinite")
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic_full = risk_aversion*sigma
    if not np.isfinite(quadratic_full).all():
        raise ValueError("risk_aversion times covariance exceeds finite range")

    raw_cost = np.asarray(cost)
    if raw_cost.dtype.kind not in "iuf" or raw_cost.shape not in ((), (n,)):
        raise ValueError("cost must be scalar or an asset vector")
    charge = np.broadcast_to(np.asarray(raw_cost, dtype=float), (n,)).copy()
    if np.any(charge < 0):
        raise ValueError("cost must be nonnegative or missing")
    allowed = np.ones(n, dtype=bool)
    if tradable is not None:
        allowed = np.asarray(tradable)
        if allowed.dtype.kind != "b" or allowed.shape != (n,):
            raise ValueError("tradable must be a boolean asset vector")
        allowed = allowed.copy()
    allowed &= np.isfinite(charge)
    charge[~allowed] = 0.0  # A frozen coordinate cannot incur a NEW order cost.
    if (beta is None) != (constraints.max_abs_beta is None):
        raise ValueError("beta and max_abs_beta must be supplied together")
    factors = None if beta is None else _vector(beta, n, "beta")
    generic = lower_factors = upper_factors = names = None
    if any(value is not None for value in
           (factor_loadings, factor_lower, factor_upper, factor_names)):
        generic, lower_factors, upper_factors, names = _factor_arrays(
            factor_loadings, factor_lower, factor_upper, stages=1, n_assets=n,
            factor_names=factor_names, required=allowed | (committed != 0.))
        generic, lower_factors, upper_factors = generic[0], lower_factors[0], upper_factors[0]
        names = tuple(names.tolist())

    def violation(weights):
        values = [0., np.abs(weights).sum()-constraints.gross_limit,
                  np.abs(weights).max()-constraints.max_weight]
        if constraints.dollar_neutral:
            values.append(abs(weights.sum()))
        if factors is not None:
            values.append(abs(float(factors @ weights))-constraints.max_abs_beta)
        if generic is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                exposure = generic.T @ weights
            if not np.isfinite(exposure).all():
                raise RuntimeError("factor exposure exceeds finite range")
            values.extend((float(np.max(lower_factors-exposure)),
                           float(np.max(exposure-upper_factors))))
        if np.any(~allowed):
            values.append(float(np.abs(weights[~allowed]-committed[~allowed]).max()))
        return max(values)

    def result(weights, gap=0., message="Independent primal and convex-gap certificates passed"):
        weights = np.array(weights, dtype=float, copy=True)
        residual = float(violation(weights))
        if not np.isfinite(residual) or residual > 1e-8:
            raise RuntimeError(f"rebalance violates planned position limits by {residual:.3g}")
        difference = weights-goal
        risk = max(0., float(difference @ sigma @ difference))
        cost_value = float(charge @ np.abs(weights-committed))
        objective = float(risk_aversion/2*risk+cost_value)
        if not np.isfinite([risk, cost_value, objective, gap]).all():
            raise RuntimeError("rebalance objective exceeds finite range")
        weights.setflags(write=False)
        exposures = None if generic is None else generic.T @ weights
        return ConstrainedRebalanceResult("solved", weights, objective, risk,
            cost_value, float(np.abs(weights-committed).sum()), float(gap), residual,
            None if factors is None else float(factors @ weights), message,
            None if generic is None else _readonly(exposures), names,
            None if generic is None else _readonly(exposures-lower_factors),
            None if generic is None else _readonly(upper_factors-exposures))

    def infeasible(message):
        return ConstrainedRebalanceResult("infeasible", None, None, None, None,
            None, None, None, None, message)

    # Exact target matching is a global optimum for zero-cost PSD tracking.
    if (not np.any(charge) and violation(goal) <= 1e-12
            and np.array_equal(goal[~allowed], committed[~allowed])):
        return result(goal, message="Exact feasible zero-cost target")
    fixed = np.where(allowed, 0., committed)
    fixed_gross = float(np.abs(fixed).sum())
    if (fixed_gross > constraints.gross_limit+1e-10
            or np.any(np.abs(fixed) > constraints.max_weight+1e-10)):
        return infeasible("Frozen committed holdings exceed gross or per-name limits")
    members = np.flatnonzero(allowed)
    m = len(members)
    if not m:
        if violation(committed) > 1e-8:
            return infeasible("Frozen committed holdings cannot satisfy exposure constraints")
        return result(committed, message="All positions frozen; committed holdings are feasible")

    from scipy.optimize import linprog

    identity, zeros = np.eye(m), np.zeros((m, m))
    # Frozen coordinates are ELIMINATED, so no post-solve repair can trade them.
    a_ub = np.vstack([
        np.hstack([identity, -identity, zeros]),
        np.hstack([-identity, -identity, zeros]),
        np.hstack([np.zeros((1, m)), np.ones((1, m)), np.zeros((1, m))]),
        np.hstack([identity, zeros, -identity]),
        np.hstack([-identity, zeros, -identity]),
    ])
    b_ub = np.r_[np.zeros(2*m), constraints.gross_limit-fixed_gross,
                 committed[members], -committed[members]]
    if factors is not None:
        beta_row = np.r_[factors[members], np.zeros(2*m)][None]
        fixed_beta = float(factors @ fixed)
        a_ub = np.vstack([a_ub, beta_row, -beta_row])
        b_ub = np.r_[b_ub, constraints.max_abs_beta-fixed_beta,
                     constraints.max_abs_beta+fixed_beta]
    if generic is not None:
        rows = np.hstack([generic[members].T, np.zeros((generic.shape[1], 2*m))])
        frozen_exposure = generic.T @ fixed
        finite_upper = np.isfinite(upper_factors)
        finite_lower = np.isfinite(lower_factors)
        a_ub = np.vstack([a_ub, rows[finite_upper], -rows[finite_lower]])
        b_ub = np.r_[b_ub, upper_factors[finite_upper]-frozen_exposure[finite_upper],
                     frozen_exposure[finite_lower]-lower_factors[finite_lower]]
    a_eq = np.r_[np.ones(m), np.zeros(2*m)][None] if constraints.dollar_neutral else None
    b_eq = np.asarray([-fixed.sum()]) if a_eq is not None else None
    lower = np.r_[np.full(m, -constraints.max_weight), np.zeros(2*m)]
    upper = np.r_[np.full(2*m, constraints.max_weight),
                  constraints.max_weight+np.abs(committed[members])]
    if not np.isfinite(upper).all():
        raise ValueError("position/turnover bounds exceed finite range")
    bounds = list(zip(lower, upper))
    lp_options = dict(dual_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9)
    feasible = linprog(np.zeros(3*m), A_ub=a_ub, b_ub=b_ub,
        A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs", options=lp_options)
    if feasible.status == 2:
        return infeasible("No plan satisfies all position limits and frozen committed holdings")
    if not feasible.success or not np.isfinite(feasible.x).all():
        raise RuntimeError(f"rebalance feasibility check failed: {feasible.message}")

    with np.errstate(over="ignore", invalid="ignore"):
        linear = quadratic_full[members] @ (fixed-goal)
    quadratic = quadratic_full[np.ix_(members, members)]
    scale = max(float(np.abs(linear).max()), float(np.abs(quadratic).max()),
                float(charge[members].max()))
    if not np.isfinite(scale):
        raise ValueError("tracking objective exceeds finite range")
    if scale == 0:
        full = fixed.copy()
        full[members] = feasible.x[:m]
        return result(full, message="Feasible plan for a constant tracking objective")
    quadratic, linear, costs = quadratic/scale, linear/scale, charge[members]/scale
    q = np.r_[linear, np.zeros(m), costs]

    def objective(x):
        return float(.5*x[:m] @ quadratic @ x[:m]+q @ x)

    def augmented(weights):
        free = weights[members]
        return np.r_[free, np.abs(free), np.abs(free-committed[members])]

    def certify(candidate):
        if np.shape(candidate) != (3*m,) or not np.isfinite(candidate).all():
            raise RuntimeError("rebalance optimizer returned an invalid candidate")
        weights = fixed.copy()
        weights[members] = candidate[:m]
        x = augmented(weights)
        residuals = np.r_[a_ub @ x-b_ub, lower-x, x-upper,
                          np.abs(a_eq @ x-b_eq) if a_eq is not None else [0.]]
        residual = float(np.max(residuals))
        if not np.isfinite(residual) or residual > 1e-8:
            raise RuntimeError(f"rebalance optimizer violates constraints by {residual:.3g}")
        gradient = q.copy()
        gradient[:m] += quadratic @ x[:m]
        certificate = linprog(gradient, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq,
            b_eq=b_eq, bounds=bounds, method="highs", options=lp_options)
        if not certificate.success or not np.isfinite(certificate.fun):
            raise RuntimeError(f"rebalance optimality certificate failed: {certificate.message}")
        gap = max(0., float(gradient @ x-certificate.fun))
        if gap > 1e-7*max(1., abs(objective(x))):
            raise RuntimeError(f"rebalance optimizer did not converge: optimality gap {gap:.3g}")
        return result(weights, gap*scale)

    # A sufficiently expensive new trade is exactly zero. The independent
    # epigraph LP verifies that claim rather than using a guessed threshold.
    if violation(committed) <= 1e-10:
        try:
            return certify(augmented(committed))
        except RuntimeError:
            pass
    if not np.any(quadratic):
        solved = linprog(q, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=bounds, method="highs", options=lp_options)
        if not solved.success:
            raise RuntimeError(f"rebalance linear optimization failed: {solved.message}")
        return certify(solved.x)

    try:
        from osqp import OSQP, OSQPException
        from scipy.sparse import block_diag, csc_matrix, eye, triu, vstack
    except ImportError as error:
        raise ImportError("Constrained tracking requires stable-finance[joint].") from error
    hessian = block_diag([csc_matrix(quadratic), csc_matrix((2*m, 2*m))], format="csc")
    matrices, l, u = [csc_matrix(a_ub)], [np.full(len(b_ub), -np.inf)], [b_ub]
    if a_eq is not None:
        matrices.append(csc_matrix(a_eq)); l.append(b_eq); u.append(b_eq)
    matrices.append(eye(3*m, format="csc")); l.append(lower); u.append(upper)

    def fallback(candidate, primary_error):
        from scipy.optimize import minimize

        initial = candidate if np.shape(candidate) == (3*m,) and np.isfinite(candidate).all() else feasible.x
        conditions = [dict(type="ineq", fun=lambda x: b_ub-a_ub @ x, jac=lambda x: -a_ub)]
        if a_eq is not None:
            conditions.append(dict(type="eq", fun=lambda x: a_eq @ x-b_eq, jac=lambda x: a_eq))
        solved_again = minimize(objective, initial,
            jac=lambda x: q+np.r_[quadratic @ x[:m], np.zeros(2*m)],
            method="SLSQP", bounds=bounds, constraints=conditions,
            options=dict(maxiter=2000, ftol=1e-12, disp=False))
        try:
            return certify(solved_again.x)
        except (ValueError, RuntimeError, TypeError) as error:
            raise RuntimeError(f"Constrained tracking could not be certified: primary={primary_error}; "
                f"fallback={solved_again.message}; certificate={error}") from error

    try:
        solver = OSQP()
        solver.setup(P=triu(hessian, format="csc"), q=q,
            A=vstack(matrices, format="csc"), l=np.concatenate(l), u=np.concatenate(u),
            verbose=False, eps_abs=1e-10, eps_rel=1e-10,
            eps_prim_inf=1e-10, eps_dual_inf=1e-10, max_iter=100000,
            polishing=True, polish_refine_iter=10, adaptive_rho=True,
            adaptive_rho_interval=50, check_termination=25, warm_starting=True)
        solver.warm_start(x=augmented(committed) if violation(committed) <= 1e-10 else feasible.x)
        solved = solver.solve(raise_error=False)
    except (OSQPException, ValueError, ArithmeticError) as error:
        return fallback(None, str(error))
    try:
        return certify(solved.x)
    except RuntimeError as error:
        if str(solved.info.status).lower() == "solved":
            raise
        return fallback(solved.x, f"{solved.info.status}: {error}")
