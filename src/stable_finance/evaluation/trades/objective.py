"""Trade from actual state while retaining the complete supplied return objective.

The caller supplies incremental expected payoffs and their complete interval-
major covariance. Only the first reference block is submitted as a current
holding. Later blocks are reoptimized reference positions, never future orders.
This module estimates no forecast, risk or cost and simulates no execution.
It is independently loadable by file path in a preserved execution namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ReturnRiskTradeResult", "solve_return_risk_rebalance"]


@dataclass(frozen=True)
class ReturnRiskTradeResult:
    """Certified original-objective plan, or explicitly infeasible current state.

    Weights and reference plans use current decision-price notionals. Expected
    payoff and variance concern the FULL reference plan; they are not realized
    P&L. Estimated cost and turnover concern only new current orders relative
    to actual plus pending holdings. The execution ledger charges actual fills.
    Infeasible results contain no plan or numerical objective. Arrays are copied
    read-only. Numerical failures raise rather than being called infeasible.
    """

    status: str
    weights: NDArray[np.float64] | None
    reference_plan: NDArray[np.float64] | None
    expected_payoff: float | None
    variance: float | None
    risk_penalty: float | None
    estimated_cost: float | None
    objective: float | None
    optimality_gap: float | None
    max_constraint_violation: float | None
    beta_exposures: NDArray[np.float64] | None
    turnover: float | None
    message: str
    factor_exposures: NDArray[np.float64] | None = None
    factor_names: tuple[str, ...] | None = None
    factor_lower_slack: NDArray[np.float64] | None = None
    factor_upper_slack: NDArray[np.float64] | None = None


def _scalar(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} scalar")
    return float(value)


def _vector(value, n, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (n,):
        raise ValueError(f"{name} must be a finite asset vector of shape {(n,)}")
    array = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _readonly(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
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


def solve_return_risk_rebalance(
    incremental_forecasts: ArrayLike,
    covariance: ArrayLike,
    *,
    previous: ArrayLike,
    pending: ArrayLike | None = None,
    cost: ArrayLike = 0.0,
    risk_aversion: float = 5.0,
    gross_limit: float = 1.0,
    max_weight: float = 0.1,
    dollar_neutral: bool = True,
    beta: ArrayLike | None = None,
    max_abs_beta: float | None = None,
    tradable: ArrayLike | None = None,
    factor_loadings: ArrayLike | None = None,
    factor_lower: ArrayLike | None = None,
    factor_upper: ArrayLike | None = None,
    factor_names: tuple[str, ...] | None = None,
) -> ReturnRiskTradeResult:
    """Maximize original return-risk utility less current net-order costs.

    ``incremental_forecasts`` is ``(K,N)`` or single-interval ``(N,)``;
    ``covariance`` is interval-major ``(K*N,K*N)``. With ``b=previous+pending``
    in current decision NAV/price units, solve

        max_X q'X - gamma/2*X'Omega*X - sum_i c_i*abs(X[0,i]-b_i).

    Every stage has the same supplied net/gross/name limits. Beta accepts
    ``(N,)`` or explicit ``(K,N)`` and requires ``max_abs_beta`` together.
    Optional ``factor_loadings`` has shape ``(N,F)`` or ``(K,N,F)``;
    ``factor_lower`` and ``factor_upper`` are scalar, ``(F,)`` or ``(K,F)``.
    Each stage satisfies ``lower <= loadings.T @ weights <= upper``.
    Generic factors and legacy beta limits both apply when supplied together.
    ``tradable=False`` and missing/positive-infinite costs freeze only the
    CURRENT committed position exactly. Future reference positions retain
    their original decision-known universe; no future quote is consulted.

    Only ``weights=reference_plan[0]`` is a current committed book. All later
    stages are reoptimized and must never be queued as orders. Pending orders
    are irrevocable commitments, not actual fills, and their sunk costs are
    not penalized again. The caller sends ``weights-previous-pending``. The
    constraint guarantee applies to this planned book, not unknown fills or
    later price/NAV drift. Forced liquidation belongs to the execution layer.

    This directly retains the supplied original objective; no forecast is
    inferred from a capped target. No clipping, gross normalization, hidden
    relaxation, parameter selection or future execution-cost assumption occurs.
    """
    raw = np.asarray(incremental_forecasts)
    if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2) or not raw.size:
        raise ValueError("incremental_forecasts must be a nonempty finite (N,) or (K,N) array")
    q_raw = np.array(raw[None] if raw.ndim == 1 else raw, dtype=float, copy=True)
    if min(q_raw.shape) == 0 or not np.isfinite(q_raw).all():
        raise ValueError("incremental forecasts must be finite with nonempty axes")
    k, n = q_raw.shape
    size = k*n
    gamma = _scalar(risk_aversion, "risk_aversion", positive=True)
    gross = _scalar(gross_limit, "gross_limit")
    cap = _scalar(max_weight, "max_weight")
    if not isinstance(dollar_neutral, (bool, np.bool_)):
        raise ValueError("dollar_neutral must be boolean")
    actual = _vector(previous, n, "previous")
    outstanding = np.zeros(n) if pending is None else _vector(pending, n, "pending")
    with np.errstate(over="ignore", invalid="ignore"):
        committed = actual+outstanding
    if not np.isfinite(committed).all():
        raise ValueError("previous plus pending must be finite")
    raw_sigma = np.asarray(covariance)
    if raw_sigma.dtype.kind not in "iuf" or raw_sigma.shape != (size, size):
        raise ValueError(f"covariance must have finite interval-major shape {(size, size)}")
    sigma = np.array(raw_sigma, dtype=float, copy=True)
    if not np.isfinite(sigma).all():
        raise ValueError("covariance must be finite")
    matrix_scale = max(float(np.abs(sigma).max()), np.finfo(float).tiny)
    if np.max(np.abs(sigma-sigma.T)) > 1e-10*matrix_scale:
        raise ValueError("covariance must be symmetric")
    sigma = (sigma+sigma.T)/2
    if np.linalg.eigvalsh(sigma)[0] < -1e-10*matrix_scale:
        raise ValueError("covariance must be positive semidefinite")
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic_full = gamma*sigma
    if not np.isfinite(quadratic_full).all():
        raise ValueError("risk_aversion times covariance exceeds finite range")
    raw_cost = np.asarray(cost)
    if raw_cost.dtype.kind not in "iuf" or raw_cost.shape not in ((), (n,)):
        raise ValueError("cost must be scalar or a current-asset vector")
    charge = np.broadcast_to(np.asarray(raw_cost, dtype=float), (n,)).copy()
    if np.any(charge < 0):
        raise ValueError("cost must be nonnegative or missing")
    current_allowed = np.ones(n, dtype=bool)
    if tradable is not None:
        raw_allowed = np.asarray(tradable)
        if raw_allowed.dtype.kind != "b" or raw_allowed.shape != (n,):
            raise ValueError("tradable must be a boolean current-asset vector")
        current_allowed = raw_allowed.copy()
    current_allowed &= np.isfinite(charge)
    charge[~current_allowed] = 0.
    if (beta is None) != (max_abs_beta is None):
        raise ValueError("beta and max_abs_beta must be supplied together")
    factors = None
    if beta is not None:
        bound = _scalar(max_abs_beta, "max_abs_beta")
        raw_beta = np.asarray(beta)
        if raw_beta.dtype.kind not in "iuf" or raw_beta.shape not in ((n,), (k, n)):
            raise ValueError("beta must have shape (N,) or explicit (K,N)")
        factors = np.broadcast_to(np.asarray(raw_beta, dtype=float), (k, n)).copy()
        if not np.isfinite(factors).all():
            raise ValueError("beta must be finite for every current and future reference asset")

    allowed = np.ones(size, dtype=bool)
    allowed[:n] = current_allowed
    fixed = np.zeros(size)
    fixed[:n] = np.where(current_allowed, 0., committed)
    generic = lower_factors = upper_factors = names = None
    if any(value is not None for value in
           (factor_loadings, factor_lower, factor_upper, factor_names)):
        generic, lower_factors, upper_factors, names = _factor_arrays(
            factor_loadings, factor_lower, factor_upper, stages=k, n_assets=n,
            factor_names=factor_names,
            required=(allowed | (fixed != 0.)).reshape(k, n))
        names = tuple(names.tolist())
    alpha = q_raw.copy()
    if dollar_neutral:
        alpha -= alpha.mean(axis=1, keepdims=True)
    alpha = alpha.ravel()

    def violation(vector):
        plan = vector.reshape(k, n)
        excess = [0., float(np.max(np.abs(plan).sum(axis=1))-gross),
                  float(np.max(np.abs(plan))-cap)]
        if dollar_neutral:
            excess.append(float(np.max(np.abs(plan.sum(axis=1)))))
        if factors is not None:
            excess.append(float(np.max(np.abs((plan*factors).sum(axis=1)))-bound))
        if generic is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                exposure = np.einsum("knf,kn->kf", generic, plan)
            if not np.isfinite(exposure).all():
                raise RuntimeError("factor exposure exceeds finite range")
            excess.extend((float(np.max(lower_factors-exposure)),
                           float(np.max(exposure-upper_factors))))
        if np.any(~current_allowed):
            excess.append(float(np.max(np.abs(plan[0, ~current_allowed]-committed[~current_allowed]))))
        return max(excess)

    def result(vector, gap=0., message="Independent primal and convex-gap certificates passed"):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise RuntimeError("return-risk optimizer returned an invalid reference plan")
        residual = float(violation(vector))
        if residual > 1e-8 or not np.isfinite(residual):
            raise RuntimeError(f"return-risk plan violates stage constraints by {residual:.3g}")
        plan = vector.reshape(k, n)
        payoff = float(q_raw.ravel() @ vector)
        variance = max(0., float(vector @ sigma @ vector))
        penalty = gamma/2*variance
        estimated = float(charge @ np.abs(plan[0]-committed))
        utility = payoff-penalty-estimated
        if not np.isfinite([payoff, variance, penalty, estimated, utility, gap]).all():
            raise RuntimeError("return-risk objective exceeds finite range")
        exposures = None if factors is None else _readonly((plan*factors).sum(axis=1))
        generic_exposures = (None if generic is None else
                             np.einsum("knf,kn->kf", generic, plan))
        return ReturnRiskTradeResult("solved", _readonly(plan[0]), _readonly(plan),
            payoff, variance, penalty, estimated, utility, float(gap), residual,
            exposures, float(np.abs(plan[0]-committed).sum()), message,
            None if generic is None else _readonly(generic_exposures), names,
            None if generic is None else _readonly(generic_exposures-lower_factors),
            None if generic is None else _readonly(upper_factors-generic_exposures))

    def infeasible(message):
        return ReturnRiskTradeResult("infeasible", None, None, None, None, None,
                                     None, None, None, None, None, None, message)

    frozen_gross = np.abs(fixed.reshape(k, n)).sum(axis=1)
    if (np.any(frozen_gross > gross+1e-10) or np.any(np.abs(fixed) > cap+1e-10)):
        return infeasible("Frozen current committed holdings exceed gross or name limits")
    members = np.flatnonzero(allowed)
    m = len(members)
    if not m:
        if violation(fixed) > 1e-8:
            return infeasible("Frozen current committed holdings cannot satisfy exposure limits")
        return result(fixed, message="All current positions frozen; no future stage")

    from scipy.optimize import linprog

    current_members = np.flatnonzero(current_allowed)
    t = len(current_members)
    free_stage = members//n
    stage = (np.arange(k)[:, None] == free_stage[None]).astype(float)
    current_select = np.zeros((t, m))
    if t:
        current_select[np.arange(t), np.searchsorted(members, current_members)] = 1.
    eye_m = np.eye(m)
    zeros_mm, zeros_mt = np.zeros((m, m)), np.zeros((m, t))
    zeros_kt = np.zeros((k, t))
    a_ub = np.vstack([
        np.hstack([eye_m, -eye_m, zeros_mt]),
        np.hstack([-eye_m, -eye_m, zeros_mt]),
        np.hstack([np.zeros((k, m)), stage, zeros_kt]),
        np.hstack([current_select, np.zeros((t, m)), -np.eye(t)]),
        np.hstack([-current_select, np.zeros((t, m)), -np.eye(t)]),
    ])
    b_ub = np.r_[np.zeros(2*m), gross-frozen_gross,
                 committed[current_members], -committed[current_members]]
    if factors is not None:
        factor_rows = stage*factors.ravel()[members][None]
        padded = np.hstack([factor_rows, np.zeros((k, m+t))])
        fixed_beta = (factors*fixed.reshape(k, n)).sum(axis=1)
        a_ub = np.vstack([a_ub, padded, -padded])
        b_ub = np.r_[b_ub, bound-fixed_beta, bound+fixed_beta]
    if generic is not None:
        count = generic.shape[2]
        rows = (stage[:, None, :] * generic.reshape(size, count)[members].T[None]).reshape(k*count, m)
        padded = np.hstack([rows, np.zeros((k*count, m+t))])
        frozen_exposure = np.einsum("knf,kn->kf", generic, fixed.reshape(k, n)).ravel()
        finite_upper = np.isfinite(upper_factors.ravel())
        finite_lower = np.isfinite(lower_factors.ravel())
        a_ub = np.vstack([a_ub, padded[finite_upper], -padded[finite_lower]])
        b_ub = np.r_[b_ub, upper_factors.ravel()[finite_upper]-frozen_exposure[finite_upper],
                     frozen_exposure[finite_lower]-lower_factors.ravel()[finite_lower]]
    a_eq = b_eq = None
    if dollar_neutral:
        fixed_net = fixed.reshape(k, n).sum(axis=1)
        nonempty = np.any(stage, axis=1)
        if np.any(np.abs(fixed_net[~nonempty]) > 1e-8):
            return infeasible("Frozen current holdings cannot satisfy dollar neutrality")
        a_eq = np.hstack([stage[nonempty], np.zeros((int(nonempty.sum()), m+t))])
        b_eq = -fixed_net[nonempty]
    lower = np.r_[np.full(m, -cap), np.zeros(m+t)]
    upper = np.r_[np.full(2*m, cap), cap+np.abs(committed[current_members])]
    if not np.isfinite(upper).all():
        raise ValueError("position/turnover bounds exceed finite range")
    bounds = list(zip(lower, upper))
    lp_options = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)

    def lp(linear):
        return linprog(linear, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                       bounds=bounds, method="highs", options=lp_options)

    feasible = lp(np.zeros(2*m+t))
    if feasible.status == 2:
        return infeasible("No reference plan satisfies frozen current holdings and all stage limits")
    if not feasible.success or not np.isfinite(feasible.x).all():
        raise RuntimeError(f"return-risk feasibility check failed: {feasible.message}")
    with np.errstate(over="ignore", invalid="ignore"):
        linear = quadratic_full[members] @ fixed-alpha[members]
    quadratic = quadratic_full[np.ix_(members, members)]
    scale = max(float(np.abs(linear).max()), float(np.abs(quadratic).max()),
                float(charge[current_members].max(initial=0.)))
    if not np.isfinite(scale):
        raise ValueError("return-risk objective scale exceeds finite range")
    if scale == 0:
        vector = fixed.copy()
        vector[members] = feasible.x[:m]
        return result(vector, message="Feasible plan for a constant original objective")
    quadratic, linear, costs = quadratic/scale, linear/scale, charge[current_members]/scale
    q = np.r_[linear, np.zeros(m), costs]

    def objective(x):
        return float(.5*x[:m] @ quadratic @ x[:m]+q @ x)

    def gradient(x):
        return q+np.r_[quadratic @ x[:m], np.zeros(m+t)]

    def augmented(vector):
        free = vector[members]
        return np.r_[free, np.abs(free), np.abs(vector[current_members]-committed[current_members])]

    def certify(candidate):
        if np.shape(candidate) != (2*m+t,) or not np.isfinite(candidate).all():
            raise RuntimeError("return-risk optimizer returned an invalid candidate")
        vector = fixed.copy()
        vector[members] = candidate[:m]
        x = augmented(vector)
        residuals = np.r_[a_ub @ x-b_ub, lower-x, x-upper,
                          np.abs(a_eq @ x-b_eq) if a_eq is not None else [0.]]
        if not np.isfinite(residuals).all() or np.max(residuals) > 1e-8:
            raise RuntimeError(f"return-risk optimizer violates constraints by {np.max(residuals):.3g}")
        grad = gradient(x)
        certificate = lp(grad)
        if not certificate.success or not np.isfinite(certificate.fun):
            raise RuntimeError(f"return-risk optimality certificate failed: {certificate.message}")
        gap = max(0., float(grad @ x-certificate.fun))
        if gap > 1e-7*max(1., abs(objective(x))):
            raise RuntimeError(f"return-risk optimizer did not converge: optimality gap {gap:.3g}")
        return result(vector, gap*scale)

    initial = feasible.x
    # Holding the current committed book can be exactly optimal even when
    # future reference positions have positive utility. Optimize the latter
    # conditionally, then independently certify the ENTIRE original objective.
    # This avoids interpreting numerical near-zero orders as intentional trades.
    current_plan = np.zeros(size)
    current_plan[:n] = committed
    current_feasible = (np.abs(committed).sum() <= gross+1e-10
        and np.max(np.abs(committed)) <= cap+1e-10
        and (not dollar_neutral or abs(committed.sum()) <= 1e-10)
        and (factors is None or abs(float(factors[0] @ committed)) <= bound+1e-10))
    if generic is not None:
        current_exposure = generic[0].T @ committed
        current_feasible = (current_feasible
            and np.all(current_exposure >= lower_factors[0]-1e-10)
            and np.all(current_exposure <= upper_factors[0]+1e-10))
    if current_feasible and t:
        if k > 1:
            try:
                conditional = solve_return_risk_rebalance(q_raw, sigma, previous=actual,
                    pending=outstanding, cost=0., risk_aversion=gamma, gross_limit=gross,
                    max_weight=cap, dollar_neutral=dollar_neutral, beta=factors,
                    max_abs_beta=None if factors is None else bound,
                    tradable=np.zeros(n, dtype=bool), factor_loadings=generic,
                    factor_lower=lower_factors, factor_upper=upper_factors,
                    factor_names=names)
                if conditional.status == "solved":
                    current_plan = np.asarray(conditional.reference_plan).ravel()
            except RuntimeError:
                pass  # The full solve below may still produce a certified plan.
        try:
            return certify(augmented(current_plan))
        except RuntimeError:
            initial = augmented(current_plan) if violation(current_plan) <= 1e-8 else feasible.x

    if not np.any(quadratic):
        solved = lp(q)
        if not solved.success:
            raise RuntimeError(f"return-risk linear optimization failed: {solved.message}")
        return certify(solved.x)

    try:
        from osqp import OSQP, OSQPException
        from scipy.sparse import block_diag, csc_matrix, eye, triu, vstack
    except ImportError as error:
        raise ImportError("Constrained original-objective trades require stable-finance[joint].") from error
    hessian = block_diag([csc_matrix(quadratic), csc_matrix((m+t, m+t))], format="csc")
    matrices, lows, highs = [csc_matrix(a_ub)], [np.full(len(b_ub), -np.inf)], [b_ub]
    if a_eq is not None:
        matrices.append(csc_matrix(a_eq)); lows.append(b_eq); highs.append(b_eq)
    matrices.append(eye(2*m+t, format="csc")); lows.append(lower); highs.append(upper)

    def fallback(candidate, primary_error):
        from scipy.optimize import minimize

        start = candidate if np.shape(candidate) == (2*m+t,) and np.isfinite(candidate).all() else initial
        conditions = [dict(type="ineq", fun=lambda x:b_ub-a_ub @ x, jac=lambda x:-a_ub)]
        if a_eq is not None:
            conditions.append(dict(type="eq", fun=lambda x:a_eq @ x-b_eq, jac=lambda x:a_eq))
        alternative = minimize(objective, start, jac=gradient, method="SLSQP",
            bounds=bounds, constraints=conditions,
            options=dict(maxiter=2000, ftol=1e-12, disp=False))
        try:
            return certify(alternative.x)
        except (ValueError, RuntimeError, TypeError) as error:
            raise RuntimeError(f"Original-objective rebalance could not be certified: "
                f"primary={primary_error}; fallback={alternative.message}; certificate={error}") from error

    try:
        solver = OSQP()
        solver.setup(P=triu(hessian, format="csc"), q=q,
            A=vstack(matrices, format="csc"), l=np.concatenate(lows), u=np.concatenate(highs),
            verbose=False, eps_abs=1e-10, eps_rel=1e-10,
            eps_prim_inf=1e-10, eps_dual_inf=1e-10, max_iter=100000,
            polishing=True, polish_refine_iter=10, adaptive_rho=True,
            adaptive_rho_interval=50, check_termination=25, warm_starting=True)
        solver.warm_start(x=initial)
        solved = solver.solve(raise_error=False)
    except (OSQPException, ValueError, ArithmeticError) as error:
        return fallback(None, str(error))
    try:
        return certify(solved.x)
    except RuntimeError as error:
        if str(solved.info.status).lower() == "solved":
            raise
        return fallback(solved.x, f"{solved.info.status}: {error}")
