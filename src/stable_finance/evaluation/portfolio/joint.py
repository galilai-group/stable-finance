"""Frictionless joint planning over disjoint decision-price payoffs.

The plan commits reference shares, not future equity weights. Only its first
block is a current target; a caller may solve again at the next decision.
Full cross-interval risk couples this decision to later forecasts. With
block-diagonal interval risk and stage-separable constraints, the first
block correctly ignores later forecasts rather than manufacturing coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from scipy.sparse import block_diag, csc_matrix, eye as sparse_eye, triu, vstack

from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.joint_risk import (
    JointReturnPanel, JointRiskModel, _clock, _same_clock,
)
from stable_finance.evaluation.portfolio.joint_constraints import JointPortfolioConstraints
from stable_finance.evaluation.portfolio.factors import (
    FactorExposureConstraints, factor_constraint_arrays,
)

__all__ = ["JointPortfolioConfig", "JointPortfolioResult", "construct_joint_portfolio"]


@dataclass(frozen=True)
class JointPortfolioConfig:
    """Fixed risk preference and reference-notional limits at EVERY stage.

    No current holdings, costs, turnover penalties or smoothness parameters
    enter target construction. ``risk_aversion`` weights terminal reference
    P&L variance. Limits constrain entry-decision-price notionals, not the
    unknown future equity or future marked dollar neutrality.
    """

    risk_aversion: float = 5.0
    constraints: JointPortfolioConstraints = JointPortfolioConstraints()

    def __post_init__(self):
        if (isinstance(self.risk_aversion, (bool, np.bool_))
                or not isinstance(self.risk_aversion, Real)
                or not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0):
            raise ValueError("risk_aversion must be finite and positive")
        if not isinstance(self.constraints, JointPortfolioConstraints):
            raise TypeError("constraints must be JointPortfolioConstraints")


@dataclass(frozen=True)
class JointPortfolioResult:
    """Current targets and a diagnostic open-loop reference-share plan.

    ``reference_plan`` and ``incremental_forecasts`` have shape (D,K,N),
    with interval-major order matching the risk covariance. The full plan
    maximizes risk-adjusted terminal P&L under the supplied conditional
    means and risk model; it is not a fully adaptive stochastic policy.
    ``current_targets`` contains only block zero on a (D,N,1) grid. Later
    blocks must not be treated as future orders or future equity weights.
    ``objective`` is expected P&L minus gamma/2 times variance. The reported
    optimality gap is in the original objective's return units.
    When requested, ``beta`` and ``beta_exposures`` record the supplied
    (D,K,N) coefficients and resulting (D,K) stage exposures; unavailable
    coefficients are retained only on explicitly excluded positions.
    Generic ``factor_loadings`` have shape (D,K,N,F), while exposures,
    budgets and slacks have shape (D,K,F). Excluded unknown coefficients
    are zeroed for multiplication; ``factor_constraints`` retains their
    original declaration for audit. Reported bounds constrain this reference
    plan, rather than actual holdings after execution or price changes.
    """

    current_targets: PortfolioWeights
    reference_plan: NDArray[np.float64]
    incremental_forecasts: NDArray[np.float64]
    expected_pnl: NDArray[np.float64]
    variance: NDArray[np.float64]
    objective: NDArray[np.float64]
    optimality_gap: NDArray[np.float64]
    decisions: NDArray
    assets: NDArray
    horizons: NDArray
    interval_seconds: NDArray[np.int64]
    risk_model: JointRiskModel
    config: JointPortfolioConfig
    beta: NDArray[np.float64] | None = None
    beta_exposures: NDArray[np.float64] | None = None
    max_abs_beta: float | None = None
    beta_available_at: NDArray | None = None
    factor_constraints: FactorExposureConstraints | None = None
    factor_names: NDArray | None = None
    factor_loadings: NDArray[np.float64] | None = None
    factor_exposures: NDArray[np.float64] | None = None
    factor_lower: NDArray[np.float64] | None = None
    factor_upper: NDArray[np.float64] | None = None
    factor_available_at: NDArray | None = None
    factor_lower_slack: NDArray[np.float64] | None = None
    factor_upper_slack: NDArray[np.float64] | None = None


def _readonly(values: ArrayLike) -> np.ndarray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _eligibility(value, d, k, n):
    """Expand only explicitly documented, boolean decision-known masks."""
    if value is None:
        return np.ones((d, k, n), dtype=bool)
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise ValueError("eligible must contain booleans")
    if array.ndim == 0:
        return np.full((d, k, n), bool(array), dtype=bool)
    if array.shape == (n,):
        return np.broadcast_to(array[None, None], (d, k, n)).copy()
    if array.shape == (d, n):
        return np.broadcast_to(array[:, None], (d, k, n)).copy()
    if array.shape == (d, k, n):
        return array.copy()
    raise ValueError("eligible must be scalar, (N,), (D,N), or explicit (D,K,N)")


def _beta_constraint(beta, max_abs_beta, available_at, decisions, allowed):
    """Validate declared exogenous stage betas and their causal availability."""
    if (beta is None) != (max_abs_beta is None):
        raise ValueError("beta and max_abs_beta must be supplied together")
    if beta is None:
        if available_at is not None:
            raise ValueError("beta_available_at requires a beta constraint")
        return None, None, None
    if (isinstance(max_abs_beta, (bool, np.bool_))
            or not isinstance(max_abs_beta, Real) or not np.isfinite(max_abs_beta)
            or max_abs_beta < 0 or not np.isfinite(float(max_abs_beta))):
        raise ValueError("max_abs_beta must be a finite nonnegative number")
    d, k, n = allowed.shape
    raw = np.asarray(beta)
    if raw.dtype.kind not in "iuf" or np.isinf(raw).any():
        raise ValueError("beta must contain numbers or NaN on excluded coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(raw, dtype=np.float64)
    if values.shape == (n,):
        values = np.broadcast_to(values[None, None], (d, k, n))
    elif values.shape == (d, n):
        values = np.broadcast_to(values[:, None], (d, k, n))
    elif values.shape != (d, k, n):
        raise ValueError("beta must have shape (N,), (D,N), or explicit (D,K,N)")
    if np.isinf(values).any() or not np.isfinite(values[allowed]).all():
        raise ValueError("beta must be finite for every eligible asset/stage")
    when, kind = _clock(decisions, "forecast decisions")
    times, time_kind = _clock(when if available_at is None else available_at,
                              "beta_available_at")
    _same_clock(kind, time_kind)
    if times.ndim == 0:
        times = np.full((d, k), times[()], dtype=times.dtype)
    elif times.shape == (d,):
        times = np.broadcast_to(times[:, None], (d, k))
    elif times.shape != (d, k):
        raise ValueError("beta_available_at must be scalar, (D,), or (D,K)")
    if np.any(times > when[:, None]):
        raise ValueError("beta_available_at is later than the forecast decision")
    return _readonly(values), float(max_abs_beta), _readonly(times)


def _solve_plan(mu, covariance, k, n, allowed, config, beta=None, max_abs_beta=None,
                *, factor_loadings=None, factor_lower=None, factor_upper=None):
    """Solve one convex reference-share QP and independently certify it."""
    m = k*n
    gross, cap = config.constraints.gross_limit, config.constraints.max_weight
    factor_values, factor_l, factor_u, _ = factor_constraint_arrays(
        factor_loadings, factor_lower, factor_upper, stages=k, n_assets=n,
        required=allowed)
    has_factors = factor_values is not None
    if not np.any(allowed) or gross == 0 or cap == 0:
        if has_factors and (np.any(factor_l > 0.) or np.any(factor_u < 0.)):
            raise RuntimeError("joint factor constraints are infeasible for the forced zero book")
        return np.zeros(m), 0.0
    # A stage-specific common mean contributes nothing under its own net
    # constraint. Removing it improves precision without changing utility.
    alpha = np.array(mu, dtype=float, copy=True).reshape(k, n)
    if config.constraints.dollar_neutral:
        alpha -= alpha.mean(axis=1, keepdims=True)
    alpha = alpha.ravel()
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic = config.risk_aversion*covariance
    if not np.isfinite(quadratic).all():
        raise ValueError("risk_aversion times covariance exceeds finite numerical range")
    scale = max(float(np.abs(alpha).max()),
                float(np.abs(quadratic).max())*k*min(gross, n*cap))
    if not np.isfinite(scale):
        raise ValueError("joint objective exceeds finite numerical range")
    if scale == 0:
        if not has_factors:
            return np.zeros(m), 0.0
        # A constant objective may still require a nonzero feasible book.
        # Preserve the exact legacy fast path when no generic family exists.
        scale = 1.
    alpha, quadratic = alpha/scale, quadratic/scale

    identity = np.eye(m)
    stage = np.kron(np.eye(k), np.ones((1, n)))
    a_ub = np.vstack([np.hstack([identity, -identity]),
                      np.hstack([-identity, -identity]),
                      np.hstack([np.zeros((k, m)), stage])])
    b_ub = np.r_[np.zeros(2*m), np.full(k, gross)]
    if beta is not None:
        # A separate absolute beta bound applies to every reference stage.
        # Excluded coordinates have no holdings and may have unknown beta.
        coefficients = np.where(allowed, beta, 0.)
        beta_rows = np.zeros((k, m))
        for stage_index in range(k):
            beta_rows[stage_index, stage_index*n:(stage_index+1)*n] = coefficients[stage_index]
        padded_beta = np.hstack([beta_rows, np.zeros((k, m))])
        a_ub = np.vstack([a_ub, padded_beta, -padded_beta])
        b_ub = np.r_[b_ub, np.full(2*k, max_abs_beta)]
    if has_factors:
        factors = factor_values.shape[2]
        factor_rows = np.zeros((k*factors, m))
        for stage_index in range(k):
            factor_rows[stage_index*factors:(stage_index+1)*factors,
                        stage_index*n:(stage_index+1)*n] = factor_values[stage_index].T
        padded_factors = np.hstack([factor_rows, np.zeros((k*factors, m))])
        lower_flat, upper_flat = factor_l.ravel(), factor_u.ravel()
        finite_upper, finite_lower = np.isfinite(upper_flat), np.isfinite(lower_flat)
        a_ub = np.vstack([a_ub, padded_factors[finite_upper], -padded_factors[finite_lower]])
        b_ub = np.r_[b_ub, upper_flat[finite_upper], -lower_flat[finite_lower]]
    a_eq = np.hstack([stage, np.zeros((k, m))]) if config.constraints.dollar_neutral else None
    b_eq = np.zeros(k) if a_eq is not None else None
    eligible_cap = cap*np.asarray(allowed, dtype=bool).ravel()
    lower = np.r_[-eligible_cap, np.zeros(m)]
    upper = np.r_[eligible_cap, eligible_cap]
    feasible_start = None
    if has_factors:
        feasible = linprog(np.zeros(2*m), A_ub=a_ub, b_ub=b_ub,
            A_eq=a_eq, b_eq=b_eq, bounds=list(zip(lower, upper)), method="highs",
            options={"dual_feasibility_tolerance": 1e-9,
                     "primal_feasibility_tolerance": 1e-9})
        if feasible.status == 2:
            raise RuntimeError("joint factor constraints are infeasible with the supplied position limits")
        if not feasible.success or not np.isfinite(feasible.x).all():
            raise RuntimeError(f"joint factor feasibility check failed: {feasible.message}")
        feasible_start = feasible.x

    def objective(x):
        return .5*x[:m] @ quadratic @ x[:m] - alpha @ x[:m]

    def certify(weights):
        if np.shape(weights) != (m,) or not np.isfinite(weights).all():
            raise RuntimeError("joint optimizer returned an invalid plan")
        # Hard-excluded coordinates are exactly zero, followed by a fresh
        # certificate. No general clipping or portfolio rescaling is applied.
        weights = np.array(weights, copy=True)
        weights[~np.asarray(allowed).ravel()] = 0.0
        x = np.r_[weights, np.abs(weights)]
        residuals = np.r_[a_ub @ x-b_ub, lower-x, x-upper,
                          np.abs(a_eq @ x) if a_eq is not None else [0.]]
        if not np.isfinite(residuals).all():
            raise RuntimeError("joint constraint residuals exceeded finite numerical range")
        residual = float(np.max(residuals))
        if residual > 1e-8:
            raise RuntimeError(f"joint plan violates stage constraints by {residual:.3g}")
        gradient = np.r_[quadratic @ weights-alpha, np.zeros(m)]
        certificate = linprog(gradient, A_ub=a_ub, b_ub=b_ub,
            A_eq=a_eq, b_eq=b_eq, bounds=list(zip(lower, upper)), method="highs",
            options={"dual_feasibility_tolerance": 1e-9,
                     "primal_feasibility_tolerance": 1e-9})
        if not certificate.success or not np.isfinite(certificate.fun):
            raise RuntimeError(f"joint optimality certificate failed: {certificate.message}")
        gap = max(0.0, float(gradient @ x-certificate.fun))
        if gap > 1e-7*max(1.0, abs(objective(x))):
            raise RuntimeError(f"joint optimizer did not converge: optimality gap {gap:.3g}")
        return weights, gap*scale

    # The exact unconstrained/equality solution is both faster and more
    # precise when all requested position limits are inactive.
    members = np.flatnonzero(np.asarray(allowed).ravel())
    reduced = quadratic[np.ix_(members, members)]
    try:
        free = np.linalg.solve(reduced, alpha[members])
        if a_eq is not None:
            equality = stage[:, members]
            equality = equality[np.any(equality, axis=1)]
            hedge = np.linalg.solve(reduced, equality.T)
            free -= hedge @ np.linalg.solve(equality @ hedge, equality @ free)
        plan = np.zeros(m)
        plan[members] = free
        beta_feasible = (beta is None or
            np.max(np.abs((np.where(allowed, beta, 0.)*plan.reshape(k, n)).sum(axis=1)))
            <= max_abs_beta+1e-10)
        factor_exposure = (None if not has_factors else
            np.einsum("kn,knf->kf", plan.reshape(k, n), factor_values))
        factors_feasible = (not has_factors or
            (np.all(factor_exposure >= factor_l-1e-10)
             and np.all(factor_exposure <= factor_u+1e-10)))
        if (np.isfinite(plan).all() and np.max(np.abs(plan)) <= cap
                and np.max(np.abs(plan.reshape(k, n)).sum(axis=1)) <= gross
                and beta_feasible and factors_feasible):
            return certify(plan)
    except np.linalg.LinAlgError:
        pass  # A singular supplied PSD model still defines a constrained QP.

    if not np.any(quadratic):
        solved = linprog(np.r_[-alpha, np.zeros(m)], A_ub=a_ub, b_ub=b_ub,
            A_eq=a_eq, b_eq=b_eq, bounds=list(zip(lower, upper)), method="highs")
        if not solved.success:
            raise RuntimeError(f"joint linear optimization failed: {solved.message}")
        return certify(solved.x[:m])

    # Optional numerical backend is needed only for a binding constrained QP.
    # Importing this module, risk estimation, and basic portfolio construction
    # remain usable without installing the joint optimization extra.
    try:
        from osqp import OSQP, OSQPException
    except ImportError as error:
        raise ImportError(
            "Constrained joint optimization requires the optional backend; "
            "install stable-finance[joint]."
        ) from error

    hessian = block_diag([csc_matrix(quadratic), csc_matrix((m, m))], format="csc")
    matrices = [csc_matrix(a_ub)]
    lower_rows, upper_rows = [np.full(len(b_ub), -np.inf)], [b_ub]
    if a_eq is not None:
        matrices.append(csc_matrix(a_eq))
        lower_rows.append(b_eq)
        upper_rows.append(b_eq)
    matrices.append(sparse_eye(2*m, format="csc"))
    lower_rows.append(lower)
    upper_rows.append(upper)

    def fallback(candidate, primary_error):
        """Solve the identical factor-constrained QP with a second algorithm.

        A near-degenerate active factor/position set can exhaust ADMM's
        iteration budget before its primal accuracy meets our certificate.
        SLSQP uses the same scaled quadratic objective, exact gradient,
        augmented variables, linear constraints and bounds. Its success flag
        is not sufficient: the unchanged independent certificate decides.
        """
        from scipy.optimize import minimize

        initial = (np.array(candidate, copy=True)
                   if np.shape(candidate) == (2*m,) and np.isfinite(candidate).all()
                   else feasible_start if has_factors else np.zeros(2*m))
        constraints = [dict(type="ineq", fun=lambda x: b_ub-a_ub @ x,
                            jac=lambda x: -a_ub)]
        if a_eq is not None:
            constraints.append(dict(type="eq", fun=lambda x: a_eq @ x-b_eq,
                                    jac=lambda x: a_eq))
        solved_again = minimize(objective, initial,
            jac=lambda x: np.r_[quadratic @ x[:m]-alpha, np.zeros(m)],
            method="SLSQP", bounds=list(zip(lower, upper)),
            constraints=constraints,
            options=dict(maxiter=2000, ftol=1e-12, disp=False))
        try:
            return certify(solved_again.x[:m])
        except (RuntimeError, ValueError, TypeError) as fallback_error:
            raise RuntimeError(
                f"joint {'factor' if has_factors else 'beta'} QP could not be independently certified: "
                f"primary={primary_error}; fallback={solved_again.message}; "
                f"certificate={fallback_error}"
            ) from fallback_error

    try:
        solver = OSQP()
        solver.setup(P=triu(hessian, format="csc"), q=np.r_[-alpha, np.zeros(m)],
            A=vstack(matrices, format="csc"), l=np.concatenate(lower_rows),
            u=np.concatenate(upper_rows), verbose=False,
            eps_abs=1e-10, eps_rel=1e-10, eps_prim_inf=1e-10, eps_dual_inf=1e-10,
            max_iter=100000, polishing=True, polish_refine_iter=10,
            adaptive_rho=True, adaptive_rho_interval=50, check_termination=25,
            warm_starting=True)
        solver.warm_start(x=feasible_start if has_factors else np.zeros(2*m))
        solved = solver.solve(raise_error=False)
    except (OSQPException, ValueError, ArithmeticError) as error:
        if beta is not None or has_factors:
            return fallback(None, str(error))
        raise RuntimeError(f"joint QP failed: {error}") from error
    if np.shape(solved.x) != (2*m,) or not np.isfinite(solved.x).all():
        if (beta is not None or has_factors) and str(solved.info.status).lower() != "solved":
            return fallback(None, f"invalid primary candidate ({solved.info.status})")
        raise RuntimeError(f"joint QP returned an invalid candidate ({solved.info.status})")
    # Acceptance depends on the independent primal and convex-gap checks,
    # never on an OSQP success string or an inaccurate-solution flag alone.
    try:
        return certify(solved.x[:m])
    except RuntimeError as error:
        if (beta is None and not has_factors) or str(solved.info.status).lower() == "solved":
            raise
        return fallback(solved.x, f"{solved.info.status}: {error}")


def construct_joint_portfolio(
    forecast: JointReturnPanel,
    *,
    risk: JointRiskModel,
    config: JointPortfolioConfig = JointPortfolioConfig(),
    eligible: ArrayLike | None = None,
    beta: ArrayLike | None = None,
    max_abs_beta: float | None = None,
    beta_available_at: ArrayLike | None = None,
    factor_constraints: FactorExposureConstraints | None = None,
) -> JointPortfolioResult:
    """Optimize a frictionless common-decision-price payoff path.

    Let ``Y_h=(P[t+delay+h]-P[t+delay])/P[t]`` and ``Y_0=0``. Forecasts are
    conditional means of Y, NOT ordinary returns divided by an unknown future
    entry price. Exact interval payoffs are ``q_k=Y_hk-Y_h(k-1)``. The plan
    chooses x to maximize ``E[q]'x - gamma/2*x'Cov(q)*x``. Full covariance has
    interval-major coordinates ``k*N+i`` and must be available at the current
    decision, with matching assets, horizons, basis and entry delay.

    All forecast values must be finite; a missing cumulative mean is never
    replaced with zero before differencing. Eligibility accepts a boolean
    scalar, (N,), (D,N), or explicit (D,K,N) stage mask. Every mask entry must
    be known when that decision is made; later quote availability is not a
    valid eligibility rule. Supplied stage limits apply to reference notionals.

    An optional exogenous factor constraint requires both ``beta`` and
    ``max_abs_beta``. Each stage obeys ``abs(sum_i beta_i*x_i) <= max_abs_beta``.
    Beta shapes are static (N,), decision-varying (D,N), or explicit (D,K,N).
    Unknown beta is permitted only on excluded coordinates. Availability can
    be scalar, (D,), or (D,K), and cannot be later than its decision; omitting
    it declares current availability. No beta is estimated or calibrated here.
    These are reference-stage constraints, not a guarantee on marked exposure
    after execution, price drift or costs. Calls without either constraint
    family preserve the existing objective and feasible set. There is no
    post-solve hedge or projection.

    ``factor_constraints`` optionally adds labeled bounds
    ``lower <= loadings.T @ x_k <= upper`` at every stage. Static, decision-
    varying and explicitly staged coefficients are resolved by their axes and
    causal availability. A supplied beta bound and generic factor bounds are
    intersected; neither overwrites the other. Missing coefficients can only
    be ignored on coordinates explicitly excluded from every possible holding.

    Only the first block is emitted as current targets. Later blocks are an
    open-loop diagnostic, not predictions of future equity weights or orders.
    Re-solving on later information gives receding-horizon control. If risk
    is block diagonal and constraints separate by stage, no later mean can
    affect the first block; the method does not add artificial time coupling.
    Execution, actual-state feedback and transaction costs remain downstream.
    """
    if not isinstance(forecast, JointReturnPanel):
        raise TypeError("forecast must be JointReturnPanel with explicit payoff semantics")
    if not isinstance(risk, JointRiskModel):
        raise TypeError("risk must be JointRiskModel")
    if not isinstance(config, JointPortfolioConfig):
        raise TypeError("config must be JointPortfolioConfig")
    for axis in ("assets", "horizons"):
        if not np.array_equal(getattr(forecast, axis), getattr(risk, axis)):
            raise ValueError(f"risk {axis} are not aligned with forecast")
    if forecast.entry_delay_seconds != risk.entry_delay_seconds:
        raise ValueError("risk and forecast entry_delay_seconds must match")
    if forecast.return_basis != "decision_price_payoff" or risk.return_basis != "decision_price_payoff_increments":
        raise ValueError("forecast and risk payoff bases do not describe decision-price payoffs")
    values = np.asarray(forecast.values, dtype=float)
    if min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("joint forecast values must be nonempty and finite")
    for decision in forecast.decisions:
        risk.assert_available(decision)
    d, n, k = values.shape
    allowed = _eligibility(eligible, d, k, n)
    beta_values, beta_bound, beta_times = _beta_constraint(
        beta, max_abs_beta, beta_available_at, forecast.decisions, allowed)
    resolved_factors = None
    if factor_constraints is not None:
        if not isinstance(factor_constraints, FactorExposureConstraints):
            raise TypeError("factor_constraints must be FactorExposureConstraints")
        resolved_factors = factor_constraints.resolve(decisions=forecast.decisions,
            assets=forecast.assets, horizons=forecast.horizons, eligible=allowed)
    increments = np.diff(values, axis=2, prepend=np.zeros((d, n, 1))).transpose(0, 2, 1)
    plan = np.zeros((d, k, n))
    expected, variance, objective, gaps = (np.zeros(d) for _ in range(4))
    for row in range(d):
        factor_options = ({} if resolved_factors is None else dict(
            factor_loadings=resolved_factors.loadings[row],
            factor_lower=resolved_factors.lower[row], factor_upper=resolved_factors.upper[row]))
        weights, gap = _solve_plan(increments[row].ravel(), risk.covariance,
            k, n, allowed[row], config,
            None if beta_values is None else beta_values[row], beta_bound, **factor_options)
        plan[row] = weights.reshape(k, n)
        expected[row] = increments[row].ravel() @ weights
        variance[row] = max(0.0, float(weights @ risk.covariance @ weights))
        objective[row] = expected[row] - config.risk_aversion*variance[row]/2
        gaps[row] = gap
    if not all(np.isfinite(x).all() for x in (expected, variance, objective, gaps)):
        raise RuntimeError("joint objective diagnostics exceeded finite numerical range")
    current = PortfolioWeights(_readonly(plan[:, 0, :, None]),
        _readonly(forecast.decisions), _readonly(forecast.assets),
        _readonly(forecast.horizons[:1]))
    with np.errstate(over="ignore", invalid="ignore"):
        exposures = (None if resolved_factors is None else
            np.einsum("dkn,dknf->dkf", plan, resolved_factors.loadings))
    if exposures is not None and not np.isfinite(exposures).all():
        raise RuntimeError("joint factor exposure diagnostics exceeded finite numerical range")
    return JointPortfolioResult(current_targets=current, reference_plan=_readonly(plan),
        incremental_forecasts=_readonly(increments), expected_pnl=_readonly(expected),
        variance=_readonly(variance), objective=_readonly(objective), optimality_gap=_readonly(gaps),
        decisions=_readonly(forecast.decisions), assets=_readonly(forecast.assets),
        horizons=_readonly(forecast.horizons),
        interval_seconds=_readonly(np.diff(np.r_[0, forecast.horizons])),
        risk_model=risk, config=config, beta=beta_values,
        beta_exposures=(None if beta_values is None else _readonly(
            (plan*np.where(allowed, beta_values, 0.)).sum(axis=2))),
        max_abs_beta=beta_bound, beta_available_at=beta_times,
        factor_constraints=factor_constraints,
        factor_names=None if resolved_factors is None else resolved_factors.factor_names,
        factor_loadings=None if resolved_factors is None else resolved_factors.loadings,
        factor_exposures=None if exposures is None else _readonly(exposures),
        factor_lower=None if resolved_factors is None else resolved_factors.lower,
        factor_upper=None if resolved_factors is None else resolved_factors.upper,
        factor_available_at=None if resolved_factors is None else resolved_factors.available_at,
        factor_lower_slack=None if exposures is None else _readonly(exposures-resolved_factors.lower),
        factor_upper_slack=None if exposures is None else _readonly(resolved_factors.upper-exposures))
