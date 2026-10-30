"""Opt-in native backtesting of an explicitly supplied return-risk objective.

This mode uses current-NAV physical holdings and its own price-mark grid. It
never estimates a predictor or risk model and never scores overlapping future
reference payoffs as P&L. The legacy grid harness remains the default path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.config import DEFAULT_BACKTEST, BacktestConfig
from stable_finance.evaluation.physical import (
    PhysicalExecutionConfig, PricePanel, execute_portfolio_targets,
    _aligned_indices, _time_deltas_seconds,
)
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.factors import (
    FactorExposureConstraints, ResolvedFactorConstraints,
)
from stable_finance.evaluation.portfolio.joint import _beta_constraint
from stable_finance.evaluation.portfolio.joint_constraints import JointPortfolioConstraints
from stable_finance.evaluation.portfolio.joint_risk import (
    JointReturnPanel, JointRiskModel, _clock, _same_clock,
)
from stable_finance.evaluation.scoring.sharpe import sharpe_ratio
from stable_finance.evaluation.trades.objective import solve_return_risk_rebalance

__all__ = ["BacktestExecutionState", "ReturnRiskBacktestSpec", "ReturnRiskTrace",
           "run_return_risk_backtest"]


def _readonly(values):
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _nonnegative(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} scalar")


@dataclass(frozen=True)
class BacktestExecutionState:
    """Starting marked equity and actual weights, not a previous target.

    Unscheduled external pending weights cannot be initialized: arrival times,
    share quantities and fill rules would be missing. Runtime pending orders
    are generated and tracked by the native physical ledger instead.
    """

    equity: float = 1.0
    weights: ArrayLike | None = None
    pending_weights: ArrayLike | None = None

    def __post_init__(self):
        _nonnegative(self.equity, "equity", positive=True)
        for name in ("weights", "pending_weights"):
            value = getattr(self, name)
            if value is None:
                continue
            raw = np.asarray(value)
            if raw.dtype.kind not in "iuf" or raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
                raise ValueError(f"{name} must be a nonempty finite asset vector")
            object.__setattr__(self, name, _readonly(np.asarray(raw, dtype=float)))
        if self.pending_weights is not None and np.any(self.pending_weights != 0):
            raise ValueError("nonzero initial pending_weights need arrival records and are unsupported")
        if (self.weights is not None and self.pending_weights is not None
                and self.weights.shape != self.pending_weights.shape):
            raise ValueError("initial actual and pending weights must share the asset axis")


@dataclass(frozen=True)
class ReturnRiskBacktestSpec:
    """Explicit original objective, constraints and causal risk/factor inputs.

    ``risk`` owns the full interval-major covariance and its availability.
    ``stage_counts`` optionally supplies a decision-known prefix per origin;
    omitted means all supplied stages, without inspecting future outcomes.
    This spec, rather than legacy estimation/weighting config fields, owns
    risk, risk aversion and position constraints in this opt-in mode.
    """

    risk: JointRiskModel
    risk_aversion: float = 5.0
    constraints: JointPortfolioConstraints = JointPortfolioConstraints()
    beta: ArrayLike | None = None
    max_abs_beta: float | None = None
    beta_available_at: ArrayLike | None = None
    stage_counts: ArrayLike | None = None
    strategy_end: object | None = None
    liquidate: bool = True
    factor_constraints: FactorExposureConstraints | None = None

    def __post_init__(self):
        if not isinstance(self.risk, JointRiskModel):
            raise TypeError("risk must be an explicit JointRiskModel")
        _nonnegative(self.risk_aversion, "risk_aversion", positive=True)
        if not isinstance(self.constraints, JointPortfolioConstraints):
            raise TypeError("constraints must be JointPortfolioConstraints")
        if (self.factor_constraints is not None
                and not isinstance(self.factor_constraints, FactorExposureConstraints)):
            raise TypeError("factor_constraints must be FactorExposureConstraints")
        if not isinstance(self.liquidate, (bool, np.bool_)):
            raise TypeError("liquidate must be boolean")
        if (self.beta is None) != (self.max_abs_beta is None):
            raise ValueError("beta and max_abs_beta must be supplied together")
        if self.max_abs_beta is not None:
            _nonnegative(self.max_abs_beta, "max_abs_beta")
        if self.beta is None and self.beta_available_at is not None:
            raise ValueError("beta_available_at requires a beta constraint")
        if self.beta is not None:
            raw = np.asarray(self.beta)
            if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2, 3) or not np.isfinite(raw).all():
                raise ValueError("beta must contain finite declared factor coefficients")
            object.__setattr__(self, "beta", _readonly(np.asarray(raw, dtype=float)))
        if self.beta_available_at is not None:
            object.__setattr__(self, "beta_available_at", _readonly(self.beta_available_at))
        if self.stage_counts is not None:
            raw = np.asarray(self.stage_counts)
            if raw.dtype.kind not in "iu" or raw.ndim not in (0, 1) or np.any(raw < 1):
                raise ValueError("stage_counts must contain positive integers, not booleans or floats")
            if np.any(raw > len(self.risk.horizons)):
                raise ValueError("stage_counts cannot exceed the supplied forecast endpoints")
            object.__setattr__(self, "stage_counts", _readonly(raw))


@dataclass(frozen=True)
class ReturnRiskTrace:
    """Auditable native objective/state handoff; omitted from flat report rows.

    ``reference_plans`` is `(D,K,N)` with zero padding beyond each declared
    prefix. Certificates are original-objective solver results. Their future
    modeled utility is never used to calculate the report's realized return.
    """

    forecast: JointReturnPanel
    spec: ReturnRiskBacktestSpec
    initial_state: BacktestExecutionState
    final_state: BacktestExecutionState
    ledger: object
    targets: PortfolioWeights
    reference_plans: np.ndarray
    certificates: tuple
    planning_cost: np.ndarray
    tradable: np.ndarray
    stage_counts: np.ndarray
    notes: tuple[str, ...]
    report_period: str = "physical_marks"
    factor_constraints: ResolvedFactorConstraints | None = None

    @property
    def report_metadata(self) -> dict:
        """Scalar authoritative settings for flat rows, without copying arrays."""
        limits = self.spec.constraints
        result = dict(input_mode="return_risk_physical",
            objective_risk_aversion=float(self.spec.risk_aversion),
            objective_gross_limit=float(limits.gross_limit),
            objective_max_weight=float(limits.max_weight),
            objective_dollar_neutral=bool(limits.dollar_neutral),
            objective_max_abs_beta=None if self.spec.max_abs_beta is None else float(self.spec.max_abs_beta),
            objective_risk_estimator=self.spec.risk.estimator,
            objective_risk_source=self.spec.risk.risk_source,
            initial_equity=float(self.initial_state.equity),
            entry_delay_seconds=int(self.forecast.entry_delay_seconds),
            report_period=self.report_period)
        if self.factor_constraints is not None:
            result.update(objective_factor_count=len(self.factor_constraints.factor_names),
                objective_factor_names=tuple(self.factor_constraints.factor_names.tolist()))
        return result


def _validate_config(config, objective, state):
    if not isinstance(config, BacktestConfig):
        raise TypeError("config must be BacktestConfig")
    if config.cost_model not in ("mid", "cross"):
        raise ValueError("return-risk physical backtesting supports cost_model='mid' or 'cross'")
    if config.rebalance != "every_decision" or getattr(config, "hold", None) is not None:
        raise ValueError("return-risk mode replans every decision; rebalance/hold overrides are unsupported")
    if getattr(config, "holding", "rebalanced") != "rebalanced":
        raise ValueError("holding overrides use a different position convention; return-risk mode owns physical state")
    for name in ("universe", "selection", "weighting", "risk_model", "shrinkage", "max_iter"):
        if getattr(config, name) != getattr(DEFAULT_BACKTEST, name):
            raise ValueError(f"{name} is owned by the supplied objective; this legacy override is unsupported")
    if any(getattr(config, name) is not None for name in ("alpha", "alpha_ic", "neutralize", "impact", "execution_days")):
        raise ValueError("return-risk mode consumes the original objective; transforms and impact/schedule overrides are unsupported")
    if config.gross_exposure not in (DEFAULT_BACKTEST.gross_exposure, objective.constraints.gross_limit):
        raise ValueError("gross_exposure conflicts with the supplied objective constraints")
    if config.equity is not None and config.equity != state.equity:
        raise ValueError("config.equity and execution_state.equity must agree")


def _interval_days(prices, decisions, days):
    """Calendar/explicit labels for mark intervals, without outcome lookahead."""
    times = np.asarray(prices.times)
    if days is not None:
        labels = np.asarray(days)
        if labels.ndim != 1:
            raise ValueError("days must be one-dimensional")
        if len(labels) == len(times):
            result = labels[:-1]
        elif len(labels) == len(times)-1:
            result = labels
        elif len(labels) == len(decisions):
            rows = np.maximum(np.searchsorted(decisions, times[:-1], side="right")-1, 0)
            result = labels[rows]
        else:
            raise ValueError("days must label forecast decisions, price marks or mark intervals")
    elif times.dtype.kind == "M":
        result = times[:-1].astype("datetime64[D]").astype(str)
    elif times.dtype == object and all(isinstance(value, datetime) for value in times):
        result = np.asarray([value.date().isoformat() for value in times[:-1]])
    else:
        # Numeric clocks use UTC calendar days; pass explicit days for an
        # exchange-local calendar or a relative synthetic time grid.
        result = times[:-1].astype("datetime64[s]").astype("datetime64[D]").astype(str)
    result = np.asarray(result).astype(str)
    changes = np.r_[True, result[1:] != result[:-1]]
    if len(set(result.tolist())) != int(changes.sum()):
        raise ValueError("a day label cannot recur after a different day's intervals")
    return result


def _market_inputs(prices, market, half_spread, config, decision_rows):
    if market is not None and half_spread is not None:
        raise ValueError("pass market or half_spread, not both")
    if market is None:
        columns = {"mid": prices.values}
        if half_spread is not None:
            spread = np.asarray(half_spread, dtype=float)
            if spread.shape != prices.values.shape:
                raise ValueError("half_spread must align with every price mark and asset")
            columns["half_spread"] = spread
        market = MarketData(columns, prices.times, prices.assets)
    if not isinstance(market, MarketData):
        raise TypeError("market must be MarketData")
    if (not np.array_equal(market.assets, prices.assets)
            or not np.array_equal(market.decisions, prices.times)):
        raise ValueError("market must align with every price mark and asset")
    shape = (len(decision_rows), len(prices.assets))
    if config.cost_model == "mid":
        return market, np.zeros(shape), np.ones(shape, dtype=bool)
    market.require("half_spread", purpose="original-objective trading and execution")
    spread = market.get("half_spread")[decision_rows]
    valid = np.isfinite(spread) & (spread >= 0)
    costs = np.where(valid, config.efq*spread, np.inf)
    return market, costs, valid


class _ObjectivePolicy:
    def __init__(self, forecast, spec, prefixes, factors, cost, tradable, linear_factors=None):
        self.forecast, self.spec, self.prefixes = forecast, spec, prefixes
        self.factors, self.cost, self.tradable = factors, cost, tradable
        self.linear_factors = linear_factors
        self.increments = np.diff(forecast.values, axis=2,
            prepend=np.zeros((*forecast.values.shape[:2], 1))).transpose(0, 2, 1)
        self.results, self.plans = [], []

    def validate(self, n_decisions, n_assets):
        if self.forecast.values.shape[:2] != (n_decisions, n_assets):
            raise ValueError("objective policy axes differ from the execution book")

    def _solve(self, row, previous, pending, *, frictionless=False):
        n = len(self.forecast.assets)
        k = int(self.prefixes[row])
        constraint = self.spec.constraints
        factor_options = {}
        if self.linear_factors is not None:
            factor_options = dict(factor_loadings=self.linear_factors.loadings[row, :k],
                factor_lower=self.linear_factors.lower[row, :k],
                factor_upper=self.linear_factors.upper[row, :k],
                factor_names=self.linear_factors.factor_names)
        return solve_return_risk_rebalance(self.increments[row, :k],
            self.spec.risk.covariance[:k*n, :k*n], previous=previous, pending=pending,
            cost=0. if frictionless else self.cost[row],
            risk_aversion=self.spec.risk_aversion,
            gross_limit=constraint.gross_limit, max_weight=constraint.max_weight,
            dollar_neutral=constraint.dollar_neutral,
            beta=None if self.factors is None else self.factors[row, :k],
            max_abs_beta=self.spec.max_abs_beta,
            tradable=None if frictionless else self.tradable[row], **factor_options)

    def plan(self, target, previous, pending, decision_index, n_decisions):
        self.validate(n_decisions, len(target))
        if (isinstance(decision_index, (bool, np.bool_))
                or not isinstance(decision_index, (int, np.integer))
                or not 0 <= decision_index < n_decisions):
            raise ValueError("decision_index must identify a current objective row")
        result = self._solve(decision_index, previous, pending)
        if result.status != "solved" or result.weights is None:
            raise RuntimeError(f"return-risk decision {decision_index} is {result.status}: {result.message}")
        full = np.zeros(self.increments.shape[1:])
        full[:int(self.prefixes[decision_index])] = result.reference_plan
        self.results.append(result)
        self.plans.append(full)
        return result.weights


def _score(ledger, interval_days, config, with_series):
    """Compound physical accounting intervals while preserving gross-cost=net."""
    from stable_finance.evaluation.backtest import PnLSeries

    gross, costs, net = ledger.gross_returns, ledger.costs, ledger.net_returns
    turnover = np.abs(ledger.trade_notional[:-1]).sum(axis=1)/ledger.pretrade_nav[:-1]
    turnover[-1] += np.abs(ledger.trade_notional[-1]).sum()/ledger.pretrade_nav[-2]
    if config.annualization == "daily":
        edges = np.flatnonzero(np.r_[True, interval_days[1:] != interval_days[:-1], True])
        g, c, t = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            wealth = ledger.nav[a:b]/ledger.nav[a]
            g.append(float(wealth @ gross[a:b]))
            c.append(float(wealth @ costs[a:b]))
            t.append(float(wealth @ turnover[a:b]))
        gross, costs, turnover = np.asarray(g), np.asarray(c), np.asarray(t)
        net = gross-costs
        labels, annual = interval_days[edges[:-1]], 252.
    else:
        labels = np.asarray(ledger.times[:-1]).astype(str)
        annual = 252.*len(net)/len(np.unique(interval_days))
    series = PnLSeries(_readonly(labels), _readonly(gross), _readonly(net), _readonly(costs)) if with_series else None
    return gross, costs, net, turnover, annual, series


def run_return_risk_backtest(forecast, prices, *, objective,
                             execution_state=None, market=None, half_spread=None,
                             config=DEFAULT_BACKTEST, days=None, with_series=False,
                             horizon_index=0):
    """Native opt-in backtest of an original single/joint return objective.

    Forecasts must explicitly declare decision-price payoffs through
    `JointReturnPanel`, including the single-endpoint case. Realized P&L comes
    from `PricePanel` marks, not a panel of overlapping forecast outcomes.
    The objective owns risk, gamma, position limits and stage prefixes. Legacy
    selection, fitting and transformation overrides are rejected in this mode.
    """
    from stable_finance.evaluation.backtest import BacktestReport

    if not isinstance(forecast, JointReturnPanel):
        raise TypeError("return-risk mode requires an explicit JointReturnPanel, including K=1")
    if not isinstance(prices, PricePanel):
        raise TypeError("return-risk mode requires PricePanel as its second argument")
    if not isinstance(objective, ReturnRiskBacktestSpec):
        raise TypeError("objective must be ReturnRiskBacktestSpec")
    if not isinstance(config, BacktestConfig):
        raise TypeError("config must be BacktestConfig")
    if (isinstance(horizon_index, (bool, np.bool_))
            or not isinstance(horizon_index, (int, np.integer)) or horizon_index != 0):
        raise ValueError("return-risk mode emits only the first current stage; horizon_index must be 0")
    state = execution_state
    if state is None:
        state = BacktestExecutionState(equity=config.equity or 1.)
    if not isinstance(state, BacktestExecutionState):
        raise TypeError("execution_state must be BacktestExecutionState")
    _validate_config(config, objective, state)
    for name in ("assets", "horizons"):
        if not np.array_equal(getattr(forecast, name), getattr(objective.risk, name)):
            raise ValueError(f"forecast and supplied risk {name} must agree exactly")
    if (forecast.return_basis != "decision_price_payoff"
            or objective.risk.return_basis != "decision_price_payoff_increments"
            or forecast.entry_delay_seconds != objective.risk.entry_delay_seconds):
        raise ValueError("forecast and risk must share the explicit payoff basis and execution delay")
    if not np.array_equal(forecast.assets, prices.assets):
        raise ValueError("forecast and physical price assets must agree exactly")
    if not np.isfinite(forecast.values).all() or min(forecast.values.shape) == 0:
        raise ValueError("the supplied forecast must be nonempty and finite")
    for decision in forecast.decisions:
        objective.risk.assert_available(decision)
    _, f_kind = _clock(forecast.decisions, "forecast decisions")
    _, p_kind = _clock(prices.times, "price times")
    _same_clock(f_kind, p_kind)
    rows = _aligned_indices(prices.times, forecast.decisions, "forecast decisions")
    d, n, k = forecast.values.shape
    for name in ("weights", "pending_weights"):
        value = getattr(state, name)
        if value is not None and value.shape != (n,):
            raise ValueError(f"execution_state.{name} must follow the forecast asset axis")
    prefixes = np.full(d, k, dtype=np.int64)
    if objective.stage_counts is not None:
        raw = objective.stage_counts
        if raw.ndim == 0:
            prefixes[:] = int(raw)
        elif raw.shape == (d,):
            prefixes = np.asarray(raw, dtype=np.int64)
        else:
            raise ValueError("stage_counts must be scalar or one prefix per decision")
        if np.any(prefixes < 1) or np.any(prefixes > k):
            raise ValueError("stage_counts cannot exceed the supplied forecast endpoints")
    factors, _, _ = _beta_constraint(objective.beta, objective.max_abs_beta,
        objective.beta_available_at, forecast.decisions, np.ones((d, k, n), dtype=bool))
    linear_factors = None
    if objective.factor_constraints is not None:
        # Every name can be held by a current or future plan, including names
        # frozen by missing quotes; no unknown loading can be assumed zero.
        linear_factors = objective.factor_constraints.resolve(decisions=forecast.decisions,
            assets=forecast.assets, horizons=forecast.horizons)
    market, costs, tradable = _market_inputs(prices, market, half_spread, config, rows)
    interval_days = _interval_days(prices, forecast.decisions, days)
    mark_annualization = 252.*(len(prices.times)-1)/len(np.unique(interval_days))
    policy = _ObjectivePolicy(forecast, objective, prefixes, factors, costs, tradable, linear_factors)
    desired = np.zeros((d, n, 1))
    for row in range(d):
        result = policy._solve(row, np.zeros(n), np.zeros(n), frictionless=True)
        if result.status != "solved" or result.weights is None:
            raise RuntimeError(f"frictionless original objective is {result.status}: {result.message}")
        desired[row, :, 0] = result.weights
    targets = PortfolioWeights(desired, forecast.decisions, forecast.assets, forecast.horizons[:1])
    ledger = execute_portfolio_targets(targets, prices, market,
        config=PhysicalExecutionConfig(model=config.cost_model,
            delay_seconds=forecast.entry_delay_seconds, efq=config.efq),
        trade_policy=policy, initial_equity=state.equity, initial_weights=state.weights,
        strategy_end=objective.strategy_end, liquidate=objective.liquidate,
        cost_bps=0., periods_per_year=mark_annualization)
    if len(policy.results) != d:
        raise RuntimeError("every decision must retain its original-objective certificate")
    gross, cost, net, turnover, annual, series = _score(ledger, interval_days, config, with_series)
    durations = _time_deltas_seconds(prices.times)
    held = ledger.posttrade_weights[:-1]
    final = BacktestExecutionState(equity=float(ledger.posttrade_nav[-1]),
        weights=ledger.posttrade_weights[-1],
        pending_weights=ledger.pending_shares[-1]*prices.values[-1]/ledger.posttrade_nav[-1])
    trace = ReturnRiskTrace(forecast, objective, state, final, ledger, targets,
        _readonly(policy.plans), tuple(policy.results), _readonly(costs),
        _readonly(tradable), _readonly(prefixes),
        ("Explicit objective owns forecasts, risk, gamma and position limits; legacy default estimation fields are unused.",
         "Physical shares/cash are marked to current NAV; only the first reference block is executed.",
         "IC and transfer coefficient are not computed in this physical-price scoring mode.",
         "Turnover is mean physical filled notional per scoring period: per mark interval for decisions, per day for daily annualization.",
         "Numeric clocks use UTC calendar days unless explicit days are supplied."),
        report_period="daily" if config.annualization == "daily" else "physical_marks",
        factor_constraints=linear_factors)
    return BacktestReport(config=config, horizon=int(forecast.horizons[0]),
        information_coefficient=np.nan, information_coefficient_se=np.nan,
        transfer_coefficient=np.nan,
        sharpe_mid=sharpe_ratio(gross, periods_per_year=annual),
        sharpe_net=sharpe_ratio(net, periods_per_year=annual),
        mean_gross_return_bps=float(np.mean(gross)*1e4),
        mean_cost_bps=float(np.mean(cost)*1e4), mean_turnover=float(np.mean(turnover)),
        mean_gross_exposure=float(np.average(np.abs(held).sum(axis=1), weights=durations)),
        periods_per_year=float(annual), n_periods=len(net),
        mean_names_held=float(np.average((np.abs(held)>1e-12).sum(axis=1), weights=durations)),
        series=series, objective_trace=trace)
