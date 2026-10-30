# Multi-horizon additions to the existing portfolio stage

Base: `harshitaggarwal01/stable-finance-harshit`, branch
`portfolio-signal-transforms`, commit
`7e81381c58f8ae97343be9991c64677d2ff9ff36`.

This extension reuses the base's `ForwardReturns`, `PortfolioWeights`,
`mean_variance_target`, selection/weighting rules, trades, execution and
`run_backtest`. Fixed mixing and frictionless joint construction introduce no
forecast training, cost model or experiment grid. The separately opt-in
[original-objective backtest](native_return_risk_backtest.md) supplies a physical
cash/share ledger for actual-state execution.

## Fixed contemporaneous target mix

The base already constructs a separate book for every supplied horizon.
The new helper combines those existing targets at the same decision:

```python
from stable_finance import combine_horizon_targets, mean_variance_target

components = mean_variance_target(forecast, covariance, risk_aversion=5.0)
target = combine_horizon_targets(
    components,
    capital_weights=[0.5, 0.5],
    target_horizon_seconds=300,
)
```

`components` has native shape `(D,N,H)`; the output has shape `(D,N,1)`.
For each decision, `target = sum_h alpha_h * components_h`. Capital fractions
are fixed ex ante, nonnegative and sum to one; the default is equal. Output
arrays are copied read-only. All component weights must be finite; supply an
explicit zero book if some reserved capital should remain cash. The helper
never changes the fractions, redistributes missing capital or rescales the
result to restore gross exposure after opposing positions cancel.

This convex combination preserves any common convex constraints already
satisfied by all component books. It does not add position limits to the
base's unconstrained mean-variance constructor. Native gross and risk-aversion
choices remain unchanged.

This is a **static mix of simultaneous current targets**. It does not infer
staggered vintages, delayed formation, forecast decay or source-horizon
holding periods.

`target_horizon_seconds` explicitly declares the downstream evaluation
interval. Score with matching, preferably nonoverlapping one-step outcomes
on the same decisions/assets. Do not average the original overlapping
multi-horizon outcome labels into P&L.

## Optional experimental joint target

`construct_joint_portfolio` is a separate experimental method. Its
`JointReturnPanel` explicitly declares cumulative expected payoffs on a
common decision-price basis:

```text
Y_h(t) = (P[t + delay + h] - P[t + delay]) / P[t]
q_1 = Y_h1; q_k = Y_hk - Y_h(k-1)
max_x  E[q]'x - risk_aversion/2 * x'Omega x
```

`Omega` spans assets and disjoint intervals. `JointRiskModel` carries the
matching asset/horizon axes, entry delay and risk-information cutoff.
`estimate_joint_risk` estimates it separately from fully mature historical
paths, with shrinkage after payoff differencing. It trains no predictor.
The native forward-VWAP labels are not implicitly converted to this payoff
basis; the caller must provide compatible forecasts and risk.

`JointPortfolioConstraints` defaults to dollar neutrality, gross at most
one and absolute name weight at most 0.1 for each reference-plan stage.
These are declared joint-model choices, not retroactive changes to the
base's other constructors. The result's `current_targets` is an ordinary
native `PortfolioWeights`; only its first stage is current. Later stages
are reference-share diagnostics, never future orders. The output horizon
is the first input endpoint. Replanning and actual holding clocks must be
compatible with that first interval.

Cross-interval covariance can make later forecasts affect the current
target. With block-diagonal risk and separate stage constraints, later
forecasts correctly have no such effect. Numerical correctness does not
establish an empirical advantage; that requires separate evidence with the
supplied forecasts, estimated risk and execution assumptions.

Install `stable-finance[joint]` for the binding constrained-QP backend.
OSQP is loaded only when that solve is needed; basic native construction
and fixed mixing do not require it. Exact free and zero solutions also
remain available without OSQP.

An optional supplied beta vector can also constrain every reference-plan
stage directly inside the same solve:

```python
plan = construct_joint_portfolio(
    forecast,
    risk=risk,
    config=JointPortfolioConfig(risk_aversion=5.0),
    beta=known_beta,
    max_abs_beta=0.10,
    beta_available_at=historical_cutoff,
)
```

Beta accepts `(N,)`, `(D,N)` or explicit joint-order `(D,K,N)` values. Its
availability must be no later than the decision, and coefficients for every
eligible position must be finite. `abs(beta @ x_k) <= max_abs_beta` is a
linear feasibility condition, not a projection or turnover penalty. A
single-endpoint panel uses this same solver for constrained single-horizon
construction; existing native closed-form constructors are unchanged.
The result separately records the supplied beta, stage exposures and bound.
Actual marked holdings can drift across a target bound after execution.
Generic multiple-factor bounds use the same solver; see
[the factor constraint guide](generic_factor_constraints.md).

## Native downstream handoff

Pass the single resulting target to native `run_backtest` at the
`PortfolioWeights` injection point, with matching single-interval realized
returns. This bypasses predictor fitting and portfolio construction.
It also bypasses cost-aware `rebalance`: the example verifies the native
execution-cost path only. To apply a trades policy, call it explicitly
after netting and declare a common-interval tracking risk model. A mixed
target alone does not identify the original constrained objective.

The examples set `rebalance="every_decision"`; source
horizons are never treated as independent realizations. Opposing targets
cancel before the existing execution model sees them, so costs are charged
on the net book once.

```sh
python examples/multi_horizon.py --output-dir /absolute/local/fixed-demo
python examples/joint_multi_horizon.py --output-dir /absolute/local/joint-demo
python -m pytest -q
```

These deterministic examples verify interfaces, algebra and native cost
accounting. They use supplied analytical forecasts, fit no prediction model,
and make no profitability claim. No external checkout or Market-1T data is
required for these examples.
