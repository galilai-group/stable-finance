# Constrained target tracking and stock-level execution attribution

These additions extend the existing portfolio and trade stages. Portfolio construction
remains frictionless. The trading decision consumes one already-combined target,
the actual marked holding and irrevocable pending orders. Execution continues to
use the existing models and physical-share cash ledger. No predictor is fitted.

## One-decision trading objective

For current marked actual weights `a`, pending-order weights `p`, and a supplied
net target `t`, choose planned committed holdings `w` by minimizing

```text
gamma/2 * (w - t)' Sigma_5m (w - t)
    + sum_i c_i * abs(w_i - a_i - p_i)
```

The caller sends only `w - a - p`. All weights use the current decision's NAV
and accounting prices. Pending orders are commitments, not filled positions.
At the next decision the caller supplies newly marked actual and pending
holdings rather than recursively treating the previous plan as a fill.

```python
import stable_finance as sf

decision = sf.solve_constrained_rebalance(
    net_target, covariance_5m,
    previous=actual_marked_weights,
    pending=pending_marked_weights,
    cost=known_one_way_cost,
    tradable=current_quote_is_valid,
    risk_aversion=5.0,
    beta=known_beta,
    constraints=sf.RebalanceConstraints(max_abs_beta=0.10),
)
if decision.status == "infeasible":
    raise RuntimeError(decision.message)
new_order_weights = decision.weights - actual_marked_weights - pending_marked_weights
```

The solver returns a read-only plan, its tracking variance, estimated new-order
cost, planned turnover, primal residual and independent convex optimality gap.
It does not return fills. An infeasible result has `weights=None`; numerical
failure raises separately. Its inputs are one asset vector and an explicitly
aligned covariance matrix, so it can be used with any existing execution loop.
Install `stable-finance[joint]` for the optional constrained numerical backend.

`Sigma_5m` explicitly prices five-minute target-tracking error. A constrained
multi-horizon target does not uniquely identify its original alpha objective:
this tracking policy is not claimed to solve the original joint alpha-risk-cost
problem. No covariance is averaged across horizons. Costs in the objective are
decision-time estimates; only observed fills debit cash, once, downstream.

Default planned books are dollar-neutral, with gross at most one and absolute
name weight at most 0.1. Beta is constrained only when explicitly requested. The generic function permits other explicitly
supplied limits. Untradable assets stay at their current committed weights.
If that makes the constraints infeasible, the solver returns an explicit
infeasible status. It never silently relaxes limits or rescales the solution.
Numerical failure is distinct from demonstrated infeasibility.

The constraints apply to the decision-price planned book. Partial fills, latency,
price changes and costs can make realized holdings cross a planned limit; the
report measures this separately. Forced end-of-session liquidation remains the
ledger's hard action and bypasses the tracking policy.

## Per-stock attribution

Use the existing physical ledger to allocate these additive quantities by stock:

```text
gross P&L_i = sum_t shares[t,i] * (price[t+1,i] - price[t,i])
slippage_i = sum_t filled_shares[t,i] * (fill_price[t,i] - mark_price[t,i])
commission_i = sum_t rate[t,i] * abs(filled_shares[t,i] * fill_price[t,i])
net P&L_i = gross P&L_i - slippage_i - commission_i
turnover_i = sum_t abs(fill_notional[t,i]) / initial_equity
```

Closing fills are included exactly once. Slippage can be negative. Decision
shortfall is diagnostic and is not added as another accounting charge. All
stock totals must reconcile to the original ledger's portfolio totals.

The change in time-weighted mean absolute portfolio beta also has an exact
signed allocation. Let `B0` and `B1` be reference and candidate portfolio beta
at one held interval, and let `s = (abs(B0)-abs(B1))/(B0-B1)` when the denominator
is nonzero, otherwise zero. Stock `i` contributes
`s * beta_i * (w0_i - w1_i)` to the reduction in absolute beta. These signed
contributions sum to `abs(B0)-abs(B1)` and may be negative. Apply the same exact
interval durations before averaging; do not sum absolute stock beta changes.

Reports also locate trade-direction reversals and activity sent when the target
is on its declared beta boundary. Boundary status uses the declared numerical
tolerance, not a newly optimized trading threshold. These are descriptive
accounting allocations and event counts, not causal evidence about the strategy.

```python
stocks = sf.execution_attribution(ledger, price_panel, beta=known_beta, beta_bound=0.10)
change = sf.paired_execution_attribution(
    reference_ledger, candidate_ledger, price_panel, beta=known_beta, beta_bound=0.10,
)
```

The reporter requires physical share/fill, commission, NAV, target and order
records; it cannot reconstruct transaction costs from a target-weight array
alone. It rejects multiple distinct fill records sharing one timestamp when
the ledger stores only one aggregate fill price. The fixed constant-delay replay
has one order per arrival and is supported. It does not invent a missing fill
price or distribute an unexplained portfolio charge across stocks.

For an end-to-end supplied-return objective, use the opt-in
[native backtest](native_return_risk_backtest.md). For additional signed
factor budgets, both solvers accept the array interface described in the
[generic factor guide](generic_factor_constraints.md). Historical empirical
replay artifacts are separate from the deterministic package tests.
