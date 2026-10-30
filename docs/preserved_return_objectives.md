# Preserve supplied return objectives through the trades stage

The portfolio stage continues to construct frictionless targets. The trades
stage can now consume the original return-risk objective explicitly, rather
than infer its marginal return from an already constrained target.

## Objective and state

For incremental expected payoffs `q` with interval-major shape `(K,N)` and the
matching full covariance `Omega`, solve at each decision

```text
max_X q'X - gamma/2 * X'Omega*X
      - sum_i c_i * abs(X[0,i] - actual_i - pending_i)
```

`K=1` preserves the original single-horizon forecast and risk covariance.
`K>1` retains every supplied forecast increment and all cross-interval risk
blocks, and reoptimizes every future reference block at the current decision.
It does not freeze future blocks at their frictionless targets or substitute
an effective five-minute alpha approximation.

```python
import stable_finance as sf

# The existing joint result already retains the original increments and risk.
decision = sf.solve_return_risk_rebalance(
    joint_result.incremental_forecasts[row],
    joint_result.risk_model.covariance,
    previous=actual_marked_weights,
    pending=pending_marked_weights,
    cost=known_current_one_way_cost,
    tradable=current_quote_is_valid,
    risk_aversion=5.0,
    beta=known_beta,
    max_abs_beta=0.10,
)
if decision.status == "infeasible":
    raise RuntimeError(decision.message)
new_order_weights = decision.weights - actual_marked_weights - pending_marked_weights
```

The caller must preserve asset order, payoff basis, decision-known risk/factor
availability and the current horizon prefix. A single interval accepts a
forecast vector `(N,)` and matching covariance `(N,N)`. A full plan accepts
increments `(K,N)` and covariance `(K*N,K*N)` in interval-major order. This
function trains nothing and never infers a forecast from capped target weights.
Results include the read-only current book and full reference plan, modeled
payoff/risk/current-cost components, constraint residual and optimality gap.

Only `X[0]` is emitted as the planned current book. Later reference blocks
are diagnostics, never queued orders or realized P&L. Costs in this objective
concern the one new current net order. No future spread, exit charge or future
tradability is invented. Forced closing and observed execution costs remain
the unchanged ledger's responsibilities.

Actual and irrevocable pending holdings are marked at the current decision's
prices and NAV. The new order is `X[0] - actual - pending`. A previous plan is
not treated as a fill and existing pending-order costs are not charged again.
Current missing quotes or costs freeze the corresponding CURRENT committed
holding. Original future reference availability is preserved.

Each reference stage retains the supplied gross, name, net and beta limits.
The fixed experiment uses gross at most 1, absolute name weight at most 0.1,
dollar neutrality and absolute frozen cohort-proxy beta at most 0.1. Frozen
holdings can make the problem infeasible; this is explicit, without clipping,
rescaling, relaxation or hidden cash fallback. Numerical failure is separate
from mathematical infeasibility. Accepted plans carry independent primal and
convex optimality certificates.

## Equivalent full objective handoff

For any saved frictionless reference plan `T`, the same objective can be
recentered on that plan if its full residual gradient is retained:

```text
g = gamma * Omega * T - q
min_X gamma/2*(X-T)'Omega*(X-T) + g'(X-T)
      + current_cost(X[0] - actual - pending)
```

The implementation passes `q` and `Omega` directly, avoiding subtraction of
nearly equal terms. The identity is tested to confirm that active target
constraints do not erase the original return reward. Current targets alone
remain insufficient to reconstruct the full constrained objective.

The existing `solve_constrained_rebalance` remains an explicitly different
target-tracking policy for users who bring weights without return forecasts.
It remains available for users who supply target weights without an original
forecast objective.

## Native handoff and factor budgets

Use `run_backtest(forecast, prices, objective=ReturnRiskBacktestSpec(...))`
for the native actual-share/pending-order loop; see the
[native integration guide](native_return_risk_backtest.md). It runs entirely
from the current package and does not load a preserved checkout.

The optional generic factor arrays impose signed lower/upper bounds on every
active reference stage, intersecting any supplied legacy beta constraint.
The [generic factor guide](generic_factor_constraints.md) documents labelled
sources, causality, missing coefficients and infeasibility handling.

The deterministic examples and tests verify algebra, constraints and physical
accounting. They do not establish empirical profitability. Historical
Market-1T replay artifacts remain separate from this package contribution.
