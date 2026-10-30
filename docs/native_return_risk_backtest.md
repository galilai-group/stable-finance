# Native original-objective backtests

`run_backtest` now has an explicit original-return-objective path. Without its
`objective` keyword, the existing forecast/weight/order paths, configuration
defaults, holding conventions and flat report schema remain unchanged.

## Public entry point

```python
report = sf.run_backtest(
    forecast_panel,              # JointReturnPanel, one or several endpoints
    price_marks,                 # PricePanel, complete positive accounting marks
    market=observed_quotes,      # MarketData on the entire accounting-mark grid
    objective=sf.ReturnRiskBacktestSpec(
        risk=known_risk,
        risk_aversion=5.0,
        constraints=sf.JointPortfolioConstraints(),
        beta=known_beta,
        max_abs_beta=0.10,
        beta_available_at=known_beta_cutoff,
        stage_counts=known_close_prefixes,
        strategy_end=close_send_time,
    ),
    execution_state=sf.BacktestExecutionState(equity=100_000.0),
    config=sf.BacktestConfig(cost_model="cross", efq=0.7,
                             annualization="decisions"),
    with_series=True,
)
trace = report.objective_trace
```

Single-horizon construction uses a `JointReturnPanel` with one endpoint and
the matching `JointRiskModel`. This explicit contract avoids reinterpreting
native forward-VWAP forecasts as decision-price payoffs. Joint inputs use
several cumulative endpoints; the engine differences them exactly once and
retains the full interval-major covariance. Assets, endpoints, payoff basis,
entry delay, risk cutoff and supplied beta availability must match.

The spec owns supplied risk, gamma and position constraints. It bypasses
automatic risk estimation, causal gross calibration, signal transforms and
selection in the ordinary pipeline. Unsupported conflicting configuration
options are rejected rather than silently applied to a different objective.
No predictor is fitted in this path.

## Execution state and accounting

The native physical ledger reuses the package's existing `execute`, `Orders`
and `MarketData` implementations. Its cash/share algorithm is ported from the
previously verified local execution adapter; it has no runtime import or path
dependency on that older checkout.

At each decision, same-time arrivals settle before the solver receives the
actual marked weights and irrevocable pending-order weights. Only the first
reference block is sent as a physical net order. Pending positions are never
treated as fills, and previously selected plans do not replace observed state.
Each subsequent decision reoptimizes every supplied future reference block.
No future reference position is queued or added to realized P&L.

Entry delay is the forecast/risk contract's exact number of seconds; the price
grid must contain each arrival time. It cannot silently round five seconds to
one minute. Current raw quotes determine planning costs, and arrival quotes
determine native fills. Estimated planning costs are not charged to NAV. The
ledger charges actual fills once, tracks cash and shares, and closes at the
declared end unless liquidation is explicitly disabled.

`BacktestExecutionState` can start with marked actual weights and equity.
External nonzero initial pending orders require arrival records and are
explicitly unsupported by this initial state contract. Pending orders created
during the backtest are fully represented and feed the next decision normally.
Missing current quotes freeze current committed positions; infeasibility is
explicit. Price/NAV drift and delayed fills can move actual exposure beyond a
planned bound, which remains a separate diagnostic.

Physical P&L uses consecutive accounting prices, not overlapping forecast
labels. The second argument is therefore `PricePanel` in this explicit mode,
not the legacy `ForwardReturns` scoring panel. Long-horizon cumulative labels
are never multiplied into P&L at every short decision. The trace retains
states, actual fills, complete reference plans and numerical certificates.
`BacktestReport.as_dict()` omits this potentially large trace without copying it.

In this explicit physical mode, `annualization="decisions"` reports each
nonoverlapping price-mark interval; `"daily"` folds currency P&L to each day's
starting NAV and preserves daily compounding. Mean return, cost and turnover
refer to that selected report period. They must not be compared to legacy
decision-grid averages with a different denominator without conversion.

## Upstream alignment

The integration base is the working-branch snapshot `7e81381c58f8ae97343be9991c64677d2ff9ff36`.
Official main was checked at `45e93beb2bf38569dd4c71c9557327ad775f3f41`.
Its updates include drift holding, order retry/cancel, array EFQ and event-time
backtests; the working branch also replaces `hold` with horizon rebalancing.
The addition preserves those source APIs and defaults; only the explicit
`objective` path selects the new physical execution convention.

Upstream drift positions are fixed-reference notionals, not automatically
weights divided by current NAV. This explicit physical mode keeps that
distinction visible. Its report carries an execution trace; legacy modes keep
their existing conventions. The integration tests exercise the native example
and the unchanged upstream paths in one checkout.

Run the self-contained English example with:

```sh
PYTHONPATH=src python examples/native_return_risk_backtest.py
```

It supplies deterministic forecasts and risk, exercises real native fills and
pending orders, and runs both single and joint objectives. It requires no
Market-1T download, training, preserved checkout or remote service. Generated
example artifacts go to a user-selected output directory or the local cache.

Optional `factor_constraints=FactorExposureConstraints(...)` in the spec
propagates the same labelled factor budgets through targets, every active
reference stage and actual-state trading. See the
[generic factor guide](generic_factor_constraints.md).
