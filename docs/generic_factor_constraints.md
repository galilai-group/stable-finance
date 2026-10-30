# Generic factor exposure constraints

The constrained single-endpoint (`K=1`) and joint portfolio paths, both trade
solvers, and the opt-in original-objective `run_backtest` accept the same
linear exposure budgets. Existing beta arguments remain supported. Omitting
generic factors preserves the old defaults, solver branches and flat report
metadata. Existing closed-form weighting rules are unchanged.

For each active planning stage `s`, the constraint is

```text
lower[s, f] <= sum_i loadings[s, i, f] * X[s, i] <= upper[s, f].
```

Each column can represent supplied market beta, industry membership, style,
or another linear factor. Budgets may be asymmetric, one-sided or exact
equalities; use `-np.inf` or `np.inf` for an open side. Net industry exposure
is linear; gross industry exposure and concentration/HHI are separate
diagnostics and are not implemented by these columns. No labels, coefficients,
normalization or risk model are inferred by this feature.

## Declare one labelled contract

```python
import numpy as np
import stable_finance as sf

factors = sf.FactorExposureConstraints(
    loadings=np.column_stack([beta, industry_membership, style]),  # (N, F)
    lower=[-0.10, -0.02, 0.00],
    upper=[ 0.10,  0.02, 0.05],
    assets=assets,
    factor_names=["proxy_beta", "industry_A", "style"],
    available_at=calibration_cutoff,
)

targets = sf.construct_joint_portfolio(
    forecast, risk=risk, factor_constraints=factors,
)
report = sf.run_backtest(
    forecast, price_marks,
    objective=sf.ReturnRiskBacktestSpec(risk, factor_constraints=factors),
    market=market,
    config=sf.BacktestConfig(cost_model="cross", efq=0.70),
)
```

`forecast` and `risk` retain the existing explicit `JointReturnPanel` and
`JointRiskModel` contracts. A one-endpoint panel is the constrained
single-horizon interface. `K>1` retains the full incremental expected payoff
and interval-major cross-stage covariance. Generic constraints add feasible
rows; they do not replace or reconstruct that objective from target weights.
The portfolio calculation remains frictionless; current trading costs remain
in the downstream trades/execution path.

## Axes and information timing

| Loadings | Required declared axes | Interpretation |
| --- | --- | --- |
| `(N,F)` | assets and factor names | Static, broadcast to requested decisions/stages |
| `(D,N,F)` | decisions | Time-varying, same across stages |
| `(K,N,F)` | horizons | Stage-specific, same across decisions |
| `(D,K,N,F)` | decisions and horizons | Fully specified |

A static `(N,F)` matrix can also declare decisions/horizons to attach varying
budgets. Bounds are scalar, `(F,)`, or the **complete declared leading axes**
plus `(F,)`. With both decisions and horizons declared, varying budgets are
`(D,K,F)`; a partial `(D,F)` budget is not silently broadcast as `(K,F)`.

Assets must match exactly, including order. Decision and horizon selection
requires exact source labels; there is no forward filling or automatic
reordering. Horizon labels are endpoint labels, not array positions. Three
dimensional arrays require an explicit leading axis, even when `D == K`.
Inputs are copied into immutable arrays.

`available_at` is scalar or follows all declared decision/stage axes. It is
the latest information time used in both loadings and budgets and must be no
later than the current decision. A future planning stage's factors must still
be known at the current decision. Omitting availability declares the input
known at each requested decision, as in the existing beta API; it does not
prove the caller's source was causal.

NaN coefficients are accepted only on explicitly hard-excluded coordinates
in portfolio construction. They are zeroed for multiplication after exclusion
is validated; the original contract preserves their missingness. Trade
solvers require coefficients on every potentially held name and any nonzero
frozen holding. Native backtesting allows every name to be held and therefore
requires complete supplied loadings. Missing historical industry labels must
remain unknown in the existing exposure report; they cannot become an
enforced industry budget without actual point-in-time inputs.

## Trades, execution state and infeasibility

The standalone trade solvers accept raw arrays from the same contract:

```python
result = sf.solve_return_risk_rebalance(
    incremental_forecasts, covariance,
    previous=actual_weights, pending=pending_weights, cost=current_cost,
    factor_loadings=factors.loadings,
    factor_lower=factors.lower, factor_upper=factors.upper,
    factor_names=factors.factor_names,
)
```

Raw arrays are for one decision: `(N,F)` or `(K,N,F)`, with bounds scalar,
`(F,)` or `(K,F)`. `solve_constrained_rebalance` accepts the single-stage
`(N,F)` version for its separate target-tracking objective. For labelled
time-varying sources, call `factors.resolve(...)` and pass the selected row.
Both trade functions are array interfaces; the labelled native backtest
performs asset/time validation before invoking them.

The current committed position is `actual + pending`; transaction costs
apply only to new net orders. An unavailable quote/cost freezes that name's
committed position. Its factor contribution remains in the constraint and
reduces the remaining budget. Future reference stages receive their own
complete factor rows. Only stage zero is submitted as current orders; the
future plan is reoptimized at subsequent decisions. A native `stage_counts`
prefix slices forecasts, covariance, loadings and budgets together. Inactive
padded stages are not constrained or executed.

The existing net/gross/name limits and optional beta bound intersect all
generic factor budgets. Supplying both beta and generic factors never
overrides either. `beta=b, max_abs_beta=c` is mathematically equivalent to a
one-column factor with bounds `[-c,c]`. Generic portfolios use the same
current decision-price reference notional units as the existing joint plan;
later reference positions are not future realized NAV weights. Callers must
provide stage loadings/budgets in that same convention.

Cash is feasible only when every lower/upper interval contains zero. A
positive floor or negative ceiling can exclude cash. Incompatible budgets,
or frozen positions that make them impossible, cause explicit infeasibility;
there is no hidden relaxation, clipping or renormalization. Standalone trade
functions return `status="infeasible"` with no weights. The portfolio/native
backtest APIs raise rather than fabricate an admissible book. Solver results
retain feasibility and convex optimality certificates, including factor rows.

## Inspect targets and actual holdings separately

`JointPortfolioResult` contains named factor loadings, stage exposures, bounds,
availability and slacks in `(D,K,F)` order. Both trade solver results contain
`factor_exposures`, `factor_names`, `factor_lower_slack` and
`factor_upper_slack`. `report.objective_trace.factor_constraints` holds the
resolved native contract and each certificate records its active-stage
exposures. The original objective, risk and execution state remain available
in the same trace.

`sf.factor_exposure_report(weights, factors)` evaluates supplied targets or
actual holdings using the same contract. Its arrays are `(D,K,F)` with a
per-decision/stage pass/fail status. A missing coefficient on a nonzero
holding raises; an exactly zero holding contributes zero. A timed factor
contract must contain the reporting timestamps explicitly. The report does
not infer fill-time coefficients or interpolate them.

Planned feasibility does not guarantee that partial/delayed fills or changing
prices leave actual holdings within bounds. Report the actual book separately;
this feature does not enforce a continuous-time exposure guarantee.

## Local validation

Run `PYTHONPATH=src python examples/generic_factor_constraints.py`. It supplies
three synthetic factors to single and joint portfolio construction, native
actual-state execution, and the target-tracking solver. It verifies all active
reference-stage bounds and physical P&L accounting, and saves target/actual
exposures plus trace arrays in the local cache. Cross-stage risk changes the
frictionless target; current costs may legitimately leave a trade plan in
cash. The example does not tune parameters to require trading and makes no
empirical profitability claim.

Tests cover labelled axes and availability, asymmetric/equality/one-sided
budgets, legacy beta equivalence, multiple simultaneous factors, all joint
stages, frozen/pending commitments, inactive prefixes, cash exclusion,
infeasibility, original-objective/cross-risk retention and unchanged defaults.
