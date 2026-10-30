# Exposure report for supplied portfolio weights

`portfolio_exposure_report` diagnoses the weights after construction, not the
forecast before construction. It accepts the existing `PortfolioWeights`
contract, including native single-horizon targets, separate per-horizon
targets, fixed multi-horizon net targets and joint first-stage targets.
It never rescales weights, changes a constraint or estimates a predictor.

```python
from stable_finance import PortfolioExposureLimits, portfolio_exposure_report

report = portfolio_exposure_report(
    target,
    beta=known_beta,             # (N,), (D,N), or explicit (D,N,H)
    industry=known_industry,     # (N,) or (D,N)
    limits=PortfolioExposureLimits(
        gross_limit=1.0,
        max_weight=0.1,
        max_abs_net=0.0,
        max_abs_beta=0.01,
        max_abs_industry=0.03,
        max_hhi=0.4,
    ),
)
data = report.as_dict()          # JSON safe; unknown values become null
```

All checks are per decision and horizon. Array columns must follow the
weights' asset order, and supplied beta estimates and industry classifications
must be known at that decision. The report does not infer their estimation
window or audit their source. Static inputs broadcast across decisions and
horizons; industry membership can change by decision.

The native `estimate_beta` output is `(H,N)`. Select a matching horizon, or
explicitly transpose and expand it to `(D,N,H)` before reporting. Do not
mistake it for time-varying `(D,N)` beta when those axis lengths coincide.
A mixed target needs beta for its declared common evaluation interval;
the report does not average source-horizon beta estimates.

## Definitions

| Field | Definition |
|---|---|
| `gross`, `net` | `sum(abs(w))`, `sum(w)` |
| `long_exposure`, `short_exposure` | Positive long and absolute short notionals, in equity units |
| `max_abs_weight`, `names_held` | Largest absolute name weight and number of nonzero positions |
| `beta_exposure` | `sum(w_i * beta_i)`, in portfolio-equity units |
| `industry_net`, `industry_gross` | Signed and absolute weight sums per labeled industry |
| `concentration_hhi` | `sum((abs(w_i) / gross)**2)` for a noncash book |
| `effective_names` | `1 / concentration_hhi` for a noncash book |

HHI uses absolute positions divided by gross, so it remains meaningful for
long-short portfolios. Cash has HHI zero and effective names zero. Beta is
not divided by gross. After mixing, recompute concentration from the net
weights; averaging component concentration would give the wrong answer.

Signal beta/industry neutralization does not ensure neutral final holdings:
selection or unequal covariance-based sizing can reintroduce exposures.
The standalone example demonstrates this with an already-neutral forecast
and the unchanged native mean-variance constructor.

## Missing data and constraint status

Weights must be finite. Missing beta or industry data on a nonzero position
is not interpreted as zero exposure. The report records gross-weighted
`beta_coverage` and `industry_coverage`, plus unknown-industry net and gross
amounts. Known industry amounts plus unknown amounts reconcile to the book.
Missing factors on exactly zero positions do not prevent evaluation; an
all-cash book has zero exposures and full effective coverage.

`PortfolioExposureLimits` declares optional nonnegative bounds. Its defaults
are all unset; the reporter imposes no new defaults on existing constructors.
`max_abs_net=0` requests dollar neutrality, `max_abs_beta=0` beta neutrality,
and `max_abs_industry=0` net neutrality in every known industry.

`violations` contains only requested checks, as raw nonnegative excess over
the declared bound. Factor violations are unknown if held-factor coverage is
incomplete. `constraint_status` distinguishes:

- `pass`: all requested checks are evaluable and within the tolerance;
- `fail`: at least one known excess exceeds the tolerance;
- `unknown`: no known failure, but a requested check cannot be evaluated;
- `not_checked`: no bounds were requested.

Default tolerance is `1e-8`, applied to pass/fail rather than hiding raw
residuals. An unknown beta cannot be reported as satisfying a beta limit.
These statuses are diagnostics, not a repair or feasibility projection.

## Target versus actual holdings

The function reports exactly the weights it receives. When called after a
constructor, it describes target weights. For realized exposure, supply the
weights actually held after trades and fills. Target reporting does not
assert that the desired portfolio was executed. No future joint reference
stage is silently treated as an actual holding.

## Run the examples

```sh
python examples/portfolio_exposure.py --output-dir /absolute/local/exposure
python examples/multi_horizon.py --output-dir /absolute/local/fixed
python examples/joint_multi_horizon.py --output-dir /absolute/local/joint
```

The examples attach reports to native component/single targets and to the
fixed/joint current target outputs. They use supplied analytical factor data
and forecasts. They test exposure identities and declared violations, not
profitability or predictive accuracy.
