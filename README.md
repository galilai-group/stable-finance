# stable-finance

Composable evaluation for financial representations, forecasts, portfolios,
and orders. **Every stage is an injection point.** Bring your own embeddings,
return forecasts (`y_hat`), portfolio weights, or orders; stable-finance only
runs the downstream adapters and metrics you ask for.

```text
┌───────────────────────────────────────────────────────────┐
│                       Dataset / I/O                       │
│                                                           │
│    market time series (any schema, calendar, bar size)    │
│                              ↓                            │
│       views  |  objectives (labels)  |  market data       │
└─────────────────────────────┬─────────────────────────────┘
                              ↓
                            model ← What you bring
                              │
                              │                    Cost and Execution Model
      ┌───────────────┬───────┴──────────┬─────────┬────┐
      ↓               ↓                  ↓         ↓    ↓
  embeddings → forward returns → portfolio weights → orders
      │               │                  │             │
      └────── IC ─────┘                  └─── Sharpe ──┘
```

## Where things live

```text
src/stable_finance/
  evaluation/              the pipeline, one package per stage
    embeddings/            stage 1: the Embeddings contract (what you bring)
    returns/               stage 2: forward returns; ridge probes (in memory or
                           streamed) and oracle forecasts
    portfolio/             stage 3: the target portfolio, as if trading were free;
                           risk.py (risk management: universe screens and the risk
                           model) and construction.py (selection and weighting, or
                           the mean-variance target)
    trades/                stage 4: trading policies, how far to move given costs;
                           rebalance and trade rules, and rl/ reserved for learned
                           policies
    scoring/               stage 5: how good each output is; ic.py (forecast IC),
                           pnl.py (positions and P&L), sharpe.py (Sharpe, EFQ curves)
    config.py, backtest.py the harness: one named configuration, entered at any level
    execution.py           the execution models: fill at the mid, across the spread
                           at an EFQ, or through the book; limit orders
    inputs.py              your table in: realized returns and market data out
  dataset/                 what generic code needs to know about a dataset
    schema.py              column roles (mid, vwap, bid_price, ...) and price methods
    anchors.py             decision grid and measurement window (AnchorSpec)
    outcomes.py            target families: returns, spread and volatility changes
    objective.py           one label definition, keyed, and its cross-sectional forms
    market.py              MarketData by level (mid, bars, quotes, book), and the
                           market requirements each stage declares
    universe.py            which names the data can price at all: quoted, and
                           present often enough to estimate
    datasets/market1t/     Market-1T's one pipeline, a module per step:
                           spec, ingest, mosaic, views, daystore, tables
tests/                     mirrors the package: dataset/ and evaluation/<stage>/
```

Each stage package holds its stage's contract (the object it produces), the
code that produces it from the stage before, and the metrics that score it.
Everything public is re-exported at the top level, so `import stable_finance
as sf` reaches all of it.

The package follows scikit-learn's estimator vocabulary: adapters expose
`fit(X, y)` and `predict(X)`, constructor arguments are inspectable
hyperparameters, and the fitted object is reusable. The first migration slice
provides:

- explicit, validated contracts for embeddings, forward returns, and portfolio
  weights;
- a standardized ridge probe from embeddings to forward-return forecasts;
- cross-sectional Spearman information coefficient at each decision time and
  forecast horizon;
- annualized portfolio Sharpe marked at mid;
- annualized portfolio Sharpe after crossing the observable best-bid/best-ask
  spread;
- weight backtesting with quoted half-spread costs and multiple horizons;
- per-horizon covariance estimation and a mean-variance allocator from
  forecasts to weights that charges a proportional transaction cost for moving
  away from the current portfolio; and
- order and fill contracts with an execution simulator, parameterized by EFQ,
  that marks the realized position path rather than the target weights.

The contracts use dense arrays with shape
`(decision time, asset, horizon)`. Decisions, assets, and horizons are carried
alongside every array and alignment is checked before evaluation. NaN denotes
a missing observation.

```python
import stable_finance as sf

# Enter at embeddings: fit the default adapter, then produce y_hat.
probe = sf.fit(train_embeddings, train_returns, alpha=1.0)
forecast = probe.predict(test_embeddings)

# Enter at y_hat: skip fitting and evaluate forecasts directly.
information_coefficients = sf.evaluate_forward_returns(forecast, test_returns)
```

`RidgeProbe` is also a scikit-learn estimator, so users who want explicit
composition can instantiate, inspect, clone, fit, and predict with it directly.
Stable-finance does not transform the supplied target: a fit against raw
returns learns raw returns; a fit against cross-sectional z-scores learns those
z-scores. Choosing or estimating the target transform is a separate pipeline
stage rather than hidden probe behavior.

The portfolio metrics report the frictionless mid-price result and the
executable quoted-spread result separately. The latter executes positive
weight changes at the best ask and negative weight changes at the best bid:

```python
from stable_finance import cross_spread_sharpe, mid_price_sharpe

# Enter at weights: no forecast model or allocator is required.
mid = mid_price_sharpe(weights, realized_mid_returns)
crossed = cross_spread_sharpe(
    weights, realized_mid_returns, best_bid, best_ask
)
```

## Forecasts to a target, to trades

The portfolio stage says what to hold as if trading were free; the trades
stage decides how far to move there given what trading costs. For
mean-variance the target is the closed form `w* = Sigma^-1 mu / risk_aversion`,
and each rebalance then solves

```text
min_w  (risk_aversion / 2) (w - w*)'Sigma (w - w*) + sum_i cost_i |w_i - w_prev_i|
```

tracking error against the target, priced by its own risk model, against the
cost of trading from `w_prev`. `cost_i` is a proportional cost per unit of
weight traded, for instance the quoted half-spread over mid. This is exactly
the cost-aware mean-variance problem, because

```text
mu'w - (risk_aversion / 2) w'Sigma w = -(risk_aversion / 2) (w - w*)'Sigma (w - w*) + const
```

so `rebalance(mean_variance_target(...))` gives the weights `allocate` gets in
one solve. The cost term is what makes the held portfolio depend on the one
already held: small forecast changes do not justify paying the spread, so
decisions are solved in order and each starts from the previous solution. The
problem is convex and solved by accelerated proximal gradient descent, whose
proximal step is the trade rule itself.

```python
from stable_finance import (
    MeanVarianceAllocator, allocate, estimate_covariance, mean_variance_target, rebalance,
)

sigma = estimate_covariance(train_returns, shrinkage=0.1)        # Sigma per horizon
target = mean_variance_target(forecast, sigma, risk_aversion=5.0)  # portfolio stage
held = rebalance(target, sigma, risk_aversion=5.0, cost=half_spread, initial=current)

# The same in one step, from a forecast.
held = allocate(forecast, sigma, risk_aversion=5.0, cost=half_spread)
held = MeanVarianceAllocator(risk_aversion=5.0, shrinkage=0.1).fit(train_returns) \
    .predict(forecast, cost=half_spread, initial=current)
```

`estimate_covariance` is the fitted estimator's public core: a pairwise sample
covariance over finite observations, shrunk toward its diagonal and clipped to
the positive semidefinite cone. A NaN cost marks an asset that cannot be
traded at that decision, so it holds its previous weight; a NaN forecast is a
zero expected return that still pays risk and cost.

Orders are the weight changes along a target path; fills are what executed:

```python
from stable_finance import (
    evaluate_fills, orders_from_weights, positions_from_fills, simulate_fills,
)

orders = orders_from_weights(weights, initial=current)
fills = simulate_fills(orders, best_bid, best_ask)   # efq=1: buys at ask, sells at bid
mid = simulate_fills(orders, efq=0)                  # midpoint fills, no quotes needed
held = positions_from_fills(fills, initial=current)  # PortfolioWeights actually held
results = evaluate_fills(fills, realized_mid_returns, initial=current)
```

The simulator fills each order in full at `mid * (1 +/- efq * half_spread)`
or not at all, at the touch the order meets on arrival when the market data
carries it, and measured against the decision price. EFQ is the effective
half-spread paid over the quoted one: `0` fills at the midpoint, `1` at the
touch, and above `1` is size and crowding.
It may be a scalar or an array that broadcasts to `(decision, asset)`, so a
per-name execution quality needs no new model. At a positive EFQ an order
without a positive quote is left unfilled and the realized position lags the
target. When every order fills, `evaluate_fills` reproduces
`cross_spread_sharpe` exactly. Depth, partial fills, impact, and commissions
remain outside the simulator; bring your own `Fills` to evaluate a richer
execution model.

In the originating market-jepa setup, the default for supervised training,
ridge fitting, and evaluation is the empirical-uniform target within each
same-date, same-anchor cross-section:

```text
uniform_i(t, h) = rankdata(y_cell)_i / (n_cell + 1)
```

Ties receive their average rank. Gaussian rank, ordinary cross-sectional
z-score `(y - mean) / std`, and raw returns remain explicit alternatives. The
ridge estimator never changes the supplied target; the stable-finance dataset
layer computes the representations explicitly and the caller chooses one.

The dataset layer can expose all four representations together, so choosing a
loss target does not discard information or push preprocessing into model
code:

```python
targets = stats.transform(raw, date, anchor, target_types, horizons)
model_target = targets.select("uniform")       # default
all_target_metadata = targets.as_dict()        # raw/zscore/uniform/rank
```

## A forecast with a known IC

Testing a portfolio or execution stage against a learned forecast confounds
how good the forecast was with how well the stage used it. `oracle_forecast`
removes the first unknown: it sorts the realized returns (rank IC 1) and
swaps forecast ranks until every decision's rank IC is exactly the target,
with the errors placed where the test wants them.

```python
from stable_finance import oracle_forecast, run_backtest

for noise in ("tails", "middle", "random"):
    forecast = oracle_forecast(realized, 0.02, noise=noise, seed=0)
    report = run_backtest(forecast, realized, half_spread=spread, history=prior)
```

`tails` exchanges best with worst, then second-best with second-worst, so the
extremes are wrong (at a low IC, inverted) and the middle is right. `middle`
exchanges outwards from the median, so the tails are right. Both hit the
target at every decision: a swap of forecast ranks moves the centred rank
sum by exactly `(f_a - f_b)(r_a - r_b)`, so the budget is spent to about 1e-6
even with ties in the realized returns (1e-4 when ties are heavy).

`random` applies random transpositions with errors spread evenly, and hits
the target *on average* rather than at every decision. One random swap
multiplies the expected IC by `1 - 2 / (n - 1)` whatever the current
ranking, so `(n - 1) / 2 * ln(1 / ic)` swaps, with `n` the names in that
cross-section, reach the target in expectation with no steering, and each decision
then scatters around it by about `1 / sqrt(n - 1)`, which is exactly the
sampling noise a real forecast of that IC shows on a cross-section of that
size. `scatter="sampling"` gives `tails` and `middle` the same per-decision
distribution, drawing each decision's target from that Gaussian and then
hitting it exactly, so the three modes then differ only in where the errors
sit -- which is everything a book that trades the tails cares about.

The values are the realized returns reassigned by the degraded ranking and
shrunk towards the cross-sectional mean by `scale`, which defaults to `|ic|`
(Grinold's `IC * volatility * score`). An unshrunk oracle has the realized
dispersion, which a cost-aware allocator or `edge_to_cost` reads as an edge
`1 / IC` times too large; `scale=1.0` gives it anyway.

## The harness is a configuration, not a constant

A Sharpe ratio is not a property of a forecast. It is a property of a forecast
plus every trading decision made downstream of it -- which names are tradable,
how much of the cross-section to hold, how risk is forecast, how weights follow
from it, how often to rebalance, what a trade costs, how a periodic return is
annualized. None of those are dictated by the model being evaluated, and
several move the answer by more than the gap between two models.

So the harness is an explicit object. `BacktestConfig` names every decision,
`DEFAULT_BACKTEST` is the library's opinion when the caller has none, and
`enumerate_configs` produces the space so the sensitivity of a result to the
harness can be measured rather than assumed.

```python
from stable_finance import DEFAULT_BACKTEST, enumerate_configs, run_backtest

report = run_backtest(forecast, realized, half_spread=spread, history=prior)
report.sharpe_mid, report.sharpe_net, report.information_coefficient

# The same forecast under every harness in the space.
space = enumerate_configs()                       # 4,080 distinct portfolios
space = enumerate_configs(cost_model="mid")       # or a named slice
rows = [run_backtest(forecast, realized, half_spread=spread, history=prior,
                     config=config).as_dict() for config in space]
```

`run_backtest` computes the IC and the Sharpe from **the same forecast array**,
which is the whole reason it is one function rather than two: a study of how
tightly the two move together is worthless if they came from separately fitted
predictors.

| Axis | Options |
|---|---|
| `universe` | `all`, `tight_50pct`, `tight_25pct` — a cross-sectional screen on each name's own quoted half-spread |
| `selection` | `all`, `quintile`, `decile`, `decile_long_only` — how much of the ranking to trade |
| `weighting` | `mean_variance` (cost-aware, path-dependent), `equal`, `inverse_vol`, `min_var`, `erc`, `rank` |
| `risk_model` | `diagonal`, `shrunk_sample`, `ledoit_wolf`, `embedding` — the last forecasts correlations from the encoder's own representation |
| `cost_model` | `mid`, `cross` — free, or the name's own quoted half-spread times `efq` |
| `rebalance` | `every_decision`, `daily` — hold the book through the session instead of retrading it |
| `annualization` | `decisions`, `daily` — whether every decision is an independent period |

`efq` is the continuous cost axis, and `trading_cost` takes it as `multiple`.
Charging a multiple of each name's own spread, rather than a flat basis-point
add-on, makes impact scale *with* illiquidity, and the quoted spread is the
only observable proxy for it. One scalar spans the whole range: `0` is
frictionless, below 1 is the fraction of notional that had to cross rather
than rest, `1` is crossing everything, above 1 is the size regime. Because net return at any multiple is
`gross - multiple * cost`, a sensitivity curve costs one backtest rather than
one per point.

Every axis is also a standalone public function (`apply_screen`,
`select_book`, `weight_book`, `estimate_risk`, `trading_cost`), so a caller who
wants one of them and none of the rest is not obliged to adopt the config
object. `enumerate_configs` removes configurations that duplicate another
point rather than reporting one portfolio several times: a book that ignores
the covariance appears under one risk model, and equal-risk-contribution on a
diagonal model is inverse-volatility, so it is not also counted as its own
option.

## Is the forecast worth its spread?

A Sharpe ratio is the wrong headline statistic for a short-horizon forecast,
and the reason is sample size rather than taste. It is a time-series quantity
over a few hundred periods with a standard error near
`sqrt((1 + S**2 / 2) / years)`, so at two or three years of data a true Sharpe
below 0.5 cannot be told apart from zero however carefully it is computed.
`edge_to_cost` is the cross-sectional alternative: the ratio of each name's
cross-sectionally centred forecast to its own one-way cost, over every
(decision, name) pair rather than every period, and therefore pinned orders of
magnitude more tightly. Above 1 the expected move clears one crossing; a round
trip needs 2.

```python
from stable_finance import edge_to_cost

ratio = edge_to_cost(mu, half_spread)       # (decision, asset)
np.nanmedian(ratio)                         # the number to report
```

The forecast must be in the units of the realized return. A probe fit against
a rank target predicts ranks, so rescale it by the slope of realized returns
on its own output before comparing it to a spread; a rank-unit forecast makes
this ratio meaningless rather than merely mis-scaled.

Two selection rules follow from the same comparison. `threshold_selection`
holds a name when `|edge| > multiple * cost`, so a wide-spread name must carry
a larger forecast to earn its place -- a screen a quantile cut cannot express,
because the tail of a forecast distribution is not the tail of an
edge-to-cost distribution whenever the two are correlated, and in equities
they are. `persistent_selection` then separates opening from closing, because
re-deciding the whole book every decision is only sensible when trading is
free: a name whose edge sits near the entry bar otherwise flips in and out and
pays a round trip each time for a forecast that barely moved.

```python
from stable_finance import persistent_selection, threshold_selection

signs = threshold_selection(mu, eligible, multiple=1.0, cost=half_spread)
held = persistent_selection(mu, eligible, enter=1.5, exit_=0.0,
                            cost=half_spread)
```

`exit_` is signed and spans the useful behaviours with one number: `0` closes
when the forecast changes sign, positive tolerates a mild adverse forecast and
closes only on a reversal worth the round trip, negative closes when
conviction merely decays. Holding costs nothing, so the exit test does not
re-check `enter` -- a position already paid for is worth keeping on weaker
evidence than it took to open, which is the whole asymmetry a cost creates.

Horizon is the lever underneath all of this. The spread is paid once whatever
the holding period, while the expected move grows with the horizon, so
edge-to-cost improves with horizon unless the IC decays faster than
`1 / sqrt(H)`. Measuring that needs one probe per horizon, which
`StreamingRidge` fits in a single pass over the pool: `X'X` dominates the cost
and each extra horizon adds only an `X'y`.

```python
probe = StreamingRidge(alpha=[a_300, a_900, a_3600])
for shard in shards:
    probe.partial_fit(shard["X"], shard["y"])   # (n_rows, n_horizons)
probe.finalize()
forecasts = probe.predict(eval_X)               # (n_rows, n_horizons)
```

Targets do not share a gram: a long horizon runs off the end of a session and
is missing on rows a short one keeps, so each accumulates over its own
labelled subset and the block fit is identical to fitting each alone.

## Attaching a market to embeddings you already have

Quotes and realized returns are properties of a `(date, anchor, ticker)` row.
They do not depend on which encoder produced an embedding for that row, so a
sweep that stored only embeddings and targets can still be turned into a
portfolio without a second forward pass. `MarketPanel` reads a month of cached
quotes without touching the cached views, and `align` reorders it onto whatever
rows the caller has:

```python
from stable_finance import MarketPanel, decision_labels, to_grid

market = MarketPanel.from_cache(cache_root, "2008-08", anchors_per_day=8)
attached = market.take(market.align(dates, anchors, tickers))
attached["half_spread"], attached["raw_targets"], attached["matched"]
```

Check `matched`. An unmatched row carries no quote, and a missing quote that is
read as zero prices a free trade in a market that never existed.
`market.market(index, dates_or_labels, tickers, decision_axis=..., asset_axis=...)`
returns the same rows as `MarketData` on a grid. A panel built from a dataset
without quotes records that, and its market data carries no bid or ask
rather than NaN ones.

## Any input level, any market data

The code is a grid. One axis is what you bring; the other is what market data
you have. `run_backtest` takes the input at any level and runs only the
stages below it, and every level is scored the same way:

| You bring | The harness runs |
|---|---|
| `Embeddings`, with `train=(embeddings, returns)` | a ridge probe, then everything below |
| `ForwardReturns` | universe, selection, risk and sizing, then trading |
| `PortfolioWeights`, the optimal portfolio | the deltas toward it, decided under cost by the trades stage, then filled |
| `Orders`, the deltas, optionally with limits | your orders, filled |

Market data comes in levels, and a richer level derives the poorer ones: a
book's first level is the best bid and ask, which give the mid and the
half-spread.

| Level | Needs | What it supports |
|---|---|---|
| `mid` | a price series | returns, and fills at the mid at no cost |
| `bars` | `high`, `low`, `close` | a spread estimated from the bars, so costs can be charged |
| `quotes` | `bid_price`, `ask_price` | the quoted spread; with `bid_size` and `ask_size`, a book of one level |
| `book` | depth on both sides | orders that walk the book for their size |

Quotes are read twice: at the decision, and again when the order reaches the
market, `delay_seconds` later (five by default in both readers), as
`exec_bid_price` and `exec_ask_price`. An EFQ fill is priced off the touch it
meets and measured against the decision price, so the drift in between is
part of the cost. Top-of-book sizes are averaged over the `depth_seconds`
before arrival, a few seconds of the touch standing in for depth the data
never saw, which is what lets the book model run on Market-1T as it is.

The execution model is the bridge between the two axes, and each one needs a
level. It is the config's `cost_model`, and every model is scored the same
way, from its fills: the position held is what the fills produced, and the
cost is each fill's distance from the decision price. A name that is not
priced at a decision is held until it is, never moved for free.

| `cost_model` | Fills at | Needs |
|---|---|---|
| `mid` | the price the return is measured from, at no cost | nothing |
| `cross` | the touch on arrival, `efq` of its half-spread paid, measured from the decision price | a spread, quoted or estimated |
| `book` | the depth, level by level, up to the order's size; the rest is unfilled | depth, and `equity` in the config |
| `impact` | the evolving mid plus the price impact of Frazzini, Israel and Moskowitz (2018), worked passively over a trading day as resting orders, so no spread | `adv`, and `equity` in the config |
| `impact_stock` | the same, with the paper's stock-specific terms: market cap, idiosyncratic volatility, market volatility, time trend | `adv`, `market_cap`, `idio_vol`, `market_vol`, `time_trend`, and `equity` |

`impact` and `impact_stock` are the institutional cost models, simple and
stock-specific, independent of EFQ. The paper fits the
realized cost of $1.7 trillion of live institutional orders, each worked with
resting limit orders over about a trading day, and finds it concave in size:
in basis points, `a + b x + c sqrt(x)` for an order of `x` percent of the
name's average daily dollar volume, with the U.S. estimates `b = -0.53` and
`c = 11.21` (Table VII, column 10). The model here does two things with it.
The position ramps linearly over the next trading day, read from the `days`
labels, so an intraday signal earns only the part of its return it has
actually bought; and each slice fills at that decision's own mid plus the
formula's impact. Orders in the same name within one day are one parent
order, since concavity would otherwise reward slicing. `execution_days=0`
fills everything at the decision, which is the paper's own accounting. The
cost does not depend on how long you take: the paper's patience is in its
coefficients. `evaluation_inputs(..., adv_lookback=...)` computes `adv` from
a volume column.

`impact_stock` adds the rest of column 10: cost falls with the log of market
cap (-0.14 per log billion), rises with the name's idiosyncratic volatility
(+0.31 per annualized percent) and the market's (+0.12), and drifts with a
time trend (+0.01 per month). The paper never reports the intercept, so the
level is pinned: a name at the reference characteristics pays the simple
model's cost. The firm-size reference is the paper's own median traded firm,
$4.9 billion; the volatility and time references are not reported by the
paper and are set to ordinary values for its sample (30%, 17%, May 2007),
overridable through `ImpactStockCoefficients`. `evaluation_inputs(...,
vol_lookback=...)` computes the volatilities from daily returns and the
time trend from the dates; market cap comes through a `market_cap` column
role.

An order may carry a `limit`, the most slippage it accepts as a fraction of
the decision price. What a model can fill within the limit fills. The rest
rests, and fills at the limit only if the execution window traded through it.
That needs the window's best prices, `window_best_buy` and `window_best_sell`,
which `evaluation_inputs` provides at every level. Queue position is not
modelled, so a resting fill is an upper bound. When fills can fall short,
`fill_path` trades toward each target from the position actually held.

```python
from stable_finance import DEFAULT_BACKTEST, Orders, run_backtest

run_backtest(embeddings, realized, market=market, train=(train_emb, train_ret))
run_backtest(forecast, realized, market=market)
run_backtest(weights, realized, market=market,
             config=DEFAULT_BACKTEST.replace(cost_model="book", equity=1e6))
run_backtest(Orders(q, decisions, assets, horizons, limit=0.0), realized,
             market=market)                    # rest at the mid
```

## Each stage says what it needs of the market

`MarketData` is the observable market on the `(decision, asset)` grid, one
array per role, as of the decision. The quantities are the dataset schema's
roles, `half_spread`, the four book quantities, which add a level axis, and
the two execution-window prices, which are the only quantities from after the
decision. `mid` and `half_spread` are derived from `bid_price` and
`ask_price`, and those from the book, when not supplied. `estimated` names
the quantities that were estimated rather than observed. Every stage that reads the market declares what it needs, and
refuses with `MissingRolesError` when a column is absent:

| Stage | Needs |
|---|---|
| embeddings to forecast (`fit`, `evaluate_forward_returns`) | nothing |
| forecast to weights, `cost_model="mid"`, `universe="all"` | nothing |
| weights to fills at the return's own starting price (`simulate_fills(orders, efq=0)`) | nothing |
| any crossing cost model, the liquidity screens, `rebalance(cost=...)`, `allocate(cost=...)`, `threshold_selection`, `persistent_selection`, `edge_to_cost`, `evaluate_weights(half_spread=...)` | `half_spread` |
| weights to fills at a positive EFQ (`simulate_fills`, `cost_model="cross"`) | `half_spread`, or both quotes |
| fills through the book (`cost_model="book"`) | `book_bid_price`, `book_bid_size`, `book_ask_price`, `book_ask_size` |
| the impact model (`cost_model="impact"`) | `adv`, average daily dollar volume |
| the stock-specific impact model (`cost_model="impact_stock"`) | `adv`, `market_cap`, `idio_vol`, `market_vol`, `time_trend` |
| limit orders, any model but `impact` | `window_best_buy` and `window_best_sell`, and a decision price |
| `cross_spread_sharpe` | `bid_price` and `ask_price` |

The cost-less path needs no market column at any stage. A return is already
measured at a price the data supports, the one-minute VWAP by default. A
cost-free fill at that price, the benchmark fill, therefore assumes nothing
new. Only costs need quotes, and the refusals are for those.

A NaN inside a column is a name that was not quoting at that decision, which
every stage handles. A missing column is a dataset that never had the
information. Filling it with NaN would make the whole universe untradable.
Filling it with zero would report a frictionless result under a
spread-charged name. Asking for the frictionless result by name, with
`cost_model="mid"` or `efq=0`, is the honest version of the second.

```python
from stable_finance import (
    DEFAULT_BACKTEST, enumerate_configs, market_from_days, run_backtest,
)

market = market_from_days(day_paths)          # whatever roles the dataset has
market = market_from_days(day_paths, lookback=20, window_seconds=300)   # and adv,
    # volatilities, arrival quotes, the one-level book, and window extremes
DEFAULT_BACKTEST.requires                     # {'cost_model': ('half_spread',)}
run_backtest(forecast, realized, market=market)      # refused without quotes
space = enumerate_configs(market=market)      # only what the market supports
```

Each config axis declares its options' needs in the module that defines
them (`COST_MODEL_REQUIRES`, `UNIVERSE_REQUIRES`, `WEIGHTING_REQUIRES`, ...).
A configuration states its needs in `requires`, and a sweep checks every
configuration before running any. Without a spread, `enumerate_configs`
yields the frictionless space: cost model `mid`, EFQ zero, universe `all`.
A custom rule that reads other roles, say one sized off the bar range, declares
`("open", "high", "low", "close")` in its table or with `@requires(...)`, and
a market without those columns refuses it the same way. `half_spread=` remains accepted wherever it was, as market data holding
that quantity alone.

## Trading events, not a lattice

An event strategy trades when something happens -- buy on a news story at
10:41:07, sell in the closing auction -- so it has no decision lattice.
`event_backtest` takes the trades as round trips and a tape of quotes, builds
the grid they imply, and scores it with the same harness:

```python
from stable_finance import DEFAULT_BACKTEST, Trades, event_backtest

trades = Trades(assets, entry, exit, size,             # epoch seconds; size signed, of equity
                exit_price=closing_cross)              # NaN: exit at the touch instead
result = event_backtest(trades, tape,                  # tape: ticker, timestamp, bid_price, ask_price
                        config=DEFAULT_BACKTEST.replace(efq=0.70))
result.report.sharpe_net                               # daily, from the harness
result.ledger.gross, result.ledger.cost                # per trade
```

Decisions are every entry and exit; each trade gets its own column, reused
only once its trade has closed, so two trades on one name at once are two
positions. Prices are the last quote at or before each time; an order meets
the touch `delay_seconds` later; an `exit_price` fills there with no spread.
A per-trade `efq` replaces the configuration's.

Two harness settings make this exact, and are available to any input:

- `holding="drift"` (in `BacktestConfig`) carries a position with its own
  return between decisions, so a trade held across several decisions earns
  exactly its price change, and an order goes out only when the target
  changes -- an unchanged target is held, not rebalanced back. The default,
  `holding="rebalanced"`, holds each name at its weight, right for a book
  retraded every decision. Drift needs returns from each decision to the
  next.
- `run_backtest(orders, ..., unfilled="retry")` (the default) keeps an order
  that could not fill working, so an exit sent when the name has no quote
  goes out again at its next quote rather than leaving the position open.
  `unfilled="cancel"` drops it.

`run_backtest(..., efq=array)` likewise takes a per-name execution quality,
broadcasting to `(decision, asset)`, in place of the configuration's scalar.

## Bringing your own data

A dataset needs no machinery of its own. Map its columns to roles, say when
decisions are made, and `evaluation_inputs` returns the realized forward
returns and the observable market on one `(decision, asset)` grid, ready for
every stage:

```python
from stable_finance import DEFAULT_BACKTEST, evaluation_inputs, run_backtest
from stable_finance.dataset import FeatureSchema

schema = FeatureSchema.from_roles(mid="px")                  # one price column is enough
# schema = FeatureSchema.from_roles(vwap="px", volume="qty", bid_price="b", ask_price="a")
realized, market = evaluation_inputs(frame, schema, decisions, horizons=[300, 900])
report = run_backtest(forecast, realized, market=market,
                      config=DEFAULT_BACKTEST.replace(cost_model="mid"))
```

`frame` is a DataFrame or a mapping of columns with a timestamp, an optional
asset column, and the schema's columns. Timestamps say when an observation
became known, which is the end of its bar for bar data. Nothing else about
time is required: no calendar, no bar length, no session.

There is no lookahead by construction. The market at a decision is the last
observation stamped at or before it. A return's price at decision `t` is
measured over the observations stamped in `(t, t + window]`, strictly after
the decision, and its forward price over `(t + h, t + h + window]`. The price
is the schema's richest unless you name one: VWAP weighted by volume, else
the time-averaged midpoint, else the close. A window with no observation has
no price, and the return is NaN rather than a guess.

**Roles decide what is possible downstream.** The roles are `mid`, `open`,
`high`, `low`, `close`, `vwap`, `volume`, `n_trades`, `bid_price`,
`ask_price`, `bid_size`, `ask_size` and `market_cap`. None is required. Every target family
and every stage declares the roles it needs. A univariate series gets returns
and frictionless portfolios, and a backtest that charges the spread is
refused with `MissingRolesError` naming the missing columns. Bars without
quotes can be charged: `spread_lookback=` estimates the half-spread from the
high, low and close of the bars before each decision (Abdi and Ranaldo,
2017), and marks it as estimated. Depth comes in as arrays on `MarketData`
directly; there is no table format for it yet.

The target families themselves (`stable_finance.dataset.outcomes`) and the
`Objective` that names a label (`stable_finance.dataset.objective`) are
generic too, for a dataset that builds its own dense sessions.

## Market-1T

`stable_finance.dataset.datasets.market1t` is the Polygon-derived one-second
US-equity dataset's full pipeline, a module per step:

| Module | Step |
|---|---|
| `spec` | its nine-column schema, NYSE calendar with holiday file, clock (`Timeline`), and store locations, local or Hugging Face |
| `ingest` | raw Polygon parquet to the sparse MDS mosaic, one range or every month, and the Fama-French industry map |
| `mosaic` | the MDS mosaic: reader, sparse-to-dense, date-ordered copy |
| `views` | sessions to model inputs: grid rebuild, views, evaluation panels and their cache, training cells |
| `daystore` | the day-major store, its labels and objective caches, and market data and returns read from it |
| `tables` | monthly cross-section tables of the targets |

The commands run it in order: `sf-shard-months` and `sf-write-mds` (raw
parquet to sparse MDS), `sf-densify` (sparse to dense sessions),
`sf-build-targets` (monthly anchor tables), `sf-daystore` (dense MDS to the
day-major store), `sf-industry-map` ((month, ticker) to FF49) and
`sf-reshard` (a date-ordered copy of a month). Each takes its default paths
from the environment (`RAW_DATA_DIR`, `METADATA_PATH`, `MOSAIC_DIR`,
`HOLIDAY_CSV`, `DAYSTORE_DIR`, ...). MosaicML Streaming is the mosaic's
backend (`uv sync --extra mds`).

```bash
sf-daystore --mosaic-dir /data/market/mds_dense --out-dir /data/market/days \
  --start 2023-01 --end 2023-12 --holiday-csv market_holidays.csv --workers 8
```

**The clock.** `market1t_timeline()` is Market-1T's one description of time:
the NYSE calendar, one-second bars stamped at their end, and a decision every
five minutes from 09:30. Decision `a` is made at the end of the bar that
starts `a` seconds after the open. Every builder reads time from it. Each
stored day is checked against it by `check_chronology`, which refuses a day
whose rows, origin or decisions disagree with the clock, or that holds data
before a name's first row.

**The day store** holds one record per trading day: the whole cross-section
as a channel-major `(tickers, channels, rows)` float32 memmap, with the
forward targets and every cross-sectional transform precomputed at every
anchor. A training cell, K stocks at one (date, anchor, resolution), is then
K contiguous slices of one file, and `views` turns it into normalized views
in one vectorized pass.

**Labels are objectives, and caches are only caches.** An `Objective` names
everything that defines a label: target families, horizons, the measurement
window at each end of a return (also how long a position takes to enter and
exit), the price, and the cross-section the standardized labels are taken
over, which is every name observed in the cell once it has 20. A day
record's stored targets are its default objective. Any other objective is
computed from the stored one-second series, and cached under a hash of its
definition when asked. A batch of K names is labelled against the whole
cross-section, not ranked among itself.

```python
from stable_finance import forward_returns_from_days, market_from_days, run_backtest
from stable_finance.dataset import Objective

record.labels(anchor, batch, Objective(horizons=(900,), window_seconds=300))
market = market_from_days(days)
for window in (60, 300, 900):                    # the execution-window sweep
    realized = forward_returns_from_days(days, Objective(window_seconds=window))
    report = run_backtest(forecast, realized, market=market, config=config)
```

Each window fills at its own benchmark, `simulate_fills(..., efq=0)`,
at no cost relative to the return, so the sweep shows how much of the signal
survives entering and exiting more slowly. The best window depends on order
size against volume, which the backtest does not model, so report the curve.

`build_session_panel` is the model-neutral injection point immediately before
encoding. It returns normalized views, labels and `ViewMetadata` separately,
and `PanelCache` preserves that separation on disk. A model integration may
implement `ProbeData`, so the probe can be fit on one of the dataset's own
periods with `RidgeProbe().fit_period(period, source)`.

## Development

Use `uv` exclusively:

```bash
uv sync
uv run pytest
```

## Migration plan

1. Establish stage contracts and metric semantics. (done)
2. Add the ridge adapter from embeddings to forward-return forecasts. (done)
3. Add covariance estimation and a mean-variance adapter from forecasts to
   weights, with a proportional transaction cost against the held portfolio.
   (done)
4. Add an order/fill contract and bid/ask execution simulator. (done)
5. Expose serialized evaluation results through a small visualization app.

Research-specific checkpoint loading, tensor collation, model augmentation,
and training remain in `market-jepa`; they depend on this package's public
dataset and outcome contracts rather than duplicating them.
