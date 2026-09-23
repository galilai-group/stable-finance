# stable-finance

Composable evaluation for financial representations, forecasts, portfolios,
and orders. **Every stage is an injection point.** Bring your own embeddings,
return forecasts (`y_hat`), portfolio weights, or orders; stable-finance only
runs the downstream adapters and metrics you ask for.

```text
┌───────────────────────────────────────────────────────────┐
│                       Dataset / I/O                       │
│                                                           │
│                     1 Hz Market Data                      │
│                              ↓                            │
│                views + targets + metadata                 │
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
- order and fill contracts with a bid/ask execution simulator that marks the
  realized position path rather than the target weights.

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

## Forecasts to weights to orders

`MeanVarianceAllocator` closes the gap between `y_hat` and weights. At each
decision it solves

```text
max_w  mu'w - (risk_aversion / 2) w'Sigma w - sum_i cost_i |w_i - w_prev_i|
```

where `mu` is the forecast, `Sigma` the fitted covariance, and `cost_i` a
proportional transaction cost per unit of weight traded, for instance the
quoted half-spread over mid. The cost term is what makes the optimal portfolio
depend on the portfolio currently held: small forecast changes do not justify
paying the spread, so decisions are solved in order and each starts from the
previous solution. With zero cost the allocator reduces to the closed-form
`Sigma^-1 mu / risk_aversion`. The problem is convex and solved by
accelerated proximal gradient descent, so nothing beyond numpy is required.

```python
from stable_finance import MeanVarianceAllocator, allocate

allocator = MeanVarianceAllocator(risk_aversion=5.0, shrinkage=0.1)
allocator.fit(train_returns)                        # Sigma per horizon
weights = allocator.predict(forecast, cost=half_spread, initial=current)

# Enter at forecasts with your own covariance: skip fitting.
weights = allocate(forecast, covariance, risk_aversion=5.0, cost=half_spread)
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
fills = simulate_fills(orders, best_bid, best_ask)   # buys at ask, sells at bid
held = positions_from_fills(fills, initial=current)  # PortfolioWeights actually held
results = evaluate_fills(fills, realized_mid_returns, initial=current)
```

The simulator fills each order in full at the touch or not at all; an order
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
| `cost_model` | `mid`, `cross`, `cross+0.5bps`, `cross+1bps`, `cross+2bps` |
| `rebalance` | `every_decision`, `daily` — hold the book through the session instead of retrading it |
| `annualization` | `decisions`, `daily` — whether every decision is an independent period |

`trading_cost` also takes a `multiple`, which is the continuous version of
the cost axis and is preferred over the flat add-ons for reasoning about size
or execution quality. A flat `+1bp` charges a 2 bp-spread mega-cap and a
40 bp-spread small-cap the same penalty, whereas impact scales *with*
illiquidity and the quoted spread is the only observable proxy for it. One
scalar then spans the whole range: `0` is frictionless, below 1 is the
fraction of notional that had to cross rather than rest, `1` is crossing
everything, above 1 is the size regime. Because net return at any multiple is
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

## Dataset architecture

The supported input is the Polygon-derived one-second US-equity dataset. Its
column semantics, exchange sessions, sparse-to-dense reconstruction, and
single-session contract live in `stable_finance.dataset`. MosaicML Streaming is
the v1 storage backend:

```bash
uv sync --extra mds
```

```python
from stable_finance.dataset.mds import StreamingMarketDataset

train = StreamingMarketDataset.from_month(mosaic_dir, "2020-01")
```

Storage is kept behind a backend-neutral `MarketSession`. A later LanceDB
backend can therefore produce the same object. Sessions retain explicit date
boundaries so a future multi-day framer can insert learned open and close
tokens without changing storage or target APIs.

`MarketSession.bar_seconds` makes native resolution explicit, and
`resample_session(session, 60)` produces schema-correct minute bars (last
quotes and sizes, OHLC extrema, summed activity, volume-weighted VWAP). Target
tables should still be built from the 1 Hz source before discarding it:
spread-change and volatility-change use within-minute information that the
nine aggregated columns cannot reconstruct. A compact downstream dataset can
therefore store minute features plus precomputed requested targets without
changing their definitions.

View geometry is configuration. `ViewSpec()` records the current 2,048-token,
50–100% of session recipe. `AnchorSpec()` independently records the session,
decision-grid, and forward-measurement geometry. Sequence length, view scale,
resolution, anchor spacing, and future slack are therefore configuration rather
than hidden dataset constants.

Raw outcome construction is also injectable and selective. Request only the
task families and horizons needed by a model or evaluation:

```python
from stable_finance.dataset import compute_pair_targets

y = compute_pair_targets(
    session.features,
    decision_index,
    horizons=[300, 900],
    types=["return"],
)
```

The return, spread-change, and volatility-change definitions are shared with
the vectorized anchor-grid path. Each requested family builds its cumulative
statistics once and reuses them across horizons; unrequested families do no
work. Risk-factor-adjusted returns are an optional extension of the same call.

The same selectivity applies when building cross-sectional target tables:

```bash
python -m stable_finance.dataset.build_targets \
  --mosaic-dir /data/market/mds --holiday-csv market_holidays.csv \
  --out-dir /data/market/targets --start 2023-01 --end 2023-12 \
  --target-types return --horizons 300 900 --transforms uniform
```

Polygon-to-MDS conversion is owned by the storage backend rather than by a
model repository:

```bash
python -m stable_finance.dataset.write_mds \
  --base-path /data/polygon/snapshots/1Hz \
  --output /data/market/mds --start 2023-01 --end 2023-12
```

Training on cross-sectional cells reads a second layout, the day-major store
(`stable_finance.dataset.daystore`): one record per trading day holding the
whole cross-section as a channel-major `(tickers, channels, rows)` float32
memmap, dense exactly as the mosaic stores it, with the forward targets and
every cross-sectional transform precomputed at every anchor. A cell -- K stocks
at one (date, anchor, resolution) -- is then K contiguous slices of one file
and a `(K, T, H)` block of labels, and `stable_finance.dataset.cells` turns it
into normalized views in one vectorized pass. The store is written from the
dense mosaic and verified against it day by day:

```bash
sf-daystore --mosaic-dir /data/market/mds_dense --out-dir /data/market/days \
  --start 2023-01 --end 2023-12 --holiday-csv market_holidays.csv --workers 8
```

The data-preparation pipeline is a set of console scripts, in order:
`sf-shard-months` (raw parquet to sparse MDS, every month, resumable),
`sf-write-mds` (one range, or `--risk-factors`), `sf-densify` (sparse to dense
sessions), `sf-build-targets` (per-month anchor tables), `sf-daystore` (dense
MDS to the day-major store), `sf-industry-map` ((month, ticker) to FF49) and
`sf-reshard` (a date-ordered copy of a month). Each takes its default paths
from the environment (`RAW_DATA_DIR`, `METADATA_PATH`, `MOSAIC_DIR`,
`HOLIDAY_CSV`, `DAYSTORE_DIR`, ...) so a consumer repository sets them once.

`build_session_panel` is the model-neutral injection point immediately before
encoding. It returns normalized views, raw/transformed targets, and
`ViewMetadata` separately. `PanelCache` preserves that separation on disk;
model integrations may encode the metadata into tokens at batch time without
embedding a token layout into the stored dataset.

A model integration may implement `MonthlyProbeData`, allowing the existing
probe to be fit with `RidgeProbe().fit_month("2020-01", source)` while keeping
checkpoint and encoder concerns outside this repository.

## Development

Use `uv` exclusively:

```bash
uv sync
uv run pytest
```

## Migration plan

1. Establish stage contracts and metric semantics.
2. Add the ridge adapter from embeddings to forward-return forecasts.
3. Add covariance estimation and a mean-variance adapter from forecasts to
   weights, with a proportional transaction cost against the held portfolio.
4. Add an order/fill contract and bid/ask execution simulator.
5. Expose serialized evaluation results through a small visualization app.

Research-specific checkpoint loading, tensor collation, model augmentation,
and training remain in `market-jepa`; they depend on this package's public
dataset and outcome contracts rather than duplicating them.
