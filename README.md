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
      ┌───────────────┬───────┴──────────┬─────────────┐
      ↓               ↓                  ↓             ↓
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
