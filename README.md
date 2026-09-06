# stable-finance

Composable evaluation for financial representations, forecasts, portfolios,
and orders. The library accepts work at any stage of the pipeline and uses
small, textbook adapters to reach the metrics that apply to it:

```text
embeddings -> forward returns -> portfolio weights -> orders
     |               |                 |              |
     +------ IC -----+                 +---- Sharpe ---+
```

The first migration slice provides:

- explicit, validated contracts for embeddings, forward returns, and portfolio
  weights;
- a standardized ridge probe from embeddings to forward-return forecasts;
- cross-sectional Spearman information coefficient at each decision time and
  forecast horizon;
- annualized Sharpe ratio; and
- weight backtesting with quoted half-spread costs and multiple horizons.

The contracts use dense arrays with shape
`(decision time, asset, horizon)`. Decisions, assets, and horizons are carried
alongside every array and alignment is checked before evaluation. NaN denotes
a missing observation.

```python
from stable_finance import RidgeProbe, evaluate_forward_returns

probe = RidgeProbe(alpha=1.0).fit(train_embeddings, train_returns)
forecast = probe.predict(test_embeddings)
information_coefficients = evaluate_forward_returns(forecast, test_returns)
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

View geometry is configuration. `ViewSpec()` records the current 2,048-token,
50–100% of session recipe; sequence length, scale range, resolution range,
anchor grid, and future slack are hyperparameters rather than dataset schema.

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
   weights.
4. Add an order/fill contract and bid/ask execution simulator.
5. Expose serialized evaluation results through a small visualization app.

Research-specific checkpoint loading, target construction, and training remain
in `market-jepa`; they should depend on this package's public contracts rather
than being copied into it.
