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
