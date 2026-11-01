"""Real forecasts without a GPU: causal intraday features, fit on one month.

The noised oracle says what a forecast of a given IC is worth; these say what
IC cheap, well-known intraday signals actually have on Market-1T, so the real
models can be placed on the oracle's IC-to-Sharpe map. No GPU and no learned
representation: the features are the ones ``data.py`` caches, all read from
rows at or before the decision --

* short-term reversal: minus the mid-quote return over the past 5, 15 and
  30 minutes;
* the last five one-minute mid-quote returns, the name's own lags in the
  spirit of Chinco, Clark-Joseph and Ye (2019, *JF*), whose 1-minute LASSO
  uses the whole cross-section's lags;
* the quote imbalance over the last five seconds, ``(bid size - ask size) /
  (bid size + ask size)``;
* the relative half-spread, the last five minutes' volume against the
  session's average so far, and the last 30 minutes' realized volatility.

Each feature is standardized within each decision's cross-section (rank to
normal score), so the models learn cross-sectional rankings rather than the
day's level, and the target is the raw forward return demeaned within the
cross-section, so the forecast is in return units, as mean-variance sizing
and edge-to-cost need. Three models, each fit on the training month and
applied unchanged to the evaluation month:

* ``reversal_15m``: the 15-minute reversal feature alone, rescaled by its
  training-month slope -- the one-line baseline;
* ``ridge``: :class:`~stable_finance.StreamingRidge` on all twelve features,
  every horizon in one pass, with the ridge penalty chosen per horizon on the
  last week of the training month;
* ``lasso``: scikit-learn's coordinate-descent LASSO per horizon (cyclic, so
  deterministic), penalty chosen the same way.

Writes one forecast file per model for ``run.py`` (``kind: "file"``) and an
IC-by-horizon table.

    uv run --extra experiments python -m experiments.signals
"""

from __future__ import annotations

import os

# One linear-algebra thread, as in run.py: a multithreaded BLAS sums in an
# order that depends on the thread count, which moves the last digits.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

import stable_finance as sf
from experiments.month_cache import load_month

OUT = Path(__file__).resolve().parent / "results" / "signals"
TRAIN, EVALUATE = "2020-07", "2020-08"
HORIZONS = (300, 600, 900, 1800, 3600, 7200)
ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
LASSO_ALPHAS = (1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Per-decision rank-to-normal-score of each feature; NaN stays NaN."""
    out = np.full(values.shape, np.nan)
    normal = NormalDist()
    for t in range(values.shape[0]):
        for k in range(values.shape[2]):
            column = values[t, :, k]
            finite = np.isfinite(column)
            n = int(finite.sum())
            if n < 3:
                continue
            ranks = np.argsort(np.argsort(column[finite], kind="stable"), kind="stable") + 1
            grid = np.array([normal.inv_cdf(r / (n + 1)) for r in range(1, n + 1)])
            out[t, np.flatnonzero(finite), k] = grid[ranks - 1]
    return out


def demeaned(realized: np.ndarray) -> np.ndarray:
    return realized - np.nanmean(realized, axis=1, keepdims=True)


def align(month: dict, assets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features and realized returns of a month on a fixed asset list."""
    index = {a: i for i, a in enumerate(month["realized"].assets.astype(str))}
    take = [index[a] for a in assets]
    horizon_index = [int(np.flatnonzero(month["realized"].horizons == h)[0]) for h in HORIZONS]
    return (month["features"][:, take].astype(np.float64),
            month["realized"].values[:, take][:, :, horizon_index])


def rows(features: np.ndarray, target: np.ndarray):
    """Flatten (decision, asset) into rows with finite features."""
    X = features.reshape(-1, features.shape[2])
    y = target.reshape(-1, target.shape[2])
    keep = np.isfinite(X).all(axis=1)
    return X[keep], y[keep], keep


def rank_ic(forecast: np.ndarray, realized: np.ndarray) -> list[float]:
    return [float(sf.grouped_rank_ic(forecast[:, :, h], realized[:, :, h], min_assets=20).mean)
            for h in range(forecast.shape[2])]


def fit_ridge(X, y, alpha):
    return sf.StreamingRidge(alpha=alpha).partial_fit(X, y).finalize()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train, evaluation = load_month(TRAIN), load_month(EVALUATE)
    assets = np.asarray(sorted(set(train["realized"].assets.astype(str))
                               & set(evaluation["realized"].assets.astype(str))))
    names = evaluation["feature_names"]
    f_train, r_train = align(train, assets)
    f_eval, r_eval = align(evaluation, assets)
    z_train, z_eval = normal_scores(f_train), normal_scores(f_eval)
    y_train = demeaned(r_train)

    # Penalties are chosen on the last week of the training month, fit on the rest.
    days = train["days"]
    cut = np.flatnonzero(days >= sorted(set(days))[-5])[0]
    X_fit, y_fit, _ = rows(z_train[:cut], y_train[:cut])
    shape_val = z_train[cut:].shape

    def validation_ic(predict) -> np.ndarray:
        X_val = z_train[cut:].reshape(-1, len(names))
        pred = np.full((X_val.shape[0], len(HORIZONS)), np.nan)
        ok = np.isfinite(X_val).all(axis=1)
        pred[ok] = predict(X_val[ok])
        return np.array(rank_ic(pred.reshape(shape_val[0], shape_val[1], -1),
                                r_train[cut:]))

    ridge_scores = {a: validation_ic(fit_ridge(X_fit, y_fit, a).predict) for a in ALPHAS}
    def best(scores: dict, grid) -> list:
        """Per horizon, the penalty with the best validation IC. A penalty that
        zeroes every coefficient has no ranking and so a NaN IC; it loses."""
        return [grid[int(np.argmax([np.nan_to_num(scores[a][h], nan=-np.inf)
                                     for a in grid]))] for h in range(len(HORIZONS))]

    ridge_alpha = best(ridge_scores, ALPHAS)

    from sklearn.linear_model import Lasso

    def lasso_predict(alpha, X, y):
        models = []
        for h in range(len(HORIZONS)):
            finite = np.isfinite(y[:, h])
            models.append(Lasso(alpha=alpha, selection="cyclic", max_iter=5000)
                          .fit(X[finite], y[finite, h]))
        return lambda Z: np.column_stack([m.predict(Z) for m in models]), models

    lasso_scores = {a: validation_ic(lasso_predict(a, X_fit, y_fit)[0]) for a in LASSO_ALPHAS}
    lasso_alpha = best(lasso_scores, LASSO_ALPHAS)

    # Refit on the whole training month with the chosen penalties, then forecast.
    X_all, y_all, _ = rows(z_train, y_train)
    X_eval = z_eval.reshape(-1, len(names))
    ok = np.isfinite(X_eval).all(axis=1)
    forecasts = {}

    ridge = fit_ridge(X_all, y_all, ridge_alpha)
    pred = np.full((X_eval.shape[0], len(HORIZONS)), np.nan)
    pred[ok] = ridge.predict(X_eval[ok])
    forecasts["ridge"] = pred

    pred = np.full((X_eval.shape[0], len(HORIZONS)), np.nan)
    lasso_models = []
    for h in range(len(HORIZONS)):
        _, models = lasso_predict(lasso_alpha[h], X_all, y_all)
        lasso_models.append(models[h])
        pred[ok, h] = models[h].predict(X_eval[ok])
    forecasts["lasso"] = pred

    # The one-feature baseline: 15-minute reversal times its training slope.
    k = names.index("rev_15m")
    slope = []
    for h in range(len(HORIZONS)):
        finite = np.isfinite(y_all[:, h])
        x = X_all[finite, k]
        slope.append(float(x @ y_all[finite, h] / (x @ x)))
    pred = np.full((X_eval.shape[0], len(HORIZONS)), np.nan)
    pred[ok] = X_eval[ok, k][:, None] * np.asarray(slope)[None, :]
    forecasts["reversal_15m"] = pred

    table = {"horizons": list(HORIZONS), "train": TRAIN, "evaluate": EVALUATE,
             "assets": len(assets), "ridge_alpha": ridge_alpha, "lasso_alpha": lasso_alpha,
             "ic": {}, "ridge_coefficients": {}, "lasso_coefficients": {}}
    decisions = evaluation["realized"].decisions
    for name, values in forecasts.items():
        cube = values.reshape(len(decisions), len(assets), len(HORIZONS))
        table["ic"][name] = rank_ic(cube, r_eval)
        np.savez(OUT / f"{name}.npz", forecast=cube, assets=assets,
                 horizons=np.asarray(HORIZONS), decisions=decisions)
    # coefficients_ is (horizons, features).
    table["ridge_coefficients"] = {n: [float(c) for c in ridge.coefficients_[:, i]]
                                   for i, n in enumerate(names)}
    table["lasso_coefficients"] = {n: [float(m.coef_[i]) for m in lasso_models]
                                   for i, n in enumerate(names)}
    table["single_feature_ic"] = {
        n: rank_ic(np.repeat(z_eval[:, :, i:i + 1], len(HORIZONS), axis=2), r_eval)
        for i, n in enumerate(names)}
    (OUT / "ic_table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps({"ic": table["ic"], "ridge_alpha": ridge_alpha,
                      "lasso_alpha": lasso_alpha}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
