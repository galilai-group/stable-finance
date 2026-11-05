"""Real forecasts without a GPU: causal intraday features, fit on one month.

The noised oracle says what a forecast of a given IC is worth; these say what
IC cheap, well-known intraday signals actually have on Market-1T, so the real
models can be placed on the oracle's IC-to-Sharpe map. No GPU and no learned
representation: the features are the ones ``month_cache.py`` caches, all read from
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

A feature with no value at a decision -- the 30-minute reversal and
volatility before 10:00, say -- enters the fit and the forecast as zero, the
neutral normal score, so the first half hour of each session is forecast from
the features it has rather than left unscored. A forecast is kept only where
the model reads a feature the row really has: on a row whose every feature
with a nonzero coefficient was imputed, the forecast is the intercept alone,
one number for every such name, which ranks nobody and which a book would
trade as one long market bet or an alphabetical one. So ``reversal_15m`` has
no forecast before 09:45 (``rev_15m`` does not exist yet), and a LASSO horizon
whose surviving features all start at 10:00 has none before then; as a
backstop, a decision whose forecast is one value across names is dropped too.
At 09:30 itself the features that do exist are the opening cross's quotes
(the five-second imbalance and the half-spread), so a book on ``ridge``
trades the open on those alone. The table reports the share of rows forecast
at each time of day, per horizon.

COMPARABILITY. ``ic`` covers every decision, 09:30-16:00. ``ic_from_1000``
drops the decisions before 10:00, and ``ic_complete_rows`` keeps only the
rows with every feature observed -- the rows the earlier table scored and the
rows ``signals_wf`` forecasts. The ridge grid widened (to 1e7) in the same
change as the imputation, so ``ridge_published_method`` refits ridge the
earlier way (complete rows only) on the old grid and on the new one, and
``ic_effects["ridge"]`` splits the change into the penalty, fitting on
imputed rows, and scoring the extra rows. LASSO's grid did not change and
reversal has no penalty: for them the change from the earlier table is the
fit (``ic_complete_rows`` against it) plus the coverage (``ic`` against
``ic_complete_rows``).

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
ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0)
#: The ridge grid the earlier table chose from, to separate the penalty's effect.
OLD_ALPHAS = ALPHAS[:5]
LASSO_ALPHAS = (1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
#: 10:00, in seconds after the open: the 30-minute features exist from here on.
FROM_1000 = 1800
#: A cross-section whose range is this small relative to its level is one value
#: plus rounding noise, as in ParametricPolicy, not a ranking.
FLAT = 1e-12


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


def impute(scores: np.ndarray) -> np.ndarray:
    """Missing normal scores as zero, the neutral score, where a row has any feature at all.

    The scores are per-decision cross-sectional ranks, so zero is the middle
    of every feature's distribution: a missing feature then says nothing about
    the name, and the row is forecast from the features it has. A row with no
    finite feature stays all NaN and gets no forecast. Causal: it reads only
    the decision's own row.
    """
    any_feature = np.isfinite(scores).any(axis=-1, keepdims=True)
    return np.where(any_feature, np.nan_to_num(scores, nan=0.0), np.nan)


def offsets(decisions) -> np.ndarray:
    """Each decision's time of day, ``"day|seconds after the open"``, as its offset string."""
    return np.asarray([str(d).split("|", 1)[1] for d in np.asarray(decisions).astype(str)])


def coverage_by_time(forecast: np.ndarray, decisions) -> dict[str, list[float]]:
    """Per time of day (seconds after the open), the share of rows with a forecast, per horizon."""
    times = offsets(decisions)
    has = np.isfinite(forecast)
    return {o: [float(v) for v in has[times == o].mean(axis=(0, 1))]
            for o in sorted(set(times), key=int)}


def uninformed(observed: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """``(rows, horizons)``: True where the row has none of the features the model reads there.

    ``observed`` is ``(rows, k)``, True for a feature that had a value before
    imputation; ``coef`` is ``(horizons, k)``. A feature is read at a horizon
    when its coefficient there is nonzero.
    """
    read = (np.asarray(coef) != 0).T.astype(np.float64)
    return np.asarray(observed, dtype=np.float64) @ read == 0


def forecast(model: tuple, X: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """``(rows, horizons)`` forecasts of rows ``X``; NaN where a row cannot be forecast.

    ``model`` is ``(predict, coef)``, coef ``(horizons, k)``. A row is
    forecast when every column of ``X`` is finite (after :func:`impute`, when
    it has any feature) and, at each horizon, it has a feature the model
    reads (:func:`uninformed`); otherwise the forecast would be the intercept
    alone, the same for every such name.
    """
    predict, coef = model[0], np.asarray(model[1])
    out = np.full((X.shape[0], coef.shape[0]), np.nan)
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        out[ok] = predict(X[ok])
    out[uninformed(observed, coef)] = np.nan
    return out


def without_flat(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The forecast cube with every one-valued cross-section set to NaN, and how many per horizon.

    A forecast equal for every name ranks nobody; a book sized on it puts the
    whole gross on one side or sorts by the asset index.
    """
    finite = np.isfinite(cube)
    some = finite.any(axis=1)
    high = np.where(finite, cube, -np.inf).max(axis=1)
    low = np.where(finite, cube, np.inf).min(axis=1)
    level = np.where(some, np.maximum(np.abs(high), np.abs(low)), 0.0)
    flat = some & (np.where(some, high - low, 0.0) <= FLAT * level)
    return np.where(flat[:, None, :], np.nan, cube), flat.sum(axis=0)


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


def fit_lasso(alpha: float, X: np.ndarray, y: np.ndarray, h: int):
    """scikit-learn's cyclic (deterministic) LASSO on one horizon's labelled rows."""
    from sklearn.linear_model import Lasso

    finite = np.isfinite(y[:, h])
    return Lasso(alpha=alpha, selection="cyclic", max_iter=5000).fit(X[finite], y[finite, h])


def stacked(models) -> tuple:
    """One predict over per-horizon models, and their ``(horizons, k)`` coefficients."""
    return (lambda Z: np.column_stack([m.predict(Z) for m in models]),
            np.array([m.coef_ for m in models]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train, evaluation = load_month(TRAIN), load_month(EVALUATE)
    assets = np.asarray(sorted(set(train["realized"].assets.astype(str))
                               & set(evaluation["realized"].assets.astype(str))))
    names = evaluation["feature_names"]
    f_train, r_train = align(train, assets)
    f_eval, r_eval = align(evaluation, assets)
    raw_train, raw_eval = normal_scores(f_train), normal_scores(f_eval)
    z_train, z_eval = impute(raw_train), impute(raw_eval)
    y_train = demeaned(r_train)
    k_all = len(names)

    # Penalties are chosen on the last week of the training month, fit on the rest.
    days = train["days"]
    cut = np.flatnonzero(days >= sorted(set(days))[-5])[0]
    X_fit, y_fit, _ = rows(z_train[:cut], y_train[:cut])
    shape_val = z_train[cut:].shape
    seen_val = np.isfinite(raw_train[cut:]).reshape(-1, k_all)

    def validation_ic(model, scores=z_train) -> np.ndarray:
        """Validation-week IC of ``model = (predict, coef)``, forecast as the evaluation is."""
        X_val = scores[cut:].reshape(-1, k_all)
        pred = forecast(model, X_val, seen_val)
        cube, _ = without_flat(pred.reshape(shape_val[0], shape_val[1], -1))
        return np.array(rank_ic(cube, r_train[cut:]))

    def ridge_model(X, y, alpha):
        fitted = fit_ridge(X, y, alpha)
        return fitted.predict, fitted.coefficients_

    def best(scores: dict, grid) -> list:
        """Per horizon, the penalty with the best validation IC. A penalty that
        zeroes every coefficient has no ranking and so a NaN IC; it loses."""
        return [grid[int(np.argmax([np.nan_to_num(scores[a][h], nan=-np.inf)
                                     for a in grid]))] for h in range(len(HORIZONS))]

    ridge_scores = {a: validation_ic(ridge_model(X_fit, y_fit, a)) for a in ALPHAS}
    ridge_alpha = best(ridge_scores, ALPHAS)

    def lasso_model(alpha, X, y):
        return stacked([fit_lasso(alpha, X, y, h) for h in range(len(HORIZONS))])

    lasso_scores = {a: validation_ic(lasso_model(a, X_fit, y_fit)) for a in LASSO_ALPHAS}
    lasso_alpha = best(lasso_scores, LASSO_ALPHAS)

    # Refit on the whole training month with the chosen penalties, then forecast.
    X_all, y_all, _ = rows(z_train, y_train)
    X_eval = z_eval.reshape(-1, k_all)
    seen_eval = np.isfinite(raw_eval).reshape(-1, k_all)
    models = {}

    ridge = fit_ridge(X_all, y_all, ridge_alpha)
    models["ridge"] = (ridge.predict, ridge.coefficients_)
    # Each horizon is fitted once, at its own penalty.
    lasso_models = [fit_lasso(lasso_alpha[h], X_all, y_all, h) for h in range(len(HORIZONS))]
    models["lasso"] = stacked(lasso_models)

    # The one-feature baseline: 15-minute reversal times its training slope.
    # An imputed zero adds nothing to x'y or x'x, so the slope is the one on
    # the rows where rev_15m was observed, and only those rows are forecast.
    k = names.index("rev_15m")
    reversal = np.zeros((len(HORIZONS), k_all))
    for h in range(len(HORIZONS)):
        finite = np.isfinite(y_all[:, h])
        x = X_all[finite, k]
        reversal[h, k] = float(x @ y_all[finite, h] / (x @ x))
    models["reversal_15m"] = (lambda Z: Z @ reversal.T, reversal)

    # The earlier method, for ridge: fit and score complete rows only, on the
    # old penalty grid and on the new one.
    Xc_fit, yc_fit, _ = rows(raw_train[:cut], y_train[:cut])
    complete_scores = {a: validation_ic(ridge_model(Xc_fit, yc_fit, a), raw_train)
                       for a in ALPHAS}
    Xc_all, yc_all, _ = rows(raw_train, y_train)
    X_eval_raw = raw_eval.reshape(-1, k_all)
    shape = (len(evaluation["realized"].decisions), len(assets), len(HORIZONS))
    published = {}
    for label, grid in (("old_grid", OLD_ALPHAS), ("new_grid", ALPHAS)):
        alphas = best(complete_scores, grid)
        model = ridge_model(Xc_all, yc_all, alphas)
        cube, _ = without_flat(forecast(model, X_eval_raw, seen_eval).reshape(shape))
        published[label] = {"alpha": alphas, "ic": rank_ic(cube, r_eval)}

    table = {"horizons": list(HORIZONS), "train": TRAIN, "evaluate": EVALUATE,
             "assets": len(assets), "ridge_alpha": ridge_alpha, "lasso_alpha": lasso_alpha,
             "ridge_alpha_grid": list(ALPHAS),
             "ridge_alpha_at_grid_edge": [a in (ALPHAS[0], ALPHAS[-1]) for a in ridge_alpha],
             "ic": {}, "ic_from_1000": {}, "ic_complete_rows": {},
             "ridge_published_method": published, "coverage": {}, "coverage_by_time": {},
             "flat_decisions_dropped": {}, "ridge_coefficients": {}, "lasso_coefficients": {}}
    decisions = evaluation["realized"].decisions
    late = offsets(decisions).astype(int) >= FROM_1000
    complete = np.isfinite(raw_eval).all(axis=2, keepdims=True)
    for name, model in models.items():
        cube, flat = without_flat(forecast(model, X_eval, seen_eval).reshape(shape))
        table["ic"][name] = rank_ic(cube, r_eval)
        table["ic_from_1000"][name] = rank_ic(cube[late], r_eval[late])
        table["ic_complete_rows"][name] = rank_ic(np.where(complete, cube, np.nan), r_eval)
        table["coverage"][name] = [float(v) for v in np.isfinite(cube).mean(axis=(0, 1))]
        table["coverage_by_time"][name] = coverage_by_time(cube, decisions)
        table["flat_decisions_dropped"][name] = [int(v) for v in flat]
        np.savez(OUT / f"{name}.npz", forecast=cube, assets=assets,
                 horizons=np.asarray(HORIZONS), decisions=decisions)
    old, new = (np.asarray(published[g]["ic"]) for g in ("old_grid", "new_grid"))
    on_complete = np.asarray(table["ic_complete_rows"]["ridge"])
    table["ic_effects"] = {"ridge": {
        "penalty": (new - old).tolist(),
        "imputed_fit": (on_complete - new).tolist(),
        "coverage": (np.asarray(table["ic"]["ridge"]) - on_complete).tolist()}}
    # coefficients_ is (horizons, features).
    table["ridge_coefficients"] = {n: [float(c) for c in ridge.coefficients_[:, i]]
                                   for i, n in enumerate(names)}
    table["lasso_coefficients"] = {n: [float(m.coef_[i]) for m in lasso_models]
                                   for i, n in enumerate(names)}
    table["single_feature_ic"] = {
        n: rank_ic(np.repeat(raw_eval[:, :, i:i + 1], len(HORIZONS), axis=2), r_eval)
        for i, n in enumerate(names)}
    (OUT / "ic_table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps({"ic": table["ic"], "ic_complete_rows": table["ic_complete_rows"],
                      "ic_effects": table["ic_effects"], "ridge_alpha": ridge_alpha,
                      "lasso_alpha": lasso_alpha,
                      "flat_decisions_dropped": table["flat_decisions_dropped"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
