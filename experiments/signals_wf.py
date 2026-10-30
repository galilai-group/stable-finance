"""Real signals refitted every day, with purged and embargoed cross-validation.

``signals.py`` fits ridge and LASSO once on July and applies them to all of
August, with the penalty picked on July's last week. That is honest but
coarse: the model never sees August, and one validation week picks the
penalty. This script is the walk-forward protocol (Lopez de Prado 2018,
"Advances in Financial Machine Learning", ch. 7): for each evaluation day the
models are fitted on every day strictly before it, the penalty is chosen by
K-fold cross-validation over contiguous blocks of training days, each fold's
training set PURGED of observations whose labels overlap the fold and
EMBARGOED for one day after it (longer than the longest horizon, two hours),
and the day is then forecast out of sample.

Both models are fitted from per-day sufficient statistics (counts, sums,
cross-products of features and targets), so a refit is a 12-by-12 solve
rather than a pass over a million rows: ridge in closed form, LASSO by cyclic
coordinate descent on the Gram matrix with a fixed order and tolerance, both
deterministic. Features are the month cache's causal features normal-scored
per decision, plus order-flow imbalance over 60 and 300 seconds when a v2
cache exists (``--ofi``); targets are forward returns demeaned across names.
The evaluation is rank IC by horizon, with 95% intervals from a bootstrap
over days, beside the month-ahead fit's (``results/signals/ic_table.json``)
when that exists.

    uv run --extra experiments python -m experiments.signals_wf [--window N] [--ofi]

``--window`` keeps only the last N training days (default: all, expanding).
Writes ``results/signals_wf/{ridge_wf,lasso_wf}.npz`` (usable as
``"kind": "file"`` forecasts) and ``ic_table.json``.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.month_cache import load_month
from experiments.signals import HORIZONS, align, demeaned, normal_scores

OUT = Path(__file__).resolve().parent / "results" / "signals_wf"
TRAIN, EVALUATE = "2020-07", "2020-08"
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
LASSO_ALPHAS = (1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
FOLDS = 5
EMBARGO_DAYS = 1
#: Penalties are re-chosen every CV_EVERY evaluation days (models are still
#: refitted daily), and validation reads every VALIDATE_EVERY-th decision:
#: rank ICs dominate the cost, and neither choice lets the future in.
CV_EVERY = 5
VALIDATE_EVERY = 4
BOOTSTRAP_DRAWS = 2000


# ── data ───────────────────────────────────────────────────────────────────

def load(use_ofi: bool) -> dict:
    """Both months on one asset list: normal-scored features, demeaned targets, raw returns."""
    train, test = load_month(TRAIN), load_month(EVALUATE)
    assets = np.asarray(sorted(set(train["realized"].assets.astype(str))
                               & set(test["realized"].assets.astype(str))))
    names = list(test["feature_names"])
    parts = []
    for month, label in ((train, TRAIN), (test, EVALUATE)):
        features, realized = align(month, assets)
        if use_ofi:
            features, ofi_names = _with_ofi(features, label, assets)
        parts.append((features, realized, month["days"], month["realized"].decisions))
    if use_ofi:
        names += ofi_names
    features = np.concatenate([p[0] for p in parts])
    realized = np.concatenate([p[1] for p in parts])
    days = np.concatenate([p[2] for p in parts])
    decisions = np.concatenate([p[3] for p in parts])
    return {"z": normal_scores(features), "y": demeaned(realized), "realized": realized,
            "days": days, "decisions": decisions, "assets": assets, "names": names,
            "eval_days": sorted(set(test["days"].tolist()))}


def _with_ofi(features: np.ndarray, month: str, assets: np.ndarray):
    from experiments.month_cache_v2 import OFI_NAMES, load_month_v2, month_path

    if not month_path(month).is_file():
        raise SystemExit(f"--ofi needs the v2 cache for {month}; build it with "
                         "python -m experiments.month_cache_v2")
    v2 = load_month_v2(month)
    index = {a: i for i, a in enumerate(v2["realized"].assets.astype(str))}
    take = [index[a] for a in assets]
    # v2's features are v1's twelve followed by the OFI columns, named in order.
    columns = [list(v2["feature_names"]).index(name) for name in OFI_NAMES]
    extra = v2["features"][:, take][:, :, columns].astype(np.float64)
    return np.concatenate([features, extra], axis=2), list(OFI_NAMES)


def day_statistics(z: np.ndarray, y: np.ndarray, days: np.ndarray) -> dict:
    """Per day: n, sum x, sum y (per horizon), x'x, x'y over rows with finite x and y."""
    labels = sorted(set(days.tolist()))
    k, h = z.shape[2], y.shape[2]
    stats = {}
    for day in labels:
        rows = days == day
        X = z[rows].reshape(-1, k)
        Y = y[rows].reshape(-1, h)
        ok = np.isfinite(X).all(axis=1)
        X, Y = X[ok], Y[ok]
        finite = np.isfinite(Y)
        Yz = np.where(finite, Y, 0.0)
        # Per-horizon counts and moments: a row missing one horizon's label
        # still counts for the others.
        n = finite.sum(axis=0).astype(np.float64)
        sx = finite.T.astype(np.float64) @ X                       # (h, k)
        xx = np.einsum("rh,ri,rj->hij", finite.astype(np.float64), X, X)
        stats[day] = {"n": n, "sx": sx, "sy": Yz.sum(axis=0), "xx": xx,
                      "xy": np.einsum("ri,rh->hi", X, Yz)}
    return stats


def pooled(stats: dict, days) -> dict:
    out = None
    for day in days:
        s = stats[day]
        out = {k: v.copy() for k, v in s.items()} if out is None else {
            k: out[k] + s[k] for k in out}
    return out


# ── models from sufficient statistics ──────────────────────────────────────

def _centered(p: dict, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Covariance of x and of (x, y) for horizon h, and the means."""
    n = max(p["n"][h], 1.0)
    mx = p["sx"][h] / n
    my = p["sy"][h] / n
    cxx = p["xx"][h] / n - np.outer(mx, mx)
    cxy = p["xy"][h] / n - mx * my
    return cxx, cxy, mx, my


def ridge_fit(p: dict, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients (h, k) and intercepts (h,); alpha penalizes ``n * |b|^2``-scaled."""
    k = p["xx"].shape[1]
    coef, intercept = [], []
    for h in range(p["xx"].shape[0]):
        cxx, cxy, mx, my = _centered(p, h)
        b = np.linalg.solve(cxx + alpha / max(p["n"][h], 1.0) * np.eye(k), cxy)
        coef.append(b)
        intercept.append(my - mx @ b)
    return np.array(coef), np.array(intercept)


def lasso_fit(p: dict, alpha: float, *, max_iter: int = 2000, tol: float = 1e-10
              ) -> tuple[np.ndarray, np.ndarray]:
    """``min (1/2) b'Cb - c'b + alpha |b|_1`` by cyclic coordinate descent, fixed order.

    That is sklearn's Lasso objective ``(1/2n) |y - Xb|^2 + alpha |b|_1`` on
    centered data, written in the Gram form so it needs only the pooled
    moments.
    """
    k = p["xx"].shape[1]
    coef, intercept = [], []
    for h in range(p["xx"].shape[0]):
        C, c, mx, my = _centered(p, h)
        b = np.zeros(k)
        for _ in range(max_iter):
            largest = 0.0
            for j in range(k):
                if C[j, j] <= 0:
                    continue
                rho = c[j] - C[j] @ b + C[j, j] * b[j]
                new = np.sign(rho) * max(abs(rho) - alpha, 0.0) / C[j, j]
                largest = max(largest, abs(new - b[j]))
                b[j] = new
            if largest < tol:
                break
        coef.append(b)
        intercept.append(my - mx @ b)
    return np.array(coef), np.array(intercept)


def predict(z: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """``(decisions, assets, h)`` forecasts; NaN where any feature is missing."""
    out = np.einsum("dak,hk->dah", np.nan_to_num(z), coef) + intercept
    return np.where(np.isfinite(z).all(axis=2, keepdims=True), out, np.nan)


def rank_ics(forecast: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """``(decisions, h)`` cross-sectional rank IC; NaN with fewer than 10 names."""
    d, _, h = forecast.shape
    out = np.full((d, h), np.nan)
    for t in range(d):
        for j in range(h):
            f, r = forecast[t, :, j], realized[t, :, j]
            ok = np.isfinite(f) & np.isfinite(r)
            if ok.sum() < 10 or np.ptp(f[ok]) == 0:
                continue
            fr = np.argsort(np.argsort(f[ok]))
            rr = np.argsort(np.argsort(r[ok]))
            out[t, j] = np.corrcoef(fr, rr)[0, 1]
    return out


# ── the protocol ───────────────────────────────────────────────────────────

def choose_alpha(data: dict, stats: dict, train_days: list, fit, grid) -> list:
    """Per horizon, the penalty with the best mean validation rank IC over purged, embargoed folds."""
    blocks = np.array_split(np.arange(len(train_days)), FOLDS)
    scores = np.zeros((len(grid), len(HORIZONS)))
    counts = np.zeros((len(grid), len(HORIZONS)))
    for block in blocks:
        if len(block) == 0:
            continue
        held = [train_days[i] for i in block]
        # Purge the fold itself and embargo the days right after it: labels
        # run at most two hours within a day, so one day clears any overlap.
        embargo = {train_days[i] for i in range(block[-1] + 1,
                                                min(block[-1] + 1 + EMBARGO_DAYS,
                                                    len(train_days)))}
        fit_days = [d for d in train_days if d not in held and d not in embargo]
        if not fit_days:
            continue
        p = pooled(stats, fit_days)
        rows = np.isin(data["days"], held) & (np.arange(len(data["days"])) % VALIDATE_EVERY == 0)
        for a, alpha in enumerate(grid):
            coef, intercept = fit(p, alpha)
            ic = rank_ics(predict(data["z"][rows], coef, intercept), data["realized"][rows])
            mean = np.nanmean(ic, axis=0)
            ok = np.isfinite(mean)
            scores[a, ok] += mean[ok]
            counts[a, ok] += 1
    mean = np.where(counts > 0, scores / np.maximum(counts, 1), -np.inf)
    return [grid[int(np.argmax(mean[:, h]))] for h in range(len(HORIZONS))]


def fit_per_horizon(p: dict, alphas: list, fit) -> tuple[np.ndarray, np.ndarray]:
    coef, intercept = [], []
    for h, alpha in enumerate(alphas):
        c, i = fit(p, alpha)
        coef.append(c[h])
        intercept.append(i[h])
    return np.array(coef), np.array(intercept)


def bootstrap_ci(ic: np.ndarray, days: np.ndarray) -> list[list[float]]:
    """95% intervals of the mean IC per horizon, resampling whole days (seeded)."""
    labels, index = np.unique(days, return_inverse=True)
    daily = np.array([[np.nanmean(ic[index == i, h]) for h in range(ic.shape[1])]
                      for i in range(len(labels))])
    draws = np.random.default_rng(0).integers(0, len(labels), (BOOTSTRAP_DRAWS, len(labels)))
    boot = np.nanmean(daily[draws], axis=1)
    return [[float(a), float(b)] for a, b in zip(*np.percentile(boot, [2.5, 97.5], axis=0))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=None,
                        help="trailing training days (default: all prior days)")
    parser.add_argument("--ofi", action="store_true", help="add OFI features from the v2 cache")
    parser.add_argument("--days", type=int, default=None,
                        help="only the first N evaluation days (for a quick check)")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(args.ofi)
    stats = day_statistics(data["z"], data["y"], data["days"])
    all_days = sorted(stats)
    eval_days = data["eval_days"][:args.days] if args.days else data["eval_days"]
    shape = (int(np.isin(data["days"], eval_days).sum()), len(data["assets"]), len(HORIZONS))
    forecasts = {"ridge_wf": np.full(shape, np.nan), "lasso_wf": np.full(shape, np.nan)}
    chosen = {"ridge_wf": {}, "lasso_wf": {}}
    eval_rows = np.isin(data["days"], eval_days)
    offset = {d: i for i, d in enumerate(np.flatnonzero(eval_rows))}
    alphas_now = {}
    for i, day in enumerate(eval_days):
        train_days = [d for d in all_days if d < day]
        if args.window:
            train_days = train_days[-args.window:]
        p = pooled(stats, train_days)
        rows = np.flatnonzero(data["days"] == day)
        target = [offset[r] for r in rows]
        for name, fit, grid in (("ridge_wf", ridge_fit, RIDGE_ALPHAS),
                                ("lasso_wf", lasso_fit, LASSO_ALPHAS)):
            if i % CV_EVERY == 0:
                alphas_now[name] = choose_alpha(data, stats, train_days, fit, grid)
            alphas = alphas_now[name]
            coef, intercept = fit_per_horizon(p, alphas, fit)
            forecasts[name][target] = predict(data["z"][rows], coef, intercept)
            chosen[name][day] = alphas
        print(f"  {day}: ridge {chosen['ridge_wf'][day]}, lasso {chosen['lasso_wf'][day]} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    realized = data["realized"][eval_rows]
    days = data["days"][eval_rows]
    decisions = data["decisions"][eval_rows]
    table = {"horizons": list(HORIZONS), "evaluate_days": eval_days, "window": args.window,
             "ofi": args.ofi, "features": data["names"], "folds": FOLDS,
             "embargo_days": EMBARGO_DAYS, "cv_every_days": CV_EVERY,
             "validate_every_decisions": VALIDATE_EVERY, "chosen_alphas": chosen,
             "ic": {}, "ic_ci": {}}
    for name, cube in forecasts.items():
        ic = rank_ics(cube, realized)
        table["ic"][name] = [float(v) for v in np.nanmean(ic, axis=0)]
        table["ic_ci"][name] = bootstrap_ci(ic, days)
        np.savez(OUT / f"{name}.npz", forecast=cube, assets=data["assets"],
                 horizons=np.asarray(HORIZONS), decisions=decisions)
    month_ahead = OUT.parent / "signals" / "ic_table.json"
    if month_ahead.is_file():
        table["month_ahead_ic"] = json.loads(month_ahead.read_text()).get("ic", {})
    table["seconds"] = time.perf_counter() - t0
    (OUT / "ic_table.json").write_text(json.dumps(table, indent=2))
    print("out-of-sample rank IC by horizon " + str(list(HORIZONS)))
    for name in forecasts:
        cells = ", ".join(f"{v:.4f} [{lo:.4f}, {hi:.4f}]" for v, (lo, hi) in
                          zip(table["ic"][name], table["ic_ci"][name]))
        print(f"  {name}: {cells}")
    for name, ics in table.get("month_ahead_ic", {}).items():
        print(f"  {name} (month-ahead): " + ", ".join(f"{v:.4f}" for v in ics))
    print(f"-> {OUT} ({table['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
