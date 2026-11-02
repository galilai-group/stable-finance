"""Train a deterministic trading policy on one month and test it on the next.

The harness's hand-written books size every decision from that decision's
forecast alone and leave turnover to whatever the forecast does. The
:class:`~stable_finance.ParametricPolicy` instead LEARNS, on the training
month, how to blend characteristics and how fast to trade toward the blended
target, by maximizing net-of-cost utility directly (Brandt, Santa-Clara and
Valkanov, 2009; DeMiguel et al., 2020; Garleanu and Pedersen, 2013). This
script fits it on July 2020 and scores it on August 2020 through the same
execution model and scoring as everything else, against baselines on the
same characteristics:

* ``signal``: the signal-weighted book on the equal blend of the
  characteristics, rebuilt every decision (the policy at eta = 1 and equal
  theta);
* ``signal, held``: the same book held for the slowest horizon's length;
* ``signal, learned eta``: the same book traded toward at the policy's own
  learned rate, so what learning theta adds beyond trading slowly shows;
* ``no position``: nothing held, the zero every net Sharpe is read against.

For the features, each characteristic enters the blend with the sign of its
rank IC on the training month, so the baseline is not hurt by a feature that
predicts with the opposite sign. Every book is treated the same at the
session boundary: a decision without a forecast (the last ones of a day,
whose horizon runs past the close) keeps the book it has, as the learned
policy does, and no book earns the overnight return, which is not in the
data. The fit runs to convergence (up to ``MAX_ITER`` iterations), and the
report gives the objective path's last change.

Two characteristic sets:

* ``oracle``: copula oracles (``copula_forecast``) at 5, 15 and 60 minutes
  with ICs 0.03, 0.02 and 0.01, errors with a 6-decision half-life -- a slow,
  decaying multi-horizon forecast whose true quality is known;
* ``features``: the twelve causal features of ``month_cache.py``,
  normal-scored per decision.

Each holds for one decision (five minutes) and earns the next five minutes'
return, so nothing overlaps; the cost is the half-spread at EFQ 70%. Scores
include the daily Sharpe with its day-block bootstrap interval.

    uv run --extra experiments python -m experiments.policy
"""

from __future__ import annotations

import os

# One linear-algebra thread, as in run.py: a multithreaded BLAS sums in an
# order that depends on the thread count, which moves the last digits.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

import stable_finance as sf
from experiments.month_cache import load_month
from experiments.signals import normal_scores

OUT = Path(__file__).resolve().parent / "results" / "policy"
TRAIN, EVALUATE = "2020-07", "2020-08"
EFQ = 0.7
ORACLE_HORIZONS, ORACLE_ICS, HALF_LIFE = (300, 900, 3600), (0.03, 0.02, 0.01), 6
#: Iteration cap for the fit; the earlier 400 had not converged.
MAX_ITER = 3000


def month_inputs(month: str, assets: np.ndarray) -> dict:
    data = load_month(month)
    realized = data["realized"]
    index = {a: i for i, a in enumerate(realized.assets.astype(str))}
    take = [index[a] for a in assets]
    hz = [int(np.flatnonzero(realized.horizons == h)[0]) for h in ORACLE_HORIZONS]
    panel = sf.ForwardReturns(realized.values[:, take][:, :, hz], realized.decisions,
                              assets, np.asarray(ORACLE_HORIZONS))
    oracle = sf.copula_forecast(panel, np.asarray(ORACLE_ICS), half_life=HALF_LIFE,
                                days=data["days"], seed=0)
    spread = data["half_spread"][:, take]
    market = sf.MarketData({q: data["market"].get(q)[:, take]
                            for q in ("bid_price", "ask_price", "exec_bid_price",
                                      "exec_ask_price")}, realized.decisions, assets)
    return {"one_step": panel.values[:, :, 0], "oracle": oracle.values,
            "features": normal_scores(data["features"][:, take].astype(np.float64)),
            "cost": spread * EFQ, "market": market, "days": data["days"],
            "realized": sf.ForwardReturns(panel.values[:, :, :1], realized.decisions, assets,
                                          np.asarray([300]))}


def score(weights: np.ndarray, month: dict) -> dict:
    from experiments.run import _inference

    target = sf.PortfolioWeights(weights[:, :, None], month["realized"].decisions,
                                 month["realized"].assets, np.asarray([300]))
    config = sf.DEFAULT_BACKTEST.replace(cost_model="cross", efq=EFQ, annualization="decisions")
    report = sf.run_backtest(target, month["realized"], market=month["market"],
                             days=month["days"], config=config, with_series=True)
    mid = sf.run_backtest(target, month["realized"], market=month["market"],
                          days=month["days"], config=config.replace(efq=0.0))
    return {"sharpe_mid": mid.sharpe_mid, "sharpe_net": report.sharpe_net,
            "mean_gross_bps": report.mean_gross_return_bps,
            "mean_cost_bps": report.mean_cost_bps, "turnover": report.mean_turnover,
            **_inference(report.series)}


def carry(weights: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Keep the previous decision's book wherever ``missing`` (no forecast at that decision)."""
    out = weights.copy()
    for t in range(1, len(out)):
        if missing[t]:
            out[t] = out[t - 1]
    return out


def feature_signs(X: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Each characteristic's sign: that of its mean per-decision rank IC on the given month."""
    signs = np.ones(X.shape[2])
    for k in range(X.shape[2]):
        value = np.nanmean([_spearman(X[t, :, k], realized[t]) for t in range(len(X))])
        signs[k] = -1.0 if value < 0 else 1.0
    return signs


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def signal_book(X: np.ndarray, gross: float = 2.0, signs: np.ndarray | None = None) -> np.ndarray:
    """The signal-weighted, dollar-neutral book on the (sign-aligned) equal blend of X.

    A decision with no characteristic at all keeps the previous book, as the
    learned policy does, rather than going flat.
    """
    if signs is not None:
        X = X * signs
    blend = np.nanmean(X, axis=2)
    missing = ~np.isfinite(blend).any(axis=1)
    blend = blend - np.nanmean(blend, axis=1, keepdims=True)
    blend = np.nan_to_num(blend)
    total = np.abs(blend).sum(axis=1, keepdims=True)
    book = np.where(total > 0, gross * blend / np.where(total > 0, total, 1), 0.0)
    return carry(book, missing)


def held(weights: np.ndarray, days: np.ndarray, every: int) -> np.ndarray:
    index = np.arange(len(days))
    starts = np.r_[True, days[1:] != days[:-1]]
    first = np.maximum.accumulate(np.where(starts, index, 0))
    formed = (index - first) % every == 0
    anchor = np.maximum.accumulate(np.where(formed, index, 0))
    return weights[anchor]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = load_month(TRAIN)["realized"].assets, load_month(EVALUATE)["realized"].assets
    assets = np.asarray(sorted(set(a.astype(str)) & set(b.astype(str))))
    train, test = month_inputs(TRAIN, assets), month_inputs(EVALUATE, assets)
    results = {}
    for name in ("oracle", "features"):
        t0 = time.perf_counter()
        policy = sf.ParametricPolicy(risk_aversion=5.0, max_iter=MAX_ITER, learning_rate=0.05)
        policy.fit(train[name], np.nan_to_num(train["one_step"]), cost=train["cost"])
        fit_seconds = time.perf_counter() - t0
        learned = policy.predict(test[name], cost=test["cost"])
        signs = feature_signs(train[name], train["one_step"]) if name == "features" else None
        baseline = signal_book(test[name], signs=signs)
        slow = sf.partial_adjustment(sf.PortfolioWeights(
            baseline[:, :, None], test["realized"].decisions, test["realized"].assets,
            np.asarray([300])), float(policy.eta_)).values[:, :, 0]
        rows = {
            "learned policy": score(learned, test),
            "signal, rebuilt each decision": score(baseline, test),
            "signal, held 12 decisions": score(held(baseline, test["days"], 12), test),
            "signal, learned eta": score(slow, test),
            "no position": {"sharpe_mid": 0.0, "sharpe_net": 0.0, "mean_gross_bps": 0.0,
                            "mean_cost_bps": 0.0, "turnover": 0.0},
        }
        path = np.asarray(policy.objective_path_)
        tail = path[-min(100, len(path) - 1) - 1:]
        results[name] = {
            "rows": rows, "theta": [float(x) for x in policy.theta_],
            "feature_signs": None if signs is None else [float(s) for s in signs],
            "eta": float(policy.eta_), "fit_seconds": fit_seconds, "iterations": len(path) - 1,
            "objective_start": float(path[0]), "objective_end": float(path[-1]),
            "objective_change_last_100": float(tail[-1] - tail[0]),
            "converged": bool(abs(tail[-1] - tail[0]) <= 1e-6 * max(abs(tail[-1]), 1e-12)),
            "monotone": bool(np.all(np.diff(path) >= -1e-12)),
        }
        print(name, json.dumps(results[name], indent=1))
    (OUT / "policy.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
