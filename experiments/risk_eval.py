"""Score each risk model out of sample, before any alpha is built on it.

Every construction in the study -- minimum variance, long-only QP, ERC, HRP,
cost-aware mean-variance -- reads a covariance, so an artefact of the risk
model shows up as a property of the construction unless the risk model is
scored on its own. This script does that with the losses the literature uses
for it, and nothing else: no forecast quality, no costs.

Each risk model is estimated on the history month's five-minute returns (July
2020) exactly as the harness estimates it, then held fixed through the
evaluation month (August 2020), as the experiments hold it, and scored decision
by decision:

* GLOBAL MINIMUM VARIANCE. The fully invested book ``w = S^-1 1 / 1'S^-1 1``
  needs no expected returns, so its realized variance measures the covariance
  alone: the better the forecast of co-movement, the lower the variance the
  book actually delivers (Engle and Colacito, 2006). Reported per day as the
  sum of the day's squared five-minute book returns.
* A LONG-SHORT SIGNAL BOOK. The dollar-neutral book of least predicted
  variance with one unit of forecast alpha, ``min w'Sw`` subject to
  ``mu'w = 1`` and ``1'w = 0``, re-solved every decision with ``mu`` the
  normal-scored five-minute reversal (minus the past five-minute mid return,
  a causal feature of the month cache). Every model's book carries the same
  forecast alpha, so the realized variances compare like with like -- the
  Engle and Colacito construction for a book that has a view.
* AN ENGLE-COLACITO TEST. A Diebold and Mariano (1995) test on the
  difference between two models' daily realized variances, with a Newey and
  West (1987) long-run variance and the Harvey, Leybourne and Newbold (1997)
  small-sample correction against Student's t; a month has about twenty days,
  so the same test is also reported on the five-minute squared returns with a
  one-day Bartlett window, which has more observations and the same null.
* BIAS STATISTICS. Realized over predicted volatility -- the root mean square
  of each book return divided by the volatility the model predicted for it --
  in half-hour buckets of the session, overall, and for single names (the
  median across names). A calibrated model gives about 1 everywhere, inside
  ``1 +- 1.96 / sqrt(2 n)`` for ``n`` returns; a model estimated over whole
  days cannot know that the open is more volatile than the lunch hour, and
  this is where that shows (Andersen and Bollerslev, 1997).

Missing returns in the evaluation month (names not quoting) earn zero, and
the books are formed on the whole cross-section the harness uses. Nothing is
random, so a rerun gives identical numbers. The month cache carries no
industry codes, so the ``factor`` model here is statistical factors alone.

    uv run --extra experiments python -m experiments.risk_eval
"""

from __future__ import annotations

import os

# One linear-algebra thread, as in run.py: a multithreaded BLAS sums in an
# order that depends on the thread count, which moves the last digits.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

import stable_finance as sf
from experiments.run import _git_sha, load_inputs
from experiments.signals import normal_scores

OUT = Path(__file__).resolve().parent / "results" / "risk_eval"

#: The models compared, as ``estimate_risk`` arguments: the four the default
#: space sweeps on returns alone, at the harness's own settings, and the
#: factor model at its default five statistical factors.
MODELS: dict[str, dict] = {
    "diagonal": {"name": "diagonal"},
    "shrunk_sample": {"name": "shrunk_sample", "shrinkage": 0.3},
    "ledoit_wolf": {"name": "ledoit_wolf"},
    "nonlinear_shrinkage": {"name": "nonlinear_shrinkage"},
    "factor": {"name": "factor", "n_factors": 5},
}

#: The same scores for the unshrunk sample covariance and for the factor model
#: at other factor counts: how much the choice of five matters.
SENSITIVITY: dict[str, dict] = {
    "sample": {"name": "shrunk_sample", "shrinkage": 0.0},
    **{f"factor_{k}": {"name": "factor", "n_factors": k} for k in (1, 3, 10, 20)},
}

#: The model every other one is tested against: the harness default.
BENCHMARK = "shrunk_sample"
BUCKET_SECONDS = 1800
OPEN_SECONDS = 9 * 3600 + 30 * 60


# ── books ───────────────────────────────────────────────────────────────────

def minimum_variance_book(sigma: np.ndarray) -> np.ndarray:
    """The fully invested global minimum-variance weights."""
    ones = np.linalg.solve(sigma, np.ones(len(sigma)))
    return ones / ones.sum()


def signal_books(sigma: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per decision, the dollar-neutral least-variance book with ``mu'w = 1``.

    With ``A = [mu, 1]`` the solution is ``S^-1 A (A'S^-1 A)^-1 (1, 0)'`` and
    its predicted variance is the ``(1, 1)`` entry of ``(A'S^-1 A)^-1``. A
    decision without a usable signal gets no book.
    """
    n_assets = mu.shape[1]
    inverse_mu = np.linalg.solve(sigma, mu.T).T            # (decision, asset)
    inverse_one = np.linalg.solve(sigma, np.ones(n_assets))
    a = np.einsum("ti,ti->t", mu, inverse_mu)               # mu' S^-1 mu
    b = mu @ inverse_one                                    # mu' S^-1 1
    c = float(inverse_one.sum())                            # 1' S^-1 1
    det = a * c - b * b
    usable = det > 1e-12 * np.maximum(a * c, 1e-300)
    safe = np.where(usable, det, 1.0)
    weights = (c * inverse_mu - b[:, None] * inverse_one[None, :]) / safe[:, None]
    weights[~usable] = 0.0
    predicted = np.where(usable, c / safe, np.nan)
    return weights, predicted


def book_returns(weights: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Each decision's book return; a name without a return earns zero."""
    r = np.nan_to_num(realized, nan=0.0)
    if weights.ndim == 1:
        return r @ weights
    return np.einsum("ti,ti->t", weights, r)


# ── statistics ──────────────────────────────────────────────────────────────

def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Bartlett-weighted long-run variance of a series about its mean."""
    d = x - x.mean()
    n = len(d)
    out = float(d @ d) / n
    for lag in range(1, min(lags, n - 1) + 1):
        out += 2 * (1 - lag / (lags + 1)) * float(d[lag:] @ d[:-lag]) / n
    return out


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, *, lags: int | None = None,
                    small_sample: bool = True) -> dict:
    """DM test of equal expected loss; negative ``stat`` favours ``a``.

    ``lags=None`` takes the Newey-West rule ``floor(4 (n/100)^(2/9))``. With
    ``small_sample`` the statistic gets Harvey, Leybourne and Newbold's
    one-step correction ``sqrt((n - 1) / n)`` and is read against Student's t
    with ``n - 1`` degrees of freedom; otherwise against the normal.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    variance = newey_west_variance(d, lags)
    if variance <= 0:
        return {"mean_difference": float(d.mean()), "stat": float("nan"),
                "p_value": float("nan"), "lags": lags, "n": n}
    stat = float(d.mean() / np.sqrt(variance / n))
    if small_sample:
        stat *= np.sqrt((n - 1) / n)
        p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    else:
        p = float(2 * stats.norm.sf(abs(stat)))
    return {"mean_difference": float(d.mean()), "relative_difference": float(
        d.mean() / np.mean(loss_b)), "stat": stat, "p_value": p, "lags": lags, "n": n}


def bias(standardized: np.ndarray, groups: np.ndarray, n_groups: int) -> list[float]:
    """Root mean square of standardized returns within each group."""
    out = []
    for g in range(n_groups):
        z = standardized[(groups == g) & np.isfinite(standardized)]
        out.append(float(np.sqrt(np.mean(z * z))) if len(z) else float("nan"))
    return out


def column_bias(z: np.ndarray) -> np.ndarray:
    """Each column's root mean square over its finite rows."""
    finite = np.isfinite(z)
    return np.sqrt(np.where(finite, z * z, 0.0).sum(axis=0) / np.maximum(finite.sum(axis=0), 1))


def daily_sum(values: np.ndarray, day_index: np.ndarray, n_days: int) -> np.ndarray:
    return np.bincount(day_index, weights=values, minlength=n_days)


# ── the experiment ──────────────────────────────────────────────────────────

def score_model(sigma: np.ndarray, realized: np.ndarray, mu: np.ndarray, day: np.ndarray,
                bucket: np.ndarray, n_days: int, n_buckets: int) -> tuple[dict, dict]:
    """Every score of one covariance, and the per-observation losses for the tests."""
    gmv = minimum_variance_book(sigma)
    gmv_returns = book_returns(gmv, realized)
    gmv_predicted = float(gmv @ sigma @ gmv)
    weights, signal_predicted = signal_books(sigma, mu)
    signal_returns = book_returns(weights, realized)
    has_book = np.isfinite(signal_predicted)
    signal_returns = np.where(has_book, signal_returns, np.nan)

    def summary(returns, predicted, books):
        squared = np.where(np.isfinite(returns), returns * returns, 0.0)
        daily = daily_sum(squared, day, n_days)
        daily_predicted = daily_sum(np.where(np.isfinite(predicted), predicted, 0.0),
                                    day, n_days)
        z = returns / np.sqrt(predicted)
        return {
            "daily_realized_variance": daily.tolist(),
            "realized_daily_vol": float(np.sqrt(daily.mean())),
            "predicted_daily_vol": float(np.sqrt(daily_predicted.mean())),
            "bias": bias(z, np.zeros(len(z), dtype=int), 1)[0],
            "bias_by_bucket": bias(z, bucket, n_buckets),
            "gross": float(np.mean(np.abs(books).sum(axis=-1))),
        }

    out = {
        "gmv": summary(gmv_returns, np.full(len(gmv_returns), gmv_predicted), gmv)
        | {"long": float(gmv[gmv > 0].sum()), "short": float(-gmv[gmv < 0].sum())},
        "signal": summary(signal_returns, signal_predicted, weights[has_book])
        | {"decisions_with_a_book": int(has_book.sum())},
    }
    # Single names: each one's own return over its own predicted volatility.
    sd = np.sqrt(np.diag(sigma))
    z_names = realized / sd[None, :]
    by_name = np.array([bias(z_names[:, i], bucket, n_buckets) for i in range(len(sd))])
    out["names"] = {"median_bias": float(np.nanmedian(column_bias(z_names))),
                    "median_bias_by_bucket": np.nanmedian(by_name, axis=0).tolist()}
    losses = {
        "gmv_daily": np.asarray(out["gmv"]["daily_realized_variance"]),
        "gmv_decision": np.where(np.isfinite(gmv_returns), gmv_returns**2, 0.0),
        "signal_daily": np.asarray(out["signal"]["daily_realized_variance"]),
        "signal_decision": np.where(np.isfinite(signal_returns), signal_returns**2, 0.0),
    }
    return out, losses


def tests_between(losses: dict, names: list[str], per_day: int) -> dict:
    """Pairwise DM tests on each book's daily and five-minute losses."""
    out = {}
    for book in ("gmv", "signal"):
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rows.append({
                    "a": a, "b": b,
                    "daily": diebold_mariano(losses[a][f"{book}_daily"],
                                             losses[b][f"{book}_daily"]),
                    "decision": diebold_mariano(losses[a][f"{book}_decision"],
                                                losses[b][f"{book}_decision"],
                                                lags=per_day, small_sample=False),
                })
        out[book] = rows
    return out


def against(losses: dict, names: list[str], benchmark: str, per_day: int) -> dict:
    """Each model's DM test against one benchmark, on both books."""
    out = {}
    for name in names:
        if name == benchmark:
            continue
        out[name] = {
            book: {
                "daily": diebold_mariano(losses[name][f"{book}_daily"],
                                         losses[benchmark][f"{book}_daily"]),
                "decision": diebold_mariano(losses[name][f"{book}_decision"],
                                            losses[benchmark][f"{book}_decision"],
                                            lags=per_day, small_sample=False),
            } for book in ("gmv", "signal")
        }
    return out


def table(results: dict) -> str:
    """The headline numbers as a Markdown table."""
    models = results["models"] | results["sensitivity"]["models"]
    tests = results["against_benchmark"] | results["sensitivity"]["against_benchmark"]
    labels = results["setup"]["bucket_labels"]
    last = len(labels) - 1
    setup = results["setup"]
    lines = [
        (f"Risk models estimated on {setup['history_month']}, scored on "
         f"{setup['eval_month']} ({setup['n_days']} days, {setup['n_assets']} names, "
         f"{setup['horizon']} s returns). Vol in bp per day; DM t against `{BENCHMARK}` "
         "(negative = lower realized variance), p from the HLN-corrected daily test; "
         "bias = realized / predicted vol."),
        "",
        ("| model | GMV realized | GMV predicted | GMV bias | GMV vs bench (t, p) "
         "| signal realized / diag | signal bias | signal vs bench (t, p) "
         f"| GMV bias {labels[0]} | GMV bias {labels[6]} | GMV bias {labels[last]} |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    diag_signal = models["diagonal"]["signal"]["realized_daily_vol"]
    for name, m in models.items():
        g, s = m["gmv"], m["signal"]
        test = tests.get(name)
        gt = (f"{test['gmv']['daily']['stat']:+.2f}, {test['gmv']['daily']['p_value']:.3f}"
              if test else "bench")
        st = (f"{test['signal']['daily']['stat']:+.2f}, {test['signal']['daily']['p_value']:.3f}"
              if test else "bench")
        lines.append(
            f"| {name} | {1e4 * g['realized_daily_vol']:.2f} | "
            f"{1e4 * g['predicted_daily_vol']:.2f} | {g['bias']:.2f} | {gt} | "
            f"{s['realized_daily_vol'] / diag_signal:.3f} | {s['bias']:.2f} | {st} | "
            f"{g['bias_by_bucket'][0]:.2f} | {g['bias_by_bucket'][6]:.2f} | "
            f"{g['bias_by_bucket'][last]:.2f} |")
    return "\n".join(lines) + "\n"


def run(history_month: str, eval_month: str, horizon: int) -> dict:
    started = time.perf_counter()
    inputs = load_inputs({"eval_month": eval_month, "history_month": history_month,
                          "horizons": [horizon]})
    history, realized_fr = inputs["history"], inputs["realized"]
    realized = realized_fr.values[:, :, 0]
    names = list(inputs["feature_names"])
    reversal = -inputs["features"][:, :, names.index("rev_5m")]
    mu = np.nan_to_num(normal_scores(reversal[:, :, None])[:, :, 0], nan=0.0)

    days = np.asarray(inputs["days"]).astype(str)
    day_labels, day = np.unique(days, return_inverse=True)
    seconds = np.array([int(str(d).split("|")[1]) for d in realized_fr.decisions])
    bucket = seconds // BUCKET_SECONDS
    n_buckets = int(bucket.max()) + 1
    bucket_labels = [
        f"{(OPEN_SECONDS + b * BUCKET_SECONDS) // 3600:02d}:"
        f"{(OPEN_SECONDS + b * BUCKET_SECONDS) % 3600 // 60:02d}" for b in range(n_buckets)]
    per_day = int(np.bincount(day).max())

    def estimate(spec: dict) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        args = dict(spec)
        sigma = sf.estimate_risk(args.pop("name"), history, **args)[0]
        return sigma, time.perf_counter() - t0

    scored, losses, seconds_taken = {}, {}, {}
    for label, spec in (MODELS | SENSITIVITY).items():
        sigma, took = estimate(spec)
        eigenvalues = np.linalg.eigvalsh(sigma)
        scored[label], losses[label] = score_model(
            sigma, realized, mu, day, bucket, len(day_labels), n_buckets)
        scored[label]["estimate_seconds"] = took
        scored[label]["condition_number"] = float(eigenvalues.max() / eigenvalues.min())
        seconds_taken[label] = took
        print(f"{label:>20}: GMV realized {1e4 * scored[label]['gmv']['realized_daily_vol']:.2f}"
              f" bp/day, predicted {1e4 * scored[label]['gmv']['predicted_daily_vol']:.2f},"
              f" bias {scored[label]['gmv']['bias']:.2f} ({took:.1f} s)", flush=True)

    main_names, extra_names = list(MODELS), list(SENSITIVITY)
    n_per_bucket = np.bincount(bucket, minlength=n_buckets)
    return {
        "setup": {
            "history_month": history_month, "eval_month": eval_month, "horizon": horizon,
            "n_assets": int(realized.shape[1]), "n_history_decisions": len(history.decisions),
            "n_eval_decisions": int(realized.shape[0]), "n_days": len(day_labels),
            "days": day_labels.tolist(), "bucket_labels": bucket_labels,
            "bucket_returns": n_per_bucket.tolist(),
            "bias_band_95": [[1 - 1.96 / np.sqrt(2 * n), 1 + 1.96 / np.sqrt(2 * n)]
                             for n in n_per_bucket.tolist()],
            "signal": "normal-scored -rev_5m (minus the past 5-minute mid return), missing = 0",
            "models": MODELS, "sensitivity_models": SENSITIVITY, "benchmark": BENCHMARK,
            "missing_return_fraction": float(1 - np.isfinite(realized).mean()),
            "git": _git_sha(), "seconds": time.perf_counter() - started,
        },
        "models": {k: scored[k] for k in main_names},
        "pairwise": tests_between(losses, main_names, per_day),
        "against_benchmark": against(losses, main_names, BENCHMARK, per_day),
        "ranking": {
            book: sorted(main_names, key=lambda k, book=book: scored[k][book]["realized_daily_vol"])
            for book in ("gmv", "signal")
        },
        "sensitivity": {
            "models": {k: scored[k] for k in extra_names},
            "against_benchmark": against(losses, extra_names, BENCHMARK, per_day),
            "against_factor": against(losses, extra_names, "factor", per_day),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history", default="2020-07", help="risk history month, YYYY-MM")
    parser.add_argument("--month", default="2020-08", help="evaluation month, YYYY-MM")
    parser.add_argument("--horizon", type=int, default=300, help="return horizon in seconds")
    args = parser.parse_args(argv)
    results = run(args.history, args.month, args.horizon)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "risk_eval.json").write_text(json.dumps(results, indent=2))
    text = table(results)
    (OUT / "table.md").write_text(text)
    print()
    print(text)
    print("ranking (lowest realized variance first):", json.dumps(results["ranking"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
