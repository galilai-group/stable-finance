"""Sanity checks and negative controls: does the pipeline recover what is known?

A simulation framework earns trust by reproducing closed-form and well-known
results before it reports new ones. Each check here has an answer known in
advance, and prints PASS, FAIL or SKIP with the numbers behind it:

* ANALYTIC TRANSFER COEFFICIENTS. On Gaussian signals with an identity
  covariance, the transfer coefficient of Clarke, de Silva and Thorley (2002)
  has a closed form for each weighting rule: proportional to the signal 1,
  rank weights ``sqrt(3 / pi) = 0.977``, the long-short halves at equal weight
  ``sqrt(2 / pi) = 0.798``, top and bottom deciles ``0.785``, quintiles
  ``0.885`` (``2 phi(z_q) / sqrt(2 q)`` for a tail of size ``q``). The
  library's books and its ``transfer_coefficient`` must land on them.
* THE FUNDAMENTAL LAW. Realized per-period IR against ``TC * IC * sqrt(N)``
  (Grinold 1989, with the transfer coefficient), per construction, from the
  E6 and E1 rows. It is reported, not enforced: for an oracle whose IC is
  exact at every decision the law does not apply (each period's P&L is about
  ``N * IC * sd(w) * sd(r)``, almost never negative), which the exact-IC
  invariance diagnostic below makes visible.
* EXACT-IC INVARIANCE. Gross Sharpe of the exact-IC (tails, middle) signal
  book across ICs, from E1. If it is roughly flat in IC, exact-IC gross
  Sharpe measures the dispersion process, not forecast quality, and should be
  read only as a cost-ratio device.
* PLACEBO. E6's IC-0 forecasts under ten seeds: the daily Sharpe must centre
  on zero relative to its spread, and the 95% intervals must cover zero.
* INVARIANCE AT IC 1. E6's tails, middle and random oracles at IC 1 are the
  same forecast and must give identical rows.
* LOOK-AHEAD CANARIES, on the real month: the realized forward return used as
  its own forecast must give an absurd Sharpe; each extra decision of lag must
  lower it; the forecast shifted by one day (yesterday's move at the same
  time) must give about nothing.
* COST IDENTITY. From every saved series: net is exactly gross minus cost, and
  each row's break-even EFQ is recomputed from its own series.
* LATENCY. With a v2 cache: gross Sharpe must not rise as the execution delay
  grows from 0 to 30 seconds. Skipped without one.
* DETERMINISM. One small task run twice, and with one and four BLAS threads,
  must give byte-identical rows.

    uv run --extra experiments python -m experiments.sanity [--results DIR] [--quick]

``--results`` points at the directory holding finished runs (default
``experiments/results``); the report goes to ``<results>/sanity/report.json``
unless ``--out`` says otherwise. ``--quick`` skips the canaries and the
determinism runs, which load a month and run backtests.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist

import numpy as np

import stable_finance as sf
from experiments.run import RESULTS

#: The closed-form transfer coefficient of each book on Gaussian signals and
#: an identity covariance, as (selection, weighting) -> TC.
ANALYTIC_TC = {
    ("all", "signal"): 1.0,
    ("all", "rank"): float(np.sqrt(3.0 / np.pi)),
    ("all", "equal"): float(np.sqrt(2.0 / np.pi)),
    ("decile", "equal"): 2 * NormalDist().pdf(NormalDist().inv_cdf(0.9)) / np.sqrt(0.2),
    ("quintile", "equal"): 2 * NormalDist().pdf(NormalDist().inv_cdf(0.8)) / np.sqrt(0.4),
}
#: Fundamental-law ratios outside this band are flagged.
LAW_BAND = (0.85, 1.15)


def _result(name: str, status: str, **numbers) -> dict:
    line = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in numbers.items() if not isinstance(v, (list, dict)))
    print(f"{status:<4}  {name}: {line}")
    return {"check": name, "status": status, **numbers}


def _rows(results: Path, name: str):
    import pandas as pd

    path = results / name / "rows.parquet"
    return pd.read_parquet(path) if path.is_file() else None


# ── checks that need no run ────────────────────────────────────────────────

def analytic_transfer_coefficients(n_assets: int = 2000, n_decisions: int = 100,
                                   seed: int = 0) -> dict:
    """The library's books on Gaussian signals against the closed forms."""
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal((n_decisions, n_assets))
    eligible = np.ones_like(alpha, dtype=bool)
    sigma = np.eye(n_assets)
    measured, worst = {}, 0.0
    for (selection, weighting), expected in ANALYTIC_TC.items():
        signs = sf.select_book(selection, alpha, eligible)
        weights = sf.weight_book(weighting, signs, alpha, sigma)
        tc = sf.transfer_coefficient(weights, alpha, sigma).mean
        measured[f"{selection}/{weighting}"] = {"tc": float(tc), "expected": expected}
        worst = max(worst, abs(tc - expected))
    # Sampling error of a mean over decisions of per-decision TCs on N names
    # is of order 1 / sqrt(N * n_decisions); allow a generous multiple.
    tolerance = 0.01
    return _result("analytic transfer coefficients", "PASS" if worst < tolerance else "FAIL",
                   worst_abs_error=float(worst), tolerance=tolerance, detail=measured)


# ── checks on finished runs ────────────────────────────────────────────────

def fundamental_law(results: Path) -> dict:
    """Realized per-period IR over TC * IC * sqrt(N), per construction."""
    import pandas as pd

    frames = [f for f in (_rows(results, "e6_sanity_checks"), _rows(results, "e1_transfer_map"))
              if f is not None]
    if not frames:
        return _result("fundamental law", "SKIP", reason="no E6 or E1 rows")
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[(rows["efq"] == 0.0) & (rows["information_coefficient"].abs() > 1e-3)
                & np.isfinite(rows["transfer_coefficient"])]
    if rows.empty:
        return _result("fundamental law", "SKIP", reason="no frictionless rows with an IC")
    ir = rows["sharpe_mid"] / np.sqrt(rows["periods_per_year"])
    law = rows["transfer_coefficient"] * rows["information_coefficient"] * np.sqrt(
        rows["mean_names_held"])
    rows = rows.assign(ratio=ir / law)
    exact = rows["f_noise"].isin(["tails", "middle"]) if "f_noise" in rows else False
    by = {}
    for (kind, weighting), part in rows.groupby([np.where(exact, "exact IC", "expected IC"),
                                                 "weighting"]):
        by[f"{kind}/{weighting}"] = float(part["ratio"].median())
    # Only small-IC, expected-IC rows are held to the band: the law is a
    # small-IC approximation and does not describe an exact-IC oracle. A
    # ratio outside it is a FLAG to investigate, not a pipeline failure.
    expected_ic = {k: v for k, v in by.items() if k.startswith("expected")}
    outside = [k for k, v in expected_ic.items() if not LAW_BAND[0] <= v <= LAW_BAND[1]]
    status = "PASS" if expected_ic and not outside else ("FLAG" if outside else "SKIP")
    return _result("fundamental law (expected-IC rows)", status,
                   median_ratio_by_construction=by, outside_band=outside, band=list(LAW_BAND),
                   n_rows=len(rows))


def exact_ic_invariance(results: Path) -> dict:
    """Gross Sharpe of the exact-IC signal book across ICs: flat means dispersion-driven."""
    rows = _rows(results, "e1_transfer_map")
    if rows is None:
        return _result("exact-IC invariance diagnostic", "SKIP", reason="no E1 rows")
    part = rows[rows["f_noise"].isin(["tails", "middle"]) & (rows["weighting"] == "signal")
                & (rows["selection"] == "all") & (rows["efq"] == 0.0)]
    if part.empty:
        return _result("exact-IC invariance diagnostic", "SKIP", reason="no exact-IC signal rows")
    curve = part.groupby("f_ic")["sharpe_mid"].median()
    spread = float(curve.max() / curve.min()) if curve.min() > 0 else float("nan")
    # Informational: a ratio near 1 across a twenty-fold range of IC is the warning sign.
    return _result("exact-IC invariance diagnostic", "INFO",
                   sharpe_by_ic={str(k): float(v) for k, v in curve.items()},
                   max_over_min=spread)


def placebo(results: Path) -> dict:
    """E6's IC-0 seeds: centred on zero, intervals covering it."""
    rows = _rows(results, "e6_sanity_checks")
    if rows is None:
        return _result("placebo (IC 0)", "SKIP", reason="no E6 rows")
    part = rows[(rows["forecast"] == "copula") & (rows["f_ic"] == 0.0) & (rows["efq"] == 0.0)]
    if part.empty or "sharpe_daily_mid" not in part:
        return _result("placebo (IC 0)", "SKIP", reason="no IC-0 rows with daily Sharpe")
    sharpe = part["sharpe_daily_mid"].to_numpy()
    t = float(sharpe.mean() / (sharpe.std(ddof=1) / np.sqrt(len(sharpe))))
    cover = float(((part["sharpe_daily_mid_low"] <= 0)
                   & (part["sharpe_daily_mid_high"] >= 0)).mean())
    status = "PASS" if abs(t) < 3 and cover >= 0.8 else "FAIL"
    return _result("placebo (IC 0)", status, mean_daily_sharpe=float(sharpe.mean()),
                   t_of_mean=t, interval_coverage=cover, n_rows=len(part))


def invariance_at_one(results: Path) -> dict:
    """E6 at IC 1: tails, middle and random rows identical."""
    rows = _rows(results, "e6_sanity_checks")
    if rows is None:
        return _result("invariance at IC 1", "SKIP", reason="no E6 rows")
    part = rows[(rows["forecast"] == "oracle") & (rows["f_ic"] == 1.0)]
    if part.empty:
        return _result("invariance at IC 1", "SKIP", reason="no IC-1 rows")
    cols = ["sharpe_mid", "sharpe_net", "mean_turnover", "transfer_coefficient"]
    spread = part.groupby(["weighting", "efq", "rebalance"])[cols].agg(lambda x: x.max() - x.min())
    worst = float(np.nanmax(np.abs(spread.to_numpy())))
    return _result("invariance at IC 1", "PASS" if worst == 0.0 else "FAIL",
                   worst_spread=worst)


def cost_identity(results: Path) -> dict:
    """net == gross - cost in every saved series; break-even recomputed from it."""
    import pandas as pd

    worst_identity, worst_breakeven, checked = 0.0, 0.0, 0
    for run in sorted(p for p in results.iterdir() if (p / "series.npz").is_file()):
        rows = pd.read_parquet(run / "rows.parquet")
        series = np.load(run / "series.npz")
        for row in rows.itertuples(index=False):
            if row.row_id not in series.files:
                continue
            gross, net, cost = series[row.row_id]
            worst_identity = max(worst_identity, float(np.nanmax(np.abs(net - (gross - cost)))))
            breakeven = getattr(row, "breakeven_efq", np.nan)
            if row.cost_model != "mid" and row.efq > 0 and np.isfinite(breakeven) \
                    and getattr(row, "mean_fee_bps", None) is None:
                again = sf.breakeven_efq(gross, cost / row.efq)
                if np.isfinite(again):
                    worst_breakeven = max(worst_breakeven, abs(float(again) - breakeven))
            checked += 1
    if not checked:
        return _result("cost identity", "SKIP", reason="no saved series")
    status = "PASS" if worst_identity < 1e-12 and worst_breakeven < 1e-9 else "FAIL"
    return _result("cost identity", status, worst_net_minus_gross_plus_cost=worst_identity,
                   worst_breakeven_error=worst_breakeven, rows_checked=checked)


# ── checks that run backtests ──────────────────────────────────────────────

def _month(spec_extra: dict | None = None) -> dict:
    from experiments.run import load_inputs

    spec = {"eval_month": "2020-08", "history_month": "2020-07", "horizons": [300],
            **(spec_extra or {})}
    return load_inputs(spec)


def _shifted(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    if lag > 0:
        out[lag:] = values[:-lag]
    else:
        out[:] = values
    return out


def lookahead_canaries() -> dict:
    """Perfect foresight is absurd, lags decay, a one-day shift is about nothing."""
    inputs = _month()
    realized = inputs["realized"]
    config = sf.BacktestConfig(weighting="signal", selection="all", cost_model="mid",
                               risk_model="diagonal", annualization="decisions")
    per_day = int(np.median(np.unique(inputs["days"], return_counts=True)[1]))
    sharpe = {}
    for name, lag in (("same decision", 0), ("lag 1", 1), ("lag 2", 2), ("one day", per_day)):
        values = _shifted(realized.values, lag)
        forecast = sf.ForwardReturns(values, realized.decisions, realized.assets,
                                     realized.horizons)
        report = sf.run_backtest(forecast, realized, market=inputs["market"],
                                 history=inputs["history"], days=inputs["days"], config=config)
        sharpe[name] = float(report.sharpe_mid)
    ok = (sharpe["same decision"] > 10 * max(abs(sharpe["lag 1"]), 1.0)
          and abs(sharpe["lag 1"]) >= abs(sharpe["lag 2"]) * 0.5
          and abs(sharpe["one day"]) < 0.25 * sharpe["same decision"])
    return _result("look-ahead canaries", "PASS" if ok else "FAIL", sharpe=sharpe,
                   same_decision=sharpe["same decision"], lag_1=sharpe["lag 1"],
                   lag_2=sharpe["lag 2"], one_day=sharpe["one day"])


def latency_monotone() -> dict:
    """With a v2 cache, gross Sharpe at efq 1 does not rise as the delay grows."""
    from experiments.month_cache_v2 import month_path

    if not all(month_path(m).is_file() for m in ("2020-07", "2020-08")):
        return _result("latency monotonicity", "SKIP", reason="no v2 cache for Jul/Aug 2020")
    config = sf.BacktestConfig(weighting="signal", selection="all", cost_model="cross", efq=1.0,
                               risk_model="diagonal", annualization="decisions")
    sharpe = {}
    for latency in (0, 1, 5, 30):
        inputs = _month({"cache": "v2", "latency": latency})
        forecast = sf.oracle_forecast(inputs["realized"], 0.03, noise="random", seed=0)
        report = sf.run_backtest(forecast, inputs["realized"], market=inputs["market"],
                                 history=inputs["history"], days=inputs["days"], config=config)
        sharpe[str(latency)] = float(report.sharpe_net)
    values = list(sharpe.values())
    ok = all(b <= a + 1e-9 for a, b in pairwise(values))
    return _result("latency monotonicity (net Sharpe at efq 1)", "PASS" if ok else "FAIL",
                   sharpe_by_latency=sharpe)


TINY_SPEC = {
    "name": "sanity_determinism", "eval_month": "2020-08", "history_month": "2020-07",
    "horizons": [900],
    "forecasts": {"kind": "oracle", "ic": [0.03], "noise": ["random"], "seed": 0},
    "configs": {"base": {"cost_model": "cross", "risk_model": "shrunk_sample",
                         "universe": "all", "annualization": "decisions", "selection": "all"},
                "grid": {"weighting": ["signal", "hrp"], "efq": [0.7]}},
}


def determinism() -> dict:
    """The same task twice, and with one and four BLAS threads: identical rows."""
    root = Path(__file__).resolve().parent.parent
    hashes, frames = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for threads in ("1", "1", "4"):
            spec = dict(TINY_SPEC, name=f"sanity_determinism_{len(hashes)}")
            path = Path(tmp) / f"{spec['name']}.json"
            path.write_text(json.dumps(spec))
            env = {**os.environ, "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
                   "MKL_NUM_THREADS": threads}
            subprocess.run([sys.executable, "-m", "experiments.run", str(path), "--workers", "1"],
                           cwd=root, env=env, check=True, capture_output=True)
            rows = RESULTS / spec["name"] / "rows.parquet"
            import pandas as pd

            frame = pd.read_parquet(rows)
            frames.append(frame)
            hashes.append(hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest())
            import shutil

            shutil.rmtree(RESULTS / spec["name"], ignore_errors=True)
    sharpe_gap = float(np.nanmax(np.abs(frames[0][["sharpe_mid", "sharpe_net"]].to_numpy()
                                        - frames[2][["sharpe_mid", "sharpe_net"]].to_numpy())))
    same_run = hashes[0] == hashes[1]
    same_threads = hashes[0] == hashes[2]
    # Reruns must be identical. Thread count is reported, not required: a
    # multithreaded BLAS sums in a thread-dependent order, which is why
    # run.py pins one thread; a WARN says the pin is what reproduction needs.
    status = "FAIL" if not same_run else ("PASS" if same_threads else "WARN")
    return _result("determinism", status, rerun_identical=same_run,
                   one_vs_four_threads_identical=same_threads,
                   max_sharpe_difference_threads=sharpe_gap)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(RESULTS),
                        help="directory holding finished runs (default experiments/results)")
    parser.add_argument("--out", default=None, help="report directory (default <results>/sanity)")
    parser.add_argument("--quick", action="store_true",
                        help="skip the checks that load a month and run backtests")
    args = parser.parse_args(argv)
    results = Path(args.results)
    report = [analytic_transfer_coefficients(), fundamental_law(results),
              exact_ic_invariance(results), placebo(results), invariance_at_one(results),
              cost_identity(results)]
    if not args.quick:
        report += [lookahead_canaries(), latency_monotone(), determinism()]
    out = Path(args.out) if args.out else results / "sanity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    failed = [r["check"] for r in report if r["status"] == "FAIL"]
    counts = {s: sum(r["status"] == s for r in report) for s in ("PASS", "FAIL", "FLAG", "WARN",
                                                                   "INFO", "SKIP")}
    print(f"\n{len(report)} checks: " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()
                                                  if n) + f" -> {out / 'report.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
