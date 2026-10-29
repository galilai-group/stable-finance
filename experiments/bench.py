"""What each stage costs on one real month: seconds and peak memory.

Times every risk model, every construction's book, the scoring of a held book
and one whole harness run on August 2020 at the 15-minute horizon, one at a
time, each under ``tracemalloc`` for its peak allocation. Every stage is
called through the library's public functions -- ``estimate_risk``,
``select_book`` and ``weight_book``, ``mean_variance_target`` and
``rebalance``, ``run_backtest`` -- so the numbers are what a caller of those
functions pays. Run it on an otherwise idle machine: the numbers are
wall-clock.

    uv run --extra experiments python -m experiments.bench
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

import stable_finance as sf
from experiments.run import load_inputs

OUT = Path(__file__).resolve().parent / "results" / "bench"
SPEC = {"eval_month": "2020-08", "history_month": "2020-07", "horizons": [900]}
GROSS, EFQ = 2.0, 0.7


def measure(work):
    """Wall-clock seconds of an untraced run, then the peak of a traced one.

    tracemalloc slows Python-level loops several-fold, so timing under it
    would overstate exactly the rules (HRP, the active set) that loop in
    Python; the two are measured on separate runs of the same work.
    """
    t0 = time.perf_counter()
    result = work()
    seconds = time.perf_counter() - t0
    tracemalloc.start()
    work()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak / 1e6


def mean_variance_book(forecast, sigma, half_spread):
    """The harness's mean-variance book: a causal risk aversion, then the
    cost-aware rebalance toward the target at EFQ times the half-spread."""
    mu = np.nan_to_num(forecast.values[:, :, 0], nan=0.0)
    gross = np.abs(np.linalg.solve(sigma, mu.T).T).sum(axis=1)
    aversion = np.maximum(np.cumsum(gross) / np.arange(1, len(gross) + 1) / GROSS, 1e-12)
    target = sf.mean_variance_target(forecast, sigma[None], risk_aversion=aversion)
    cost = np.where(np.isfinite(half_spread), EFQ * half_spread, np.inf)
    return sf.rebalance(target, sigma[None], risk_aversion=aversion, cost=cost).values[:, :, 0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(SPEC)
    realized, history, market, days = (inputs[k] for k in ("realized", "history", "market",
                                                          "days"))
    forecast, oracle_s, oracle_mb = measure(
        lambda: sf.oracle_forecast(realized, 0.03, noise="random", seed=0))
    seconds, peak = {"oracle forecast": oracle_s}, {"oracle forecast": oracle_mb}
    for risk in sf.RISK_MODELS:
        if risk == "embedding":
            continue
        _, s, mb = measure(lambda risk=risk: sf.estimate_risk(risk, history))
        seconds[f"risk: {risk}"], peak[f"risk: {risk}"] = s, mb
    sigma = sf.estimate_risk("shrunk_sample", history)[0]
    mu = forecast.values[:, :, 0]
    half_spread = market.get("half_spread")
    eligible = np.isfinite(mu) & np.isfinite(half_spread)
    signs = sf.select_book("all", mu, eligible)
    for weighting in ("equal", "rank", "signal", "hrp", "min_var_qp"):
        _, s, mb = measure(lambda weighting=weighting: sf.weight_book(
            weighting, signs, mu, sigma, gross_exposure=GROSS))
        seconds[weighting], peak[weighting] = s, mb
    _, s, mb = measure(lambda: mean_variance_book(forecast, sigma, half_spread))
    seconds["mean_variance"], peak["mean_variance"] = s, mb

    config = sf.BacktestConfig(weighting="signal", selection="all", risk_model="shrunk_sample",
                               cost_model="cross", efq=EFQ, annualization="decisions")
    held = sf.PortfolioWeights(
        sf.weight_book("signal", signs, mu, sigma, gross_exposure=GROSS)[:, :, None],
        realized.decisions, realized.assets, realized.horizons)
    _, s, mb = measure(lambda: sf.run_backtest(held, realized, market=market, days=days,
                                               config=config))
    seconds["scoring a held book (fills, P&L)"] = s
    peak["scoring a held book (fills, P&L)"] = mb
    _, s, mb = measure(lambda: sf.run_backtest(forecast, realized, market=market,
                                               history=history, days=days, config=config))
    seconds["whole harness run (signal)"], peak["whole harness run (signal)"] = s, mb
    report = {"decisions": len(realized.decisions), "assets": len(realized.assets),
              "seconds": seconds, "peak_mb": peak}
    (OUT / "bench.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
