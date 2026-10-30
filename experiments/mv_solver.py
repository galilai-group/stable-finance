"""Is the mean-variance result a property of the allocator, or of where its solver stopped?

The harness's cost-aware mean-variance allocator solves one convex rebalance
per decision, each starting from the book the last one chose. By default
(``BacktestConfig.mv_solver=None``) it runs FISTA for at most ``max_iter=400``
iterations, a budget chosen when the panels were a few hundred names. On the
real month that budget runs out at every decision, so the book is wherever the
iterations stopped, and the error carries forward into the next decision's
starting point. ``mv_solver="restart"`` solves the same problem to the step
tolerance with accelerated proximal gradient and adaptive restart. This script
measures, on August 2020 at the 15-minute horizon with the IC-0.03 oracle that
:mod:`experiments.bench` times:

- the iterations each decision's solve ran and the share that met the step
  tolerance before its cap;
- every decision's optimality residuals: the proximal-gradient fixed-point
  residual in weight units (the step one more plain iteration would take) and
  the largest violation of the optimality (KKT) conditions in return units;
- every answer's distance from the exact optimum GIVEN THE SAME PREVIOUS BOOK,
  found by a primal-dual active-set solve that is accepted only once it
  verifies the optimality conditions, so the reference is checked rather than
  assumed;
- the seconds the rebalance took under each solver;
- what changes in the configuration's scores: the report's Sharpe ratios,
  turnover and cost, the daily Sharpe with its day-block bootstrap interval
  (as :mod:`experiments.run` writes it) and the break-even EFQ.

``--variants N`` also runs solver alternatives on the first N decisions, each
along its own path: FISTA at several budgets, and two restart variants that
are not in the library -- the textbook unscaled metric and the function-value
restart test -- which are the evidence for the library's choice of a
variance-scaled metric and the gradient test.

    uv run --extra experiments python -m experiments.mv_solver --efq 0.7 --variants 200

Writes ``results/mv_solver/summary.json`` and ``results/mv_solver/per_decision.npz``.
One EFQ costs three to four minutes of one core on the real month (two
harness runs of about 1.5 minutes each, then the exact references);
``--variants 200`` adds about three, most of it FISTA at 5000 iterations. The
command above with ``--efq 0.7 1.0 0.0`` took 13 minutes.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.linalg as sla

import stable_finance as sf
from experiments.run import _breakeven, _inference, load_inputs, restrict_decisions
from stable_finance.evaluation import backtest

OUT = Path(__file__).resolve().parent / "results" / "mv_solver"
SPEC = {"eval_month": "2020-08", "history_month": "2020-07", "horizons": [900]}
SOLVERS = (None, "restart")
SCORES = ("sharpe_mid", "sharpe_net", "mean_turnover", "mean_gross_return_bps",
          "mean_cost_bps", "mean_gross_exposure", "transfer_coefficient", "mean_names_held")


# ── the harness, instrumented ───────────────────────────────────────────────

@contextmanager
def recording(record: dict):
    """Route the harness's rebalance through one that also returns iteration counts.

    The harness calls :func:`~stable_finance.rebalance` once per run; the
    replacement makes the same call with ``return_n_iter=True`` and keeps the
    problem it was handed, so the diagnostics below re-examine exactly the
    problem the harness solved.
    """
    original = backtest.rebalance

    def instrumented(target, covariance, **kwargs):
        t0 = time.perf_counter()
        held, n_iter = original(target, covariance, return_n_iter=True, **kwargs)
        record.update(
            seconds=time.perf_counter() - t0, n_iter=n_iter[:, 0],
            goal=np.nan_to_num(target.values[:, :, 0], nan=0.0),
            sigma=np.asarray(covariance, dtype=np.float64)[0],
            aversion=np.asarray(kwargs["risk_aversion"], dtype=np.float64),
            cost=np.asarray(kwargs["cost"], dtype=np.float64),
            weights=held.values[:, :, 0].copy(), max_iter=int(kwargs["max_iter"]),
        )
        return held

    backtest.rebalance = instrumented
    try:
        yield
    finally:
        backtest.rebalance = original


# ── optimality, checked ─────────────────────────────────────────────────────

def _signal(goal, sigma, aversion):
    """``lambda_t * Sigma @ w*_t``, computed exactly as :func:`~stable_finance.rebalance` does."""
    return aversion[:, None] * np.einsum("hij,djh->dih", sigma[None], goal[:, :, None])[:, :, 0]


def _soft(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _objective(w, previous, sigma, signal, cost, lam) -> float:
    finite = np.isfinite(cost)
    return float(0.5 * lam * w @ sigma @ w - signal @ w
                 + np.sum(cost[finite] * np.abs(w[finite] - previous[finite])))


def residuals(w, previous, sigma, signal, cost, lam, largest) -> tuple[float, float]:
    """The fixed-point residual of one plain proximal-gradient step, and the KKT violation.

    The first is in weight units: how far one more unscaled iteration (step
    ``1 / (lam * L)``) would move ``w``; it is zero exactly at the optimum and
    is the quantity the step tolerance (1e-10) is a proxy for. The second is
    in return units, comparable with the cost per unit of weight: a traded
    name's marginal utility must equal its cost, an untraded one's must lie
    inside the cost band. Names with an infinite cost are held and exempt.
    """
    gradient = lam * (sigma @ w) - signal
    step = 1.0 / (lam * largest)
    fixed_point = float(np.max(np.abs(w - (previous + _soft(w - step * gradient - previous,
                                                            step * cost)))))
    finite = np.isfinite(cost)
    trade = w - previous
    traded = finite & (trade != 0)
    held = finite & (trade == 0)
    boundary = np.abs(gradient[traded] + cost[traded] * np.sign(trade[traded]))
    band = np.maximum(np.abs(gradient[held]) - cost[held], 0.0)
    return fixed_point, float(max(boundary.max(initial=0.0), band.max(initial=0.0)))


def exact_optimum(signal, sigma, previous, cost, lam, start, max_outer=50):
    """The optimum given ``previous``, by a primal-dual active-set solve from ``start``'s trades.

    Guess which names trade and in which direction; on that guess the
    optimality conditions are one linear system (traded names: marginal
    utility equals minus cost times direction), solved by Cholesky. Accept the
    answer only if it verifies every condition -- each traded name moves in its
    guessed direction and each untraded name's marginal utility lies inside
    its cost band -- and otherwise move the violators across and solve again.
    Returns the optimum, whether it verified, and the solves it took. The
    checks allow rounding: a trade of 1e-12 the wrong way is no trade, and the
    band is widened by 1e-12 of the largest marginal utility, both far below
    the distances being measured.
    """
    finite = np.isfinite(cost)
    slack = 1e-12 * max(float(np.max(np.abs(signal))), 1e-300)
    pattern = np.where(finite, np.sign(start - previous), 0.0)
    w = previous.copy()
    for outer in range(1, max_outer + 1):
        active = pattern != 0
        w = previous.copy()
        if active.any():
            rhs = (signal[active] - cost[active] * pattern[active]
                   - lam * (sigma[np.ix_(active, ~active)] @ previous[~active]))
            factor = sla.cho_factor(lam * sigma[np.ix_(active, active)], lower=True,
                                    check_finite=False)
            w[active] = sla.cho_solve(factor, rhs, check_finite=False)
        gradient = lam * (sigma @ w) - signal
        trade = w - previous
        wrong_way = active & (trade * pattern < -1e-12)
        outside = ~active & finite & (np.abs(gradient) > cost * (1.0 + 1e-9) + slack)
        if not wrong_way.any() and not outside.any():
            return w, True, outer
        pattern = np.where(wrong_way, 0.0, pattern)
        pattern = np.where(outside, -np.sign(gradient), pattern)
    return w, False, max_outer


def diagnose(record: dict) -> dict:
    """Per-decision residuals and distance from the verified optimum along one path."""
    goal, sigma, aversion, cost, weights = (record[k] for k in
                                            ("goal", "sigma", "aversion", "cost", "weights"))
    signal = _signal(goal, sigma, aversion)
    largest = float(np.linalg.eigvalsh(sigma).max())
    n_decisions, n_assets = weights.shape
    out = {k: np.empty(n_decisions) for k in ("fixed_point", "kkt", "distance", "gap")}
    out["verified"] = np.zeros(n_decisions, dtype=bool)
    out["solves"] = np.zeros(n_decisions, dtype=np.int64)
    previous = np.zeros(n_assets)
    for d in range(n_decisions):
        lam, w = float(aversion[d]), weights[d]
        out["fixed_point"][d], out["kkt"][d] = residuals(w, previous, sigma, signal[d],
                                                         cost[d], lam, largest)
        exact, out["verified"][d], out["solves"][d] = exact_optimum(
            signal[d], sigma, previous, cost[d], lam, start=w)
        out["distance"][d] = np.max(np.abs(w - exact))
        out["gap"][d] = (_objective(w, previous, sigma, signal[d], cost[d], lam)
                         - _objective(exact, previous, sigma, signal[d], cost[d], lam))
        previous = w
    return out


def _quantiles(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    q = np.percentile(values, [0, 10, 50, 90, 99, 100])
    return {"min": q[0], "p10": q[1], "median": q[2], "p90": q[3], "p99": q[4], "max": q[5],
            "mean": float(values.mean())}


# ── solver alternatives on a window ─────────────────────────────────────────

def _restart_variant(signal, sigma, previous, cost, lam, step, *, test, max_iter, tol):
    """The library's restart loop with a choice of metric (via ``step``) and restart test."""
    weights, momentum, t = previous.copy(), previous.copy(), 1.0
    value = _objective(weights, previous, sigma, signal, cost, lam)
    for iteration in range(1, max_iter + 1):
        gradient = lam * (sigma @ momentum) - signal
        updated = previous + _soft(momentum - step * gradient - previous, step * cost)
        change = updated - weights
        if np.max(np.abs(change)) < tol:
            return updated, iteration
        if test == "function":
            new_value = _objective(updated, previous, sigma, signal, cost, lam)
            restart, value = new_value > value, new_value
        else:
            restart = np.dot((momentum - updated) / step, change) > 0.0
        if restart:
            t, momentum = 1.0, updated
        else:
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum, t = updated + ((t - 1.0) / t_next) * change, t_next
        weights = updated
    return weights, max_iter


def variants(record: dict, n: int, tol: float = 1e-10) -> dict:
    """Every solver alternative on the first ``n`` decisions, each along its own path.

    The library's own solvers go through :func:`~stable_finance.rebalance`;
    the two restart variants it does not ship run the same loop here. Each
    path is compared with the library restart solver's path, which the full
    diagnostics verify against the exact optimum.
    """
    goal, sigma, aversion, cost = (record[k] for k in ("goal", "sigma", "aversion", "cost"))
    goal, aversion, cost = goal[:n], aversion[:n], cost[:n]
    target = sf.PortfolioWeights(goal[:, :, None], np.arange(n), np.arange(goal.shape[1]),
                                 [900])
    paths, out = {}, {}
    library = [("fista_400", None, 400), ("fista_1000", None, 1000),
               ("fista_5000", None, 5000), ("restart", "restart", 400)]
    for name, solver, budget in library:
        t0 = time.perf_counter()
        held, n_iter = sf.rebalance(target, sigma[None], risk_aversion=aversion, cost=cost,
                                    max_iter=budget, tol=tol, solver=solver,
                                    return_n_iter=True)
        seconds = time.perf_counter() - t0
        paths[name] = held.values[:, :, 0]
        cap = budget if solver is None else max(budget, sf.RESTART_MAX_ITER)
        out[name] = {"seconds": seconds, "iterations": _quantiles(n_iter[:, 0]),
                     "converged_share": float(np.mean(n_iter[:, 0] < cap))}
    signal = _signal(goal, sigma, aversion)
    largest = float(np.linalg.eigvalsh(sigma).max())
    variance = np.diag(sigma)
    scale = np.where(variance > 0, variance, largest)
    curvature = float(np.linalg.eigvalsh(sigma / np.sqrt(np.outer(scale, scale))).max())
    for name, scaled, test in (("restart_unscaled_gradient", False, "gradient"),
                               ("restart_scaled_function", True, "function")):
        previous, rows, counts = np.zeros(goal.shape[1]), [], []
        t0 = time.perf_counter()
        for d in range(n):
            lam = float(aversion[d])
            step = (1.0 / (lam * curvature * scale) if scaled
                    else np.full(goal.shape[1], 1.0 / (lam * largest)))
            previous, used = _restart_variant(signal[d], sigma, previous, cost[d], lam, step,
                                              test=test, max_iter=sf.RESTART_MAX_ITER,
                                              tol=tol)
            rows.append(previous)
            counts.append(used)
        paths[name] = np.asarray(rows)
        out[name] = {"seconds": time.perf_counter() - t0, "iterations": _quantiles(counts),
                     "converged_share": float(np.mean(np.asarray(counts) < sf.RESTART_MAX_ITER))}
    for name, entry in out.items():
        gap = np.abs(paths[name] - paths["restart"])
        entry["max_abs_from_restart_path"] = float(gap.max())
        entry["median_abs_from_restart_path"] = float(np.median(gap.max(axis=1)))
    return {"decisions": n, "solvers": out}


# ── the run ─────────────────────────────────────────────────────────────────

def run_efq(inputs: dict, forecast, efq: float) -> tuple[dict, dict, dict]:
    """Both solvers on one EFQ: scores, iterations, residuals, exact distances, seconds."""
    summary, arrays = {"efq": efq}, {}
    reports, records = {}, {}
    for solver in SOLVERS:
        name = solver or "fista"
        config = sf.DEFAULT_BACKTEST.replace(cost_model="cross", efq=efq, mv_solver=solver)
        record: dict = {}
        t0 = time.perf_counter()
        with recording(record):
            report = sf.run_backtest(forecast, inputs["realized"], market=inputs["market"],
                                     history=inputs["history"], days=inputs["days"],
                                     config=config, with_series=True)
        total = time.perf_counter() - t0
        reports[name], records[name] = report, record
        cap = (record["max_iter"] if solver is None
               else max(record["max_iter"], sf.RESTART_MAX_ITER))
        t0 = time.perf_counter()
        checks = diagnose(record)
        median_cost = float(np.median(record["cost"][np.isfinite(record["cost"])]))
        print(f"  efq={efq} {name}: harness {total:.1f}s, rebalance {record['seconds']:.1f}s, "
              f"checks {time.perf_counter() - t0:.1f}s", flush=True)
        row = {k: getattr(report, k) for k in SCORES}
        row.update(_inference(report.series))
        row["breakeven_efq"] = _breakeven(report.series, efq) if efq > 0 else float("nan")
        summary[name] = {
            "label": config.label,
            "harness_seconds": total,
            "rebalance_seconds": record["seconds"],
            "iteration_cap": cap,
            "iterations": _quantiles(record["n_iter"]),
            "converged_share": float(np.mean(record["n_iter"] < cap)),
            "converged": int(np.sum(record["n_iter"] < cap)),
            "decisions": len(record["n_iter"]),
            "fixed_point_residual": _quantiles(checks["fixed_point"]),
            "kkt_violation": _quantiles(checks["kkt"]),
            "kkt_violation_over_median_cost": (
                float(np.max(checks["kkt"]) / median_cost) if median_cost > 0 else float("nan")),
            "distance_from_exact": _quantiles(checks["distance"]),
            "objective_gap": _quantiles(checks["gap"]),
            "exact_verified_share": float(checks["verified"].mean()),
            "exact_solves": _quantiles(checks["solves"]),
            "scores": row,
        }
        for key in ("fixed_point", "kkt", "distance", "gap", "verified"):
            arrays[f"efq{efq}_{name}_{key}"] = checks[key]
        arrays[f"efq{efq}_{name}_n_iter"] = record["n_iter"]
    fista, restart = summary["fista"]["scores"], summary["restart"]["scores"]
    summary["change"] = {k: restart[k] - fista[k] for k in fista}
    path_gap = np.abs(records["restart"]["weights"] - records["fista"]["weights"])
    summary["path_difference"] = {"max_abs_weight": float(path_gap.max()),
                                  "median_decision_max_abs_weight":
                                      float(np.median(path_gap.max(axis=1))),
                                  "mean_abs_weight": float(path_gap.mean())}
    summary["rebalance_speed_ratio"] = (summary["restart"]["rebalance_seconds"]
                                        / summary["fista"]["rebalance_seconds"])
    return summary, arrays, records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--efq", type=float, nargs="+", default=[0.7])
    parser.add_argument("--variants", type=int, default=0,
                        help="also compare solver alternatives on the first N decisions")
    parser.add_argument("--decisions", type=int, default=0,
                        help="run on the first N decisions only (a smoke test); 0 is all")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(SPEC)
    if args.decisions:
        inputs = restrict_decisions(inputs, inputs["realized"].decisions[:args.decisions])
    forecast = sf.oracle_forecast(inputs["realized"], 0.03, noise="random", seed=0)
    summary = {"spec": SPEC, "forecast": {"kind": "oracle", "ic": 0.03, "noise": "random",
                                          "seed": 0},
               "assets": len(inputs["realized"].assets),
               "decisions": len(inputs["realized"].decisions),
               "restart_max_iter": sf.RESTART_MAX_ITER, "runs": []}
    arrays = {}
    for efq in args.efq:
        run, run_arrays, records = run_efq(inputs, forecast, efq)
        summary["runs"].append(run)
        arrays.update(run_arrays)
        if "covariance" not in summary:
            sigma = records["fista"]["sigma"]
            d = np.sqrt(np.diag(sigma))
            summary["covariance"] = {
                "condition_number": float(np.linalg.cond(sigma)),
                "correlation_condition_number": float(np.linalg.cond(sigma / np.outer(d, d)))}
        if args.variants and efq == args.efq[0]:
            t0 = time.perf_counter()
            summary["variants"] = {"efq": efq, **variants(records["fista"], args.variants)}
            print(f"  variants on {args.variants} decisions: {time.perf_counter() - t0:.1f}s",
                  flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(OUT / "per_decision.npz", **arrays)
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
