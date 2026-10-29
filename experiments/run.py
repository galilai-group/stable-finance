"""Run an experiment spec: forecasts x horizons x harness configurations.

An experiment is a JSON spec naming a month to evaluate on, a month of risk
history before it, a set of forecasts (noised oracles at chosen ICs and noise
modes, copula oracles at chosen half-lives, or a saved forecast file), the
horizons to score, and a grid of
:class:`~stable_finance.BacktestConfig` settings. Every (forecast, horizon)
pair is one :func:`~stable_finance.sweep_configs` call, so a grid shares its
risk models and books exactly as the library does, and the tasks run in
parallel processes.

EVERY NUMBER IS REPRODUCIBLE. Oracle noise is seeded from the spec, tasks are
written in a fixed order whatever finished first, and the manifest records the
git commit, the spec and a hash of every input array. Running a spec twice
gives byte-identical result rows.

Each row is a configuration's :meth:`~stable_finance.BacktestReport.as_dict`
plus the forecast's parameters, the horizon, and the break-even EFQ of its
own profit path. EFQ itself is a grid axis: fills are priced off the touch on
arrival, so cost is not exactly linear in EFQ and each point is its own run.

    uv run --extra experiments python -m experiments.run experiments/specs/e1_transfer_map.json
"""

from __future__ import annotations

import os

# One linear-algebra thread per worker: the parallelism is across tasks, and
# letting every worker's BLAS take every core oversubscribes the machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import itertools
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import stable_finance as sf
from experiments.month_cache import load_month

RESULTS = Path(__file__).resolve().parent / "results"


# ── inputs ──────────────────────────────────────────────────────────────────

def _horizon_panel(panel: sf.ForwardReturns, horizons) -> sf.ForwardReturns:
    index = [int(np.flatnonzero(panel.horizons == h)[0]) for h in horizons]
    return sf.ForwardReturns(panel.values[:, :, index], panel.decisions, panel.assets,
                             np.asarray(horizons))


def _common_assets(*panels):
    common = sorted(set.intersection(*(set(p.assets.astype(str)) for p in panels)))
    return np.asarray(common)


def _restrict(panel: sf.ForwardReturns, assets) -> sf.ForwardReturns:
    index = {a: i for i, a in enumerate(panel.assets.astype(str))}
    take = [index[a] for a in assets]
    return sf.ForwardReturns(panel.values[:, take], panel.decisions, np.asarray(assets),
                             panel.horizons)


def load_inputs(spec: dict) -> dict:
    """The evaluation month, its risk history, and the market, on one asset grid.

    Names are the ones present in both months and quoted on at least
    ``min_presence`` of the evaluation decisions, so every risk model has a
    full history row for every name it is asked about.
    """
    evaluation, history = load_month(spec["eval_month"]), load_month(spec["history_month"])
    horizons = spec["horizons"]
    realized = _horizon_panel(evaluation["realized"], horizons)
    past = _horizon_panel(history["realized"], horizons)
    spread = evaluation["half_spread"]
    quoted = np.isfinite(spread).mean(axis=0) >= spec.get("min_presence", 0.95)
    seen = np.isfinite(past.values[:, :, 0]).mean(axis=0)
    keep = set(realized.assets[quoted].astype(str)) & set(past.assets[seen >= 0.95].astype(str))
    assets = np.asarray(sorted(keep))
    realized, past = _restrict(realized, assets), _restrict(past, assets)
    index = {a: i for i, a in enumerate(evaluation["realized"].assets.astype(str))}
    take = [index[a] for a in assets]
    market_columns = {q: evaluation["market"].get(q)[:, take]
                      for q in ("bid_price", "ask_price", "exec_bid_price", "exec_ask_price")}
    market = sf.MarketData(market_columns, realized.decisions, assets)
    features = evaluation["features"][:, take]
    return {"realized": realized, "history": past, "market": market, "days": evaluation["days"],
            "features": features, "feature_names": evaluation["feature_names"]}


def _hash(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


# ── forecasts ───────────────────────────────────────────────────────────────

def forecast_specs(spec: dict) -> list[dict]:
    """One dict per forecast, in a fixed order."""
    f = spec["forecasts"]
    if f["kind"] == "oracle":
        keys = ("ic", "noise", "scatter", "persistence")
        grid = [f.get(k, [None]) if isinstance(f.get(k), list) else [f.get(k)] for k in keys]
        return [{"kind": "oracle", **dict(zip(keys, combo)), "seed": f.get("seed", 0)}
                for combo in itertools.product(*grid)]
    if f["kind"] == "copula":
        keys = ("ic", "half_life")
        grid = [f.get(k, [None]) if isinstance(f.get(k), list) else [f.get(k)] for k in keys]
        return [{"kind": "copula", **dict(zip(keys, combo)), "seed": f.get("seed", 0)}
                for combo in itertools.product(*grid)]
    if f["kind"] == "file":
        return [{"kind": "file", "path": p, "name": Path(p).stem} for p in f["paths"]]
    raise ValueError(f"unknown forecast kind {f['kind']!r}")


def restrict_decisions(inputs: dict, decisions) -> dict:
    """The evaluation inputs on a subset of their decisions, in their order.

    A forecast file may cover only some decisions -- an encoder evaluated at
    eight anchors a day rather than seventy-eight -- and a decision with no
    forecast must not be traded as though the forecast were flat, so the run
    is restricted to the decisions the forecast has.
    """
    labels = inputs["realized"].decisions.astype(str)
    wanted = set(np.asarray(decisions).astype(str))
    take = np.flatnonzero(np.isin(labels, list(wanted)))
    if len(take) == len(labels):
        return inputs
    realized = inputs["realized"]
    market = inputs["market"]
    return {**inputs,
            "realized": sf.ForwardReturns(realized.values[take], realized.decisions[take],
                                          realized.assets, realized.horizons),
            "market": sf.MarketData({q: v[take] for q, v in market.columns.items()},
                                    realized.decisions[take], realized.assets),
            "days": inputs["days"][take], "features": inputs["features"][take]}


def make_forecast(fspec: dict, inputs: dict) -> sf.ForwardReturns:
    realized = inputs["realized"]
    if fspec["kind"] == "oracle":
        kwargs = {"noise": fspec["noise"], "seed": fspec["seed"]}
        if fspec.get("scatter") is not None:
            kwargs["scatter"] = fspec["scatter"]
        if fspec.get("persistence") not in (None, 1):
            kwargs["persistence"] = fspec["persistence"]
            kwargs["days"] = inputs["days"]
        return sf.oracle_forecast(realized, fspec["ic"], **kwargs)
    if fspec["kind"] == "copula":
        return sf.copula_forecast(realized, fspec["ic"], half_life=fspec["half_life"],
                                  days=inputs["days"], seed=fspec["seed"])
    data = np.load(fspec["path"], allow_pickle=False)
    values = data["forecast"]
    index = {a: i for i, a in enumerate(data["assets"].astype(str))}
    take = [index.get(a, -1) for a in realized.assets.astype(str)]
    horizons = [int(np.flatnonzero(data["horizons"] == h)[0]) for h in realized.horizons]
    rows = {d: i for i, d in enumerate(data["decisions"].astype(str))}
    order = [rows[d] for d in realized.decisions.astype(str)]
    out = np.full((len(order), len(take), len(horizons)), np.nan)
    known = np.asarray(take) >= 0
    out[:, known] = values[np.ix_(order, np.asarray(take)[known])][:, :, horizons]
    return sf.ForwardReturns(out, realized.decisions, realized.assets, realized.horizons)


# ── configurations ──────────────────────────────────────────────────────────

def configs_for(spec: dict, horizon: int, spacing: int) -> list[sf.BacktestConfig]:
    """The spec's grid at one horizon.

    ``rebalance="horizon"`` at a horizon no longer than the decision spacing
    is ``every_decision``, so it is run once, under that name.
    """
    base = dict(spec["configs"].get("base", {}))
    grid = spec["configs"].get("grid", {})
    names = list(grid)
    out = []
    for combo in itertools.product(*(grid[n] for n in names)):
        settings = {**base, **dict(zip(names, combo))}
        if settings.get("rebalance") == "horizon" and horizon <= spacing:
            settings["rebalance"] = "every_decision"
        if any(all(settings.get(k) == v for k, v in rule.items())
               for rule in spec["configs"].get("exclude", [])):
            continue
        out.append(sf.BacktestConfig(**settings))
    unique = list(dict.fromkeys(out))
    return unique


# ── one task ────────────────────────────────────────────────────────────────

_INPUTS: dict | None = None


def _init(spec):
    global _INPUTS
    _INPUTS = load_inputs(spec)


def _breakeven(series, efq: float) -> float:
    """The EFQ at which this book's mean net return reaches zero.

    Fills are priced off the touch on ARRIVAL, so a period's cost can be
    negative when the quote moved in the order's favour, and cost is only
    approximately linear in EFQ. The Sharpe at each EFQ is therefore taken
    from its own run (EFQ is a grid axis), and only the break-even is read
    off one path.
    """
    if efq <= 0:
        return float("nan")
    return float(sf.breakeven_efq(series.gross, series.cost / efq))


def run_task(spec: dict, fspec: dict, horizon: int, chunk: int = 0) -> tuple[list[dict], dict]:
    inputs = _INPUTS if _INPUTS is not None else load_inputs(spec)
    if fspec["kind"] == "file":
        covered = np.load(fspec["path"], allow_pickle=False)["decisions"]
        inputs = restrict_decisions(inputs, covered)
    realized = inputs["realized"]
    forecast = make_forecast(fspec, inputs)
    hi = int(np.flatnonzero(realized.horizons == horizon)[0])
    spacing = spec.get("decision_spacing", 300)
    configs = _chunk(configs_for(spec, horizon, spacing), spec, chunk)
    t0 = time.perf_counter()
    reports = sf.sweep_configs(forecast, realized, configs, market=inputs["market"],
                               history=inputs["history"], days=inputs["days"],
                               horizon_index=hi, with_series=True, on_error=spec.get("on_error", "skip"))
    elapsed = time.perf_counter() - t0
    rows, series = [], {}
    tag = {k: v for k, v in fspec.items() if k != "kind"}
    for report in reports:
        row = {"forecast": fspec["kind"], **{f"f_{k}": v for k, v in tag.items()},
               "eval_horizon": horizon, **report.as_dict()}
        if report.config.cost_model != "mid":
            row["breakeven_efq"] = _breakeven(report.series, report.config.efq)
        row["row_id"] = hashlib.sha256(json.dumps(
            {k: row[k] for k in sorted(row) if k.startswith("f_") or k in
             ("eval_horizon",)} | {"label": report.config.label},
            sort_keys=True, default=str).encode()).hexdigest()[:16]
        rows.append(row)
        series[row["row_id"]] = np.stack([report.series.gross, report.series.net,
                                          report.series.cost])
    timing = {"forecast": tag, "horizon": horizon, "chunk": chunk, "configs": len(configs),
              "seconds": elapsed}
    return rows, series | {"__timing__": timing}


# ── the experiment ──────────────────────────────────────────────────────────

def _git_sha() -> str:
    """The commit the run starts from, marked ``-dirty`` if the tree has changes."""
    here = Path(__file__).parent
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=here, text=True).strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _chunk(configs: list, spec: dict, chunk: int) -> list:
    """One of ``config_chunks`` contiguous slices of a task's configurations.

    A spec whose (forecast, horizon) tasks are long -- many cost-aware books
    in one task -- can split each into chunks, so it parallelizes over more
    workers and a killed run loses one chunk rather than the whole task.
    Slices are contiguous because neighbouring configurations share books.
    """
    n = int(spec.get("config_chunks", 1))
    if n <= 1:
        return configs
    bounds = np.linspace(0, len(configs), n + 1).round().astype(int)
    return configs[bounds[chunk]:bounds[chunk + 1]]


def _task_key(spec: dict, fspec: dict, horizon: int, chunk: int = 0) -> str:
    """A task's identity: everything that determines its rows."""
    payload = {k: spec.get(k) for k in ("eval_month", "history_month", "horizons", "configs",
                                         "decision_spacing", "min_presence", "on_error")}
    payload.update(forecast=fspec, horizon=horizon)
    if int(spec.get("config_chunks", 1)) > 1:
        payload.update(config_chunks=int(spec["config_chunks"]), chunk=chunk)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _run_and_save(spec: dict, fspec: dict, horizon: int, chunk: int, path: str) -> str:
    """Run one task and checkpoint it atomically, so a killed run loses only it."""
    result = run_task(spec, fspec, horizon, chunk)
    partial = Path(path).with_suffix(".partial")
    with open(partial, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    partial.rename(path)
    return path


def run(spec_path: str, workers: int = 4) -> Path:
    """Run a spec, resuming from any task checkpoints a killed run left.

    Each (forecast, horizon) task is checkpointed under ``tasks/`` the moment
    it finishes; a rerun skips the tasks already there. The result files are
    assembled in task order once every task is present, so a resumed run
    writes exactly the rows an uninterrupted one would.
    """
    import pandas as pd

    spec = json.loads(Path(spec_path).read_text())
    git_sha = _git_sha()
    out = RESULTS / spec["name"]
    checkpoints = out / "tasks"
    checkpoints.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(spec)
    chunks = range(max(int(spec.get("config_chunks", 1)), 1))
    tasks = [(f, h, c) for f in forecast_specs(spec) for h in spec["horizons"] for c in chunks]
    paths = [checkpoints / f"{_task_key(spec, f, h, c)}.pkl" for f, h, c in tasks]
    pending = [(f, h, c, str(p)) for (f, h, c), p in zip(tasks, paths) if not p.is_file()]
    print(f"{spec['name']}: {len(tasks) - len(pending)} of {len(tasks)} tasks already done",
          flush=True)
    t0 = time.perf_counter()
    if pending and workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(spec,)) as pool:
            futures = [pool.submit(_run_and_save, spec, f, h, c, p) for f, h, c, p in pending]
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()
                print(f"  {spec['name']}: task {done}/{len(pending)} done at "
                      f"{time.strftime('%H:%M:%S')}", flush=True)
    elif pending:
        _init(spec)
        for done, (f, h, c, p) in enumerate(pending, start=1):
            _run_and_save(spec, f, h, c, p)
            print(f"  {spec['name']}: task {done}/{len(pending)} done", flush=True)
    rows, series, timings = [], {}, []
    for path in paths:
        with open(path, "rb") as handle:
            task_rows, task_series = pickle.load(handle)
        timings.append(task_series.pop("__timing__"))
        rows.extend(task_rows)
        series.update(task_series)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "rows.parquet", index=False)
    np.savez_compressed(out / "series.npz", **series)
    manifest = {
        "spec": spec, "git_sha": git_sha, "n_rows": len(rows), "n_tasks": len(tasks),
        "resumed_tasks": len(tasks) - len(pending),
        "wall_seconds_this_session": time.perf_counter() - t0,
        "task_seconds_total": float(sum(t["seconds"] for t in timings)),
        "workers": workers, "timings": timings,
        "inputs": {"assets": len(inputs["realized"].assets),
                   "decisions": len(inputs["realized"].decisions),
                   "realized": _hash(inputs["realized"].values),
                   "history": _hash(inputs["history"].values)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"{spec['name']}: {len(rows)} rows from {len(tasks)} tasks in "
          f"{manifest['wall_seconds_this_session']:.0f}s -> {out}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    run(args.spec, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
