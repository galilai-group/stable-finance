"""Run an experiment spec: forecasts x horizons x harness configurations.

An experiment is a JSON spec naming a month to evaluate on, a month of risk
history before it, a set of forecasts (noised oracles at chosen ICs and noise
modes, copula oracles at chosen half-lives, factor-structured oracles whose
errors are factor bets and whose IC drifts, or a saved forecast file), the
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
A spec with ``"save_daily": true`` also writes ``daily.npz``, every row's P&L
summed over each trading day, which ``walkforward.py`` pools across months.

WHICH DATA. Three optional spec keys pick the evaluation data rather than the
harness, and a spec that names none of them runs exactly as before:

* ``"cache": "v1" | "v2"`` (default ``"v1"``): the month cache to read
  (:mod:`experiments.month_cache` or :mod:`experiments.month_cache_v2`). On
  v2 the market also carries ``adv``, each name's average daily dollar volume
  over the ``"adv_days"`` (default 20) trading days strictly before the
  decision's day, the history month's days included, which the ``impact``
  cost model needs.
* ``"returns": "vwap" | "mid"`` (default ``"vwap"``): the store's VWAP
  forward returns or v2's midquote-to-midquote ones, for the evaluation and
  the risk history alike. The universe is chosen from the VWAP panel either
  way, so the two are scored on the same names.
* ``"latency": 0 | 1 | 5 | 30`` (default 5): seconds between the decision and
  the touch an order meets. 5 is v1's execution quote; 0 is the decision's
  own quote; 1 and 30 need v2.

``returns`` and ``latency`` may be lists. Each combination is then a task axis
of its own (the inputs differ, so it cannot share a sweep), and every row
carries the keys the spec names, which also enter its ``row_id``.

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
from experiments.month_cache_v2 import load_month_v2
from stable_finance.dataset.characteristics import trailing_from_days

RESULTS = Path(__file__).resolve().parent / "results"

#: Options of the data keys (see the module docstring). Defaults reproduce v1.
CACHES = ("v1", "v2")
RETURNS = ("vwap", "mid")
LATENCIES = (0, 1, 5, 30)
DATA_DEFAULTS = {"returns": "vwap", "latency": 5}

#: Trading days in the ``adv`` window when the spec does not say.
ADV_DAYS = 20


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


def data_variants(spec: dict) -> list[dict]:
    """The spec's data settings, one dict per combination of ``returns`` and ``latency``.

    Only the keys the spec names appear, in a fixed order, so a spec naming
    neither has the one empty variant and its tasks, rows and row ids are
    exactly what they were before these keys existed.
    """
    named = [k for k in DATA_DEFAULTS if k in spec]
    options = [spec[k] if isinstance(spec[k], list) else [spec[k]] for k in named]
    return [dict(zip(named, combo)) for combo in itertools.product(*options)]


def variant_key(variant: dict | None) -> str:
    return json.dumps(variant or {}, sort_keys=True)


def _data_setting(spec: dict, variant: dict | None, key: str):
    """One data setting: the variant's, else the spec's (its first, if a list), else the default."""
    if variant and key in variant:
        return variant[key]
    value = spec.get(key, DATA_DEFAULTS[key])
    return value[0] if isinstance(value, list) else value


def check_data_keys(spec: dict) -> None:
    """Refuse a data setting the chosen cache cannot supply, before any month is read."""
    cache = spec.get("cache", "v1")
    if cache not in CACHES:
        raise ValueError(f"cache must be one of {CACHES}; got {cache!r}")
    adv_days = spec.get("adv_days", ADV_DAYS)
    if adv_days is not None and (not isinstance(adv_days, int) or adv_days < 1):
        raise ValueError(f"adv_days must be a positive number of days or null; got {adv_days!r}")
    for variant in data_variants(spec):
        returns, latency = (_data_setting(spec, variant, k) for k in ("returns", "latency"))
        if returns not in RETURNS:
            raise ValueError(f"returns must be one of {RETURNS}; got {returns!r}")
        if latency not in LATENCIES:
            raise ValueError(f"latency must be one of {LATENCIES} seconds; got {latency!r}")
        if cache == "v1" and (returns != "vwap" or latency not in (0, 5)):
            raise ValueError(f"returns={returns!r}, latency={latency!r} needs \"cache\": \"v2\"; "
                             "v1 holds VWAP returns and the touch at 0 and 5 seconds only")


def _load(spec: dict, month: str) -> dict:
    return load_month_v2(month) if spec.get("cache", "v1") == "v2" else load_month(month)


def trailing_adv(evaluation: dict, history: dict, assets, adv_days: int | None) -> np.ndarray:
    """``(decisions, assets)`` average daily dollar volume before each decision's day.

    The days are the history month's and the evaluation month's, in date
    order; each decision reads the ``adv_days`` days strictly before its own
    (all of them when ``None``), so nothing of its own day enters. This is
    :func:`~stable_finance.dataset.characteristics.trailing_from_days`, the
    rule ``market_from_days(lookback=...)`` uses: the mean over the days in
    the window on which the name traded, NaN when it traded on none.
    """
    days, dollars, closes = [], [], []
    for month in (history, evaluation):
        labels = month["days"]
        first = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        index = {a: i for i, a in enumerate(month["realized"].assets.astype(str))}
        take = np.array([index.get(a, -1) for a in np.asarray(assets).astype(str)])
        known = take >= 0
        for name, out in (("dollar_volume", dollars), ("day_close", closes)):
            block = np.full((len(first), len(take)), np.nan)
            block[:, known] = month[name][first][:, take[known]]
            out.append(block)
        days.append(labels[first])
    labels = np.concatenate(days)
    order = np.argsort(labels, kind="stable")
    labels, unique = np.unique(labels[order], return_index=True)
    pick = order[unique]
    lookback = len(labels) if adv_days is None else int(adv_days)
    adv = trailing_from_days(np.concatenate(closes)[pick], np.concatenate(dollars)[pick],
                             lookback)[0]
    row = {d: i for i, d in enumerate(labels.tolist())}
    return adv[[row[d] for d in evaluation["days"].tolist()]]


def assemble_inputs(spec: dict, evaluation: dict, history: dict,
                    variant: dict | None = None) -> dict:
    """One data variant's inputs from two loaded months (see :func:`load_inputs`)."""
    horizons = spec["horizons"]
    realized = _horizon_panel(evaluation["realized"], horizons)
    past = _horizon_panel(history["realized"], horizons)
    spread = evaluation["half_spread"]
    quoted = np.isfinite(spread).mean(axis=0) >= spec.get("min_presence", 0.95)
    seen = np.isfinite(past.values[:, :, 0]).mean(axis=0)
    keep = set(realized.assets[quoted].astype(str)) & set(past.assets[seen >= 0.95].astype(str))
    assets = np.asarray(sorted(keep))
    realized, past = _restrict(realized, assets), _restrict(past, assets)
    if _data_setting(spec, variant, "returns") == "mid":
        # Chosen from the VWAP panel above, so mid and VWAP runs score the same names.
        realized = _restrict(_horizon_panel(evaluation["realized_mid"], horizons), assets)
        past = _restrict(_horizon_panel(history["realized_mid"], horizons), assets)
    index = {a: i for i, a in enumerate(evaluation["realized"].assets.astype(str))}
    take = [index[a] for a in assets]
    market_columns = {q: evaluation["market"].get(q)[:, take]
                      for q in ("bid_price", "ask_price", "exec_bid_price", "exec_ask_price")}
    latency = _data_setting(spec, variant, "latency")
    if latency != 5:
        market = evaluation["market"]
        quotes = evaluation.get("exec", {0: (market.get("bid_price"), market.get("ask_price"))})
        bid, ask = quotes[latency]
        market_columns["exec_bid_price"], market_columns["exec_ask_price"] = bid[:, take], ask[:, take]
    if spec.get("cache", "v1") == "v2":
        market_columns["adv"] = trailing_adv(evaluation, history, assets,
                                             spec.get("adv_days", ADV_DAYS))
    market = sf.MarketData(market_columns, realized.decisions, assets)
    features = evaluation["features"][:, take]
    return {"realized": realized, "history": past, "market": market, "days": evaluation["days"],
            "features": features, "feature_names": evaluation["feature_names"]}


def load_inputs(spec: dict, variant: dict | None = None) -> dict:
    """The evaluation month, its risk history, and the market, on one asset grid.

    Names are the ones present in both months and quoted on at least
    ``min_presence`` of the evaluation decisions, so every risk model has a
    full history row for every name it is asked about. ``variant`` picks one
    data setting of a spec whose ``returns`` or ``latency`` is a list (default:
    the first of each).
    """
    check_data_keys(spec)
    evaluation, history = _load(spec, spec["eval_month"]), _load(spec, spec["history_month"])
    return assemble_inputs(spec, evaluation, history, variant)


def load_all_inputs(spec: dict) -> dict[str, dict]:
    """Every data variant's inputs, keyed by :func:`variant_key`; the months are read once."""
    check_data_keys(spec)
    evaluation, history = _load(spec, spec["eval_month"]), _load(spec, spec["history_month"])
    return {variant_key(v): assemble_inputs(spec, evaluation, history, v)
            for v in data_variants(spec)}


def _hash(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


# ── forecasts ───────────────────────────────────────────────────────────────

def _grid(f: dict, keys: tuple[str, ...], defaults: dict | None = None,
          optional: tuple[str, ...] = ()) -> list[dict]:
    """Every combination of a forecast grid's options; a scalar is one option.

    ``seed`` may be a list too, running the same forecast under each seed, so
    a result can be told apart from a lucky draw. ``scale`` and the keys in
    ``optional`` are passed on only when the grid sets them, so a spec that
    does not name them keeps its forecasts and its row ids. ``defaults`` fills
    keys the grid leaves out, so a row records the value the forecast used
    rather than a blank.
    """
    keys = (*keys, "seed", *(k for k in (*optional, "scale") if k in f))
    defaults = {"seed": 0, **(defaults or {})}
    options = [f[k] if isinstance(f.get(k), list) else [f.get(k, defaults.get(k))]
               for k in keys]
    return [{"kind": f["kind"], **dict(zip(keys, combo))}
            for combo in itertools.product(*options)]


#: A factor oracle's grid keys, named as :func:`~stable_finance.factor_forecast`
#: names them, and the library's defaults for the ones a spec leaves out.
#: ``quality_strength`` enters a row only when the grid names it.
FACTOR_KEYS = ("ic", "n_factors", "factor_share", "ic_persistence", "ic_dispersion",
               "quality", "half_life")
FACTOR_DEFAULTS = {"n_factors": 3, "factor_share": 0.3, "ic_dispersion": 0.0}


def forecast_specs(spec: dict) -> list[dict]:
    """One dict per forecast, in a fixed order.

    ``forecasts`` is one grid or a list of grids of different kinds, run in
    the order given.
    """
    grids = spec["forecasts"] if isinstance(spec["forecasts"], list) else [spec["forecasts"]]
    out = []
    for f in grids:
        if f["kind"] == "oracle":
            out += _grid(f, ("ic", "noise", "scatter", "persistence"))
        elif f["kind"] == "copula":
            out += _grid(f, ("ic", "half_life"))
        elif f["kind"] == "factor":
            out += _grid(f, FACTOR_KEYS, FACTOR_DEFAULTS, optional=("quality_strength",))
        elif f["kind"] == "file":
            out += [{"kind": "file", "path": p, "name": Path(p).stem} for p in f["paths"]]
        else:
            raise ValueError(f"unknown forecast kind {f['kind']!r}")
    return out


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
        if fspec.get("scale") is not None:
            kwargs["scale"] = fspec["scale"]
        return sf.oracle_forecast(realized, fspec["ic"], **kwargs)
    if fspec["kind"] == "copula":
        return sf.copula_forecast(realized, fspec["ic"], half_life=fspec["half_life"],
                                  days=inputs["days"], scale=fspec.get("scale"),
                                  seed=fspec["seed"])
    if fspec["kind"] == "factor":
        # The loadings (and "vol" quality) come from the history's first
        # horizon, the spec's shortest; "spread" quality reads the market.
        options = {k: fspec[k] for k in (*FACTOR_KEYS[1:], "quality_strength", "scale")
                   if fspec.get(k) is not None}
        return sf.factor_forecast(realized, fspec["ic"], history=inputs["history"],
                                  market=inputs["market"], days=inputs["days"],
                                  seed=fspec["seed"], **options)
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

#: Each data variant's inputs, keyed by :func:`variant_key`, once a worker has loaded them.
_INPUTS: dict[str, dict] | None = None


def _init(spec):
    global _INPUTS
    _INPUTS = load_all_inputs(spec)


def _breakeven(series, efq: float) -> float:
    """The EFQ at which this book's mean net return reaches zero.

    Fills are priced off the touch on ARRIVAL, so a period's cost can be
    negative when the quote moved in the order's favour, and cost is only
    approximately linear in EFQ. The Sharpe at each EFQ is therefore taken
    from its own run (EFQ is a grid axis), and only the break-even is read
    off one path.

    A configuration that charges fees (``fee_bps``, ``fee_per_share``,
    ``sec_fee_rate``) pays them at every EFQ, so they come off the gross
    rather than scaling with the spread: the break-even is then
    ``mean(gross - fee) / mean((cost - fee) / efq)``.
    """
    if efq <= 0:
        return float("nan")
    if series.fee is None:
        return float(sf.breakeven_efq(series.gross, series.cost / efq))
    return float(sf.breakeven_efq(series.gross, (series.cost - series.fee) / efq,
                                  fees=series.fee))


#: Resamples behind each bootstrap interval. The generator is seeded and
#: depends only on the number of days, so every row of a run resamples the
#: same days: intervals are reproducible and paired across configurations.
BOOTSTRAP_DRAWS = 2000


def _sharpe(daily: np.ndarray) -> np.ndarray:
    """Annualized Sharpe of daily P&L along the last axis; NaN when flat."""
    sd = daily.std(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, daily.mean(axis=-1) / sd * np.sqrt(252.0), np.nan)


def _inference(series) -> dict:
    """Daily Sharpe ratios and 95% day-block bootstrap intervals for them.

    The day is the honest unit for an intraday book. A Sharpe ratio computed
    per decision and annualized by the decisions in a year treats some
    twenty thousand overlapping, intraday-seasonal periods as independent,
    which inflates it; the P&L summed over each day is close to independent
    across days. The intervals resample whole days with replacement -- days
    are the natural blocks, since P&L is correlated within one -- with a
    seeded generator. One month gives about twenty days, so the intervals are
    wide and themselves approximate: read them as the scale of the
    uncertainty, not as exact coverage.
    """
    days = np.char.partition(np.asarray(series.periods).astype(str), "|")[:, 0]
    _, index = np.unique(days, return_inverse=True)
    n = int(index.max()) + 1
    draws = np.random.default_rng(0).integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    out = {"n_days": n}
    for name, values in (("mid", series.gross), ("net", series.net)):
        daily = np.bincount(index, weights=np.nan_to_num(values), minlength=n)
        out[f"sharpe_daily_{name}"] = float(_sharpe(daily))
        boot = _sharpe(daily[draws])
        boot = boot[np.isfinite(boot)]
        low, high = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
        out[f"sharpe_daily_{name}_low"], out[f"sharpe_daily_{name}_high"] = float(low), float(high)
    return out


def _daily(series) -> tuple[np.ndarray, np.ndarray]:
    """A profit path summed over each trading day: ``(days, (3, n_days))``.

    Rows are gross, net and cost, the same day sums ``_inference`` takes its
    Sharpe ratios from (NaN periods count as zero), so a figure or a pooled
    statistic built from them agrees with the row. The per-period series
    carries no period labels once saved, and a held or daily book has fewer
    periods than decisions, so the day of each period cannot be recovered
    afterwards; a study that conditions on the day -- the walk-forward's
    regime buckets, a pooled Sharpe over months -- needs the sums written now.
    """
    days = np.char.partition(np.asarray(series.periods).astype(str), "|")[:, 0]
    labels, index = np.unique(days, return_inverse=True)
    sums = np.stack([np.bincount(index, weights=np.nan_to_num(values), minlength=len(labels))
                     for values in (series.gross, series.net, series.cost)])
    return labels, sums


def run_task(spec: dict, fspec: dict, horizon: int, chunk: int = 0,
             variant: dict | None = None) -> tuple[list[dict], dict]:
    key, variant = variant_key(variant), variant or {}
    inputs = (_INPUTS[key] if _INPUTS is not None and key in _INPUTS
              else load_inputs(spec, variant))
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
    rows, series, daily = [], {}, {}
    tag = {k: v for k, v in fspec.items() if k != "kind"}
    for report in reports:
        row = {"forecast": fspec["kind"], **{f"f_{k}": v for k, v in tag.items()}, **variant,
               "eval_horizon": horizon, **report.as_dict()}
        if report.config.cost_model != "mid":
            row["breakeven_efq"] = _breakeven(report.series, report.config.efq)
        row.update(_inference(report.series))
        row["row_id"] = hashlib.sha256(json.dumps(
            {k: row[k] for k in sorted(row) if k.startswith("f_") or k in
             ("eval_horizon", *variant)} | {"label": report.config.label},
            sort_keys=True, default=str).encode()).hexdigest()[:16]
        rows.append(row)
        series[row["row_id"]] = np.stack([report.series.gross, report.series.net,
                                          report.series.cost])
        if spec.get("save_daily"):
            daily[row["row_id"]] = _daily(report.series)
    timing = {"forecast": tag, "horizon": horizon, "chunk": chunk, "configs": len(configs),
              "seconds": elapsed, **({"data": variant} if variant else {})}
    extra = {"__daily__": daily} if spec.get("save_daily") else {}
    return rows, series | {"__timing__": timing} | extra


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


def _task_key(spec: dict, fspec: dict, horizon: int, chunk: int = 0,
              variant: dict | None = None) -> str:
    """A task's identity: everything that determines its rows.

    The data keys enter only when the spec names them, so a spec that names
    none keeps the keys, and the checkpoints, it had before they existed. A
    variant enters as its resolved settings, not the spec's lists, so adding a
    latency to a list leaves the finished tasks of the others valid.
    """
    payload = {k: spec.get(k) for k in ("eval_month", "history_month", "horizons", "configs",
                                         "decision_spacing", "min_presence", "on_error")}
    payload.update(forecast=fspec, horizon=horizon)
    if int(spec.get("config_chunks", 1)) > 1:
        payload.update(config_chunks=int(spec["config_chunks"]), chunk=chunk)
    payload.update({k: spec[k] for k in ("cache", "adv_days") if k in spec})
    if variant:
        payload.update(data=variant)
    # Named only when set, like the chunks: a checkpoint written without day
    # sums must not be reused by a run that needs them, and every key a spec
    # without the option produced stays the same.
    if spec.get("save_daily"):
        payload.update(save_daily=True)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _write_daily(path: Path, daily: dict) -> None:
    """Every row's day sums on the run's union of days, NaN where a row has none.

    ``__days__`` holds the day labels and each row id a ``(3, n_days)`` array
    of gross, net and cost. A day a row has no period on (a forecast file
    covering part of the month) is NaN rather than zero, so it is not read as
    a flat day.
    """
    days = np.asarray(sorted({str(d) for labels, _ in daily.values() for d in labels}))
    column = {d: i for i, d in enumerate(days.tolist())}
    arrays = {}
    for row_id, (labels, sums) in daily.items():
        full = np.full((3, len(days)), np.nan)
        full[:, [column[str(d)] for d in labels]] = sums
        arrays[row_id] = full
    np.savez_compressed(path, __days__=days, **arrays)


def _run_and_save(spec: dict, fspec: dict, horizon: int, chunk: int, path: str,
                  variant: dict | None = None) -> str:
    """Run one task and checkpoint it atomically, so a killed run loses only it."""
    result = run_task(spec, fspec, horizon, chunk, variant)
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
    variants = data_variants(spec)
    all_inputs = load_all_inputs(spec)
    inputs = all_inputs[variant_key(variants[0])]
    # Recorded only for a spec that names the data keys, so other manifests keep their shape.
    data_hashes = [] if variants == [{}] else [
        {**v, "realized": _hash(x["realized"].values), "history": _hash(x["history"].values),
         "exec_bid_price": _hash(x["market"].get("exec_bid_price"))}
        for v, x in ((v, all_inputs[variant_key(v)]) for v in variants)]
    chunks = range(max(int(spec.get("config_chunks", 1)), 1))
    tasks = [(f, h, c, v) for f in forecast_specs(spec) for h in spec["horizons"]
             for v in variants for c in chunks]
    paths = [checkpoints / f"{_task_key(spec, f, h, c, v)}.pkl" for f, h, c, v in tasks]
    pending = [(f, h, c, v, str(p)) for (f, h, c, v), p in zip(tasks, paths) if not p.is_file()]
    print(f"{spec['name']}: {len(tasks) - len(pending)} of {len(tasks)} tasks already done",
          flush=True)
    t0 = time.perf_counter()
    if pending and workers > 1:
        del all_inputs     # each worker loads its own; the parent keeps only the manifest's
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(spec,)) as pool:
            futures = [pool.submit(_run_and_save, spec, f, h, c, p, v)
                       for f, h, c, v, p in pending]
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()
                print(f"  {spec['name']}: task {done}/{len(pending)} done at "
                      f"{time.strftime('%H:%M:%S')}", flush=True)
    elif pending:
        _init(spec)
        for done, (f, h, c, v, p) in enumerate(pending, start=1):
            _run_and_save(spec, f, h, c, p, v)
            print(f"  {spec['name']}: task {done}/{len(pending)} done", flush=True)
    rows, series, timings, daily = [], {}, [], {}
    for path in paths:
        with open(path, "rb") as handle:
            task_rows, task_series = pickle.load(handle)
        timings.append(task_series.pop("__timing__"))
        daily.update(task_series.pop("__daily__", {}))
        rows.extend(task_rows)
        series.update(task_series)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "rows.parquet", index=False)
    np.savez_compressed(out / "series.npz", **series)
    if spec.get("save_daily"):
        _write_daily(out / "daily.npz", daily)
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
    if inputs["market"].has("adv"):
        manifest["inputs"]["adv"] = _hash(inputs["market"].get("adv"))
    if data_hashes:
        manifest["data"] = data_hashes
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
