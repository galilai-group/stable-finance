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
* THE FUNDAMENTAL LAW. Realized IR per held book against ``TC * IC *
  sqrt(N)`` (Grinold 1989, with the transfer coefficient), on E1x's
  random-error oracle (E1's without it), the one whose IC is a target in
  expectation. The books are rebuilt from the run's own spec, so every factor
  is measured in one set of units: TC in the identity metric the rank IC
  lives in, both at the formations the P&L is booked at, N the screened
  universe the TC spans. Beside the law, a second prediction keeps what the
  law drops: the value IC (which the P&L carries, not the rank IC) times the
  cross-sectional dispersion, whose swings cost IR. Realized / predicted is
  reported by IC with a day-bootstrap interval. An oracle whose IC is exact
  at every decision is outside the law altogether (each period's P&L is
  about ``N * IC * sd(w) * sd(r)``, almost never negative), which the
  exact-IC invariance diagnostic below makes visible.
* EXACT-IC INVARIANCE. Gross Sharpe of the exact-IC signal book across ICs,
  from E1, one curve per placement (tails, middle), horizon and rebalance
  rule. If the curves are roughly flat in IC, exact-IC gross Sharpe measures
  the dispersion process, not forecast quality, and should be read only as
  a cost-ratio device.
* PLACEBO. E6's IC-0 forecasts under ten seeds, each distinct book once: the
  daily gross Sharpe must centre on zero relative to its spread across
  seeds, and the 95% intervals must exclude zero no more often than 5%,
  within the interval of that rate.
* INVARIANCE AT IC 1. E6's tails, middle and random oracles at IC 1 are the
  same forecast and must give identical rows.
* LOOK-AHEAD CANARIES, on the real month: the realized forward return used as
  its own forecast must give an absurd Sharpe, and the same return shifted
  far enough to end before the decision -- the first lag clear of the
  forward window, and one day -- must stay an order of magnitude below it.
  Lag 1 shares the post-decision VWAP minute with its target and is reported
  only. The canary catches a misalignment of whole decisions, not a leak of
  seconds or one entering outside the forecast.
* COST IDENTITY. From every saved series: net is exactly gross minus cost, and
  each row's break-even EFQ is recomputed from its own series, with how many
  rows each covers.
* LATENCY. With a v2 cache, one book at execution delays of 0, 1, 5 and 30
  seconds. A delay only swaps the quotes an order meets (returns still start
  at the decision), so gross must not move with it. Whether waiting costs
  more is reported as mean cost per period and daily net Sharpe, with and
  without each day's first decision, the 09:30:00 open, whose quotes are
  wide for its first seconds; a delay that lowers cost is a FLAG. Skipped
  without one.
* DETERMINISM. One small task run twice, and with one and four BLAS threads,
  must give byte-identical rows and per-period series.

    uv run --extra experiments python -m experiments.sanity [--results DIR] [--quick]
        [--only CHECK ...]

``--results`` points at the directory holding finished runs (default
``experiments/results``); the report goes to ``<results>/sanity/report.json``
unless ``--out`` says otherwise. ``--quick`` skips the checks that load a
month and run backtests: the law, the canaries, the latency and determinism
runs. ``--only`` runs just the named checks (see :data:`CHECKS`).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist

import numpy as np

import stable_finance as sf
from experiments.run import BOOTSTRAP_DRAWS, RESULTS
from stable_finance.evaluation._grid import _rankdata

#: The closed-form transfer coefficient of each book on Gaussian signals and
#: an identity covariance, as (selection, weighting) -> TC.
ANALYTIC_TC = {
    ("all", "signal"): 1.0,
    ("all", "rank"): float(np.sqrt(3.0 / np.pi)),
    ("all", "equal"): float(np.sqrt(2.0 / np.pi)),
    ("decile", "equal"): 2 * NormalDist().pdf(NormalDist().inv_cdf(0.9)) / np.sqrt(0.2),
    ("quintile", "equal"): 2 * NormalDist().pdf(NormalDist().inv_cdf(0.8)) / np.sqrt(0.4),
}
#: Realized over predicted IR: a ratio whose interval lies wholly outside this
#: band is flagged (see :func:`fundamental_law` for which side of it counts).
LAW_BAND = (0.85, 1.15)
#: The runs the law is read from, in order of preference: the expected-IC
#: oracles of E1x, else E1's.
LAW_RUNS = ("e1x_expected_ic", "e1_transfer_map")
#: The books the identity-metric law describes: those built from the forecast
#: alone. HRP and minimum variance size by the covariance, a transfer the
#: identity-metric TC does not measure and the rank IC has no counterpart for.
LAW_WEIGHTINGS = ("equal", "rank", "signal")
#: The predictions realized IR is read against, as the report names them.
LAW_PREDICTIONS = ("law", "value_dispersion")
#: The harness's configuration fields, as the columns a row records them in.
CONFIG_FIELDS = tuple(f.name for f in dataclasses.fields(sf.BacktestConfig))


def _result(name: str, status: str, **numbers) -> dict:
    line = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in numbers.items() if not isinstance(v, (list, dict)))
    print(f"{status:<4}  {name}: {line}")
    return {"check": name, "status": status, **numbers}


def _rows(results: Path, name: str):
    import pandas as pd

    path = results / name / "rows.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _missing(value) -> bool:
    value = _scalar(value)
    return value is None or (isinstance(value, float) and math.isnan(value))


def _row_config(row) -> sf.BacktestConfig:
    """The configuration a row was written from, read back from its columns.

    A field the row leaves blank -- a column the run wrote for other rows,
    NaN or None here -- takes the default, as it did when the row ran; numpy
    scalars become the Python values the configuration was built from.
    """
    settings = {}
    for name in CONFIG_FIELDS:
        if name in row and not _missing(row[name]):
            value = _scalar(row[name])
            settings[name] = tuple(value) if isinstance(value, (list, np.ndarray)) else value
    return sf.BacktestConfig(**settings)


def _row_forecast(row) -> dict:
    """The forecast spec a row was written from: its kind and ``f_`` columns, blanks left out."""
    spec = {"kind": _scalar(row["forecast"])}
    for key, value in row.items():
        if key.startswith("f_") and not _missing(value):
            spec[key[2:]] = _scalar(value)
    return spec


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


# ── the fundamental law ────────────────────────────────────────────────────

def _row_correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row's entries under ``mask``; NaN below three of them."""
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(mask, left, 0.0)
        y = np.where(mask, right, 0.0)
        x = np.where(mask, x - x.sum(axis=1, keepdims=True) / n[:, None], 0.0)
        y = np.where(mask, y - y.sum(axis=1, keepdims=True) / n[:, None], 0.0)
        out = (x * y).sum(axis=1) / np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    return np.where((n >= 3) & np.isfinite(out), out, np.nan)


def formation_statistics(mu: np.ndarray, outcome: np.ndarray,
                         screened: np.ndarray) -> dict[str, np.ndarray]:
    """Each row's rank IC, value IC, outcome dispersion and breadth over the screen.

    Rows are decisions -- here the formations of a held book -- and the
    cross-section is the screened names with a known forecast and a known
    outcome, the names the harness scores its IC on. The rank IC is the
    harness's (:func:`~stable_finance.grouped_rank_ic` on one decision); the
    value IC is the Pearson correlation a signal book's P&L carries; the
    dispersion is the outcome's cross-sectional standard deviation. A row
    whose outcome is never known has breadth 0 and NaN for the rest.
    """
    known = screened & np.isfinite(mu) & np.isfinite(outcome)
    rank_mu, rank_outcome = np.full(mu.shape, np.nan), np.full(mu.shape, np.nan)
    dispersion = np.full(len(mu), np.nan)
    for t in np.flatnonzero(known.any(axis=1)):
        m = known[t]
        rank_mu[t, m], rank_outcome[t, m] = _rankdata(mu[t, m]), _rankdata(outcome[t, m])
        dispersion[t] = outcome[t, m].std()
    return {"ic_rank": _row_correlation(rank_mu, rank_outcome, known),
            "ic_value": _row_correlation(mu, outcome, known),
            "dispersion": dispersion, "breadth": known.sum(axis=1)}


def identity_tc(book: np.ndarray, alpha: np.ndarray, screened: np.ndarray) -> np.ndarray:
    """Each row's transfer coefficient in the identity metric: corr(weights, alpha).

    The metric the analytic transfer coefficients and the rank IC both live
    in, where the harness's own TC whitens by the covariance. It reads the
    screened names, a name the book does not hold counting as a zero weight,
    with roundoff residue zeroed under the harness's threshold.
    """
    held = np.where(np.isfinite(book) & (np.abs(book) <= 1e-12), 0.0, book)
    return _row_correlation(held, alpha, screened & np.isfinite(held) & np.isfinite(alpha))


def day_draws(day, n_draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> np.ndarray:
    """Period weights for a day bootstrap: row 0 the sample itself, then one row per draw.

    Each draw resamples the days with replacement, and a period counts as
    often as its day was drawn, so a statistic computed under every row is
    its sampling distribution with the within-day correlation kept. The
    draws are the ones :func:`experiments.run._inference` makes for a row's
    daily intervals: the same generator, seed and number of days.
    """
    _, index = np.unique(np.asarray(day), return_inverse=True)
    n = int(index.max()) + 1
    draws = np.random.default_rng(seed).integers(0, n, size=(n_draws, n))
    counts = np.zeros((n_draws, n))
    np.add.at(counts, (np.arange(n_draws)[:, None], draws), 1.0)
    return np.vstack([np.ones(len(index)), counts[:, index]])


def _moments(weights: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample sd of ``values`` under each row of period weights; NaN drops out."""
    known = np.isfinite(values)
    w, v = weights[:, known], values[known]
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = w @ v / total
        var = (w * (v[None, :] - mean[:, None]) ** 2).sum(axis=1) / (total - 1)
    return mean, np.sqrt(var)


def law_draws(gross: np.ndarray, tc: np.ndarray, stats: dict[str, np.ndarray],
              weights: np.ndarray) -> dict[str, np.ndarray]:
    """Realized IR and its two predictions under each row of ``weights``.

    ``gross`` is the book's P&L at each formation, ``tc`` its identity TC
    there and ``stats`` the formations' :func:`formation_statistics`. EVERY
    FACTOR READS THE SAME PERIODS, the formations whose outcome is known: a
    formation whose forward return is never known (the day's last, whose
    return runs past the close) books nothing, and counting it as a zero
    P&L in the IR while the IC skips it would mix two time bases. Returned,
    per row of ``weights``:

    * ``ir``: mean over sd of the P&L;
    * ``law``: ``TC * IC * sqrt(N)`` from the mean TC, the mean rank IC and
      the mean breadth;
    * ``value_dispersion``: TC times the IR of ``IC_value * dispersion``,
      which is a signal book's P&L period by period up to its scale: the law
      with the forecast's own IR measured in the units the P&L is in, the
      value IC's shortfall from the rank IC and the dispersion's swings kept;
    * ``tc``, ``ic_rank``, ``ic_value``, ``breadth`` (the means) and
      ``n_periods``.
    """
    known = np.isfinite(stats["ic_rank"]) & np.isfinite(stats["ic_value"])
    w = weights[:, known]
    mean, sd = _moments(w, gross[known])
    tc_mean = _moments(w, tc[known])[0]
    ic_rank = _moments(w, stats["ic_rank"][known])[0]
    ic_value = _moments(w, stats["ic_value"][known])[0]
    breadth = _moments(w, stats["breadth"][known].astype(np.float64))[0]
    signal_mean, signal_sd = _moments(w, (stats["ic_value"] * stats["dispersion"])[known])
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"ir": mean / sd, "law": tc_mean * ic_rank * np.sqrt(breadth),
                "value_dispersion": tc_mean * signal_mean / signal_sd,
                "tc": tc_mean, "ic_rank": ic_rank, "ic_value": ic_value, "breadth": breadth,
                "n_periods": w.sum(axis=1)}


def _ratio_interval(draws: np.ndarray) -> dict:
    """The sample's ratio (row 0) and the 95% interval of the draws behind it."""
    boot = draws[1:][np.isfinite(draws[1:])]
    low, high = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    return {"ratio": float(draws[0]), "low": float(low), "high": float(high)}


def _law_inputs(out: Path, horizons) -> dict:
    """The run's inputs, from the spec its manifest records (else the month pair)."""
    from experiments.run import load_inputs

    manifest = out / "manifest.json"
    spec = json.loads(manifest.read_text()).get("spec") if manifest.is_file() else None
    if spec is None:
        spec = {"eval_month": "2020-08", "history_month": "2020-07",
                "horizons": sorted(int(h) for h in horizons)}
    return load_inputs(spec)


def fundamental_law(results: Path) -> dict:
    """Realized IR of the random-error oracle's held books against the law, in one set of units.

    The rows are E1x's random-placement oracles (E1's when there is no E1x),
    held to the horizon, at EFQ 0 since gross does not depend on it, on the
    books :data:`LAW_WEIGHTINGS` names. Each book is rebuilt from the run's
    spec -- the forecast from the row's ``f_`` columns, the configuration from
    its config columns -- and its saved gross series is checked against the
    rebuild. The earlier version of this check read four mismatched factors
    off the rows: a covariance-metric TC averaged over carried decisions
    while P&L is booked only at formations, N the names held although the TC
    spans the whole screen, and a rank IC over every decision; it reported
    realized IR at 3-6 times the law. Here all of them are measured at the
    formations (see :func:`law_draws`), and realized / predicted is reported
    by IC as the median over seeds and books, with a 95% interval from one
    set of day draws shared by every row.

    The law is a small-IC approximation that a book's realized IR falls
    short of as IC grows (Clarke, de Silva and Thorley 2005): the value IC
    drops below the rank IC and the P&L inherits the dispersion's swings.
    The value-IC x dispersion prediction keeps both, so it is held to
    :data:`LAW_BAND` on both sides; the law only from above, a realized IR
    beating it beyond its interval being what a look-ahead or a unit error
    would produce. Either is a FLAG to investigate, not a pipeline failure;
    a rebuild that does not reproduce the saved P&L is a FAIL.
    """
    import pandas as pd

    from experiments.run import make_forecast
    from stable_finance.evaluation import backtest

    name = "fundamental law"
    out = next((results / n for n in LAW_RUNS
                if (results / n / "rows.parquet").is_file()
                and (results / n / "series.npz").is_file()), None)
    if out is None:
        return _result(name, "SKIP", reason="no E1x or E1 run with series")
    rows = pd.read_parquet(out / "rows.parquet")
    rows = rows[(rows["forecast"] == "oracle") & (rows["f_noise"] == "random")
                & (rows["rebalance"] == "horizon") & rows["weighting"].isin(LAW_WEIGHTINGS)
                & (rows["efq"] == 0.0) & (rows["f_ic"] > 0) & (rows["f_ic"] < 0.5)]
    if rows.empty:
        return _result(name, "SKIP", reason=f"no random-error held rows in {out.name}")
    inputs = _law_inputs(out, rows["eval_horizon"].unique())
    realized, series = inputs["realized"], np.load(out / "series.npz")
    keys = [json.dumps(_row_forecast(r), sort_keys=True, default=str) for _, r in rows.iterrows()]
    sigmas, draws, by_ic, entries, worst = {}, {}, {}, [], 0.0
    for (key, horizon), part in rows.groupby([keys, "eval_horizon"], sort=False):
        forecast = make_forecast(json.loads(key), inputs)
        hi = int(np.flatnonzero(realized.horizons == horizon)[0])
        context = backtest._prepare(forecast, realized, inputs["market"], inputs["days"], hi)
        formed = backtest._formations(context)
        mu, outcome = context["mu"][formed], context["outcome_values"][formed]
        if horizon not in draws:
            draws[horizon] = day_draws(context["day"][formed])
        stats = {}
        for _, row in part.iterrows():
            config = _row_config(row)
            gross = series[row["row_id"]][0] if row["row_id"] in series.files else None
            if gross is None or len(gross) != formed.sum():
                continue
            risk = (hi, config.risk_model, config.shrinkage)
            if risk not in sigmas:
                sigmas[risk] = backtest._risk(context, config, inputs["history"], realized,
                                              None, hi)
            weights, alpha = backtest._book(context, config, sigmas[risk])
            fills = backtest._fills_toward(context, backtest._hold_from(weights, formed), config)
            held = sf.positions_from_fills(fills).values[:, :, 0][formed]
            # The rebuild must be the row: its gross at each formation is the saved series.
            rebuilt = np.where(np.isfinite(held) & np.isfinite(outcome), held * outcome,
                               0.0).sum(axis=1)
            worst = max(worst, float(np.max(np.abs(rebuilt - gross))))
            screened = sf.apply_screen(config.universe, context["spread"],
                                       shape=context["mu"].shape)[formed]
            if config.universe not in stats:
                stats[config.universe] = formation_statistics(mu, outcome, screened)
            result = law_draws(gross, identity_tc(held, alpha[formed], screened),
                               stats[config.universe], draws[horizon])
            ratios = {p: result["ir"] / result[p] for p in LAW_PREDICTIONS}
            by_ic.setdefault(float(row["f_ic"]), []).append(ratios)
            entries.append({
                "ic": float(row["f_ic"]), "seed": _scalar(row.get("f_seed")),
                "horizon": int(horizon), "weighting": row["weighting"],
                "selection": row["selection"],
                **{k: float(result[k][0]) for k in ("ir", *LAW_PREDICTIONS, "tc", "ic_rank",
                                                    "ic_value", "breadth")},
                **{f"ratio_{p}": float(ratios[p][0]) for p in LAW_PREDICTIONS},
                "names_held": float(row["mean_names_held"]),
                "n_periods": int(result["n_periods"][0])})
    if not entries:
        return _result(name, "SKIP", reason=f"no row of {out.name} has a series to rebuild")
    pooled = {}
    for ic in sorted(by_ic):
        pooled[f"{ic:g}"] = {"ic": ic, "n_rows": len(by_ic[ic]), **{
            p: _ratio_interval(np.median(np.stack([r[p] for r in by_ic[ic]]), axis=0))
            for p in LAW_PREDICTIONS}}
    above = [k for k, e in pooled.items() if e["law"]["low"] > LAW_BAND[1]]
    unexplained = [k for k, e in pooled.items() if e["value_dispersion"]["high"] < LAW_BAND[0]
                   or e["value_dispersion"]["low"] > LAW_BAND[1]]
    status = "FAIL" if worst > 1e-12 else ("FLAG" if above or unexplained else "PASS")
    first, last = pooled[min(pooled, key=float)], pooled[max(pooled, key=float)]
    span = {p: f"{first[p]['ratio']:.2f} at IC {first['ic']:g} -> "
               f"{last[p]['ratio']:.2f} at IC {last['ic']:g}" for p in LAW_PREDICTIONS}
    return _result(name, status, run=out.name, n_rows=len(entries),
                   ratio_law=span["law"], ratio_value_dispersion=span["value_dispersion"],
                   law_above_band=above, value_dispersion_outside_band=unexplained,
                   band=list(LAW_BAND), rebuild_error=worst, pooled_by_ic=pooled, rows=entries)


# ── checks on finished runs ────────────────────────────────────────────────


def exact_ic_invariance(results: Path) -> dict:
    """Gross Sharpe of the exact-IC signal book across ICs, per curve: flat is dispersion-driven.

    Tails and middle are opposite books -- one holds the extremes inverted,
    the other right -- so their Sharpe ratios have opposite signs, and the
    horizon and rebalance rule set their scale. Pooled, the median sits near
    zero and a max-over-min ratio is undefined; each placement, horizon and
    rebalance rule is therefore its own curve, summarized by its range over
    its mean absolute Sharpe (per decision, as the rows annualize).
    """
    name = "exact-IC invariance diagnostic"
    rows = _rows(results, "e1_transfer_map")
    if rows is None:
        return _result(name, "SKIP", reason="no E1 rows")
    part = rows[rows["f_noise"].isin(["tails", "middle"]) & (rows["weighting"] == "signal")
                & (rows["selection"] == "all") & (rows["efq"] == 0.0)]
    if part.empty:
        return _result(name, "SKIP", reason="no exact-IC signal rows")
    curves = {}
    for (noise, horizon, rebalance), group in part.groupby(["f_noise", "eval_horizon",
                                                            "rebalance"]):
        curve = group.groupby("f_ic")["sharpe_mid"].median()
        curves[f"{noise}/{horizon}s/{rebalance}"] = {
            "sharpe_by_ic": {f"{k:g}": float(v) for k, v in curve.items()},
            "relative_range": float((curve.max() - curve.min()) / curve.abs().mean())}
    # Informational: a range of a few percent across a twenty-fold range of IC is the warning sign.
    ranges = [c["relative_range"] for c in curves.values()]
    return _result(name, "INFO", n_curves=len(curves), max_relative_range=max(ranges),
                   median_relative_range=float(np.median(ranges)), curves=curves)


#: The level of the placebo's tests: a 95% interval should exclude zero this often.
TEST_LEVEL = 0.05


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The Wilson score 95% interval of a proportion of ``n``."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def placebo(results: Path) -> dict:
    """E6's IC-0 seeds: daily gross Sharpe centred on zero, intervals of the right size.

    EACH DISTINCT BOOK COUNTS ONCE. Gross does not depend on EFQ, so a book's
    copies at each EFQ are one test of the placebo, not several, and collapse
    to the frictionless row. The configurations run on one seed share its
    forecast and their Sharpe ratios move together, so a seed is one cluster:
    the t statistic is taken over the seeds' mean Sharpe, with each
    configuration's own t across the seeds beside it. The size is the share
    of books whose 95% day-bootstrap interval excludes zero, with a Wilson
    interval that treats the books as independent and so is, if anything,
    too narrow; the check fails when even its lower end is above 5%.
    """
    name = "placebo (IC 0)"
    rows = _rows(results, "e6_sanity_checks")
    if rows is None:
        return _result(name, "SKIP", reason="no E6 rows")
    part = rows[(rows["forecast"] == "copula") & (rows["f_ic"] == 0.0)]
    if part.empty or "sharpe_daily_mid" not in part:
        return _result(name, "SKIP", reason="no IC-0 rows with daily Sharpe")
    book = [c for c in part.columns if (c.startswith("f_") or c in CONFIG_FIELDS) and c != "efq"]
    part = part.sort_values("efq", kind="stable").drop_duplicates(book)
    sharpe = part["sharpe_daily_mid"].to_numpy(dtype=np.float64)
    seeds = part.groupby("f_seed")["sharpe_daily_mid"].mean()
    t = float(seeds.mean() / (seeds.std(ddof=1) / np.sqrt(len(seeds))))
    t_iid = float(sharpe.mean() / (sharpe.std(ddof=1) / np.sqrt(len(sharpe))))
    varying = [c for c in book if not c.startswith("f_") and part[c].nunique(dropna=False) > 1]
    by_config = {}
    for key, group in part.groupby(varying) if varying else ():
        values = group["sharpe_daily_mid"].to_numpy(dtype=np.float64)
        label = "|".join(f"{c}={v}" for c, v in zip(varying, np.atleast_1d(key)))
        by_config[label] = float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))
    rejected = ~((part["sharpe_daily_mid_low"] <= 0) & (part["sharpe_daily_mid_high"] >= 0))
    size_low, size_high = _wilson(int(rejected.sum()), len(part))
    status = "PASS" if abs(t) < 3 and size_low <= TEST_LEVEL else "FAIL"
    return _result(name, status, mean_daily_sharpe=float(sharpe.mean()), t_clustered_by_seed=t,
                   t_iid=t_iid, size=float(rejected.mean()), size_low=size_low,
                   size_high=size_high, n_books=len(part), n_seeds=len(seeds),
                   t_by_configuration=by_config)


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
    """net == gross - cost in every saved series; break-even recomputed from it, with coverage.

    The identity holds by construction -- the harness computes net as gross
    less cost -- so what it checks is that each saved series is its row's.
    The break-even is recomputed for every row that has one (a cost model
    that charges, a positive EFQ, a finite break-even) except those that
    charge fees, whose break-even takes the fee out of the cost first, from a
    fee path the series does not save. A row says it charges fees by a
    finite ``mean_fee_bps``: in a run that also has fee rows the column is
    NaN, not None, on the rest, and testing for None skipped every one of
    them. How many rows each test covered is reported beside its error.
    """
    import pandas as pd

    worst_identity, worst_breakeven = 0.0, 0.0
    coverage = {"rows": 0, "with_series": 0, "breakeven_rows": 0, "breakeven_checked": 0,
                "breakeven_skipped_fees": 0, "breakeven_skipped_not_finite": 0}
    for run in sorted(p for p in results.iterdir() if (p / "series.npz").is_file()):
        rows = pd.read_parquet(run / "rows.parquet")
        series = np.load(run / "series.npz")
        for row in rows.itertuples(index=False):
            coverage["rows"] += 1
            if row.row_id not in series.files:
                continue
            coverage["with_series"] += 1
            gross, net, cost = series[row.row_id]
            worst_identity = max(worst_identity, float(np.nanmax(np.abs(net - (gross - cost)))))
            if row.cost_model == "mid" or not row.efq > 0:
                continue
            coverage["breakeven_rows"] += 1
            fee = _scalar(getattr(row, "mean_fee_bps", None))
            if not _missing(fee):
                coverage["breakeven_skipped_fees"] += 1
                continue
            breakeven = getattr(row, "breakeven_efq", np.nan)
            if not np.isfinite(breakeven):
                coverage["breakeven_skipped_not_finite"] += 1
                continue
            again = float(sf.breakeven_efq(gross, cost / row.efq))
            # A finite break-even the row's own series cannot reproduce is an error too.
            worst_breakeven = max(worst_breakeven, abs(again - breakeven)
                                  if np.isfinite(again) else float("inf"))
            coverage["breakeven_checked"] += 1
    if not coverage["with_series"]:
        return _result("cost identity", "SKIP", reason="no saved series")
    status = "PASS" if worst_identity < 1e-12 and worst_breakeven < 1e-9 else "FAIL"
    return _result("cost identity", status, worst_net_minus_gross_plus_cost=worst_identity,
                   worst_breakeven_error=worst_breakeven, **coverage)


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


#: The store's forward returns run between 60-second VWAPs, the first starting
#: strictly after the decision (:mod:`experiments.month_cache`), so the return
#: booked at a decision ends this long after the decision plus its horizon.
VWAP_SECONDS = 60


def _decision_spacing(decisions) -> float:
    """Seconds between consecutive decisions of one day, from ``"date|anchor"`` labels."""
    parts = np.char.partition(np.asarray(decisions).astype(str), "|")
    gaps = np.diff(parts[:, 2].astype(np.float64))[parts[1:, 0] == parts[:-1, 0]]
    return float(np.median(gaps[gaps > 0]))


def first_causal_lag(horizon: int, spacing: float, vwap_seconds: int = VWAP_SECONDS) -> int:
    """The fewest decisions a realized return must be shifted by to end before the decision.

    The return booked ``lag`` decisions earlier ends ``horizon + vwap_seconds``
    after that decision, so it is known at this one once ``lag * spacing``
    covers that: lag 2 for a 300-second horizon on 300-second decisions.
    """
    return max(math.ceil((horizon + vwap_seconds) / spacing), 1)


def lookahead_canaries() -> dict:
    """Perfect foresight is absurd; a shifted return that ends before the decision is far below it.

    The forecast is the realized forward return itself, shifted by:

    * 0 decisions, perfect foresight, which must be absurd: the forecast and
      the return it is scored against are aligned;
    * 1 decision, reported only. With a 300-second horizon that return ends
      at the VWAP minute after THIS decision, the window the scored return
      starts from, so it shares it with its target and is not causal;
    * :func:`first_causal_lag` decisions and one day (the same time
      yesterday), whose returns ended before the decision: causal forecasts,
      which must stay an order of magnitude below perfect foresight.

    WHAT IT CAN DETECT: a misalignment of whole decisions between forecast
    and outcome, a forecast that sees its own return or a return series
    shifted so that a causal forecast does, which turns one of the causal
    Sharpe ratios absurd. WHAT IT CANNOT: a leak of seconds (a feature that
    reads part of the post-decision VWAP minute, as lag 1 does), information
    that enters through the universe screen, the risk history or the cost
    quotes instead of the forecast, or a leak far weaker than foresight. Nor
    does it test that the lags decay: their Sharpe ratios are noise around
    zero (each one's iid t is reported), and so is any gap between them.
    """
    from experiments.run import _inference

    inputs = _month()
    realized = inputs["realized"]
    config = sf.BacktestConfig(weighting="signal", selection="all", cost_model="mid",
                               risk_model="diagonal", annualization="decisions")
    causal = first_causal_lag(int(realized.horizons[0]), _decision_spacing(realized.decisions))
    per_day = int(np.median(np.unique(inputs["days"], return_counts=True)[1]))
    shifts = {"same decision": 0, "lag 1": 1, f"lag {causal}": causal, "one day": per_day}
    sharpe, t_iid, daily = {}, {}, {}
    for name, lag in shifts.items():
        values = _shifted(realized.values, lag)
        forecast = sf.ForwardReturns(values, realized.decisions, realized.assets,
                                     realized.horizons)
        report = sf.run_backtest(forecast, realized, market=inputs["market"],
                                 history=inputs["history"], days=inputs["days"], config=config,
                                 with_series=True)
        gross = report.series.gross
        sharpe[name] = float(report.sharpe_mid)
        t_iid[name] = float(np.mean(gross) / np.std(gross, ddof=1) * np.sqrt(len(gross)))
        days = _inference(report.series)
        daily[name] = [days[f"sharpe_daily_mid{s}"] for s in ("", "_low", "_high")]
    worst_causal = max(abs(sharpe[f"lag {causal}"]), abs(sharpe["one day"]), 1.0)
    ok = sharpe["same decision"] > 10 * worst_causal
    return _result("look-ahead canaries", "PASS" if ok else "FAIL",
                   same_decision=sharpe["same decision"], lag_1=sharpe["lag 1"],
                   first_causal_lag=causal, first_causal=sharpe[f"lag {causal}"],
                   one_day=sharpe["one day"], t_iid=t_iid, sharpe=sharpe,
                   sharpe_daily_with_interval=daily)


#: Execution delays the latency check compares, in seconds: the v2 cache's quotes.
LATENCY_GRID = (0, 1, 5, 30)
#: How far daily gross Sharpe may move with the delay, relative to no delay. A
#: delay swaps only the execution quotes; gross moves only through the few
#: fills a later quote finds missing.
GROSS_TOLERANCE = 0.01


def _first_of_day(labels) -> np.ndarray:
    """Each day's first decision among ``"date|anchor"`` labels: the 09:30:00 open."""
    day = np.char.partition(np.asarray(labels).astype(str), "|")[:, 0]
    return np.r_[True, day[1:] != day[:-1]]


def _cost_and_sharpe(series, keep: np.ndarray) -> dict:
    """Mean cost per period (bps) and daily gross and net Sharpe over the periods kept."""
    from experiments.run import _sharpe

    day = np.char.partition(np.asarray(series.periods).astype(str), "|")[keep, 0]
    _, index = np.unique(day, return_inverse=True)
    daily = {k: np.bincount(index, weights=np.nan_to_num(getattr(series, k)[keep]))
             for k in ("gross", "net")}
    return {"mean_cost_bps": float(np.nanmean(series.cost[keep]) * 1e4),
            "sharpe_daily_net": float(_sharpe(daily["net"])),
            "sharpe_daily_gross": float(_sharpe(daily["gross"]))}


def latency_monotone() -> dict:
    """With a v2 cache: does a longer execution delay cost more? Gross must not move.

    The book is the IC-0.03 random oracle's signal book over the whole
    cross-section, crossing at EFQ 1 every five minutes, at each delay in
    :data:`LATENCY_GRID`. A delay changes only the quotes an order meets:
    the cost is the move of the mid over the delay plus EFQ times the spread
    then, and the returns still start at the decision. So daily gross Sharpe
    must agree across delays (a FAIL otherwise), and the cost carries the
    whole effect. The earlier version ranked per-decision net Sharpe, which
    on a negative mean grows more negative as its sd falls: it passed while
    the delay was cutting both cost and the daily loss.

    Mean cost per period and daily net Sharpe are reported with and without
    each day's first decision, the 09:30:00 open, whose quoted spread
    narrows by half in its first 30 seconds, with the open's share of the
    cost a 30-second delay saves. A delay that lowers cost or raises daily
    net Sharpe, with the open or without it, is a FLAG: waiting should not
    pay on an informed book, and where it does it is the quotes at the
    decision that need a look.
    """
    from experiments.month_cache_v2 import month_path
    from experiments.run import load_all_inputs, variant_key

    name = "latency (mean cost and daily net Sharpe at efq 1)"
    if not all(month_path(m).is_file() for m in ("2020-07", "2020-08")):
        return _result(name, "SKIP", reason="no v2 cache for Jul/Aug 2020")
    spec = {"eval_month": "2020-08", "history_month": "2020-07", "horizons": [300],
            "cache": "v2", "latency": list(LATENCY_GRID)}
    all_inputs = load_all_inputs(spec)
    config = sf.BacktestConfig(weighting="signal", selection="all", cost_model="cross", efq=1.0,
                               risk_model="diagonal", annualization="decisions")
    numbers = {"with the open": {}, "without the open": {}}
    open_cost, total_cost = {}, {}
    for latency in LATENCY_GRID:
        inputs = all_inputs[variant_key({"latency": latency})]
        forecast = sf.oracle_forecast(inputs["realized"], 0.03, noise="random", seed=0)
        series = sf.run_backtest(forecast, inputs["realized"], market=inputs["market"],
                                 history=inputs["history"], days=inputs["days"], config=config,
                                 with_series=True).series
        opening = _first_of_day(series.periods)
        for label, keep in (("with the open", np.ones_like(opening)),
                            ("without the open", ~opening)):
            numbers[label][latency] = _cost_and_sharpe(series, keep)
        open_cost[latency] = float(np.nansum(series.cost[opening]))
        total_cost[latency] = float(np.nansum(series.cost))
    gross = [numbers["with the open"][lat]["sharpe_daily_gross"] for lat in LATENCY_GRID]
    gross_change = max(abs(g / gross[0] - 1.0) for g in gross)
    helps = {}
    for label, by in numbers.items():
        cost = [by[lat]["mean_cost_bps"] for lat in LATENCY_GRID]
        sharpe = [by[lat]["sharpe_daily_net"] for lat in LATENCY_GRID]
        helps[label] = (any(b < a for a, b in pairwise(cost))
                        or any(b > a for a, b in pairwise(sharpe)))
    first, last = LATENCY_GRID[0], LATENCY_GRID[-1]
    saved = total_cost[first] - total_cost[last]
    status = ("FAIL" if gross_change > GROSS_TOLERANCE
              else "FLAG" if any(helps.values()) else "PASS")

    def span(key: str, digits: int) -> str:
        values = [numbers[label][lat][key] for label in numbers for lat in (first, last)]
        return (f"{values[0]:.{digits}f} -> {values[1]:.{digits}f} at {first} -> {last} s "
                f"(without the open {values[2]:.{digits}f} -> {values[3]:.{digits}f})")

    return _result(name, status, mean_cost_bps=span("mean_cost_bps", 2),
                   sharpe_daily_net=span("sharpe_daily_net", 0),
                   open_share_of_saving=(open_cost[first] - open_cost[last]) / saved
                   if saved else float("nan"),
                   max_gross_change=gross_change,
                   delay_helps_with_open=helps["with the open"],
                   delay_helps_without_open=helps["without the open"], by_latency=numbers)


TINY_SPEC = {
    "name": "sanity_determinism", "eval_month": "2020-08", "history_month": "2020-07",
    "horizons": [900],
    "forecasts": {"kind": "oracle", "ic": [0.03], "noise": ["random"], "seed": 0},
    "configs": {"base": {"cost_model": "cross", "risk_model": "shrunk_sample",
                         "universe": "all", "annualization": "decisions", "selection": "all"},
                "grid": {"weighting": ["signal", "hrp"], "efq": [0.7]}},
}


def _arrays_hash(*paths: Path) -> str:
    """A hash of every array the ``.npz`` files hold, key by key.

    The arrays rather than the files: an archive records when it was
    written, so two identical runs never give identical bytes.
    """
    digest = hashlib.sha256()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key in sorted(data.files):
                array = np.ascontiguousarray(data[key])
                digest.update(f"{path.name}|{key}|{array.dtype}|{array.shape}|".encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def determinism() -> dict:
    """The same task twice, and with one and four BLAS threads: identical rows and series.

    The rows are each configuration's summary; ``series.npz`` holds the
    per-period gross, net and cost that every daily statistic, the inference
    and the figures are computed from, and ``periods.npz`` their labels. A
    rerun that matched the rows could still differ there, so both are
    compared: the rows by their pandas hash, the series array by array.
    """
    import shutil

    import pandas as pd

    root = Path(__file__).resolve().parent.parent
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, threads in enumerate(("1", "1", "4")):
            spec = dict(TINY_SPEC, name=f"sanity_determinism_{k}")
            path = Path(tmp) / f"{spec['name']}.json"
            path.write_text(json.dumps(spec))
            env = {**os.environ, "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
                   "MKL_NUM_THREADS": threads}
            subprocess.run([sys.executable, "-m", "experiments.run", str(path), "--workers", "1"],
                           cwd=root, env=env, check=True, capture_output=True)
            out = RESULTS / spec["name"]
            frame = pd.read_parquet(out / "rows.parquet")
            with np.load(out / "series.npz", allow_pickle=False) as data:
                series = {key: data[key] for key in data.files}
            saved = [p for p in (out / "series.npz", out / "periods.npz") if p.is_file()]
            rows = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
            runs.append({"frame": frame, "series": series,
                         "rows_hash": hashlib.sha256(rows).hexdigest(),
                         "series_hash": _arrays_hash(*saved)})
            shutil.rmtree(out, ignore_errors=True)
    one, again, four = runs
    columns = ["sharpe_mid", "sharpe_net"]
    sharpe_gap = float(np.nanmax(np.abs(one["frame"][columns].to_numpy()
                                        - four["frame"][columns].to_numpy())))
    series_gap = max(float(np.nanmax(np.abs(values - four["series"][key])))
                     for key, values in one["series"].items())
    same = {f"{kind}_{name}": first[f"{kind}_hash"] == other[f"{kind}_hash"]
            for name, first, other in (("rerun", one, again), ("threads", one, four))
            for kind in ("rows", "series")}
    # Reruns must be identical. Thread count is reported, not required: a
    # multithreaded BLAS sums in a thread-dependent order, which is why
    # run.py pins one thread; a WARN says the pin is what reproduction needs.
    status = ("FAIL" if not (same["rows_rerun"] and same["series_rerun"])
              else "PASS" if same["rows_threads"] and same["series_threads"] else "WARN")
    return _result("determinism", status, rerun_rows_identical=same["rows_rerun"],
                   rerun_series_identical=same["series_rerun"],
                   one_vs_four_threads_rows_identical=same["rows_threads"],
                   one_vs_four_threads_series_identical=same["series_threads"],
                   max_sharpe_difference_threads=sharpe_gap,
                   max_series_difference_threads=series_gap)


#: Every check in report order, as (name, takes the results directory, loads a month).
CHECKS = (
    ("analytic_transfer_coefficients", False, False),
    ("fundamental_law", True, True),
    ("exact_ic_invariance", True, False),
    ("placebo", True, False),
    ("invariance_at_one", True, False),
    ("cost_identity", True, False),
    ("lookahead_canaries", False, True),
    ("latency_monotone", False, True),
    ("determinism", False, True),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(RESULTS),
                        help="directory holding finished runs (default experiments/results)")
    parser.add_argument("--out", default=None, help="report directory (default <results>/sanity)")
    parser.add_argument("--quick", action="store_true",
                        help="skip the checks that load a month and run backtests")
    parser.add_argument("--only", nargs="+", metavar="CHECK",
                        choices=[name for name, _, _ in CHECKS],
                        help="run just these checks: " + ", ".join(n for n, _, _ in CHECKS))
    args = parser.parse_args(argv)
    results = Path(args.results)
    report = []
    for name, reads_results, loads_month in CHECKS:
        if (args.only and name not in args.only) or (args.quick and loads_month):
            continue
        check = globals()[name]
        report.append(check(results) if reads_results else check())
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
