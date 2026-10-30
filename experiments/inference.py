"""Inference on a finished run: honest Sharpe ratios and grid-level multiple testing.

A run (``experiments/run.py``) writes one row per configuration and each
row's profit path. The rows' Sharpe ratios are point estimates, and a grid
of them invites two mistakes this script exists to prevent.

THE FIRST IS TREATING INTRADAY PERIODS AS INDEPENDENT. A per-decision Sharpe
annualized by ``sqrt(periods a year)`` assumes some twenty thousand
uncorrelated observations a year, but overlapping horizons, persistent
forecast errors and the intraday volatility pattern make neighbouring
decisions' P&L correlated. Three corrections are reported side by side:

* the P&L summed over each trading day, the primary unit (close to
  independent across days), with its own Sharpe ratio;
* Lo's (2002) autocorrelation-adjusted annualization of the per-period
  Sharpe, ``SR * q / sqrt(q + 2 sum_k (q - k) rho_k)`` with ``q`` periods a
  year and the autocorrelations ``rho_k`` estimated up to ``lags``;
* a Newey-West (Bartlett-kernel) standard error of the Sharpe ratio by the
  delta method on the first two moments (Lo 2002's GMM form), with
  ``lags = max(floor(4 (T/100)^(2/9)), h / spacing)`` for a book booked every
  decision at horizon ``h`` -- at least the overlap of its forward returns,
  as the research notes ask -- and the Newey-West rule alone otherwise.

THE SECOND IS READING THE BEST OF A GRID AS AN ESTIMATE. Within each group of
configurations that saw the same forecast -- by default every
(forecast, horizon, EFQ) cell, so the configurations differ only in how the
book is built -- the daily net P&L is tested jointly:

* the Hansen-Lunde-Nason (2011) Model Confidence Set at 90%: the
  configurations statistically indistinguishable from the best, on the mean
  daily loss (minus the net P&L);
* Hansen's (2005) SPA test of whether ANY configuration beats a benchmark
  (default: rank weights on the whole cross-section), and Romano-Wolf (2005)
  StepM listing which do, with the family-wise error held at 5%;
* the Ledoit-Wolf (2008) test of each configuration's daily Sharpe against
  the benchmark's: the library's HAC version and their studentized
  circular-block bootstrap, with a 95% interval for the difference;
* the deflated Sharpe ratio (Bailey and Lopez de Prado 2014) with the number
  of trials set to the EFFECTIVE number of independent configurations, the
  participation ratio ``(sum lambda)^2 / sum lambda^2`` of the eigenvalues of
  the configurations' return correlation matrix: twenty configurations that
  hold nearly the same book are closer to one trial than to twenty;
* the probability of backtest overfitting by combinatorially symmetric
  cross-validation over blocks of days.

MCS, SPA and StepM come from ``arch``, with its stationary bootstrap. Its
SPA (and StepM, which runs it) does not divide the statistic by the loss
differential's standard error -- the ``studentize`` flag only feeds the
recentring threshold -- so the differentials are studentized here before they
are passed in, with arch's own stationary-bootstrap variance. That is
exactly Hansen's statistic, whose standard errors are estimated once on the
full sample, and it stops the most volatile configuration from deciding the
test alone.

EVERYTHING IS SEEDED AND PAIRED. Every bootstrap draws its indices from one
seed (``--seed``), and a group's configurations are resampled on the same
days, so differences between them are judged on common shocks. The mean
block length is Politis and White's (2004, corrected 2009) estimate from
arch, the median over the group's columns, never below one day; ``--block``
fixes it instead. A month has about twenty days, so every interval here is
wide and approximate: read it as the scale of the uncertainty. Pooling
months (``walkforward.py``) is what tightens it.

Also reported, from the same series: the seed-to-seed spread of a forecast
run under several seeds beside the time-sampling standard error, the
rejection rate of zero-IC placebo rows at the 5% level (the tests' size), a
bootstrap interval for each break-even EFQ (a ratio of mean gross P&L to
mean cost, resampled on days), and each row's mean P&L by half hour of the
session, since five-minute P&L is far from stationary within the day.

WHICH DAY A PERIOD BELONGS TO. ``series.npz`` holds each row's per-period
gross, net and cost without labels. Runs since ``periods.npz`` was added save
the labels; for older runs they are rebuilt from the spec's evaluation
decisions with the library's own rule (every decision; the formation
decisions of a held book, every ``round(h / spacing)``-th of each day; or
the days of a daily book), and every rebuilt row is checked against the
daily Sharpe ratios ``run.py`` stored, which were computed from the true
labels.

    uv run --extra experiments python -m experiments.inference e3_horizon_overlap
    uv run --extra experiments python -m experiments.inference e3_horizon_overlap \\
        --benchmark weighting=equal,rebalance=every_decision

Writes ``results/<name>/inference.parquet`` (one row per configuration),
``inference.json`` (settings, checks and one entry per group),
``inference_daily.npz`` (every row's day sums, ``__days__`` plus a
``(3, n_days)`` gross/net/cost array per row id, NaN on a day a row has no
period: the format of ``run.py``'s ``daily.npz``) and
``inference_tod.parquet`` (mean P&L by half hour).
"""

from __future__ import annotations

import os

# One linear-algebra thread, as in run.py: results must not depend on it.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

import stable_finance as sf
from experiments.run import RESULTS
from stable_finance.evaluation.config import OPTIONAL_SETTINGS

ROOT = Path(__file__).resolve().parent.parent

#: Bootstrap resamples behind every test and interval, as in run.py.
REPS = 2000
#: The Model Confidence Set holds the best configuration with probability 1 - MCS_SIZE.
MCS_SIZE = 0.10
#: Family-wise size of SPA and StepM, and the level of every per-row test.
TEST_SIZE = 0.05
#: The default benchmark within each group: rank weights on the whole cross-section.
DEFAULT_BENCHMARK = "weighting=rank,selection=all"
#: Above this many configurations the MCS uses arch's ``max`` statistic: the
#: ``R`` statistic keeps a (reps, k, k) array of pairwise differences.
MCS_R_LIMIT = 100
#: Width of a time-of-day bucket, in seconds past the session's origin.
TOD_BUCKET = 1800
#: Trading days a year, the annualization of every daily Sharpe ratio.
DAYS_A_YEAR = 252.0
#: The most CSCV blocks the overfitting probability uses.
PBO_MAX_BLOCKS = 16

CONFIG_FIELDS = tuple(f.name for f in dataclasses.fields(sf.BacktestConfig))
PNL = ("mid", "net")  # the series' first two rows: gross (marked at the mid) and net


# ── loading ─────────────────────────────────────────────────────────────────

def run_dir(name: str) -> Path:
    """A run's directory: ``results/<name>``, or ``name`` itself if it is one."""
    path = Path(name)
    if (path / "rows.parquet").is_file():
        return path
    return RESULTS / name


def load_run(out: Path) -> tuple:
    """Rows, series by row id, and the spec the run was made with.

    The spec is the manifest's copy -- the one the rows were actually written
    with -- and only falls back to ``experiments/specs/<name>.json``.
    """
    import pandas as pd

    rows = pd.read_parquet(out / "rows.parquet")
    with np.load(out / "series.npz", allow_pickle=False) as data:
        series = {key: data[key] for key in data.files}
    spec = None
    if (out / "manifest.json").is_file():
        spec = json.loads((out / "manifest.json").read_text()).get("spec")
    if spec is None:
        path = ROOT / "experiments" / "specs" / f"{out.name}.json"
        spec = json.loads(path.read_text()) if path.is_file() else None
    return rows, series, spec


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _missing(value) -> bool:
    value = _scalar(value)
    return value is None or (isinstance(value, float) and math.isnan(value))


def config_label(row) -> str:
    """The configuration's label, rebuilt from the row's config columns.

    The same rule as :attr:`~stable_finance.BacktestConfig.label`: optional
    settings appear only when set. Used to name configurations in the output;
    benchmarks are matched on the columns themselves (:func:`match_rows`).
    """
    parts = []
    for name in CONFIG_FIELDS:
        if name not in row or (_missing(row[name]) and name in OPTIONAL_SETTINGS):
            continue
        if name == "holding" and row[name] == "rebalanced":
            continue
        parts.append(f"{name}={_scalar(row[name])}")
    return "|".join(parts)


# ── period labels ───────────────────────────────────────────────────────────

def saved_periods(out: Path) -> dict[str, np.ndarray]:
    """Each row's period labels from ``periods.npz``, when the run saved them."""
    path = out / "periods.npz"
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        sets = {key: data[key].astype(str) for key in data.files if key.startswith("set_")}
        return {row_id: sets[f"set_{index}"]
                for row_id, index in zip(data["row_id"].astype(str), data["set"])}


def _split(labels) -> tuple[np.ndarray, np.ndarray] | None:
    """``(day, seconds past the session origin)`` of ``"date|anchor"`` labels, else None."""
    parts = np.char.partition(np.asarray(labels).astype(str), "|")
    if len(parts) == 0 or not np.all(parts[:, 1] == "|"):
        return None
    try:
        return parts[:, 0], parts[:, 2].astype(np.float64)
    except ValueError:
        return None


def decision_spacing(decisions, days) -> float:
    """The median gap between consecutive decisions on one day, in seconds.

    The library's rule (``backtest._decision_seconds``), so a held book's
    formations are rebuilt exactly as it set them: epoch-second decisions or
    ``"date|anchor"`` labels, gaps across a day boundary and zero gaps
    ignored.
    """
    decisions = np.asarray(decisions)
    try:
        stamps = decisions.astype(np.float64)
    except (TypeError, ValueError):
        split = _split(decisions)
        if split is None:
            raise ValueError("decisions carry no times") from None
        stamps = split[1]
    _, day = np.unique(np.asarray(days).astype(str), return_inverse=True)
    gaps = np.diff(stamps)[day[1:] == day[:-1]]
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        raise ValueError("need two decisions on one day to measure their spacing")
    return float(np.median(gaps))


def position_in_day(days) -> np.ndarray:
    """Each decision's index within its own day, the day being the label run it sits in."""
    days = np.asarray(days).astype(str)
    index = np.arange(len(days))
    starts = np.r_[True, days[1:] != days[:-1]]
    return index - np.maximum.accumulate(np.where(starts, index, 0))


def hold_length(horizon: int, spacing: float) -> int:
    """Decisions a held book spans: ``max(round(h / spacing), 1)``, the library's rule."""
    return max(round(float(horizon) / spacing), 1)


def period_labels(decisions, days, *, rebalance: str, annualization: str,
                  horizon: int) -> np.ndarray:
    """The labels a configuration's series is indexed by, by the library's rule.

    Daily accounting folds a series into its days, in sorted order; a held
    book has one period per formation, every ``round(h / spacing)``-th
    decision counted from each day's first; everything else has one period
    per decision (``backtest._report``).
    """
    decisions = np.asarray(decisions).astype(str)
    days = np.asarray(days).astype(str)
    if annualization == "daily":
        return np.unique(days)
    if rebalance == "horizon":
        every = hold_length(horizon, decision_spacing(decisions, days))
        return decisions[position_in_day(days) % every == 0]
    return decisions


def rebuilt_periods(spec: dict, rows, series: dict, wanted: list[str]) -> tuple[dict, list]:
    """Labels for rows the run saved none for, rebuilt from the spec's decisions.

    The evaluation decisions come from ``run.load_inputs`` -- the inputs the
    rows were run on -- restricted to a forecast file's decisions exactly as
    ``run.run_task`` restricts them. A row whose rebuilt labels do not match
    its series length is left out and reported, never guessed at.
    """
    from experiments.run import load_inputs

    inputs = load_inputs(spec)
    decisions = inputs["realized"].decisions.astype(str)
    days = np.asarray(inputs["days"]).astype(str)
    covered: dict[str, np.ndarray] = {}
    out, failed = {}, []
    for row in rows[rows["row_id"].isin(wanted)].to_dict("records"):
        take = slice(None)
        if row.get("forecast") == "file":
            path = Path(row["f_path"])
            path = path if path.is_absolute() else ROOT / path
            if str(path) not in covered:
                if not path.is_file():
                    failed.append((row["row_id"], f"forecast file {path} is missing"))
                    continue
                with np.load(path, allow_pickle=False) as data:
                    covered[str(path)] = data["decisions"].astype(str)
            take = np.isin(decisions, covered[str(path)])
        try:
            labels = period_labels(decisions[take], days[take], rebalance=row["rebalance"],
                                   annualization=row["annualization"],
                                   horizon=int(row["eval_horizon"]))
        except ValueError as error:
            failed.append((row["row_id"], str(error)))
            continue
        if len(labels) != series[row["row_id"]].shape[1]:
            failed.append((row["row_id"], (f"{len(labels)} rebuilt labels for "
                                          f"{series[row['row_id']].shape[1]} periods")))
            continue
        out[row["row_id"]] = labels
    return out, failed


def daily_sums(values: np.ndarray, labels) -> tuple[np.ndarray, np.ndarray]:
    """A series summed over each day: ``(days, (3, n_days))``, NaN periods as zero.

    The same sums ``run._inference`` takes its daily Sharpe ratios from.
    """
    days = np.char.partition(np.asarray(labels).astype(str), "|")[:, 0]
    unique, index = np.unique(days, return_inverse=True)
    sums = np.stack([np.bincount(index, weights=np.nan_to_num(v), minlength=len(unique))
                     for v in values])
    return unique, sums


def saved_daily(out: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Day sums ``run.py`` saved itself (``"save_daily": true``), the last resort."""
    path = out / "daily.npz"
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        days = data["__days__"].astype(str)
        return {key: (days, data[key]) for key in data.files if key != "__days__"}


# ── one series ──────────────────────────────────────────────────────────────

def newey_west_lags(n: int) -> int:
    """The Newey-West (1994) rule ``floor(4 (n/100)^(2/9))``, as ``sf.sharpe_difference`` uses."""
    return math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))


def _bartlett(series: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of a mean-zero series."""
    n = len(series)
    variance = float(series @ series) / n
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * float(series[lag:] @ series[:-lag]) / n
    return variance


def autocorrelations(x: np.ndarray, lags: int) -> np.ndarray:
    """Sample autocorrelations at lags 1..lags (the biased, positive-definite estimator)."""
    d = x - x.mean()
    g0 = float(d @ d)
    if g0 <= 0:
        return np.full(lags, np.nan)
    return np.array([float(d[k:] @ d[:-k]) / g0 for k in range(1, min(lags, len(x) - 1) + 1)])


def lo_scale(rho: np.ndarray, q: float) -> float:
    """Lo's (2002) factor turning a per-period Sharpe into an annual one.

    ``q / sqrt(q + 2 sum_{k=1}^{L} (q - k) rho_k)``, the autocorrelations past
    ``L`` taken as zero. With no autocorrelation it is ``sqrt(q)``, the naive
    factor; positive autocorrelation shrinks it, negative grows it. NaN when
    the estimated long-run variance is not positive.
    """
    k = np.arange(1, len(rho) + 1, dtype=np.float64)
    keep = k < q
    denominator = q + 2.0 * float(np.sum((q - k[keep]) * rho[keep]))
    return q / math.sqrt(denominator) if denominator > 0 else np.nan


def sharpe_hac(x: np.ndarray, lags: int) -> tuple[float, float]:
    """The per-period Sharpe (population moments) and its Newey-West standard error.

    Lo (2002), the delta method on ``(mean, mean of squares)``: with
    ``sigma^2 = m2 - mu^2`` the gradient of ``mu / sigma`` is
    ``(m2 / sigma^3, -mu / (2 sigma^3))``, and the variance is the long-run
    variance of the gradient-weighted moment deviations over ``T``.
    """
    n = len(x)
    mu, m2 = float(x.mean()), float(np.mean(x ** 2))
    variance = m2 - mu ** 2
    if n < 3 or variance <= 0:
        return np.nan, np.nan
    sd = math.sqrt(variance)
    projected = m2 / sd ** 3 * (x - mu) - mu / (2.0 * sd ** 3) * (x ** 2 - m2)
    return mu / sd, math.sqrt(max(_bartlett(projected, lags), 0.0) / n)


def _sharpe(x: np.ndarray) -> float:
    """Per-period ``ddof=1`` Sharpe, as ``sf.sharpe_ratio`` at one period a year."""
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else np.nan


def series_statistics(x: np.ndarray, *, q: float, lags: int) -> dict:
    """Naive and Lo-adjusted annual Sharpe of a per-period series, with HAC and iid errors."""
    x = x[np.isfinite(x)]
    sr = _sharpe(x)
    sr0, se = sharpe_hac(x, lags)
    scale = lo_scale(autocorrelations(x, lags), q) if np.isfinite(sr) else np.nan
    iid = math.sqrt((1.0 + sr0 ** 2 / 2.0) / len(x)) if np.isfinite(sr0) else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sharpe": sr * math.sqrt(q), "sharpe_lo": sr * scale,
                "lo_ratio": scale / math.sqrt(q), "sharpe_se_hac": se * math.sqrt(q),
                "sharpe_se_iid": iid * math.sqrt(q),
                "sharpe_t_hac": sr0 / se if se > 0 else np.nan,
                "sharpe_t_iid": sr0 / iid if iid > 0 else np.nan}


def daily_statistics(x: np.ndarray) -> dict:
    """Annual Sharpe of daily P&L, its HAC standard error and probabilistic Sharpe."""
    x = x[np.isfinite(x)]
    lags = newey_west_lags(len(x))
    sr = _sharpe(x)
    sr0, se = sharpe_hac(x, lags)
    scale = math.sqrt(DAYS_A_YEAR)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sharpe_daily": sr * scale, "sharpe_daily_se_hac": se * scale,
                "sharpe_daily_t_hac": sr0 / se if se > 0 else np.nan,
                "psr_daily": sf.probabilistic_sharpe(x)}


def row_lags(row, n_periods: int, spacing: float | None, fixed: int | None) -> int:
    """HAC lags for a row: the Newey-West rule, raised to the overlap of its returns.

    A book booked every decision at horizon ``h`` earns overlapping forward
    returns, so its P&L is correlated over ``h / spacing`` decisions by
    construction; a held book or a daily series is not.
    """
    if fixed is not None:
        return int(fixed)
    lags = newey_west_lags(n_periods)
    if (spacing and row["rebalance"] != "horizon" and row["annualization"] != "daily"):
        lags = max(lags, math.ceil(float(row["eval_horizon"]) / spacing))
    return min(lags, max(n_periods - 2, 0))


# ── bootstrap helpers ───────────────────────────────────────────────────────

def block_lengths(matrix: np.ndarray, fixed: int | None) -> tuple[int, int, dict]:
    """Stationary and circular mean block lengths for a group, in days.

    Politis-White via arch, column by column; the median over columns,
    rounded and never below one day nor above half the sample. A flat column
    has no estimate and is skipped; with none left the block is one day.
    """
    n = matrix.shape[0]
    if fixed is not None:
        return int(fixed), int(fixed), {"source": "fixed"}
    from arch.bootstrap import optimal_block_length

    live = matrix[:, np.nanstd(matrix, axis=0) > 0]
    raw = {"stationary": np.nan, "circular": np.nan}
    if live.shape[1] and n >= 8:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            estimate = optimal_block_length(live)
        # arch names the columns "stationary"/"circular" (older releases b_sb/b_cb).
        sb, cb = (("stationary", "circular") if "stationary" in estimate
                  else ("b_sb", "b_cb"))
        raw = {"stationary": float(np.nanmedian(estimate[sb].to_numpy())),
               "circular": float(np.nanmedian(estimate[cb].to_numpy()))}

    def clip(value: float) -> int:
        if not np.isfinite(value):
            return 1
        return int(min(max(round(value), 1), max(n // 2, 1)))

    return (clip(raw["stationary"]), clip(raw["circular"]),
            {"source": "politis_white", "pw_stationary": raw["stationary"],
             "pw_circular": raw["circular"]})


def stationary_indices(n: int, block: int, reps: int, seed: int) -> np.ndarray:
    """``(reps, n)`` day indices from arch's stationary bootstrap, seeded."""
    from arch.bootstrap import StationaryBootstrap

    bootstrap = StationaryBootstrap(block, np.arange(n), seed=seed)
    return np.stack([data[0][0] for data in bootstrap.bootstrap(reps)]).astype(np.intp)


def stationary_variance(d: np.ndarray, block: int) -> np.ndarray:
    """Columnwise variance of ``sqrt(T) * mean`` under the stationary bootstrap.

    Politis and Romano's kernel with ``p = 1 / block``, the estimator arch's
    SPA uses for its recentring threshold, so studentizing with it changes
    the statistic and nothing else.
    """
    t = len(d)
    p = 1.0 / block
    demeaned = d - d.mean(axis=0)
    variance = np.sum(demeaned ** 2, axis=0) / t
    for i in range(1, t):
        kappa = (1.0 - i / t) * (1.0 - p) ** i + (i / t) * (1.0 - p) ** (t - i)
        variance += 2.0 * kappa * np.sum(demeaned[: t - i] * demeaned[i:], axis=0) / t
    return variance


# ── a group of configurations ───────────────────────────────────────────────

def match_rows(frame, selector: str) -> np.ndarray:
    """Rows matching ``key=value`` pairs separated by commas or ``|``.

    A full configuration label is one such list. Values compare as numbers
    when both sides are numeric (``efq=0.7`` matches 0.7 whatever its
    type), ``None``/``null`` matches a missing value, and anything else
    compares as text.
    """
    pairs = [item.split("=", 1) for item in selector.replace("|", ",").split(",") if item]
    keep = np.ones(len(frame), dtype=bool)
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"benchmark terms are key=value; got {'='.join(pair)!r}")
        key, wanted = pair[0].strip(), pair[1].strip()
        if key not in frame:
            return np.zeros(len(frame), dtype=bool)
        try:
            number = float(wanted)
        except ValueError:
            number = None
        hits = []
        for value in frame[key].tolist():
            value = _scalar(value)
            if wanted in ("None", "null"):
                hits.append(_missing(value))
            elif number is not None and isinstance(value, (int, float)) and not _missing(value):
                hits.append(float(value) == number)
            else:
                hits.append(str(value) == wanted)
        keep &= np.asarray(hits, dtype=bool)
    return keep


def effective_trials(matrix: np.ndarray) -> tuple[float, float]:
    """Participation ratio of the correlation eigenvalues, and the top eigenvalue's share.

    ``(sum lambda)^2 / sum lambda^2``: the number of configurations if they
    were uncorrelated, one if they were the same strategy, and in between the
    number of equally weighted independent directions the grid explores.
    Flat columns carry no information and are left out.
    """
    live = matrix[:, np.std(matrix, axis=0) > 0]
    if live.shape[1] < 2:
        return float(live.shape[1]), 1.0
    eigenvalues = np.clip(np.linalg.eigvalsh(np.corrcoef(live, rowvar=False)), 0.0, None)
    return (float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2)),
            float(eigenvalues.max() / eigenvalues.sum()))


def block_matrix(row_ids: list[str], horizons: dict, labels: dict, values: dict,
                 grid: dict, spacing: float) -> np.ndarray | None:
    """The group's net P&L on its finest common grid of periods, or None.

    A group mixes books booked every decision with books held for the
    horizon, so their series have different lengths; summing every series
    into blocks of the longest hold in the group (counted from each day's
    first decision, the library's formations) puts them on one grid with
    many more observations than days, which is what a correlation matrix of
    hundreds of configurations needs. Only for rows labelled by decision.
    """
    every = max(hold_length(horizons[r], spacing) if horizons[r] else 1 for r in row_ids)
    out = []
    for row_id in row_ids:
        cells = np.array([grid.get(label, -1) for label in labels[row_id]])
        if np.any(cells < 0):
            return None
        out.append(cells)
    day_of, position = grid["__day__"], grid["__position__"]
    offsets = {}
    n_cells = 0
    for day, count in grid["__per_day__"]:
        offsets[day] = n_cells
        n_cells += -(-count // every)
    matrix = np.zeros((n_cells, len(row_ids)))
    for j, (row_id, cells) in enumerate(zip(row_ids, out)):
        index = np.array([offsets[day_of[c]] + position[c] // every for c in cells])
        matrix[:, j] = np.bincount(index, weights=np.nan_to_num(values[row_id]),
                                   minlength=n_cells)
    used = np.any(matrix != 0, axis=1)
    return matrix[used]


def decision_grid(labels: dict) -> tuple[dict, float | None]:
    """Every decision label in the run, with its day and position within the day.

    The union of the rows' labels is the run's decision list as long as one
    row is booked every decision; if every row is held, it is the formation
    list, on which each hold is one step, and the grid is still right.
    """
    union = sorted({label for array in labels.values() if _split(array[:1]) is not None
                    for label in array.tolist()},
                   key=lambda s: (s.partition("|")[0], float(s.partition("|")[2])))
    if not union:
        return {}, None
    days, _ = _split(union)
    position = position_in_day(days)
    grid = {label: i for i, label in enumerate(union)}
    per_day = []
    for day in days:
        if per_day and per_day[-1][0] == day:
            per_day[-1][1] += 1
        else:
            per_day.append([day, 1])
    grid.update(__day__=days, __position__=position, __per_day__=per_day)
    return grid, decision_spacing(np.asarray(union), days)


def studentized_differences(benchmark: np.ndarray, models: np.ndarray,
                            block: int) -> np.ndarray:
    """``(L_benchmark - L_model) / omega``, the loss differentials Hansen's SPA tests.

    A model identical to the benchmark has no differential and no variance;
    its column is zero, so it can never be called better.
    """
    differences = benchmark[:, None] - models
    omega = np.sqrt(np.maximum(stationary_variance(differences, block), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(omega > 1e-300, differences / omega, 0.0)


def sharpe_difference_bootstrap(matrix: np.ndarray, bench: int, *, block: int, reps: int,
                                seed: int) -> list[dict]:
    """Ledoit-Wolf (2008) tests of each column's Sharpe against the benchmark column's.

    The HAC test is the library's :func:`~stable_finance.sharpe_difference`.
    The bootstrap is their Section 3.2: circular blocks of ``block`` days
    (seeded, the same draws for every column, so the comparison is paired),
    each resample's difference studentized with their block standard error
    ``(1/l) sum_j zeta_j^2``, ``zeta_j`` the block sums of the projected
    moment deviations over ``sqrt(block)``. The two-sided p-value is the share
    of ``|delta* - delta| / s*`` at or above ``|delta| / s`` and the 95%
    interval ``delta -/+ q95 * s``. Per day; :func:`group_statistics`
    annualizes.
    """
    n, k = matrix.shape
    length = n // block
    span = length * block
    starts = np.random.default_rng(seed).integers(0, n, size=(reps, length))
    index = ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, span)
    x = matrix[index]  # (reps, span, k)
    mu, m2 = x.mean(axis=1), np.mean(x ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = m2 - mu ** 2
        sr = mu / np.sqrt(var)
        g_mean, g_square = m2 / var ** 1.5, -mu / (2.0 * var ** 1.5)
    out = []
    for j in range(k):
        if j == bench:
            out.append({})
            continue
        try:
            hac = sf.sharpe_difference(matrix[:, j], matrix[:, bench])
        except ValueError:
            out.append({})
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            projected = (g_mean[:, None, j] * (x[:, :, j] - mu[:, None, j])
                         - g_mean[:, None, bench] * (x[:, :, bench] - mu[:, None, bench])
                         + g_square[:, None, j] * (x[:, :, j] ** 2 - m2[:, None, j])
                         - g_square[:, None, bench] * (x[:, :, bench] ** 2 - m2[:, None, bench]))
            zeta = projected.reshape(reps, length, block).sum(axis=2) / math.sqrt(block)
            s_star = np.sqrt(np.mean(zeta ** 2, axis=1) / span)
            stat = np.abs(sr[:, j] - sr[:, bench] - hac.difference) / s_star
        stat = stat[np.isfinite(stat)]
        if hac.standard_error > 0 and len(stat):
            observed = abs(hac.difference) / hac.standard_error
            p_boot = float((1 + np.sum(stat >= observed)) / (len(stat) + 1))
            half = float(np.quantile(stat, 1.0 - TEST_SIZE)) * hac.standard_error
        else:
            p_boot, half = np.nan, np.nan
        out.append({"difference": hac.difference, "se_hac": hac.standard_error,
                    "p_hac": hac.p_value, "p_boot": p_boot,
                    "low": hac.difference - half, "high": hac.difference + half})
    return out


def breakeven_interval(gross: np.ndarray, unit_cost: np.ndarray,
                       index: np.ndarray) -> tuple[float, float]:
    """95% bootstrap interval of a break-even EFQ, resampling days.

    The break-even is a ratio, mean gross over mean cost at EFQ 1, so it is
    resampled as one: both sums over the same drawn days. A resample with no
    positive gross has a break-even of zero (no execution rescues it), one
    that paid nothing an infinite one; the percentiles take the nearest draw
    rather than interpolating across an infinity.
    """
    g, c = gross[index].sum(axis=1), unit_cost[index].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(g <= 0, 0.0, np.where(c <= 0, np.inf, g / c))
    low, high = np.quantile(ratio, [TEST_SIZE / 2, 1 - TEST_SIZE / 2], method="nearest")
    return float(low), float(high)


def pbo_blocks(n_days: int, wanted: int | None) -> int:
    """CSCV blocks: as asked, else the most (even, at most 16) that leave two days a block."""
    if wanted:
        return int(wanted)
    blocks = min(PBO_MAX_BLOCKS, n_days // 2)
    return blocks - blocks % 2


def group_statistics(group, daily: dict, *, benchmark: str, which: int,
                     reps: int, seed: int, block: int | None, pbo_n: int | None,
                     block_data: tuple | None) -> tuple[dict, dict]:
    """Joint tests on one group's daily P&L: MCS, SPA, StepM, Sharpe differences, DSR, PBO.

    Returns the group's summary and per-row results by row id. Only days on
    which every configuration in the group has a period are used, so all are
    judged on the same days.
    """
    import pandas as pd
    from arch.bootstrap import MCS, SPA, StepM

    ids = [r for r in group["row_id"] if r in daily]
    summary: dict = {"n_configs": len(ids)}
    per_row: dict = {r: {} for r in ids}
    if len(ids) < 2:
        summary["skipped"] = "fewer than two configurations with daily P&L"
        return summary, per_row
    full = np.stack([daily[r][which] for r in ids], axis=1)
    common = np.all(np.isfinite(full), axis=1)
    pnl = full[common]
    n = len(pnl)
    summary["n_days"] = int(n)
    if n < 4:
        summary["skipped"] = f"only {n} common days"
        return summary, per_row
    block_sb, block_cb, block_info = block_lengths(pnl, block)
    summary["block"] = {"stationary": block_sb, "circular": block_cb, **block_info}
    losses = pd.DataFrame(-pnl, columns=ids)

    # Model Confidence Set: the configurations indistinguishable from the best.
    # Configurations with the same daily P&L (a setting that changed nothing)
    # are one model to it, since its pairwise statistic divides by their
    # zero-variance difference; each duplicate takes its representative's
    # p-value. A group the statistic still cannot handle falls back to the
    # max statistic, and failing that is reported as skipped.
    _, first, inverse = np.unique(pnl, axis=1, return_index=True, return_inverse=True)
    distinct = [ids[i] for i in first]
    method = "R" if len(distinct) <= MCS_R_LIMIT else "max"
    pvalues = None
    if len(distinct) >= 2:
        for attempt in dict.fromkeys((method, "max")):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mcs = MCS(losses[distinct], size=MCS_SIZE, reps=reps, block_size=block_sb,
                              method=attempt, bootstrap="stationary", seed=seed)
                    mcs.compute()
                pvalues, method = mcs.pvalues["Pvalue"], attempt
                break
            except (IndexError, ValueError, np.linalg.LinAlgError) as error:
                summary.setdefault("mcs_errors", []).append(f"{attempt}: {error}")
    if len(distinct) < 2:
        summary["mcs"] = {"size": MCS_SIZE, "skipped": "every configuration has the same P&L",
                          "included": list(ids)}
        for r in ids:
            per_row[r]["mcs_pvalue"], per_row[r]["in_mcs"] = 1.0, True
    elif pvalues is None:
        summary["mcs"] = {"size": MCS_SIZE, "skipped": "the MCS statistic failed on this group"}
    else:
        for j, r in enumerate(ids):
            value = float(pvalues.loc[distinct[int(np.ravel(inverse)[j])]])
            per_row[r]["mcs_pvalue"] = value
            per_row[r]["in_mcs"] = bool(value > MCS_SIZE)
        summary["mcs"] = {"size": MCS_SIZE, "method": method, "distinct_models": len(distinct),
                          "included": [r for r in ids if per_row[r]["in_mcs"]]}

    # SPA / StepM / Sharpe differences against the benchmark.
    hits = np.flatnonzero(match_rows(group.set_index("row_id").loc[ids].reset_index(), benchmark))
    if len(hits) == 0:
        summary["benchmark"] = {"selector": benchmark, "row_id": None,
                                "note": "no configuration in this group matches"}
    else:
        bench = int(hits[0])
        bench_id = ids[bench]
        summary["benchmark"] = {"selector": benchmark, "row_id": bench_id,
                                "ties": [ids[h] for h in hits[1:]]}
        for r in ids:
            per_row[r]["is_benchmark"] = r == bench_id
        others = [r for r in ids if r != bench_id]
        z = studentized_differences(losses[bench_id].to_numpy(), losses[others].to_numpy(),
                                    block_sb)
        models = pd.DataFrame(-z, columns=others)
        spa = SPA(np.zeros(n), models, block_size=block_sb, reps=reps, bootstrap="stationary",
                  seed=seed)
        spa.compute()
        stepm = StepM(np.zeros(n), models, size=TEST_SIZE, block_size=block_sb, reps=reps,
                      bootstrap="stationary", seed=seed)
        stepm.compute()
        superior = set(stepm.superior_models)
        for r in others:
            per_row[r]["stepm_superior"] = r in superior
        summary["spa"] = {key: float(v) for key, v in spa.pvalues.items()}
        summary["stepm"] = {"size": TEST_SIZE, "superior": sorted(superior)}
        scale = math.sqrt(DAYS_A_YEAR)
        tests = sharpe_difference_bootstrap(pnl, bench, block=block_cb, reps=reps, seed=seed)
        for r, test in zip(ids, tests):
            if test:
                per_row[r].update({
                    "sharpe_diff_daily": test["difference"] * scale,
                    "sharpe_diff_se_hac": test["se_hac"] * scale,
                    "sharpe_diff_p_hac": test["p_hac"], "sharpe_diff_p_boot": test["p_boot"],
                    "sharpe_diff_low": test["low"] * scale,
                    "sharpe_diff_high": test["high"] * scale})

    # Deflated Sharpe with the effective number of trials.
    sharpes = np.array([_sharpe(pnl[:, j]) for j in range(pnl.shape[1])])
    finite = sharpes[np.isfinite(sharpes)]
    variance = float(finite.var(ddof=1)) if len(finite) >= 2 else 0.0
    n_eff_day, top_day = effective_trials(pnl)
    n_eff_block = top_block = None
    if block_data is not None:
        n_eff_block, top_block = effective_trials(block_data)
    n_eff = n_eff_block if n_eff_block is not None else n_eff_day
    trials = max(1, round(n_eff))
    hurdle = sf.expected_max_sharpe(trials, variance)
    hurdle_all = sf.expected_max_sharpe(len(ids), variance)
    for j, r in enumerate(ids):
        per_row[r]["dsr_daily"] = sf.probabilistic_sharpe(pnl[:, j], benchmark=hurdle)
        per_row[r]["dsr_daily_all_trials"] = sf.probabilistic_sharpe(pnl[:, j],
                                                                     benchmark=hurdle_all)
    best = int(np.nanargmax(np.where(np.isfinite(sharpes), sharpes, -np.inf)))
    summary["dsr"] = {
        "n_eff_day": n_eff_day, "top_eigen_share_day": top_day,
        "n_eff_block": n_eff_block, "top_eigen_share_block": top_block,
        "n_block_periods": None if block_data is None else int(block_data.shape[0]),
        "n_eff_used": n_eff, "n_trials": trials, "sharpe_var_daily": variance,
        "hurdle_daily": hurdle, "hurdle_annual": hurdle * math.sqrt(DAYS_A_YEAR),
        "best_row_id": ids[best], "best_sharpe_daily": sharpes[best] * math.sqrt(DAYS_A_YEAR),
        "best_dsr": per_row[ids[best]]["dsr_daily"],
        "best_dsr_all_trials": per_row[ids[best]]["dsr_daily_all_trials"]}

    # Probability of backtest overfitting over blocks of days.
    blocks = pbo_blocks(n, pbo_n)
    if blocks >= 2:
        try:
            result = sf.pbo(pnl, n_blocks=blocks)
            summary["pbo"] = {"probability": result.probability,
                              "probability_of_loss": result.probability_of_loss,
                              "n_blocks": blocks, "block_days": result.block_length,
                              "n_splits": result.n_combinations}
        except ValueError as error:
            summary["pbo"] = {"skipped": str(error)}
    return summary, per_row


# ── extras over the whole run ───────────────────────────────────────────────

def seed_dispersion(table) -> list[dict]:
    """The spread of each statistic across seeds, beside its time-sampling error.

    A forecast run under several seeds is one configuration seen through
    several noise draws; the seed-to-seed standard deviation says how much a
    single-seed result could mislead, and comparing it with the mean HAC
    standard error says whether the noise draw or the month is the larger
    source of uncertainty.
    """
    if "f_seed" not in table or table["f_seed"].nunique() < 2:
        return []
    keys = [c for c in table.columns if c.startswith("f_") and c != "f_seed"]
    keys += ["eval_horizon", "label"]
    out = []
    for key, part in table.groupby(keys, dropna=False, sort=False):
        if len(part) < 2:
            continue
        entry = {k: _scalar(v) for k, v in zip(keys, key)}
        entry["n_seeds"] = len(part)
        for column in ("sharpe_daily_net", "sharpe_net", "sharpe_lo_net", "sharpe_daily_mid"):
            values = part[column].to_numpy(dtype=float)
            entry[f"{column}_mean"] = float(np.nanmean(values))
            entry[f"{column}_sd"] = float(np.nanstd(values, ddof=1))
        entry["sharpe_daily_se_hac_net_mean"] = float(np.nanmean(part["sharpe_daily_se_hac_net"]))
        entry["sharpe_se_hac_net_mean"] = float(np.nanmean(part["sharpe_se_hac_net"]))
        out.append({k: (None if _missing(v) else v) for k, v in entry.items()})
    return out


def size_check(table) -> dict:
    """How often zero-IC placebo rows reject a zero gross Sharpe at the 5% level.

    The gross (mid) P&L of an IC-0 forecast has mean zero, so a test of the
    right size rejects about 5% of the time; the iid test, which ignores the
    serial correlation, shows how far that assumption over-rejects. Net P&L
    is left out: it carries the cost, so its mean is not zero.
    """
    if "f_ic" not in table:
        return {}
    placebo = table[table["f_ic"].astype(float) == 0.0]
    if placebo.empty:
        return {}
    critical = 1.959963984540054
    out = {"n_rows": len(placebo), "level": TEST_SIZE}
    for column in ("sharpe_t_hac_mid", "sharpe_t_iid_mid", "sharpe_daily_t_hac_mid"):
        values = placebo[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        out[f"reject_{column}"] = float(np.mean(np.abs(values) > critical)) if len(values) else None
    return out


def time_of_day(row_id: str, values: np.ndarray, labels) -> list[dict]:
    """Mean P&L (bps) by half hour past the session origin, at each period's label."""
    split = _split(labels)
    if split is None:
        return []
    bucket = (split[1] // TOD_BUCKET).astype(np.int64)
    out = []
    for b in np.unique(bucket):
        take = bucket == b
        gross, net, cost = (np.nan_to_num(v[take]) for v in values)
        out.append({"row_id": row_id, "bucket_s": int(b * TOD_BUCKET), "n": int(take.sum()),
                    "mean_mid_bps": float(gross.mean() * 1e4),
                    "mean_net_bps": float(net.mean() * 1e4),
                    "mean_cost_bps": float(cost.mean() * 1e4),
                    "sd_net_bps": float(net.std(ddof=1) * 1e4) if take.sum() > 1 else np.nan})
    return out


# ── the analysis ────────────────────────────────────────────────────────────

def default_group(rows) -> list[str]:
    """(forecast, horizon, EFQ): every forecast parameter, the horizon and the EFQ."""
    keys = ["forecast", *[c for c in rows.columns if c.startswith("f_")], "eval_horizon", "efq"]
    return [k for k in keys if k in rows]


def _group_name(keys: list[str], values) -> str:
    return "|".join(f"{k}={_scalar(v)}" for k, v in zip(keys, values))


def analyze(name: str, *, benchmark: str = DEFAULT_BENCHMARK, group_by: list[str] | None = None,
            pnl: str = "net", reps: int = REPS, seed: int = 0, block: int | None = None,
            lags: int | None = None, pbo_n: int | None = None) -> dict:
    """Run every statistic on ``results/<name>`` and write the outputs; returns the JSON."""
    import pandas as pd

    out = run_dir(name)
    rows, series, spec = load_run(out)
    rows = rows.copy()
    rows["label"] = [config_label(r) for r in rows.to_dict("records")]
    ids = rows["row_id"].astype(str).tolist()

    labels = saved_periods(out)
    source = {r: "periods.npz" for r in ids if r in labels}
    missing = [r for r in ids if r not in labels]
    failed: list = []
    if missing and spec is not None:
        try:
            rebuilt, failed = rebuilt_periods(spec, rows, series, missing)
        except (OSError, KeyError, ValueError) as error:  # no month cache here, say
            rebuilt, failed = {}, [(r, f"cannot rebuild: {error}") for r in missing]
        labels.update(rebuilt)
        source.update({r: "rebuilt" for r in rebuilt})
    fallback = saved_daily(out)

    daily_by_row: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in ids:
        if r in labels:
            daily_by_row[r] = daily_sums(series[r], labels[r])
        elif r in fallback:
            daily_by_row[r] = fallback[r]
            source[r] = "daily.npz"
    all_days = np.asarray(sorted({d for days, _ in daily_by_row.values() for d in days.tolist()}))
    column = {d: i for i, d in enumerate(all_days.tolist())}
    daily: dict[str, np.ndarray] = {}
    for r, (days, sums) in daily_by_row.items():
        full = np.full((3, len(all_days)), np.nan)
        full[:, [column[d] for d in days.tolist()]] = sums
        daily[r] = full

    grid, spacing = decision_grid(labels)
    if spacing is None:
        spacing = float(spec.get("decision_spacing", 300)) if spec else None

    # Per-row statistics.
    records, tod = [], []
    stored_check = []
    for row in rows.to_dict("records"):
        r = row["row_id"]
        values = series[r]
        q = float(row["periods_per_year"])
        record = {"row_id": r, "label_source": source.get(r)}
        n_lags = row_lags(row, values.shape[1], spacing, lags)
        record["lags"] = n_lags
        for i, kind in enumerate(PNL):
            stats = series_statistics(values[i], q=q, lags=n_lags)
            record.update({f"{key}_{kind}": v for key, v in stats.items()})
            if r in daily:
                stats = daily_statistics(daily[r][i][np.isfinite(daily[r][i])])
                record.update({f"{key}_{kind}": v for key, v in stats.items()})
                stored = row.get(f"sharpe_daily_{kind}")
                if stored is not None and np.isfinite(stored):
                    stored_check.append(abs(record[f"sharpe_daily_{kind}"] - stored))
        record["n_days"] = int(np.isfinite(daily[r][1]).sum()) if r in daily else 0
        if r in labels:
            tod += time_of_day(r, values, labels[r])
        records.append(record)
    stats_frame = pd.DataFrame(records)

    # Groups.
    group_by = group_by or default_group(rows)
    which = PNL.index(pnl)
    horizons = {r: (int(h) if reb == "horizon" else 0)
                for r, h, reb in zip(rows["row_id"], rows["eval_horizon"], rows["rebalance"])}
    groups, per_row = [], {}
    keys = group_by if group_by != ["all"] else []
    grouped = rows.groupby(keys, dropna=False, sort=False) if keys else [((), rows)]
    for key, members in grouped:
        key = key if isinstance(key, tuple) else (key,)
        member_ids = [r for r in members["row_id"] if r in daily]
        block_data = None
        if grid and all(r in labels for r in member_ids) and member_ids:
            block_data = block_matrix(member_ids, horizons, labels,
                                      {r: series[r][which] for r in member_ids}, grid, spacing)
        summary, results = group_statistics(
            members, daily, benchmark=benchmark, which=which, reps=reps, seed=seed,
            block=block, pbo_n=pbo_n, block_data=block_data)
        gid = _group_name(keys, key) if keys else "all"
        summary = {"group": gid, "keys": {k: _scalar(v) for k, v in zip(keys, key)}, **summary}
        groups.append(summary)
        for r, result in results.items():
            per_row[r] = {"group": gid, **result}

        # Break-even EFQ intervals: resampled on the group's days and block.
        if "block" in summary:
            index = stationary_indices(summary["n_days"], summary["block"]["stationary"],
                                       reps, seed)
            common = np.all(np.isfinite(np.stack([daily[m][which] for m in member_ids])),
                            axis=0)
            for row in members.to_dict("records"):
                r = row["row_id"]
                if r not in daily or row["cost_model"] == "mid" or not row["efq"] > 0:
                    continue
                gross, cost = daily[r][0][common], daily[r][2][common] / row["efq"]
                low, high = breakeven_interval(gross, cost, index)
                per_row[r].update(breakeven_efq_low=low, breakeven_efq_high=high)

    extra = pd.DataFrame([{"row_id": r, **v} for r, v in per_row.items()])
    identity = rows[["row_id", "label", *[c for c in rows.columns if c.startswith("f_")],
                     *[c for c in ("forecast", "eval_horizon") if c in rows],
                     *[c for c in CONFIG_FIELDS if c in rows],
                     *[c for c in ("breakeven_efq",) if c in rows]]]
    table = identity.merge(stats_frame, on="row_id", how="left")
    if not extra.empty:
        table = table.merge(extra, on="row_id", how="left")
    table.to_parquet(out / "inference.parquet", index=False)
    if tod:
        pd.DataFrame(tod).to_parquet(out / "inference_tod.parquet", index=False)
    if daily:
        np.savez_compressed(out / "inference_daily.npz", __days__=all_days, **daily)

    report = {
        "name": name, "spec_name": spec.get("name") if spec else None,
        "settings": {"benchmark": benchmark, "group_by": group_by, "pnl": pnl, "reps": reps,
                     "seed": seed, "block": block, "lags": lags, "mcs_size": MCS_SIZE,
                     "test_size": TEST_SIZE, "pbo_blocks": pbo_n,
                     "decision_spacing": spacing},
        "checks": {
            "rows": len(ids), "labels": {k: sum(1 for v in source.values() if v == k)
                                         for k in ("periods.npz", "rebuilt", "daily.npz")},
            "label_failures": [{"row_id": r, "reason": why} for r, why in failed],
            "max_abs_daily_sharpe_vs_rows": float(max(stored_check)) if stored_check else None,
            "max_abs_sharpe_vs_rows": float(np.nanmax(np.abs(np.r_[
                table["sharpe_mid"].to_numpy(float) - rows["sharpe_mid"].to_numpy(float),
                table["sharpe_net"].to_numpy(float) - rows["sharpe_net"].to_numpy(float)]))),
            "n_days": len(all_days),
        },
        "groups": groups,
        "seed_dispersion": seed_dispersion(table),
        "size_check": size_check(table),
    }
    (out / "inference.json").write_text(json.dumps(_clean(report), indent=2))
    print_summary(report, table)
    return report


def _clean(value):
    """JSON-safe: numpy scalars as Python, NaN and infinity as null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    value = _scalar(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _quantiles(values) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "n/a"
    low, mid, high = np.quantile(values, [0.1, 0.5, 0.9])
    return f"{mid:.2f} [{low:.2f}, {high:.2f}]"


def _number(value, spec: str) -> str:
    return "-" if value is None or not np.isfinite(value) else format(value, spec)


def print_summary(report: dict, table) -> None:
    """A few lines per run and one per group; the files hold everything else."""
    checks = report["checks"]
    print(f"{report['name']}: {checks['rows']} rows, {checks['n_days']} days, "
          f"{len(report['groups'])} groups by {','.join(report['settings']['group_by'])}")
    print(f"  labels: {checks['labels']}, failures: {len(checks['label_failures'])}; "
          f"recomputed vs stored Sharpe max |diff|: per period "
          f"{_number(checks['max_abs_sharpe_vs_rows'], '.1e')}, daily "
          f"{_number(checks['max_abs_daily_sharpe_vs_rows'], '.1e')}")
    print(f"  per-period net Sharpe, Lo / naive (median [p10, p90]): "
          f"{_quantiles(table['lo_ratio_net'])}; HAC / iid s.e.: "
          f"{_quantiles(table['sharpe_se_hac_net'] / table['sharpe_se_iid_net'])}")
    # Name groups and configurations by what varies, not by what every row shares.
    group_keys = [k for k in report["settings"]["group_by"]
                  if len({str(g["keys"].get(k)) for g in report["groups"]}) > 1]
    axes = [c for c in CONFIG_FIELDS if c in table and c not in report["settings"]["group_by"]
            and table[c].astype(str).nunique() > 1]
    by_id = table.set_index("row_id")

    def short(row_id):
        return "/".join(str(by_id.at[row_id, c]) for c in axes) or by_id.at[row_id, "label"]

    print(f"  group: {','.join(group_keys) or '(one)'}; configuration: {'/'.join(axes)}")
    print(f"  {'group':<28} {'k':>4} {'best by daily net SR':<38} {'MCS':>4} {'SPA p':>6} "
          f"{'StepM':>5} {'N_eff':>6} {'DSR':>5} {'PBO':>5}")
    for g in report["groups"]:
        name = ",".join(str(g["keys"].get(k)) for k in group_keys)[:28] or "all"
        if "dsr" not in g:
            print(f"  {name:<28} {g['n_configs']:>4} {g.get('skipped', '')}")
            continue
        d = g["dsr"]
        best = f"{short(d['best_row_id'])[:30]} {d['best_sharpe_daily']:.1f}"
        stepm = len(g["stepm"]["superior"]) if "stepm" in g else "-"
        print(f"  {name:<28} {g['n_configs']:>4} {best:<38} {len(g['mcs']['included']):>4} "
              f"{_number(g.get('spa', {}).get('consistent'), '.3f'):>6} {stepm:>5} "
              f"{_number(d['n_eff_used'], '.1f'):>6} {_number(d['best_dsr'], '.2f'):>5} "
              f"{_number(g.get('pbo', {}).get('probability'), '.2f'):>5}")
    size = report["size_check"]
    if size:
        print(f"  size check on {size['n_rows']} IC-0 rows (gross, 5% level): "
              + ", ".join(f"{k[len('reject_'):]} {v:.2f}" for k, v in size.items()
                          if k.startswith("reject_") and v is not None))
    seeds = report["seed_dispersion"]
    if seeds:
        ratio = [s["sharpe_daily_net_sd"] / s["sharpe_daily_se_hac_net_mean"] for s in seeds
                 if s["sharpe_daily_net_sd"] is not None and s["sharpe_daily_se_hac_net_mean"]]
        print(f"  seeds: {len(seeds)} configurations under several seeds; seed sd / time "
              f"s.e. of the daily net Sharpe: {_quantiles(ratio)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="a run under results/ (or the path of a run directory)")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                        help="key=value pairs (or a config label) naming the benchmark in each "
                             f"group; ties go to the first row (default {DEFAULT_BENCHMARK!r})")
    parser.add_argument("--group", default=None,
                        help="comma-separated row columns to group by, or 'all' for one group "
                             "(default: forecast, its parameters, eval_horizon, efq)")
    parser.add_argument("--pnl", choices=PNL, default="net", help="series the group tests use")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=None,
                        help="fixed mean block length in days (default: Politis-White)")
    parser.add_argument("--lags", type=int, default=None,
                        help="fixed HAC lags for the per-period series (default: see the docstring)")
    parser.add_argument("--pbo-blocks", type=int, default=None)
    args = parser.parse_args(argv)
    group = args.group.split(",") if args.group else None
    analyze(args.name, benchmark=args.benchmark, group_by=group, pnl=args.pnl, reps=args.reps,
            seed=args.seed, block=args.block, lags=args.lags, pbo_n=args.pbo_blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
