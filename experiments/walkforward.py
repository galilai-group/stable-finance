"""Walk-forward: one spec over every consecutive (history, evaluation) month pair.

One month is an anecdote. It holds about twenty trading days and 1,638
decisions, so the day-level Sharpe ratio of a configuration has a standard
error near one annualized unit before any correction, and two constructions
whose returns are 0.9 correlated cannot be told apart unless their Sharpe
ratios differ by about 1.5. It is also ONE market: August 2020 was calm and
trending, while the store's eighteen months run from the quiet 2019 summer
through the March 2020 crash -- quoted spreads several times wider, realized
volatility several times higher -- and back. Alpha tends to scale with
volatility and cost with spread, so a ranking or a break-even EFQ measured
in one month need not survive the next. This script runs a spec, unchanged,
on every month and asks the three questions a single month cannot answer:

1. HOW MUCH DOES EACH CONFIGURATION MOVE FROM MONTH TO MONTH? Per
   configuration (a ``row_id``, which names the forecast, the horizon and the
   harness settings and not the month): the mean, spread, range and share of
   positive months of every metric, and a Sharpe ratio pooled over every day
   of every month with a paired day-block bootstrap interval, which is where
   seventeen months buy their power.
2. DO THE RANKINGS HOLD? Within each (forecast, horizon) group the
   configurations are ranked month by month, and agreement is measured by
   Kendall's W (0: rankings unrelated across months, 1: identical), its
   chi-square test, the mean and worst pairwise Spearman correlation between
   months, and how often a month's top decile stays in another month's top
   decile. The probability of backtest overfitting (CSCV, Bailey et al. 2017)
   is computed on the pooled daily P&L with contiguous blocks that are close
   to one month each, so "the best configuration in some months" is scored
   against "the best configuration out of sample".
3. DOES THE ANSWER DEPEND ON THE REGIME? Every day is described from the
   cache itself -- the cross-sectional median of its realized volatility
   (from non-overlapping five-minute returns), the mean quoted relative
   half-spread, and the cross-sectional dispersion of five-minute returns --
   and bucketed into terciles pooled over all days; each month is bucketed by
   its mean the same way. Each configuration's day-summed P&L is then
   summarized per bucket: Sharpe ratios, mean daily P&L and the break-even
   EFQ, which says whether the edge survives when spreads widen.

Every month must also be COMPLETE to be comparable. Each evaluation month's
cached days are checked against the NYSE calendar (and against the store's
own month index when it is on disk): a missing day, a thinly quoted day, a
known early close and the March 2020 market-wide halts are flagged, printed
and kept with the results rather than silently averaged over.

EVERYTHING GOES THROUGH ``run.py``. Each pair gets a per-month spec named
``<name>_wf/<YYYY-MM>`` (the evaluation month) with ``eval_month`` and
``history_month`` substituted and ``save_daily`` set, and is run in-process
by :func:`experiments.run.run`, so it resumes from task checkpoints, writes
the usual ``rows.parquet``, ``series.npz`` and ``manifest.json``, and is
byte-identical on a rerun. A pair runs only when both months are cached
(``market1t-v2-YYYY-MM.npz`` when the spec says ``"cache": "v2"``, else
``market1t-YYYY-MM.npz``); nothing is downloaded or built here.

The protocol's segments are pre-declared (research item 2): 2019-08 to
2019-12 is the DESIGN period on which the grid and the primary metric are
frozen, February to April 2020 the STRESS regime, November and December 2020
the sealed HOLDOUT; every other month is TEST. The segment only labels rows
and splits the summaries, unless ``--seal-holdout`` is given, which refuses to
run or aggregate holdout months until the design is final.

    uv run --extra experiments python -m experiments.walkforward \\
        experiments/specs/e4_harness_sensitivity.json --months 2019-08..2020-12 --workers 4

Outputs under ``results/<name>_wf/``: one directory per evaluation month
(run.py's own files plus ``daily.npz``), and the aggregate:
``rows.parquet`` (every month's rows with ``eval_month``, ``history_month``,
``segment`` and ``month_missing_days``), ``daily.npz`` (the day-by-row P&L
matrices), ``regimes.parquet``, ``coverage.json``, ``configs.parquet``,
``stability.parquet``, ``regime_summary.parquet``, ``summary.json`` and
``figures/wf_*-{light,dark}.png``.
"""

from __future__ import annotations

import os

# One linear-algebra thread, as in run.py: the parallelism is across tasks.
# Set before numpy is imported, which is when the BLAS reads it.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import calendar
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

import stable_finance as sf
from experiments import run as harness
from experiments.month_cache import ROOT, cache_dir

#: NYSE full-day closures over the store's span and the year after it, as
#: published by the exchange (they match the TFWM ``market_holidays_nyse.csv``).
#: A weekday of a month that is not here and not in the cache is a missing day.
NYSE_HOLIDAYS = frozenset({
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", "2019-07-04",
    "2019-09-02", "2019-11-28", "2019-12-25",
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", "2020-07-03",
    "2020-09-07", "2020-11-26", "2020-12-25",
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31", "2021-07-05",
    "2021-09-06", "2021-11-25", "2021-12-24",
})
#: Years the holiday list covers; outside them every weekday is expected.
CALENDAR_YEARS = frozenset({2019, 2020, 2021})
#: 13:00 closes: about 42 of the 78 decisions have a market behind them.
NYSE_EARLY_CLOSES = frozenset({"2019-07-03", "2019-11-29", "2019-12-24", "2020-11-27",
                               "2020-12-24", "2021-11-26"})
#: Level-1 market-wide circuit breakers (15-minute halts) of March 2020.
MARKET_WIDE_HALTS = frozenset({"2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"})

#: Pre-declared protocol segments (first and last evaluation month, inclusive).
#: Any other month is ``test``.
SEGMENTS = (("design", "2019-08", "2019-12"), ("stress", "2020-02", "2020-04"),
            ("holdout", "2020-11", "2020-12"))

#: Metrics summarized per configuration across months, when the rows have them.
METRICS = ("sharpe_daily_net", "sharpe_daily_mid", "sharpe_net", "sharpe_mid",
           "breakeven_efq", "mean_gross_return_bps", "mean_cost_bps", "mean_turnover",
           "information_coefficient", "transfer_coefficient")
#: Metrics whose month-by-month rankings are tested for stability.
RANK_METRICS = ("sharpe_daily_net", "sharpe_daily_mid", "breakeven_efq")
#: Day descriptors computed from the cache, each bucketed into terciles.
REGIME_VARIABLES = ("realized_vol_bps", "half_spread_bps", "dispersion_bps")
TERCILES = ("low", "mid", "high")
#: A day quoted on less than this share of its (decision, name) cells, among
#: the names the month's screen keeps, is flagged as thin.
THIN_DAY = 0.9
#: Share of a month's top configurations tested for staying on top.
TOP_SHARE = 0.1


# ── months and pairs ────────────────────────────────────────────────────────

def _month_index(month: str) -> int:
    year, number = (int(part) for part in month.split("-"))
    return year * 12 + number - 1


def _month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def shift_month(month: str, by: int) -> str:
    return _month_label(_month_index(month) + by)


def parse_months(text: str) -> list[str]:
    """Evaluation months from ``A..B`` (inclusive), ``A,B,C`` or one ``A``."""
    months: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if ".." in part:
            first, last = (p.strip() for p in part.split(".."))
            if _month_index(last) < _month_index(first):
                raise ValueError(f"empty month range {part!r}")
            months += [_month_label(i)
                       for i in range(_month_index(first), _month_index(last) + 1)]
        elif part:
            months.append(_month_label(_month_index(part)))
    return list(dict.fromkeys(months))


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Explicit ``HISTORY:EVAL`` pairs, comma separated."""
    pairs = []
    for part in text.split(","):
        history, _, evaluation = part.strip().partition(":")
        if not evaluation:
            raise ValueError(f"a pair is HISTORY:EVAL, got {part!r}")
        pairs.append((_month_label(_month_index(history)),
                      _month_label(_month_index(evaluation))))
    return pairs


def segment_of(month: str) -> str:
    for name, first, last in SEGMENTS:
        if _month_index(first) <= _month_index(month) <= _month_index(last):
            return name
    return "test"


def cache_version(spec: dict) -> str:
    return "v2" if spec.get("cache") == "v2" else "v1"


def cache_path(month: str, version: str = "v1") -> Path:
    """Where ``month_cache`` (v1) or ``month_cache_v2`` (v2) writes a month."""
    stem = f"market1t-v2-{month}" if version == "v2" else f"market1t-{month}"
    return cache_dir() / f"{stem}.npz"


def plan(spec: dict, pairs: list[tuple[str, str]], *, seal_holdout: bool = False):
    """The pairs that can run, and the ones that cannot with the reason.

    A pair needs both months cached in the spec's cache version. Each
    evaluation month names its own result directory, so it may appear once.
    """
    evaluations = [e for _, e in pairs]
    repeated = sorted({e for e in evaluations if evaluations.count(e) > 1})
    if repeated:
        raise ValueError(f"evaluation months must be distinct, repeated: {repeated}")
    version = cache_version(spec)
    runnable, skipped = [], []
    for history, evaluation in pairs:
        if seal_holdout and segment_of(evaluation) == "holdout":
            skipped.append({"history_month": history, "eval_month": evaluation,
                            "reason": "holdout sealed"})
            continue
        missing = [str(cache_path(m, version)) for m in (history, evaluation)
                   if not cache_path(m, version).is_file()]
        if missing:
            skipped.append({"history_month": history, "eval_month": evaluation,
                            "reason": "cache missing", "missing": missing})
        else:
            runnable.append((history, evaluation))
    return runnable, skipped


# ── running ────────────────────────────────────────────────────────────────

def month_spec(spec: dict, name: str, history: str, evaluation: str) -> dict:
    """The spec for one pair: same forecasts and grid, months substituted."""
    return {**spec, "name": f"{name}_wf/{evaluation}", "eval_month": evaluation,
            "history_month": history, "save_daily": True}


def _done(out: Path, spec: dict) -> bool:
    """A month whose results were assembled from exactly this spec."""
    manifest = out / "manifest.json"
    if not (manifest.is_file() and (out / "rows.parquet").is_file()
            and (out / "daily.npz").is_file()):
        return False
    return json.loads(manifest.read_text()).get("spec") == json.loads(json.dumps(spec))


def run_month(spec: dict, name: str, history: str, evaluation: str, workers: int) -> Path:
    """Run one pair through ``run.py`` in-process, unless it is already done."""
    one = month_spec(spec, name, history, evaluation)
    out = harness.RESULTS / one["name"]
    if _done(out, one):
        print(f"{one['name']}: already done", flush=True)
        return out
    specs = harness.RESULTS / f"{name}_wf" / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    path = specs / f"{evaluation}.json"
    path.write_text(json.dumps(one, indent=2))
    try:
        return harness.run(str(path), workers=workers)
    finally:
        # The sequential path keeps the month's inputs in a module global.
        harness._INPUTS = None


# ── regimes and coverage, from the caches ──────────────────────────────────

def month_regime(month: str, version: str = "v1", *, min_presence: float = 0.95,
                 spacing: int = 300):
    """One row per cached day of ``month``: its market regime and its coverage.

    Names are those quoted on at least ``min_presence`` of the month's
    decisions, the evaluation screen of ``run.load_inputs`` (the history
    screen is left out: the regime describes the month's market, not one
    book). Realized volatility is each name's daily volatility from the
    non-overlapping ``spacing``-second forward returns -- the mean square
    over the day's quoted decisions times the day's decision count, so a half
    day reads at a full day's scale -- then the cross-sectional median; the
    half-spread is the mean
    quoted relative half-spread over the day's (decision, name) cells;
    dispersion is the cross-sectional standard deviation of the same returns,
    averaged over the day's decisions. All three are in basis points.
    """
    import pandas as pd

    data = np.load(cache_path(month, version), allow_pickle=False)
    days = np.char.partition(data["decisions"].astype(str), "|")[:, 0]
    spread = data["half_spread"].astype(np.float64)
    horizons = np.asarray(data["horizons"])
    h = int(np.flatnonzero(horizons == spacing)[0]) if spacing in horizons \
        else int(np.argmin(horizons))
    returns = data["realized"][:, :, h].astype(np.float64)
    # A forward return longer than the spacing overlaps the next decision's;
    # scaling its square keeps the day's variance roughly additive.
    overlap = min(spacing / float(horizons[h]), 1.0)
    keep = np.isfinite(spread).mean(axis=0) >= min_presence
    spread, returns = spread[:, keep], returns[:, keep]
    labels = np.unique(days)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for day in labels:
            rows = days == day
            r, s = returns[rows], spread[rows]
            n = int(rows.sum())
            rv = np.sqrt(np.nanmean(r ** 2, axis=0) * n * overlap)
            out.append({
                "eval_month": month, "day": str(day), "n_decisions": n,
                "n_names": int(keep.sum()),
                "quoted_share": float(np.isfinite(s).mean()),
                "realized_vol_bps": float(np.nanmedian(rv) * 1e4),
                "half_spread_bps": float(np.nanmean(s) * 1e4),
                "dispersion_bps": float(np.nanmean(np.nanstd(r, axis=1)) * 1e4),
            })
    return pd.DataFrame(out)


def expected_days(month: str) -> list[str]:
    """The month's NYSE trading days: weekdays less the exchange's closures."""
    year, number = (int(part) for part in month.split("-"))
    days = [f"{month}-{d:02d}" for d in range(1, calendar.monthrange(year, number)[1] + 1)
            if calendar.weekday(year, number, d) < 5]
    return [d for d in days if d not in NYSE_HOLIDAYS]


def store_days(month: str) -> list[str] | None:
    """The days the Hugging Face store's month index lists, if it is on disk.

    Read from the local snapshot only, never downloaded: ``months.py`` deletes
    a month's raw files once its cache is written, and the check is
    opportunistic.
    """
    from stable_finance.dataset.datasets.market1t.spec import cache_root, parse_hub

    _, repo_id, revision, subdir = parse_hub(ROOT)
    local = cache_root() / (repo_id.replace("/", "--") + (f"@{revision}" if revision else ""))
    index = local / subdir / month[:4] / month[5:7] / "index.json"
    if not index.is_file():
        return None
    return sorted(d["date"] for d in json.loads(index.read_text())["days"])


def month_coverage(month: str, regime, rows=None) -> dict:
    """What a month is missing, against the calendar, the store and its rows.

    ``flagged`` is set by a missing or unexpected day, a day quoted on less
    than ``THIN_DAY`` of its cells that is not a known early close, a day the
    store lists but the cache lacks, or rows covering fewer days than the
    cache. Early closes and the March 2020 halts are listed but do not flag.
    """
    cached = regime["day"].astype(str).tolist()
    expected = expected_days(month)
    year = int(month[:4])
    missing = sorted(set(expected) - set(cached))
    unexpected = sorted(set(cached) - set(expected))
    full = int(regime["n_decisions"].max()) if len(regime) else 0
    short = regime[regime["n_decisions"] < full]["day"].astype(str).tolist()
    thin = regime[regime["quoted_share"] < THIN_DAY]["day"].astype(str).tolist()
    early = [d for d in cached if d in NYSE_EARLY_CLOSES]
    halts = [d for d in cached if d in MARKET_WIDE_HALTS]
    store = store_days(month)
    missing_vs_store = sorted(set(store) - set(cached)) if store is not None else None
    row_days = None
    if rows is not None and "n_days" in rows and len(rows):
        row_days = {"min": int(rows["n_days"].min()), "max": int(rows["n_days"].max())}
    unexplained_thin = [d for d in thin if d not in NYSE_EARLY_CLOSES]
    flagged = bool(missing or unexpected or short or unexplained_thin or missing_vs_store
                   or (row_days is not None and row_days["min"] < len(cached)))
    return {"eval_month": month, "calendar_known": year in CALENDAR_YEARS,
            "n_expected": len(expected), "n_cached": len(cached), "missing_days": missing,
            "unexpected_days": unexpected, "short_days": short, "thin_days": thin,
            "early_closes": early, "market_wide_halts": halts,
            "store_days": None if store is None else len(store),
            "missing_vs_store": missing_vs_store, "row_days": row_days, "flagged": flagged}


def tercile(values) -> np.ndarray:
    """Rank terciles (0, 1, 2) of ``values``; NaN stays unassigned (-1).

    Ranks rather than quantile cuts, so ties and small samples still split as
    evenly as they can, and ``ordinal`` ranks keep ties in input order, which
    is chronological here, so the split is deterministic.
    """
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), -1, dtype=np.int64)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n:
        ranks = rankdata(values[finite], method="ordinal")
        out[finite] = np.minimum((3 * (ranks - 1)) // n, 2).astype(np.int64)
    return out


def bucket_regimes(regimes):
    """Day terciles pooled over every day, and month terciles of month means."""
    regimes = regimes.sort_values(["eval_month", "day"]).reset_index(drop=True)
    means = regimes.groupby("eval_month", sort=True)[list(REGIME_VARIABLES)].mean()
    for variable in REGIME_VARIABLES:
        regimes[f"{variable}_day_tercile"] = [
            TERCILES[i] if i >= 0 else None for i in tercile(regimes[variable])]
        month_bucket = dict(zip(means.index, tercile(means[variable])))
        regimes[f"{variable}_month_tercile"] = [
            TERCILES[month_bucket[m]] if month_bucket[m] >= 0 else None
            for m in regimes["eval_month"]]
    return regimes


# ── statistics ─────────────────────────────────────────────────────────────

def kendall_w(matrix) -> tuple[float, float, float]:
    """Kendall's coefficient of concordance of ``(months, configs)`` scores.

    Each month ranks the configurations (ties averaged); ``W = 12 S /
    (m^2 (n^3 - n) - m T)`` with ``S`` the spread of the configurations' rank
    sums and ``T`` the tie correction. Returns ``(W, chi2, p)``, the large-
    sample test of ``W = 0`` being ``m (n - 1) W ~ chi2(n - 1)``. NaN when
    fewer than two months or two configurations, or every month is one tie.
    """
    from scipy.stats import chi2, rankdata

    matrix = np.asarray(matrix, dtype=np.float64)
    m, n = matrix.shape
    if m < 2 or n < 2:
        return float("nan"), float("nan"), float("nan")
    ranks = np.vstack([rankdata(row) for row in matrix])
    totals = ranks.sum(axis=0)
    s = float(((totals - totals.mean()) ** 2).sum())
    ties = 0.0
    for row in matrix:
        _, counts = np.unique(row, return_counts=True)
        ties += float((counts ** 3 - counts).sum())
    denominator = m ** 2 * (n ** 3 - n) - m * ties
    if denominator <= 0:
        return float("nan"), float("nan"), float("nan")
    w = 12.0 * s / denominator
    statistic = m * (n - 1) * w
    return float(w), float(statistic), float(chi2.sf(statistic, n - 1))


def spearman_between(matrix) -> np.ndarray:
    """``(months, months)`` Spearman correlations of the configurations' scores."""
    from scipy.stats import rankdata

    ranks = np.vstack([rankdata(row) for row in np.asarray(matrix, dtype=np.float64)])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(ranks)


def top_persistence(matrix, share: float = TOP_SHARE) -> float:
    """Mean share of one month's top configurations in another month's top.

    Over every ordered pair of distinct months. Chance level is ``share``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    m, n = matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    k = max(round(share * n), 1)
    # Stable descending order: ties go to the earlier configuration.
    tops = [set(np.argsort(-row, kind="stable")[:k].tolist()) for row in matrix]
    shares = [len(tops[i] & tops[j]) / k for i in range(m) for j in range(m) if i != j]
    return float(np.mean(shares))


def _sharpe(daily, axis=-1):
    """Annualized Sharpe of daily P&L ignoring NaN days; NaN when flat."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(daily, axis=axis)
        sd = np.nanstd(daily, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, mean / sd * np.sqrt(252.0), np.nan)


def pooled_sharpe(daily: np.ndarray, *, chunk: int = 512) -> dict:
    """``(rows, days)`` daily P&L -> Sharpe per row with a 95% day-block interval.

    As in ``run._inference`` the generator is seeded and depends only on the
    number of days, so every row resamples the same days and intervals are
    paired across configurations; a day a row has no P&L on is NaN and is
    left out of that row's draws. A draw is a vector of day counts, so every
    row's resampled sums are one product with the count matrix rather than a
    fancy-indexed copy per row -- seventeen months of a large grid are
    thousands of rows times hundreds of days times 2,000 draws. Each row is
    centred on its own mean first, which leaves the variance unchanged and
    keeps the sum-of-squares form from cancelling.
    """
    rows, n = daily.shape
    point = _sharpe(daily)
    low, high = np.full(rows, np.nan), np.full(rows, np.nan)
    if n < 2 or rows == 0:
        return {"sharpe": point, "low": low, "high": high}
    draws = np.random.default_rng(0).integers(0, n, size=(harness.BOOTSTRAP_DRAWS, n))
    offsets = draws + n * np.arange(harness.BOOTSTRAP_DRAWS)[:, None]
    counts = np.bincount(offsets.ravel(), minlength=harness.BOOTSTRAP_DRAWS * n)
    counts = counts.reshape(harness.BOOTSTRAP_DRAWS, n).astype(np.float64)
    for start in range(0, rows, chunk):
        block = daily[start:start + chunk]
        valid = np.isfinite(block)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nan_to_num(np.nanmean(block, axis=1))
        x = np.where(valid, block - centre[:, None], 0.0)
        k = counts @ valid.T.astype(np.float64)
        s1, s2 = counts @ x.T, counts @ (x * x).T
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / k
            variance = (s2 - k * mean ** 2) / (k - 1)
            boot = np.where((k > 1) & (variance > 0),
                            (mean + centre) / np.sqrt(variance) * np.sqrt(252.0), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low[start:start + chunk], high[start:start + chunk] = np.nanpercentile(
                boot, [2.5, 97.5], axis=0)
    return {"sharpe": point, "low": low, "high": high}


def _breakeven(gross: np.ndarray, cost: np.ndarray, efq: np.ndarray,
               model: np.ndarray) -> np.ndarray:
    """Break-even EFQ of summed daily P&L, as ``run._breakeven`` does per row."""
    out = np.full(len(gross), np.nan)
    for i in range(len(gross)):
        if model[i] == "mid" or not efq[i] > 0:
            continue
        g, c = gross[i], cost[i]
        finite = np.isfinite(g) & np.isfinite(c)
        if finite.sum():
            out[i] = sf.breakeven_efq(g[finite], c[finite] / efq[i])
    return out


# ── aggregation ────────────────────────────────────────────────────────────

def _config_columns(rows) -> list[str]:
    """Columns that name a configuration: the forecast, the horizon, the harness."""
    fields = [f.name for f in dataclasses.fields(sf.BacktestConfig)]
    forecast = ["forecast", *sorted(c for c in rows.columns if c.startswith("f_"))]
    return [c for c in (*forecast, "eval_horizon", *fields) if c in rows.columns]


def _group_label(row, columns) -> str:
    parts = [f"{c[2:]}={row[c]}" for c in columns if c.startswith("f_") and _present(row[c])]
    return " ".join([str(row["forecast"]), *parts, f"h={row['eval_horizon']}"])


def _present(value) -> bool:
    return not (value is None or (isinstance(value, float) and math.isnan(value)))


def config_table(rows):
    """One row per configuration: its identity, a short label and its group."""
    columns = _config_columns(rows)
    configs = rows.drop_duplicates("row_id")[["row_id", *columns]].reset_index(drop=True)
    group_columns = [c for c in columns if c == "forecast" or c.startswith("f_")]
    configs["group"] = [_group_label(r, [*group_columns]) for _, r in configs.iterrows()]
    harness_columns = [c for c in columns if c not in group_columns and c != "eval_horizon"]
    varying = [c for c in harness_columns if configs[c].astype(str).nunique() > 1]
    configs["label"] = ["|".join(f"{c}={r[c]}" for c in varying) or "base"
                        for _, r in configs.iterrows()]
    return configs


def load_months(name: str, pairs: list[tuple[str, str]]):
    """Every month's rows and day sums, months in order, rows in run order."""
    import pandas as pd

    frames, daily = [], {}
    for history, evaluation in pairs:
        out = harness.RESULTS / f"{name}_wf" / evaluation
        frame = pd.read_parquet(out / "rows.parquet")
        frame.insert(0, "history_month", history)
        frame.insert(0, "eval_month", evaluation)
        frame.insert(2, "segment", segment_of(evaluation))
        frames.append(frame)
        data = np.load(out / "daily.npz", allow_pickle=False)
        daily[evaluation] = {"days": data["__days__"].astype(str),
                             **{k: data[k] for k in data.files if k != "__days__"}}
    return pd.concat(frames, ignore_index=True, sort=False), daily


def daily_matrices(row_ids: list[str], daily: dict) -> dict:
    """``(rows, days)`` gross, net and cost over every month's days in order."""
    months = sorted(daily)
    days = np.concatenate([daily[m]["days"] for m in months])
    day_month = np.concatenate([[m] * len(daily[m]["days"]) for m in months])
    out = {k: np.full((len(row_ids), len(days)), np.nan) for k in ("gross", "net", "cost")}
    start = 0
    for m in months:
        width = len(daily[m]["days"])
        for i, row_id in enumerate(row_ids):
            if row_id in daily[m]:
                for k, values in zip(("gross", "net", "cost"), daily[m][row_id]):
                    out[k][i, start:start + width] = values
        start += width
    return {"days": days, "day_month": day_month, "row_ids": np.asarray(row_ids), **out}


def summarize_configs(rows, configs, matrices, metric: str):
    """Across-month moments of every metric and the pooled daily statistics."""
    months = sorted(rows["eval_month"].unique())
    summary = configs.copy()
    summary["n_months"] = 0
    for m in [m for m in METRICS if m in rows.columns]:
        wide = rows.pivot_table(index="row_id", columns="eval_month", values=m,
                                aggfunc="first", dropna=False).reindex(
            index=summary["row_id"], columns=months)
        values = wide.to_numpy(dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary[f"mean_{m}"] = np.nanmean(values, axis=1)
            summary[f"std_{m}"] = np.nanstd(values, axis=1, ddof=1)
            summary[f"min_{m}"] = np.nanmin(values, axis=1)
            summary[f"max_{m}"] = np.nanmax(values, axis=1)
        if m == metric:
            count = np.isfinite(values).sum(axis=1)
            summary["n_months"] = count
            with np.errstate(invalid="ignore", divide="ignore"):
                summary[f"share_positive_{m}"] = np.where(
                    count > 0, (values > 0).sum(axis=1) / count, np.nan)
                summary[f"t_months_{m}"] = (summary[f"mean_{m}"] / summary[f"std_{m}"]
                                            * np.sqrt(count))
    for kind, key in (("net", "net"), ("mid", "gross")):
        pooled = pooled_sharpe(matrices[key])
        summary[f"pooled_sharpe_daily_{kind}"] = pooled["sharpe"]
        summary[f"pooled_sharpe_daily_{kind}_low"] = pooled["low"]
        summary[f"pooled_sharpe_daily_{kind}_high"] = pooled["high"]
    summary["pooled_n_days"] = np.isfinite(matrices["net"]).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary["pooled_mean_daily_net_bps"] = np.nanmean(matrices["net"], axis=1) * 1e4
    if "efq" in summary and "cost_model" in summary:
        summary["pooled_breakeven_efq"] = _breakeven(
            matrices["gross"], matrices["cost"], summary["efq"].to_numpy(dtype=float),
            summary["cost_model"].astype(str).to_numpy())
    # Deflated Sharpe within each group, deflated by the group's own trials,
    # on the days every one of them has.
    summary["dsr_daily_net"] = np.nan
    summary["pooled_rank_in_group"] = np.nan
    for index in summary.groupby("group", sort=False).groups.values():
        index = np.asarray(index)
        block = matrices["net"][index]
        common = np.all(np.isfinite(block), axis=0)
        if common.sum() >= 3:
            summary.loc[index, "dsr_daily_net"] = sf.deflated_sharpe_from_trials(
                block[:, common].T)
        summary.loc[index, "pooled_rank_in_group"] = summary.loc[
            index, "pooled_sharpe_daily_net"].rank(ascending=False, method="min")
    return summary


def _pbo(block: np.ndarray, n_months: int) -> dict:
    """CSCV PBO on ``(configs, days)`` with blocks near one month each."""
    n_blocks = min(n_months - n_months % 2, 16)
    common = np.all(np.isfinite(block), axis=0)
    if n_blocks < 2 or block.shape[0] < 2 or common.sum() < 2 * n_blocks:
        return {"pbo": float("nan"), "pbo_blocks": n_blocks,
                "pbo_probability_of_loss": float("nan")}
    try:
        result = sf.pbo(block[:, common].T, n_blocks=n_blocks)
    except ValueError:
        return {"pbo": float("nan"), "pbo_blocks": n_blocks,
                "pbo_probability_of_loss": float("nan")}
    return {"pbo": result.probability, "pbo_blocks": n_blocks,
            "pbo_probability_of_loss": result.probability_of_loss}


def ranking_stability(rows, configs, matrices) -> list[dict]:
    """Kendall's W, Spearman between months, top-decile persistence and PBO.

    Within each (forecast, horizon) group -- ranking an IC 0.10 forecast
    against an IC 0.01 one says nothing about constructions -- and over all
    rows, for every month and for each segment with two months or more that
    is not every month. A configuration missing a month is left out of that
    scope's ranking (W needs complete blocks) and counted in ``n_dropped``; one
    the metric does not apply to in any month (the break-even EFQ of a
    frictionless book) is not counted.
    """
    group_of = dict(zip(configs["row_id"], configs["group"]))
    rows = rows.assign(group=rows["row_id"].map(group_of))
    position = {r: i for i, r in enumerate(matrices["row_ids"].tolist())}
    scopes = [("all", sorted(rows["eval_month"].unique()))]
    for segment in [s for s, _, _ in SEGMENTS] + ["test"]:
        months = sorted(rows.loc[rows["segment"] == segment, "eval_month"].unique())
        if len(months) >= 2 and months != scopes[0][1]:
            scopes.append((segment, months))
    groups = [("all rows", rows)] + [(g, part) for g, part in rows.groupby("group", sort=False)]
    out = []
    for segment, months in scopes:
        day_mask = np.isin(matrices["day_month"], months)
        for group, part in groups:
            part = part[part["eval_month"].isin(months)]
            ids = list(dict.fromkeys(part["row_id"]))
            pbo = _pbo(matrices["net"][[position[i] for i in ids]][:, day_mask], len(months))
            for metric in [m for m in RANK_METRICS if m in part.columns]:
                wide = part.pivot_table(index="eval_month", columns="row_id", values=metric,
                                        aggfunc="first", dropna=False).reindex(
                    index=months, columns=ids)
                values = wide.to_numpy(dtype=np.float64)
                complete = np.all(np.isfinite(values), axis=0)
                applicable = np.any(np.isfinite(values), axis=0)
                values = values[:, complete]
                w, statistic, p = kendall_w(values)
                rho = spearman_between(values) if values.shape[1] >= 2 and len(months) >= 2 \
                    else np.full((1, 1), np.nan)
                off = rho[~np.eye(len(rho), dtype=bool)] if len(rho) > 1 else np.array([])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out.append({
                        "segment": segment, "group": group, "metric": metric,
                        "n_months": len(months), "n_configs": int(complete.sum()),
                        "n_dropped": int((applicable & ~complete).sum()), "kendall_w": w,
                        "chi2": statistic, "p_value": p,
                        "mean_spearman": float(np.nanmean(off)) if len(off) else np.nan,
                        "min_spearman": float(np.nanmin(off)) if len(off) else np.nan,
                        "top_decile_persistence": top_persistence(values),
                        **pbo,
                    })
    return out


def summarize_regimes(configs, matrices, regimes):
    """Per configuration and tercile of each regime variable, day and month unit.

    A day's bucket comes from its own regime (``unit="day"``, terciles pooled
    over every day) or from its month's (``unit="month"``); the statistics are
    always over the bucket's days, so the two units are directly comparable.
    """
    import pandas as pd

    by_day = regimes.set_index("day")
    efq = configs["efq"].to_numpy(dtype=float) if "efq" in configs else None
    model = configs["cost_model"].astype(str).to_numpy() if "cost_model" in configs else None
    frames = []
    for variable in REGIME_VARIABLES:
        for unit in ("day", "month"):
            column = by_day[f"{variable}_{unit}_tercile"]
            buckets = np.asarray([column.get(d) for d in matrices["days"]], dtype=object)
            for name in TERCILES:
                mask = buckets == name
                if not mask.any():
                    continue
                net, gross = matrices["net"][:, mask], matrices["gross"][:, mask]
                net_sharpe, mid_sharpe = pooled_sharpe(net), pooled_sharpe(gross)
                breakeven = (_breakeven(gross, matrices["cost"][:, mask], efq, model)
                             if efq is not None else np.full(len(configs), np.nan))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_net = np.nanmean(net, axis=1) * 1e4
                    mean_gross = np.nanmean(gross, axis=1) * 1e4
                level = float(regimes.loc[regimes[f"{variable}_{unit}_tercile"] == name,
                                          variable].mean())
                frames.append(pd.DataFrame({
                    "row_id": matrices["row_ids"], "group": configs["group"].to_numpy(),
                    "label": configs["label"].to_numpy(), "variable": variable,
                    "unit": unit, "tercile": name, "level_mean": level,
                    "n_days": np.isfinite(net).sum(axis=1),
                    "mean_daily_net_bps": mean_net, "mean_daily_mid_bps": mean_gross,
                    "sharpe_daily_net": net_sharpe["sharpe"],
                    "sharpe_daily_net_low": net_sharpe["low"],
                    "sharpe_daily_net_high": net_sharpe["high"],
                    "sharpe_daily_mid": mid_sharpe["sharpe"],
                    "breakeven_efq": breakeven,
                }))
    return pd.concat(frames, ignore_index=True)


def aggregate(name: str, pairs: list[tuple[str, str]], spec: dict, *,
              metric: str = "sharpe_daily_net", skipped: list | None = None,
              figures: bool = True) -> dict:
    """Combine the months of a walk-forward into its summaries and figures."""
    import pandas as pd

    out = harness.RESULTS / f"{name}_wf"
    version = cache_version(spec)
    pairs = sorted(pairs, key=lambda pair: _month_index(pair[1]))
    rows, daily = load_months(name, pairs)
    min_presence = spec.get("min_presence", 0.95)
    regimes = pd.concat([month_regime(e, version, min_presence=min_presence,
                                      spacing=spec.get("decision_spacing", 300))
                         for _, e in pairs], ignore_index=True)
    regimes = bucket_regimes(regimes)
    coverage = [month_coverage(e, regimes[regimes["eval_month"] == e],
                               rows[rows["eval_month"] == e]) for _, e in pairs]
    missing = {c["eval_month"]: len(c["missing_days"]) for c in coverage}
    rows["month_missing_days"] = rows["eval_month"].map(missing).astype("int64")

    configs = config_table(rows)
    matrices = daily_matrices(configs["row_id"].tolist(), daily)
    summary = summarize_configs(rows, configs, matrices, metric)
    stability = pd.DataFrame(ranking_stability(rows, configs, matrices))
    regime_rows = summarize_regimes(configs, matrices, regimes)

    rows.to_parquet(out / "rows.parquet", index=False)
    np.savez_compressed(out / "daily.npz", **matrices)
    regimes.to_parquet(out / "regimes.parquet", index=False)
    summary.to_parquet(out / "configs.parquet", index=False)
    stability.to_parquet(out / "stability.parquet", index=False)
    regime_rows.to_parquet(out / "regime_summary.parquet", index=False)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))

    month_table = []
    means = regimes.groupby("eval_month")[list(REGIME_VARIABLES)].mean()
    first = regimes.drop_duplicates("eval_month").set_index("eval_month")
    for (history, evaluation), cover in zip(pairs, coverage):
        part = rows[rows["eval_month"] == evaluation]
        best = part.sort_values(metric, ascending=False, na_position="last").head(1)
        label = dict(zip(configs["row_id"], configs["label"]))
        group = dict(zip(configs["row_id"], configs["group"]))
        month_table.append({
            "eval_month": evaluation, "history_month": history,
            "segment": segment_of(evaluation),
            "causal": _month_index(history) < _month_index(evaluation),
            "consecutive": _month_index(history) == _month_index(evaluation) - 1,
            "n_rows": len(part), "n_days": cover["n_cached"],
            "missing_days": cover["missing_days"], "flagged": cover["flagged"],
            **{v: float(means.loc[evaluation, v]) for v in REGIME_VARIABLES},
            **{f"{v}_tercile": first.loc[evaluation, f"{v}_month_tercile"]
               for v in REGIME_VARIABLES},
            f"median_{metric}": float(part[metric].median()) if metric in part else None,
            "best": None if best.empty else {
                "group": group[best["row_id"].iloc[0]],
                "label": label[best["row_id"].iloc[0]], metric: float(best[metric].iloc[0])},
        })
    headline = stability[(stability["segment"] == "all") & (stability["metric"] == metric)]
    top = summary.sort_values(f"pooled_{metric}" if f"pooled_{metric}" in summary else metric,
                              ascending=False, na_position="last").head(10)
    report = {
        "name": name, "metric": metric, "cache": version,
        "months": [e for _, e in pairs], "n_rows": len(rows),
        "n_configs": len(configs), "n_days": len(matrices["days"]),
        "skipped": skipped or [], "flagged_months": [c["eval_month"] for c in coverage
                                                     if c["flagged"]],
        "per_month": month_table,
        "stability": json.loads(headline.to_json(orient="records")),
        "top_configs": json.loads(top[[c for c in (
            "group", "label", "n_months", f"mean_{metric}", f"std_{metric}",
            f"share_positive_{metric}", f"pooled_{metric}", f"pooled_{metric}_low",
            f"pooled_{metric}_high", "pooled_breakeven_efq", "dsr_daily_net")
            if c in top]].to_json(orient="records")),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    if figures:
        report["figures"] = [str(p) for p in draw_figures(name, rows, summary, regimes,
                                                          regime_rows, coverage, metric)]
    _print_report(report, stability, metric)
    return report


def _print_report(report: dict, stability, metric: str) -> None:
    print(f"\n{report['name']}_wf: {report['n_rows']} rows, {report['n_configs']} "
          f"configurations, {len(report['months'])} months, {report['n_days']} days")
    for s in report["skipped"]:
        print(f"  skipped {s['history_month']} -> {s['eval_month']}: {s['reason']}")
    print(f"  {'month':8} {'segment':8} {'days':>4} {'vol':>7} {'spread':>7} "
          f"{'median ' + metric:>24}  flags")
    for m in report["per_month"]:
        flags = []
        if m["missing_days"]:
            flags.append(f"missing {len(m['missing_days'])} days")
        if m["flagged"] and not m["missing_days"]:
            flags.append("see coverage.json")
        if not m["causal"]:
            flags.append("NON-CAUSAL pair")
        elif not m["consecutive"]:
            flags.append("history not the prior month")
        median = m[f"median_{metric}"]
        print(f"  {m['eval_month']:8} {m['segment']:8} {m['n_days']:>4} "
              f"{m['realized_vol_bps']:>7.1f} {m['half_spread_bps']:>7.2f} "
              f"{median if median is not None else float('nan'):>24.2f}  {', '.join(flags)}")
    part = stability[(stability["segment"] == "all") & (stability["metric"] == metric)]
    print(f"  ranking stability of {metric} across months:")
    for _, r in part.iterrows():
        print(f"    {r['group'][:60]:60} n={r['n_configs']:>4} W={r['kendall_w']:.3f} "
              f"(p={r['p_value']:.3g}) mean rho={r['mean_spearman']:.3f} "
              f"top-decile kept={r['top_decile_persistence']:.2f} PBO={r['pbo']:.2f}")


# ── figures ────────────────────────────────────────────────────────────────

def _weighting_color(mode: str, weighting) -> str:
    from experiments.figures import SERIES, SLOT, WEIGHTING_ORDER, _color

    if weighting in SLOT or weighting in WEIGHTING_ORDER:
        return _color(mode, weighting, WEIGHTING_ORDER)
    # Slots 5 and 7 are the ones no construction owns.
    return SERIES[mode][(5, 7)[sum(map(ord, str(weighting))) % 2]]


def _weightings(frame) -> list:
    from experiments.figures import WEIGHTING_ORDER

    present = list(dict.fromkeys(frame["weighting"])) if "weighting" in frame else [None]
    return ([w for w in WEIGHTING_ORDER if w in present]
            + sorted(str(w) for w in present if w not in WEIGHTING_ORDER and w is not None)
            or [None])


def _save(fig, out: Path, name: str, mode: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{mode}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def small_multiples(rows, summary, regimes, coverage, metric: str, mode: str,
                    out: Path, title: str) -> Path:
    """One panel per evaluation month; configurations in one fixed order.

    The x order is each configuration's mean across months, so a ranking that
    holds draws the same falling cloud in every panel and one that does not
    is visibly scrambled. Whiskers are the month's own day-block intervals.
    """
    from experiments.figures import INK, LABELS, _plt

    plt = _plt(mode)
    months = sorted(rows["eval_month"].unique())
    order = summary.sort_values(f"mean_{metric}", ascending=False, na_position="last")
    x_of = {r: i for i, r in enumerate(order["row_id"])}
    cols = min(4, len(months))
    nrows = math.ceil(len(months) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.1 * cols, 2.4 * nrows), sharey=True,
                             squeeze=False)
    cover = {c["eval_month"]: c for c in coverage}
    tags = regimes.drop_duplicates("eval_month").set_index("eval_month")
    dense = len(order) > 60
    for k, month in enumerate(months):
        ax = axes[k // cols, k % cols]
        part = rows[rows["eval_month"] == month]
        x = part["row_id"].map(x_of).to_numpy()
        for weighting in _weightings(part):
            mask = (part["weighting"] == weighting).to_numpy() if weighting is not None \
                else np.ones(len(part), dtype=bool)
            color = _weighting_color(mode, weighting) if weighting is not None \
                else INK[mode]["secondary"]
            if f"{metric}_low" in part and not dense:
                ax.vlines(x[mask], part[f"{metric}_low"][mask], part[f"{metric}_high"][mask],
                          color=color, alpha=0.3, linewidth=1)
            ax.scatter(x[mask], part[metric][mask], s=8 if dense else 16, color=color,
                       edgecolor=INK[mode]["surface"], linewidth=0.5, zorder=3,
                       label=LABELS.get(weighting, str(weighting)) if k == 0 else None)
        ax.axhline(0, color=INK[mode]["axis"], linewidth=1)
        ax.set_xticks([])
        ax.set_title(month, loc="left")
        note = (f"vol {tags.loc[month, 'realized_vol_bps_month_tercile']}, "
                f"spread {tags.loc[month, 'half_spread_bps_month_tercile']}")
        if cover[month]["missing_days"]:
            note += f", {len(cover[month]['missing_days'])} days missing"
        ax.text(1.0, 1.02, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=INK[mode]["secondary"])
        if k % cols == 0:
            ax.set_ylabel(metric.replace("_", " "))
        if k // cols == nrows - 1:
            ax.set_xlabel("configurations, by mean across months")
    for k in range(len(months), nrows * cols):
        axes[k // cols, k % cols].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.0 + 0.06 / nrows))
    fig.suptitle(f"{title}: {metric.replace('_', ' ')} by evaluation month",
                 y=1.0 + 0.12 / nrows)
    fig.tight_layout()
    path = _save(fig, out, "wf_months", mode)
    plt.close(fig)
    return path


def trajectories(rows, summary, metric: str, mode: str, out: Path, title: str) -> Path:
    """Metric by month: every configuration thin, each construction's median bold.

    One panel per (forecast, horizon) group when there are at most nine,
    otherwise one panel over every row. A panel with more than
    ``max_lines`` configurations draws the interquartile band of each
    construction instead of its lines. Stress and holdout months are shaded.
    """
    from experiments.figures import INK, LABELS, _plt

    max_lines = 120
    plt = _plt(mode)
    months = sorted(rows["eval_month"].unique())
    group_of = dict(zip(summary["row_id"], summary["group"]))
    rows = rows.assign(group=rows["row_id"].map(group_of))
    groups = list(dict.fromkeys(rows["group"]))
    panels = [(g, rows[rows["group"] == g]) for g in groups] if len(groups) <= 9 \
        else [("all rows", rows)]
    cols = min(3, len(panels))
    nrows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 2.5 * nrows), squeeze=False)
    x = np.arange(len(months))
    for k, (group, part) in enumerate(panels):
        ax = axes[k // cols, k % cols]
        for i, month in enumerate(months):
            if segment_of(month) in ("stress", "holdout"):
                ax.axvspan(i - 0.5, i + 0.5, color=INK[mode]["grid"], alpha=0.8, zorder=0)
        lines = part["row_id"].nunique() <= max_lines
        for weighting in _weightings(part):
            sub = part[part["weighting"] == weighting] if weighting is not None else part
            color = _weighting_color(mode, weighting) if weighting is not None \
                else INK[mode]["secondary"]
            wide = sub.pivot_table(index="row_id", columns="eval_month", values=metric,
                                   aggfunc="first", dropna=False).reindex(columns=months)
            if lines:
                for values in wide.to_numpy(dtype=np.float64):
                    ax.plot(x, values, color=color, alpha=0.3, linewidth=0.8, marker=None)
            else:
                ax.fill_between(x, wide.quantile(0.25).to_numpy(),
                                wide.quantile(0.75).to_numpy(), color=color, alpha=0.15,
                                linewidth=0)
            ax.plot(x, wide.median().to_numpy(), marker="o", color=color,
                    label=LABELS.get(weighting, str(weighting)))
        ax.axhline(0, color=INK[mode]["axis"], linewidth=1)
        ax.set_xticks(x, months, rotation=60 if len(months) > 4 else 0, fontsize=7)
        ax.set_title(group[:48], fontsize=8)
        if k % cols == 0:
            ax.set_ylabel(metric.replace("_", " "))
    for k in range(len(panels), nrows * cols):
        axes[k // cols, k % cols].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1),
               bbox_to_anchor=(0.5, 1.0 + 0.06 / nrows))
    fig.suptitle(f"{title}: each configuration by month, bold the median of each construction "
                 "(shaded: stress and holdout)", y=1.0 + 0.13 / nrows)
    fig.tight_layout()
    path = _save(fig, out, "wf_trajectories", mode)
    plt.close(fig)
    return path


def regime_bars(regime_rows, summary, mode: str, out: Path, title: str) -> Path:
    """Pooled daily net Sharpe by day tercile of each regime variable.

    Bars are the median over each construction's configurations, one row of
    panels per EFQ when a spec sweeps a few (cost is what turns a spread
    regime into a loss, and pooling EFQs would blur exactly that); terciles
    are ordinal, so they take the blue magnitude ramp, not series colours.
    """
    from experiments.figures import INK, LABELS, SEQUENTIAL, _plt

    plt = _plt(mode)
    keep = [c for c in ("weighting", "efq") if c in summary]
    part = regime_rows[regime_rows["unit"] == "day"].merge(
        summary[["row_id", *keep]], on="row_id", how="left")
    levels = sorted(part["efq"].dropna().unique()) if "efq" in part else []
    levels = levels if 2 <= len(levels) <= 4 else [None]
    names = {"realized_vol_bps": "realized volatility", "half_spread_bps": "quoted half-spread",
             "dispersion_bps": "cross-sectional dispersion"}
    weightings = _weightings(summary)
    fig, axes = plt.subplots(len(levels), len(REGIME_VARIABLES),
                             figsize=(3.4 * len(REGIME_VARIABLES), 2.6 * len(levels)),
                             sharey="row", squeeze=False)
    # More of the variable is more contrast with the surface in either mode.
    shades = (SEQUENTIAL[1], SEQUENTIAL[3], SEQUENTIAL[5]) if mode == "light" \
        else (SEQUENTIAL[5], SEQUENTIAL[3], SEQUENTIAL[1])
    width = 0.8 / len(TERCILES)
    for i, level in enumerate(levels):
        at = part if level is None else part[np.isclose(part["efq"], level)]
        for j, variable in enumerate(REGIME_VARIABLES):
            ax = axes[i, j]
            sub = at[at["variable"] == variable]
            for t, (name, shade) in enumerate(zip(TERCILES, shades)):
                values = []
                for weighting in weightings:
                    cell = sub[sub["tercile"] == name]
                    if weighting is not None:
                        cell = cell[cell["weighting"] == weighting]
                    values.append(cell["sharpe_daily_net"].median() if len(cell) else np.nan)
                ax.bar(np.arange(len(weightings)) + (t - 1) * width, values,
                       width=width * 0.92, color=shade,
                       label=f"{name} tercile" if i == 0 and j == 0 else None)
            ax.axhline(0, color=INK[mode]["axis"], linewidth=1)
            ax.set_xticks(range(len(weightings)),
                          [LABELS.get(w, str(w)) for w in weightings], fontsize=7)
            if i == 0:
                ax.set_title(f"by {names[variable]}")
            if j == 0:
                ax.set_ylabel("pooled daily net Sharpe" if level is None
                              else f"EFQ {level:g}\npooled daily net Sharpe")
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0 + 0.08 / len(levels)))
    fig.suptitle(f"{title}: net Sharpe in calm and stressed days (day terciles)",
                 y=1.0 + 0.15 / len(levels))
    fig.tight_layout()
    path = _save(fig, out, "wf_regimes", mode)
    plt.close(fig)
    return path


def draw_figures(name, rows, summary, regimes, regime_rows, coverage, metric) -> list[Path]:
    out = harness.RESULTS / f"{name}_wf" / "figures"
    made = []
    for mode in ("light", "dark"):
        made.append(small_multiples(rows, summary, regimes, coverage, metric, mode, out, name))
        made.append(trajectories(rows, summary, metric, mode, out, name))
        made.append(regime_bars(regime_rows, summary, mode, out, name))
    return made


# ── entry point ────────────────────────────────────────────────────────────

def walk_forward(spec_path: str, *, months: str | None = None, pairs: str | None = None,
                 name: str | None = None, workers: int = 4, aggregate_only: bool = False,
                 seal_holdout: bool = False, metric: str = "sharpe_daily_net",
                 figures: bool = True) -> dict:
    """Run a spec over its month pairs and aggregate them.

    ``months`` are EVALUATION months, each paired with the month before it as
    risk history; ``pairs`` gives ``HISTORY:EVAL`` pairs explicitly instead (a
    non-consecutive or non-causal pair is allowed, for smoke tests, and is
    marked as such in the summary).
    """
    spec = json.loads(Path(spec_path).read_text())
    name = name or spec["name"]
    if pairs:
        wanted = parse_pairs(pairs)
    elif months:
        wanted = [(shift_month(m, -1), m) for m in parse_months(months)]
    else:
        wanted = [(spec["history_month"], spec["eval_month"])]
    runnable, skipped = plan(spec, wanted, seal_holdout=seal_holdout)
    for s in skipped:
        print(f"skip {s['history_month']} -> {s['eval_month']}: {s['reason']}"
              + (f" ({', '.join(s['missing'])})" if s.get("missing") else ""), flush=True)
    for history, evaluation in runnable:
        if _month_index(history) >= _month_index(evaluation):
            print(f"WARNING: {history} -> {evaluation} uses future data as risk history; "
                  "smoke tests only", flush=True)
    if aggregate_only:
        runnable = [(h, e) for h, e in runnable
                    if (harness.RESULTS / f"{name}_wf" / e / "daily.npz").is_file()]
        for history, evaluation in runnable:
            if not _done(harness.RESULTS / f"{name}_wf" / evaluation,
                         month_spec(spec, name, history, evaluation)):
                print(f"WARNING: {name}_wf/{evaluation} was run from a different spec",
                      flush=True)
    else:
        for history, evaluation in runnable:
            run_month(spec, name, history, evaluation, workers)
    if not runnable:
        raise SystemExit("no month pair with both caches present (and results, with "
                         "--aggregate-only)")
    return aggregate(name, runnable, spec, metric=metric, skipped=skipped, figures=figures)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec")
    parser.add_argument("--months", help="evaluation months: 2019-08..2020-12, or A,B,C")
    parser.add_argument("--pairs", help="explicit HISTORY:EVAL pairs, comma separated")
    parser.add_argument("--name", help="results name (default: the spec's)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metric", default="sharpe_daily_net",
                        help="primary metric for orderings and the headline")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="do not run; aggregate the months already run")
    parser.add_argument("--seal-holdout", action="store_true",
                        help="skip the pre-declared holdout months (2020-11, 2020-12)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    walk_forward(args.spec, months=args.months, pairs=args.pairs, name=args.name,
                 workers=args.workers, aggregate_only=args.aggregate_only,
                 seal_holdout=args.seal_holdout, metric=args.metric,
                 figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
