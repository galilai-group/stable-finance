"""Month cache v2: everything v1 caches, plus what cost realism and order flow need.

Version 1 (:mod:`experiments.month_cache`) keeps what the first round of
experiments read at the 78 five-minute decisions: the store's VWAP forward
returns, the quotes at the decision and five seconds later, and twelve causal
features. Four questions it cannot answer need more of the day, and all of it
is only readable while the day is unpacked, so v2 reads it in the same pass:

* MIDQUOTE RETURNS. The store's returns run between 60-second VWAPs of
  TRADES. A trade prints at the bid or the ask, so trade-price returns carry
  bounce that a short-horizon reversal signal can mistake for alpha (Heston,
  Korajczyk and Sadka 2010); a one-minute VWAP averages most of it away, but
  prices the minute after the decision (about thirty seconds of implicit
  delay) and, by averaging, adds a little positive autocorrelation (Working
  1960). ``realized_mid`` is the midquote-to-midquote return over the same
  horizons, from the same decision row, the frictionless benchmark that the
  latency and spread costs are then charged against:
  ``mid[t + h] / mid[t] - 1``, with the midpoint defined exactly as
  :class:`~stable_finance.MarketData` defines it (a crossed, one-sided or
  zero-bid book has none). It follows the label's rule for which cells exist
  (a name quoted at the decision, a day with at least ``MIN_ROWS`` rows for
  it, a forward window that ends before the close), so a cell present in one
  panel and missing from the other is a data gap -- a VWAP window with no
  trade, a broken quote -- never a different definition. On 2020-08-12 every
  finite VWAP cell has a midquote return, and about 0.25% more cells have
  only the latter. That day it also equals, bit for bit, the library's own
  label code run with a one-second window at the midpoint
  (``Objective(price="mid", window_seconds=1)``), which differs from it only
  in keeping a crossed book's midpoint.
* LATENCY. v1's execution quotes are the touch five seconds after the
  decision, the default of
  :func:`~stable_finance.dataset.datasets.market1t.daystore.market_from_days`.
  v2 adds the touch 1, 5 and 30 seconds after (``exec_bid_price_1s`` ...
  ``exec_ask_price_30s``), read from the same row
  ``market_from_days(delay_seconds=d)`` reads, so latency becomes a
  sensitivity rather than a constant. The 5-second pair equals v1's exactly.
* CAPACITY. ``dollar_volume`` is each name's dollar volume over its whole
  day (volume times the bar VWAP, summed), and ``day_close`` its last quoted
  midpoint, both as ``market_from_days(lookback=...)`` computes them. Each
  decision row carries ITS DAY'S value, which is NOT causal on its own day:
  :mod:`experiments.run` builds the ``adv`` market column from the days
  strictly before each decision, never from this row directly.
* ORDER FLOW. ``ofi_60s`` and ``ofi_300s``: best-level order-flow imbalance
  (Cont, Kukanov and Stoikov 2014) summed over the last 60 and 300 seconds and
  divided by the average depth over the same window. With one-second bid and
  ask prices ``P`` and sizes ``q``, the event at second ``n`` is::

      e_n = 1{Pb_n >= Pb_n-1} qb_n - 1{Pb_n <= Pb_n-1} qb_n-1
          - 1{Pa_n <= Pa_n-1} qa_n + 1{Pa_n >= Pa_n-1} qa_n-1

  and ``OFI = sum e_n / mean((qb + qa) / 2)``. A second with either quote
  broken contributes nothing. The store has only the top of the book, so
  multi-level (integrated) OFI is not available.

FEATURES ARE CAUSAL, as in v1: every feature at a decision reads only rows at
or before the decision's own row. The store stamps each one-second bar at its
END, and the decision at offset ``a`` reads row ``a``.

V1 COMPATIBILITY. A v2 file holds every v1 array under its v1 name, computed by
the same calls with the same arguments (``market_from_days``,
``forward_returns_from_days``, ``month_cache.decision_features``), so its
``realized``, ``features`` and quotes equal the cached v1 files bit for bit
(checked by :func:`check_day`). One column needs care: the cached v1 files
were written by the builder's first version, whose half-spread feature is the
market's own, and v2 follows the files rather than today's
``decision_features`` (see :func:`build_day`). The OFI features are stored
apart (``ofi_features``) so ``features`` keeps v1's twelve columns;
:func:`load_month_v2` appends them.

Outputs, beside v1's under ``$STABLE_FINANCE_CACHE/experiments``:
``days-v2/YYYY-MM-DD.npz`` per day (written atomically, so a killed build
resumes at the day it stopped on) and ``market1t-v2-YYYY-MM.npz`` per month.

    uv run --extra hf --extra mds --extra experiments python -m experiments.month_cache_v2 2020-08
    # one day only, with the v1 comparison and the realized_mid sanity check:
    ... python -m experiments.month_cache_v2 --days 2020-08-12 --check

:mod:`experiments.months` builds every month of the store in order and deletes
each month's download once its cache is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from experiments.month_cache import (
    FEATURE_NAMES,
    QUANTITIES,
    ROOT,
    decision_features,
)
from experiments.month_cache import cache_dir as v1_cache_dir
from stable_finance import ForwardReturns, MarketData
from stable_finance.dataset.datasets.market1t.daystore import (
    FEATURES,
    MIN_ROWS,
    DayRecord,
    _closes_and_dollars,
    day_dir,
    decompress_day,
    forward_returns_from_days,
    market_from_days,
)
from stable_finance.dataset.datasets.market1t.spec import resolve_root

#: Execution delays, in seconds, at which the touch is cached. 5 is v1's.
LATENCIES = (1, 5, 30)

#: Order-flow windows, in seconds, in column order of ``ofi_features``.
OFI_WINDOWS = (60, 300)
OFI_NAMES = tuple(f"ofi_{w}s" for w in OFI_WINDOWS)

#: The v2-only arrays that sit on the (decision, asset) grid.
EXEC_QUANTITIES = tuple(f"exec_{side}_price_{d}s" for d in LATENCIES for side in ("bid", "ask"))
DAILY_QUANTITIES = ("dollar_volume", "day_close")

#: Tickers per block in the order-flow pass, which bounds its float64 scratch.
_BLOCK = 128


def cache_dir(out: str | Path | None = None) -> Path:
    """Where v2 files go: v1's directory unless ``out`` says otherwise."""
    if out is None:
        return v1_cache_dir()
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    return path


def month_path(month: str, out: str | Path | None = None) -> Path:
    return cache_dir(out) / f"market1t-v2-{month}.npz"


def store_root(month: str, root: str | Path | None = None) -> Path:
    """A local directory holding ``month`` of the day store.

    Resolved ONCE per month: a Hub root downloads the month (or checks the
    download) here, and every day is then read from the local copy, so a
    month costs one round of Hub metadata calls rather than one per day and a
    restart after the download needs no network.
    """
    return resolve_root(ROOT if root is None else root, f"{month}-01", f"{month}-31")


def trading_dates(month: str, root: str | Path | None = None) -> list[str]:
    """The month's trading days, from the store's month index."""
    index = store_root(month, root) / month[:4] / month[5:7] / "index.json"
    return [d["date"] for d in json.loads(index.read_text())["days"]]


def _save_atomic(target: Path, **arrays) -> Path:
    """``np.savez`` through a temporary file, so a killed write leaves no file.

    v1 writes in place, and a build killed mid-write leaves a truncated file
    that the next run takes for a finished one. Writing through a file handle
    also stops ``np.savez`` from appending ``.npz`` to the temporary name.
    """
    partial = target.with_name(target.name + ".partial")
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(partial, target)
    return target


# ── the v2 arrays of one day ────────────────────────────────────────────────

def _mid_at(bid: np.ndarray, ask: np.ndarray, quoted: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Midpoints at ``rows`` (any shape), in float64, NaN where the book is broken."""
    safe = np.clip(rows, 0, bid.shape[1] - 1)
    b = bid[:, safe].astype(np.float64)
    a = ask[:, safe].astype(np.float64)
    return np.where(quoted[:, safe], (b + a) / 2, np.nan)


def midquote_returns(record: DayRecord, bid: np.ndarray, ask: np.ndarray, quoted: np.ndarray,
                     horizons) -> np.ndarray:
    """``(anchors, tickers, horizons)`` midquote-to-midquote forward returns.

    Starts at the decision row, as the store's VWAP label does (its first
    window is ``[t, t + window)``), and ends at the row ``h`` seconds later.
    The availability rule is the label's (``_standard_target_arrays``): the
    name is quoted at the decision, its day has at least ``MIN_ROWS`` rows,
    and the label's forward window ``[t + h, t + h + window)`` ends inside
    the session.
    """
    bar = record.bar_seconds
    window = record.default_objective.window_seconds // bar
    t = (record.anchors - record.open_tod) // bar                       # (A,)
    h = np.asarray(horizons, dtype=np.int64) // bar                     # (H,)
    future = t[:, None] + h[None, :]                                    # (A, H)
    first = record.first_row.astype(np.int64)[:, None, None]            # (N, 1, 1)
    valid = ((t[None, :, None] >= first) & (record.rows - first >= MIN_ROWS)
             & ((future + window) <= record.rows)[None] & (future > t[:, None])[None])
    now = _mid_at(bid, ask, quoted, t)[:, :, None]                      # (N, A, 1)
    then = _mid_at(bid, ask, quoted, future)                            # (N, A, H)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(valid & (now > 0) & np.isfinite(then), then / now - 1.0, np.nan)
    return out.transpose(1, 0, 2).astype(np.float32)


def execution_quotes(record: DayRecord, bid: np.ndarray, ask: np.ndarray) -> dict[str, np.ndarray]:
    """The touch ``d`` seconds after each decision, for every ``d`` in :data:`LATENCIES`.

    The row and the masking are ``market_from_days``'s: the first row at or
    after arrival, NaN for a name not yet quoted there or an arrival past the
    close, and the raw quote otherwise (a broken book is the fill model's to
    refuse, as it is for v1's pair).
    """
    bar = record.bar_seconds
    out = {}
    for delay in LATENCIES:
        arrival = -(-(record.anchors + delay - record.open_tod) // bar)
        inside = (arrival >= 0) & (arrival < record.rows)
        safe = np.clip(arrival, 0, record.rows - 1)
        live = (record.first_row[:, None] <= safe[None, :]) & inside[None, :]
        out[f"exec_bid_price_{delay}s"] = np.where(live, bid[:, safe], np.nan).T.astype(np.float32)
        out[f"exec_ask_price_{delay}s"] = np.where(live, ask[:, safe], np.nan).T.astype(np.float32)
    return out


def order_flow(record: DayRecord, bid: np.ndarray, ask: np.ndarray, bid_size: np.ndarray,
               ask_size: np.ndarray, quoted: np.ndarray) -> np.ndarray:
    """``(anchors, tickers, len(OFI_WINDOWS))`` depth-normalized best-level OFI.

    Each window ends at the decision row and covers the events of its last
    ``w`` seconds, ``(t - w, t]``; the average depth is over the quoted rows
    of the same span. A window that would start before the open is NaN, as
    v1's lagged features are, and so is a name not quoted at the decision.
    Event sums run in float64 over blocks of tickers, which keeps the scratch
    to a few tens of megabytes.
    """
    bar = record.bar_seconds
    t = (record.anchors - record.open_tod) // bar
    n, rows = bid.shape
    out = np.full((len(t), n, len(OFI_WINDOWS)), np.nan, dtype=np.float32)
    t_safe = np.clip(t, 0, rows - 1)
    for lo in range(0, n, _BLOCK):
        hi = min(lo + _BLOCK, n)
        pb, pa = bid[lo:hi], ask[lo:hi]
        qb = np.nan_to_num(bid_size[lo:hi].astype(np.float64))
        qa = np.nan_to_num(ask_size[lo:hi].astype(np.float64))
        ok = quoted[lo:hi]
        both = ok[:, 1:] & ok[:, :-1]
        with np.errstate(invalid="ignore"):
            event = ((pb[:, 1:] >= pb[:, :-1]) * qb[:, 1:] - (pb[:, 1:] <= pb[:, :-1]) * qb[:, :-1]
                     - (pa[:, 1:] <= pa[:, :-1]) * qa[:, 1:] + (pa[:, 1:] >= pa[:, :-1]) * qa[:, :-1])
        event = np.where(both, event, 0.0)
        # cum[:, k] is the sum over rows < k; the event of row n sits at column n.
        zeros = np.zeros((hi - lo, 1))
        cum_event = np.concatenate([zeros, zeros, np.cumsum(event, axis=1)], axis=1)
        depth = np.where(ok, (qb + qa) / 2, 0.0)
        cum_depth = np.concatenate([zeros, np.cumsum(depth, axis=1)], axis=1)
        cum_count = np.concatenate([zeros, np.cumsum(ok, axis=1, dtype=np.float64)], axis=1)
        live_now = ok[:, t_safe] & (t >= 0)[None, :]                       # (block, A)
        for k, w in enumerate(OFI_WINDOWS):
            start = t_safe - w + 1                                         # first row in the window
            fits = (t - w >= 0)[None, :] & live_now
            s = np.clip(start, 0, rows)
            flow = cum_event[:, t_safe + 1] - cum_event[:, s]
            count = cum_count[:, t_safe + 1] - cum_count[:, s]
            with np.errstate(invalid="ignore", divide="ignore"):
                average = (cum_depth[:, t_safe + 1] - cum_depth[:, s]) / count
                value = np.where(fits & (count > 0) & (average > 0), flow / average, np.nan)
            out[:, lo:hi, k] = value.T
    return out


def v2_day_arrays(record: DayRecord, horizons) -> dict[str, np.ndarray]:
    """Every v2-only array of one day, on the record's own ticker order."""
    schema = record.schema
    take = lambda role: np.asarray(record.features[:, schema.role_index(role), :],
                                   dtype=np.float32)
    bid, ask = take("bid_price"), take("ask_price")
    live = np.arange(record.rows)[None, :] >= record.first_row[:, None]
    with np.errstate(invalid="ignore"):
        quoted = live & np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    out = {"realized_mid": midquote_returns(record, bid, ask, quoted, horizons)}
    out.update(execution_quotes(record, bid, ask))
    out["ofi_features"] = order_flow(record, bid, ask, take("bid_size"), take("ask_size"), quoted)
    del bid, ask, quoted, live
    closes, dollars = _closes_and_dollars(record)
    anchors = len(record.anchors)
    out["dollar_volume"] = np.broadcast_to(dollars, (anchors, record.n_tickers)).astype(np.float32)
    out["day_close"] = np.broadcast_to(closes, (anchors, record.n_tickers)).astype(np.float32)
    return out


# ── days and months ─────────────────────────────────────────────────────────

def build_day(date: str, out: Path, *, root: str | Path | None = None) -> tuple[Path, dict]:
    """Unpack one day, cache v1's arrays and v2's, delete the unpack.

    Returns the file and the seconds each stage took (zero for a day already
    cached). ``root`` is the day store (default: the Hub copy, resolved to its
    local download).
    """
    target = out / f"{date}.npz"
    if target.is_file():
        return target, {}
    clock = time.perf_counter()
    timings = {}

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name] = now - clock
        clock = now

    day = day_dir(store_root(date[:7], root), date)
    decompress_day(day, remove_zst=False)
    lap("unpack")
    record = DayRecord(day)
    market = market_from_days([record])
    realized = forward_returns_from_days([record])
    features = decision_features(record)
    lap("v1")
    extra = v2_day_arrays(record, realized.horizons)
    lap("v2")
    by_ticker = {t: i for i, t in enumerate(record.tickers.astype(str))}
    columns = np.array([by_ticker[t] for t in market.assets.astype(str)])
    features = features[:, columns]
    # The cached v1 months were written by the first version of the builder,
    # which replaced the half-spread feature with the market's own (float64
    # (ask - bid) / 2 / mid, cast once). The committed decision_features
    # computes it in float32 and differs from those files by one unit in the
    # last place on about a quarter of the cells. v2 follows the files, which
    # every published number was computed from.
    features[:, :, FEATURE_NAMES.index("half_spread")] = market.get("half_spread")
    _save_atomic(target,
                 decisions=realized.decisions, assets=realized.assets,
                 horizons=realized.horizons,
                 realized=realized.values.astype(np.float32),
                 features=features,
                 **{q: market.get(q).astype(np.float32) for q in QUANTITIES},
                 **{k: v[:, columns] for k, v in extra.items()})
    record._features = None
    del record, market
    unpacked = Path(day) / FEATURES
    if unpacked.is_file():
        unpacked.unlink()
    lap("save")
    return target, timings


#: Arrays stacked onto the month's asset grid, in file order after the axes.
STACKED = ("realized", "features", *QUANTITIES, "realized_mid", *EXEC_QUANTITIES,
           *DAILY_QUANTITIES, "ofi_features")


def build_month(month: str, *, out: str | Path | None = None, root: str | Path | None = None,
                log=print) -> Path:
    """Every day of ``month`` cached, then stacked on the month's asset grid."""
    out = cache_dir(out)
    target = month_path(month, out)
    if target.is_file():
        return target
    days_dir = out / "days-v2"
    days_dir.mkdir(exist_ok=True)
    dates = trading_dates(month, root)
    if not dates:
        raise ValueError(f"{month}: the store's month index lists no days")
    paths = []
    for i, date in enumerate(dates):
        t0 = time.perf_counter()
        path, timings = build_day(date, days_dir, root=root)
        paths.append(path)
        stages = " ".join(f"{k} {v:.1f}s" for k, v in timings.items()) or "cached"
        log(f"{date} ({i + 1}/{len(dates)}) {time.perf_counter() - t0:.1f}s [{stages}]")
    stack_days(paths, target)
    return target


def stack_days(paths, target: Path) -> Path:
    """Stack day files onto the union of their assets, NaN where a name is absent."""
    parts = [np.load(p, allow_pickle=False) for p in paths]
    assets = np.asarray(sorted({str(a) for p in parts for a in p["assets"]}))
    index = {a: i for i, a in enumerate(assets.tolist())}

    def stack(key):
        blocks = []
        for p in parts:
            values = p[key]
            block = np.full((values.shape[0], len(assets), *values.shape[2:]), np.nan,
                            dtype=np.float32)
            block[:, [index[str(a)] for a in p["assets"]]] = values
            blocks.append(block)
        return np.concatenate(blocks)

    return _save_atomic(target,
                        decisions=np.concatenate([p["decisions"] for p in parts]),
                        assets=assets, horizons=parts[0]["horizons"],
                        feature_names=np.asarray(FEATURE_NAMES),
                        ofi_feature_names=np.asarray(OFI_NAMES),
                        latencies=np.asarray(LATENCIES),
                        **{key: stack(key) for key in STACKED})


def write_v1(month: str, *, out: str | Path | None = None) -> Path:
    """The v1 month file, cut from the v2 one.

    v2 holds every v1 array under its v1 name, so v1's file is a subset: an
    experiment on ``"cache": "v1"`` then runs on a month whose download is
    already gone. Same keys, same order, same bytes as v1's ``build_month``.
    """
    target = cache_dir(out) / f"market1t-{month}.npz"
    if target.is_file():
        return target
    data = np.load(month_path(month, out), allow_pickle=False)
    keys = ("decisions", "assets", "horizons", "feature_names", "realized", "features",
            *QUANTITIES)
    return _save_atomic(target, **{k: data[k] for k in keys})


def load_month_v2(month: str) -> dict:
    """A cached v2 month as harness inputs: v1's keys, plus v2's.

    v1's keys mean what they mean in :func:`experiments.month_cache.load_month`
    (``realized`` is still the store's VWAP return), except that ``features``
    carries the OFI columns after v1's twelve, named in ``feature_names``.
    Added: ``realized_mid`` (:class:`ForwardReturns`), ``exec`` mapping each
    latency in seconds to its ``(bid, ask)`` pair (0 is the decision's own
    quotes), and ``dollar_volume`` and ``day_close`` (each row its day's value,
    not causal within the day).
    """
    data = np.load(build_month(month), allow_pickle=False)
    decisions, assets, horizons = data["decisions"], data["assets"], data["horizons"]
    realized = ForwardReturns(data["realized"].astype(np.float64), decisions, assets, horizons)
    realized_mid = ForwardReturns(data["realized_mid"].astype(np.float64), decisions, assets,
                                  horizons)
    market = MarketData({q: data[q].astype(np.float64) for q in QUANTITIES
                         if q not in ("mid", "half_spread")}, decisions, assets)
    days = np.char.partition(decisions.astype(str), "|")[:, 0]
    execution = {0: (market.get("bid_price"), market.get("ask_price"))}
    for delay in data["latencies"].tolist():
        execution[int(delay)] = (data[f"exec_bid_price_{delay}s"].astype(np.float64),
                                 data[f"exec_ask_price_{delay}s"].astype(np.float64))
    names = (*data["feature_names"].astype(str), *data["ofi_feature_names"].astype(str))
    return {"realized": realized, "market": market,
            "features": np.concatenate([data["features"], data["ofi_features"]], axis=2),
            "feature_names": names, "days": days,
            "half_spread": data["half_spread"].astype(np.float64),
            "realized_mid": realized_mid, "exec": execution,
            "dollar_volume": data["dollar_volume"].astype(np.float64),
            "day_close": data["day_close"].astype(np.float64)}


# ── checks ──────────────────────────────────────────────────────────────────

def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal shapes, dtypes and values, NaN equal to NaN: equal bytes in practice."""
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


def check_day(path: str | Path, v1_path: str | Path | None = None) -> dict:
    """Compare a v2 day file with v1's, and sanity-check what v1 does not have.

    * Every v1 array must equal the v1 day cache's exactly, and the 5-second
      execution pair must equal v1's ``exec_`` pair.
    * ``realized_mid`` against ``realized`` (VWAP), per horizon: both finite on
      nearly the same cells, a correlation that rises with the horizon (the
      price and window differences are a fixed few basis points, the return
      grows like the square root of the horizon), and the lag-one
      autocorrelation of consecutive 5-minute returns under each. Midquote
      returns should be close to uncorrelated. A one-minute VWAP averages
      away most single-trade bounce, and averaging prices over a window
      itself adds a small positive autocorrelation (Working 1960), so the
      VWAP figure is not expected to be the more negative one.
    * Latency: the share of names whose touch moved between 1 and 30 seconds
      grows with the delay, and the 30-second touch's mean absolute midpoint
      drift is larger than the 1-second one.
    * OFI: share finite, and its correlation with the contemporaneous mid
      return over the same window (positive and large in Cont et al.).
    """
    v2 = np.load(path, allow_pickle=False)
    report: dict = {"file": str(path)}
    if v1_path is not None and Path(v1_path).is_file():
        v1 = np.load(v1_path, allow_pickle=False)
        report["v1_equal"] = {k: _same(v1[k], v2[k]) for k in v1.files}
        report["exec_5s_equal_v1"] = (_same(v1["exec_bid_price"], v2["exec_bid_price_5s"])
                                      and _same(v1["exec_ask_price"], v2["exec_ask_price_5s"]))
    vwap, mid = v2["realized"].astype(np.float64), v2["realized_mid"].astype(np.float64)
    rows = {}
    for k, h in enumerate(v2["horizons"].tolist()):
        a, b = vwap[:, :, k], mid[:, :, k]
        both = np.isfinite(a) & np.isfinite(b)
        rows[int(h)] = {
            "finite_vwap": int(np.isfinite(a).sum()), "finite_mid": int(np.isfinite(b).sum()),
            "finite_both": int(both.sum()),
            "corr": float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 2 else None,
            "sd_vwap_bp": float(np.std(a[both]) * 1e4), "sd_mid_bp": float(np.std(b[both]) * 1e4),
            "mean_abs_diff_bp": float(np.mean(np.abs(a[both] - b[both])) * 1e4),
        }
    report["realized_mid_vs_vwap"] = rows
    # Lag-one autocorrelation of consecutive non-overlapping 5-minute returns,
    # pooled over names: bid-ask bounce makes trade-price returns revert.
    first = int(np.flatnonzero(v2["horizons"] == 300)[0]) if 300 in v2["horizons"] else 0
    for name, panel in (("vwap", vwap[:, :, first]), ("mid", mid[:, :, first])):
        x, y = panel[:-1].ravel(), panel[1:].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        report[f"lag1_autocorr_5m_{name}"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    bid0, ask0 = v2["bid_price"].astype(np.float64), v2["ask_price"].astype(np.float64)
    mid0 = (bid0 + ask0) / 2
    drift = {}
    for d in LATENCIES:
        b, a = (v2[f"exec_{s}_price_{d}s"].astype(np.float64) for s in ("bid", "ask"))
        ok = np.isfinite(b) & np.isfinite(a) & np.isfinite(mid0) & (mid0 > 0)
        moved = (b != bid0) | (a != ask0)
        drift[d] = {"share_touch_moved": float(moved[ok].mean()),
                    "mean_abs_mid_drift_bp": float(np.mean(np.abs((b + a)[ok] / 2 / mid0[ok] - 1))
                                                   * 1e4)}
    report["latency"] = drift
    ofi = v2["ofi_features"].astype(np.float64)
    report["ofi"] = {name: {"finite": float(np.isfinite(ofi[:, :, k]).mean()),
                            "sd": float(np.nanstd(ofi[:, :, k]))}
                     for k, name in enumerate(OFI_NAMES)}
    feats, names = v2["features"].astype(np.float64), list(FEATURE_NAMES)
    # rev_5m is MINUS the last five minutes' mid return; OFI over the same
    # five minutes should move with that return, so with rev_5m against it.
    x, y = ofi[:, :, OFI_NAMES.index("ofi_300s")], -feats[:, :, names.index("rev_5m")]
    ok = np.isfinite(x) & np.isfinite(y)
    report["ofi"]["corr_ofi_300s_with_last_5m_mid_return"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    dv = v2["dollar_volume"][0].astype(np.float64)
    report["dollar_volume"] = {"names": int(np.isfinite(dv).sum()),
                               "median_musd": float(np.nanmedian(dv) / 1e6),
                               "max_musd": float(np.nanmax(dv) / 1e6),
                               "max_name": str(v2["assets"][int(np.nanargmax(dv))]),
                               "constant_within_day": bool(np.all(
                                   (v2["dollar_volume"] == v2["dollar_volume"][:1])
                                   | np.isnan(v2["dollar_volume"])))}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("months", nargs="*", help="YYYY-MM months to build")
    parser.add_argument("--days", nargs="+", default=(),
                        help="build only these YYYY-MM-DD day files (no month file)")
    parser.add_argument("--root", help="a local day store root instead of the Hub copy")
    parser.add_argument("--out", help="output directory (default $STABLE_FINANCE_CACHE/experiments)")
    parser.add_argument("--check", action="store_true",
                        help="after each day, compare with v1's day cache and print sanity stats")
    args = parser.parse_args(argv)
    if not args.months and not args.days:
        parser.error("name months or --days")
    out = cache_dir(args.out)
    for date in args.days:
        days_dir = out / "days-v2"
        days_dir.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        path, timings = build_day(date, days_dir, root=args.root)
        stages = " ".join(f"{k} {v:.1f}s" for k, v in timings.items()) or "cached"
        print(f"{date}: {path} {time.perf_counter() - t0:.1f}s [{stages}] "
              f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        if args.check:
            v1 = v1_cache_dir() / "days" / f"{date}.npz"
            print(json.dumps(check_day(path, v1), indent=1), flush=True)
    for month in args.months:
        path = build_month(month, out=args.out, root=args.root)
        print(f"{month}: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
