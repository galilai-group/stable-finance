"""One Market-1T month as a compact, reusable evaluation cache.

The day store ships each trading day as a compressed ``(tickers, 9, 23400)``
second-by-second array -- about 70 MB compressed and 600 MB unpacked -- so a
month unpacked is larger than a laptop's free disk. Every experiment here
needs only what is known AT the 78 five-minute decisions: the realized forward
returns the store already carries, the quotes at the decision and on arrival,
and a handful of causal features. So the month is streamed one day at a time:
unpack the day, read those quantities, delete the unpacked array, move on.
Peak disk is the compressed month plus one unpacked day.

The output is one ``.npz`` per month on the ``(decision, asset)`` grid that
:func:`~stable_finance.dataset.datasets.market1t.daystore.market_from_days`
uses -- decisions are ``"YYYY-MM-DD|anchor"`` labels, assets every ticker seen
in the month, sorted -- so a month loads straight into ``ForwardReturns`` and
``MarketData``.

FEATURES ARE CAUSAL. Every feature at a decision reads only rows at or before
the decision's own row: past mid-quote returns (5, 15 and 30 minutes, and the
last five one-minute returns), the quote imbalance over the last five
seconds, the relative half-spread, the last five minutes' volume against the
session's average so far, and the realized volatility of the last 30
minutes. Returns are the store's: between 60-second forward VWAPs, starting
strictly after the decision.

    uv run --extra hf --extra mds --extra experiments python -m experiments.month_cache 2020-08

Set ``STABLE_FINANCE_CACHE`` to choose where the download and the cache live.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from stable_finance import ForwardReturns, MarketData
from stable_finance.dataset.datasets.market1t.daystore import (
    FEATURES,
    DayRecord,
    discover_days,
    forward_returns_from_days,
    market_from_days,
)
from stable_finance.dataset.datasets.market1t.spec import resolve_root

ROOT = "hf://datasets/fin-ai-lab/Market-1T-1Hz-2019H2-2020-daystore/1Hz_daystore"

#: Market quantities kept per decision. ``mid`` and ``half_spread`` are
#: derived from the quotes; the ``exec_`` pair is the touch five seconds after
#: the decision, which the EFQ execution model fills against.
QUANTITIES = ("mid", "half_spread", "bid_price", "ask_price", "exec_bid_price",
              "exec_ask_price", "bid_size", "ask_size")

#: Decision-time features, in column order.
FEATURE_NAMES = ("rev_5m", "rev_15m", "rev_30m", "lag_1m_1", "lag_1m_2", "lag_1m_3",
                 "lag_1m_4", "lag_1m_5", "imbalance_5s", "half_spread", "volume_surprise",
                 "realized_vol_30m")


def cache_dir() -> Path:
    base = Path(os.environ.get("STABLE_FINANCE_CACHE", Path.home() / ".cache" / "stable-finance"))
    out = base / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    return out


def trading_dates(month: str) -> list[str]:
    """The month's trading days, from the store's month index."""
    start, end = f"{month}-01", f"{month}-31"
    local = resolve_root(ROOT, start, end)
    index = local / month[:4] / month[5:7] / "index.json"
    return [d["date"] for d in json.loads(index.read_text())["days"]]


def _ratio_return(now: np.ndarray, then: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((then > 0) & (now > 0), now / then - 1.0, np.nan)


def decision_features(record: DayRecord) -> np.ndarray:
    """``(anchors, tickers, features)`` causal features at each decision."""
    schema = record.schema
    take = lambda role: np.asarray(record.features[:, schema.role_index(role), :],
                                   dtype=np.float32)
    bid, ask = take("bid_price"), take("ask_price")
    bid_size, ask_size, volume = take("bid_size"), take("ask_size"), take("volume")
    live = np.arange(record.rows)[None, :] >= record.first_row[:, None]
    quoted = live & np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    mid = np.where(quoted, (bid + ask) / 2, np.nan).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        half_spread = np.where(quoted, (ask - bid) / (ask + bid), np.nan).astype(np.float32)
    del bid, ask

    with np.errstate(invalid="ignore", divide="ignore"):
        log_mid = np.log(mid)
    one_second = np.diff(log_mid, axis=1, prepend=np.nan)
    volume = np.where(live, volume, 0.0)
    cum_volume = np.cumsum(volume, axis=1, dtype=np.float64)

    n = record.n_tickers
    out = np.full((len(record.anchors), n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for a, anchor in enumerate(record.anchors):
        row = record.anchor_row(int(anchor))
        if not 0 <= row < record.rows:
            continue
        now = mid[:, row]
        f = out[a]
        for j, back in enumerate((300, 900, 1800)):
            if row - back >= 0:
                f[:, j] = -_ratio_return(now, mid[:, row - back])
        for lag in range(1, 6):
            hi, lo = row - 60 * (lag - 1), row - 60 * lag
            if lo >= 0:
                f[:, 2 + lag] = _ratio_return(mid[:, hi], mid[:, lo])
        lo = max(row - 4, 0)
        b = bid_size[:, lo:row + 1].mean(axis=1)
        s = ask_size[:, lo:row + 1].mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            f[:, 8] = np.where(b + s > 0, (b - s) / (b + s), np.nan)
        f[:, 9] = half_spread[:, row]
        if row >= 300:
            recent = cum_volume[:, row] - cum_volume[:, row - 300]
            average = cum_volume[:, row] / (row + 1) * 300
            with np.errstate(invalid="ignore", divide="ignore"):
                f[:, 10] = np.where(average > 0, recent / average, np.nan)
        if row >= 1800:
            window = one_second[:, row - 1799:row + 1]
            f[:, 11] = np.nanstd(window, axis=1) * np.sqrt(1800)
    return out


def build_day(date: str, out: Path) -> Path:
    """Unpack one day, cache what the experiments need, delete the unpack."""
    target = out / f"{date}.npz"
    if target.is_file():
        return target
    (day,) = discover_days(ROOT, date, date)
    record = DayRecord(day)
    market = market_from_days([record])
    realized = forward_returns_from_days([record])
    features = decision_features(record)
    by_ticker = {t: i for i, t in enumerate(record.tickers.astype(str))}
    columns = np.array([by_ticker[t] for t in market.assets.astype(str)])
    features = features[:, columns]
    np.savez(target,
             decisions=realized.decisions, assets=realized.assets,
             horizons=realized.horizons,
             realized=realized.values.astype(np.float32),
             features=features,
             **{q: market.get(q).astype(np.float32) for q in QUANTITIES})
    record._features = None
    del record, market
    unpacked = Path(day) / FEATURES
    if unpacked.is_file():
        unpacked.unlink()
    return target


def build_month(month: str) -> Path:
    """Every day of ``month`` cached, then stacked on the month's asset grid."""
    out = cache_dir()
    days_dir = out / "days"
    days_dir.mkdir(exist_ok=True)
    target = out / f"market1t-{month}.npz"
    if target.is_file():
        return target
    dates = trading_dates(month)
    parts = []
    for i, date in enumerate(dates):
        t0 = time.perf_counter()
        parts.append(np.load(build_day(date, days_dir), allow_pickle=False))
        print(f"{date} ({i + 1}/{len(dates)}) {time.perf_counter() - t0:.1f}s", flush=True)
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

    np.savez(target,
             decisions=np.concatenate([p["decisions"] for p in parts]),
             assets=assets, horizons=parts[0]["horizons"],
             feature_names=np.asarray(FEATURE_NAMES),
             realized=stack("realized"), features=stack("features"),
             **{q: stack(q) for q in QUANTITIES})
    return target


def load_month(month: str) -> dict:
    """A cached month as harness inputs: realized returns, market, features, days."""
    data = np.load(build_month(month), allow_pickle=False)
    decisions, assets, horizons = data["decisions"], data["assets"], data["horizons"]
    realized = ForwardReturns(data["realized"].astype(np.float64), decisions, assets, horizons)
    market = MarketData({q: data[q].astype(np.float64) for q in QUANTITIES
                         if q not in ("mid", "half_spread")}, decisions, assets)
    days = np.char.partition(decisions.astype(str), "|")[:, 0]
    return {"realized": realized, "market": market, "features": data["features"],
            "feature_names": tuple(data["feature_names"].astype(str)), "days": days,
            "half_spread": data["half_spread"].astype(np.float64)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("months", nargs="+", help="YYYY-MM")
    args = parser.parse_args(argv)
    for month in args.months:
        path = build_month(month)
        print(f"{month}: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
