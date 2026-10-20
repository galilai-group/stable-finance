"""Small stand-ins shared by tests in several folders."""

from pathlib import Path

import numpy as np

from stable_finance.dataset import CrossSectionalTargetBundle


class IdentityStats:
    """A target table that standardizes nothing: every transform is the raw value."""

    def transform(self, raw, *_args):
        values = np.asarray(raw)
        return CrossSectionalTargetBundle(values, values, values, values)


def market1t_features(rng, n_rows, seed_price):
    """A plausible dense session: positive quotes, bid < ask, integer sizes."""
    steps = rng.normal(0, 0.01, n_rows).cumsum()
    mid = seed_price * np.exp(steps)
    half = 0.01 + 0.002 * rng.random(n_rows)
    f = np.empty((n_rows, 9), dtype=np.float32)
    f[:, 0] = mid - half                                   # bid
    f[:, 4] = mid + half                                   # ask
    f[:, 7] = rng.poisson(30, n_rows) * (rng.random(n_rows) < 0.6)   # volume
    f[:, 8] = np.minimum(f[:, 7], rng.poisson(3, n_rows))  # n
    f[:, 1] = np.where(f[:, 7] > 0, mid + rng.normal(0, 0.003, n_rows), np.nan)
    # The dense mosaic forward-fills vwap over zero-volume seconds.
    idx = np.arange(n_rows); src = np.where(np.isnan(f[:, 1]), 0, idx)
    np.maximum.accumulate(src, out=src); f[:, 1] = f[src, 1]
    if np.isnan(f[0, 1]):
        f[:, 1] = np.where(np.isnan(f[:, 1]), mid, f[:, 1])
    f[:, 2] = np.maximum(f[:, 1], mid) + 0.001               # high
    f[:, 3] = np.minimum(f[:, 1], mid) - 0.001               # low
    f[:, 5] = rng.poisson(200, n_rows) + 1                   # bid_size
    f[:, 6] = rng.poisson(200, n_rows) + 1                   # ask_size
    return f


def market1t_store(root, dates=("2023-01-03",), n_tickers: int = 24, *, seed: int = 0,
                   late=(), objective=None) -> list[Path]:
    """A small real Market-1T day store, written from synthetic dense sessions.

    ``late`` names tickers whose grid starts 5,000 rows in. Enough names (24)
    for a cross-section to clear MIN_NAMES.
    """
    from stable_finance.dataset.datasets.market1t.daystore import (
        build_day, month_dir, write_day, write_month_index,
    )
    from stable_finance.dataset.datasets.market1t.spec import timeline_bounds_est

    rng = np.random.default_rng(seed)
    root, paths, entries = Path(root), [], {}
    tickers = [f"T{k:02d}" for k in range(n_tickers)]
    for d, date in enumerate(dates):
        ts_open, ts_close = timeline_bounds_est(date)
        records = []
        for k, ticker in enumerate(tickers):
            first = 5_000 if ticker in late else 0
            records.append({"ticker": ticker, "date": date, "grid_start": ts_open + first,
                            "features": market1t_features(rng, ts_close - ts_open - first,
                                                          20.0 + k + d)})
        arrays = build_day(records, date, None, objective=objective)
        mdir = month_dir(root, date[:7]); mdir.mkdir(parents=True, exist_ok=True)
        entries.setdefault(mdir, []).append(write_day(mdir / date, arrays, compress=False))
        paths.append(mdir / date)
    for mdir, month_entries in entries.items():
        write_month_index(mdir, month_entries)
    return paths
