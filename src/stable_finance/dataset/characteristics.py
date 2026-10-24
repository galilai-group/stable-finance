"""Trailing quantities the impact models read: dollar volume, volatilities, the time trend.

All of them are computed from daily closes and daily dollar volumes, so the
same functions serve a table of observations (through
:func:`~stable_finance.evaluation.inputs.evaluation_inputs`) and a day store
(through :func:`~stable_finance.dataset.datasets.market1t.daystore.market_from_days`).
Nothing here looks past a decision: a day enters a trailing window only once
the last name's close for it is in.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "average_daily_dollar_volume",
    "daily_returns",
    "months_since_1926",
    "trailing_from_days",
    "volatilities",
]

#: Daily returns needed before a volatility is reported. Fewer make the
#: residual standard deviation a draw rather than an estimate.
MIN_DAYS = 20


def daily_returns(stamps, prices):
    """Each UTC day with a close-to-close return: its number, last stamp, and return."""
    day = stamps // 86_400
    last = np.r_[day[1:] != day[:-1], True]
    days, ends, closes = day[last], stamps[last], prices[last]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = closes[1:] / closes[:-1] - 1.0
    return days[1:], ends[1:], returns


def volatilities(daily, when, lookback, min_days: int = MIN_DAYS):
    """Idiosyncratic and market volatility, annualized in percent, as of each decision.

    ``daily`` holds one ``(days, ends, returns)`` triple per name, as
    :func:`daily_returns` gives them; ``when`` and ``lookback`` are in the
    units of ``ends``. Returns are aligned by day across names, and a day is
    known once the last of the names' closes for it is in, so no decision
    sees a day's cross-section before it is complete. The market is the
    equal-weighted mean of the names observed each day; each name's
    idiosyncratic volatility is the residual of its daily returns on it.
    """
    days = np.unique(np.concatenate([d for d, _, _ in daily])) if daily else np.zeros(0, int)
    panel = np.full((len(days), len(daily)), np.nan)
    ends = np.zeros(len(days), dtype=np.int64)
    for j, (d, e, r) in enumerate(daily):
        rows = np.searchsorted(days, d)
        panel[rows, j] = r
        np.maximum.at(ends, rows, e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # an all-NaN day is NaN
        market = np.nanmean(panel, axis=1) if len(days) else np.zeros(0)
    lo = np.searchsorted(ends, when - lookback, side="right")
    hi = np.searchsorted(ends, when, side="right")
    idio = np.full((len(when), len(daily)), np.nan)
    mvol = np.full(len(when), np.nan)
    for t in range(len(when)):
        m = market[lo[t]:hi[t]]
        rows = panel[lo[t]:hi[t]]
        ok_m = np.isfinite(m)
        if ok_m.sum() >= min_days:
            mvol[t] = np.std(m[ok_m], ddof=1) * np.sqrt(252) * 100
        for j in range(len(daily)):
            r = rows[:, j]
            ok = np.isfinite(r) & ok_m
            if ok.sum() < min_days:
                continue
            x, y = m[ok], r[ok]
            beta = np.cov(x, y, ddof=1)[0, 1] / max(np.var(x, ddof=1), 1e-30)
            resid = y - y.mean() - beta * (x - x.mean())
            idio[t, j] = np.std(resid, ddof=2) * np.sqrt(252) * 100
    return idio, np.broadcast_to(mvol[:, None], idio.shape).copy()


def months_since_1926(when):
    """The paper's time trend: months since June 1926, which is 1."""
    dates = np.asarray(when, dtype="datetime64[s]").astype("datetime64[M]")
    return (dates - np.datetime64("1926-06", "M")).astype(np.int64) + 1


def average_daily_dollar_volume(stamps, volume, price, when, lookback):
    """Dollar volume in ``(t - lookback, t]`` per distinct UTC day observed in it."""
    dollars = np.nan_to_num(volume * price, nan=0.0)
    total = np.concatenate([[0.0], np.cumsum(dollars)])
    day = stamps // 86_400
    new_day = np.r_[True, day[1:] != day[:-1]] if len(day) else np.zeros(0, bool)
    days_seen = np.concatenate([[0], np.cumsum(new_day)])
    lo = np.searchsorted(stamps, when - lookback, side="right")
    hi = np.searchsorted(stamps, when, side="right")
    # Days in the window: those starting inside it, plus the one straddling its start.
    n_days = days_seen[hi] - days_seen[lo] + (lo < hi).astype(np.int64) * (
        ~new_day[np.minimum(lo, max(len(day) - 1, 0))] if len(day) else 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_days > 0, (total[hi] - total[lo]) / np.maximum(n_days, 1), np.nan)


def trailing_from_days(closes, dollars, lookback: int, min_days: int = MIN_DAYS
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``adv``, ``idio_vol`` and ``market_vol`` for each day of a day-by-day panel.

    ``closes`` and ``dollars`` are ``(days, names)``: each day's closing price
    and dollar volume per name, NaN where a name was not traded. Row ``r`` of
    each result uses only the ``lookback`` days before day ``r``, so the
    first row is NaN throughout and a decision on day ``r`` sees nothing of
    that day. Average daily dollar volume is the mean over the days in the
    window on which the name traded; the volatilities need ``min_days`` of
    daily returns in it.
    """
    closes = np.asarray(closes, dtype=np.float64)
    dollars = np.asarray(dollars, dtype=np.float64)
    if closes.ndim != 2 or dollars.shape != closes.shape:
        raise ValueError("closes and dollars must be (days, names) and alike")
    if lookback < 1:
        raise ValueError("lookback must be at least one day")
    n_days, n_names = closes.shape
    traded = np.isfinite(dollars) & (dollars > 0)
    total = np.concatenate([np.zeros((1, n_names)), np.cumsum(np.where(traded, dollars, 0.0), axis=0)])
    count = np.concatenate([np.zeros((1, n_names)), np.cumsum(traded, axis=0)])
    rows = np.arange(n_days)
    lo, hi = np.maximum(rows - lookback, 0), rows               # days [lo, hi) precede r
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        adv = np.where(n > 0, (total[hi] - total[lo]) / np.maximum(n, 1), np.nan)
    index = np.arange(n_days, dtype=np.int64)
    daily = []
    for j in range(n_names):
        seen = np.flatnonzero(np.isfinite(closes[:, j]))
        if len(seen) < 2:
            daily.append((np.zeros(0, int), np.zeros(0, int), np.zeros(0)))
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r = closes[seen[1:], j] / closes[seen[:-1], j] - 1.0
        daily.append((index[seen[1:]], index[seen[1:]], r))
    idio, mvol = volatilities(daily, rows - 1, lookback, min_days=min_days)
    return adv, idio, mvol
