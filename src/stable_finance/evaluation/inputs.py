"""Your data in, evaluation inputs out: the one interface a dataset needs.

Give :func:`evaluation_inputs` a table of observations -- a pandas DataFrame
or a mapping of columns -- with a timestamp, an optional asset column, and
whichever market columns you have, mapped to roles by a
:class:`~stable_finance.dataset.schema.FeatureSchema`. Say when decisions are
made and at which horizons, and it returns the realized
:class:`~stable_finance.evaluation.returns.contract.ForwardReturns` and the
observable :class:`~stable_finance.dataset.market.MarketData` on one
``(decision, asset)`` grid, ready for every stage::

    schema = FeatureSchema.from_roles(mid="px")            # one price column is enough
    realized, market = evaluation_inputs(frame, schema, decisions, horizons=[300, 900])

TIMESTAMPS say when an observation became known: the end of its bar for bar
data. Epoch seconds, ``datetime64`` or tz-aware pandas datetimes. Nothing
else about time is required -- no calendar, no bar length, no session.

NO LOOKAHEAD, by construction. The market at a decision is the last
observation stamped at or before it. A return's price at a decision is
measured over the observations stamped in ``(t, t + window]`` -- strictly
after the decision -- and its forward price over ``(t + h, t + h + window]``.
The price is the schema's richest (VWAP weighted by volume, else the
time-averaged midpoint, else the close) or the ``price`` you name; a window
with no observation has no price, and the return is NaN.

ANY LEVEL OF MARKET DATA (:data:`~stable_finance.dataset.market.MARKET_LEVELS`)
works. A bare price series is a ``mid``-level market; OHLCV bars, quotes and
their sizes come through as the roles they are mapped to. Bars carry no
spread, but ``spread_lookback`` estimates one from their high, low and close
over the bars before each decision (Abdi and Ranaldo, 2017), marked as
estimated, which lets costs be charged on bar data. The market also carries
the execution window's best buy and sell prices, ``window_best_buy`` and
``window_best_sell``, over the same ``(t, t + window]`` the return starts
from: the lowest ask and highest bid, else the lowest low and highest high,
else the price's own range. They are what a limit order is filled against.
``adv_lookback`` adds the average daily dollar volume over the bars before
each decision, which the ``impact`` execution models size orders against.
``vol_lookback`` adds what ``impact_stock`` also needs: each name's
idiosyncratic volatility and the market's volatility, from the daily
returns in the lookback, and the time trend from the decision's date. A
``market_cap`` role carries the firm's size through like any other column.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import MarketData, bar_half_spread
from stable_finance.dataset.schema import FeatureSchema, price_values, resolve_price
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = ["epoch_seconds", "evaluation_inputs", "to_grid"]


def epoch_seconds(values) -> NDArray[np.int64]:
    """Integer epoch seconds from ints, ``datetime64``, or pandas datetimes."""
    accessor = getattr(values, "dt", None)
    if accessor is not None and getattr(accessor, "tz", None) is not None:
        values = accessor.tz_convert("UTC").dt.tz_localize(None)
    array = np.asarray(values)
    if array.dtype.kind == "M":
        return array.astype("datetime64[s]").astype(np.int64)
    if array.dtype.kind in "iu":
        return array.astype(np.int64)
    if array.dtype.kind == "f" and np.all(np.isfinite(array)) and np.all(array == np.floor(array)):
        return array.astype(np.int64)
    raise TypeError(f"cannot read timestamps of dtype {array.dtype}")


def _window_prices(stamps, prices, weights, starts, window):
    """Weighted mean price over ``(start, start + window]`` for each start."""
    numerator = np.concatenate([[0.0], np.cumsum(np.nan_to_num(prices * weights, nan=0.0))])
    weight = np.concatenate([[0.0], np.cumsum(weights)])
    lo = np.searchsorted(stamps, starts, side="right")
    hi = np.searchsorted(stamps, starts + window, side="right")
    total = weight[hi] - weight[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (numerator[hi] - numerator[lo]) / total, np.nan)


def _range_extreme(values, lo, hi, reduce, fill):
    """``reduce`` over ``values[lo:hi]`` for every query, by a sparse table.

    Each query is two lookups into a table of power-of-two blocks, so a
    window per decision costs ``O(n log n)`` once rather than a pass each.
    An empty or all-NaN range is NaN.
    """
    out = np.full(len(lo), np.nan)
    if not len(values):
        return out
    table = [np.where(np.isfinite(values), values, fill)]
    while (1 << len(table)) <= len(values):
        half = 1 << (len(table) - 1)
        table.append(reduce(table[-1][:-half], table[-1][half:]))
    length = hi - lo
    ok = length > 0
    level = np.zeros(len(lo), dtype=np.int64)
    level[ok] = np.floor(np.log2(length[ok])).astype(np.int64)
    for k in np.unique(level[ok]):
        rows = ok & (level == k)
        out[rows] = reduce(table[k][lo[rows]], table[k][hi[rows] - (1 << k)])
    out[~np.isfinite(out)] = np.nan
    return out


def _estimated_half_spread(stamps, high, low, close, when, lookback):
    """Bar half-spread from the pairs of bars in ``(t - lookback, t]``."""
    terms = bar_half_spread(high, low, close)
    good = np.isfinite(terms)
    total = np.concatenate([[0.0], np.cumsum(np.where(good, terms, 0.0))])
    count = np.concatenate([[0], np.cumsum(good)])
    lo = np.searchsorted(stamps, when - lookback, side="right")
    hi = np.maximum(np.searchsorted(stamps, when, side="right") - 1, lo)  # pairs [lo, hi)
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, (total[hi] - total[lo]) / np.maximum(n, 1), np.nan)
    return np.sqrt(np.maximum(mean, 0.0)) / 2


def _daily_returns(stamps, prices):
    """Each UTC day with a close-to-close return: its number, last stamp, and return."""
    day = stamps // 86_400
    last = np.r_[day[1:] != day[:-1], True]
    days, ends, closes = day[last], stamps[last], prices[last]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = closes[1:] / closes[:-1] - 1.0
    return days[1:], ends[1:], returns


def _volatilities(daily, when, lookback, min_days: int = 20):
    """Idiosyncratic and market volatility, annualized in percent, as of each decision.

    Returns are aligned by day across names, and a day is known once the
    last of the names' closes for it is in, so no decision sees a day's
    cross-section before it is complete.
    """
    days = np.unique(np.concatenate([d for d, _, _ in daily]))
    panel = np.full((len(days), len(daily)), np.nan)
    ends = np.zeros(len(days), dtype=np.int64)
    for j, (d, e, r) in enumerate(daily):
        rows = np.searchsorted(days, d)
        panel[rows, j] = r
        np.maximum.at(ends, rows, e)
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


def _months_since_1926(when):
    """The paper's time trend: months since June 1926, which is 1."""
    dates = np.asarray(when, dtype="datetime64[s]").astype("datetime64[M]")
    return (dates - np.datetime64("1926-06", "M")).astype(np.int64) + 1


def _average_daily_dollar_volume(stamps, volume, price, when, lookback):
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


def evaluation_inputs(
    frame: Mapping,
    schema: FeatureSchema,
    decisions: ArrayLike,
    horizons: ArrayLike,
    *,
    time: str = "timestamp",
    asset: str | None = "ticker",
    price: str | None = None,
    window_seconds: int = 60,
    spread_lookback: int | None = None,
    adv_lookback: int | None = None,
    vol_lookback: int | None = None,
) -> tuple[ForwardReturns, MarketData]:
    """Realized forward returns and the market at each decision, from a table.

    ``decisions`` are the decision timestamps; ``horizons`` are seconds. The
    table's columns are read by the schema's column names; ``asset=None`` (or
    a table without that column) is a single series named ``"series"``. The
    market data carries every role the schema maps, so a stage that needs a
    column the table lacks is refused downstream rather than guessed at.

    ``spread_lookback`` (seconds) estimates the half-spread from the bars in
    the lookback before each decision; it needs ``high``, ``low`` and
    ``close`` and is refused when the table has quotes, whose spread is
    observed rather than estimated.

    ``adv_lookback`` (seconds) computes ``adv``, the average daily dollar
    volume: the dollar volume (volume times VWAP, else close, else mid) of
    the observations in the lookback before each decision, divided by the
    number of distinct UTC days observed in it. It needs ``volume``.

    ``vol_lookback`` (seconds) computes ``idio_vol`` and ``market_vol`` from
    the daily close-to-close returns of the price (last observation of each
    UTC day) over the lookback before each decision: the market is the
    equal-weighted mean of the names observed that day, idiosyncratic
    volatility is the residual of each name's daily returns on it, both
    annualized by 252 and in percent, and both NaN with fewer than twenty
    days. ``time_trend``, months since June 1926, comes from the decisions.
    """
    columns = list(frame.columns) if hasattr(frame, "columns") else list(frame.keys())
    missing = [c for c in (time, *schema.columns) if c not in columns]
    if missing:
        raise KeyError(f"table has no columns {missing}")
    price = resolve_price(schema, price)
    stamps = epoch_seconds(frame[time])
    values = np.column_stack([np.asarray(frame[c], dtype=np.float64) for c in schema.columns])
    labels = (np.asarray(frame[asset]).astype(str) if asset is not None and asset in columns
              else np.full(len(stamps), "series"))
    when = epoch_seconds(np.asarray(decisions))
    hs = np.asarray(horizons, dtype=np.int64).ravel()
    if when.ndim != 1 or not len(when) or not len(hs) or (hs <= 0).any():
        raise ValueError("decisions must be a non-empty 1-D array and horizons positive seconds")
    if window_seconds < 1:
        raise ValueError("window_seconds must be positive")
    if spread_lookback is not None:
        if schema.has("bid_price", "ask_price"):
            raise ValueError("the table has quotes, so its spread is observed; "
                             "spread_lookback estimates one only for bars")
        if not schema.has("high", "low", "close"):
            raise ValueError("spread_lookback estimates the spread from high, low and "
                             "close, which the schema does not all map")
        if spread_lookback < 1:
            raise ValueError("spread_lookback must be positive seconds")
    if adv_lookback is not None:
        if not schema.has("volume"):
            raise ValueError("adv_lookback needs a volume column")
        if adv_lookback < 1:
            raise ValueError("adv_lookback must be positive seconds")
    if vol_lookback is not None and vol_lookback < 1:
        raise ValueError("vol_lookback must be positive seconds")
    adv_price = next((r for r in ("vwap", "close", "mid") if schema.has(r)), None)
    if adv_lookback is not None and adv_price is None:
        raise ValueError("adv_lookback needs a price: vwap, close or mid")
    buy_role, sell_role = next(((b, s) for b, s in (("ask_price", "bid_price"), ("low", "high"))
                                if schema.has(b, s)), (None, None))
    assets = np.unique(labels)
    roles = tuple(schema.role_map)
    returns = np.full((len(when), len(assets), len(hs)), np.nan)
    market = np.full((len(roles), len(when), len(assets)), np.nan)
    best = np.full((2, len(when), len(assets)), np.nan)
    estimated = np.full((len(when), len(assets)), np.nan)
    adv = np.full((len(when), len(assets)), np.nan)
    daily = []                                   # per asset: (day end stamps, returns)
    role_columns = [schema.role_index(r) for r in roles]
    for j, name in enumerate(assets):
        rows = np.flatnonzero(labels == name)
        order = rows[np.argsort(stamps[rows], kind="stable")]
        ts, x = stamps[order], values[order]
        p = price_values(x, schema, price)
        w = (np.nan_to_num(x[:, schema.role_index("volume")], nan=0.0) if price == "vwap"
             else np.isfinite(p).astype(np.float64))
        base = _window_prices(ts, p, w, when, window_seconds)
        for k, h in enumerate(hs):
            forward = _window_prices(ts, p, w, when + h, window_seconds)
            with np.errstate(invalid="ignore", divide="ignore"):
                returns[:, j, k] = np.where(base != 0, forward / base - 1.0, np.nan)
        lo = np.searchsorted(ts, when, side="right")
        hi = np.searchsorted(ts, when + window_seconds, side="right")
        buy, sell = ((p, p) if buy_role is None else
                     (x[:, schema.role_index(buy_role)], x[:, schema.role_index(sell_role)]))
        best[0, :, j] = _range_extreme(buy, lo, hi, np.minimum, np.inf)
        best[1, :, j] = _range_extreme(sell, lo, hi, np.maximum, -np.inf)
        if vol_lookback is not None:
            daily.append(_daily_returns(ts, p))
        if adv_lookback is not None:
            adv[:, j] = _average_daily_dollar_volume(
                ts, x[:, schema.role_index("volume")], x[:, schema.role_index(adv_price)],
                when, adv_lookback)
        if spread_lookback is not None:
            estimated[:, j] = _estimated_half_spread(
                ts, *(x[:, schema.role_index(r)] for r in ("high", "low", "close")),
                when, spread_lookback)
        last = np.searchsorted(ts, when, side="right") - 1
        seen = last >= 0
        if roles:
            market[:, seen, j] = x[last[seen]][:, role_columns].T
    realized = ForwardReturns(returns, when, assets, hs)
    columns = {role: market[k] for k, role in enumerate(roles)}
    columns["window_best_buy"], columns["window_best_sell"] = best
    if spread_lookback is not None:
        columns["half_spread"] = estimated
    if adv_lookback is not None:
        columns["adv"] = adv
    if vol_lookback is not None:
        columns["idio_vol"], columns["market_vol"] = _volatilities(daily, when, vol_lookback)
        columns["time_trend"] = np.broadcast_to(_months_since_1926(when)[:, None],
                                                (len(when), len(assets))).copy()
    data = MarketData(columns, when, assets,
                      frozenset({"half_spread"}) if spread_lookback is not None else frozenset())
    return realized, data


def to_grid(
    values: ArrayLike,
    decisions: ArrayLike,
    assets: ArrayLike,
    *,
    decision_axis: ArrayLike,
    asset_axis: ArrayLike,
) -> NDArray[np.float64]:
    """Scatter flat rows onto ``(n_decisions, n_assets, n_columns)``.

    Cells that no row fills stay NaN, which every downstream stage reads as
    "not observed" rather than as a zero return or a free trade. Rows whose
    decision or asset is outside the requested axes are dropped.
    """
    flat = np.asarray(values, dtype=np.float64)
    if flat.ndim == 1:
        flat = flat[:, None]
    decision_of = {key: i for i, key in enumerate(np.asarray(decision_axis).tolist())}
    asset_of = {key: i for i, key in enumerate(np.asarray(asset_axis).tolist())}
    rows = np.array([decision_of.get(key, -1)
                     for key in np.asarray(decisions).tolist()])
    cols = np.array([asset_of.get(key, -1) for key in np.asarray(assets).tolist()])
    keep = (rows >= 0) & (cols >= 0)
    grid = np.full((len(decision_of), len(asset_of), flat.shape[1]), np.nan)
    grid[rows[keep], cols[keep]] = flat[keep]
    return grid
