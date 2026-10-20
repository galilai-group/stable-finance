"""Your data in, evaluation inputs out: the one interface a dataset needs.

Give :func:`evaluation_inputs` a table of observations -- a pandas DataFrame
or a mapping of columns -- with a timestamp, an optional asset column, and
whichever market columns you have, mapped to roles by a
:class:`~stable_finance.dataset.schema.FeatureSchema`. Say when decisions are
made and at which horizons, and it returns the realized
:class:`~stable_finance.evaluation.returns.contract.ForwardReturns` and the
observable :class:`~stable_finance.evaluation.market_data.MarketData` on one
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
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.schema import FeatureSchema, price_values, resolve_price
from stable_finance.evaluation.market_data import MarketData
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
) -> tuple[ForwardReturns, MarketData]:
    """Realized forward returns and the market at each decision, from a table.

    ``decisions`` are the decision timestamps; ``horizons`` are seconds. The
    table's columns are read by the schema's column names; ``asset=None`` (or
    a table without that column) is a single series named ``"series"``. The
    market data carries every role the schema maps, so a stage that needs a
    column the table lacks is refused downstream rather than guessed at.
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
    assets = np.unique(labels)
    roles = tuple(schema.role_map)
    returns = np.full((len(when), len(assets), len(hs)), np.nan)
    market = np.full((len(roles), len(when), len(assets)), np.nan)
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
        last = np.searchsorted(ts, when, side="right") - 1
        seen = last >= 0
        if roles:
            market[:, seen, j] = x[last[seen]][:, role_columns].T
    realized = ForwardReturns(returns, when, assets, hs)
    data = MarketData({role: market[k] for k, role in enumerate(roles)}, when, assets)
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
