"""Raw forward outcomes from any dense session, by role rather than column.

Each target family needs some roles of the schema (:data:`TARGET_ROLES`):
``return`` needs a price (see :mod:`stable_finance.dataset.schema`),
``volatility_change`` a midpoint, and ``spread_change`` both sides of the
book. :func:`check_targets` says up front whether a schema can support a
request, and the builders call it, so a univariate series asked for a spread
change fails with the roles it lacks rather than an index error.

Every horizon and window is in seconds. A session stored at ``bar_seconds``
per row has them converted to rows, and they must be whole bars.
"""

from __future__ import annotations

import numpy as np

from stable_finance.dataset.anchors import (
    DEFAULT_ANCHOR_SPEC,
    RETURN_VWAP_WINDOW,
    TO_CLOSE,
    AnchorSpec,
)
from stable_finance.dataset.schema import (
    FeatureSchema,
    MissingRolesError,
    default_schema,
    has_mid,
    mid_series,
    price_cumsums,
    price_values,
    resolve_price,
    windowed_price,
)


_schema = default_schema


# Returns compare equally sized forward price windows at t and t+h. This avoids
# exposing a trailing or instantaneous baseline already visible in the model
# input. With the vwap price, volume weighting also excludes forward-filled
# prices during seconds with no trade.
def forward_price(features: np.ndarray, idx, window: int = RETURN_VWAP_WINDOW, *,
                  schema: FeatureSchema | None = None, price: str | None = None):
    """The price over ``[idx, idx + window)`` rows, by ``price`` method.

    ``idx`` may be a scalar or an integer array; the result matches its shape.
    NaN where the window runs past the end of the day or holds no weight.
    """
    schema = _schema(schema)
    cs_num, cs_weight = price_cumsums(features, schema, resolve_price(schema, price))
    out = windowed_price(cs_num, cs_weight, len(features), idx, window)
    return out if np.ndim(idx) else float(out)


def forward_vwap(features: np.ndarray, idx, window: int = RETURN_VWAP_WINDOW, *,
                 schema: FeatureSchema | None = None):
    """Volume-weighted trade price over ``[idx, idx + window)``.

    ``idx`` may be a scalar or an integer array; the result matches its shape.
    NaN where the window would run past the end of the day or holds no traded
    volume -- both mean "this row has no target", and the caller's existing
    NaN filter drops it.

    Shared by ``anchor_targets`` (eval, and the anchor-stats builder) and
    ``targets.compute_pair_targets`` (training) so the thing a model is trained
    on and the thing it is scored on cannot drift apart.
    """
    return forward_price(features, idx, window, schema=schema, price="vwap")


ANCHOR_TARGET_TYPES = ("return", "volatility_change", "spread_change")
PAIR_TARGET_TYPES = (
    "return",
    "spread_change",
    "volatility_change",
)


def target_roles(target_type: str, schema: FeatureSchema,
                 price: str | None = None) -> None:
    """Raise :class:`MissingRolesError` if ``schema`` cannot form ``target_type``."""
    if target_type == "return":
        resolve_price(schema, price)
    elif target_type == "spread_change":
        schema.require("bid_price", "ask_price", purpose="spread_change")
    elif target_type == "volatility_change":
        if not has_mid(schema):
            raise MissingRolesError(
                "volatility_change needs a mid column or both bid_price and "
                f"ask_price (roles: {schema.describe_roles()})")
    else:
        raise ValueError(f"unknown target type: {target_type!r}")


def check_targets(schema: FeatureSchema, types, price: str | None = None) -> None:
    """Validate a whole request: every family must be formable from ``schema``."""
    for target_type in types:
        target_roles(str(target_type), schema, price)


def supported_targets(schema: FeatureSchema, price: str | None = None) -> tuple[str, ...]:
    """The families in :data:`PAIR_TARGET_TYPES` that ``schema`` can form."""
    out = []
    for target_type in PAIR_TARGET_TYPES:
        try:
            target_roles(target_type, schema, price)
        except MissingRolesError:
            continue
        out.append(target_type)
    return tuple(out)


def get_target_names(
    horizons: list[int],
    types: list[str],
    rf_tickers: list[str] | None = None,
    *,
    window_seconds: int | None = None,
) -> list[str]:
    """Return names in the same type-major order as :func:`compute_pair_targets`.

    A ``window_seconds`` other than the standard one-minute measurement window
    is appended as ``@<seconds>``: ``return_900`` is measured over one minute
    at each end, ``return_900@300`` over five.
    """
    suffix = ("" if window_seconds is None or int(window_seconds) == RETURN_VWAP_WINDOW
              else f"@{int(window_seconds)}")
    names = [
        f"{target_type}_{horizon:03d}{suffix}"
        for target_type in types
        for horizon in horizons
    ]
    if rf_tickers:
        names.extend(
            f"return_adj__{ticker}__{horizon:03d}"
            for ticker in rf_tickers
            for horizon in horizons
        )
    return names


def anchor_indices(
    session_len: int | None = None,
    *,
    spec: AnchorSpec = DEFAULT_ANCHOR_SPEC,
) -> np.ndarray:
    """Row indices of the decision instants; seconds past the open at 1 Hz.

    ``session_len`` is in rows. :func:`anchor_offsets` gives the same instants
    in seconds, which is how stores and tables key them.
    """
    length = spec.session_rows if session_len is None else int(session_len)
    return np.arange(0, length, spec.step_rows, dtype=np.int64)


def anchor_offsets(spec: AnchorSpec = DEFAULT_ANCHOR_SPEC) -> np.ndarray:
    """The decision instants in seconds past the session origin."""
    return np.arange(0, spec.session_seconds, spec.step_seconds, dtype=np.int64)


def _nan_cumsums(r: np.ndarray):
    """Prefix sums of a NaN-holed series, for O(1) windowed nanstd."""
    ok = np.isfinite(r)
    rn = np.where(ok, r, 0.0)
    s1 = np.concatenate([[0.0], np.cumsum(rn)])
    s2 = np.concatenate([[0.0], np.cumsum(rn * rn)])
    cn = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    return s1, s2, cn


def window_spread(features: np.ndarray, idx: int, window: int = RETURN_VWAP_WINDOW, *,
                  schema: FeatureSchema | None = None):
    """Time-averaged quoted spread over ``[idx, idx + window)``.

    The scalar counterpart of the cumsum path in ``anchor_targets``; shared so
    the TRAINING target and the SCORING target cannot drift apart. Plain mean,
    not volume-weighted: the spread is a property of the book at each instant,
    not of the trades that happened to print.
    """
    schema = _schema(schema)
    bid, ask = schema.role_index("bid_price"), schema.role_index("ask_price")
    n = len(features)
    if idx < 0 or idx + window > n:
        return np.nan
    w = features[idx:idx + window]
    return float(np.nanmean(w[:, ask] - w[:, bid]))


def window_realized_volatility(
    features: np.ndarray,
    idx: int,
    window: int = RETURN_VWAP_WINDOW,
    *,
    schema: FeatureSchema | None = None,
):
    """Realized vol of the midpoint over ``[idx, idx + window)``.

    Population standard deviation (ddof=0) of the 1 Hz log returns, matching
    the vectorized implementation used by :func:`anchor_targets`.
    """
    n = len(features)
    if idx < 0 or idx + window > n:
        return np.nan
    mids = mid_series(features[idx:idx + window], _schema(schema))
    if len(mids) < 3:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(mids))
    return float(np.nanstd(r))


def _windowed_mean(cs, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Mean over ``[lo, hi)`` from a cumulative sum with a leading zero."""
    lo = np.clip(lo, 0, len(cs) - 1)
    hi = np.clip(hi, 0, len(cs) - 1)
    n = (hi - lo).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.where(n > 0, n, 1.0), np.nan)


def _vwap_cumsums(features: np.ndarray, schema: FeatureSchema | None = None):
    return price_cumsums(features, _schema(schema), "vwap")


_windowed_vwap = windowed_price


def resolve_future(
    indices: np.ndarray, horizons: np.ndarray, close_index: int | None
) -> np.ndarray:
    """Where each (anchor, horizon) measures its forward window.

    A fixed horizon measures at ``anchor + horizon``. The :data:`TO_CLOSE`
    sentinel measures at the closing auction instead, which is the same row
    for every anchor -- so the holding period shortens through the day exactly
    as a position held into the auction does, rather than the target becoming
    unmeasurable and quietly dropping the late anchors.
    """
    t = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
    hs = np.asarray(horizons, dtype=np.int64)
    future = t + hs[None, :]
    to_close = hs == TO_CLOSE
    if to_close.any():
        if close_index is None:
            raise ValueError(
                "a TO_CLOSE horizon needs close_index; pass the AnchorSpec "
                "whose session this is, and keep post_close_seconds >= the "
                "measurement window so the auction window is retained"
            )
        future = np.where(to_close[None, :], int(close_index), future)
    return future


def horizon_rows(horizons, bar_seconds: int = 1) -> np.ndarray:
    """Horizons in seconds as row offsets; :data:`TO_CLOSE` passes through."""
    hs = np.asarray(horizons, dtype=np.int64)
    bar = int(bar_seconds)
    if bar == 1:
        return hs
    fixed = hs != TO_CLOSE
    if (hs[fixed] % bar).any():
        raise ValueError(f"horizons {hs[fixed].tolist()} are not whole {bar}-second bars")
    return np.where(fixed, hs // bar, hs)


def _standard_target_arrays(
    features: np.ndarray,
    indices: np.ndarray,
    horizons: np.ndarray,
    types: tuple[str, ...],
    window: int,
    close_index: int | None = None,
    *,
    schema: FeatureSchema | None = None,
    price: str | None = None,
    bar_seconds: int = 1,
) -> dict[str, np.ndarray]:
    """Compute requested target families over all points and horizons.

    ``indices``, ``window`` and ``close_index`` are rows; ``horizons`` are
    seconds and are converted at ``bar_seconds`` per row.
    """
    unknown = set(types) - set(PAIR_TARGET_TYPES)
    if unknown:
        raise ValueError(f"unknown target types: {sorted(unknown)}")
    schema = _schema(schema)
    check_targets(schema, types, price)

    n = len(features)
    t = np.asarray(indices, dtype=np.int64)[:, None]
    hs = horizon_rows(horizons, bar_seconds)
    future = resolve_future(indices, hs, close_index)
    # An anchor at or past the measurement point has no forward window to
    # measure: at the close this is the last seconds of the session, where a
    # position could still be entered but not held for any length of time.
    valid_window = (t >= 0) & ((future + window) <= n) & (future > t)
    future_safe = np.where(valid_window, future, 0)
    result: dict[str, np.ndarray] = {}

    if "return" in types:
        cs_num, cs_weight = price_cumsums(features, schema, resolve_price(schema, price))
        base = windowed_price(
            cs_num, cs_weight, n, t[:, 0], window
        )[:, None]
        forward = windowed_price(
            cs_num, cs_weight, n, future_safe, window
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            result["return"] = np.where(
                valid_window & np.isfinite(base) & (base != 0),
                forward / np.where(base != 0, base, 1.0) - 1.0,
                np.nan,
            )

    if "spread_change" in types:
        spread = (
            features[:, schema.role_index("ask_price")].astype(np.float64)
            - features[:, schema.role_index("bid_price")].astype(np.float64)
        )
        cs = np.concatenate([[0.0], np.nancumsum(spread)])
        base = _windowed_mean(cs, t, t + window)
        forward = _windowed_mean(cs, future_safe, future_safe + window)
        result["spread_change"] = np.where(
            valid_window, forward - base, np.nan
        )

    if "volatility_change" in types:
        mid = mid_series(features, schema)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(mid))
        s1, s2, counts = _nan_cumsums(log_returns)

        base = _windowed_std(
            s1,
            s2,
            counts,
            np.broadcast_to(t, future.shape),
            np.broadcast_to(t + window - 1, future.shape),
        )
        forward = _windowed_std(
            s1,
            s2,
            counts,
            future_safe,
            future_safe + window - 1,
        )
        result["volatility_change"] = np.where(
            valid_window, forward - base, np.nan
        )

    return result


def _risk_factor_day(risk_factor: dict, date: str, index: int):
    date_index = risk_factor["date_to_idx"].get(date)
    if date_index is None:
        return None, 0
    values = risk_factor["features"][date_index]
    if index < 0 or index >= len(values):
        return None, 0
    return values, len(values)


def compute_pair_targets(
    focal_features: np.ndarray,
    t_idx: int,
    horizons: list[int],
    types: list[str],
    rf_data: dict[str, dict] | None = None,
    rf_price_mode: str | None = None,
    date_str: str | None = None,
    rf_t_idx: int | None = None,
    *,
    measurement_window: int = RETURN_VWAP_WINDOW,
    close_index: int | None = None,
    schema: FeatureSchema | None = None,
    price: str | None = None,
    bar_seconds: int = 1,
) -> np.ndarray:
    """Compute selected raw targets for one decision instant.

    Work is lazy by target family: VWAP, spread, and log-return prefix sums are
    built only when their corresponding target is requested. All requested
    horizons share those prefix sums.

    ``horizons`` may include :data:`TO_CLOSE`, which measures to the closing
    auction rather than to a fixed offset; ``close_index`` says which row that
    is, and the session must retain a measurement window past it.

    ``measurement_window`` and every horizon are seconds; ``t_idx`` and
    ``close_index`` are rows at ``bar_seconds`` per row.
    """
    schema = _schema(schema)
    type_order = tuple(types)
    if int(measurement_window) % int(bar_seconds):
        raise ValueError("measurement_window is not a whole number of bars")
    arrays = _standard_target_arrays(
        focal_features,
        np.asarray([t_idx]),
        horizons,
        type_order,
        int(measurement_window) // int(bar_seconds),
        close_index,
        schema=schema,
        price=price,
        bar_seconds=bar_seconds,
    )
    hs = horizon_rows(horizons, bar_seconds)
    output = [arrays[target_type][0] for target_type in type_order]

    if rf_data and rf_price_mode and date_str is not None and rf_t_idx is not None:
        if rf_price_mode not in {"mid", "vwap"}:
            raise ValueError("rf_price_mode must be 'mid', 'vwap', or None")
        n_focal = len(focal_features)
        focal_future = np.minimum(t_idx + hs, n_focal - 1)
        focal_mids = mid_series(focal_features[np.r_[t_idx, focal_future]], schema)
        focal_mid, focal_future_mid = focal_mids[0], focal_mids[1:]
        focal_gross = (
            focal_future_mid / focal_mid
            if focal_mid != 0
            else np.full(len(hs), np.nan)
        )

        for risk_factor in rf_data.values():
            values, n_risk_factor = _risk_factor_day(
                risk_factor, date_str, rf_t_idx
            )
            adjusted = np.full(len(hs), np.nan, dtype=np.float64)
            if values is not None:
                rf_future = np.minimum(rf_t_idx + hs, n_risk_factor - 1)
                matching = (focal_future - t_idx) == (rf_future - rf_t_idx)
                rf_prices = price_values(
                    values[np.r_[rf_t_idx, rf_future]], schema, rf_price_mode)
                rf_base, rf_forward = rf_prices[0], rf_prices[1:]
                if rf_base != 0:
                    adjusted[matching] = (
                        focal_gross[matching]
                        - rf_forward[matching] / rf_base
                    )
            output.append(adjusted)

    if not output:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(output).astype(np.float32, copy=False)


def _windowed_std(s1, s2, cn, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Population std (ddof=0, matching ``np.nanstd``) over ``r[lo:hi]``."""
    lo = np.clip(lo, 0, len(cn) - 1)
    hi = np.clip(hi, 0, len(cn) - 1)
    n = (cn[hi] - cn[lo]).astype(np.float64)
    a1 = s1[hi] - s1[lo]
    a2 = s2[hi] - s2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = a1 / n
        var = a2 / n - mean * mean
        out = np.sqrt(np.maximum(var, 0.0))
    # nanstd needs >= 2 points to be meaningful (1 diff is std 0, not signal).
    return np.where(n >= 2, out, np.nan)


def anchor_targets(
    features: np.ndarray,
    horizons,
    anchors: np.ndarray | None = None,
    *,
    types=ANCHOR_TARGET_TYPES,
    spec: AnchorSpec = DEFAULT_ANCHOR_SPEC,
    schema: FeatureSchema | None = None,
    price: str | None = None,
) -> np.ndarray:
    """Raw forward targets for one (ticker, date) at every anchor.

    Vectorized counterpart of ``targets.compute_pair_targets`` for the anchor
    grid, with ONE deliberate difference: no close clamp. Where that function
    uses ``fut_idx = min(t + h, N - 1)``, this emits NaN.

    Args:
        features: (N, C) raw rows at ``spec.bar_seconds``, ffilled, NOT
            normalized, with columns as ``schema`` (default Market-1T).
        horizons: forward horizons in seconds.
        anchors: row indices; defaults to the spec's anchor grid.
        types: requested target families. Unrequested families are not built.
        price: the return's price method; None is the schema's default.

    Returns:
        (A, T, H) float64, ``T`` ordered as ``types``.
    """
    if anchors is None:
        anchors = anchor_indices(len(features), spec=spec)
    anchors = np.asarray(anchors, dtype=np.int64)
    hs = np.asarray(list(horizons), dtype=np.int64)
    requested = tuple(types)
    unknown = set(requested) - set(ANCHOR_TARGET_TYPES)
    if unknown:
        raise ValueError(f"unknown anchor target types: {sorted(unknown)}")
    if not requested:
        raise ValueError("types cannot be empty")
    # anchor_targets owns a whole session, so it knows where the close is and
    # whether the tail to measure it was retained. A TO_CLOSE horizon against
    # a spec that did not keep that tail is a silent column of NaN otherwise.
    if (hs == TO_CLOSE).any() and not spec.measures_close:
        raise ValueError(
            "a TO_CLOSE horizon needs a session that retains the auction "
            f"window: spec keeps {spec.post_close_seconds}s past the close "
            f"but the measurement window is {spec.measurement_window_seconds}s"
            " (use CLOSING_AUCTION_SPEC)"
        )
    arrays = _standard_target_arrays(
        features,
        anchors,
        hs,
        requested,
        spec.window_rows,
        spec.close_index,
        schema=schema,
        price=price,
        bar_seconds=spec.bar_seconds,
    )
    return np.stack([arrays[name] for name in requested], axis=1)
