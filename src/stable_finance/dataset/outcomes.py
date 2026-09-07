"""Raw forward outcomes for the supported one-second equity dataset."""

from __future__ import annotations

import numpy as np

from stable_finance.dataset.anchors import (
    DEFAULT_ANCHOR_SPEC,
    RETURN_VWAP_WINDOW,
    AnchorSpec,
)
from stable_finance.dataset.schema import MARKET_SCHEMA

# Resolve columns from the dataset schema so target semantics cannot drift when
# the physical column order changes.
_BID = MARKET_SCHEMA.index("bid_price")
_ASK = MARKET_SCHEMA.index("ask_price")
_VWAP = MARKET_SCHEMA.index("vwap_all")
_VOL = MARKET_SCHEMA.index("volume")

# Returns compare equally sized forward VWAP windows at t and t+h. This avoids
# exposing a trailing or instantaneous baseline already visible in the model
# input. Volume weighting also excludes forward-filled prices during seconds
# with no trade.
def forward_vwap(features: np.ndarray, idx, window: int = RETURN_VWAP_WINDOW):
    """Volume-weighted trade price over ``[idx, idx + window)``.

    ``idx`` may be a scalar or an integer array; the result matches its shape.
    NaN where the window would run past the end of the day or holds no traded
    volume -- both mean "this row has no target", and the caller's existing
    NaN filter drops it.

    Shared by ``anchor_targets`` (eval, and the anchor-stats builder) and
    ``targets.compute_pair_targets`` (training) so the thing a model is trained
    on and the thing it is scored on cannot drift apart.
    """
    cs_num, cs_volume = _vwap_cumsums(features)
    out = _windowed_vwap(cs_num, cs_volume, len(features), idx, window)
    return out if np.ndim(idx) else float(out)

ANCHOR_TARGET_TYPES = ("return", "volatility_change", "spread_change")
PAIR_TARGET_TYPES = (
    "return",
    "spread_change",
    "volatility",
    "volatility_change",
)


def get_target_names(
    horizons: list[int],
    types: list[str],
    rf_tickers: list[str] | None = None,
) -> list[str]:
    """Return names in the same type-major order as :func:`compute_pair_targets`."""
    names = [
        f"{target_type}_{horizon:03d}"
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
    """The 1 Hz row indices of the decision instants."""
    length = spec.session_seconds if session_len is None else int(session_len)
    return np.arange(0, length, spec.step_seconds, dtype=np.int64)


def _nan_cumsums(r: np.ndarray):
    """Prefix sums of a NaN-holed series, for O(1) windowed nanstd."""
    ok = np.isfinite(r)
    rn = np.where(ok, r, 0.0)
    s1 = np.concatenate([[0.0], np.cumsum(rn)])
    s2 = np.concatenate([[0.0], np.cumsum(rn * rn)])
    cn = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    return s1, s2, cn


def window_spread(features: np.ndarray, idx: int, window: int = RETURN_VWAP_WINDOW):
    """Time-averaged quoted spread over ``[idx, idx + window)``.

    The scalar counterpart of the cumsum path in ``anchor_targets``; shared so
    the TRAINING target and the SCORING target cannot drift apart. Plain mean,
    not volume-weighted: the spread is a property of the book at each instant,
    not of the trades that happened to print.
    """
    n = len(features)
    if idx < 0 or idx + window > n:
        return np.nan
    w = features[idx:idx + window]
    return float(np.nanmean(w[:, _ASK] - w[:, _BID]))


def window_realized_volatility(
    features: np.ndarray,
    idx: int,
    window: int = RETURN_VWAP_WINDOW,
):
    """Realized vol of the midpoint over ``[idx, idx + window)``.

    Population standard deviation (ddof=0) of the 1 Hz log returns, matching
    the vectorized implementation used by :func:`anchor_targets`.
    """
    n = len(features)
    if idx < 0 or idx + window > n:
        return np.nan
    w = features[idx:idx + window]
    mids = (w[:, _BID] + w[:, _ASK]) / 2.0
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


def _vwap_cumsums(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vw = features[:, _VWAP].astype(np.float64)
    volume = np.nan_to_num(features[:, _VOL].astype(np.float64), nan=0.0)
    numerator = np.nan_to_num(vw * volume, nan=0.0)
    return (
        np.concatenate([[0.0], np.cumsum(numerator)]),
        np.concatenate([[0.0], np.cumsum(volume)]),
    )


def _windowed_vwap(
    cs_num: np.ndarray,
    cs_volume: np.ndarray,
    length: int,
    idx,
    window: int,
):
    """Read one or many VWAP windows from shared prefix sums."""
    indices = np.asarray(idx, dtype=np.int64)
    lo, hi = indices, indices + int(window)
    valid = (lo >= 0) & (hi <= length)
    lo_safe = np.clip(lo, 0, length)
    hi_safe = np.clip(hi, 0, length)
    denominator = cs_volume[hi_safe] - cs_volume[lo_safe]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            valid & (denominator > 0),
            (cs_num[hi_safe] - cs_num[lo_safe])
            / np.where(denominator > 0, denominator, 1.0),
            np.nan,
        )


def _standard_target_arrays(
    features: np.ndarray,
    indices: np.ndarray,
    horizons: np.ndarray,
    types: tuple[str, ...],
    window: int,
) -> dict[str, np.ndarray]:
    """Compute requested target families over all points and horizons."""
    unknown = set(types) - set(PAIR_TARGET_TYPES)
    if unknown:
        raise ValueError(f"unknown target types: {sorted(unknown)}")

    n = len(features)
    t = np.asarray(indices, dtype=np.int64)[:, None]
    hs = np.asarray(horizons, dtype=np.int64)
    future = t + hs[None, :]
    valid_window = (t >= 0) & ((future + window) <= n)
    future_safe = np.where(valid_window, future, 0)
    result: dict[str, np.ndarray] = {}

    if "return" in types:
        cs_num, cs_volume = _vwap_cumsums(features)
        base = _windowed_vwap(
            cs_num, cs_volume, n, t[:, 0], window
        )[:, None]
        forward = _windowed_vwap(
            cs_num, cs_volume, n, future_safe, window
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            result["return"] = np.where(
                valid_window & np.isfinite(base) & (base != 0),
                forward / np.where(base != 0, base, 1.0) - 1.0,
                np.nan,
            )

    if "spread_change" in types:
        spread = (
            features[:, _ASK].astype(np.float64)
            - features[:, _BID].astype(np.float64)
        )
        cs = np.concatenate([[0.0], np.nancumsum(spread)])
        base = _windowed_mean(cs, t, t + window)
        forward = _windowed_mean(cs, future_safe, future_safe + window)
        result["spread_change"] = np.where(
            valid_window, forward - base, np.nan
        )

    if "volatility" in types or "volatility_change" in types:
        mid = (
            features[:, _BID].astype(np.float64)
            + features[:, _ASK].astype(np.float64)
        ) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(mid))
        s1, s2, counts = _nan_cumsums(log_returns)

        if "volatility" in types:
            # Preserve the legacy pair target: realized volatility over
            # [t, min(t+h, close)).
            end = np.minimum(future, n)
            result["volatility"] = _windowed_std(
                s1, s2, counts, t, end - 1
            )

        if "volatility_change" in types:
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
) -> np.ndarray:
    """Compute selected raw targets for one decision instant.

    Work is lazy by target family: VWAP, spread, and log-return prefix sums are
    built only when their corresponding target is requested. All requested
    horizons share those prefix sums.
    """
    type_order = tuple(types)
    hs = np.asarray(horizons, dtype=np.int64)
    arrays = _standard_target_arrays(
        focal_features,
        np.asarray([t_idx]),
        hs,
        type_order,
        int(measurement_window),
    )
    output = [arrays[target_type][0] for target_type in type_order]

    if rf_data and rf_price_mode and date_str is not None and rf_t_idx is not None:
        if rf_price_mode not in {"mid", "vwap"}:
            raise ValueError("rf_price_mode must be 'mid', 'vwap', or None")
        n_focal = len(focal_features)
        focal_mid = (
            focal_features[t_idx, _BID] + focal_features[t_idx, _ASK]
        ) / 2.0
        focal_future = np.minimum(t_idx + hs, n_focal - 1)
        focal_future_mid = (
            focal_features[focal_future, _BID]
            + focal_features[focal_future, _ASK]
        ) / 2.0
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
                if rf_price_mode == "mid":
                    rf_base = (
                        values[rf_t_idx, _BID] + values[rf_t_idx, _ASK]
                    ) / 2.0
                    rf_forward = (
                        values[rf_future, _BID] + values[rf_future, _ASK]
                    ) / 2.0
                else:
                    rf_base = values[rf_t_idx, _VWAP]
                    rf_forward = values[rf_future, _VWAP]
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
    spec: AnchorSpec = DEFAULT_ANCHOR_SPEC,
) -> np.ndarray:
    """Raw forward targets for one (ticker, date) at every anchor.

    Vectorized counterpart of ``targets.compute_pair_targets`` for the anchor
    grid, with ONE deliberate difference: no close clamp. Where that function
    uses ``fut_idx = min(t + h, N - 1)``, this emits NaN.

    Args:
        features: (N, 9) raw 1 Hz rows, ffilled, NOT normalized.
        horizons: forward horizons in seconds.
        anchors: 1 Hz row indices; defaults to the standard grid.

    Returns:
        (A, T, H) float64, ``T`` ordered as :data:`ANCHOR_TARGET_TYPES`.
    """
    if anchors is None:
        anchors = anchor_indices(len(features), spec=spec)
    anchors = np.asarray(anchors, dtype=np.int64)
    hs = np.asarray(list(horizons), dtype=np.int64)
    arrays = _standard_target_arrays(
        features,
        anchors,
        hs,
        ANCHOR_TARGET_TYPES,
        spec.measurement_window_seconds,
    )
    return np.stack([arrays[name] for name in ANCHOR_TARGET_TYPES], axis=1)
