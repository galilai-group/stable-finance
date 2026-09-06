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
    vw = features[:, _VWAP].astype(np.float64)
    vol = np.nan_to_num(features[:, _VOL].astype(np.float64), nan=0.0)
    num = np.nan_to_num(vw * vol, nan=0.0)
    c_num = np.concatenate([[0.0], np.cumsum(num)])
    c_vol = np.concatenate([[0.0], np.cumsum(vol)])

    n = len(features)
    a = np.asarray(idx, dtype=np.int64)
    lo, hi = a, a + int(window)
    ok = (lo >= 0) & (hi <= n)
    lo_s, hi_s = np.clip(lo, 0, n), np.clip(hi, 0, n)
    den = c_vol[hi_s] - c_vol[lo_s]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ok & (den > 0),
                       (c_num[hi_s] - c_num[lo_s]) / np.where(den > 0, den, 1.0),
                       np.nan)
    return out if np.ndim(idx) else float(out)

ANCHOR_TARGET_TYPES = ("return", "volatility_change", "spread_change")
PAIR_TARGET_TYPES = (
    "return",
    "spread_change",
    "volatility",
    "volatility_change",
)


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
    n = len(features)
    if anchors is None:
        anchors = anchor_indices(n, spec=spec)
    anchors = np.asarray(anchors, dtype=np.int64)
    hs = np.asarray(list(horizons), dtype=np.int64)

    bid = features[:, _BID].astype(np.float64)
    ask = features[:, _ASK].astype(np.float64)
    mid = (bid + ask) / 2.0
    spread = ask - bid

    with np.errstate(divide="ignore", invalid="ignore"):
        logmid = np.log(mid)
    r = np.diff(logmid)
    s1, s2, cn = _nan_cumsums(r)

    t = anchors[:, None]                      # (A, 1)
    fut = t + hs[None, :]                     # (A, H)
    # No clamp: a window that would run past the last row has no target.
    # Every target now needs its forward window to fit, not just the horizon.
    measurement_window = spec.measurement_window_seconds
    valid = (fut + measurement_window) <= n
    fut_safe = np.where(valid, fut, 0)

    # Both ends are forward VWAP windows of the same shape, so the base is
    # unknowable at t and no statistic computed at t can be correlated with its
    # measurement error.
    base = forward_vwap(features, anchors, measurement_window)[:, None]
    fwd_px = forward_vwap(features, fut_safe, measurement_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(valid & np.isfinite(base) & (base != 0),
                       fwd_px / np.where(base != 0, base, 1.0) - 1.0, np.nan)

    # Change targets also compare two forward windows so their baseline is not
    # a spread or volatility measurement already visible in the input.
    w = measurement_window
    cs_spread = np.concatenate([[0.0], np.nancumsum(spread)])
    spr_base = _windowed_mean(cs_spread, t, t + w)              # (A,1)
    spr_fwd = _windowed_mean(cs_spread, fut_safe, fut_safe + w)  # (A,H)
    spr = np.where(valid, spr_fwd - spr_base, np.nan)

    # Realized vol over each window. mids[a:b] -> log-returns r[a:b-1], hence
    # the -1 on the upper bound.
    vol_base = _windowed_std(s1, s2, cn, np.broadcast_to(t, fut.shape),
                             np.broadcast_to(t + w - 1, fut.shape))
    vol_fwd = _windowed_std(s1, s2, cn, fut_safe, fut_safe + w - 1)
    vol = np.where(valid, vol_fwd - vol_base, np.nan)

    out = np.empty(
        (len(anchors), len(ANCHOR_TARGET_TYPES), len(hs)), dtype=np.float64
    )
    out[:, ANCHOR_TARGET_TYPES.index("return"), :] = ret
    out[:, ANCHOR_TARGET_TYPES.index("volatility_change"), :] = vol
    out[:, ANCHOR_TARGET_TYPES.index("spread_change"), :] = spr
    return out
