"""What a model is trained and scored on, kept apart from data and loader.

The layers are separate on purpose:

- the STORE holds the raw time series, by ticker and day;
- the LOADER cuts views from it: rows up to a decision, and nothing after;
- an OBJECTIVE turns the session around a decision into labels.

An :class:`Objective` names everything that defines a label and nothing else:
the target families, the horizons, the measurement window (how long the
position takes to enter and to exit -- the execution window), the price
method, and the cross-section the standardized versions are taken over. It
reads only the session it is given; a context and its future never cross a
session boundary.

Labels are produced in two steps. ``raw`` is a fact about one name: its
forward return, spread change or volatility change. The standardized forms
(z-score, uniform, rank) are facts about a name relative to a cross-section
and are derived from the raw values of every name in a cell
(:func:`cross_section_transforms`),
so filtering to a universe or re-ranking is a cheap downstream step and a
batch of K names still gets labels from the full cross-section.

CACHES ARE OPTIMIZATIONS, NEVER DEFINITIONS. A day store's stored targets,
the monthly target tables and the panel cache all hold one objective's
labels. :attr:`Objective.key` names that objective, so a cached label is
always tied to the definition that produced it, and a different horizon or
window is a different key -- computed on demand from the stored series, and
cached under its own key if asked -- rather than a rebuild of the store.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

import numpy as np

from stable_finance.dataset.labels.families import (
    ANCHOR_TARGET_TYPES,
    anchor_targets,
    check_targets,
    get_target_names,
    supported_targets,
)
from stable_finance.dataset.labels.tables import MIN_NAMES, QUANTILE_LEVELS
from stable_finance.dataset.spec.schema import FeatureSchema, resolve_price
from stable_finance.dataset.spec.timeline import (
    DEFAULT_TARGET_HORIZONS,
    RETURN_VWAP_WINDOW,
    TO_CLOSE,
    AnchorSpec,
)


#: Who a standardized label is relative to. ``observed`` is every name with a
#: finite raw value in the (date, anchor, type, horizon) cell, once the cell
#: has ``min_names`` of them: the definition every reported IC uses.
CROSS_SECTIONS = ("observed",)

OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class Objective:
    """One label definition. Unset fields resolve against a dataset's schema.

    ``types=None`` is every anchor family the schema supports; ``price=None``
    is its richest price. :meth:`resolve` fills both, and only a resolved
    objective has a :attr:`key`.
    """

    types: tuple[str, ...] | None = None
    horizons: tuple[int, ...] = DEFAULT_TARGET_HORIZONS
    window_seconds: int = RETURN_VWAP_WINDOW
    price: str | None = None
    cross_section: str = "observed"
    min_names: int = MIN_NAMES

    def __post_init__(self) -> None:
        if self.types is not None:
            types = tuple(str(t) for t in self.types)
            unknown = set(types) - set(ANCHOR_TARGET_TYPES)
            if unknown or not types or len(set(types)) != len(types):
                raise ValueError(
                    f"types must be distinct members of {ANCHOR_TARGET_TYPES}; got {types}")
            object.__setattr__(self, "types", types)
        horizons = tuple(int(h) for h in self.horizons)
        if not horizons or len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be non-empty and distinct")
        if any(h <= 0 and h != TO_CLOSE for h in horizons):
            raise ValueError("horizons must be positive seconds or TO_CLOSE")
        object.__setattr__(self, "horizons", horizons)
        if int(self.window_seconds) < 1:
            raise ValueError("window_seconds must be positive")
        object.__setattr__(self, "window_seconds", int(self.window_seconds))
        if self.cross_section not in CROSS_SECTIONS:
            raise ValueError(f"cross_section must be one of {CROSS_SECTIONS}")
        if int(self.min_names) < 1:
            raise ValueError("min_names must be at least 1")

    # -- resolution against a dataset ------------------------------------------

    @property
    def resolved(self) -> bool:
        return self.types is not None and (self.price is not None or "return" not in self.types)

    def resolve(self, schema: FeatureSchema) -> "Objective":
        """Fill unset fields from ``schema`` and check it can form every label."""
        types = self.types
        price = self.price
        if types is None:
            supported = set(supported_targets(schema, price))
            types = tuple(t for t in ANCHOR_TARGET_TYPES if t in supported)
        if "return" in types:
            price = resolve_price(schema, price)
        check_targets(schema, types, price)
        return dataclasses.replace(self, types=types, price=price)

    # -- what it is --------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"version": OBJECTIVE_VERSION, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, value: dict) -> "Objective":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in fields})

    @property
    def key(self) -> str:
        """A short stable hash of the definition; caches are filed under it."""
        if not self.resolved:
            raise ValueError("resolve the objective against a schema before keying it")
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def names(self) -> list[str]:
        """Label names, type-major, with an ``@window`` suffix off the default."""
        if self.types is None:
            raise ValueError("resolve the objective to name its labels")
        return get_target_names(list(self.horizons), list(self.types),
                                window_seconds=self.window_seconds)

    @property
    def lookahead_seconds(self) -> int | None:
        """How far past a decision the labels read; None when one reads to the close."""
        if TO_CLOSE in self.horizons:
            return None
        return max(self.horizons) + self.window_seconds

    # -- labels ------------------------------------------------------------------

    def raw(self, features: np.ndarray, rows, *, schema: FeatureSchema,
            timeline=None) -> np.ndarray:
        """``(len(rows), T, H)`` raw labels for one session at decision ``rows``.

        ``timeline`` is the session's clock (default Market-1T's); the
        measurement window is this objective's own.
        """
        resolved = self if self.resolved else self.resolve(schema)
        return anchor_targets(features, resolved.horizons, np.asarray(rows, dtype=np.int64),
                              types=resolved.types, spec=resolved.spec(timeline),
                              schema=schema, price=resolved.price)

    def spec(self, timeline=None) -> AnchorSpec:
        """A clock and this objective's window, in the form the target functions take."""
        from stable_finance.dataset.spec.timeline import as_timeline

        return as_timeline(timeline).anchor_spec(self.window_seconds)

    def transform(self, raw: np.ndarray, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """The cross-sectional forms of a day's ``(A, N, T, H)`` raw labels."""

        return cross_section_transforms(raw, mask, min_names=self.min_names)


# ── cross-sectional transforms ───────────────────────────────────────────────

def _normal_ppf_vectorized(u: np.ndarray) -> np.ndarray:
    """Vectorized inverse standard-normal CDF (Acklam), the scalar routine in
    ``targets._normal_ppf`` applied elementwise; SciPy is not a dependency."""
    u = np.asarray(u, dtype=np.float64)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398,
         2.445134137142996, 3.754408661907416]
    out = np.full(u.shape, np.nan)
    lo, hi = u < 0.02425, u > 0.97575
    mid = ~(lo | hi) & np.isfinite(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.sqrt(-2 * np.log(u[lo]))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = np.sqrt(-2 * np.log(1 - u[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = u[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


def cross_section_transforms(raw: np.ndarray, mask: np.ndarray | None = None, *,
                             min_names: int = MIN_NAMES) -> dict[str, np.ndarray]:
    """Every standard target transform of a day, from its full cross-section.

    ``mask`` restricts the cross-section: ``(N,)`` or ``(A, N)`` booleans, True
    for a name that belongs to it. A masked-out name is NaN in every output
    and does not count toward ``min_names``. Without a mask the cross-section
    is every name observed in the cell, the library's definition.

    ``raw`` is ``(A, N, T, H)``. Returns ``zscore``/``uniform``/``rank`` of the
    same shape and ``count`` of ``(A, T, H)``. The arithmetic follows the
    table path exactly: population moments finalized to float32 and masked
    below MIN_NAMES (``targets.finalize``); the uniform score is the average
    rank among the float32 order statistics over ``n + 1``
    (``AnchorTargetStats.uniform_score``); the Gaussian rank interpolates the
    64-level empirical quantiles and clips to half a level from either end
    (``rank_score``). Only the summation order differs, which is inside float32
    rounding of the stored moments.
    """
    raw = np.asarray(raw)
    if raw.ndim != 4:
        raise ValueError(f"raw targets must be (A, N, T, H); got {raw.shape}")
    A, N, T, H = raw.shape
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape == (N,):
            keep = np.broadcast_to(keep, (A, N))
        if keep.shape != (A, N):
            raise ValueError(f"mask must be ({N},) or ({A}, {N}); got {keep.shape}")
        raw = np.where(keep[:, :, None, None], raw, np.nan)
    # (cells, N): one row per (anchor, type, horizon) cross-section.
    cells = np.ascontiguousarray(
        raw.astype(np.float32).transpose(0, 2, 3, 1).reshape(-1, N))
    finite = np.isfinite(cells)
    count = finite.sum(axis=1)
    n = count.astype(np.float64)
    v = np.where(finite, cells, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = v.sum(axis=1) / n
        variance = (v * v).sum(axis=1) / n - mean * mean
        sigma = np.sqrt(np.maximum(variance, 0.0))
    thin = count < min_names
    mean32 = np.where(thin, np.nan, mean).astype(np.float32)
    sigma32 = np.where(thin | (sigma <= 0), np.nan, sigma).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (cells.astype(np.float64) - mean32[:, None]) / sigma32[:, None]
    z = np.where(np.isfinite(sigma32)[:, None] & finite, z, np.nan)

    uniform = np.full(cells.shape, np.nan, dtype=np.float64)
    rank = np.full(cells.shape, np.nan, dtype=np.float64)
    levels = QUANTILE_LEVELS
    bound = 0.5 / (len(levels) * 2)
    for c in range(len(cells)):
        ok = finite[c]
        m = int(count[c])
        if m == 0:
            continue
        vals = cells[c, ok]                      # float32, the table's dtype
        if m >= min_names:
            grid = np.sort(vals)
            left = np.searchsorted(grid, vals, side="left")
            right = np.searchsorted(grid, vals, side="right")
            uniform[c, ok] = (left + right + 1.0) / (2.0 * (m + 1.0))
        q = np.nanquantile(vals, levels).astype(np.float32)
        if np.isfinite(q).all():
            p = np.interp(vals.astype(np.float64), q.astype(np.float64), levels)
            rank[c, ok] = _normal_ppf_vectorized(np.clip(p, bound, 1.0 - bound))

    def back(x):
        return np.ascontiguousarray(
            x.reshape(A, T, H, N).transpose(0, 3, 1, 2)).astype(np.float32)

    return {
        "zscore": back(z), "uniform": back(uniform), "rank": back(rank),
        "count": count.reshape(A, T, H).astype(np.int32),
    }
