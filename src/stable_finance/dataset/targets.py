"""Cross-sectional target metadata for the supported equity dataset.

The raw forward outcome and its standard target transforms belong to the data
pipeline, not to a model. :class:`AnchorTargetStats` loads the precomputed
cross-section and returns every representation together; a consumer can select
one for fitting without recomputing or hiding the others.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_NAMES = 20
TARGET_TRANSFORMS = ("raw", "zscore", "uniform", "rank")


@dataclass(frozen=True)
class CrossSectionalTargetBundle:
    """One raw target array and all standard cross-sectional transforms.

    ``uniform`` is exactly ``rankdata(y_cell) / (n_cell + 1)`` with average
    ties. ``rank`` is the historical Gaussian-rank transform and is retained
    as an explicit alternative.
    """

    raw: np.ndarray
    zscore: np.ndarray
    uniform: np.ndarray
    rank: np.ndarray

    def select(self, transform: str) -> np.ndarray:
        if transform not in TARGET_TRANSFORMS:
            raise ValueError(
                f"unknown target transform {transform!r}; expected one of "
                f"{TARGET_TRANSFORMS}"
            )
        return getattr(self, transform)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in TARGET_TRANSFORMS}


class AnchorTargetStats:
    """Precomputed cross-sections with target-bundle lookup by cell."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.table_set = self.path.parent.name
        with np.load(path, allow_pickle=False) as data:
            self.dates = [str(x) for x in data["dates"]]
            self.anchors = data["anchors"].astype(np.int64)
            self.types = [str(x) for x in data["types"]]
            self.horizons = data["horizons"].astype(np.int64)
            self.mu = data["mu"]
            self.sigma = data["sigma"]
            self.count = data["count"]
            self.quantiles = (
                data["quantiles"] if "quantiles" in data.files else None
            )
            self.quantile_levels = (
                data["quantile_levels"]
                if "quantile_levels" in data.files
                else None
            )
            self.sorted_values = (
                data["sorted_values"] if "sorted_values" in data.files else None
            )
        self._reindex()

    def _reindex(self) -> None:
        self._date_idx = {value: i for i, value in enumerate(self.dates)}
        self._anchor_idx = {int(value): i for i, value in enumerate(self.anchors)}
        self._type_idx = {value: i for i, value in enumerate(self.types)}
        self._h_idx = {int(value): i for i, value in enumerate(self.horizons)}

    @classmethod
    def load_months(cls, paths) -> "AnchorTargetStats":
        parts = [cls(path) for path in paths]
        if not parts:
            raise ValueError("no anchor-stat tables given")
        first = parts[0]
        for part in parts[1:]:
            if part.types != first.types or not np.array_equal(
                part.horizons, first.horizons
            ):
                raise ValueError("anchor tables disagree on types/horizons")
            if not np.array_equal(part.anchors, first.anchors):
                raise ValueError("anchor tables disagree on the anchor grid")
            if part.table_set != first.table_set:
                raise ValueError(
                    "anchor tables come from different target generations: "
                    f"{first.table_set!r} vs {part.table_set!r}"
                )

        merged = object.__new__(cls)
        merged.path, merged.table_set = first.path, first.table_set
        merged.dates = [date for part in parts for date in part.dates]
        merged.anchors, merged.types = first.anchors, first.types
        merged.horizons = first.horizons
        for name in ("mu", "sigma", "count"):
            setattr(merged, name, np.concatenate([getattr(p, name) for p in parts]))
        if any(part.quantiles is None for part in parts):
            merged.quantiles = merged.quantile_levels = None
        else:
            merged.quantiles = np.concatenate([p.quantiles for p in parts])
            merged.quantile_levels = first.quantile_levels
        if any(part.sorted_values is None for part in parts):
            merged.sorted_values = None
        else:
            width = max(part.sorted_values.shape[-1] for part in parts)
            padded = []
            for part in parts:
                values = part.sorted_values
                if values.shape[-1] < width:
                    pad = [(0, 0)] * values.ndim
                    pad[-1] = (0, width - values.shape[-1])
                    values = np.pad(values, pad, constant_values=np.nan)
                padded.append(values)
            merged.sorted_values = np.concatenate(padded)
        merged._reindex()
        return merged

    def _cell(self, raw, date, anchor, types, horizons):
        raw = np.asarray(raw, dtype=np.float64)
        di = self._date_idx.get(str(date))
        ai = self._anchor_idx.get(int(anchor))
        ti = [self._type_idx.get(str(value), -1) for value in types]
        hi = [self._h_idx.get(int(value), -1) for value in horizons]
        if di is None or ai is None or any(i < 0 for i in ti + hi):
            return raw, None
        shape = (len(ti), len(hi))
        if raw.size != int(np.prod(shape)):
            raise ValueError(f"raw targets have shape {raw.shape}; expected {shape}")
        return raw, (di, ai, ti, hi, shape)

    def transform(
        self, raw: np.ndarray, date: str, anchor: int, types, horizons,
    ) -> CrossSectionalTargetBundle:
        """Return raw, z-score, uniform-rank, and Gaussian-rank metadata."""
        raw_array = np.asarray(raw, dtype=np.float64)
        return CrossSectionalTargetBundle(
            raw=raw_array.copy(),
            zscore=self.zscore(raw_array, date, anchor, types, horizons),
            uniform=self.uniform_score(raw_array, date, anchor, types, horizons),
            rank=self.rank_score(raw_array, date, anchor, types, horizons),
        )

    def zscore(self, raw, date, anchor, types, horizons) -> np.ndarray:
        raw, cell = self._cell(raw, date, anchor, types, horizons)
        if cell is None:
            return np.full(raw.shape, np.nan)
        di, ai, ti, hi, shape = cell
        mu = self.mu[di, ai][np.ix_(ti, hi)]
        sigma = self.sigma[di, ai][np.ix_(ti, hi)]
        with np.errstate(invalid="ignore", divide="ignore"):
            result = (raw.reshape(shape) - mu) / sigma
        return np.where(
            np.isfinite(sigma) & (sigma > 0), result, np.nan
        ).reshape(raw.shape)

    def uniform_score(self, raw, date, anchor, types, horizons) -> np.ndarray:
        """Return exact ``rankdata(y_cell) / (n_cell + 1)`` values."""
        if self.sorted_values is None:
            raise ValueError(
                "anchor tables carry no exact order statistics; rebuild them "
                "to use the uniform target"
            )
        raw, cell = self._cell(raw, date, anchor, types, horizons)
        if cell is None:
            return np.full(raw.shape, np.nan)
        di, ai, ti, hi, shape = cell
        grids = self.sorted_values[di, ai][np.ix_(ti, hi)]
        counts = self.count[di, ai][np.ix_(ti, hi)]
        values, out = raw.reshape(shape), np.full(shape, np.nan)
        for index in np.ndindex(shape):
            n, value = int(counts[index]), values[index]
            if n < MIN_NAMES or not np.isfinite(value):
                continue
            grid = grids[index][:n]
            if not np.isfinite(grid).all():
                continue
            value = np.float32(value)
            left = int(np.searchsorted(grid, value, side="left"))
            right = int(np.searchsorted(grid, value, side="right"))
            out[index] = (left + right + 1.0) / (2.0 * (n + 1.0))
        return out.reshape(raw.shape)

    def rank_score(self, raw, date, anchor, types, horizons) -> np.ndarray:
        """Return the historical ``Phi^-1(F_cell(y))`` target."""
        if self.quantiles is None:
            raise ValueError("anchor tables carry no quantiles; rebuild them")
        raw, cell = self._cell(raw, date, anchor, types, horizons)
        if cell is None:
            return np.full(raw.shape, np.nan)
        di, ai, ti, hi, shape = cell
        grids = self.quantiles[di, ai][np.ix_(ti, hi)]
        levels = np.asarray(self.quantile_levels, dtype=np.float64)
        values, out = raw.reshape(shape), np.full(shape, np.nan)
        for index in np.ndindex(shape):
            grid, value = grids[index], values[index]
            if not np.isfinite(value) or not np.isfinite(grid).all():
                continue
            percentile = float(np.interp(value, grid, levels))
            bound = 0.5 / (len(levels) * 2)
            out[index] = _normal_ppf(min(max(percentile, bound), 1.0 - bound))
        return out.reshape(raw.shape)

    def unstandardize_panel(self, z, dates, anchors, types, horizons) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        out = np.full(z.shape, np.nan)
        ti = np.array([self._type_idx.get(str(value), -1) for value in types])
        hi = np.array([self._h_idx.get(int(value), -1) for value in horizons])
        if ti.min() < 0 or hi.min() < 0:
            return out
        di = np.array([self._date_idx.get(str(value), -1) for value in dates])
        ai = np.array([self._anchor_idx.get(int(value), -1) for value in anchors])
        ok = (di >= 0) & (ai >= 0)
        if not ok.any():
            return out
        n = int(ok.sum())
        mu = self.mu[di[ok], ai[ok]][:, ti][:, :, hi].reshape(n, -1)
        sigma = self.sigma[di[ok], ai[ok]][:, ti][:, :, hi].reshape(n, -1)
        out[ok] = np.where(
            np.isfinite(sigma) & (sigma > 0), z[ok] * sigma + mu, np.nan
        )
        return out


def accumulate(sums, squares, counts, values) -> None:
    """Accumulate finite observations into cross-sectional moments."""
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0)
    sums += values
    squares += values * values
    counts += finite


def finalize(sums, squares, counts, *, min_names: int = MIN_NAMES):
    """Finalize population moments, masking thin or degenerate cells."""
    n = counts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / n
        variance = squares / n - mean * mean
        sigma = np.sqrt(np.maximum(variance, 0.0))
    thin = counts < min_names
    mean = np.where(thin, np.nan, mean)
    sigma = np.where(thin | (sigma <= 0), np.nan, sigma)
    return mean.astype(np.float32), sigma.astype(np.float32)


def build_cross_section_metadata(
    sums: np.ndarray,
    squares: np.ndarray,
    counts: np.ndarray,
    samples_by_decision,
    quantile_levels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the arrays needed for every standard target representation.

    Each element of ``samples_by_decision`` has shape
    ``(n_assets, *cell_shape)``. The asset axis may vary between decisions;
    exact order statistics are NaN-padded to the largest cross-section.
    """
    mean, sigma = finalize(sums, squares, counts)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or len(levels) < 2 or levels[0] != 0 or levels[-1] != 1:
        raise ValueError("quantile_levels must be a 1-D grid from 0 to 1")
    samples = [
        np.asarray(values, dtype=np.float32) for values in samples_by_decision
    ]
    if len(samples) != len(counts):
        raise ValueError("samples_by_decision must align with the decision axis")
    cell_shape = counts.shape[1:]
    width = max((len(values) for values in samples), default=0)
    quantiles = np.full(
        (len(samples), *cell_shape, len(levels)), np.nan, dtype=np.float32
    )
    sorted_values = np.full(
        (len(samples), *cell_shape, width), np.nan, dtype=np.float32
    )
    for index, values in enumerate(samples):
        expected = (len(values), *cell_shape)
        if values.shape != expected:
            raise ValueError(
                f"decision samples have shape {values.shape}; expected {expected}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            quantiles[index] = np.moveaxis(
                np.nanquantile(values, levels, axis=0), 0, -1
            ).astype(np.float32)
        sorted_values[index, ..., :len(values)] = np.moveaxis(
            np.sort(values, axis=0), 0, -1
        )
    return {
        "mu": mean,
        "sigma": sigma,
        "count": counts.astype(np.int32),
        "quantiles": quantiles,
        "quantile_levels": levels.astype(np.float32),
        "sorted_values": sorted_values,
    }


def _normal_ppf(u: float) -> float:
    """Inverse standard-normal CDF (Acklam), avoiding a SciPy dependency."""
    import math

    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398,
         2.445134137142996, 3.754408661907416]
    if u < 0.02425:
        q = math.sqrt(-2 * math.log(u))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if u > 0.97575:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = u - 0.5, (u - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
