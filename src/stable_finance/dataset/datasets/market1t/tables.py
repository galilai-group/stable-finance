"""Monthly cross-sectional target tables: what they hold, and the command that builds them.

TABLES. Cross-sectional target metadata: moments and order statistics per cell.

The raw forward outcome and its standard target transforms belong to the data
pipeline, not to a model. :class:`AnchorTargetStats` loads the precomputed
cross-section and returns every representation together; a consumer can select
one for fitting without recomputing or hiding the others.

BUILDING TABLES. Build monthly cross-sectional anchor tables and empirical target transforms.

One pass over a month's mosaic shards folds every (ticker, date) into running
moments of the raw forward targets on the 5-minute anchor grid, then writes
moments and order statistics per (date, anchor, target_type, horizon). Training and
eval both look the target up here, so the cross-section a sample is
standardized against is identical no matter which pipeline reads it.

The table is the ONLY place the cross-section exists: a streaming sample sees
one stock, so ``mu``/``sigma`` across ~700 names at a shared instant can never
be assembled at training time.

Cost is ~1 decode per ticker-day and the arithmetic is vectorized over all
(anchor, type, horizon) cells at once. The exact uniform target stores the
finite per-cell order statistic, so its table is materially larger than the
moment-only format.

Usage:
    sf-build-targets \
        --out-dir /data/lab/market-jepa-mosaic/xs_anchor_stats \
        --start 2023-01 --end 2023-02
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stable_finance.dataset.anchors import DEFAULT_TARGET_HORIZONS
from stable_finance.dataset.datasets.market1t.spec import (
    MARKET_SCHEMA,
    as_timeline,
    month_dirs,
)
from stable_finance.dataset.datasets.market1t.views import sparse_to_dense_grid
from stable_finance.dataset.objective import MIN_NAMES, QUANTILE_LEVELS
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES as TARGET_TYPES,
    anchor_offsets,
    anchor_targets,
)
from stable_finance.dataset.schema import default_schema


# ── tables ───────────────────────────────────────────────────────────────────


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
            # Tables written before these were recorded measured over the
            # standard one-minute window at the schema's default price.
            self.window_seconds = (int(data["window_seconds"])
                                   if "window_seconds" in data.files else 60)
            self.price = (str(data["price"]) or None) if "price" in data.files else None
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
    *,
    include_quantiles: bool = True,
    include_sorted: bool = True,
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
    result = {
        "mu": mean,
        "sigma": sigma,
        "count": counts.astype(np.int32),
    }
    if not include_quantiles and not include_sorted:
        return result
    if samples_by_decision is None:
        raise ValueError(
            "samples_by_decision is required for rank or uniform metadata"
        )
    samples = [np.asarray(values, dtype=np.float32)
               for values in samples_by_decision]
    if len(samples) != len(counts):
        raise ValueError("samples_by_decision must align with the decision axis")
    cell_shape = counts.shape[1:]
    width = max((len(values) for values in samples), default=0)
    quantiles = (
        np.full((len(samples), *cell_shape, len(levels)), np.nan,
                dtype=np.float32)
        if include_quantiles else None
    )
    sorted_values = (
        np.full((len(samples), *cell_shape, width), np.nan, dtype=np.float32)
        if include_sorted else None
    )
    for index, values in enumerate(samples):
        expected = (len(values), *cell_shape)
        if values.shape != expected:
            raise ValueError(
                f"decision samples have shape {values.shape}; expected {expected}"
            )
        if include_quantiles:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                quantiles[index] = np.moveaxis(
                    np.nanquantile(values, levels, axis=0), 0, -1
                ).astype(np.float32)
        if include_sorted:
            sorted_values[index, ..., :len(values)] = np.moveaxis(
                np.sort(values, axis=0), 0, -1
            )
    if include_quantiles:
        result["quantiles"] = quantiles
        result["quantile_levels"] = levels.astype(np.float32)
    if include_sorted:
        result["sorted_values"] = sorted_values
    return result


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


# ── building tables ──────────────────────────────────────────────────────────

# Market-1T's fill columns, kept for scripts that import them from here. The
# builder itself reads the fill rules off each store's own schema.
FFILL_IDX = MARKET_SCHEMA.forward_fill_indices
ZEROFILL_IDX = MARKET_SCHEMA.zero_fill_indices


def _shard_partial(month_dir: str, shard_meta: dict, timeline,
                   target_types: tuple[str, ...], horizons: tuple[int, ...],
                   keep_samples: bool, schema=None, spec=None,
                   price: str | None = None):
    """Fold one shard's ticker-days into per-date moment arrays.

    Returns ``{date: (sums, sqs, cnts)}``, each (A, T, H). Runs in a worker
    process; the timeline pickles (Market-1T's re-reads its holiday file).
    """
    import tempfile

    from stable_finance.dataset.datasets.market1t.mosaic import open_shard

    schema = default_schema(schema)
    timeline = as_timeline(timeline)
    spec = spec if spec is not None else timeline.anchor_spec()
    anchors_tod = anchor_offsets(spec)
    hs = list(horizons)
    shape = (len(anchors_tod), len(target_types), len(hs))

    out: dict[str, tuple] = {}
    with tempfile.TemporaryDirectory(prefix="xs_shard_") as tmpdir:
        reader = open_shard(month_dir, shard_meta, tmpdir)
        _fold_shard(
            reader, shard_meta["samples"], timeline, anchors_tod, hs,
            target_types, shape, out, keep_samples,
            schema=schema, spec=spec, price=price,
        )
    return out


def _fold_shard(reader, n_samples, timeline, anchors_tod, hs, target_types,
                shape, out, keep_samples, *, schema=None, spec=None,
                price=None):
    """Accumulate every ticker-day in one shard into ``out``, keyed by date.

    ``timeline`` may also be a Market-1T holiday schedule, as it was before
    timelines existed.
    """
    schema = default_schema(schema)
    timeline = as_timeline(timeline)
    spec = spec if spec is not None else timeline.anchor_spec()
    bar = timeline.bar_seconds
    for i in range(n_samples):
        sample = reader.get_item(i)
        date_str = str(sample["date"])
        if not timeline.is_open(date_str):
            continue  # market closed — a stray sample on a holiday
        try:
            ts_open, ts_close = timeline.bounds(date_str)
        except ValueError:
            continue
        res = sparse_to_dense_grid(
            ts_sec=sample["ts_interval"],
            features=sample["features"],
            ts_open_sec=ts_open,
            ts_close_sec=ts_close,
            ffill_indices=schema.forward_fill_indices,
            zerofill_indices=schema.zero_fill_indices,
            trim_leading_nan=True,
            bar_seconds=bar,
        )
        if res is None or len(res[1]) < 10:
            continue
        canonical_sec, features = res

        # Decisions are instants measured from the timeline's origin (for
        # Market-1T the standard 09:30 open, the dataset's meta.tod_sec
        # convention). A ticker that starts quoting late has its grid trimmed,
        # so its row 0 is NOT the origin and the decision must be shifted by
        # that offset before indexing.
        tod_offset = int(canonical_sec[0]) - timeline.origin(date_str)
        rows = (anchors_tod - tod_offset) // bar
        keep = (rows >= 0) & (rows < len(features))
        if not keep.any():
            continue

        y = np.full(shape, np.nan)
        y[keep] = anchor_targets(
            features, hs, rows[keep], types=target_types,
            spec=spec, schema=schema, price=price,
        )

        if date_str not in out:
            out[date_str] = (
                np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int64),
                [],
            )
        accumulate(*out[date_str][:3], y)
        # Retain the raw value so the cell's empirical CDF can be quantized
        # later. (mu, sigma) is not enough for a RANK target: assuming the
        # cross-section is Gaussian makes Phi^-1(Phi(z)) collapse back to z,
        # so the whole point is the shape the two moments throw away.
        # ~600 ticker-days x (A,T,H) float32 is ~34 MB per date — cheap.
        if keep_samples:
            out[date_str][3].append(y.astype(np.float32))


def build_month(
    ym: str,
    month_dir: Path,
    out_dir: Path,
    holiday_csv,
    workers: int,
    *,
    target_types: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] = DEFAULT_TARGET_HORIZONS,
    transforms: tuple[str, ...] = TARGET_TRANSFORMS,
    objective=None,
) -> Path:
    """Build one month, calculating only requested families and horizons.

    ``holiday_csv`` is the holiday file (its path, or a schedule or timeline)
    that gives short days their early close.
    ``objective`` sets types, horizons, price and measurement window together;
    without one, ``target_types`` (default: every family the schema supports)
    and ``horizons`` define it.
    """
    from joblib import Parallel, delayed

    from stable_finance.dataset.objective import Objective

    schema = MARKET_SCHEMA
    timeline = as_timeline(holiday_csv)
    if objective is None:
        objective = Objective(types=None if target_types is None else tuple(target_types),
                              horizons=tuple(horizons))
    objective = objective.resolve(schema)
    target_types, horizons, price = objective.types, objective.horizons, objective.price
    spec = objective.spec(timeline)

    unknown_types = set(target_types) - set(TARGET_TYPES)
    unknown_transforms = set(transforms) - set(TARGET_TRANSFORMS)
    if unknown_types:
        raise ValueError(f"unknown target types: {sorted(unknown_types)}")
    if unknown_transforms:
        raise ValueError(f"unknown target transforms: {sorted(unknown_transforms)}")
    if not target_types or not horizons:
        raise ValueError("target_types and horizons cannot be empty")
    keep_samples = bool({"uniform", "rank"} & set(transforms))
    with open(month_dir / "index.json") as f:
        shards = json.load(f)["shards"]

    parts = Parallel(n_jobs=workers, backend="loky")(
        delayed(_shard_partial)(
            str(month_dir), s, timeline, target_types, horizons, keep_samples,
            schema, spec, price,
        )
        for s in shards
    )

    anchors_tod = anchor_offsets(spec)
    hs = list(horizons)
    shape = (len(anchors_tod), len(target_types), len(hs))
    merged: dict[str, tuple] = {}
    for part in parts:
        for date_str, (s, q, c, vals) in part.items():
            if date_str not in merged:
                merged[date_str] = (
                    np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int64),
                    [],
                )
            m = merged[date_str]
            m[0][:] += s
            m[1][:] += q
            m[2][:] += c
            m[3].extend(vals)

    dates = sorted(merged)
    D = len(dates)
    sums = np.stack([merged[d][0] for d in dates]) if D else np.zeros((0, *shape))
    sqs = np.stack([merged[d][1] for d in dates]) if D else np.zeros((0, *shape))
    cnts = (
        np.stack([merged[d][2] for d in dates]) if D
        else np.zeros((0, *shape), dtype=np.int64)
    )
    table = build_cross_section_metadata(
        sums, sqs, cnts,
        ([np.stack(merged[d][3]) for d in dates] if keep_samples else None),
        QUANTILE_LEVELS,
        include_quantiles="rank" in transforms,
        include_sorted="uniform" in transforms,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ym}.npz"
    np.savez_compressed(
        path,
        dates=np.array(dates),
        anchors=anchors_tod.astype(np.int32),
        types=np.array(target_types),
        horizons=np.array(hs, dtype=np.int32),
        price=np.array(price or ""),
        window_seconds=np.int32(spec.measurement_window_seconds),
        **table,
    )
    usable = int(np.isfinite(table["sigma"]).sum())
    print(
        f"{ym}: {D} dates, "
        f"{usable}/{table['sigma'].size} usable cells "
        f"({100.0 * usable / max(1, table['sigma'].size):.1f}%), "
        f"median names/cell {int(np.median(cnts[cnts > 0])) if cnts.any() else 0} "
        f"-> {path}",
        flush=True,
    )
    return path


def main(argv=None, *, default_mosaic_dir=None, default_holiday_csv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_mosaic_dir = default_mosaic_dir or os.environ.get("MOSAIC_DIR")
    default_holiday_csv = default_holiday_csv or os.environ.get("HOLIDAY_CSV")
    p.add_argument("--mosaic-dir", default=default_mosaic_dir,
                   required=default_mosaic_dir is None, help="sparse mosaic root ($MOSAIC_DIR)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", required=True, help="first month, YYYY-MM")
    p.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    p.add_argument("--holiday-csv", default=default_holiday_csv,
                   required=default_holiday_csv is None, help="($HOLIDAY_CSV)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target-types", nargs="+", choices=TARGET_TYPES, default=None,
                   help="default: every family the store's schema supports")
    p.add_argument("--price", choices=("vwap", "mid", "close"), default=None,
                   help="price method of the return target (default: the schema's richest)")
    p.add_argument("--window", type=int, default=None,
                   help="measurement (execution) window in seconds; default 60")
    p.add_argument("--horizons", nargs="+", type=int,
                   default=list(DEFAULT_TARGET_HORIZONS))
    p.add_argument("--transforms", nargs="+", choices=TARGET_TRANSFORMS,
                   default=list(TARGET_TRANSFORMS))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    months = month_dirs(Path(args.mosaic_dir), args.start, args.end)
    if not months:
        raise SystemExit(f"No months in [{args.start}, {args.end}] under {args.mosaic_dir}")


    for ym, md in months:
        if (out_dir / f"{ym}.npz").is_file() and not args.overwrite:
            print(f"{ym}: exists, skipping (--overwrite to rebuild)", flush=True)
            continue
        t0 = time.time()
        from stable_finance.dataset.objective import Objective

        build_month(
            ym, md, out_dir, args.holiday_csv, args.workers,
            transforms=tuple(args.transforms),
            objective=Objective(
                types=tuple(args.target_types) if args.target_types else None,
                horizons=tuple(args.horizons), price=args.price,
                window_seconds=args.window if args.window is not None else 60),
        )
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
