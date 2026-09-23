"""The observable market state on a decision grid, independent of any model.

A quote and a realized return are properties of a (date, anchor, ticker) row.
They do not depend on which encoder produced an embedding for that row, which
is what makes this module the join that lets a portfolio be built from
embeddings that were computed for something else entirely: score a sweep once,
then attach the market to it as often as the analysis needs, with no second
forward pass.

:class:`MarketPanel` reads one month of cached panel metadata -- quotes and
raw targets -- WITHOUT touching the cached views, which are the only large
array in a panel. :meth:`MarketPanel.align` then reorders it onto an arbitrary
set of rows, and :func:`to_grid` scatters flat rows onto the
``(decision, asset, horizon)`` rectangle every other stage speaks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["MarketPanel", "decision_labels", "find_cached_month", "to_grid"]


def _row_keys(dates: ArrayLike, anchors: ArrayLike,
              tickers: ArrayLike) -> NDArray[np.str_]:
    """The join key, as one string per row.

    A structured dtype or a tuple list would both work; a single string column
    is used because it makes the key visible in any debugger and because the
    dict lookup that follows is the same cost either way.
    """
    date = np.asarray(dates).astype("U10")
    anchor = np.char.mod("%d", np.asarray(anchors).astype(np.int64))
    ticker = np.asarray(tickers).astype("U8")
    return np.char.add(np.char.add(date, np.char.add("|", anchor)),
                       np.char.add("|", ticker))


def find_cached_month(
    root: str | Path, month: str, *, anchors_per_day: int,
    stats_tag: str | None = None,
) -> Path:
    """Locate a ready cache directory for ``month`` at this anchor schedule.

    THE KEY HASH IS DELIBERATELY NOT REQUIRED. A panel's hash covers the view
    construction -- normalization groups, sequence length, aggregation -- none
    of which changes a bid, an ask, or a realized return. Demanding the exact
    hash would force a caller to reconstruct an encoder's view settings just to
    read the market, and would miss a perfectly good quote table sitting under
    a neighbouring key.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"panel cache root {base} does not exist")
    for key_dir in sorted(base.iterdir()):
        directory = key_dir / month
        if not (directory / ".ready").is_file():
            continue
        try:
            key = json.loads((directory / "meta.json").read_text())["key"]
        except (OSError, ValueError, KeyError):
            continue
        if int(key.get("anchors_per_day", -1)) != int(anchors_per_day):
            continue
        if stats_tag is not None and key.get("stats_tag") != stats_tag:
            continue
        if (directory / "quotes.npy").is_file():
            return directory
    raise FileNotFoundError(
        f"no ready panel cache for {month} at {anchors_per_day} anchors/day"
        + (f" with stats {stats_tag!r}" if stats_tag else "")
    )


@dataclass(frozen=True)
class MarketPanel:
    """One month of quotes and realized targets, keyed by row."""

    dates: NDArray
    anchors: NDArray[np.int64]
    tickers: NDArray
    quotes: NDArray[np.float64]
    raw_targets: NDArray[np.float64]
    month: str

    @classmethod
    def from_cache(
        cls, root: str | Path, month: str, *, anchors_per_day: int,
        stats_tag: str | None = None,
    ) -> "MarketPanel":
        """Read a month's market columns, skipping the cached views entirely."""
        directory = find_cached_month(
            root, month, anchors_per_day=anchors_per_day, stats_tag=stats_tag
        )
        load = lambda name: np.load(directory / name, allow_pickle=False)  # noqa: E731
        quotes = load("quotes.npy").astype(np.float64)
        raw = load("raw_targets.npy").astype(np.float64)
        dates, anchors, tickers = load("dates.npy"), load("anchors.npy"), load("tickers.npy")
        lengths = {len(quotes), len(raw), len(dates), len(anchors), len(tickers)}
        if len(lengths) != 1:
            raise RuntimeError(f"panel cache at {directory} has ragged columns")
        return cls(dates=dates, anchors=anchors.astype(np.int64), tickers=tickers,
                   quotes=quotes, raw_targets=raw, month=month)

    @property
    def bid(self) -> NDArray[np.float64]:
        return self.quotes[:, 0]

    @property
    def ask(self) -> NDArray[np.float64]:
        return self.quotes[:, 1]

    def half_spread(self) -> NDArray[np.float64]:
        """Quoted half-spread over mid; NaN where the book was not two-sided.

        A crossed book (ask below bid) is treated as unquoted rather than
        clamped to zero, because a negative half-spread would book a REBATE
        for crossing the spread -- turning the most illiquid rows in the
        archive into the most profitable ones.
        """
        bid, ask = self.bid, self.ask
        broken = ~np.isfinite(bid) | ~np.isfinite(ask) | (bid <= 0) | (ask < bid)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(broken, np.nan, (ask - bid) / 2 / ((bid + ask) / 2))

    def align(self, dates: ArrayLike, anchors: ArrayLike,
              tickers: ArrayLike) -> NDArray[np.int64]:
        """Row indices of this panel matching the supplied rows; -1 for misses.

        The caller is expected to CHECK THE MISS RATE rather than trust it. A
        silent miss is the failure mode that matters here: an unmatched row
        carries no quote, so it would be priced as a free trade in a market
        that never existed.
        """
        keys = _row_keys(self.dates, self.anchors, self.tickers)
        if len(np.unique(keys)) != len(keys):
            raise RuntimeError(f"{self.month} panel has duplicate row keys")
        lookup = {key: index for index, key in enumerate(keys.tolist())}
        wanted = _row_keys(dates, anchors, tickers)
        return np.array([lookup.get(key, -1) for key in wanted.tolist()],
                        dtype=np.int64)

    def take(self, index: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """Quotes, half-spread and raw targets for aligned rows; NaN on a miss."""
        idx = np.asarray(index, dtype=np.int64)
        hit = idx >= 0
        safe = np.where(hit, idx, 0)
        blank = lambda a: np.where(hit[:, None], a[safe], np.nan)  # noqa: E731
        spread = np.where(hit, self.half_spread()[safe], np.nan)
        return {"quote": blank(self.quotes), "raw_targets": blank(self.raw_targets),
                "half_spread": spread, "matched": hit}


def decision_labels(dates: ArrayLike, anchors: ArrayLike) -> NDArray[np.str_]:
    """One label per (date, anchor), the decision identity of a row.

    A decision is a wall-clock instant shared across the cross-section, so
    every stock observed at the same date and anchor belongs to the same
    decision. Sorting these labels sorts chronologically because the date is
    ISO and the anchor is zero-padded.
    """
    anchor = np.char.zfill(np.char.mod("%d", np.asarray(anchors).astype(np.int64)), 4)
    return np.char.add(np.asarray(dates).astype("U10"), np.char.add("|", anchor))


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
