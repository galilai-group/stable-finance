"""Evaluation inputs read from the dataset's stores, independent of any model.

Market data and realized forward returns for a set of day records, on one
shared decision and asset grid, and the panel cache's quotes joined onto
arbitrary rows.

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

from stable_finance.dataset.spec.schema import MissingRolesError
from stable_finance.evaluation.market_data import MarketData
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = ["MarketPanel", "decision_labels", "find_cached_month", "forward_returns_from_days",
           "market_from_days", "to_grid"]


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
        # A panel from a dataset without quotes says so in its meta.json; one
        # that should have them and does not is a damaged cache, skipped.
        quotes_absent = json.loads((directory / "meta.json").read_text()).get("quotes") is False
        if quotes_absent or (directory / "quotes.npy").is_file():
            return directory
    raise FileNotFoundError(
        f"no ready panel cache for {month} at {anchors_per_day} anchors/day"
        + (f" with stats {stats_tag!r}" if stats_tag else "")
    )


@dataclass(frozen=True)
class MarketPanel:
    """One month of quotes and realized targets, keyed by row.

    ``quotes`` is None when the panel's dataset has no bid and ask; every
    quote-reading method then raises
    :class:`~stable_finance.dataset.schema.MissingRolesError` rather than
    returning NaN, which would read as names that were not quoting.
    """

    dates: NDArray
    anchors: NDArray[np.int64]
    tickers: NDArray
    quotes: NDArray[np.float64] | None
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
        quotes = (load("quotes.npy").astype(np.float64)
                  if (directory / "quotes.npy").is_file() else None)
        raw = load("raw_targets.npy").astype(np.float64)
        dates, anchors, tickers = load("dates.npy"), load("anchors.npy"), load("tickers.npy")
        lengths = {len(raw), len(dates), len(anchors), len(tickers)}
        if quotes is not None:
            lengths.add(len(quotes))
        if len(lengths) != 1:
            raise RuntimeError(f"panel cache at {directory} has ragged columns")
        return cls(dates=dates, anchors=anchors.astype(np.int64), tickers=tickers,
                   quotes=quotes, raw_targets=raw, month=month)

    @property
    def has_quotes(self) -> bool:
        return self.quotes is not None

    def _quotes(self) -> NDArray[np.float64]:
        if self.quotes is None:
            raise MissingRolesError(
                f"the {self.month} panel's dataset has no bid_price or ask_price")
        return self.quotes

    @property
    def bid(self) -> NDArray[np.float64]:
        return self._quotes()[:, 0]

    @property
    def ask(self) -> NDArray[np.float64]:
        return self._quotes()[:, 1]

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
        """Quotes, half-spread and raw targets for aligned rows; NaN on a miss.

        ``quote`` and ``half_spread`` are absent, not NaN, for a panel without
        quotes.
        """
        idx = np.asarray(index, dtype=np.int64)
        hit = idx >= 0
        safe = np.where(hit, idx, 0)
        blank = lambda a: np.where(hit[:, None], a[safe], np.nan)  # noqa: E731
        out = {"raw_targets": blank(self.raw_targets), "matched": hit}
        if self.has_quotes:
            out["quote"] = blank(self.quotes)
            out["half_spread"] = np.where(hit, self.half_spread()[safe], np.nan)
        return out

    def market(self, index: ArrayLike, decisions: ArrayLike, assets: ArrayLike, *,
               decision_axis: ArrayLike, asset_axis: ArrayLike) -> MarketData:
        """The aligned rows as :class:`MarketData` on a ``(decision, asset)`` grid.

        ``index`` is :meth:`align`'s output for rows labelled by ``decisions``
        and ``assets``; the grid axes are ``decision_axis`` and ``asset_axis``.
        A missed row is NaN, a name not observed. A panel without quotes gives
        market data without bid and ask, which the execution stages refuse.
        """
        if not self.has_quotes:
            return MarketData.empty(decision_axis, asset_axis)
        taken = self.take(index)
        grid = to_grid(taken["quote"], decisions, assets,
                       decision_axis=decision_axis, asset_axis=asset_axis)
        return MarketData({"bid_price": grid[:, :, 0], "ask_price": grid[:, :, 1]},
                          decision_axis, asset_axis)


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


def market_from_days(days, *, anchors: ArrayLike | None = None,
                     roles: tuple[str, ...] | None = None) -> MarketData:
    """The market a day store recorded, as :class:`MarketData`, by role.

    ``days`` are day-record paths or :class:`~stable_finance.dataset.storage.daystore.DayRecord`
    objects. Each decision is a (date, anchor) with ``anchors`` in seconds past
    the origin (default: each day's own anchor grid) and is labelled by
    :func:`decision_labels`; assets are every ticker seen, sorted. A column is
    included for every role the store's schema maps (or just ``roles``), so a
    dataset's columns flow downstream unchanged and a stage that needs one it
    lacks is refused: a univariate store yields ``mid`` only, which supports
    frictionless portfolios and nothing priced at the touch.

    Values are the decision row's bar. A name before its first quote, or an
    anchor off the session, is NaN.
    """
    from stable_finance.dataset.storage.daystore import DayRecord

    records = [d if isinstance(d, DayRecord) else DayRecord(d) for d in days]
    if not records:
        raise ValueError("no days")
    schema = records[0].schema
    for record in records:
        if record.schema != schema:
            raise ValueError(f"{record.date}: schema differs from {records[0].date}")
    wanted = tuple(schema.role_map) if roles is None else tuple(roles)
    schema.require(*wanted, purpose="market_from_days")
    assets = np.asarray(sorted({str(t) for r in records for t in r.tickers}))
    asset_of = {ticker: i for i, ticker in enumerate(assets.tolist())}
    labels, blocks = [], []
    for record in records:
        grid = (record.anchors if anchors is None
                else np.asarray(anchors, dtype=np.int64))
        columns = [asset_of[str(t)] for t in record.tickers]
        for anchor in grid:
            values = record.market_at(int(anchor), wanted)
            block = np.full((len(wanted), len(assets)), np.nan)
            block[:, columns] = values
            blocks.append(block)
            labels.append((record.date, int(anchor)))
    stacked = np.stack(blocks, axis=1)                   # (roles, decisions, assets)
    decisions = decision_labels([d for d, _ in labels], [a for _, a in labels])
    return MarketData({role: stacked[k] for k, role in enumerate(wanted)}, decisions, assets)


def forward_returns_from_days(days, objective=None, *, target_type: str = "return",
                              transform: str = "raw", anchors: ArrayLike | None = None,
                              persist: bool = False, cache_root=None) -> ForwardReturns:
    """One target family of a day store as :class:`ForwardReturns`.

    The grid is :func:`market_from_days`'s -- decisions are (date, anchor)
    labels, assets every ticker sorted -- so the two align without a join.
    The horizon axis is the objective's horizons. ``objective`` defaults to
    each record's stored one; any other (a different horizon set, execution
    window or price) is computed from the stored features, and cached under
    its key when ``persist``. That is the execution-window sweep::

        for window in (60, 300, 900):
            realized = forward_returns_from_days(days, Objective(window_seconds=window))

    ``transform`` picks raw or a cross-sectional form. Horizons that measure
    to the close have no fixed length and are not accepted here.
    """
    from stable_finance.dataset.storage.daystore import DayRecord

    records = [d if isinstance(d, DayRecord) else DayRecord(d) for d in days]
    if not records:
        raise ValueError("no days")
    resolved = [(objective or r.default_objective).resolve(r.schema) for r in records]
    keys = {o.key for o in resolved}
    if len(keys) != 1:
        raise ValueError("the days resolve to different objectives; pass one explicitly")
    objective = resolved[0]
    if target_type not in objective.types:
        raise ValueError(f"{target_type!r} is not among the objective's types {objective.types}")
    if any(h <= 0 for h in objective.horizons):
        raise ValueError("ForwardReturns horizons must be fixed positive durations")
    ti = objective.types.index(target_type)
    assets = np.asarray(sorted({str(t) for r in records for t in r.tickers}))
    asset_of = {ticker: i for i, ticker in enumerate(assets.tolist())}
    labels, blocks = [], []
    for record in records:
        arrays = record.objective_arrays(objective, persist=persist, cache_root=cache_root)
        values = np.asarray(arrays[transform], dtype=np.float64)       # (A, N, T, H)
        anchor_index = {int(a): i for i, a in enumerate(record.anchors)}
        grid = record.anchors if anchors is None else np.asarray(anchors, dtype=np.int64)
        columns = [asset_of[str(t)] for t in record.tickers]
        for anchor in grid:
            block = np.full((len(assets), len(objective.horizons)), np.nan)
            row = anchor_index.get(int(anchor))
            if row is not None:
                block[columns] = values[row, :, ti, :]
            blocks.append(block)
            labels.append((record.date, int(anchor)))
    decisions = decision_labels([d for d, _ in labels], [a for _, a in labels])
    return ForwardReturns(np.stack(blocks), decisions, assets, np.asarray(objective.horizons))
