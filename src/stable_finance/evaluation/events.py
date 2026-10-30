"""Event-time trading: round trips at arbitrary times, scored by the same harness.

The grid harness (:func:`~stable_finance.evaluation.backtest.run_backtest`)
takes a book on a fixed lattice of decisions. An event strategy -- buy on a
news story at 10:41:07, sell at the close -- has no lattice: its decisions
are whenever something happened. :func:`event_backtest` takes the trades as
a list of round trips and a tape of quotes, builds the grid they imply, and
hands it to ``run_backtest`` with ``holding="drift"``, so every round trip
earns exactly its price change and pays exactly its own fills.

    trades = Trades(assets, entry, exit, size)              # epoch seconds
    result = event_backtest(trades, tape, config=DEFAULT_BACKTEST.replace(efq=0.7))
    result.report.sharpe_net, result.ledger.net

THE GRID. Decisions are every entry and exit time. Each trade gets its own
column (a SLOT), reused by a later trade only once its trade is closed, so
two trades on one name at once are two positions, and the grid is as wide
as the most trades ever open together rather than the number of trades.

THE MARKET, per slot while its trade is open, from the tape (the last quote
at or before each time, as :func:`~stable_finance.evaluation.inputs.evaluation_inputs`
reads it): ``bid_price`` / ``ask_price`` at each decision, from which
:class:`~stable_finance.dataset.market.MarketData` derives the mid and the
half-spread (a crossed, one-sided or zero-bid quote is unquoted), and
``exec_bid_price`` / ``exec_ask_price`` ``delay_seconds`` later, the touch an
order sent at the decision meets. A trade with an ``exit_price`` exits there
with no spread -- an auction, e.g. the closing cross -- otherwise at the
touch, at the first decision from its exit time on that the name quotes.

THE RETURNS. Each slot's mid-to-mid return from one decision to the next
while open (a missing mid holds the last good one; the last leg ends at the
exit price when there is one). Under drift holding these compound to
``exit / entry - 1`` per trade.

THE COST. The configured cost model at each fill (``cross``: EFQ times the
half-spread at the arrival touch, measured against the decision's mid; a
per-trade ``efq`` overrides the config's), plus the configuration's fees on
each fill (``fee_bps``, ``fee_per_share`` at the decision's mid,
``sec_fee_rate`` on the sale). An entry that cannot fill is retried at the
next decision the name quotes.
"""

from __future__ import annotations

import heapq
import datetime as dt
from dataclasses import dataclass
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.anchors import TO_CLOSE
from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.backtest import BacktestReport, run_backtest
from stable_finance.evaluation.config import DEFAULT_BACKTEST, BacktestConfig
from stable_finance.evaluation.execution import _efq_array, fee_cost, fill_path
from stable_finance.evaluation.inputs import epoch_seconds
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.scoring.pnl import positions_from_fills

__all__ = ["EventLedger", "EventResult", "Trades", "event_backtest", "event_inputs"]


@dataclass(frozen=True)
class Trades:
    """Round trips: ``size`` (signed, a fraction of equity) of ``assets`` from ``entry`` to ``exit``.

    Times are epoch seconds (or anything :func:`epoch_seconds` reads).
    ``exit_price``, where finite, is an auction price the trade exits at with
    no spread (e.g. the closing cross, with ``exit`` the session close); NaN
    exits at the touch. ``efq``, if given, is each trade's own execution
    quality in place of the configuration's.
    """

    assets: NDArray
    entry: NDArray[np.int64]
    exit: NDArray[np.int64]
    size: NDArray[np.float64]
    exit_price: NDArray[np.float64] | None = None
    efq: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        assets = np.asarray(self.assets).astype(str)
        entry, exit_ = epoch_seconds(self.entry), epoch_seconds(self.exit)
        size = np.asarray(self.size, dtype=np.float64)
        n = len(assets)
        if assets.ndim != 1 or not n:
            raise ValueError("trades need a non-empty 1-D array of assets")
        for name, value in (("entry", entry), ("exit", exit_), ("size", size)):
            if value.shape != (n,):
                raise ValueError(f"{name} must have one value per trade, shape ({n},)")
        if not np.isfinite(size).all():
            raise ValueError("size must be finite")
        if (exit_ <= entry).any():
            raise ValueError("every trade must exit after it enters")
        price = None
        if self.exit_price is not None:
            price = np.asarray(self.exit_price, dtype=np.float64)
            if price.shape != (n,) or (np.isfinite(price) & (price <= 0)).any():
                raise ValueError("exit_price must be one positive price (or NaN) per trade")
        efq = None if self.efq is None else _efq_array(self.efq)
        if efq is not None and efq.shape != (n,):
            raise ValueError("efq must be one value per trade")
        for name, value in (("assets", assets), ("entry", entry), ("exit", exit_), ("size", size),
                            ("exit_price", price), ("efq", efq)):
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.assets)


@dataclass(frozen=True)
class EventLedger:
    """Per trade, in the order given: where it filled and what it earned, as returns on equity."""

    slot: NDArray[np.int64]
    entry_fill_time: NDArray[np.int64]     # -1 where the entry never filled
    exit_fill_time: NDArray[np.int64]      # -1 where the exit never filled (still open)
    entry_price: NDArray[np.float64]
    exit_price: NDArray[np.float64]
    gross: NDArray[np.float64]
    cost: NDArray[np.float64]

    @property
    def net(self) -> NDArray[np.float64]:
        return self.gross - self.cost


@dataclass(frozen=True)
class EventResult:
    report: BacktestReport
    ledger: EventLedger


def _quoted(bid, ask):
    return np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)


class _Tape:
    """Quotes by asset, read as of a time (the last at or before it)."""

    def __init__(self, frame: Mapping, time: str, asset: str, bid: str, ask: str):
        stamps = epoch_seconds(frame[time])
        names = np.asarray(frame[asset]).astype(str)
        order = np.lexsort((stamps, names))
        self.names, self.stamps = names[order], stamps[order]
        self.bid = np.asarray(frame[bid], dtype=np.float64)[order]
        self.ask = np.asarray(frame[ask], dtype=np.float64)[order]
        uniq, start = np.unique(self.names, return_index=True)
        self.span = dict(zip(uniq, zip(start, np.r_[start[1:], len(names)])))

    def asof(self, name: str, times: NDArray[np.int64]):
        lo, hi = self.span.get(name, (0, 0))
        i = np.searchsorted(self.stamps[lo:hi], times, side="right") - 1
        ok = i >= 0
        j = lo + np.maximum(i, 0)
        return np.where(ok, self.bid[j], np.nan), np.where(ok, self.ask[j], np.nan)


def _slots(start: NDArray[np.int64], end: NDArray[np.int64]) -> NDArray[np.int64]:
    """Column per trade over decision indices [start, end]; a column is reused only after its trade's end."""
    slot = np.empty(len(start), np.int64)
    free: list[int] = []
    busy: list[tuple[int, int]] = []              # (end, slot)
    n = 0
    for i in np.argsort(start, kind="stable"):
        while busy and busy[0][0] < start[i]:
            heapq.heappush(free, heapq.heappop(busy)[1])
        if free:
            slot[i] = heapq.heappop(free)
        else:
            slot[i], n = n, n + 1
        heapq.heappush(busy, (int(end[i]), int(slot[i])))
    return slot


def _days(decisions: NDArray[np.int64], timezone: str) -> NDArray[np.str_]:
    zone = ZoneInfo(timezone)
    return np.array([dt.datetime.fromtimestamp(int(t), zone).strftime("%Y-%m-%d") for t in decisions])


def event_inputs(trades: Trades, tape: Mapping, *, time: str = "timestamp", asset: str = "ticker",
                 bid: str = "bid_price", ask: str = "ask_price", delay_seconds: int = 5,
                 max_exit_wait: int = 1_000_000):
    """The grid an event backtest runs on: ``(weights, realized, market, slot, exit index, efq)``.

    ``weights`` holds each trade's size in its slot from its entry decision
    to the decision before its exit; ``realized`` the slot's decision-to-
    decision mid return while open; ``market`` the quotes (see the module
    docstring). A trade exiting at the touch whose name does not quote at its
    exit time is held to the first later decision it does, at most
    ``max_exit_wait`` seconds on.
    """
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")
    reader = _Tape(tape, time, asset, bid, ask)
    auction = (np.isfinite(trades.exit_price) if trades.exit_price is not None
               else np.zeros(len(trades), bool))
    decisions = np.unique(np.r_[trades.entry, trades.exit])
    a = np.searchsorted(decisions, trades.entry)
    b = np.searchsorted(decisions, trades.exit)
    # a touch exit waits for a quote: move its exit to the first decision from then on that has one;
    # a trade that never finds one stays open, and its slot is never handed to another trade
    stuck = ~auction.copy()
    for i in np.flatnonzero(~auction):
        js = np.arange(b[i], len(decisions))
        js = js[decisions[js] <= trades.exit[i] + max_exit_wait]
        xb, xa = reader.asof(trades.assets[i], decisions[js] + delay_seconds)
        mb, ma = reader.asof(trades.assets[i], decisions[js])
        ok = np.flatnonzero(_quoted(xb, xa) & _quoted(mb, ma))
        if len(ok):
            b[i], stuck[i] = js[ok[0]], False
    slot = _slots(a, np.where(stuck, len(decisions), b))
    n_d, n_a = len(decisions), int(slot.max()) + 1
    cols = {q: np.full((n_d, n_a), np.nan) for q in ("bid_price", "ask_price", "exec_bid_price", "exec_ask_price")}
    weights, realized = np.zeros((n_d, n_a)), np.full((n_d, n_a), np.nan)
    efq = None if trades.efq is None else np.zeros((n_d, n_a))
    for i in range(len(trades)):
        s, js = slot[i], np.arange(a[i], b[i] + 1)
        qb, qa = reader.asof(trades.assets[i], decisions[js])
        xb, xa = reader.asof(trades.assets[i], decisions[js] + delay_seconds)
        if auction[i]:
            qb[-1] = qa[-1] = xb[-1] = xa[-1] = trades.exit_price[i]
        cols["bid_price"][js, s], cols["ask_price"][js, s] = qb, qa
        cols["exec_bid_price"][js, s], cols["exec_ask_price"][js, s] = xb, xa
        mid = np.where(_quoted(qb, qa), (qb + qa) / 2, np.nan)
        good = np.flatnonzero(np.isfinite(mid))
        if len(good):                                          # a missing mid holds the last good one
            last = np.maximum.accumulate(np.where(np.isfinite(mid), np.arange(len(mid)), -1))
            mid = np.where(last >= 0, mid[np.maximum(last, 0)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            realized[js[:-1], s] = mid[1:] / mid[:-1] - 1
        weights[a[i]:b[i], s] = trades.size[i]
        if efq is not None:
            efq[js, s] = trades.efq[i]
    horizon = np.array([TO_CLOSE if auction.all() else int(np.median(trades.exit - trades.entry))])
    assets = np.array([f"slot{k}" for k in range(n_a)])
    return (PortfolioWeights(weights[:, :, None], decisions, assets, horizon),
            ForwardReturns(realized[:, :, None], decisions, assets, horizon),
            MarketData(cols, decisions, assets), slot, a, b, efq)


def event_backtest(trades: Trades, tape: Mapping, *, config: BacktestConfig = DEFAULT_BACKTEST,
                   time: str = "timestamp", asset: str = "ticker", bid: str = "bid_price",
                   ask: str = "ask_price", delay_seconds: int = 5, timezone: str = "America/New_York",
                   with_series: bool = False) -> EventResult:
    """Trade ``trades`` against ``tape`` under ``config`` (its cost model and EFQ; ``holding`` is set to drift).

    Days, for daily annualization, are calendar dates in ``timezone``.
    Returns the harness's report and a per-trade :class:`EventLedger`.
    """
    if config.cost_model not in ("mid", "cross"):
        raise ValueError("event_backtest prices fills at the mid or the touch: cost_model 'mid' or 'cross'")
    config = config.replace(holding="drift", rebalance="every_decision")
    weights, realized, market, slot, a, b, efq = event_inputs(
        trades, tape, time=time, asset=asset, bid=bid, ask=ask, delay_seconds=delay_seconds)
    days = _days(weights.decisions, timezone)
    report = run_backtest(weights, realized, market=market, days=days, config=config, efq=efq,
                          with_series=with_series)
    # the same fills, kept for the ledger
    fills = fill_path(weights, market, model=config.cost_model, efq=config.efq if efq is None else efq,
                      drift=realized.values)
    held = positions_from_fills(fills, drift=realized.values).values[:, :, 0]
    q, price, mid = fills.quantity[:, :, 0], fills.price[:, :, 0], fills.mid_price[:, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        slip = np.where(q != 0, (price - mid) / mid, 0.0)
    # Each fill's fees, as the report charged them, so a trade's ledger cost
    # sums with the others' to the report's.
    fee = (fee_cost(q, market.get("mid") if config.fee_per_share is not None else None,
                    **config.fees) if config.fees else None)
    pnl = np.where(np.isfinite(realized.values[:, :, 0]), held * realized.values[:, :, 0], 0.0)
    n = len(trades)
    out = {k: np.full(n, np.nan) for k in ("entry_price", "exit_price", "gross", "cost")}
    t_in, t_out = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for i in range(n):
        s, js = slot[i], np.arange(a[i], b[i] + 1)
        traded = js[q[js, s] != 0]
        if len(traded):
            t_in[i], out["entry_price"][i] = weights.decisions[traded[0]], price[traded[0], s]
        if len(traded) > 1:
            t_out[i], out["exit_price"][i] = weights.decisions[traded[-1]], price[traded[-1], s]
        out["gross"][i] = pnl[js, s].sum()
        out["cost"][i] = (q[js, s] * slip[js, s]).sum()
        if fee is not None:
            out["cost"][i] += fee[js, s].sum()
    return EventResult(report, EventLedger(slot, t_in, t_out, out["entry_price"], out["exit_price"],
                                           out["gross"], out["cost"]))
