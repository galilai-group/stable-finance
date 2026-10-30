"""Physical-share execution of portfolio targets through upstream models.

The decision clock and price-mark clock are separate. Orders fix shares when
sent and arrive after an exact number of seconds, never after a rounded bar.
Only then is native ``execute`` called, using the market observed at arrival.
This is the existing local physical ledger moved into the package without a
runtime dependency on the preserved checkout. Default all-cash arithmetic is
retained, and a caller may explicitly supply initial marked holdings.
This adapter does not implement scheduled institutional or depth/limit fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Real
from operator import index
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import MarketData
from stable_finance.evaluation.execution import (
    ImpactStockCoefficients, Orders, execute, execution_requires,
)
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.scoring.sharpe import sharpe_ratio

__all__ = ["PricePanel", "PhysicalTradePolicy", "PhysicalExecutionConfig",
           "PortfolioOrderRecord", "PhysicalExecutionReport", "execute_portfolio_targets"]


def _integer(name: str, value, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        result = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _time_deltas_seconds(times: np.ndarray) -> np.ndarray:
    if np.issubdtype(times.dtype, np.datetime64):
        if np.isnat(times).any():
            raise ValueError("times cannot contain NaT")
        return np.asarray(np.diff(times) / np.timedelta64(1, "s"), dtype=float)
    if np.issubdtype(times.dtype, np.number) and not np.issubdtype(
        times.dtype, np.complexfloating
    ):
        numeric = np.asarray(times, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("times must be finite")
        return np.diff(numeric)
    if times.dtype == object and all(isinstance(value, datetime) for value in times):
        aware = [value.utcoffset() is not None for value in times]
        if any(aware) and not all(aware):
            raise ValueError("times cannot mix timezone-aware and naive datetimes")
        seconds = np.asarray([
            value.timestamp() if is_aware else (value - datetime(1970, 1, 1)).total_seconds()
            for value, is_aware in zip(times, aware)
        ])
        return np.diff(seconds)
    raise TypeError("times must be numeric seconds or datetime values")


def _time_axes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare native aware datetimes by instant, other labels exactly.

    Python compares datetimes sharing a timezone by wall time, ignoring a DST
    fold. Normalize only native aware datetime axes to UTC; do not convert
    numeric or numpy datetime64 labels to floating seconds and lose precision.
    """
    arrays = (np.asarray(left), np.asarray(right))
    native = [array.dtype == object and all(isinstance(value, datetime) for value in array)
              for array in arrays]
    awareness = []
    for array, is_native in zip(arrays, native):
        if not is_native:
            awareness.append(None)
            continue
        aware = [value.utcoffset() is not None for value in array]
        if any(aware) and not all(aware):
            raise ValueError("times cannot mix timezone-aware and naive datetimes")
        awareness.append(bool(aware and aware[0]))
    if all(native):
        if awareness[0] != awareness[1]:
            raise ValueError("times cannot mix timezone-aware and naive datetimes")
        if awareness[0]:
            return arrays[0].shape == arrays[1].shape and all(
                first.astimezone(timezone.utc) == second.astimezone(timezone.utc)
                for first, second in zip(*arrays)
            )
    elif any(awareness):
        other = arrays[1 if native[0] else 0]
        if np.issubdtype(other.dtype, np.datetime64):
            raise ValueError("times cannot mix timezone-aware and naive datetimes")
    return np.array_equal(*arrays)


@dataclass(frozen=True)
class PricePanel:
    """Complete positive physical price marks on an immutable time/asset grid.

    Numeric times are seconds. Datetime64 and native datetimes are accepted,
    with timezone awareness kept consistent. The caller supplies adjusted
    prices; missing held prices are never skipped or implicitly replaced.
    """

    values: ArrayLike
    times: ArrayLike
    assets: ArrayLike

    def __post_init__(self):
        raw = np.asarray(self.values)
        if raw.dtype.kind not in "iuf":
            raise ValueError("prices must contain real numbers")
        values = np.array(raw, dtype=float, copy=True)
        times = np.array(self.times, copy=True)
        assets = np.array(self.assets, copy=True)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times must be one-dimensional with at least two marks")
        if assets.ndim != 1 or not len(assets):
            raise ValueError("assets must be nonempty and one-dimensional")
        if any(value is None or (isinstance(value, Real) and not np.isfinite(value))
               or (isinstance(value, (str, np.str_)) and not value)
               or (isinstance(value, np.datetime64) and np.isnat(value)) for value in assets):
            raise ValueError("assets must contain nonmissing labels")
        try:
            unique = len(np.unique(assets)) == len(assets)
        except (TypeError, ValueError) as error:
            raise ValueError("assets must be comparable unique labels") from error
        if not unique:
            raise ValueError("assets must be unique")
        if values.shape != (len(times), len(assets)):
            raise ValueError("prices must have shape (len(times), len(assets))")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("prices must be complete, finite and strictly positive")
        if not np.all(_time_deltas_seconds(times) > 0):
            raise ValueError("times must be strictly increasing and unique")
        for name, array in (("values", values), ("times", times), ("assets", assets)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@runtime_checkable
class PhysicalTradePolicy(Protocol):
    """A trades callback with current actual and irrevocable pending holdings."""

    def validate(self, n_decisions: int, n_assets: int) -> None: ...

    def plan(self, target: ArrayLike, previous: ArrayLike, pending: ArrayLike,
             decision_index: int, n_decisions: int) -> ArrayLike: ...


@dataclass(frozen=True)
class PhysicalExecutionConfig:
    """Supported upstream execution assumption and physical arrival latency.

    ``impact``/``impact_stock`` with schedule zero apply upstream's empirical
    cost formula instantaneously; they do not simulate its one-day execution
    strategy. ``efq`` applies to cross only. Book, resting limits and positive
    schedules require richer fill state and are explicitly unsupported here.
    """

    model: str = "cross"
    delay_seconds: int = 5
    efq: float = 0.7
    schedule: int = 0
    impact: tuple[float, ...] | None = None
    limit: float | None = None

    def __post_init__(self):
        if self.model not in ("mid", "cross", "impact", "impact_stock"):
            raise ValueError("this ledger supports mid, cross, impact and impact_stock; "
                             "book/partial-depth execution is not implemented")
        _integer("delay_seconds", self.delay_seconds, minimum=0)
        if _integer("schedule", self.schedule, minimum=0) != 0:
            raise ValueError("scheduled institutional execution is not implemented; schedule must be 0")
        if self.limit is not None:
            raise ValueError("limit/resting execution is not implemented")
        if (isinstance(self.efq, (bool, np.bool_)) or not isinstance(self.efq, Real)
                or not np.isfinite(self.efq) or self.efq < 0):
            raise ValueError("efq must be a finite nonnegative scalar")
        if self.impact is not None:
            expected = len(ImpactStockCoefficients._fields) if self.model == "impact_stock" else 3
            coefficients = np.asarray(self.impact, dtype=float)
            if self.model not in ("impact", "impact_stock"):
                raise ValueError("impact coefficients apply only to impact models")
            if coefficients.shape != (expected,) or not np.isfinite(coefficients).all():
                raise ValueError(f"{self.model} impact coefficients must have {expected} finite entries")
            object.__setattr__(self, "impact", tuple(float(value) for value in coefficients))


@dataclass(frozen=True)
class PortfolioOrderRecord:
    """A sent order and the fill actually observed at its arrival boundary."""

    decision_time: object
    arrival_time: object
    decision_nav: float
    requested_shares: NDArray[np.float64]
    filled_shares: NDArray[np.float64]
    unfilled_shares: NDArray[np.float64]
    decision_prices: NDArray[np.float64]
    reference_prices: NDArray[np.float64]
    fill_prices: NDArray[np.float64]
    decision_shortfall: float
    arrival_slippage: float
    commission: float
    terminal: bool


@dataclass(frozen=True)
class PhysicalExecutionReport:
    """Currency cash/share ledger, with returns on nonoverlapping mark intervals.

    ``nav`` starts at initial_equity and compounds net_returns. Boundary states
    ``shares``, ``cash``, ``posttrade_nav`` include fills arriving there; nav
    instead records pretrade marks except at the final boundary. Final fills'
    costs are included once in the final interval. Slippage can be negative;
    commissions cannot. Decision shortfall is an attribution, not a second
    charge to NAV. trade_notional is signed actual fill dollars; mark_notional
    is its value at the accounting mark. Gross returns use the shares actually
    filled, not a second frictionless portfolio. All share quantities are
    physical, not weights.

    Decision-grid arrays preserve both stages: target_weights is the
    frictionless desired book; planned_weights is the trades policy's chosen
    committed book. actual_decision_weights and pending_decision_weights are
    marked after same-time arrivals and before sending the new order, using
    decision_nav. planned_trade_weights is planned minus actual minus pending.
    Order sizing fixes physical shares at that decision; future execution
    prices and fees do not retroactively rescale the plan.
    """

    times: NDArray
    assets: NDArray
    target_decisions: NDArray
    target_weights: NDArray[np.float64]
    planned_weights: NDArray[np.float64]
    actual_decision_weights: NDArray[np.float64]
    pending_decision_weights: NDArray[np.float64]
    planned_trade_weights: NDArray[np.float64]
    decision_nav: NDArray[np.float64]
    trade_policy: PhysicalTradePolicy | None
    config: PhysicalExecutionConfig
    initial_equity: float
    mark_step_seconds: float
    nav: NDArray[np.float64]
    pretrade_nav: NDArray[np.float64]
    posttrade_nav: NDArray[np.float64]
    cash: NDArray[np.float64]
    shares: NDArray[np.float64]
    posttrade_weights: NDArray[np.float64]
    requested_shares: NDArray[np.float64]
    filled_shares: NDArray[np.float64]
    unfilled_shares: NDArray[np.float64]
    pending_shares: NDArray[np.float64]
    trade_notional: NDArray[np.float64]
    mark_notional: NDArray[np.float64]
    fill_prices: NDArray[np.float64]
    arrival_slippage: NDArray[np.float64]
    decision_shortfall: NDArray[np.float64]
    commissions: NDArray[np.float64]
    commission_rates: NDArray[np.float64]
    gross_returns: NDArray[np.float64]
    costs: NDArray[np.float64]
    net_returns: NDArray[np.float64]
    order_records: tuple[PortfolioOrderRecord, ...]
    periods_per_year: float
    sharpe_gross: float
    sharpe_net: float
    total_return: float
    liquidated: bool
    initial_weights: NDArray[np.float64] | None = None


def _timestamp_keys(values):
    array = np.asarray(values)
    if array.dtype == object and all(isinstance(value, datetime) for value in array):
        aware = [value.utcoffset() is not None for value in array]
        if any(aware) and not all(aware):
            raise ValueError("times cannot mix timezone-aware and naive datetimes")
        if aware and aware[0]:
            return np.asarray([value.astimezone(timezone.utc) for value in array], dtype=object)
    return array


def _aligned_indices(grid, labels, name):
    try:
        positions = np.searchsorted(_timestamp_keys(grid), _timestamp_keys(labels))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be exact timestamps on the price grid") from error
    if np.any(positions >= len(grid)) or not _time_axes_equal(grid[positions], labels):
        raise ValueError(f"{name} must be exact timestamps on the price grid")
    if np.any(np.diff(positions) <= 0):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return positions


def _arrival_indices(times, send_indices, delay):
    keys = _timestamp_keys(times)
    if np.issubdtype(keys.dtype, np.datetime64):
        arrivals = keys[send_indices] + np.timedelta64(int(delay), "s")
    elif keys.dtype == object and all(isinstance(value, datetime) for value in keys):
        arrivals = np.asarray([value + timedelta(seconds=int(delay))
                               for value in keys[send_indices]], dtype=object)
    else:
        arrivals = keys[send_indices] + delay
    try:
        return _aligned_indices(np.asarray(times), arrivals, "arrival times")
    except ValueError as error:
        raise ValueError("price grid must contain each exact arrival time; "
                         "a five-second delay cannot be rounded to a minute bar") from error


def _current_market(market, marks):
    # exec_* in evaluation_inputs refer to a FUTURE touch relative to this
    # row. At a physical arrival event use current observations exactly once.
    # Do not derive a current touch from possibly delayed book columns either.
    names = ("mid", "close", "vwap", "bid_price", "ask_price", "half_spread",
             "adv", "market_cap", "idio_vol", "market_vol", "time_trend")
    columns = {name: market.columns[name] for name in names
               if name in market.columns}
    if "mid" not in columns and not {"bid_price", "ask_price"} <= columns.keys():
        if not any(name in columns for name in ("close", "vwap")):
            columns["mid"] = marks
    return MarketData(columns, market.decisions, market.assets,
                      market.estimated & columns.keys())


def _arrival_market(market, row):
    return MarketData({name: value[row:row + 1] for name, value in market.columns.items()},
                      market.decisions[row:row + 1], market.assets, market.estimated)


def execute_portfolio_targets(
    targets: PortfolioWeights,
    prices: PricePanel,
    market: MarketData | None = None,
    *,
    config: PhysicalExecutionConfig = PhysicalExecutionConfig(),
    trade_policy: PhysicalTradePolicy | None = None,
    initial_equity: float | None = None,
    initial_weights: ArrayLike | None = None,
    cost_bps: ArrayLike = 0.0,
    strategy_end=None,
    liquidate: bool = True,
    periods_per_year: float,
) -> PhysicalExecutionReport:
    """Send one net book's targets and settle upstream fills into real cash.

    Targets have exactly one horizon; their decision grid may be coarser than
    the uniformly spaced prices. MarketData, when supplied, is aligned to *all* price marks, with
    each row's raw quotes/ADV/characteristics observable at that row's time.
    Precomputed delayed exec_* quotes are ignored. Exact arrival marks are
    mandatory; missing quotes produce recorded unfilled orders, never free
    fills. At a decision, desired shares use the NAV and price then known,
    minus actual and outstanding shares; future fills cannot resize an order.

    The optional trades-stage policy receives the combined target and the
    actual and outstanding shares marked at the decision price and NAV. Its
    planned weights replace the desired holding for order sizing only; the
    frictionless target remains separately reported. Estimated policy costs
    affect the decision, while only actual fills debit cash. A pending order
    is committed and cannot be cancelled here: the policy prices additional
    trades relative to actual plus pending holdings. Forced terminal closing
    bypasses the policy, so a high estimated cost cannot prevent liquidation.

    ``initial_weights`` optionally starts with physical holdings marked at the
    first price boundary and initial_equity. The remaining equity is cash;
    no externally pending orders are invented. Omission preserves the original
    all-cash starting state exactly.

    ``cost_bps`` is an additional commission on absolute executed fill dollars.
    Impact models require initial_equity explicitly in the ADV's currency.
    Other models default to one unit. ``strategy_end`` is the timestamp when
    liquidation is SENT; all targets must precede it. The price grid must
    continue through its delayed arrival. With delay zero it defaults to the
    final mark. No interest, dividends, borrow fees or corporate actions are
    invented. PricePanel must already use consistent adjusted prices.
    """
    if not isinstance(config, PhysicalExecutionConfig):
        raise TypeError("config must be PhysicalExecutionConfig")
    if trade_policy is not None and (not isinstance(trade_policy, PhysicalTradePolicy)
            or not callable(trade_policy.validate) or not callable(trade_policy.plan)):
        raise TypeError("trade_policy must implement callable validate and plan methods")
    if not isinstance(liquidate, (bool, np.bool_)):
        raise TypeError("liquidate must be a boolean")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    if initial_equity is None:
        if config.model in ("impact", "impact_stock"):
            raise ValueError("impact models require explicit initial_equity in the ADV's currency")
        initial_equity = 1.0
    if (isinstance(initial_equity, (bool, np.bool_)) or not isinstance(initial_equity, Real)
            or not np.isfinite(initial_equity) or initial_equity <= 0):
        raise ValueError("initial_equity must be finite and positive")
    if len(targets.horizons) != 1:
        raise ValueError("execution takes one net target book (one horizon), not independent books")
    desired = np.asarray(targets.values[:, :, 0], dtype=float)
    if not np.isfinite(desired).all():
        raise ValueError("target weights must be finite; missing targets must be resolved before execution")
    if not np.array_equal(targets.assets, prices.assets):
        raise ValueError("target and price assets are not aligned")
    if not len(targets.decisions):
        raise ValueError("targets must contain at least one decision")
    if trade_policy is not None:
        trade_policy.validate(len(targets.decisions), len(targets.assets))
    mark_deltas = _time_deltas_seconds(prices.times)
    if not np.allclose(mark_deltas, mark_deltas[0], rtol=1e-10, atol=0.0):
        raise ValueError("price marks must have uniform spacing for per-mark annualization")
    decisions = _aligned_indices(prices.times, targets.decisions, "target decisions")
    arrivals = _arrival_indices(prices.times, decisions, config.delay_seconds)
    if market is None:
        market = MarketData({"mid": prices.values}, prices.times, prices.assets)
    if not (_time_axes_equal(market.decisions, prices.times)
            and np.array_equal(market.assets, prices.assets)):
        raise ValueError("market must align with every price mark and asset")
    market = _current_market(market, prices.values)
    market.require(*execution_requires(config.model), purpose="physical portfolio execution")
    count, assets = prices.values.shape
    rates = np.asarray(cost_bps, dtype=float)
    if rates.ndim != 0 and rates.shape not in ((assets,), (count, assets)):
        raise ValueError("cost_bps must be scalar, one per asset, or one per price mark and asset")
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("cost_bps must be finite and nonnegative")
    rates = np.broadcast_to(rates, (count, assets)).copy() / 1e4
    close_send = close_arrival = None
    if liquidate:
        if strategy_end is None:
            if config.delay_seconds:
                raise ValueError("delayed liquidation needs strategy_end and marks through its arrival")
            strategy_end = prices.times[-1]
        close_send = int(_aligned_indices(prices.times, np.asarray([strategy_end]), "strategy_end")[0])
        if close_send <= decisions[-1]:
            raise ValueError("strategy_end must be after every target decision")
        close_arrival = int(_arrival_indices(prices.times, np.asarray([close_send]), config.delay_seconds)[0])
    elif strategy_end is not None:
        end_index = int(_aligned_indices(prices.times, np.asarray([strategy_end]), "strategy_end")[0])
        if end_index < decisions[-1]:
            raise ValueError("strategy_end cannot precede target decisions")

    share_path = np.zeros((count, assets))
    cash_path, pre_nav, post_nav = (np.zeros(count) for _ in range(3))
    requested, filled, missed, pending_path = (np.zeros((count, assets)) for _ in range(4))
    fill_prices = np.full((count, assets), np.nan)
    slippage, shortfall, commissions = (np.zeros(count) for _ in range(3))
    if initial_weights is None:
        start_weights = np.zeros(assets)
        held, pending, cash = np.zeros(assets), np.zeros(assets), float(initial_equity)
    else:
        raw_start = np.asarray(initial_weights)
        if raw_start.dtype.kind not in "iuf" or raw_start.shape != (assets,):
            raise ValueError("initial_weights must be a finite asset vector")
        start_weights = np.array(raw_start, dtype=float, copy=True)
        if not np.isfinite(start_weights).all():
            raise ValueError("initial_weights must be a finite asset vector")
        held = float(initial_equity) * start_weights / prices.values[0]
        pending = np.zeros(assets)
        cash = float(initial_equity) * (1.0-float(start_weights.sum()))
    start_weights.setflags(write=False)
    records, queue = [], {}
    terminal_failed = False
    rows = {int(row): j for j, row in enumerate(decisions)}
    planned = np.zeros_like(desired)
    actual_decision, pending_decision, planned_trades = (
        np.zeros_like(desired) for _ in range(3)
    )
    decision_nav = np.zeros(len(decisions))

    def mark_nav(row):
        value = float(cash + np.dot(held, prices.values[row]))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("portfolio NAV must remain finite and positive")
        return value

    def settle(row, packet):
        nonlocal held, pending, cash, terminal_failed
        sent, sent_nav, quantities, terminal = packet
        # Queue contents are authoritative. Reconstruct instead of repeatedly
        # subtracting floats, which otherwise leaves spurious share balances
        # after large-currency orders have all arrived.
        pending = sum((item[2] for item in queue.values()), np.zeros(assets))
        if not np.any(quantities):
            return
        observed = _arrival_market(market, row)
        # The upstream reference is the unit in which Orders/Fills.quantity
        # are weights. Convert physical shares into it and back consistently.
        reference = next((observed.get(name)[0] for name in ("mid", "close", "vwap")
                          if observed.has(name)), None)
        if reference is None:
            raise ValueError("current execution market has no reference price")
        # A missing quote still cannot fill. Accounting marks only keep the
        # normalization finite for those unfilled assets; never overwrite an
        # explicitly missing execution reference, or another asset's valid one.
        valid_reference = np.isfinite(reference) & (reference > 0)
        normalization = np.where(valid_reference, reference, prices.values[row])
        equity = mark_nav(row)
        delta_weight = quantities * normalization / equity
        order = Orders(delta_weight[None, :, None], observed.decisions,
                       prices.assets, np.asarray([1]))
        fills = execute(order, observed, model=config.model, efq=config.efq,
                        equity=equity, schedule=config.schedule, impact=config.impact)
        if terminal and not np.array_equal(fills.quantity, order.values):
            terminal_failed = True
        executed = fills.quantity[0, :, 0] * equity / normalization
        px = fills.price[0, :, 0]
        traded = executed != 0
        paid_prices = np.where(traded, px, 0.0)
        charge = float(np.dot(rates[row], np.abs(executed * paid_prices)))
        arrival_cost = float(np.dot(executed, np.where(traded, px - prices.values[row], 0.0)))
        decision_cost = float(np.dot(executed, np.where(traded, px - prices.values[sent], 0.0)))
        cash -= float(np.dot(executed, paid_prices)) + charge
        held += executed
        filled[row] += executed
        missed[row] += quantities - executed
        fill_prices[row] = np.where(traded, px, fill_prices[row])
        commissions[row] += charge
        slippage[row] += arrival_cost
        shortfall[row] += decision_cost
        records.append(PortfolioOrderRecord(
            decision_time=prices.times[sent], arrival_time=prices.times[row],
            decision_nav=sent_nav, requested_shares=quantities.copy(),
            filled_shares=executed.copy(), unfilled_shares=(quantities - executed).copy(),
            decision_prices=prices.values[sent].copy(),
            reference_prices=fills.mid_price[0, :, 0].copy(),
            fill_prices=px.copy(), decision_shortfall=decision_cost,
            arrival_slippage=arrival_cost, commission=charge, terminal=terminal,
        ))
        mark_nav(row)

    def send(row, arrival, weights, terminal=False):
        nonlocal pending
        equity = mark_nav(row)
        quantity = equity * weights / prices.values[row] - held - pending
        requested[row] += quantity
        pending += quantity
        packet = (row, equity, quantity.copy(), terminal)
        if arrival == row:
            settle(row, packet)
        else:
            queue[arrival] = packet

    for row in range(count):
        pre_nav[row] = mark_nav(row)
        if row in queue:
            settle(row, queue.pop(row))
        if row in rows:
            j = rows[row]
            equity = mark_nav(row)
            actual_decision[j] = held * prices.values[row] / equity
            pending_decision[j] = pending * prices.values[row] / equity
            decision_nav[j] = equity
            chosen = (desired[j] if trade_policy is None else trade_policy.plan(
                desired[j], actual_decision[j], pending_decision[j], j, len(decisions),
            ))
            chosen = np.asarray(chosen, dtype=float)
            if chosen.shape != (assets,) or not np.isfinite(chosen).all():
                raise ValueError("trade_policy must return one finite planned weight per asset")
            planned[j] = chosen
            planned_trades[j] = planned[j] - actual_decision[j] - pending_decision[j]
            send(row, int(arrivals[j]), planned[j])
        if close_send is not None and row == close_send:
            send(row, close_arrival, np.zeros(assets), terminal=True)
        cash_path[row], share_path[row], pending_path[row] = cash, held, pending
        post_nav[row] = mark_nav(row)
    if queue:
        raise ValueError("price grid ended before outstanding orders arrived")
    share_roundoff = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(filled).sum(axis=0))
    is_flat = bool(np.all(np.abs(held) <= share_roundoff))
    if liquidate and (terminal_failed or not is_flat):
        raise ValueError("terminal liquidation did not fill completely; portfolio remains open")
    gross = np.sum(share_path[:-1] * np.diff(prices.values, axis=0), axis=1) / pre_nav[:-1]
    costs = (slippage[:-1] + commissions[:-1]) / pre_nav[:-1]
    costs[-1] += (slippage[-1] + commissions[-1]) / pre_nav[-2]
    net = gross - costs
    nav = pre_nav.copy()
    nav[-1] = post_nav[-1]
    return PhysicalExecutionReport(
        times=prices.times.copy(), assets=prices.assets.copy(),
        target_decisions=targets.decisions.copy(), target_weights=desired.copy(),
        planned_weights=planned, actual_decision_weights=actual_decision,
        pending_decision_weights=pending_decision, planned_trade_weights=planned_trades,
        decision_nav=decision_nav, trade_policy=trade_policy,
        config=config, initial_equity=float(initial_equity), mark_step_seconds=float(mark_deltas[0]),
        nav=nav,
        pretrade_nav=pre_nav, posttrade_nav=post_nav, cash=cash_path, shares=share_path,
        posttrade_weights=share_path * prices.values / post_nav[:, None],
        requested_shares=requested, filled_shares=filled, unfilled_shares=missed,
        pending_shares=pending_path,
        trade_notional=filled * np.nan_to_num(fill_prices, nan=0.0),
        mark_notional=filled * prices.values,
        fill_prices=fill_prices, arrival_slippage=slippage, decision_shortfall=shortfall,
        commissions=commissions, commission_rates=rates, gross_returns=gross,
        costs=costs, net_returns=net, order_records=tuple(records),
        periods_per_year=float(periods_per_year),
        sharpe_gross=sharpe_ratio(gross, periods_per_year=periods_per_year),
        sharpe_net=sharpe_ratio(net, periods_per_year=periods_per_year),
        total_return=float(post_nav[-1] / initial_equity - 1.0),
        liquidated=bool(liquidate and is_flat), initial_weights=start_weights,
    )
