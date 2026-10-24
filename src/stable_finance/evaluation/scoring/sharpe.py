"""How good a portfolio's return series is: Sharpe ratios and their EFQ curve.

The Sharpe ratio of a periodic return series, the Sharpe of a weight path
marked at mid and after crossing the quoted spread, and the net Sharpe swept
across execution quality with its break-even EFQ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from stable_finance.evaluation._grid import require_aligned
from stable_finance.dataset.market import (
    MarketData,
    market_quantity,
    requires,
    _validated_quotes,
)
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns


def sharpe_ratio(returns: ArrayLike, *, periods_per_year: float = 252.0) -> float:
    """Annualized arithmetic Sharpe ratio of a periodic return series."""
    values = np.asarray(returns, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    volatility = values.std(ddof=1)
    if volatility <= 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * values.mean() / volatility)


def _portfolio_returns_at_mid(
    weights: PortfolioWeights,
    realized_mid_returns: ForwardReturns,
) -> tuple[np.ndarray, ...]:
    """Return portfolio P&L marked at the mid price for every horizon."""
    require_aligned(weights, realized_mid_returns)
    results = []
    for horizon in range(len(weights.horizons)):
        weight = weights.values[:, :, horizon]
        outcome = realized_mid_returns.values[:, :, horizon]
        valid = np.isfinite(weight) & np.isfinite(outcome)
        returns = np.where(valid, weight * outcome, 0.0).sum(axis=1)
        returns[~valid.any(axis=1)] = np.nan
        results.append(returns)
    return tuple(results)


@requires()
def mid_price_sharpe(
    weights: PortfolioWeights,
    realized_mid_returns: ForwardReturns,
    *,
    periods_per_year: float = 252.0,
) -> tuple[float, ...]:
    """Annualized Sharpe at the mid price, independently per horizon.

    ``realized_mid_returns`` must contain mid-to-mid forward returns. This is
    the frictionless result: no spread, impact, or commission is charged.
    """
    return tuple(
        sharpe_ratio(returns, periods_per_year=periods_per_year)
        for returns in _portfolio_returns_at_mid(weights, realized_mid_returns)
    )


@requires("bid_price", "ask_price")
def cross_spread_sharpe(
    weights: PortfolioWeights,
    realized_mid_returns: ForwardReturns,
    best_bid: ArrayLike | MarketData,
    best_ask: ArrayLike | None = None,
    *,
    periods_per_year: float = 252.0,
) -> tuple[float, ...]:
    """Annualized Sharpe after crossing the quoted spread on every trade.

    Bid and ask must have shape ``(n_decisions, n_assets)`` and represent the
    observable quote at each rebalance. A positive weight change buys at the
    best ask; a negative weight change sells at the best bid. The execution
    price is compared with that quote's midpoint and charged against mid-marked
    P&L. The first portfolio is entered from cash. Market impact, quote depth,
    partial fills, and commissions remain outside this metric.

    :class:`MarketData` may stand in for both quotes; one without
    ``bid_price`` and ``ask_price`` is refused.
    """
    if isinstance(best_bid, MarketData):
        if best_ask is not None:
            raise ValueError("pass market data or two quote arrays, not both")
        market = best_bid
        best_bid = market_quantity(market, "bid_price", stage=cross_spread_sharpe,
                                   like=weights)
        best_ask = market.get("ask_price")
    elif best_ask is None:
        raise TypeError("cross_spread_sharpe needs best_ask, or market data in place of both")
    gross_returns = _portfolio_returns_at_mid(weights, realized_mid_returns)
    expected = weights.values.shape[:2]
    bid, ask = _validated_quotes(best_bid, best_ask)
    if bid.shape != expected:
        raise ValueError(f"best_bid and best_ask must have shape {expected}")

    sharpes = []
    for horizon, gross in enumerate(gross_returns):
        weight = weights.values[:, :, horizon]
        previous = np.vstack([np.zeros((1, expected[1])), weight[:-1]])
        trade = weight - previous
        traded = np.isfinite(trade) & (trade != 0)
        valid_quote = (
            np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
        )
        missing_execution_quote = traded & ~valid_quote

        midpoint = (bid + ask) / 2
        execution_price = np.where(trade > 0, ask, bid)
        # Signed trade makes both directions a positive cost: buys execute
        # above mid and sells (negative trade) execute below mid.
        execution_cost = trade * (execution_price - midpoint) / midpoint
        costs = np.where(traded & valid_quote, execution_cost, 0.0).sum(axis=1)
        net = gross - costs
        net[missing_execution_quote.any(axis=1)] = np.nan
        sharpes.append(sharpe_ratio(net, periods_per_year=periods_per_year))
    return tuple(sharpes)


@dataclass(frozen=True)
class ExecutionQualityCurve:
    """Net Sharpe as a function of EFQ, the effective-to-quoted spread ratio.

    ``efq`` is a fraction, as it is used as a multiplier; the literature
    quotes EFQ as a percent and figures should display it that way.
    """

    efq: np.ndarray
    sharpe: np.ndarray
    standard_error: np.ndarray
    gross_sharpe: float
    breakeven: float
    periods: int

    def at(self, efq: float) -> float:
        """Net Sharpe at one EFQ, interpolated within the swept grid.

        NaN outside it rather than the nearest endpoint: ``np.interp`` clamps,
        which would report a comfortably positive Sharpe at an execution
        quality the sweep never visited.
        """
        if efq < self.efq[0] or efq > self.efq[-1]:
            return float("nan")
        return float(np.interp(efq, self.efq, self.sharpe))


def sharpe_vs_execution_quality(
    gross: ArrayLike,
    cost_at_unit_efq: ArrayLike,
    *,
    efq_grid: ArrayLike | None = None,
    periods_per_year: float = 252.0,
) -> ExecutionQualityCurve:
    """Sweep net Sharpe across execution quality, from one backtest.

    ``gross`` and ``cost_at_unit_efq`` are per-period portfolio returns and the
    per-period cost that would be charged at ``EFQ = 1``. Because cost is
    linear in EFQ, the whole curve is ``gross - efq * cost`` and needs no
    re-running: one backtest produces the entire sensitivity.

    This is the honest way to present a Sharpe ratio that a cost assumption
    can dominate. Rather than defending one number, it reports where the
    result lives as a function of the single assumption that matters, and
    ``breakeven`` says where it dies. A curve whose break-even sits inside
    ``EFQ_INFORMED_RANGE`` is the quantitative form of "there is real
    predictive structure here and the market charges very nearly all of it".

    The standard error is Lo (2002) for an annualized Sharpe over the sample's
    own length, so a reader can see directly that a result of this size cannot
    be distinguished from zero at this sample length -- which is a property of
    the horizon available, not a flaw in the estimate.
    """
    gross = np.asarray(gross, dtype=np.float64).ravel()
    cost = np.asarray(cost_at_unit_efq, dtype=np.float64).ravel()
    if gross.shape != cost.shape:
        raise ValueError("gross and cost_at_unit_efq must have the same shape")
    keep = np.isfinite(gross) & np.isfinite(cost)
    gross, cost = gross[keep], cost[keep]
    if len(gross) < 2:
        raise ValueError("need at least two periods")
    if np.any(cost < 0):
        raise ValueError("cost cannot be negative")

    grid = (np.linspace(0.0, 1.5, 61) if efq_grid is None
            else np.asarray(efq_grid, dtype=np.float64).ravel())
    years = len(gross) / periods_per_year
    sharpe = np.array([sharpe_ratio(gross - ratio * cost,
                                    periods_per_year=periods_per_year)
                       for ratio in grid])
    # Lo (2002): the SE of an annualized Sharpe grows with the ratio itself,
    # so the band widens exactly where the curve is most flattering.
    standard_error = np.sqrt((1.0 + sharpe ** 2 / 2.0) / years)
    return ExecutionQualityCurve(
        efq=grid,
        sharpe=sharpe,
        standard_error=standard_error,
        gross_sharpe=sharpe_ratio(gross, periods_per_year=periods_per_year),
        breakeven=breakeven_efq(gross, cost),
        periods=int(len(gross)),
    )


def breakeven_efq(gross: ArrayLike, cost_at_unit_efq: ArrayLike) -> float:
    """The EFQ at which the expected net return reaches zero.

    Costs are linear in EFQ, so a strategy's whole sensitivity to execution
    quality collapses to one number: it makes money exactly while

        EFQ < mean(gross) / mean(cost at EFQ = 1)

    This is the quantity to report. A Sharpe ratio quoted at one cost
    assumption invites the reader to argue about the assumption; a break-even
    EFQ hands them the assumption as a threshold and lets them compare it to
    what their own execution achieves. It is also where the weak-form
    efficiency reading becomes quantitative -- a break-even close to 1 says
    the edge is almost exactly the cost of taking it.

    Returns ``inf`` when nothing is ever charged and ``nan`` when the gross
    return is not positive, since no execution quality rescues that.
    """
    gross = np.asarray(gross, dtype=np.float64)
    cost = np.asarray(cost_at_unit_efq, dtype=np.float64)
    if gross.shape != cost.shape:
        raise ValueError("gross and cost must have the same shape")
    mean_gross, mean_cost = float(np.mean(gross)), float(np.mean(cost))
    if mean_gross <= 0:
        return float("nan")
    return float("inf") if mean_cost <= 0 else mean_gross / mean_cost
