"""Run configurations end to end: forecast in, IC and Sharpe out.

This is the wiring, and it is deliberately thin. Every decision it makes is
named in the :class:`~stable_finance.evaluation.config.BacktestConfig` it was handed, and
every stage it calls is a public function that can be used on its own. The
point is that two runs differing only in their config differ only in the
harness -- so a change in Sharpe between them cannot be a change in what was
forecast, in which names were held, or in how the metric was computed.

THE IC AND THE SHARPE COME FROM THE SAME FORECAST ARRAY. That is the whole
reason this is one function rather than two: a study of how tightly IC and
Sharpe move together is worthless if the two numbers were produced by
separately fitted predictors.

Use :func:`run_backtest` for one configuration and :func:`sweep_configs` for
many. They agree exactly; the sweep is only faster, because it notices that
most of the space shares work.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation.config import DEFAULT_BACKTEST, BacktestConfig, check_config
from stable_finance.evaluation.embeddings.contract import Embeddings
from stable_finance.evaluation.execution import trading_cost
from stable_finance.evaluation.market_data import MarketData, require_market_aligned
from stable_finance.evaluation.portfolio.construction import (
    mean_variance_target, select_book, weight_book,
)
from stable_finance.evaluation.portfolio.risk import estimate_risk
from stable_finance.evaluation.portfolio.universe import apply_screen
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.scoring.ic import grouped_rank_ic
from stable_finance.evaluation.scoring.sharpe import sharpe_ratio
from stable_finance.evaluation.trades.rebalance import rebalance

__all__ = ["BacktestReport", "run_backtest", "sweep_configs"]

#: Axes that determine the weights. Two configurations agreeing on all of them
#: hold the identical book, so the expensive part is computed once and the
#: remaining axes -- which act only on an existing book -- are evaluated on top.
_BOOK_AXES = ("universe", "selection", "weighting", "risk_model", "cost_model",
              "efq", "gross_exposure", "shrinkage", "max_iter")


@dataclass(frozen=True)
class PnLSeries:
    """The per-period profit path behind a report's scalar scores.

    A Sharpe ratio is a mean over a standard deviation, and two very different
    profit paths can share one. Keeping the path lets a caller plot what the
    configuration actually did -- and lets a test assert that the scalar and
    the series are the same number, which is the only way to know a chart and
    a table were not computed from different things.

    ``periods`` are the labels the series is indexed by: one per decision, or
    one per day when the configuration annualizes daily.
    """

    periods: NDArray[np.str_]
    gross: NDArray[np.float64]
    net: NDArray[np.float64]
    cost: NDArray[np.float64]


@dataclass(frozen=True)
class BacktestReport:
    """One configuration's scores on one horizon.

    ``sharpe_mid`` marks every trade at the midpoint and ``sharpe_net``
    charges the configured cost model; reporting both is what keeps a
    frictionless number from being read as an executable one.
    ``information_coefficient`` is measured on the same forecast over the same
    screened universe, so the pair is comparable by construction.
    """

    config: BacktestConfig
    horizon: int
    information_coefficient: float
    information_coefficient_se: float
    sharpe_mid: float
    sharpe_net: float
    mean_gross_return_bps: float
    mean_cost_bps: float
    mean_turnover: float
    mean_gross_exposure: float
    periods_per_year: float
    n_periods: int
    mean_names_held: float
    #: The profit path, only when the caller asked for it. It is deliberately
    #: absent from :meth:`as_dict`: a sweep writes one row per configuration,
    #: and four thousand embedded arrays would turn a 3 MB result file into a
    #: gigabyte one.
    series: PnLSeries | None = None

    def as_dict(self) -> dict:
        """A flat, JSON-serializable row: config axes beside the scores."""
        row = {k: v for k, v in asdict(self).items()
               if k not in ("config", "series")}
        return {**asdict(self.config), **row}


# ── shared setup ────────────────────────────────────────────────────────────

def _day_index(days: ArrayLike | None, n_decisions: int) -> NDArray[np.int64]:
    """Map each decision to a day, defaulting to one day per decision."""
    if days is None:
        return np.arange(n_decisions, dtype=np.int64)
    labels = np.asarray(days).ravel()
    if len(labels) != n_decisions:
        raise ValueError(f"days must have length {n_decisions}, got {len(labels)}")
    _, index = np.unique(labels, return_inverse=True)
    return index.astype(np.int64)


def _market(forecast, market, half_spread) -> MarketData:
    """The market data a run is checked against.

    ``half_spread`` is the historical argument and is wrapped as market data
    holding only that quantity. Neither given is a market with no quotes,
    which supports exactly the frictionless configurations.
    """
    if market is not None and half_spread is not None:
        raise ValueError("pass market or half_spread, not both")
    if market is None:
        if half_spread is None:
            return MarketData.empty(forecast.decisions, forecast.assets)
        spread = np.asarray(half_spread, dtype=np.float64)
        expected = forecast.values.shape[:2]
        if spread.shape != expected:
            raise ValueError(
                f"half_spread must have shape {expected}, got {spread.shape}")
        return MarketData({"half_spread": spread}, forecast.decisions, forecast.assets)
    require_market_aligned(market, forecast)
    return market


def _prepare(forecast, realized, market: MarketData, days, horizon_index) -> dict:
    """Everything downstream needs that no configuration can change."""
    n_decisions, n_assets, _ = forecast.values.shape
    # None, not NaN, when the market has no spread: NaN is a name that was not
    # quoting, and reading a missing column that way would screen out the
    # whole universe instead of trading it frictionlessly.
    spread = market.get("half_spread") if market.has("half_spread") else None
    horizon = int(forecast.horizons[horizon_index])
    axis = np.asarray([horizon])
    outcome_values = realized.values[:, :, horizon_index]
    return {
        "mu": forecast.values[:, :, horizon_index],
        "outcome_values": outcome_values,
        "outcome": ForwardReturns(outcome_values[:, :, None], realized.decisions,
                                  realized.assets, axis),
        "spread": spread, "axis": axis, "horizon": horizon,
        "decisions": forecast.decisions, "assets": forecast.assets,
        "day": _day_index(days, n_decisions),
        "n_decisions": n_decisions, "n_assets": n_assets,
    }


def _risk(context, config, history, realized, embeddings, horizon_index):
    """The covariance for one risk model, on the horizon being scored."""
    source = history if history is not None else realized
    panel = ForwardReturns(
        source.values[:, :, horizon_index][:, :, None],
        source.decisions, source.assets, context["axis"],
    )
    return estimate_risk(config.risk_model, panel, embeddings=embeddings,
                         shrinkage=config.shrinkage, min_samples=2)[0]


# ── the book ────────────────────────────────────────────────────────────────

def _weights(context: dict, config: BacktestConfig,
             sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Target weights under one configuration's book-determining axes."""
    mu, spread = context["mu"], context["spread"]

    # THE SCREEN AND THE BOOK ARE DIFFERENT SETS. ``screened`` is what the
    # portfolio may trade at all; ``signs`` is what this configuration chose to
    # hold out of it. Both allocator families consume the second, which is what
    # keeps `selection` a real axis for mean-variance rather than a label on
    # four identical portfolios.
    screened = apply_screen(config.universe, spread, shape=mu.shape)
    signs = select_book(config.selection, mu, screened)
    cost = _cost(config, spread, mu.shape)

    if config.weighting != "mean_variance":
        return weight_book(config.weighting, signs, mu, sigma,
                           gross_exposure=config.gross_exposure)

    # A name outside the BOOK gets a zero expected return, so the optimizer
    # unwinds it as risk and cost allow. A name outside the UNIVERSE gets an
    # infinite cost, which the proximal step reads as "hold whatever you have".
    # The distinction matters: zeroing mu lets a position be exited, while an
    # infinite cost freezes it, and using the latter for both would trap the
    # portfolio in names the selection had already dropped.
    masked = np.where(signs != 0, mu, 0.0)
    forecast = ForwardReturns(masked[:, :, None], context["decisions"],
                              context["assets"], context["axis"])
    # lambda is set so the COST-FREE target averages the requested gross
    # exposure. mu is in return units and Sigma in squared ones, so their ratio
    # has no natural scale; fixing the exposure is what makes one forecast's
    # spread bill comparable with another's.
    raw = np.linalg.solve(sigma, np.nan_to_num(masked, nan=0.0).T).T
    risk_aversion = max(float(np.abs(raw).sum(axis=1).mean()) / config.gross_exposure, 1e-12)
    # The portfolio stage says what to hold; the trades stage how far to move
    # there, given what trading costs.
    target = mean_variance_target(forecast, sigma[None], risk_aversion=risk_aversion)
    return rebalance(
        target, sigma[None], risk_aversion=risk_aversion,
        cost=np.where(screened, np.nan_to_num(cost, nan=0.0), np.inf),
        max_iter=config.max_iter,
    ).values[:, :, 0]


def _cost(config: BacktestConfig, spread, shape) -> NDArray[np.float64]:
    """Per-unit trading cost; zero for ``mid`` whether or not there are quotes."""
    if spread is None:
        # check_config has already refused every other model without a spread.
        return np.zeros(shape, dtype=np.float64)
    return trading_cost(config.cost_model, spread, multiple=config.efq)


def _hold_within_day(weights, day):
    """Keep each day's first decision's weights and hold them all day.

    Applied AFTER weighting rather than by subsampling the decisions, so the
    book is still marked and still earns a return at every decision -- it just
    does not retrade. Turnover falls; exposure does not.
    """
    starts = np.r_[True, day[1:] != day[:-1]]
    anchor = np.maximum.accumulate(np.where(starts, np.arange(len(day)), 0))
    return weights[anchor]


# ── scoring ─────────────────────────────────────────────────────────────────

def _score(context: dict, values: NDArray[np.float64],
           config: BacktestConfig, *,
           with_series: bool = False) -> BacktestReport:
    """Mark a book, charge it, and annualize -- the axes that act on a book."""
    mu, spread, day = context["mu"], context["spread"], context["day"]
    outcome_values, n_assets = context["outcome_values"], context["n_assets"]
    if config.rebalance == "daily":
        values = _hold_within_day(values, day)
    cost = _cost(config, spread, mu.shape)

    valid = np.isfinite(values) & np.isfinite(outcome_values)
    gross = np.where(valid, values * outcome_values, 0.0).sum(axis=1)
    previous = np.vstack([np.zeros((1, n_assets)), values[:-1]])
    turnover_per_asset = np.abs(values - previous)
    # An unquoted name carries an INFINITE cost, and 0 * inf is a NaN with a
    # warning attached. The zero is the meaningful half: such a name is
    # screened out of the book, so its turnover is zero and it is charged
    # nothing. Zeroing the cost before the multiply says that directly instead
    # of producing a NaN and then masking it.
    finite_cost = np.isfinite(cost)
    charged = np.where(
        finite_cost & np.isfinite(turnover_per_asset),
        turnover_per_asset * np.where(finite_cost, cost, 0.0), 0.0,
    )
    costs = charged.sum(axis=1)
    net = gross - costs

    if config.annualization == "daily":
        order = np.argsort(day, kind="stable")
        edges = np.flatnonzero(np.r_[True, day[order][1:] != day[order][:-1], True])
        fold = lambda series: np.array(  # noqa: E731
            [series[order][a:b].sum() for a, b in zip(edges[:-1], edges[1:])]
        )
        gross, net, costs_series = fold(gross), fold(net), fold(costs)
        # One label per folded period, in the same order as the folded series.
        period_labels = np.asarray(day)[order][edges[:-1]]
        periods_per_year = 252.0
    else:
        costs_series = costs
        period_labels = np.asarray(day)
        periods_per_year = (context["n_decisions"]
                            / max(len(np.unique(day)), 1)) * 252.0

    # IC IS MEASURED ON THE SCREENED UNIVERSE, not on the book: the portfolio
    # can only express a view on names it is allowed to trade, but restricting
    # the IC to the tails it actually held would score the forecast on the
    # subset it was most confident about and call that its ranking ability.
    screened = apply_screen(config.universe, spread, shape=mu.shape)
    ic = grouped_rank_ic(np.where(screened, mu, np.nan), outcome_values,
                         min_assets=2)
    return BacktestReport(
        config=config, horizon=context["horizon"],
        information_coefficient=float(ic.mean),
        information_coefficient_se=float(ic.standard_error),
        sharpe_mid=sharpe_ratio(gross, periods_per_year=periods_per_year),
        sharpe_net=sharpe_ratio(net, periods_per_year=periods_per_year),
        mean_gross_return_bps=float(np.nanmean(gross) * 1e4),
        mean_cost_bps=float(np.nanmean(costs_series) * 1e4),
        mean_turnover=float(turnover_per_asset.sum(axis=1).mean()),
        mean_gross_exposure=float(np.abs(values).sum(axis=1).mean()),
        periods_per_year=float(periods_per_year),
        n_periods=int(len(net)),
        mean_names_held=float((np.abs(values) > 0).sum(axis=1).mean()),
        series=PnLSeries(
            periods=period_labels.astype(str),
            gross=gross.copy(), net=net.copy(), cost=costs_series.copy(),
        ) if with_series else None,
    )


# ── public API ──────────────────────────────────────────────────────────────

def run_backtest(
    forecast: ForwardReturns,
    realized: ForwardReturns,
    *,
    market: MarketData | None = None,
    half_spread: ArrayLike | None = None,
    history: ForwardReturns | None = None,
    embeddings: Embeddings | None = None,
    days: ArrayLike | None = None,
    config: BacktestConfig = DEFAULT_BACKTEST,
    horizon_index: int = 0,
    with_series: bool = False,
) -> BacktestReport:
    """Score one forecast under one harness configuration.

    ``forecast`` must already be in the units the realized returns are in; a
    rank-unit forecast traded against a return-unit spread would make the cost
    term meaningless. ``history`` supplies the returns the risk model is
    estimated from and should cover a period strictly before ``realized``; it
    defaults to ``realized`` itself, which is in-sample and appropriate only
    for tests. ``days`` labels each decision's trading day and is required by
    the ``daily`` rebalance and annualization options.

    ``market`` is the observable market on the forecast's grid; ``half_spread``
    is shorthand for a market holding only that. Each configuration axis
    declares what it needs (:attr:`BacktestConfig.requires`), and a
    configuration the market cannot support is refused with
    :class:`~stable_finance.dataset.schema.MissingRolesError` before anything
    runs. With neither given only frictionless configurations run, such as
    ``config=DEFAULT_BACKTEST.replace(cost_model="mid")``.
    """
    market = _market(forecast, market, half_spread)
    check_config(config, market)
    context = _prepare(forecast, realized, market, days, horizon_index)
    sigma = _risk(context, config, history, realized, embeddings, horizon_index)
    return _score(context, _weights(context, config, sigma), config,
                  with_series=with_series)


def sweep_configs(
    forecast: ForwardReturns,
    realized: ForwardReturns,
    configs,
    *,
    market: MarketData | None = None,
    half_spread: ArrayLike | None = None,
    history: ForwardReturns | None = None,
    embeddings: Embeddings | None = None,
    days: ArrayLike | None = None,
    horizon_index: int = 0,
    on_error: str = "raise",
    book_cache_size: int = 8,
    with_series: bool = False,
) -> tuple[BacktestReport, ...]:
    """Score one forecast under many configurations, sharing the work.

    Identical to calling :func:`run_backtest` in a loop, and much faster,
    because the space is enormously redundant: a covariance depends only on the
    risk model, and a book depends only on the axes that produce weights.
    ``rebalance`` and ``annualization`` act on a book that already exists, so
    every configuration sharing the other axes reuses one solve -- which for
    the cost-aware allocator is essentially the entire cost of the run.

    ``on_error='skip'`` drops configurations that raise and keeps the rest,
    for sweeps wide enough that one degenerate corner should not lose the
    others. ``book_cache_size`` bounds how many weight panels are held at once;
    the default suffices whenever the book-reusing axes vary fastest.

    Every configuration is checked against ``market`` before any is run, so a
    sweep refuses up front rather than failing partway; with ``on_error='skip'``
    the unsupported ones are dropped instead. ``enumerate_configs(market=...)``
    builds a space that needs no dropping.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    market = _market(forecast, market, half_spread)
    configs = tuple(configs)
    if on_error == "raise":
        for config in configs:
            check_config(config, market)
    else:
        configs = tuple(c for c in configs if c.supported_by(market))
    context = _prepare(forecast, realized, market, days, horizon_index)

    # The risk cache is tiny (one matrix per risk model) and kept whole. The
    # book cache is NOT: a book is a full weight panel, and the space holds
    # over a thousand distinct ones, which would be gigabytes held for nothing.
    # It is bounded instead, which costs nothing when the axes that reuse a
    # book vary fastest -- as they do in enumerate_configs -- and merely
    # recomputes on an unlucky ordering rather than exhausting memory.
    risk_cache: dict[tuple, NDArray[np.float64]] = {}
    book_cache: "OrderedDict[tuple, NDArray[np.float64]]" = OrderedDict()
    reports = []
    for config in configs:
        try:
            risk_key = (config.risk_model, config.shrinkage)
            if risk_key not in risk_cache:
                risk_cache[risk_key] = _risk(
                    context, config, history, realized, embeddings, horizon_index
                )
            book_key = tuple(getattr(config, axis) for axis in _BOOK_AXES)
            if book_key in book_cache:
                book_cache.move_to_end(book_key)
            else:
                book_cache[book_key] = _weights(
                    context, config, risk_cache[risk_key]
                )
                if len(book_cache) > book_cache_size:
                    book_cache.popitem(last=False)
            reports.append(_score(context, book_cache[book_key], config,
                                  with_series=with_series))
        except Exception:
            if on_error == "raise":
                raise
    return tuple(reports)
