"""Run configurations end to end, from whichever level you start at.

You may bring the pipeline's input at ANY level, and the harness runs only
the stages below it:

=====================  =====================================================
you bring              the harness runs
=====================  =====================================================
``Embeddings``         a ridge probe fit on ``train``, then everything below
``ForwardReturns``     universe, selection, risk and sizing, then trading
``PortfolioWeights``   the optimal portfolio: trading toward it under the
                       execution model, which decides the deltas under cost
``Orders``             your deltas, filled under the execution model
=====================  =====================================================

Every level is scored the same way, so a Sharpe from embeddings and one from
orders are the same statistic. The IC exists only where there is a forecast.
Every level runs against every level of market data that its configuration
allows: ``cost_model='mid'`` needs none, ``'cross'`` a spread (quoted, or
estimated from bars), and ``'book'`` depth.

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

from stable_finance.evaluation.config import (
    DEFAULT_BACKTEST,
    OPTIONAL_SETTINGS,
    BacktestConfig,
    check_config,
)
from stable_finance.evaluation.embeddings.contract import Embeddings
from stable_finance.evaluation.execution import (
    Orders,
    _efq_array,
    execute,
    fill_path,
    impact_cost,
    impact_stock_cost,
    orders_from_weights,
    trading_cost,
)
from stable_finance.dataset.market import MarketData, require_market_aligned
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.construction import (
    mean_variance_target, select_book, weight_book,
)
from stable_finance.evaluation.portfolio.risk import apply_screen, estimate_risk
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.returns.probes import RidgeProbe
from stable_finance.evaluation.scoring.ic import grouped_rank_ic
from stable_finance.evaluation.scoring.pnl import positions_from_fills
from stable_finance.evaluation.scoring.sharpe import sharpe_ratio
from stable_finance.evaluation.trades.rebalance import rebalance

__all__ = ["LEVELS", "BacktestReport", "run_backtest", "sweep_configs"]

#: Axes that determine the weights. Two configurations agreeing on all of them
#: hold the identical book, so the expensive part is computed once and the
#: remaining axes -- which act only on an existing book -- are evaluated on top.
_BOOK_AXES = ("universe", "selection", "weighting", "risk_model", "cost_model",
              "efq", "gross_exposure", "shrinkage", "max_iter", "equity", "impact")


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
        """A flat, JSON-serializable row: config axes beside the scores.

        ``equity`` is left out when unset, as in the config's label, so rows
        written before it existed keep the same columns.
        """
        row = {k: v for k, v in asdict(self).items()
               if k not in ("config", "series")}
        config = {k: v for k, v in asdict(self.config).items()
                  if not (k in OPTIONAL_SETTINGS and v is None)}
        return {**config, **row}


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


def _market(like, market, half_spread) -> MarketData:
    """The market data a run is checked against.

    ``half_spread`` is the historical argument and is wrapped as market data
    holding only that quantity. Neither given is a market with no quotes,
    which supports exactly the frictionless configurations.
    """
    if market is not None and half_spread is not None:
        raise ValueError("pass market or half_spread, not both")
    if market is None:
        if half_spread is None:
            return MarketData.empty(like.decisions, like.assets)
        spread = np.asarray(half_spread, dtype=np.float64)
        expected = like.values.shape[:2]
        if spread.shape != expected:
            raise ValueError(
                f"half_spread must have shape {expected}, got {spread.shape}")
        return MarketData({"half_spread": spread}, like.decisions, like.assets)
    require_market_aligned(market, like)
    return market


def _prepare(like, realized, market: MarketData, days, horizon_index) -> dict:
    """Everything downstream needs that no configuration can change.

    ``like`` is the input on the grid: a forecast, whose values are ``mu``, or
    weights or orders, which have no forecast and so no ``mu``.
    """
    for axis in ("decisions", "assets", "horizons"):
        if not np.array_equal(getattr(like, axis), getattr(realized, axis)):
            raise ValueError(f"{axis} of the input and the realized returns are not aligned")
    n_decisions, n_assets, _ = like.values.shape
    # None, not NaN, when the market has no spread: NaN is a name that was not
    # quoting, and reading a missing column that way would screen out the
    # whole universe instead of trading it frictionlessly.
    spread = market.get("half_spread") if market.has("half_spread") else None
    horizon = int(like.horizons[horizon_index])
    axis = np.asarray([horizon])
    outcome_values = realized.values[:, :, horizon_index]
    return {
        "mu": (like.values[:, :, horizon_index] if isinstance(like, ForwardReturns)
               else None),
        "market": market,
        "outcome_values": outcome_values,
        "outcome": ForwardReturns(outcome_values[:, :, None], realized.decisions,
                                  realized.assets, axis),
        "spread": spread, "axis": axis, "horizon": horizon,
        "decisions": like.decisions, "assets": like.assets,
        "day": _day_index(days, n_decisions),
        # what the caller called each decision's day (else the decision itself), for the P&L series
        "day_labels": np.asarray(days).ravel() if days is not None else np.asarray(like.decisions),
        "days_given": days is not None,
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
    cost = _cost(config, spread, mu.shape, context=context, target=target.values[:, :, 0])
    return rebalance(
        target, sigma[None], risk_aversion=risk_aversion,
        cost=np.where(screened, np.nan_to_num(cost, nan=0.0), np.inf),
        max_iter=config.max_iter,
    ).values[:, :, 0]


def _cost(config: BacktestConfig, spread, shape, *, context=None,
          target=None) -> NDArray[np.float64]:
    """Per-unit trading cost; zero for ``mid`` whether or not there are quotes.

    The impact model's cost depends on the order, so the allocator plans
    against the cost of the move the cost-free target would make: the one
    linearization the solver can take without knowing its own answer.
    """
    if config.cost_model in ("impact", "impact_stock"):
        market = context["market"]
        adv = market.get("adv")
        moves = np.abs(target - np.vstack([np.zeros((1, shape[1])), target[:-1]]))
        size = 100.0 * np.nan_to_num(moves) * config.equity / np.where(adv > 0, adv, np.nan)
        if config.cost_model == "impact_stock":
            traits = [market.get(n) for n in ("market_cap", "idio_vol", "market_vol", "time_trend")]
            known = np.isfinite(size) & np.all([np.isfinite(x) for x in traits], axis=0)
            planned = impact_stock_cost(np.nan_to_num(size), *[np.nan_to_num(x) for x in traits],
                                        coefficients=config.impact)
            return np.where(known, planned, np.inf)
        return np.where(np.isfinite(size), impact_cost(np.nan_to_num(size), coefficients=config.impact), np.inf)
    if spread is None:
        # check_config has already refused every other model without a spread.
        return np.zeros(shape, dtype=np.float64)
    efq = config.efq if context is None else context.get("efq", config.efq)
    return trading_cost(config.cost_model, spread, multiple=efq)


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
    day = context["day"]
    if config.rebalance == "daily":
        values = _hold_within_day(values, day)
    # Every model is scored from its fills, so a cross fill is billed the
    # touch the order met (the arrival quotes, when the market carries them)
    # while the allocator planned against the quoted spread at the decision.
    return _score_fills(context, _fills_toward(context, values, config), config,
                        with_series=with_series)


def _fills_toward(context: dict, values: NDArray[np.float64], config: BacktestConfig):
    """Trade toward a target path from the positions actually held."""
    target = PortfolioWeights(values[:, :, None], context["decisions"],
                              context["assets"], context["axis"])
    if config.cost_model in ("impact", "impact_stock"):
        # Every order completes, so there is nothing to chase: the orders are
        # the target's moves, each worked over its schedule.
        return execute(orders_from_weights(target), context["market"], model=config.cost_model,
                       equity=config.equity, schedule=_schedule(context, config),
                       impact=config.impact)
    # Otherwise chase the target from what is really held: a name that is
    # not priced at a decision is held, not moved for free.
    return fill_path(target, context["market"], model=config.cost_model,
                     efq=context.get("efq", config.efq),
                     equity=config.equity, drift=_drift(context, config))


def _drift(context: dict, config: BacktestConfig):
    """The decision-to-decision returns positions move with, under drift holding; else None."""
    return context["outcome"].values if config.holding == "drift" else None


def _schedule(context: dict, config: BacktestConfig) -> int:
    """How many decisions the impact model works an order over.

    One trading day is however many decisions the data has in a day, read
    from the ``days`` labels; a run without them cannot say what a day is.
    """
    days = 1.0 if config.execution_days is None else config.execution_days
    if days == 0:
        return 0
    if not context["days_given"]:
        raise ValueError(f"cost_model={config.cost_model!r} works orders over execution_days, so it "
                         "needs days labelling each decision's trading day")
    per_day = float(np.median(np.bincount(context["day"])))
    return int(round(days * per_day))


def _score_fills(context: dict, fills, config: BacktestConfig, *,
                 with_series: bool = False) -> BacktestReport:
    """Mark the positions fills produced, and charge each fill's own cost."""
    positions = positions_from_fills(fills, drift=_drift(context, config)).values[:, :, 0]
    outcome_values = context["outcome_values"]
    valid = np.isfinite(positions) & np.isfinite(outcome_values)
    gross = np.where(valid, positions * outcome_values, 0.0).sum(axis=1)
    quantity = fills.quantity[:, :, 0]
    traded = quantity != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        slippage = np.where(traded, (fills.price[:, :, 0] - fills.mid_price[:, :, 0])
                            / fills.mid_price[:, :, 0], 0.0)
    costs = (quantity * slippage).sum(axis=1)
    return _report(context, config, gross, costs, np.abs(quantity), positions,
                   with_series=with_series)


def _report(context: dict, config: BacktestConfig, gross, costs, turnover_per_asset,
            values, *, with_series: bool) -> BacktestReport:
    """Annualize a profit path and summarize it, with the IC where there is a forecast."""
    mu, spread, day = context["mu"], context["spread"], context["day"]
    net = gross - costs

    if config.annualization == "daily":
        order = np.argsort(day, kind="stable")
        edges = np.flatnonzero(np.r_[True, day[order][1:] != day[order][:-1], True])
        fold = lambda series: np.array(  # noqa: E731
            [series[order][a:b].sum() for a, b in zip(edges[:-1], edges[1:])]
        )
        gross, net, costs_series = fold(gross), fold(net), fold(costs)
        # One label per folded period, in the same order as the folded series.
        period_labels = context["day_labels"][order][edges[:-1]]
        periods_per_year = 252.0
    else:
        costs_series = costs
        period_labels = np.asarray(context["decisions"])
        periods_per_year = (context["n_decisions"]
                            / max(len(np.unique(day)), 1)) * 252.0

    # IC IS MEASURED ON THE SCREENED UNIVERSE, not on the book: the portfolio
    # can only express a view on names it is allowed to trade, but restricting
    # the IC to the tails it actually held would score the forecast on the
    # subset it was most confident about and call that its ranking ability.
    if mu is None:
        ic_mean = ic_se = float("nan")
    else:
        screened = apply_screen(config.universe, spread, shape=mu.shape)
        ic = grouped_rank_ic(np.where(screened, mu, np.nan), context["outcome_values"],
                             min_assets=2)
        ic_mean, ic_se = float(ic.mean), float(ic.standard_error)
    return BacktestReport(
        config=config, horizon=context["horizon"],
        information_coefficient=ic_mean,
        information_coefficient_se=ic_se,
        sharpe_mid=sharpe_ratio(gross, periods_per_year=periods_per_year),
        sharpe_net=sharpe_ratio(net, periods_per_year=periods_per_year),
        mean_gross_return_bps=float(np.nanmean(gross) * 1e4),
        mean_cost_bps=float(np.nanmean(costs_series) * 1e4),
        mean_turnover=float(turnover_per_asset.sum(axis=1).mean()),
        mean_gross_exposure=float(np.abs(values).sum(axis=1).mean()),
        periods_per_year=float(periods_per_year),
        n_periods=int(len(net)),
        # A position is a holding above roundoff: a path of fills that returns
        # to flat leaves a weight of a few ulps, not a name held.
        mean_names_held=float((np.abs(values) > 1e-12).sum(axis=1).mean()),
        series=PnLSeries(
            periods=period_labels.astype(str),
            gross=gross.copy(), net=net.copy(), cost=costs_series.copy(),
        ) if with_series else None,
    )


# ── public API ──────────────────────────────────────────────────────────────

#: What the harness accepts, from the top of the pipeline to the bottom.
LEVELS = (Embeddings, ForwardReturns, PortfolioWeights, Orders)


def _forecast(source, train, probe) -> ForwardReturns:
    """Turn embeddings into a forecast with a probe fit on ``train``."""
    if train is None:
        raise ValueError("embeddings need train=(embeddings, returns) to fit the probe on")
    train_embeddings, train_returns = train
    return (probe or RidgeProbe()).fit(train_embeddings, train_returns).predict(source)


def _check_axes(config: BacktestConfig, market: MarketData, source) -> None:
    """Check the axes a level uses; below the forecast only trading remains."""
    if isinstance(source, ForwardReturns):
        check_config(config, market)
    else:
        check_config(config.replace(universe="all"), market)


def run_backtest(
    forecast,
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
    train: tuple[Embeddings, ForwardReturns] | None = None,
    probe: RidgeProbe | None = None,
    efq: ArrayLike | None = None,
    unfilled: str = "retry",
) -> BacktestReport:
    """Score one input, at any level, under one harness configuration.

    ``forecast`` is the input at any level -- embeddings, a forecast, target
    weights, or orders (see the module docstring) -- and the harness runs the
    stages below it. Embeddings need
    ``train``, the embeddings and realized returns to fit ``probe`` on (a
    default :class:`RidgeProbe` if not given), and also serve the
    ``embedding`` risk model. Below the forecast only the trading axes apply:
    ``cost_model``, ``efq``, ``equity``, ``rebalance`` (weights only) and
    ``annualization``.

    A forecast must already be in the units the realized returns are in; a
    rank-unit forecast traded against a return-unit spread would make the cost
    term meaningless. ``history`` supplies the returns the risk model is
    estimated from and should cover a period strictly before ``realized``; it
    defaults to ``realized`` itself, which is in-sample and appropriate only
    for tests. ``days`` labels each decision's trading day and is required by
    the ``daily`` rebalance and annualization options.

    ``market`` is the observable market on the input's grid; ``half_spread``
    is shorthand for a market holding only that. Each configuration axis
    declares what it needs (:attr:`BacktestConfig.requires`), and a
    configuration the market cannot support is refused with
    :class:`~stable_finance.dataset.schema.MissingRolesError` before anything
    runs. With neither given only frictionless configurations run, such as
    ``config=DEFAULT_BACKTEST.replace(cost_model="mid")``.

    ``efq`` is a per-name (or per-decision) execution quality, an array that
    broadcasts to ``(decision, asset)``, used in place of ``config.efq`` for
    both the allocator's charge and the fills. The config keeps a scalar so a
    configuration stays one hashable point of the sweep; this is for a
    caller who has measured execution quality by name.

    ``unfilled`` says what becomes of an ORDER (input at the orders level)
    that cannot fill where it is sent, e.g. a name with no quote then.
    ``"retry"`` (default) keeps it working: the orders are read as the
    position path they imply, and that path is chased from what is really
    held (:func:`fill_path`), so a missed exit goes out again at the next
    decision the name quotes instead of leaving the position open for good.
    ``"cancel"`` drops it, as the orders were literally sent. When every
    order fills the two are identical. The impact models complete every
    order, so it does not apply to them.

    ``config.holding="drift"`` carries positions with their own returns
    between decisions (see :data:`~stable_finance.evaluation.config.HOLDING_RULES`);
    ``realized`` must then hold each name's return from each decision to the
    next. It is supported for the ``mid`` and ``cross`` cost models.
    """
    source = forecast
    if isinstance(source, Embeddings):
        embeddings = source if embeddings is None else embeddings
        source = _forecast(source, train, probe)
    if not isinstance(source, LEVELS):
        raise TypeError(f"run_backtest takes one of {[c.__name__ for c in LEVELS]}, "
                        f"not {type(source).__name__}")
    market = _market(source, market, half_spread)
    if unfilled not in ("retry", "cancel"):
        raise ValueError(f"unfilled must be 'retry' or 'cancel', not {unfilled!r}")
    if config.holding == "drift" and config.cost_model not in ("mid", "cross"):
        raise ValueError("holding='drift' is supported for the mid and cross cost models")
    _check_axes(config, market, source)
    context = _prepare(source, realized, market, days, horizon_index)
    if efq is not None:
        context["efq"] = np.broadcast_to(_efq_array(efq), (context["n_decisions"], context["n_assets"]))
    if isinstance(source, ForwardReturns):
        sigma = _risk(context, config, history, realized, embeddings, horizon_index)
        return _score(context, _weights(context, config, sigma), config,
                      with_series=with_series)
    if isinstance(source, PortfolioWeights):
        return _score(context, source.values[:, :, horizon_index], config,
                      with_series=with_series)
    one = slice(horizon_index, horizon_index + 1)
    orders = Orders(source.values[:, :, one], source.decisions, source.assets,
                    source.horizons[one],
                    limit=None if source.limit is None else source.limit[:, :, one])
    if unfilled == "retry" and config.cost_model not in ("impact", "impact_stock"):
        # The path the orders imply, chased from what is really held.
        path = PortfolioWeights(np.cumsum(np.nan_to_num(orders.values, nan=0.0), axis=0),
                                orders.decisions, orders.assets, orders.horizons)
        fills = fill_path(path, market, model=config.cost_model, efq=context.get("efq", config.efq),
                          equity=config.equity, limit=orders.limit, drift=_drift(context, config))
    else:
        fills = execute(orders, market, model=config.cost_model, efq=context.get("efq", config.efq),
                        equity=config.equity, impact=config.impact,
                        schedule=(_schedule(context, config)
                                  if config.cost_model in ("impact", "impact_stock") else 0))
    return _score_fills(context, fills, config, with_series=with_series)


def sweep_configs(
    forecast,
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
    train: tuple[Embeddings, ForwardReturns] | None = None,
    probe: RidgeProbe | None = None,
) -> tuple[BacktestReport, ...]:
    """Score one input under many configurations, sharing the work.

    Identical to calling :func:`run_backtest` in a loop, and much faster,
    because the space is enormously redundant: a covariance depends only on the
    risk model, and a book depends only on the axes that produce weights.
    ``rebalance`` and ``annualization`` act on a book that already exists, so
    every configuration sharing the other axes reuses one solve -- which for
    the cost-aware allocator is essentially the entire cost of the run.
    Embeddings are turned into a forecast once. Weights and orders have no
    book to share, so they are run one configuration at a time.

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
    source = forecast
    if isinstance(source, Embeddings):
        embeddings = source if embeddings is None else embeddings
        source = _forecast(source, train, probe)
    market = _market(source, market, half_spread)
    configs = tuple(configs)
    if on_error == "raise":
        for config in configs:
            _check_axes(config, market, source)
    else:
        configs = tuple(c for c in configs if c.supported_by(market))
    if not isinstance(source, ForwardReturns):
        reports = []
        for config in configs:
            try:
                reports.append(run_backtest(source, realized, market=market, days=days,
                                            config=config, horizon_index=horizon_index,
                                            with_series=with_series))
            except Exception:
                if on_error == "raise":
                    raise
        return tuple(reports)
    context = _prepare(source, realized, market, days, horizon_index)
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
