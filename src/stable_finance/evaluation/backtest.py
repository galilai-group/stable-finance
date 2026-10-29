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
    FEE_SETTINGS,
    Orders,
    _efq_array,
    execute,
    fee_cost,
    fee_per_unit,
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
from stable_finance.evaluation.portfolio.signals import (
    alpha_from_rule, apply_signal, estimate_beta,
)
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.returns.probes import RidgeProbe
from stable_finance.evaluation.scoring.ic import grouped_rank_ic
from stable_finance.evaluation.scoring.pnl import positions_from_fills
from stable_finance.evaluation.scoring.sharpe import sharpe_ratio
from stable_finance.evaluation.scoring.tc import transfer_coefficient
from stable_finance.evaluation.trades.rebalance import rebalance

__all__ = ["LEVELS", "BacktestReport", "run_backtest", "sweep_configs"]

#: Axes that determine the weights. Two configurations agreeing on all of them
#: hold the identical book, so the expensive part is computed once and the
#: remaining axes -- which act only on an existing book -- are evaluated on top.
_BOOK_AXES = ("universe", "selection", "weighting", "risk_model", "cost_model",
              "efq", "gross_exposure", "shrinkage", "max_iter", "equity", "impact",
              "alpha", "alpha_ic", "neutralize", "mv_solver", *FEE_SETTINGS)


#: The book axes only the cost-aware allocator reads. A path-independent rule
#: sizes a book from the forecast and the risk model alone, so configurations
#: differing only in what trading costs hold the same book and share it.
_COST_AXES = ("cost_model", "efq", "equity", "impact", "max_iter", "mv_solver",
              *FEE_SETTINGS)


def _book_key(config: BacktestConfig) -> tuple:
    """The settings that determine a configuration's book, and nothing else."""
    axes = (_BOOK_AXES if config.weighting == "mean_variance"
            else tuple(a for a in _BOOK_AXES if a not in _COST_AXES))
    return tuple((axis, getattr(config, axis)) for axis in axes)


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

    ``fee`` is the part of ``cost`` that is fees, when the configuration
    charges any (else None), so ``cost - fee`` is what the cost model itself
    charged -- the part that scales with execution quality, which a
    break-even EFQ is read from.
    """

    periods: NDArray[np.str_]
    gross: NDArray[np.float64]
    net: NDArray[np.float64]
    cost: NDArray[np.float64]
    fee: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class BacktestReport:
    """One configuration's scores on one horizon.

    ``sharpe_mid`` marks every trade at the midpoint and ``sharpe_net``
    charges the configured cost model; reporting both is what keeps a
    frictionless number from being read as an executable one.
    ``information_coefficient`` is measured on the same forecast over the same
    screened universe, so the pair is comparable by construction.

    ``transfer_coefficient`` is how much of the forecast the book expresses
    (:func:`~stable_finance.evaluation.scoring.tc.transfer_coefficient`): the
    correlation, over the screened names, of the positions actually HELD --
    after the solver, the fills and ``rebalance`` -- with the forecast
    the book was built from, AFTER the signal transforms, both in the risk
    model's units. It is measured against the transformed forecast rather than
    the raw one because the transforms are what the book was asked to express;
    the IC is what scores the raw forecast. Below the forecast level there is
    no forecast to transfer, and it is NaN.
    """

    config: BacktestConfig
    horizon: int
    information_coefficient: float
    information_coefficient_se: float
    #: NaN where there is no forecast.
    transfer_coefficient: float
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
    #: The mean fee per period, in basis points of equity, when the
    #: configuration charges fees; it is part of ``mean_cost_bps``, not on top
    #: of it. None otherwise, and then absent from :meth:`as_dict`.
    mean_fee_bps: float | None = None

    def as_dict(self) -> dict:
        """A flat, JSON-serializable row: config axes beside the scores.

        ``equity`` is left out when unset, as in the config's label, so rows
        written before it existed keep the same columns; so is
        ``mean_fee_bps`` for a configuration that charges no fees.
        """
        row = {k: v for k, v in asdict(self).items()
               if k not in ("config", "series")
               and not (k == "mean_fee_bps" and v is None)}
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


def _history_panel(context, history, realized) -> ForwardReturns:
    """The risk history at the horizon being scored, checked against the input.

    The history is matched to the input BY LABEL, not by position: its assets
    must be the input's assets in the input's order, and its horizon is found
    by value. A history whose columns were merely the same length used to be
    read as though they lined up, which silently paired one name's risk with
    another's forecast.
    """
    source = history if history is not None else realized
    if not np.array_equal(source.assets, context["assets"]):
        raise ValueError("assets of the history and the input are not aligned")
    matches = np.flatnonzero(np.asarray(source.horizons) == context["horizon"])
    if len(matches) == 0:
        raise ValueError(f"the history has no horizon {context['horizon']}; "
                         f"it has {list(source.horizons)}")
    return ForwardReturns(source.values[:, :, matches[0]][:, :, None],
                          source.decisions, source.assets, context["axis"])


def _risk(context, config, history, realized, embeddings, horizon_index):
    """The covariance for one risk model, on the horizon being scored.

    The ``factor`` model adds industry factors when the market carries an
    ``industry`` code, taking each name's FIRST known code: classifications
    are static data, and the earliest code in the period is the one in force
    when the model is fitted, so a reclassification later in the period
    cannot leak into it.
    """
    panel = _history_panel(context, history, realized)
    if embeddings is not None and not np.array_equal(embeddings.assets, context["assets"]):
        raise ValueError("assets of the embeddings and the input are not aligned")
    industry = None
    if config.risk_model == "factor" and context["market"].has("industry"):
        codes = context["market"].get("industry")
        known = np.isfinite(codes)
        first = np.argmax(known, axis=0)
        industry = np.where(known.any(axis=0),
                            codes[first, np.arange(codes.shape[1])], np.nan)
    return estimate_risk(config.risk_model, panel, embeddings=embeddings,
                         shrinkage=config.shrinkage, min_samples=2,
                         industry=industry)[0]


def _beta(context, history, realized) -> NDArray[np.float64]:
    """Each name's beta at the scored horizon, estimated once per run."""
    if "beta" not in context:
        context["beta"] = estimate_beta(_history_panel(context, history, realized))[0]
    return context["beta"]


def _signal(context: dict, config: BacktestConfig, sigma, screened) -> NDArray[np.float64]:
    """The forecast the book is built from, after the configured transforms.

    Neutralization runs first, so the alpha is scaled from the part of the
    forecast the book is meant to express; both read only the screened names.
    The IC is still measured on the raw forecast: these transforms change what
    the portfolio expresses, not what the model predicted.
    """
    mu = context["mu"]
    if config.neutralize is not None:
        market = context["market"]
        mu = apply_signal(
            config.neutralize, mu, eligible=screened,
            beta=context.get("beta"),
            industry=market.get("industry") if market.has("industry") else None,
        )
    if config.alpha is not None:
        mu = alpha_from_rule(config.alpha, mu, np.sqrt(np.maximum(np.diag(sigma), 0.0)),
                             ic=config.alpha_ic, eligible=screened)
    return mu


# ── the book ────────────────────────────────────────────────────────────────

def _weights(context: dict, config: BacktestConfig,
             sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Target weights under one configuration's book-determining axes."""
    return _book(context, config, sigma)[0]


def _book(context: dict, config: BacktestConfig, sigma: NDArray[np.float64]
          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Target weights, and the transformed forecast they were built from.

    The forecast is returned beside the weights, rather than stashed in
    ``context``, because the context is shared by every configuration of a
    sweep and the transformed forecast is not: two configurations differing
    only in ``neutralize`` would read each other's. Returned together, the pair
    is cached together, so the transfer coefficient a sweep reports is the one
    :func:`run_backtest` would.
    """
    spread = context["spread"]
    mu = context["mu"]

    # THE SCREEN AND THE BOOK ARE DIFFERENT SETS. ``screened`` is what the
    # portfolio may trade at all; ``signs`` is what this configuration chose to
    # hold out of it. Both allocator families consume the second, which is what
    # keeps `selection` a real axis for mean-variance rather than a label on
    # four identical portfolios.
    screened = apply_screen(config.universe, spread, shape=mu.shape)
    mu = _signal(context, config, sigma, screened)
    signs = select_book(config.selection, mu, screened)
    if config.weighting != "mean_variance":
        return weight_book(config.weighting, signs, mu, sigma,
                           gross_exposure=config.gross_exposure), mu

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
    # spread bill comparable with another's. The average runs over the
    # decisions UP TO each one, never past it: a lambda fit on the whole window
    # let a forecast made next month resize the book held today.
    risk_aversion = causal_risk_aversion(masked, sigma, config.gross_exposure)
    # The portfolio stage says what to hold; the trades stage how far to move
    # there, given what trading costs.
    target = mean_variance_target(forecast, sigma[None], risk_aversion=risk_aversion)
    cost = _cost(config, spread, mu.shape, context=context, target=target.values[:, :, 0])
    return rebalance(
        target, sigma[None], risk_aversion=risk_aversion,
        cost=np.where(screened, np.nan_to_num(cost, nan=0.0), np.inf),
        max_iter=config.max_iter, solver=config.mv_solver,
    ).values[:, :, 0], mu


def _cost(config: BacktestConfig, spread, shape, *, context=None,
          target=None) -> NDArray[np.float64]:
    """Per-unit trading cost the allocator is charged: the cost model's, plus fees.

    Fees are added on top of whatever the model charges, at
    :func:`~stable_finance.evaluation.execution.fee_per_unit`, so the book is
    planned against the same fees :func:`_score_fills` bills. A name the model
    cannot trade stays at infinite cost, and so does one without the price a
    per-share fee needs.
    """
    cost = _model_cost(config, spread, shape, context=context, target=target)
    if not config.fees:
        return cost
    return cost + fee_per_unit(_share_price(context, config), **config.fees)


def _share_price(context: dict, config: BacktestConfig) -> NDArray[np.float64] | None:
    """The share price a per-share fee is converted at: the mid, else not needed."""
    return context["market"].get("mid") if config.fee_per_share is not None else None


def _model_cost(config: BacktestConfig, spread, shape, *, context=None,
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

def causal_risk_aversion(mu: ArrayLike, sigma: ArrayLike,
                         gross_exposure: float) -> NDArray[np.float64]:
    """One lambda per decision, from the decisions up to and including it.

    ``lambda_t = mean_{s <= t} |Sigma^-1 mu_s|_1 / gross_exposure``: the
    cost-free mean-variance book averages the requested gross exposure over
    the forecasts seen so far. Only the past and the present enter, so a run
    cut short gives the same weights on the decisions it kept.
    """
    raw = np.linalg.solve(np.asarray(sigma, dtype=np.float64),
                          np.nan_to_num(np.asarray(mu, dtype=np.float64), nan=0.0).T).T
    gross = np.abs(raw).sum(axis=1)
    expanding = np.cumsum(gross) / np.arange(1, len(gross) + 1)
    return np.maximum(expanding / gross_exposure, 1e-12)


def _decision_seconds(context: dict) -> float:
    """The spacing between consecutive decisions on one day, in seconds.

    Read from the decisions themselves: epoch seconds, as
    :func:`~stable_finance.evaluation.inputs.evaluation_inputs` takes them, or
    the ``"date|anchor"`` labels the Market-1T readers write, whose anchor is
    seconds past the session's origin. The median gap within a day is used,
    so overnight gaps and an occasional missing decision do not count.
    """
    decisions = np.asarray(context["decisions"])
    try:
        stamps = decisions.astype(np.float64)
    except (TypeError, ValueError):
        parts = np.char.partition(decisions.astype(str), "|")
        if not np.all(parts[:, 1] == "|"):
            raise ValueError("this needs the decisions' times: pass decisions as epoch "
                             "seconds or 'date|anchor' labels") from None
        stamps = parts[:, 2].astype(np.float64)
    gaps = np.diff(stamps)
    if context["days_given"]:
        day = context["day"]
        gaps = gaps[day[1:] == day[:-1]]
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        raise ValueError("need two decisions on one day to measure their spacing; "
                         "pass days labelling each decision's trading day")
    return float(np.median(gaps))


def _decisions_spanning(context: dict, *, seconds: float | None = None,
                        days: float | None = None) -> int:
    """How many decisions a span of time covers: the one time-to-decisions rule.

    A span in ``days`` is that many times the decisions the data has in a
    day, read from the ``days`` labels (the impact model's schedule). A span
    in ``seconds`` is that over the spacing of the decisions (a held book's
    horizon). Both round to the nearest whole decision.
    """
    if days is not None:
        if not context["days_given"]:
            raise ValueError("a span in days needs days labelling each decision's trading day")
        per_day = float(np.median(np.bincount(context["day"])))
        return int(round(days * per_day))
    return max(round(float(seconds) / _decision_seconds(context)), 1)


def _formations(context: dict) -> NDArray[np.bool_]:
    """The decisions where a horizon-held book is set: every k-th of each day.

    k is the forecast horizon in decisions. Blocks restart at each day, since
    a forward return does not run across the close; without ``days`` the
    decisions are one run.
    """
    if "formed" not in context:
        if context["horizon"] <= 0:
            raise ValueError("rebalance='horizon' needs a horizon of fixed length; a book "
                             "held to the close is already set once a day (rebalance='daily')")
        every = _decisions_spanning(context, seconds=context["horizon"])
        day = context["day"]
        index = np.arange(len(day))
        if context["days_given"]:
            starts = np.r_[True, day[1:] != day[:-1]]
            index = index - np.maximum.accumulate(np.where(starts, index, 0))
        context["formed"] = index % every == 0
    return context["formed"]


def _hold_from(weights, formed):
    """Carry each formation decision's weights until the next formation."""
    anchor = np.maximum.accumulate(np.where(formed, np.arange(len(formed)), 0))
    return weights[anchor]


def _score(context: dict, values: NDArray[np.float64],
           config: BacktestConfig, *,
           with_series: bool = False, alpha: NDArray[np.float64] | None = None,
           sigma: NDArray[np.float64] | None = None) -> BacktestReport:
    """Mark a book, charge it, and annualize -- the axes that act on a book.

    ``alpha`` and ``sigma`` are the transformed forecast and the covariance the
    book was built from; given both, the report carries a transfer coefficient.
    """
    day = context["day"]
    if config.rebalance == "daily":
        values = _hold_within_day(values, day)
    if config.rebalance == "horizon":
        values = _hold_from(values, _formations(context))
    # Every model is scored from its fills, so a cross fill is billed the
    # touch the order met (the arrival quotes, when the market carries them)
    # while the allocator planned against the quoted spread at the decision.
    return _score_fills(context, _fills_toward(context, values, config), config,
                        with_series=with_series, alpha=alpha, sigma=sigma)


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
    return _decisions_spanning(context, days=days)


def _score_fills(context: dict, fills, config: BacktestConfig, *,
                 with_series: bool = False, alpha: NDArray[np.float64] | None = None,
                 sigma: NDArray[np.float64] | None = None) -> BacktestReport:
    """Mark the positions fills produced, and charge each fill's own cost."""
    positions = positions_from_fills(fills, drift=_drift(context, config)).values[:, :, 0]
    outcome_values = context["outcome_values"]
    valid = np.isfinite(positions) & np.isfinite(outcome_values)
    if config.rebalance == "horizon":
        # A held book earns its forward return ONCE, at the decision it was
        # set: the decisions it is merely carried through are inside that same
        # return, and booking them again would count it once per decision.
        valid &= _formations(context)[:, None]
    gross = np.where(valid, positions * outcome_values, 0.0).sum(axis=1)
    quantity = fills.quantity[:, :, 0]
    traded = quantity != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        slippage = np.where(traded, (fills.price[:, :, 0] - fills.mid_price[:, :, 0])
                            / fills.mid_price[:, :, 0], 0.0)
    costs = (quantity * slippage).sum(axis=1)
    fees = None
    if config.fees:
        # Fees are billed on the quantity each fill actually traded, beside
        # its slippage rather than inside its price (see the execution
        # module), so cost is still exactly slippage plus fees.
        fees = fee_cost(quantity, _share_price(context, config), **config.fees).sum(axis=1)
        costs = costs + fees
    return _report(context, config, gross, costs, np.abs(quantity), positions,
                   with_series=with_series, alpha=alpha, sigma=sigma, fees=fees)


def _report(context: dict, config: BacktestConfig, gross, costs, turnover_per_asset,
            values, *, with_series: bool, alpha: NDArray[np.float64] | None = None,
            sigma: NDArray[np.float64] | None = None, fees=None) -> BacktestReport:
    """Annualize a profit path and summarize it, with the IC where there is a forecast.

    ``fees`` is the per-decision part of ``costs`` that is fees, folded into
    periods the same way, or None when the configuration charges none.
    """
    mu, spread, day = context["mu"], context["spread"], context["day"]
    net = gross - costs

    if config.annualization == "daily":
        order = np.argsort(day, kind="stable")
        edges = np.flatnonzero(np.r_[True, day[order][1:] != day[order][:-1], True])
        fold = lambda series: np.array(  # noqa: E731
            [series[order][a:b].sum() for a, b in zip(edges[:-1], edges[1:])]
        )
        gross, net, costs_series = fold(gross), fold(net), fold(costs)
        fee_series = None if fees is None else fold(fees)
        # One label per folded period, in the same order as the folded series.
        period_labels = context["day_labels"][order][edges[:-1]]
        periods_per_year = 252.0
    elif config.rebalance == "horizon":
        # One period per held book: the decisions it is carried through are
        # folded into the one at which it was set, which labels the period.
        block = np.cumsum(_formations(context)) - 1
        edges = np.flatnonzero(np.r_[True, block[1:] != block[:-1], True])
        fold = lambda series: np.add.reduceat(series, edges[:-1])
        gross, net, costs_series = fold(gross), fold(net), fold(costs)
        fee_series = None if fees is None else fold(fees)
        period_labels = np.asarray(context["decisions"])[edges[:-1]]
        periods_per_year = (len(edges) - 1) / max(len(np.unique(day)), 1) * 252.0
    else:
        costs_series = costs
        fee_series = fees
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
    # THE TC READS THE SAME SCREENED NAMES AS THE IC, so the two factors of the
    # fundamental law are measured over one cross-section. The positions are
    # the HELD ones, with roundoff residue zeroed under the same threshold as
    # ``mean_names_held``: a book that returned to flat is flat, not a vector of
    # ulps whose correlation with the forecast is noise. NaN is the module
    # constant, not a fresh float, so two NaN reports compare equal as rows.
    tc = np.nan
    if mu is not None and alpha is not None and sigma is not None:
        held = np.where(np.isfinite(values) & (np.abs(values) <= 1e-12), 0.0, values)
        mean = transfer_coefficient(held, alpha, sigma, eligible=screened).mean
        tc = float(mean) if np.isfinite(mean) else np.nan
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
        transfer_coefficient=tc,
        series=PnLSeries(
            periods=period_labels.astype(str),
            gross=gross.copy(), net=net.copy(), cost=costs_series.copy(),
            fee=None if fee_series is None else fee_series.copy(),
        ) if with_series else None,
        mean_fee_bps=None if fee_series is None else float(np.nanmean(fee_series) * 1e4),
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
        check_config(config.replace(universe="all", alpha=None, alpha_ic=None,
                                    neutralize=None), market)


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
    ``cost_model``, ``efq``, ``equity``, the fees, ``rebalance`` (weights
    only) and ``annualization``.

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
        if config.neutralize in ("beta", "beta_industry"):
            _beta(context, history, realized)
        weights, alpha = _book(context, config, sigma)
        return _score(context, weights, config, with_series=with_series,
                      alpha=alpha, sigma=sigma)
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
    # Each entry is a book AND the transformed forecast it was built from, so
    # a reused book reports the transfer coefficient its first solve would.
    book_cache: "OrderedDict[tuple, tuple[NDArray[np.float64], NDArray[np.float64]]]" = (
        OrderedDict())
    reports = []
    for config in configs:
        try:
            risk_key = (config.risk_model, config.shrinkage)
            if risk_key not in risk_cache:
                risk_cache[risk_key] = _risk(
                    context, config, history, realized, embeddings, horizon_index
                )
            if config.neutralize in ("beta", "beta_industry"):
                _beta(context, history, realized)
            book_key = _book_key(config)
            if book_key in book_cache:
                book_cache.move_to_end(book_key)
            else:
                book_cache[book_key] = _book(
                    context, config, risk_cache[risk_key]
                )
                if len(book_cache) > book_cache_size:
                    book_cache.popitem(last=False)
            weights, alpha = book_cache[book_key]
            reports.append(_score(context, weights, config, with_series=with_series,
                                  alpha=alpha, sigma=risk_cache[risk_key]))
        except Exception:
            if on_error == "raise":
                raise
    return tuple(reports)
