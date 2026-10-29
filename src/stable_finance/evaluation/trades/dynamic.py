"""Dynamic trading rules: trade partway, or not at all, rather than to the target.

Every other book in this package is rebuilt at each decision and traded to.
With proportional costs that is not optimal, and the literature gives three
standard answers, each a function of a target path and the book already held:

* :func:`aim_portfolio` (Garleanu and Pedersen 2013, "Dynamic trading with
  predictable returns and transaction costs", JF 68(6):2309-2340). When
  signals predict returns over different horizons and decay at different
  rates, the optimal policy trades partway toward an AIM portfolio that
  weights each signal by its persistence: a slowly decaying signal will still
  be right after the costly trade has been made, a fast one may not be.
* :func:`partial_adjustment`, the trading rule of the same paper: move a fixed
  fraction ``rate`` of the gap to the target each decision, the fraction set
  by how costly trading is relative to risk.
* :func:`no_trade_band` (Constantinides 1986, "Capital market equilibrium with
  transaction costs", JPE 94(4):842-862): with proportional costs there is a
  region around the current holding inside which no trade pays; a name is
  traded only when its target leaves the band, and then only to the band's
  edge.

All three are deterministic, take and return
:class:`~stable_finance.evaluation.portfolio.contract.PortfolioWeights`, and
are scored by the same harness as any other weights (enter
:func:`~stable_finance.run_backtest` at the weights level).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = ["aim_portfolio", "no_trade_band", "partial_adjustment"]


def aim_portfolio(
    forecast: ForwardReturns,
    covariance: ArrayLike,
    *,
    risk_aversion: float | ArrayLike,
    half_lives: ArrayLike | None = None,
    cost_aversion: float = 0.0,
    spacing: float | None = None,
) -> PortfolioWeights:
    """The Garleanu-Pedersen aim portfolio from forecasts at several horizons.

    ``forecast`` holds expected returns over each of its horizons (seconds);
    ``covariance`` is the ``(n_assets, n_assets)`` risk of one decision's
    return. Each horizon's forecast is first put per decision, ``mu_h * spacing
    / h``, with ``spacing`` the seconds between decisions (default: the
    shortest horizon). In Garleanu and Pedersen's model a signal ``k`` that
    mean-reverts at rate ``phi_k`` enters the aim portfolio as

        aim_t = (risk_aversion * Sigma)^-1 sum_k mu_k,t / (1 + phi_k * a / gamma),

    where ``a / gamma`` is trading cost relative to risk aversion
    (``cost_aversion`` here) and ``phi_k = ln 2 / half_life_k`` with the
    half-life in decisions. ``half_lives`` defaults to each horizon in
    decisions, ``h / spacing``: a forecast of the next hour has done its work
    in an hour. With ``cost_aversion=0`` every signal counts fully and the aim
    is the frictionless mean-variance portfolio of their sum; as trading gets
    costlier the fast signals are shrunk most. With one horizon equal to the
    spacing and no cost aversion it is exactly
    :func:`~stable_finance.evaluation.portfolio.construction.mean_variance_target`.

    A horizon whose forecast is NaN for a name contributes nothing for it; a
    name with no forecast at any horizon has a NaN aim. ``risk_aversion`` is a
    positive scalar or one value per decision. Returns weights on the
    shortest horizon's axis.
    """
    mu = np.asarray(forecast.values, dtype=np.float64)
    horizons = np.asarray(forecast.horizons, dtype=np.float64)
    if np.any(horizons <= 0):
        raise ValueError("aim_portfolio needs horizons of fixed, positive length")
    step = float(horizons.min() if spacing is None else spacing)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("spacing must be a positive number of seconds")
    lives = horizons / step if half_lives is None else np.asarray(half_lives, dtype=np.float64)
    if lives.shape != horizons.shape or not np.all(np.isfinite(lives) & (lives > 0)):
        raise ValueError("half_lives must be one positive value per horizon")
    if not np.isfinite(cost_aversion) or cost_aversion < 0:
        raise ValueError("cost_aversion must be finite and non-negative")
    sigma = np.asarray(covariance, dtype=np.float64)
    n_decisions, n_assets, _ = mu.shape
    if sigma.shape != (n_assets, n_assets):
        raise ValueError(f"covariance must have shape {(n_assets, n_assets)}, got {sigma.shape}")
    aversion = np.broadcast_to(np.asarray(risk_aversion, dtype=np.float64), (n_decisions,))
    if not np.all(np.isfinite(aversion) & (aversion > 0)):
        raise ValueError("risk_aversion must be finite and positive")

    persistence = 1.0 / (1.0 + np.log(2.0) / lives * cost_aversion)
    per_decision = mu * (step / horizons)
    combined = np.nansum(per_decision * persistence, axis=2)
    known = np.isfinite(mu).any(axis=2)
    weights = np.linalg.solve(sigma, np.where(known, combined, 0.0).T).T / aversion[:, None]
    weights = np.where(known, weights, np.nan)
    shortest = int(np.argmin(horizons))
    return PortfolioWeights(weights[:, :, None], forecast.decisions, forecast.assets,
                            [forecast.horizons[shortest]])


def _start(initial: ArrayLike | None, shape: tuple[int, int]) -> NDArray[np.float64]:
    if initial is None:
        return np.zeros(shape)
    start = np.broadcast_to(np.asarray(initial, dtype=np.float64), shape).copy()
    if not np.all(np.isfinite(start)):
        raise ValueError("initial weights must be finite")
    return start


def partial_adjustment(target: PortfolioWeights, rate: float, *,
                       initial: ArrayLike | None = None) -> PortfolioWeights:
    """Trade a fixed fraction ``rate`` of the gap to the target each decision.

    ``w_t = w_{t-1} + rate * (target_t - w_{t-1})``, from ``initial`` (default
    flat). ``rate=1`` trades to the target; a smaller rate halves the gap
    every ``ln 2 / -ln(1 - rate)`` decisions, which is the policy's half-life.
    A name whose target is NaN at a decision holds its weight.
    """
    if not np.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    values = np.asarray(target.values, dtype=np.float64)
    held = np.empty_like(values)
    current = _start(initial, values.shape[1:])
    for t in range(values.shape[0]):
        goal = values[t]
        # (1 - rate) w + rate g rather than w + rate (g - w): at rate 1 it is the
        # target exactly, not the target up to a rounding of the gap.
        current = np.where(np.isfinite(goal), (1.0 - rate) * current + rate * goal, current)
        held[t] = current
    return PortfolioWeights(held, target.decisions, target.assets, target.horizons)


def no_trade_band(target: PortfolioWeights, band: float, *, cost: ArrayLike | None = None,
                  initial: ArrayLike | None = None) -> PortfolioWeights:
    """Hold each name until its target leaves a band, then trade to the band's edge.

    The band's half-width is ``band`` in weight, or ``band * cost`` per name and
    decision when ``cost`` (broadcasting to ``(n_decisions, n_assets)``, the cost
    per unit of weight traded, e.g. the half-spread) is given, so names that
    are dear to trade get a wider band. Trading only to the edge, not to the
    target, is the optimal policy of Constantinides (1986) under proportional
    costs: the last unit traded is the one just worth its cost. ``band=0``
    trades to the target; a band wider than any move holds ``initial``
    (default flat). A NaN target, or a NaN or infinite cost, holds the name.
    """
    if not np.isfinite(band) or band < 0:
        raise ValueError("band must be finite and non-negative")
    values = np.asarray(target.values, dtype=np.float64)
    n_decisions, n_assets, n_horizons = values.shape
    if cost is None:
        width = np.full((n_decisions, n_assets), float(band))
    else:
        width = band * np.broadcast_to(np.asarray(cost, dtype=np.float64),
                                       (n_decisions, n_assets))
    held = np.empty_like(values)
    current = _start(initial, (n_assets, n_horizons))
    for t in range(n_decisions):
        goal = values[t]
        half = width[t][:, None]
        tradable = np.isfinite(goal) & np.isfinite(half)
        upper, lower = goal + half, goal - half
        moved = np.clip(current, lower, upper)
        current = np.where(tradable, moved, current)
        held[t] = current
    return PortfolioWeights(held, target.decisions, target.assets, target.horizons)
