"""Dynamic trading rules: trade partway, or not at all, rather than to the target.

Every other book in this package is rebuilt at each decision and traded to.
These rules move toward a target path instead, from the book already held:

* :func:`partial_adjustment`: move a fixed fraction ``rate`` of the gap to the
  target each decision, the form of Garleanu and Pedersen's (2013, "Dynamic
  trading with predictable returns and transaction costs", JF
  68(6):2309-2340) trading rule.
* :func:`no_trade_band`: hold a name until its target leaves a band around
  the current holding, then trade only to the band's edge, the form of the
  no-trade region of Constantinides (1986, "Capital market equilibrium with
  transaction costs", JPE 94(4):842-862).

They are PARAMETRIC POLICIES, the same contract as
:class:`~stable_finance.evaluation.trades.rl.parametric.ParametricPolicy`:
``rate`` and ``band`` are free parameters to be chosen (on a training period,
say), not the papers' optimal values. In Garleanu and Pedersen both the rate
and the aim portfolio's weights
(:func:`~stable_finance.evaluation.portfolio.construction.aim_portfolio`)
follow from one quadratic cost coefficient
together with risk aversion and the signals' dynamics; tuned separately they
are two hyperparameters of a policy. Any cost a rule takes is the policy's
BELIEF about cost: it shapes the trades and is never charged. What a book
pays is decided by the harness's cost models on its fills, the same as for
any other weights (enter :func:`~stable_finance.run_backtest` at the weights
level).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation.portfolio.contract import PortfolioWeights

__all__ = ["no_trade_band", "partial_adjustment"]


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


def no_trade_band(target: PortfolioWeights, band: float, *,
                  cost_belief: ArrayLike | None = None,
                  initial: ArrayLike | None = None) -> PortfolioWeights:
    """Hold each name until its target leaves a band, then trade to the band's edge.

    The band's half-width is ``band`` in weight, or ``band * cost_belief`` per
    name and decision when ``cost_belief`` (broadcasting to ``(n_decisions,
    n_assets)``) is given: the policy's belief about the cost per unit of
    weight traded -- the quoted half-spread, say -- so names it believes dear
    to trade get a wider band. The belief only shapes the trades; nothing
    charges it. Trading only to the edge, not to the target, is the form of
    Constantinides' (1986) no-trade region under proportional costs; here
    ``band`` is a free parameter, not his optimal width. ``band=0`` trades to
    the target; a band wider than any move holds ``initial`` (default flat). A
    NaN target, or a NaN or infinite belief, holds the name.
    """
    if not np.isfinite(band) or band < 0:
        raise ValueError("band must be finite and non-negative")
    values = np.asarray(target.values, dtype=np.float64)
    n_decisions, n_assets, n_horizons = values.shape
    if cost_belief is None:
        width = np.full((n_decisions, n_assets), float(band))
    else:
        width = band * np.broadcast_to(np.asarray(cost_belief, dtype=np.float64),
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
