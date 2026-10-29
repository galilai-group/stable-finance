"""Learned trading policies: where reinforcement-learning methods go.

The rules in :mod:`stable_finance.evaluation.trades.rules` and the allocator
in :mod:`stable_finance.evaluation.trades.rebalance` are hand-written
policies for the same decision a learned one makes: given the portfolio held,
the target, the forecast and what trading costs, how far to move. A policy
here takes that state and returns the same ``PortfolioWeights`` (or
``Orders``) the rules do, so the harness in
:mod:`stable_finance.evaluation.backtest` and the execution models in
:mod:`stable_finance.evaluation.execution` score it unchanged.

:class:`~stable_finance.evaluation.trades.rl.parametric.ParametricPolicy` is
the first: a parametric portfolio policy with transaction costs, fit
deterministically on net-of-cost utility.
"""

from stable_finance.evaluation.trades.rl.parametric import ParametricPolicy, to_weights

__all__ = ["ParametricPolicy", "to_weights"]
