"""Learned trading policies: where reinforcement-learning methods will go.

Empty for now. The rules in :mod:`stable_finance.evaluation.trades.rules` and
the allocator in :mod:`stable_finance.evaluation.trades.rebalance` are
hand-written policies for the same decision a learned one would make: given
the portfolio held, the target, the forecast and what trading costs, how far
to move. A policy added here should take that state and return the same
``PortfolioWeights`` (or ``Orders``) the rules do, so the harness in
:mod:`stable_finance.evaluation.backtest` and the execution models in
:mod:`stable_finance.evaluation.execution` score it unchanged.
"""

__all__: list[str] = []
