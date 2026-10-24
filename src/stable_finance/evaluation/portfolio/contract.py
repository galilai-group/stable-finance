"""Stage 3: portfolio weights, on the forward-return grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stable_finance.evaluation.returns.contract import ForwardReturns


@dataclass(frozen=True)
class PortfolioWeights:
    """Portfolio weights on the same grid as :class:`ForwardReturns`.

    As the portfolio stage's output this is the OPTIMAL PORTFOLIO: what to
    hold at each decision as if trading were free. The deltas from the
    portfolio held to it, with no regard to cost, are
    :func:`~stable_finance.evaluation.execution.orders_from_weights`; the
    trades stage decides how much of them to trade given cost, and hands
    back the same contract as the path actually held.

    Weights are fractions of portfolio equity. They may be long or short and
    need not have unit gross exposure; evaluation reports the result supplied
    by the caller rather than silently rescaling it.
    """

    values: NDArray[np.floating]
    decisions: NDArray
    assets: NDArray
    horizons: NDArray[np.integer]

    def __post_init__(self) -> None:
        returns_contract = ForwardReturns(
            self.values, self.decisions, self.assets, self.horizons
        )
        object.__setattr__(self, "values", returns_contract.values)
        object.__setattr__(self, "decisions", returns_contract.decisions)
        object.__setattr__(self, "assets", returns_contract.assets)
        object.__setattr__(self, "horizons", returns_contract.horizons)
