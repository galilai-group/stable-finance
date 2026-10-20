"""Stage 3: portfolio weights, on the forward-return grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stable_finance.evaluation.returns.contract import ForwardReturns


@dataclass(frozen=True)
class PortfolioWeights:
    """Portfolio weights on the same grid as :class:`ForwardReturns`.

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
