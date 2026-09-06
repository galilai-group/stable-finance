"""Canonical array contracts shared by the evaluation stages.

The first two axes always identify a decision and an asset. A third axis, when
present, identifies a forecast horizon. Keeping those axes explicit prevents a
surprisingly common class of research bugs: comparing predictions and outcomes
whose rows happen to have the same length but refer to different observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _one_dimensional(name: str, value: ArrayLike) -> NDArray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


@dataclass(frozen=True)
class ForwardReturns:
    """Realized or predicted forward returns.

    ``values`` has shape ``(n_decisions, n_assets, n_horizons)``. Missing
    observations are represented by NaN and are ignored by metrics.
    """

    values: NDArray[np.floating]
    decisions: NDArray
    assets: NDArray
    horizons: NDArray[np.integer]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        decisions = _one_dimensional("decisions", self.decisions)
        assets = _one_dimensional("assets", self.assets)
        horizons = _one_dimensional("horizons", self.horizons)
        expected = (len(decisions), len(assets), len(horizons))
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}, got {values.shape}")
        if np.any(np.asarray(horizons, dtype=np.float64) <= 0):
            raise ValueError("horizons must contain positive durations")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "horizons", horizons)


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


def require_aligned(left: ForwardReturns | PortfolioWeights,
                    right: ForwardReturns | PortfolioWeights) -> None:
    """Raise when two stage outputs do not describe the identical grid."""
    for axis in ("decisions", "assets", "horizons"):
        if not np.array_equal(getattr(left, axis), getattr(right, axis)):
            raise ValueError(f"{axis} are not aligned")

