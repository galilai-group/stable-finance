"""Stage 2: forward returns, realized or forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stable_finance.evaluation._grid import _one_dimensional


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
