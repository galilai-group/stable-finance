"""Stage 2: forward returns, realized or forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stable_finance.dataset.anchors import TO_CLOSE
from stable_finance.evaluation._grid import _one_dimensional


@dataclass(frozen=True)
class ForwardReturns:
    """Realized or predicted forward returns.

    ``values`` has shape ``(n_decisions, n_assets, n_horizons)``. Missing
    observations are represented by NaN and are ignored by metrics.

    A horizon is a positive duration, or
    :data:`~stable_finance.dataset.anchors.TO_CLOSE`: held into the closing
    auction, so its length differs by decision.
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
        h = np.asarray(horizons, dtype=np.float64)
        if np.any((h <= 0) & (h != TO_CLOSE)):
            raise ValueError("horizons must contain positive durations or TO_CLOSE")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "horizons", horizons)
