"""Stage 1, what the model brings: representations on a decision and asset grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from stable_finance.evaluation._grid import _one_dimensional

if TYPE_CHECKING:
    from stable_finance.evaluation.returns.contract import ForwardReturns


@dataclass(frozen=True)
class Embeddings:
    """Model representations on a decision-time and asset grid.

    ``values`` has shape ``(n_decisions, n_assets, n_features)``. The feature
    axis belongs to the producing model, while decisions and assets form the
    public alignment contract.
    """

    values: NDArray[np.floating]
    decisions: NDArray
    assets: NDArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        decisions = _one_dimensional("decisions", self.decisions)
        assets = _one_dimensional("assets", self.assets)
        expected_prefix = (len(decisions), len(assets))
        if values.ndim != 3 or values.shape[:2] != expected_prefix:
            raise ValueError(
                "values must have shape "
                f"({expected_prefix[0]}, {expected_prefix[1]}, n_features), "
                f"got {values.shape}"
            )
        if values.shape[2] == 0:
            raise ValueError("embeddings must contain at least one feature")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "assets", assets)


def require_same_observations(embeddings: Embeddings,
                              targets: ForwardReturns) -> None:
    """Raise unless embeddings and targets identify the same training rows."""
    for axis in ("decisions", "assets"):
        if not np.array_equal(getattr(embeddings, axis), getattr(targets, axis)):
            raise ValueError(f"{axis} are not aligned")
