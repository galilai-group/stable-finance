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


def require_aligned(left: ForwardReturns | PortfolioWeights | Orders | Fills,
                    right: ForwardReturns | PortfolioWeights | Orders | Fills) -> None:
    """Raise when two stage outputs do not describe the identical grid."""
    for axis in ("decisions", "assets", "horizons"):
        if not np.array_equal(getattr(left, axis), getattr(right, axis)):
            raise ValueError(f"{axis} are not aligned")


def require_same_observations(embeddings: Embeddings,
                              targets: ForwardReturns) -> None:
    """Raise unless embeddings and targets identify the same training rows."""
    for axis in ("decisions", "assets"):
        if not np.array_equal(getattr(embeddings, axis), getattr(targets, axis)):
            raise ValueError(f"{axis} are not aligned")


@dataclass(frozen=True)
class Orders:
    """Signed trade requests on the :class:`PortfolioWeights` grid.

    ``values`` is the requested change in weight, as a fraction of portfolio
    equity: positive buys, negative sells, zero and NaN mean no order. Convert
    to shares outside this contract by multiplying by equity and dividing by
    the reference price.
    """

    values: NDArray[np.floating]
    decisions: NDArray
    assets: NDArray
    horizons: NDArray[np.integer]

    def __post_init__(self) -> None:
        contract = ForwardReturns(self.values, self.decisions, self.assets, self.horizons)
        object.__setattr__(self, "values", contract.values)
        object.__setattr__(self, "decisions", contract.decisions)
        object.__setattr__(self, "assets", contract.assets)
        object.__setattr__(self, "horizons", contract.horizons)


@dataclass(frozen=True)
class Fills:
    """Executions produced from :class:`Orders`.

    ``quantity`` is the filled weight change (zero where nothing traded),
    ``price`` the execution price (NaN where nothing traded), and ``mid_price``
    the reference midpoint used to measure execution cost. All three share the
    ``(n_decisions, n_assets, n_horizons)`` grid.
    """

    quantity: NDArray[np.floating]
    price: NDArray[np.floating]
    mid_price: NDArray[np.floating]
    decisions: NDArray
    assets: NDArray
    horizons: NDArray[np.integer]

    def __post_init__(self) -> None:
        contract = ForwardReturns(self.quantity, self.decisions, self.assets, self.horizons)
        price = np.asarray(self.price, dtype=np.float64)
        mid_price = np.asarray(self.mid_price, dtype=np.float64)
        for name, array in (("price", price), ("mid_price", mid_price)):
            if array.shape != contract.values.shape:
                raise ValueError(
                    f"{name} must have shape {contract.values.shape}, got {array.shape}"
                )
        if not np.isfinite(contract.values).all():
            raise ValueError("quantity must be finite; use 0 for unfilled orders")
        traded = contract.values != 0
        if not (np.isfinite(price[traded]) & (price[traded] > 0)).all():
            raise ValueError("every non-zero fill needs a positive price")
        if not (np.isfinite(mid_price[traded]) & (mid_price[traded] > 0)).all():
            raise ValueError("every non-zero fill needs a positive mid_price")
        object.__setattr__(self, "quantity", contract.values)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "mid_price", mid_price)
        object.__setattr__(self, "decisions", contract.decisions)
        object.__setattr__(self, "assets", contract.assets)
        object.__setattr__(self, "horizons", contract.horizons)
