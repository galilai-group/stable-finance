"""Stage 4: orders requested and fills obtained, on the weight grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stable_finance.evaluation.returns.contract import ForwardReturns


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
