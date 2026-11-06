"""Factor exposure and concentration of the weights actually held."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["PortfolioExposure", "portfolio_exposure"]


@dataclass(frozen=True)
class PortfolioExposure:
    """Per-decision factor exposures and gross-normalized concentration.

    ``factor_exposure`` has shape ``(decision, factor)``, in the order of
    ``factor_names``. A held name with an unknown loading makes that factor's
    exposure unknown, rather than treating the name as neutral.
    ``concentration`` is ``sum((abs(w) / sum(abs(w))) ** 2)``: one for a
    one-name book and ``1 / n`` for equal absolute weights across ``n`` names.
    Cash has zero factor exposure and zero concentration.
    """

    factor_names: tuple[str, ...]
    factor_exposure: NDArray[np.float64]
    concentration: NDArray[np.float64]


def portfolio_exposure(
    weights: ArrayLike,
    factor_loadings: ArrayLike | None = None,
    *,
    factor_names: tuple[str, ...] = (),
) -> PortfolioExposure:
    """Describe a ``(decision, asset)`` weight path without changing it.

    ``factor_loadings`` is ``(asset, factor)`` for fixed loadings or
    ``(decision, asset, factor)`` for loadings observed at each decision.
    Supply the same estimates the portfolio used; this function estimates
    nothing. Without loadings only concentration is reported. Weights within
    ``1e-12`` of zero are treated as roundoff, as in the backtest's held-name
    count. Nonfinite weights leave that decision's diagnostics unknown.
    """
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("weights must be a 2-D (decision, asset) array")
    held = np.where(np.abs(values) <= 1e-12, 0.0, values)
    held[~np.isfinite(held)] = np.nan
    gross = np.abs(held).sum(axis=1)
    concentration = np.divide(
        (held * held).sum(axis=1), gross * gross,
        out=np.zeros(len(held)), where=gross != 0,
    )
    if factor_loadings is None:
        loadings = np.empty((*held.shape, 0))
    else:
        loadings = np.asarray(factor_loadings, dtype=np.float64)
        if loadings.ndim == 2 and loadings.shape[0] == held.shape[1]:
            loadings = np.broadcast_to(loadings, (len(held), *loadings.shape))
        if loadings.ndim != 3 or loadings.shape[:2] != held.shape:
            raise ValueError("factor_loadings must have shape (asset, factor) "
                             "or (decision, asset, factor)")
    names = tuple(factor_names) or tuple(f"factor_{i}" for i in range(loadings.shape[2]))
    if len(names) != loadings.shape[2]:
        raise ValueError("factor_names must name each loading column")
    exposure = np.where(held[:, :, None] == 0, 0.0,
                        held[:, :, None] * np.where(np.isfinite(loadings), loadings, np.nan)
                        ).sum(axis=1)
    return PortfolioExposure(names, exposure, concentration)
