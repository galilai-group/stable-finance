"""The axes every evaluation contract shares, and the check that two agree.

The first two axes always identify a decision and an asset. A third axis, when
present, identifies a forecast horizon. Keeping those axes explicit prevents a
surprisingly common class of research bugs: comparing predictions and outcomes
whose rows happen to have the same length but refer to different observations.
"""

from __future__ import annotations

import numpy as np

from stable_finance.dataset.market import _one_dimensional  # noqa: F401


def require_aligned(left, right) -> None:
    """Raise when two stage outputs do not describe the identical grid."""
    for axis in ("decisions", "assets", "horizons"):
        if not np.array_equal(getattr(left, axis), getattr(right, axis)):
            raise ValueError(f"{axis} are not aligned")


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average-tie ranks without requiring scipy."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = starts.cumsum()
    bounds = np.r_[np.flatnonzero(starts), len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = 0.5 * (bounds[dense] + bounds[dense - 1] + 1)
    return ranks
