"""The axes every evaluation contract shares, and the check that two agree.

The first two axes always identify a decision and an asset. A third axis, when
present, identifies a forecast horizon. Keeping those axes explicit prevents a
surprisingly common class of research bugs: comparing predictions and outcomes
whose rows happen to have the same length but refer to different observations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _one_dimensional(name: str, value: ArrayLike) -> NDArray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


def require_aligned(left, right) -> None:
    """Raise when two stage outputs do not describe the identical grid."""
    for axis in ("decisions", "assets", "horizons"):
        if not np.array_equal(getattr(left, axis), getattr(right, axis)):
            raise ValueError(f"{axis} are not aligned")
