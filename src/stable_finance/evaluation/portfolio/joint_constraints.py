"""Position limits for the optional joint multi-horizon reference plan.

These constrain each planning interval. They add no single-horizon allocator
and do not change the peer's existing construction rules.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np

__all__ = ["JointPortfolioConstraints"]


@dataclass(frozen=True)
class JointPortfolioConstraints:
    """Gross and concentration limits; dollar neutrality is optional."""

    gross_limit: float = 1.0
    max_weight: float = 0.1
    dollar_neutral: bool = True

    def __post_init__(self) -> None:
        for name in ("gross_limit", "max_weight"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.dollar_neutral, (bool, np.bool_)):
            raise ValueError("dollar_neutral must be boolean")
