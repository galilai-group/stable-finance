"""Small stand-ins shared by tests in several folders."""

import numpy as np

from stable_finance.dataset import CrossSectionalTargetBundle


class IdentityStats:
    """A target table that standardizes nothing: every transform is the raw value."""

    def transform(self, raw, *_args):
        values = np.asarray(raw)
        return CrossSectionalTargetBundle(values, values, values, values)

