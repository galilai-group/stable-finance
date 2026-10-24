"""What the data says about which names it can price at all."""
from __future__ import annotations

import numpy as np

from stable_finance.dataset import presence_screen, tradable


def test_tradable_needs_a_finite_non_negative_half_spread():
    spread = np.array([[1e-4, np.nan, -1e-4, 0.0]])
    np.testing.assert_array_equal(tradable(spread), [[True, False, False, True]])


def test_presence_screen_is_an_asset_mask():
    observed = np.array([[True, True], [True, False], [True, False]])
    np.testing.assert_array_equal(presence_screen(observed, minimum=0.6), [True, False])
