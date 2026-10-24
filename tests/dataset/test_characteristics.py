"""Trailing dollar volume and volatilities from a day-by-day panel."""
from __future__ import annotations

import numpy as np

from stable_finance.dataset import trailing_from_days


def test_trailing_from_days_uses_only_the_days_before_each_row():
    rng = np.random.default_rng(0)
    n, m = 40, 3
    closes = 100 * np.exp(rng.normal(0, 0.01, (n, m)).cumsum(axis=0))
    dollars = rng.uniform(1e6, 2e6, (n, m))
    adv, idio, mvol = trailing_from_days(closes, dollars, lookback=25)
    assert np.isnan(adv[0]).all()
    np.testing.assert_allclose(adv[30], dollars[5:30].mean(axis=0))
    np.testing.assert_allclose(adv[3], dollars[:3].mean(axis=0))
    # Twenty daily returns need twenty-one closes.
    assert np.isnan(idio[:21]).all() and np.isnan(mvol[:21]).all()
    assert np.isfinite(idio[21:]).all() and np.isfinite(mvol[21:]).all()
    # A day's own close and volume never reach its own row.
    louder, higher = dollars.copy(), closes.copy()
    louder[30] *= 100
    higher[30] *= 2
    adv2, idio2, mvol2 = trailing_from_days(higher, louder, lookback=25)
    np.testing.assert_allclose(adv2[30], adv[30])
    np.testing.assert_allclose(idio2[30], idio[30])
    assert adv2[31, 0] > adv[31, 0] and idio2[31, 0] > idio[31, 0]


def test_a_name_that_did_not_trade_is_left_out_of_its_own_average():
    closes = np.full((4, 1), 100.0)
    dollars = np.array([[1e6], [np.nan], [3e6], [0.0]])
    adv, _, _ = trailing_from_days(closes, dollars, lookback=3)
    np.testing.assert_allclose(adv[:, 0], [np.nan, 1e6, 1e6, 2e6])
