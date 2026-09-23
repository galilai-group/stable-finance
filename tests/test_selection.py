"""Cost-aware selection: which names are worth their own spread, and when."""

import numpy as np
import pytest

from stable_finance.metrics import edge_to_cost
from stable_finance.selection import (
    persistent_selection,
    select_book,
    tail_selection,
    threshold_selection,
)


def book(forecast, cost=None, *, eligible=None):
    forecast = np.asarray(forecast, dtype=float)
    if eligible is None:
        eligible = np.ones(forecast.shape, dtype=bool)
    if cost is None:
        cost = np.full(forecast.shape, 1e-4)
    return forecast, np.asarray(eligible), np.asarray(cost, dtype=float)


class TestThresholdSelection:
    def test_a_higher_multiple_holds_a_subset_of_a_lower_one(self):
        rng = np.random.default_rng(0)
        mu = rng.normal(scale=3e-4, size=(20, 40))
        f, el, c = book(mu)
        previous = None
        for multiple in (0.0, 0.5, 1.0, 2.0, 4.0):
            signs = threshold_selection(f, el, multiple=multiple, cost=c)
            if previous is not None:
                # Tightening can only remove names, never add or flip one.
                assert np.all((signs == 0) | (signs == previous))
            previous = signs

    def test_a_wide_spread_name_needs_a_larger_forecast(self):
        mu = np.array([[-3e-4, 0.0, 3e-4]])
        f, el, _ = book(mu)
        cheap = np.full(mu.shape, 1e-4)
        assert threshold_selection(f, el, multiple=1.0, cost=cheap)[0].tolist() \
            == [-1, 0, 1]
        # Same forecasts, same threshold; only the last name's cost changed.
        dear = np.array([[1e-4, 1e-4, 1e-3]])
        assert threshold_selection(f, el, multiple=1.0, cost=dear)[0].tolist() \
            == [-1, 0, 0]

    def test_the_cut_is_on_the_median_not_on_zero(self):
        # Every forecast is positive; a rule reading raw sign would go all long.
        mu = np.array([[1e-3, 2e-3, 3e-3]])
        f, el, c = book(mu)
        signs = threshold_selection(f, el, multiple=1.0, cost=c)
        assert signs.sum() == 0 and set(signs[0].tolist()) == {-1, 0, 1}

    def test_an_unpriced_name_is_never_selected(self):
        mu = np.array([[-3e-4, 0.0, 3e-4]])
        f, el, _ = book(mu)
        cost = np.array([[1e-4, 1e-4, np.nan]])
        assert threshold_selection(f, el, multiple=0.0, cost=cost)[0, 2] == 0

    def test_long_only_drops_the_short_leg(self):
        mu = np.array([[-3e-4, 0.0, 3e-4]])
        f, el, c = book(mu)
        signs = threshold_selection(f, el, multiple=0.5, cost=c, long_only=True)
        assert signs.min() == 0 and signs.max() == 1

    def test_shapes_and_multiples_are_validated(self):
        f, el, c = book(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            threshold_selection(f, el, multiple=-1.0, cost=c)
        with pytest.raises(ValueError):
            threshold_selection(f, el, multiple=1.0, cost=np.zeros((2, 2)))


class TestPersistentSelection:
    def test_a_position_survives_a_forecast_too_weak_to_have_opened_it(self):
        # Decision 0 opens on a strong edge; decision 1's edge is far below the
        # entry bar but has not turned, so the position is kept. This is the
        # asymmetry the rule exists for.
        mu = np.array([[-5e-4, 0.0, 5e-4], [-1e-5, 0.0, 1e-5]])
        f, el, c = book(mu)
        signs = persistent_selection(f, el, enter=2.0, exit_=0.0, cost=c)
        assert signs[0].tolist() == [-1, 0, 1]
        assert signs[1].tolist() == [-1, 0, 1]
        # The pointwise rule at the same threshold closes everything.
        assert threshold_selection(f, el, multiple=2.0, cost=c)[1].tolist() \
            == [0, 0, 0]

    def test_a_sign_flip_closes_the_position_when_exit_is_zero(self):
        mu = np.array([[-5e-4, 0.0, 5e-4], [5e-4, 0.0, -5e-4]])
        f, el, c = book(mu)
        signs = persistent_selection(f, el, enter=2.0, exit_=0.0, cost=c)
        # Reversed far enough to re-open on the other side in the same decision.
        assert signs[1].tolist() == [1, 0, -1]

    def test_a_positive_exit_tolerates_a_mild_reversal(self):
        # Edge reverses by 0.5x cost: inside a 1.0x tolerance band, outside a
        # zero one.
        mu = np.array([[-5e-4, 0.0, 5e-4], [5e-5, 0.0, -5e-5]])
        f, el, c = book(mu)
        assert persistent_selection(f, el, enter=2.0, exit_=0.0, cost=c)[1].tolist() \
            == [0, 0, 0]
        assert persistent_selection(f, el, enter=2.0, exit_=1.0, cost=c)[1].tolist() \
            == [-1, 0, 1]

    def test_a_negative_exit_closes_on_decayed_conviction(self):
        mu = np.array([[-5e-4, 0.0, 5e-4], [-1e-5, 0.0, 1e-5]])
        f, el, c = book(mu)
        # Still favourable, but only 0.1x cost; a -0.5 exit wants 0.5x to hold.
        assert persistent_selection(f, el, enter=2.0, exit_=-0.5, cost=c)[1].tolist() \
            == [0, 0, 0]

    def test_an_ineligible_name_is_closed_not_held(self):
        mu = np.array([[-5e-4, 0.0, 5e-4], [-5e-4, 0.0, 5e-4]])
        f, _, c = book(mu)
        eligible = np.ones(mu.shape, dtype=bool)
        eligible[1, 2] = False
        assert persistent_selection(f, eligible, enter=2.0, exit_=0.0,
                                    cost=c)[1, 2] == 0

    def test_it_turns_over_less_than_the_pointwise_rule_it_generalizes(self):
        rng = np.random.default_rng(3)
        # A forecast that persists weakly, as a real one does.
        mu = np.cumsum(rng.normal(scale=1e-4, size=(60, 30)), axis=0) * 0.3 \
            + rng.normal(scale=2e-4, size=(60, 30))
        f, el, c = book(mu)
        churn = lambda s: np.abs(np.diff(s.astype(float), axis=0)).sum()
        held = persistent_selection(f, el, enter=1.0, exit_=0.0, cost=c)
        pointwise = threshold_selection(f, el, multiple=1.0, cost=c)
        assert churn(held) < churn(pointwise)


class TestEdgeToCost:
    def test_it_is_the_centred_forecast_over_the_spread(self):
        mu = np.array([[1e-4, 2e-4, 6e-4]])
        spread = np.full(mu.shape, 1e-4)
        ratio = edge_to_cost(mu, spread)
        # Median is 2e-4, so the edges are -1e-4, 0, 4e-4 over a 1e-4 cost.
        assert ratio[0] == pytest.approx([1.0, 0.0, 4.0])

    def test_it_agrees_with_what_the_threshold_selects(self):
        rng = np.random.default_rng(5)
        mu = rng.normal(scale=3e-4, size=(15, 25))
        spread = rng.uniform(5e-5, 5e-4, size=mu.shape)
        eligible = np.ones(mu.shape, dtype=bool)
        ratio = edge_to_cost(mu, spread, eligible=eligible)
        for multiple in (0.5, 1.0, 2.0):
            selected = threshold_selection(mu, eligible, multiple=multiple,
                                           cost=spread) != 0
            assert np.array_equal(selected, np.nan_to_num(ratio) > multiple)

    def test_an_unquoted_name_is_nan_not_infinite(self):
        mu = np.array([[1e-4, 2e-4, 3e-4]])
        spread = np.array([[1e-4, 1e-4, np.nan]])
        assert np.isnan(edge_to_cost(mu, spread)[0, 2])
        spread = np.array([[1e-4, 1e-4, 0.0]])
        assert np.isnan(edge_to_cost(mu, spread)[0, 2])
