"""Execution quality as an axis: E/Q, the break-even, and the Sharpe curve."""

import numpy as np
import pytest

from stable_finance.costs import (
    EQ_EMPIRICAL_RANGE,
    breakeven_eq,
    effective_half_spread,
    trading_cost,
)
from stable_finance.metrics import sharpe_ratio, sharpe_vs_execution_quality


class TestEffectiveHalfSpread:
    def test_at_one_it_is_the_quoted_half_spread(self):
        quoted = np.array([[1e-4, 5e-4, 2e-3]])
        assert effective_half_spread(quoted, 1.0) == pytest.approx(quoted)

    def test_it_is_the_same_scaling_trading_cost_applies(self):
        # The two spellings must not drift apart: `multiple` IS E/Q.
        quoted = np.array([[1e-4, 5e-4, 2e-3]])
        for eq in (0.0, 0.4, 1.0, 2.5):
            assert trading_cost("cross", quoted, multiple=eq) == \
                pytest.approx(effective_half_spread(quoted, eq))

    def test_a_midpoint_fill_costs_nothing_where_a_market_was_quoted(self):
        quoted = np.array([[1e-4, 5e-4]])
        assert effective_half_spread(quoted, 0.0) == pytest.approx(0.0)

    def test_an_unquoted_name_is_untradable_at_every_ratio(self):
        # Including zero: E/Q is a claim about execution quality, not a claim
        # that a market which was not quoting can be traded in.
        quoted = np.array([[1e-4, np.nan]])
        for eq in (0.0, 0.5, 1.0):
            assert np.isinf(effective_half_spread(quoted, eq)[0, 1])

    def test_a_negative_ratio_is_refused(self):
        with pytest.raises(ValueError, match="eq must be"):
            effective_half_spread(np.array([[1e-4]]), -0.1)


class TestBreakevenEq:
    def test_it_is_the_ratio_of_mean_gross_to_mean_cost(self):
        gross = np.full(50, 6e-4)
        cost = np.full(50, 2e-4)
        assert breakeven_eq(gross, cost) == pytest.approx(3.0)

    def test_a_strategy_that_loses_money_gross_has_no_break_even(self):
        assert np.isnan(breakeven_eq(np.full(10, -1e-4), np.full(10, 1e-4)))

    def test_a_free_strategy_never_breaks_even(self):
        assert np.isinf(breakeven_eq(np.full(10, 1e-4), np.zeros(10)))

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            breakeven_eq(np.zeros(5), np.zeros(4))


class TestSharpeVsExecutionQuality:
    def _series(self, mean=5e-4, cost=5e-4, n=400, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(mean, 1e-3, n), np.full(n, cost)

    def test_the_curve_starts_at_the_gross_sharpe(self):
        gross, cost = self._series()
        curve = sharpe_vs_execution_quality(gross, cost)
        assert curve.eq[0] == 0.0
        assert curve.sharpe[0] == pytest.approx(curve.gross_sharpe)
        assert curve.gross_sharpe == pytest.approx(sharpe_ratio(gross))

    def test_it_falls_monotonically_as_execution_worsens(self):
        gross, cost = self._series()
        curve = sharpe_vs_execution_quality(gross, cost)
        assert np.all(np.diff(curve.sharpe) < 0)

    def test_it_crosses_zero_exactly_at_the_break_even(self):
        # The claim the whole figure rests on: the break-even is where the
        # curve dies, so reporting the scalar and plotting the curve say the
        # same thing.
        gross, cost = self._series()
        curve = sharpe_vs_execution_quality(gross, cost)
        assert curve.at(curve.breakeven) == pytest.approx(0.0, abs=2e-3)
        assert curve.at(curve.breakeven * 0.8) > 0
        assert curve.at(curve.breakeven * 1.2) < 0

    def test_it_will_not_extrapolate_past_the_swept_grid(self):
        gross, cost = self._series()
        curve = sharpe_vs_execution_quality(gross, cost, eq_grid=[0.0, 0.5, 1.0])
        assert np.isnan(curve.at(1.4))
        assert np.isfinite(curve.at(1.0))

    def test_one_backtest_answers_every_ratio(self):
        # The point of the linearity: no re-running. A point on the curve
        # must equal the Sharpe of the explicitly netted series.
        gross, cost = self._series()
        curve = sharpe_vs_execution_quality(gross, cost, eq_grid=[0.3, 0.75, 1.0])
        for index, eq in enumerate([0.3, 0.75, 1.0]):
            assert curve.sharpe[index] == pytest.approx(sharpe_ratio(gross - eq * cost))

    def test_a_varying_cost_is_charged_period_by_period(self):
        # Costs are not a scalar in practice: a decision holding wide names
        # pays more. Netting must be elementwise, not on the mean.
        rng = np.random.default_rng(2)
        gross = rng.normal(5e-4, 1e-3, 300)
        cost = rng.uniform(1e-4, 6e-4, 300)
        curve = sharpe_vs_execution_quality(gross, cost, eq_grid=[1.0])
        assert curve.sharpe[0] == pytest.approx(sharpe_ratio(gross - cost))
        assert curve.sharpe[0] != pytest.approx(sharpe_ratio(gross - cost.mean()))

    def test_the_error_band_reflects_the_sample_length(self):
        # Lo (2002): halving the sample widens the band by sqrt(2).
        gross, cost = self._series(n=504)
        short = sharpe_vs_execution_quality(gross[:252], cost[:252], eq_grid=[0.0])
        long = sharpe_vs_execution_quality(gross, cost, eq_grid=[0.0])
        assert short.standard_error[0] > long.standard_error[0]
        assert short.periods == 252 and long.periods == 504

    def test_non_finite_periods_are_dropped_not_propagated(self):
        gross, cost = self._series(n=100)
        gross[5] = np.nan
        curve = sharpe_vs_execution_quality(gross, cost, eq_grid=[0.5])
        assert curve.periods == 99
        assert np.isfinite(curve.sharpe[0])

    def test_the_empirical_range_brackets_full_crossing(self):
        # The band drawn on the figure must not include regimes better than
        # a midpoint fill, nor claim crossing is atypical.
        low, high = EQ_EMPIRICAL_RANGE
        assert 0.0 < low < high <= 1.0

    def test_mismatched_inputs_are_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            sharpe_vs_execution_quality(np.zeros(5), np.zeros(4))
        with pytest.raises(ValueError, match="cannot be negative"):
            sharpe_vs_execution_quality(np.zeros(5), -np.ones(5))
