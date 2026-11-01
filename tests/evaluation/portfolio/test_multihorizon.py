"""Fixed-capital target mixing and handoff to the existing execution path."""

import numpy as np
import pytest

from stable_finance import (
    DEFAULT_BACKTEST,
    ForwardReturns,
    PortfolioWeights,
    combine_horizon_targets,
    mean_variance_target,
    orders_from_weights,
    run_backtest,
)
from stable_finance.dataset import TO_CLOSE


def targets():
    return PortfolioWeights(
        [[[.10, -.06], [-.10, .06]], [[.04, .08], [-.04, -.08]]],
        [0, 300], ["B", "A"], [900, 300],
    )


def test_fixed_capital_fractions_preserve_labels_and_have_the_hand_computed_weights():
    source = targets()
    result = combine_horizon_targets(source, capital_weights=[.25, .75], target_horizon=300)
    np.testing.assert_allclose(result.values[:, :, 0], [[-.02, .02], [.07, -.07]])
    np.testing.assert_array_equal(result.decisions, source.decisions)
    np.testing.assert_array_equal(result.assets, source.assets)
    np.testing.assert_array_equal(result.horizons, [300])


def test_equal_default_and_a_single_source_leave_capital_allocation_explicit():
    source = targets()
    equal = combine_horizon_targets(source, target_horizon=300)
    np.testing.assert_allclose(equal.values[:, :, 0], [[.02, -.02], [.06, -.06]])
    single = PortfolioWeights(source.values[:, :, :1], source.decisions, source.assets, [900])
    np.testing.assert_array_equal(
        combine_horizon_targets(single, target_horizon=300).values, single.values)


def test_opposing_books_and_cash_reserves_are_not_rescaled_to_fill_gross():
    source = PortfolioWeights([[[.1, -.1], [-.1, .1]]], [0], ["A", "B"], [300, 900])
    np.testing.assert_array_equal(combine_horizon_targets(source, target_horizon=300).values, 0.)
    cash = PortfolioWeights([[[.1, 0.], [-.1, 0.]]], [0], ["A", "B"], [300, 900])
    mixed = combine_horizon_targets(cash, capital_weights=[.25, .75], target_horizon=300)
    np.testing.assert_allclose(mixed.values[:, :, 0], [[.025, -.025]])
    all_cash = combine_horizon_targets(cash, capital_weights=[0., 1.], target_horizon=300)
    np.testing.assert_array_equal(all_cash.values, 0.)


def test_source_and_output_horizons_can_use_the_existing_to_close_label():
    source = PortfolioWeights([[[.1, .2], [-.1, -.2]]], [0], ["A", "B"], [TO_CLOSE, 300])
    result = combine_horizon_targets(source, target_horizon=TO_CLOSE)
    np.testing.assert_allclose(result.values[:, :, 0], [[.15, -.15]])
    np.testing.assert_array_equal(result.horizons, [TO_CLOSE])


def test_mixing_does_not_mutate_or_alias_component_books():
    source = targets()
    before = source.values.copy()
    result = combine_horizon_targets(source, target_horizon=300)
    np.testing.assert_array_equal(source.values, before)
    result.values[:] = 0.
    result.assets[0] = "Z"
    np.testing.assert_array_equal(source.values, before)
    np.testing.assert_array_equal(source.assets, ["B", "A"])


@pytest.mark.parametrize("fractions", [[1.], [[.5, .5]], [.4, .4], [-.1, 1.1],
                                     [np.nan, .5], [np.inf, 0.], [True, False]])
def test_invalid_capital_allocations_raise_instead_of_being_normalized(fractions):
    with pytest.raises(ValueError, match="capital_weights"):
        combine_horizon_targets(targets(), capital_weights=fractions, target_horizon=300)


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_unknown_sources_are_rejected_even_when_their_capital_fraction_is_zero(missing):
    source = targets()
    source.values[0, 0, 1] = missing
    with pytest.raises(ValueError, match="finite"):
        combine_horizon_targets(source, capital_weights=[1., 0.], target_horizon=300)


@pytest.mark.parametrize("horizon", [0, -2, True, 300.5])
def test_output_horizon_must_be_an_explicit_duration_or_to_close(horizon):
    with pytest.raises(ValueError, match="target_horizon"):
        combine_horizon_targets(targets(), target_horizon=horizon)


def test_native_mean_variance_targets_are_used_without_another_optimizer():
    forecast = ForwardReturns([[[.02, .06], [-.02, -.06]]], [0], ["A", "B"], [300, 900])
    sigma = np.array([np.eye(2)*.1, np.eye(2)*.2])
    components = mean_variance_target(forecast, sigma, risk_aversion=2.)
    combined = combine_horizon_targets(components, capital_weights=[.25, .75], target_horizon=300)
    np.testing.assert_allclose(combined.values[:, :, 0], [[.1375, -.1375]])


def test_existing_orders_and_harness_charge_the_net_book_once():
    combined = combine_horizon_targets(targets(), capital_weights=[.25, .75], target_horizon=300)
    orders = orders_from_weights(combined)
    np.testing.assert_allclose(orders.values[:, :, 0], [[-.02, .02], [.09, -.09]])
    realized = ForwardReturns([[[.01], [-.01]], [[.02], [-.02]]],
                              combined.decisions, combined.assets, combined.horizons)
    report = run_backtest(combined, realized, half_spread=np.full((2, 2), .001),
        config=DEFAULT_BACKTEST.replace(efq=.7, annualization="decisions"), with_series=True)
    np.testing.assert_allclose(report.series.gross, [-.0004, .0028], atol=1e-14)
    np.testing.assert_allclose(report.series.cost, [.000028, .000126], atol=1e-14)
    np.testing.assert_allclose(report.series.net, [-.000428, .002674], atol=1e-14)
    assert report.mean_turnover == pytest.approx(.11)
