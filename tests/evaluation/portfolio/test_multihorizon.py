"""Static multi-horizon target combination reuses native constructors."""

import numpy as np
import pytest

from stable_finance.evaluation.portfolio.construction import mean_variance_target
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.multihorizon import combine_horizon_targets
from stable_finance.evaluation.returns.contract import ForwardReturns


def books(values=None, *, decisions=None, assets=None, horizons=None):
    if values is None:
        values = [[[.10, -.06], [-.10, .06]], [[.04, .08], [-.04, -.08]]]
    values = np.asarray(values, dtype=float)
    d, n, h = values.shape
    return PortfolioWeights(values, np.arange(d) if decisions is None else decisions,
        np.arange(n) if assets is None else assets,
        np.arange(1, h+1)*300 if horizons is None else horizons)


def combine(value=None, **kwargs):
    return combine_horizon_targets(books() if value is None else value,
                                  target_horizon_seconds=300, **kwargs)


def test_fixed_fractions_match_hand_calculation_and_preserve_labels():
    source = books(assets=["A", "B"])
    result = combine(source, capital_weights=[.25, .75])
    np.testing.assert_allclose(result.values[:, :, 0], [[-.02, .02], [.07, -.07]])
    np.testing.assert_array_equal(result.decisions, source.decisions)
    np.testing.assert_array_equal(result.assets, source.assets)
    np.testing.assert_array_equal(result.horizons, [300])


def test_default_equal_fractions_and_single_horizon_identity():
    source = books()
    np.testing.assert_allclose(combine(source).values[:, :, 0], source.values.mean(axis=2))
    one = books(source.values[:, :, :1], horizons=[900])
    np.testing.assert_array_equal(combine(one).values, one.values)


def test_zero_fraction_and_explicit_cash_do_not_redistribute_capital():
    source = books()
    np.testing.assert_array_equal(combine(source, capital_weights=[1., 0.]).values,
                                   source.values[:, :, :1])
    zero = books(np.stack([source.values[:, :, 0], np.zeros((2, 2))], axis=2))
    np.testing.assert_allclose(combine(zero).values[:, :, 0], source.values[:, :, 0]/2)


def test_opposing_horizon_books_emit_exact_zero_net_target():
    source = books([[[.1, -.1], [-.1, .1]]])
    np.testing.assert_array_equal(combine(source).values, 0.)


def test_convex_combination_preserves_common_neutrality_gross_and_name_bounds():
    source = books()
    result = combine(source, capital_weights=[.7, .3])
    np.testing.assert_allclose(result.values.sum(axis=1), 0., atol=1e-14)
    assert np.abs(result.values).max() <= .1
    assert np.abs(result.values).sum(axis=1).max() <= .2
    # No hidden scaling fills the unused gross allowance.
    assert np.abs(result.values).sum() < np.abs(source.values).sum()


def test_asset_and_source_horizon_permutations_preserve_economic_target():
    source = books()
    expected = combine(source, capital_weights=[.3, .7]).values
    permuted = books(source.values[:, ::-1, ::-1], assets=source.assets[::-1],
                     horizons=source.horizons[::-1])
    actual = combine(permuted, capital_weights=[.7, .3]).values
    np.testing.assert_allclose(actual[:, ::-1], expected)


def test_appending_future_targets_does_not_change_existing_targets():
    source = books()
    prefix = books(source.values[:1], decisions=source.decisions[:1])
    np.testing.assert_array_equal(combine(prefix).values, combine(source).values[:1])


def test_native_mean_variance_targets_are_reused_without_new_optimization():
    forecast = ForwardReturns([[[.02, .06], [-.02, -.06]]], [0], ["A", "B"], [300, 900])
    covariance = np.array([np.eye(2)*.1, np.eye(2)*.3])
    native = mean_variance_target(forecast, covariance, risk_aversion=2.)
    combined = combine(native, capital_weights=[.4, .6])
    np.testing.assert_allclose(native.values, [[[.1, .1], [-.1, -.1]]])
    np.testing.assert_allclose(combined.values, [[[.1], [-.1]]])


def test_returned_arrays_are_independent_readonly_copies():
    source = books()
    before = source.values.copy()
    result = combine(source)
    for field in ("values", "decisions", "assets", "horizons"):
        assert not getattr(result, field).flags.writeable
        assert not np.shares_memory(getattr(source, field), getattr(result, field))
    source.values[:] = 0.
    np.testing.assert_array_equal(result.values[:, :, 0], before.mean(axis=2))


@pytest.mark.parametrize("fractions", [[1.], [[.5, .5]], [.6, .6], [-.1, 1.1],
    [np.nan, .5], [np.inf, 0.], [True, False], [".5", ".5"]])
def test_invalid_capital_fractions_are_rejected(fractions):
    with pytest.raises(ValueError, match="capital_weights"):
        combine(capital_weights=fractions)


@pytest.mark.parametrize("value", [0, -1, 300., True, np.bool_(True), 2**63])
def test_output_interval_requires_explicit_positive_integer_seconds(value):
    with pytest.raises(ValueError, match="target_horizon_seconds"):
        combine_horizon_targets(books(), target_horizon_seconds=value)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_missing_or_infinite_targets_are_rejected_even_with_zero_capital(bad):
    source = books()
    source.values[0, 0, 1] = bad
    with pytest.raises(ValueError, match="target weights must be finite"):
        combine(source, capital_weights=[1., 0.])


@pytest.mark.parametrize("axis, labels", [("decisions", [0, 0]), ("assets", ["A", "A"]),
                                        ("horizons", [300, 300])])
def test_duplicate_axes_are_rejected(axis, labels):
    with pytest.raises(ValueError, match=axis):
        combine(books(**{axis:labels}))


@pytest.mark.parametrize("horizons", [[300.5, 900], [True, True]])
def test_source_horizons_require_integer_seconds(horizons):
    with pytest.raises(ValueError):
        combine(books(horizons=horizons))


def test_nonfinite_labels_and_empty_axes_are_rejected():
    with pytest.raises(ValueError, match="decisions"):
        combine(books(decisions=[0., np.nan]))
    with pytest.raises(ValueError, match="empty"):
        combine(books(np.zeros((0, 2, 2))))


def test_wrong_contract_is_rejected():
    with pytest.raises(TypeError, match="PortfolioWeights"):
        combine_horizon_targets(np.zeros((1, 2, 2)), target_horizon_seconds=300)
