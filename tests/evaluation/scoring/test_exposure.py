"""Exposure diagnostics score the supplied book and preserve missing information."""

import numpy as np
import pytest

from stable_finance.evaluation.execution import orders_from_weights, simulate_fills
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.scoring.exposure import portfolio_exposure_report
from stable_finance.evaluation.scoring.pnl import positions_from_fills


def weights(values=None):
    if values is None:
        values = [[[.3, .2], [-.1, -.2], [.1, 0.]], [[0., .1], [0., .2], [0., -.1]]]
    values = np.asarray(values, dtype=float)
    return PortfolioWeights(values, np.arange(len(values)), list("ABC")[:values.shape[1]],
                            np.arange(1, values.shape[2]+1)*300)


def test_signed_exposures_and_gross_normalized_concentration_have_native_book_axes():
    source = weights()
    report = portfolio_exposure_report(source)
    np.testing.assert_allclose(report["gross"], [[.5, .4], [0., .4]])
    np.testing.assert_allclose(report["net"], [[.3, 0.], [0., .2]], atol=1e-15)
    np.testing.assert_allclose(report["max_abs_weight"], [[.3, .2], [0., .2]])
    np.testing.assert_array_equal(report["names_held"], [[3, 2], [0, 3]])
    np.testing.assert_allclose(report["concentration_hhi"], [[.44, .5], [0., .375]])
    np.testing.assert_allclose(report["effective_names"], [[1/.44, 2.], [0., 1/.375]])
    for name in ("decisions", "assets", "horizons"):
        np.testing.assert_array_equal(report[name], getattr(source, name))
    assert report["factor_exposure"] is None
    assert report["factor_tolerance"] is None
    np.testing.assert_array_equal(report["constraint_status"], "not_checked")


def test_multiple_factor_columns_bounds_and_names_keep_decision_horizon_factor_order():
    report = portfolio_exposure_report(weights(),
        factor_loadings=[[1., 1.], [2., 0.], [.5, 1.]],
        factor_names=["beta", "industry_A"], factor_lower=[-.05, 0.], factor_upper=[.1, .25])
    expected = np.array([[[.15, .4], [-.2, .2]], [[0., 0.], [.45, 0.]]])
    np.testing.assert_allclose(report["factor_exposure"], expected, atol=1e-15)
    np.testing.assert_allclose(report["lower_slack"], expected-[-.05, 0.], atol=1e-15)
    np.testing.assert_allclose(report["upper_slack"], [.1, .25]-expected, atol=1e-15)
    np.testing.assert_allclose(report["violations"], [[[.05, .15], [.15, 0.]], [[0., 0.], [.35, 0.]]], atol=1e-15)
    np.testing.assert_array_equal(report["constraint_status"], [["fail", "fail"], ["pass", "fail"]])
    np.testing.assert_array_equal(report["factor_names"], ["beta", "industry_A"])
    np.testing.assert_array_equal(report["factor_coverage"], 1.)


def test_missing_held_loadings_stay_unknown_and_known_failures_take_precedence():
    source = weights([[[.2], [-.1], [0.]], [[0.], [-.1], [0.]], [[0.], [0.], [0.]]])
    report = portfolio_exposure_report(source, factor_loadings=[[np.nan, 1.], [1., 0.], [np.nan, np.nan]],
        factor_lower=[-.2, 0.], factor_upper=[.2, .1])
    assert np.isnan(report["factor_exposure"][0, 0, 0])
    np.testing.assert_allclose(report["factor_coverage"][:, 0, 0], [1/3, 1., 1.])
    np.testing.assert_allclose(report["factor_exposure"][1:, 0, 0], [-.1, 0.])
    np.testing.assert_array_equal(report["constraint_status"][:, 0], ["fail", "pass", "pass"])
    unknown = portfolio_exposure_report(source, factor_loadings=[[np.nan], [1.], [np.nan]],
        factor_lower=-.2, factor_upper=.2)
    np.testing.assert_array_equal(unknown["constraint_status"][:, 0], ["unknown", "pass", "pass"])
    assert np.isnan(unknown["violations"][0, 0, 0])


def test_factor_reporting_without_bounds_does_not_claim_to_check_constraints():
    report = portfolio_exposure_report(weights(), factor_loadings=[[1.], [1.], [1.]])
    np.testing.assert_allclose(report["factor_exposure"][:, :, 0], report["net"], atol=1e-15)
    np.testing.assert_array_equal(report["factor_names"], ["factor_0"])
    np.testing.assert_array_equal(report["constraint_status"], "not_checked")
    assert report["violations"] is report["lower_slack"] is report["factor_lower"] is None
    assert report["factor_tolerance"] is None


def test_equality_one_sided_limits_and_tolerance_preserve_raw_violations():
    source = weights([[[.100000001], [-.1]]])
    report = portfolio_exposure_report(source, factor_loadings=[[1., 1.], [1., 0.]],
        factor_lower=[0., -np.inf], factor_upper=[0., .2])
    assert report["violations"][0, 0, 0] == pytest.approx(1e-9)
    assert np.isposinf(report["lower_slack"][0, 0, 1])
    np.testing.assert_array_equal(report["constraint_status"], "pass")
    strict = portfolio_exposure_report(source, factor_loadings=[[1.], [1.]],
        factor_lower=0., factor_upper=0., tolerance=0.)
    np.testing.assert_array_equal(strict["constraint_status"], "fail")


@pytest.mark.parametrize("rescaling", [1e-12, 1e12])
def test_factor_units_change_raw_violations_without_changing_limit_status(rescaling):
    source = weights([[[.05 + .5e-8], [0.]], [[.05 + 1.5e-8], [0.]]])
    loadings = np.array([[2.], [-1.]])
    original = portfolio_exposure_report(source, factor_loadings=loadings,
        factor_lower=-.1, factor_upper=.1)
    changed = portfolio_exposure_report(source, factor_loadings=loadings*rescaling,
        factor_lower=-.1*rescaling, factor_upper=.1*rescaling)
    np.testing.assert_array_equal(original["constraint_status"][:, 0], ["pass", "fail"])
    np.testing.assert_array_equal(changed["constraint_status"], original["constraint_status"])
    np.testing.assert_allclose(changed["violations"], original["violations"]*rescaling, rtol=1e-7)
    np.testing.assert_allclose(original["factor_tolerance"], [2e-8])
    np.testing.assert_allclose(changed["factor_tolerance"], [2e-8*rescaling])
    np.testing.assert_allclose(changed["factor_exposure"], original["factor_exposure"]*rescaling)


def test_tolerance_scaling_uses_known_loadings_and_a_unit_fallback_for_empty_columns():
    report = portfolio_exposure_report(weights([[[0.], [.1]]]),
        factor_loadings=[[np.nan, 0., np.nan], [2., 0., np.nan]],
        factor_lower=-1., factor_upper=1.)
    np.testing.assert_allclose(report["factor_tolerance"], [2e-8, 1e-8, 1e-8])
    np.testing.assert_array_equal(report["constraint_status"], "unknown")


def test_actual_fills_and_price_drift_are_checked_separately_from_a_feasible_target():
    target = weights([[[.1], [-.1]], [[.1], [-.1]]])
    bid, ask = np.array([[99., np.nan], [99., 99.]]), np.array([[101., np.nan], [101., 101.]])
    fills = simulate_fills(orders_from_weights(target), bid, ask)
    actual = positions_from_fills(fills, drift=[[[.1], [0.]], [[0.], [0.]]])
    parameters = dict(factor_loadings=[[1.], [1.]], factor_names=["beta"],
                      factor_lower=-.01, factor_upper=.01)
    planned_report = portfolio_exposure_report(target, **parameters)
    actual_report = portfolio_exposure_report(actual, **parameters)
    np.testing.assert_array_equal(planned_report["constraint_status"], "pass")
    np.testing.assert_array_equal(actual_report["constraint_status"], "fail")
    np.testing.assert_allclose(actual_report["factor_exposure"][:, 0, 0], [.1, .11])
    assert fills.quantity[0, 1, 0] == 0.


def test_reports_do_not_mutate_or_alias_weights_names_axes_or_limits():
    source = weights()
    before = source.values.copy()
    factors, lower, names = np.ones((3, 1)), np.array([-.2]), np.array(["beta"])
    report = portfolio_exposure_report(source, factor_loadings=factors, factor_names=names,
        factor_lower=lower, factor_upper=.2)
    report["factor_lower"][:] = 0.
    report["factor_names"][:] = "other"
    report["assets"][:] = "Z"
    report["gross"][:] = 0.
    np.testing.assert_array_equal(source.values, before)
    np.testing.assert_array_equal(source.assets, list("ABC"))
    np.testing.assert_array_equal(factors, 1.)
    np.testing.assert_array_equal(lower, [-.2])
    np.testing.assert_array_equal(names, ["beta"])


@pytest.mark.parametrize("kwargs", [dict(factor_lower=-.1), dict(factor_names=["beta"]),
    dict(factor_loadings=[1., 1., 1.]), dict(factor_loadings=np.ones((2, 1))),
    dict(factor_loadings=np.ones((3, 0))), dict(factor_loadings=np.full((3, 1), np.inf)),
    dict(factor_loadings=np.ones((3, 1)), factor_names=[""]),
    dict(factor_loadings=np.ones((3, 2)), factor_names=["beta", "beta"]),
    dict(factor_loadings=np.ones((3, 1)), factor_lower=0.),
    dict(factor_loadings=np.ones((3, 1)), factor_lower=[0., 0.], factor_upper=1.),
    dict(factor_loadings=np.ones((3, 1)), factor_lower=1., factor_upper=0.),
    dict(factor_loadings=np.ones((3, 1)), factor_lower=np.inf, factor_upper=np.inf),
    dict(factor_loadings=np.ones((3, 1)), factor_lower=-np.inf, factor_upper=-np.inf),
    dict(factor_loadings=np.ones((3, 1)), factor_lower=np.nan, factor_upper=1.),
])
def test_invalid_factor_definitions_or_incomplete_budgets_raise(kwargs):
    with pytest.raises(ValueError, match="factor"):
        portfolio_exposure_report(weights(), **kwargs)


@pytest.mark.parametrize("tolerance", [-1., np.nan, np.inf, True])
def test_invalid_tolerance_raises(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        portfolio_exposure_report(weights(), tolerance=tolerance)


def test_unknown_weights_cannot_be_reported_as_known_gross_or_factor_exposure():
    source = weights()
    source.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="weights must be finite"):
        portfolio_exposure_report(source)
