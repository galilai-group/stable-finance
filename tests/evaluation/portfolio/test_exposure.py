"""Exposure diagnostics use final weights and preserve uncertainty."""

from dataclasses import fields
import json

import numpy as np
import pytest

from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.portfolio.exposure import (
    PortfolioExposureLimits, portfolio_exposure_report,
)


def book(values=(.4, .1, -.3, -.2), *, assets=None, decisions=None, horizons=None):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :, None]
    elif array.ndim == 2:
        array = array[:, :, None]
    d, n, h = array.shape
    return PortfolioWeights(array, np.arange(d) if decisions is None else decisions,
        np.arange(n) if assets is None else assets,
        np.arange(1, h+1)*300 if horizons is None else horizons)


def test_final_neutral_book_has_nonzero_beta_and_sector_exposure():
    report = portfolio_exposure_report(book(), beta=[1.5, .5, 1., 2.],
        industry=["Tech", "Tech", "Finance", "Finance"],
        limits=PortfolioExposureLimits(max_abs_net=0., max_abs_beta=0., max_abs_industry=0.))
    np.testing.assert_allclose(report.gross, 1.)
    np.testing.assert_allclose(report.net, 0., atol=1e-15)
    np.testing.assert_allclose(report.long_exposure, .5)
    np.testing.assert_allclose(report.short_exposure, .5)
    np.testing.assert_allclose(report.beta_exposure, -.05, atol=1e-15)
    np.testing.assert_array_equal(report.industry_labels, ["Finance", "Tech"])
    np.testing.assert_allclose(report.industry_net, [[[-.5, .5]]])
    np.testing.assert_allclose(report.industry_gross, [[[.5, .5]]])
    np.testing.assert_allclose(report.concentration_hhi, .3)
    np.testing.assert_allclose(report.effective_names, 10/3)
    np.testing.assert_allclose(report.max_abs_weight, .4)
    np.testing.assert_array_equal(report.names_held, 4)
    np.testing.assert_array_equal(report.constraint_status, "fail")
    np.testing.assert_allclose(report.violations["max_abs_beta"], .05, atol=1e-15)
    np.testing.assert_allclose(report.violations["max_abs_industry"], .5)


def test_concentration_is_gross_normalized_and_scale_invariant():
    first = portfolio_exposure_report(book())
    second = portfolio_exposure_report(book(np.asarray([.4, .1, -.3, -.2])*3))
    np.testing.assert_allclose(first.concentration_hhi, second.concentration_hhi)
    np.testing.assert_allclose(first.effective_names, second.effective_names)
    np.testing.assert_allclose(second.gross, 3.)
    assert first.constraint_status[0, 0] == "not_checked"
    assert not first.violations


def test_cash_has_zero_exposures_and_full_coverage_without_factor_data():
    report = portfolio_exposure_report(book([0., 0.]),
        limits=PortfolioExposureLimits(gross_limit=0., max_weight=0., max_abs_net=0.,
                                      max_abs_beta=0., max_abs_industry=0., max_hhi=0.))
    np.testing.assert_array_equal(report.beta_exposure, 0.)
    np.testing.assert_array_equal(report.beta_coverage, 1.)
    np.testing.assert_array_equal(report.industry_coverage, 1.)
    np.testing.assert_array_equal(report.concentration_hhi, 0.)
    np.testing.assert_array_equal(report.effective_names, 0.)
    assert report.industry_net.shape == (1, 1, 0)
    assert report.constraint_status[0, 0] == "pass"


def test_missing_beta_on_held_name_is_unknown_but_on_zero_name_is_irrelevant():
    limits = PortfolioExposureLimits(max_abs_beta=.5)
    incomplete = portfolio_exposure_report(book([.2, -.1, 0.]), beta=[1., np.nan, np.nan], limits=limits)
    assert np.isnan(incomplete.beta_exposure[0, 0])
    assert incomplete.beta_coverage[0, 0] == pytest.approx(2/3)
    assert np.isnan(incomplete.violations["max_abs_beta"][0, 0])
    assert incomplete.constraint_status[0, 0] == "unknown"
    complete = portfolio_exposure_report(book([.2, 0., 0.]), beta=[1., np.nan, np.nan], limits=limits)
    assert complete.beta_exposure[0, 0] == pytest.approx(.2)
    assert complete.beta_coverage[0, 0] == 1.
    assert complete.constraint_status[0, 0] == "pass"


def test_tiny_unknown_held_name_is_not_hidden_by_coverage_rounding():
    report = portfolio_exposure_report(book([1., 1e-30]), beta=[1., np.nan],
        industry=["A", None], limits=PortfolioExposureLimits(max_abs_beta=2., max_abs_industry=2.))
    assert report.beta_coverage[0, 0] == report.industry_coverage[0, 0] == 1.
    assert np.isnan(report.beta_exposure[0, 0])
    assert np.isnan(report.violations["max_abs_industry"][0, 0])
    assert report.constraint_status[0, 0] == "unknown"


def test_known_constraint_failure_dominates_unknown_factor_constraint():
    report = portfolio_exposure_report(book([.7, -.4]),
        limits=PortfolioExposureLimits(gross_limit=1., max_abs_beta=0.))
    assert report.violations["gross_limit"][0, 0] == pytest.approx(.1)
    assert np.isnan(report.violations["max_abs_beta"][0, 0])
    assert report.constraint_status[0, 0] == "fail"


def test_industry_unknown_bucket_reconciles_net_and_gross():
    weights = book([[.2, -.1, .3, -.4], [.1, -.2, 0., .1]])
    report = portfolio_exposure_report(weights,
        industry=[["Tech", "Tech", None, ""], ["Tech", "Finance", np.nan, "Finance"]],
        limits=PortfolioExposureLimits(max_abs_industry=.2))
    np.testing.assert_allclose(report.industry_net.sum(axis=2)+report.unknown_industry_net, report.net, atol=1e-15)
    np.testing.assert_allclose(report.industry_gross.sum(axis=2)+report.unknown_industry_gross, report.gross)
    np.testing.assert_allclose(report.unknown_industry_net[:, 0], [-.1, 0.])
    np.testing.assert_allclose(report.unknown_industry_gross[:, 0], [.7, 0.])
    np.testing.assert_allclose(report.industry_coverage[:, 0], [.3, 1.])
    assert report.constraint_status[:, 0].tolist() == ["unknown", "pass"]


def test_numeric_and_time_varying_industry_codes():
    report = portfolio_exposure_report(book([[.3, -.2], [.1, -.4]]),
        industry=[[20., 10.], [10., 20.]])
    np.testing.assert_array_equal(report.industry_labels, [10., 20.])
    np.testing.assert_allclose(report.industry_net[:, 0], [[-.2, .3], [.1, -.4]])


def test_large_integer_industry_identifier_is_not_coerced_into_another_code():
    large = 2**53+1
    report = portfolio_exposure_report(book([.3, -.2]), industry=[large, float(large-1)])
    assert report.industry_labels.tolist() == [float(large-1), large]
    np.testing.assert_allclose(report.industry_net, [[[-.2, .3]]])
    np.testing.assert_allclose(report.industry_gross.sum(axis=2), report.gross)


@pytest.mark.parametrize("factor", [np.array([1., 2.]), np.array([[1., 2.], [2., 1.]]),
    np.array([[[1., 3.], [2., 4.]], [[2., 4.], [1., 3.]]])])
def test_explicit_beta_broadcasts_to_each_decision_and_horizon(factor):
    weights = book(np.array([[[.2, .1], [-.1, -.2]], [[.3, .2], [-.2, -.1]]]))
    report = portfolio_exposure_report(weights, beta=factor)
    if factor.ndim == 1:
        expected = [[0., -.3], [-.1, 0.]]
    elif factor.ndim == 2:
        expected = [[0., -.3], [.4, .3]]
    else:
        expected = [[0., -.5], [.4, .5]]
    np.testing.assert_allclose(report.beta_exposure, expected, atol=1e-15)


def test_asset_permutation_preserves_exposures_with_corresponding_factor_order():
    weights, beta = book(), np.array([1.5, .5, 1., 2.])
    sectors, order = np.array(["Tech", "Tech", "Finance", "Finance"]), [2, 0, 3, 1]
    original = portfolio_exposure_report(weights, beta=beta, industry=sectors)
    permuted = portfolio_exposure_report(book(weights.values[:, order], assets=weights.assets[order]),
                                         beta=beta[order], industry=sectors[order])
    for name in ("gross", "net", "beta_exposure", "industry_net", "industry_gross", "concentration_hhi"):
        np.testing.assert_allclose(getattr(original, name), getattr(permuted, name), atol=1e-15)
    np.testing.assert_array_equal(original.industry_labels, permuted.industry_labels)


def test_report_is_readonly_independent_and_json_safe():
    weights, beta = book(), np.array([1., np.nan, 2., 3.])
    report = portfolio_exposure_report(weights, beta=beta,
        limits=PortfolioExposureLimits(max_abs_beta=1.), industry=["Tech", None, "Finance", "Finance"])
    before = report.gross.copy()
    weights.values[:] = 0.
    beta[:] = 0.
    np.testing.assert_array_equal(report.gross, before)
    for field in fields(report):
        value = getattr(report, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
    with pytest.raises(TypeError):
        report.violations["new"] = np.zeros((1, 1))
    assert not report.violations["max_abs_beta"].flags.writeable
    encoded = json.dumps(report.as_dict(), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["beta_exposure"] == [[None]]
    assert decoded["violations"]["max_abs_beta"] == [[None]]
    assert decoded["industry_labels"] == ["Finance", "Tech"]


def test_datetime_labels_serialize_without_losing_readability():
    report = portfolio_exposure_report(book(decisions=np.array(["2020-09-01T10:00:00"], dtype="datetime64[ns]")))
    assert report.as_dict()["decisions"] == ["2020-09-01T10:00:00.000000000"]
    json.dumps(report.as_dict(), allow_nan=False)


def test_longdouble_axis_and_limits_serialize_without_recursive_scalar_conversion():
    report = portfolio_exposure_report(book(decisions=np.array([0], dtype=np.longdouble)),
        limits=PortfolioExposureLimits(gross_limit=np.longdouble(1.)))
    decoded = json.loads(json.dumps(report.as_dict(), allow_nan=False))
    assert decoded["decisions"] == [0.]
    assert decoded["limits"]["gross_limit"] == 1.


def test_raw_violation_and_tolerance_are_separate():
    report = portfolio_exposure_report(book([.5+1e-10, -.5]), tolerance=1e-8,
        limits=PortfolioExposureLimits(gross_limit=1., max_abs_net=0.))
    assert report.violations["gross_limit"][0, 0] > 0.
    assert report.constraint_status[0, 0] == "pass"
    strict = portfolio_exposure_report(book([.5+1e-10, -.5]), tolerance=0.,
        limits=PortfolioExposureLimits(max_abs_net=0.))
    assert strict.constraint_status[0, 0] == "fail"


@pytest.mark.parametrize("kwargs", [dict(beta=[1., np.inf, 1., 1.]), dict(beta=[True]*4),
    dict(beta=np.ones((2, 4))), dict(industry=[1., 2., np.inf, 1.]),
    dict(industry=["Tech", 1, "Tech", 1]), dict(industry=[True]*4),
    dict(industry=np.zeros((1, 4, 1))), dict(tolerance=-1.), dict(tolerance=True),
    dict(limits={"gross_limit": 1.})])
def test_invalid_factor_shapes_values_and_options_raise(kwargs):
    with pytest.raises((ValueError, TypeError)):
        portfolio_exposure_report(book(), **kwargs)


@pytest.mark.parametrize("kwargs", [dict(gross_limit=-1.), dict(max_weight=np.inf),
    dict(max_abs_net=True), dict(max_abs_beta=np.nan), dict(max_hhi=1.01)])
def test_invalid_limits_raise(kwargs):
    with pytest.raises(ValueError):
        PortfolioExposureLimits(**kwargs)


@pytest.mark.parametrize("values", [[np.nan, 0.], [np.inf, 0.], [1e308, 1e308]])
def test_nonfinite_or_overflowing_book_is_rejected(values):
    with pytest.raises(ValueError):
        portfolio_exposure_report(book(values))


def test_finite_factors_with_overflowing_beta_exposure_are_rejected():
    with pytest.raises(ValueError, match="beta exposure exceeds"):
        portfolio_exposure_report(book([1e200]), beta=[1e200])


def test_duplicate_axis_is_rejected_instead_of_silently_double_counting():
    with pytest.raises(ValueError, match="unique"):
        portfolio_exposure_report(book([.2, -.1], assets=["A", "A"]))


def test_object_axis_with_nonfinite_label_is_rejected():
    with pytest.raises(ValueError, match="finite labels"):
        portfolio_exposure_report(book(decisions=np.array([np.nan], dtype=object)))


def test_missing_factor_coverage_stays_in_fraction_range_for_fortran_weights():
    rng = np.random.default_rng(9)
    values = np.asfortranarray(rng.normal(size=(60, 40)))
    weights = PortfolioWeights(values[:, :, None], np.arange(60), np.arange(40), [300])
    report = portfolio_exposure_report(weights)
    assert np.all((report.industry_coverage >= 0.) & (report.industry_coverage <= 1.))
    assert np.all((report.beta_coverage >= 0.) & (report.beta_coverage <= 1.))
    np.testing.assert_allclose(report.industry_coverage, 0., atol=1e-15)
