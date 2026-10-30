"""Factor budgets are labelled, causal, and independent of portfolio solvers."""

from dataclasses import fields
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stable_finance.evaluation.portfolio.factors import (
    FactorExposureConstraints, factor_constraint_arrays, factor_exposure_report,
)


ASSETS = ["A", "B", "C"]
NAMES = ["beta", "technology"]
LOADINGS = np.array([[1.5, 1.], [.5, 0.], [1., 1.]])


def contract(loadings=LOADINGS, lower=(-.1, -.2), upper=(.1, .3), **kwargs):
    return FactorExposureConstraints(loadings, lower, upper, ASSETS, NAMES, **kwargs)


def resolve(value, **kwargs):
    return value.resolve(decisions=kwargs.pop("decisions", [100, 200]),
        assets=kwargs.pop("assets", ASSETS), horizons=kwargs.pop("horizons", [60, 300]), **kwargs)


def test_static_constraints_broadcast_across_time_and_planning_stages():
    result = resolve(contract())
    assert result.loadings.shape == (2, 2, 3, 2)
    np.testing.assert_array_equal(result.loadings, np.broadcast_to(LOADINGS, (2, 2, 3, 2)))
    np.testing.assert_array_equal(result.lower, np.broadcast_to([-.1, -.2], (2, 2, 2)))
    np.testing.assert_array_equal(result.upper, np.broadcast_to([.1, .3], (2, 2, 2)))
    np.testing.assert_array_equal(result.available_at, [[100, 100], [200, 200]])
    np.testing.assert_array_equal(result.factor_names, NAMES)


def test_one_decision_helper_preserves_asymmetric_equality_and_unbounded_budgets():
    a, lo, hi, names = factor_constraint_arrays(LOADINGS, [-np.inf, .05], [.2, .05],
        stages=2, n_assets=3, factor_names=NAMES)
    np.testing.assert_array_equal(a, np.stack([LOADINGS, LOADINGS]))
    np.testing.assert_array_equal(lo, [[-np.inf, .05], [-np.inf, .05]])
    np.testing.assert_array_equal(hi, [[.2, .05], [.2, .05]])
    np.testing.assert_array_equal(names, NAMES)
    for value in (a, lo, hi, names):
        assert not value.flags.writeable


def test_raw_helper_default_names_and_null_family_are_explicit():
    *_, names = factor_constraint_arrays(LOADINGS, -.1, .1, stages=1, n_assets=3)
    np.testing.assert_array_equal(names, ["factor_0", "factor_1"])
    assert factor_constraint_arrays(None, None, None, stages=1, n_assets=3) == (None,)*4
    with pytest.raises(ValueError, match="require factor_loadings"):
        factor_constraint_arrays(None, 0., None, stages=1, n_assets=3)


def test_missing_coefficients_are_allowed_only_on_explicitly_hard_zero_names():
    missing = LOADINGS.copy()
    missing[2] = np.nan
    value = contract(missing)
    with pytest.raises(ValueError, match="potentially held"):
        resolve(value)
    result = resolve(value, eligible=[True, True, False])
    np.testing.assert_array_equal(result.loadings[:, :, 2], 0.)
    assert np.isnan(value.loadings[2]).all()
    # A later decision that can hold C must not inherit the earlier exclusion.
    with pytest.raises(ValueError, match="potentially held"):
        resolve(value, eligible=[[True, True, False], [True, True, True]])


def test_exclusions_can_vary_by_stage_without_changing_the_source():
    staged = np.stack([LOADINGS, LOADINGS])
    staged[0, 2] = np.nan
    staged[1, 1] = np.nan
    needed = np.array([[True, True, False], [True, False, True]])
    a, *_ = factor_constraint_arrays(staged, -.1, .1, stages=2, n_assets=3, required=needed)
    np.testing.assert_array_equal(a[0, 2], 0.)
    np.testing.assert_array_equal(a[1, 1], 0.)
    assert np.isnan(staged[0, 2]).all()


def test_infinite_coefficient_is_rejected_even_on_an_excluded_name():
    invalid = LOADINGS.copy()
    invalid[2, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        factor_constraint_arrays(invalid, -.1, .1, stages=1, n_assets=3,
                                 required=[True, True, False])


@pytest.mark.parametrize("lower,upper", [
    (None, .1), (-.1, None), (np.nan, .1), (-.1, np.nan),
    (np.inf, np.inf), (-np.inf, -np.inf), (.2, .1),
    ([0.], [.1, .2]), ([[0., 0.]], .1),
])
def test_invalid_or_implicitly_missing_budgets_fail(lower, upper):
    with pytest.raises(ValueError):
        factor_constraint_arrays(LOADINGS, lower, upper, stages=2, n_assets=3)


@pytest.mark.parametrize("kwargs", [
    dict(stages=0, n_assets=3), dict(stages=True, n_assets=3),
    dict(stages=1.5, n_assets=3), dict(stages=1, n_assets=0),
    dict(stages=1, n_assets=3, factor_names=["beta", "beta"]),
    dict(stages=1, n_assets=3, factor_names=["beta", ""]),
    dict(stages=1, n_assets=3, factor_names=["beta"]),
    dict(stages=1, n_assets=3, factor_names=[1, 2]),
    dict(stages=1, n_assets=3, required=[1, 1, 0]),
    dict(stages=1, n_assets=3, required=[[True], [True], [False]]),
])
def test_raw_contract_rejects_unsafe_axes_and_masks(kwargs):
    with pytest.raises(ValueError):
        factor_constraint_arrays(LOADINGS, -.1, .1, **kwargs)


def test_equal_time_and_stage_lengths_still_require_explicit_axis_meaning():
    matrix = np.stack([LOADINGS, LOADINGS*2])
    time_value = contract(matrix, decisions=[100, 200])
    stage_value = contract(matrix, horizons=[60, 300])
    time_result, stage_result = resolve(time_value), resolve(stage_value)
    np.testing.assert_array_equal(time_result.loadings[1, 0], LOADINGS*2)
    np.testing.assert_array_equal(time_result.loadings[0, 1], LOADINGS)
    np.testing.assert_array_equal(stage_result.loadings[1, 0], LOADINGS)
    np.testing.assert_array_equal(stage_result.loadings[0, 1], LOADINGS*2)
    with pytest.raises(ValueError):
        contract(matrix)
    with pytest.raises(ValueError):
        contract(matrix, decisions=[100, 200], horizons=[60, 300])


def test_time_and_stage_subsets_select_exact_labels_without_reordering_assets():
    values = np.arange(3*3*3*2, dtype=float).reshape(3, 3, 3, 2)
    bounds = np.arange(18, dtype=float).reshape(3, 3, 2)
    value = contract(values, bounds, bounds+2., decisions=[100, 200, 300],
        horizons=[60, 300, 900], available_at=[[90, 90, 90], [190, 190, 190], [290, 290, 290]])
    result = resolve(value, decisions=[100, 300], horizons=[60, 900])
    np.testing.assert_array_equal(result.loadings, values[[0, 2]][:, [0, 2]])
    np.testing.assert_array_equal(result.lower, bounds[[0, 2]][:, [0, 2]])
    np.testing.assert_array_equal(result.available_at, [[90, 90], [290, 290]])
    for request in (dict(decisions=[150]), dict(decisions=[400]), dict(horizons=[120])):
        with pytest.raises(ValueError, match="exist exactly"):
            resolve(value, **request)
    with pytest.raises(ValueError, match="exact requested asset order"):
        resolve(value, assets=["B", "A", "C"])


def test_static_loadings_can_carry_independent_time_and_stage_budgets():
    lower = np.array([[[-.1, -.2], [-.3, -.4]], [[-.5, -.6], [-.7, -.8]]])
    value = contract(lower=lower, upper=-lower, decisions=[100, 200], horizons=[60, 300])
    result = resolve(value)
    np.testing.assert_array_equal(result.loadings, np.broadcast_to(LOADINGS, (2, 2, 3, 2)))
    np.testing.assert_array_equal(result.lower, lower)
    for partial in (lower[0], lower[:, 0]):
        with pytest.raises(ValueError, match="full declared"):
            contract(lower=partial, decisions=[100, 200], horizons=[60, 300])


@pytest.mark.parametrize("kwargs", [
    dict(decisions=[100, 100]), dict(decisions=[200, 100]), dict(decisions=[np.nan]),
    dict(horizons=[0]), dict(horizons=[300, 60]), dict(horizons=[60, 60]),
    dict(horizons=[60.]), dict(horizons=[True]), dict(horizons=[2**63]),
])
def test_invalid_declared_decision_and_horizon_axes_fail(kwargs):
    with pytest.raises(ValueError):
        contract(**kwargs)


def test_availability_is_checked_for_every_requested_stage():
    value = contract(decisions=[100, 200], horizons=[60, 300],
                     available_at=[[90, 101], [190, 190]])
    with pytest.raises(ValueError, match="later than"):
        resolve(value)
    result = resolve(value, decisions=[200])
    np.testing.assert_array_equal(result.available_at, [[190, 190]])
    with pytest.raises(ValueError, match="all declared leading axes"):
        contract(decisions=[100, 200], horizons=[60, 300], available_at=[90, 190])


def test_availability_cannot_cross_timestamp_bases():
    value = contract(available_at=np.datetime64("2020-01-01"))
    with pytest.raises(ValueError, match="same timestamp basis"):
        resolve(value)
    with pytest.raises(ValueError, match="same timestamp basis"):
        contract(decisions=[100], available_at=np.datetime64("2020-01-01"))


def test_aware_datetimes_align_by_instant_and_check_future_information():
    utc = datetime(2020, 8, 3, 14, tzinfo=timezone.utc)
    eastern = timezone(timedelta(hours=-4))
    value = contract(decisions=[utc, utc+timedelta(minutes=5)], available_at=utc)
    result = resolve(value, decisions=[utc.astimezone(eastern)])
    assert result.available_at[0, 0] == utc
    future = contract(available_at=utc+timedelta(seconds=1))
    with pytest.raises(ValueError, match="later than"):
        resolve(future, decisions=[utc])
    with pytest.raises(ValueError, match="mix timezone-aware"):
        contract(decisions=[utc, utc.replace(tzinfo=None)])


def test_contract_and_resolution_are_copied_readonly():
    original = LOADINGS.copy()
    lower = np.array([-.1, -.2])
    value = contract(original, lower)
    original[:] = 99.
    lower[:] = 99.
    result = resolve(value)
    np.testing.assert_array_equal(result.loadings[0, 0], LOADINGS)
    np.testing.assert_array_equal(result.lower[0, 0], [-.1, -.2])
    for instance in (value, result):
        for field in fields(instance):
            array = getattr(instance, field.name)
            if array is not None:
                assert not array.flags.writeable


def book(values):
    return SimpleNamespace(values=np.asarray(values, dtype=float), decisions=[100],
                           assets=ASSETS, horizons=[60, 300])


def test_report_uses_native_weight_axis_and_reports_signed_budget_slacks():
    weights = book([[[.2, -.1], [-.3, .1], [.1, 0.]]])
    report = factor_exposure_report(weights, contract())
    # A' w for the two supplied books, independent of any solver or signal.
    expected = np.array([[[.25, .3], [-.1, -.1]]])
    np.testing.assert_allclose(report.exposures, expected, atol=1e-15)
    np.testing.assert_allclose(report.lower_slack, expected-[-.1, -.2], atol=1e-15)
    np.testing.assert_allclose(report.upper_slack, [.1, .3]-expected, atol=1e-15)
    np.testing.assert_allclose(report.violations, [[[.15, 0.], [0., 0.]]], atol=1e-15)
    np.testing.assert_array_equal(report.status, [["fail", "pass"]])
    for field in fields(report):
        assert not getattr(report, field.name).flags.writeable


def test_report_checks_actual_holdings_independently_of_feasible_targets():
    target = book([[[.05, .05], [-.05, -.05], [0., 0.]]])
    actual = book([[[.2, .2], [-.2, -.2], [0., 0.]]])
    np.testing.assert_array_equal(factor_exposure_report(target, contract()).status, "pass")
    np.testing.assert_array_equal(factor_exposure_report(actual, contract()).status, "fail")


def test_report_never_turns_unknown_held_loadings_into_zero_exposure():
    missing = LOADINGS.copy()
    missing[2] = np.nan
    value = contract(missing)
    zero = book([[[.05, .05], [-.05, -.05], [0., 0.]]])
    assert np.isfinite(factor_exposure_report(zero, value).exposures).all()
    zero.values[0, 2, 0] = 1e-30
    with pytest.raises(ValueError, match="potentially held"):
        factor_exposure_report(zero, value)


@pytest.mark.parametrize("tolerance", [-1., np.nan, np.inf, True])
def test_report_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        factor_exposure_report(book(np.zeros((1, 3, 2))), contract(), tolerance)


def test_report_does_not_mutate_weights_or_infer_factors():
    weights = book([[[.05, .05], [-.05, -.05], [0., 0.]]])
    before = weights.values.copy()
    factor_exposure_report(weights, contract())
    np.testing.assert_array_equal(weights.values, before)
    with pytest.raises(TypeError, match="FactorExposureConstraints"):
        factor_exposure_report(weights, None)


def test_contract_module_loads_in_isolation_without_package_imports():
    path = Path(__file__).parents[3]/"src/stable_finance/evaluation/portfolio/factors.py"
    spec = importlib.util.spec_from_file_location("_isolated_factor_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = module.factor_constraint_arrays([[1.]], -.1, .1, stages=1, n_assets=1)
        np.testing.assert_array_equal(result[0], [[[1.]]])
    finally:
        sys.modules.pop(spec.name, None)
