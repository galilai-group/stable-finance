import numpy as np
import pytest

from stable_finance.evaluation.scoring.exposure import portfolio_exposure


def test_factor_exposure_reads_signed_weights_not_gross_weights():
    weights = np.array([[0.6, -0.4, 0.0], [0.2, 0.3, -0.5]])
    loadings = np.array([[1.5, 1.0], [0.5, 1.0], [2.0, 0.0]])
    report = portfolio_exposure(weights, loadings, factor_names=("beta", "industry_1"))
    assert report.factor_names == ("beta", "industry_1")
    np.testing.assert_allclose(report.factor_exposure, weights @ loadings)
    np.testing.assert_allclose(report.concentration, [0.52, 0.38])


def test_concentration_is_invariant_to_gross_and_handles_cash():
    weights = [[0.5, -0.5], [2.0, -2.0], [0.0, 0.0], [1.0, 0.0]]
    report = portfolio_exposure(weights)
    np.testing.assert_allclose(report.concentration, [0.5, 0.5, 0.0, 1.0])
    assert report.factor_exposure.shape == (4, 0)
    assert report.factor_names == ()


def test_loadings_are_observed_at_each_decision():
    weights = np.array([[0.5, -0.5], [0.5, -0.5]])
    loadings = np.array([[[1.0], [2.0]], [[2.0], [1.0]]])
    report = portfolio_exposure(weights, loadings, factor_names=("beta",))
    np.testing.assert_allclose(report.factor_exposure[:, 0], [-0.5, 0.5])


def test_unknown_held_loadings_are_not_reported_as_neutral():
    weights = [[0.5, -0.5], [0.5, 0.0], [0.0, 0.0]]
    loadings = [[1.0, 2.0], [np.nan, 1.0]]
    report = portfolio_exposure(weights, loadings)
    np.testing.assert_allclose(report.factor_exposure,
                               [[np.nan, 0.5], [0.5, 1.0], [0.0, 0.0]])


def test_nonfinite_weights_remain_unknown_and_roundoff_is_cash():
    report = portfolio_exposure([[np.nan, 0.0], [1e-15, -1e-15]], [[1.0], [2.0]])
    np.testing.assert_allclose(report.concentration, [np.nan, 0.0])
    np.testing.assert_allclose(report.factor_exposure[:, 0], [np.nan, 0.0])


def test_inputs_match_the_decision_asset_and_factor_axes():
    with pytest.raises(ValueError, match="weights must"):
        portfolio_exposure([0.5, -0.5])
    with pytest.raises(ValueError, match="factor_loadings must"):
        portfolio_exposure([[0.5, -0.5]], np.ones((3, 1)))
    with pytest.raises(ValueError, match="factor_names must"):
        portfolio_exposure([[0.5, -0.5]], np.ones((2, 1)), factor_names=("a", "b"))
