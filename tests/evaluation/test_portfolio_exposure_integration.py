"""Completed native and multi-horizon targets share one exposure diagnostic."""

import json

import numpy as np

from stable_finance import (
    PortfolioExposureLimits, portfolio_exposure_report,
    combine_horizon_targets, mean_variance_target,
)
from examples.portfolio_exposure import run_example, supplied_inputs


def test_neutral_alpha_can_produce_exposed_native_weights_and_unchanged_diagnostics(tmp_path):
    result = run_example(tmp_path)
    assert result["status"] == "passed"
    assert result["alpha_is_dollar_beta_and_industry_neutral"]
    assert result["weights_unchanged"]
    assert result["predictor_fits"] == result["factor_estimates"] == 0
    row = result["first_rows"]["single_horizon_target"]
    np.testing.assert_allclose(row["net"], .0375)
    np.testing.assert_allclose(row["beta"], -.0125)
    np.testing.assert_allclose(row["hhi"], 17/45)
    np.testing.assert_allclose(row["effective_names"], 45/17)
    assert row["constraint_status"] == "fail"
    assert result["first_rows"]["fixed_multi_horizon_target"]["hhi"] == row["hhi"]
    np.testing.assert_allclose(result["first_rows"]["joint_current_target"]["net"], 0., atol=1e-14)
    assert abs(result["first_rows"]["joint_current_target"]["beta"]) > .01
    saved = json.loads((tmp_path/"result.json").read_text())
    assert saved == result
    assert set(saved["reports"]) == {"per_horizon_targets", "single_horizon_target",
        "fixed_multi_horizon_target", "joint_current_target"}
    assert "never repairs" in (tmp_path/"exposure_report.md").read_text()


def test_missing_held_beta_is_unknown_in_both_single_and_mixed_targets():
    forecast, covariance, beta, industry = supplied_inputs()
    components = mean_variance_target(forecast, covariance, risk_aversion=2.)
    beta[0] = np.nan
    for fractions in ([1., 0.], [.5, .5]):
        target = combine_horizon_targets(components, capital_weights=fractions,
                                         target_horizon_seconds=300)
        before = target.values.copy()
        report = portfolio_exposure_report(target, beta=beta, industry=industry,
            limits=PortfolioExposureLimits(max_abs_beta=.1))
        assert np.isnan(report.beta_exposure).all()
        assert np.all(report.beta_coverage < 1.)
        assert np.all(report.constraint_status == "unknown")
        np.testing.assert_array_equal(target.values, before)
        # Missing exposure is not serialized as a healthy zero or JSON NaN.
        encoded = json.dumps(report.as_dict(), allow_nan=False)
        assert "null" in encoded
