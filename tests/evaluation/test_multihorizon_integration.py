"""The additive horizon mixer uses native construction and scoring."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from stable_finance import RidgeProbe


def example_module():
    path = Path(__file__).resolve().parents[2]/"examples/multi_horizon.py"
    spec = importlib.util.spec_from_file_location("multi_horizon_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_pipeline_nets_before_costs_and_scores_one_step_returns(tmp_path, monkeypatch):
    def forbid_training(*args, **kwargs):
        raise AssertionError("external forecast example must not fit a predictor")

    monkeypatch.setattr(RidgeProbe, "fit", forbid_training)
    example = example_module()
    result = example.run_example(tmp_path)
    assert result["status"] == "passed"
    assert result["predictor_fits"] == 0
    assert result["source_horizons_seconds"] == [300, 900]
    assert result["evaluation_interval_seconds"] == 300
    np.testing.assert_allclose(result["first_component_targets"], [[.05, -.05], [-.05, .05]])
    np.testing.assert_array_equal(result["first_net_target"], [0., 0.])
    net = result["reports"]["equal_capital_mix"]
    control = result["reports"]["single_horizon_control"]
    assert net["mean_cost_bps"] == net["mean_turnover"] == 0.
    np.testing.assert_array_equal(net["gross_series"], 0.)
    np.testing.assert_array_equal(net["net_series"], 0.)
    assert control["mean_cost_bps"] > 0.
    np.testing.assert_allclose(control["gross_series"], [.00015, -.0001, .00025, .0001])
    # Exposure is recomputed from the combined net book, not averaged from
    # component concentration or beta diagnostics.
    exposure = result["target_exposures"]
    np.testing.assert_array_equal(exposure["equal_capital_mix"]["beta_exposure"], 0.)
    np.testing.assert_array_equal(exposure["equal_capital_mix"]["concentration_hhi"], 0.)
    assert all(row == ["pass"] for row in exposure["equal_capital_mix"]["constraint_status"])
    np.testing.assert_allclose(exposure["single_horizon_control"]["beta_exposure"], .02)
    np.testing.assert_allclose(exposure["components"]["concentration_hhi"], .5)
    # Four one-step rows, not a sum over the two overlapping forecast labels.
    assert len(control["gross_series"]) == 4
    saved = json.loads((tmp_path/"result.json").read_text())
    assert saved == result
    assert example.file_hash(tmp_path/"external_inputs.npz") == saved["input_sha256"]


def test_integration_example_fails_if_harness_charges_a_cancelled_book(tmp_path, monkeypatch):
    example = example_module()
    original = example.run_backtest

    def charge_cancelled_book(weights, *args, **kwargs):
        report = original(weights, *args, **kwargs)
        if np.all(weights.values == 0):
            from dataclasses import replace
            return replace(report, mean_cost_bps=1.)
        return report

    monkeypatch.setattr(example, "run_backtest", charge_cancelled_book)
    with pytest.raises(AssertionError):
        example.run_example(tmp_path)


def test_joint_first_target_enters_the_unchanged_native_backtest(tmp_path):
    from examples.joint_multi_horizon import run_example

    result = run_example(tmp_path)
    assert result["status"] == "passed"
    assert result["block_diagonal_decoupling"]
    assert result["only_first_stage_scored"]
    assert result["predictor_fits"] == 0
    np.testing.assert_allclose(result["current_long_weight"], .04285714285714286)
    np.testing.assert_allclose(result["after_later_forecast_change"], .06)
    assert set(result["reports"]) == {
        "native_single", "native_fixed_mix", "joint_full", "joint_no_cross"}
    exposure = result["reports"]["joint_full"]["target_exposure"]
    np.testing.assert_allclose(exposure["net"], 0., atol=1e-14)
    np.testing.assert_allclose(exposure["beta_exposure"], .4*.04285714285714286)
    assert len(exposure["horizons"]) == 1
