"""The documented native entry point runs without a preserved checkout."""

import json

import numpy as np

from examples.native_return_risk_backtest import run_example


def test_documented_single_joint_example_uses_native_state_and_serializable_reports(tmp_path):
    result = run_example(tmp_path)
    assert result["status"] == "passed"
    assert result["predictor_fits"] == result["dataset_downloads"] == 0
    assert result["preserved_checkout_required"] is False
    saved = json.loads((tmp_path/"result.json").read_text())
    for mode in ("single", "joint"):
        row = saved["reports"][mode]
        assert row["pending_feedback_verified"] and row["missing_arrival_fill_verified"]
        assert row["physical_nav_reconciled"] and row["terminal_liquidation_verified"]
        assert row["report"]["input_mode"] == "return_risk_physical"
        assert row["report"]["objective_gross_limit"] == 1.
        assert row["report"]["information_coefficient"] is None
        assert "objective_trace" not in row["report"]
    with np.load(tmp_path/"native_traces.npz") as trace:
        assert trace["single_reference_plans"].shape == (4, 1, 4)
        assert trace["joint_reference_plans"].shape == (4, 2, 4)
        np.testing.assert_array_equal(trace["joint_reference_plans"][-1, 1], 0.)
        assert np.any(trace["single_pending"][1] != 0)
        assert not np.allclose(trace["single_planned"][0], trace["joint_planned"][0])
