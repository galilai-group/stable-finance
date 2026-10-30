"""The public example exercises a complete supplied-factor pipeline."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def test_generic_factor_public_example(tmp_path):
    path = Path(__file__).resolve().parents[2]/"examples/generic_factor_constraints.py"
    spec = importlib.util.spec_from_file_location("generic_factor_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_example(tmp_path)
    assert result["status"] == "passed"
    assert result["factor_names"] == ["proxy_beta", "synthetic_group", "style"]
    assert result["tracking_constraints_passed"]
    assert result["cross_interval_coupling_verified"]
    saved = json.loads((tmp_path/"result.json").read_text())
    assert saved == result
    for label in ("single", "joint"):
        assert saved["reports"][label]["targets_passed"]
        assert saved["reports"][label]["all_reference_stages_passed"]
    with np.load(tmp_path/"traces.npz", allow_pickle=False) as arrays:
        assert arrays["joint_plans"].shape == (3, 2, 6)
        assert arrays["single_plans"].shape == (3, 1, 6)
        assert np.isfinite(arrays["joint_nav"]).all()
