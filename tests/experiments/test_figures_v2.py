"""experiments/figures_v2.py: the MCS figure counts distinct configurations."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from experiments import figures_v2


def _rows():
    # Signal: a and its copy b in the set, c out; HRP: d in, e not tested.
    return pd.DataFrame({
        "row_id": list("abcde"),
        "weighting": ["signal", "signal", "signal", "hrp", "hrp"],
        "duplicate_of": [None, "a", None, None, None],
        "in_mcs_pnl": [True, True, False, True, None],
        "in_mcs_sharpe": [False, False, True, True, None],
    })


def test_mcs_shares_count_each_distinct_configuration_once():
    shares = figures_v2.mcs_shares(_rows())
    assert shares.at["signal", "pnl"] == 0.5          # a in, c out; the copy b ignored
    assert shares.at["signal", "sharpe"] == 0.5
    assert shares.at["hrp", "pnl"] == 1.0             # e's group had no set
    assert shares.at["hrp", "sharpe"] == 1.0


def test_mcs_figure_skips_outputs_without_duplicate_flags(tmp_path):
    pytest.importorskip("matplotlib")
    old = tmp_path / "old_run"
    old.mkdir()
    pd.DataFrame({"row_id": ["a"], "weighting": ["signal"], "in_mcs": [True]}).to_parquet(
        old / "inference.parquet")
    assert figures_v2.mcs(tmp_path, tmp_path / "figures", "light") == []
    new = tmp_path / "new_run"
    new.mkdir()
    _rows().to_parquet(new / "inference.parquet")
    (new / "inference.json").write_text('{"settings": {"pnl": "net"}}')
    made = figures_v2.mcs(tmp_path, tmp_path / "figures", "light")
    assert [p.name for p in made] == ["mcs-light.png"]
    assert np.all([p.is_file() for p in made])
