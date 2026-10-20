"""Every module that moved is still importable at its old path."""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from stable_finance._moved import MOVED

pytest.importorskip("streaming")      # some moved modules need the mds extra


@pytest.mark.parametrize("old", sorted(MOVED))
def test_an_old_path_resolves_to_where_it_moved(old):
    module = importlib.import_module(old)
    targets = MOVED[old]
    if len(targets) == 1:
        assert module is importlib.import_module(targets[0])
    else:
        for target in targets:
            moved = importlib.import_module(target)
            for name, value in vars(moved).items():
                if not name.startswith("__") and name in module.__dict__:
                    break
            else:
                raise AssertionError(f"{old} exposes nothing from {target}")


def test_old_names_import_from_old_paths():
    from stable_finance.backtest import run_backtest
    from stable_finance.data import ForwardReturns, MarketData
    from stable_finance.dataset.daystore import DayRecord
    from stable_finance.dataset.schema import MARKET_SCHEMA
    from stable_finance.dataset.write_mds import DENSE_MDS_COLUMNS
    from stable_finance.metrics import _validated_quotes, grouped_rank_ic, sharpe_ratio

    import stable_finance as sf

    assert run_backtest is sf.run_backtest and DayRecord.__module__.endswith("storage.daystore")
    assert ForwardReturns is sf.ForwardReturns and MarketData is sf.MarketData
    assert grouped_rank_ic is sf.grouped_rank_ic and sharpe_ratio is sf.sharpe_ratio
    assert DENSE_MDS_COLUMNS and MARKET_SCHEMA and _validated_quotes


def test_python_dash_m_on_an_old_path_runs_the_moved_module():
    out = subprocess.run([sys.executable, "-m", "stable_finance.dataset.daystore", "--help"],
                         capture_output=True, text=True, check=True)
    assert "day" in out.stdout.lower()
