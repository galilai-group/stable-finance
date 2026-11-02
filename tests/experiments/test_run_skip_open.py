"""The runner's skip_open option: which decisions it drops, and how it keys tasks."""
from __future__ import annotations

import numpy as np

from experiments import run


def test_the_first_decision_of_each_day_is_dropped():
    labels = np.array(["2020-08-03|0000", "2020-08-03|0300", "2020-08-03|0600",
                       "2020-08-04|0300", "2020-08-04|0600"])
    kept = run._after_open(labels)
    # A day whose opening decision is missing loses its earliest one instead.
    assert kept.tolist() == ["2020-08-03|0300", "2020-08-03|0600", "2020-08-04|0600"]


def test_skip_open_enters_the_task_key_only_when_set():
    spec = {"eval_month": "2020-08", "history_month": "2020-07", "horizons": [900],
            "configs": {"base": {}, "grid": {}}}
    forecast = {"kind": "oracle", "ic": 0.03, "noise": "random", "seed": 0}
    plain = run._task_key(spec, forecast, 900)
    assert run._task_key({**spec, "skip_open": False}, forecast, 900) == plain
    assert run._task_key({**spec, "skip_open": True}, forecast, 900) != plain
