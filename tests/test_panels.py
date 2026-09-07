from __future__ import annotations

import numpy as np

from stable_finance.dataset import (
    CrossSectionalTargetBundle,
    MARKET_SCHEMA,
    MarketSession,
    ViewSpec,
    build_session_panel,
    choose_cell_aggregation,
)
from stable_finance.dataset.calendar import standard_open_est


class _IdentityStats:
    def transform(self, raw, *_args):
        values = np.asarray(raw)
        return CrossSectionalTargetBundle(values, values, values, values)


def _session():
    n = 300
    values = np.zeros((n, len(MARKET_SCHEMA.columns)), dtype=np.float64)
    values[:, 0] = np.linspace(100, 101, n)
    values[:, 1] = values[:, 0]
    values[:, 2] = values[:, 0] + 0.1
    values[:, 3] = values[:, 0] - 0.1
    values[:, 4] = values[:, 0] + 0.02
    values[:, 5:7] = 100
    values[:, 7:] = 10
    start = standard_open_est("2023-01-03")
    return MarketSession(
        "A", "2023-01-03", np.arange(start, start + n), values,
    )


def test_panel_keeps_metadata_separate_from_view_columns():
    rows = build_session_panel(
        _session(), [150], _IdentityStats(),
        view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)),
        target_types=("return",),
        horizons=(10,),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.view.shape == (4, len(MARKET_SCHEMA.columns))
    assert row.metadata.start_seconds == 147
    assert row.metadata.end_seconds == 151
    assert row.metadata.aggregation_seconds == 1


def test_cell_aggregation_is_reproducible_and_feasible():
    first = choose_cell_aggregation("2023-01-03", 20, 20, 4, choices=(3, 4, 8))
    second = choose_cell_aggregation("2023-01-03", 20, 20, 4, choices=(3, 4, 8))
    assert first == second
    assert first in (3, 4)
