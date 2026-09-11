from __future__ import annotations

import numpy as np

from stable_finance.dataset import (
    CrossSectionalTargetBundle,
    FeatureSchema,
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


def test_panel_carries_the_natural_unit_quote_at_the_decision():
    """The executable price cannot be recovered from the view.

    ``prepare_view`` standardizes each observation against its own window, so
    the bid and ask inside ``row.view`` are in units that differ per row. The
    execution stage charges the spread that was actually quoted, so the quote
    rides alongside the view in natural units instead.
    """
    session = _session()
    rows = build_session_panel(
        session, [150], _IdentityStats(),
        view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)),
        target_types=("return",),
        horizons=(10,),
    )

    bid = session.features[150, MARKET_SCHEMA.index("bid_price")]
    ask = session.features[150, MARKET_SCHEMA.index("ask_price")]
    assert rows[0].quote.tolist() == [bid, ask]
    # The DECISION row, not the window's first or last bar: bid_price rises
    # monotonically here, so reading the wrong end of the view would show up.
    assert rows[0].quote[0] != session.features[147, MARKET_SCHEMA.index("bid_price")]


def test_quote_columns_follow_the_session_schema_rather_than_a_constant():
    """A backend that reorders its columns must move the quote with it.

    Indices read off a module-level constant would silently return two other
    features here, and the execution stage would price every fill against
    whatever happened to sit in those positions.
    """
    reversed_columns = tuple(reversed(MARKET_SCHEMA.columns))
    schema = FeatureSchema(
        columns=reversed_columns,
        forward_fill=MARKET_SCHEMA.forward_fill,
        zero_fill=MARKET_SCHEMA.zero_fill,
        aggregation=tuple(
            (name, MARKET_SCHEMA.aggregation_rules[name]) for name in reversed_columns
        ),
        normalization=MARKET_SCHEMA.normalization,
    )
    original = _session()
    session = MarketSession(
        original.ticker, original.date, original.timestamps,
        original.features[:, ::-1].copy(), schema,
    )
    rows = build_session_panel(
        session, [150], _IdentityStats(),
        view_spec=ViewSpec(sequence_length=4, aggregation_seconds=(1, 1)),
        target_types=("return",),
        horizons=(10,),
    )

    assert rows[0].quote.tolist() == [
        session.features[150, schema.index("bid_price")],
        session.features[150, schema.index("ask_price")],
    ]
    # And that is the same market the default-ordered session reported.
    assert rows[0].quote.tolist() == [
        original.features[150, MARKET_SCHEMA.index("bid_price")],
        original.features[150, MARKET_SCHEMA.index("ask_price")],
    ]
