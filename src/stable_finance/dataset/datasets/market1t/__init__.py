"""Market-1T: the Polygon-derived one-second US-equity dataset.

One processing pipeline, one module per step:

``spec``      its nine-column schema, NYSE calendar and clock, store locations
``ingest``    raw Polygon parquet to the sparse MDS mosaic; the industry map
``mosaic``    the MDS mosaic: reader, densify (sparse to dense), reshard
``views``     sessions to model inputs: grid rebuild, views, panels, cells
``daystore``  the day-major store, its labels, and evaluation inputs read from it
``tables``    monthly cross-section tables of the targets

Labels are defined by the generic :class:`~stable_finance.dataset.objective.Objective`
and computed by :mod:`stable_finance.dataset.outcomes`; everything else here
is Market-1T's own.
"""

from stable_finance.dataset.datasets.market1t.spec import (
    EASTERN_TIME,
    MARKET_SCHEMA,
    Market1TCalendar,
    MarketSchedule,
    market1t_timeline,
    standard_open_est,
    timeline_bounds_est,
)

__all__ = [
    "EASTERN_TIME",
    "MARKET_SCHEMA",
    "Market1TCalendar",
    "MarketSchedule",
    "market1t_timeline",
    "standard_open_est",
    "timeline_bounds_est",
]
