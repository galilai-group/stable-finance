"""Market-1T: the Polygon-derived one-second US-equity dataset.

Everything specific to it lives here: its column schema
(:data:`MARKET_SCHEMA`), its NYSE calendar with the holiday file and the
one-second end-of-bar offset (:class:`Market1TCalendar`), Polygon parquet
ingestion into the sparse mosaic (:mod:`.ingest`, :mod:`.shard_months`), and
the Fama-French industry map (:mod:`.industry`).

The generic pipeline -- densify, the day store, target tables, panels and
cells -- reads Market-1T stores through the same schema and calendar
interfaces as any other dataset, and defaults to this one when a store or a
caller names nothing else.
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
