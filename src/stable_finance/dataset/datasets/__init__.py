"""Concrete datasets, one package each.

A dataset package holds only what is specific to that dataset: its schema
and calendar (its spec), the reader that ingests its raw files into a store,
and any reference data that comes with it. Everything else -- storage,
loading, labels -- is generic and lives beside this package. A new dataset
is a new package here; datasets without a raw-file reader of their own come in
through :mod:`stable_finance.dataset.ingest` instead.

``market1t``  the Polygon-derived one-second US-equity dataset, Market-1T
"""
