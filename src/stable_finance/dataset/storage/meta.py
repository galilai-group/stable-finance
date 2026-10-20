"""``dataset.json``: what a store holds, written beside the data.

A store written with this file can be read without knowing which dataset it
came from: its schema (columns, rules, roles) and its timeline (calendar, bar
length, decision grid) are recorded rather than assumed. A store without one is
Market-1T at 1 Hz, which is every store written before the file existed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stable_finance.dataset.spec.schema import FeatureSchema
from stable_finance.dataset.spec.timeline import SessionCalendar, Timeline, as_timeline


STORE_META = "dataset.json"
META_VERSION = 1

# The two ticker-day record layouts of the MDS mosaic (storage.mosaic). They
# live here, beside the description of what a store holds, so modules that
# name a layout do not need MosaicML Streaming installed to import.
MDS_COLUMNS = {
    "ticker": "str",
    "date": "str",
    "ts_interval": "ndarray:int32",
    "features": "ndarray:float32",
}

# THE DENSE LAYOUT. ``features`` is the reconstructed session --
# (session_rows, n_columns), forward- and zero-filled -- and ``grid_start``
# is the epoch second of its first row, so the timestamps are an arange and
# are not stored. SessionPreprocessor.transform adopts such a record instead
# of rebuilding the grid; see its ``_from_dense`` for why the format exists
# and what it measures.
DENSE_MDS_COLUMNS = {
    "ticker": "str",
    "date": "str",
    "grid_start": "int32",
    "features": "ndarray:float32",
}




@dataclass(frozen=True)
class StoreMeta:
    """A store's schema and the timeline its rows are stamped on."""

    schema: FeatureSchema
    clock: dict

    def timeline(self, schedule=None) -> Timeline:
        """The store's timeline.

        A Market-1T calendar is recorded by kind only, because its holiday file
        is a path on the writer's machine; pass the reader's ``schedule``
        (a holiday file, its path, or a whole timeline) to keep short days.
        """
        if isinstance(schedule, Timeline):
            return schedule
        if isinstance(schedule, SessionCalendar) and not self.is_market1t:
            return Timeline.from_dict(self.clock, calendar=schedule)
        if self.is_market1t:
            from stable_finance.dataset.datasets.market1t.spec import market1t_timeline

            recorded = Timeline.from_dict(self.clock)
            return market1t_timeline(schedule, post_close_seconds=recorded.post_close_seconds)
        return Timeline.from_dict(self.clock)

    @property
    def is_market1t(self) -> bool:
        return (self.clock.get("calendar") or {"kind": "market1t"}).get("kind") == "market1t"

    @property
    def bar_seconds(self) -> int:
        return int(self.clock.get("bar_seconds", 1))

    def to_dict(self) -> dict:
        # Stored rows are always stamped at the END of their bar (see
        # dataset.spec.timeline); recorded so a reader never has to assume it.
        return {"version": META_VERSION, "schema": self.schema.to_dict(),
                "timeline": self.clock, "stamp": "end"}


def needs_holiday_file(month_dirs) -> list[str]:
    """Months among ``(label, dir)`` pairs stored on the Market-1T calendar.

    Those months take their short days from a holiday file the store does not
    carry, so a command converting them must be given one.
    """
    return [label for label, directory in month_dirs if read_store_meta(directory).is_market1t]


def require_holiday_file(month_dirs, holiday_csv) -> None:
    """Exit with a message when a Market-1T month is converted without its holidays."""
    if holiday_csv:
        return
    missing = needs_holiday_file(month_dirs)
    if missing:
        raise SystemExit(
            f"{missing[0]} is a Market-1T month: pass --holiday-csv so short days "
            "get their early close")


def write_store_meta(directory: str | Path, schema: FeatureSchema, *, timeline) -> Path:
    path = Path(directory) / STORE_META
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = StoreMeta(schema, as_timeline(timeline).to_dict())
    path.write_text(json.dumps(meta.to_dict(), indent=1))
    return path


def read_store_meta(directory: str | Path) -> StoreMeta:
    """The store's description; Market-1T at 1 Hz when it carries none."""
    path = Path(directory) / STORE_META
    if not path.is_file():
        from stable_finance.dataset.datasets.market1t.spec import MARKET_SCHEMA

        return StoreMeta(MARKET_SCHEMA, as_timeline(None).to_dict())
    value = json.loads(path.read_text())
    return StoreMeta(FeatureSchema.from_dict(value["schema"]), value["timeline"])
