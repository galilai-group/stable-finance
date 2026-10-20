"""Where a store's months are: month identifiers, period directories, local or Hub roots.

MONTHS. Calendar-month identifiers used by fitting and evaluation APIs.

PERIODS. Discovery of the dataset's month- and quarter-partitioned storage.

SOURCES. Where a store's bytes come from: a local directory or a Hugging Face repo.

Every store is laid out as ``<root>/YYYY/MM/...``, so a root is all a reader
needs. A root is either a local path, returned untouched, or a Hub spec::

    hf://datasets/<org>/<repo>[@<revision>]/<subdir>

which is materialized under a local cache and returned as an ordinary path.
Only the months a caller asks for are downloaded, so pointing a one-month eval
at a multi-month repo fetches one month.

The cache is ``$STABLE_FINANCE_CACHE`` (default ``~/.cache/stable-finance``),
one real directory per repo rather than the Hub's symlinked blob cache: the
day store decompresses ``features.npy`` in place beside its ``.zst``, and that
needs a directory the reader owns. Downloads are resumable and a second call
for the same months is a metadata check.

Keep ``<subdir>`` named like the local store (``1Hz_mosaic_mnth``,
``1Hz_daystore``, ...): :func:`infer_period_frequency` reads the frequency off
the root's name, and the resolved path ends in ``<subdir>``.

MONTH DIRECTORIES. Months of a store that hold samples.
"""

from __future__ import annotations

import calendar
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path



# ── months ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class Month:
    """A validated calendar month with inclusive date bounds."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if self.year < 1 or not 1 <= self.month <= 12:
            raise ValueError(f"invalid calendar month: {self.year}-{self.month}")

    @classmethod
    def parse(cls, value: "Month | str") -> "Month":
        if isinstance(value, cls):
            return value
        try:
            year, month = (int(part) for part in str(value)[:7].split("-"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"month must be YYYY-MM, got {value!r}") from error
        return cls(year, month)

    @property
    def first_day(self) -> dt.date:
        return dt.date(self.year, self.month, 1)

    @property
    def last_day(self) -> dt.date:
        return dt.date(
            self.year, self.month, calendar.monthrange(self.year, self.month)[1]
        )

    def offset(self, count: int) -> "Month":
        ordinal = self.year * 12 + self.month - 1 + count
        year, zero_based_month = divmod(ordinal, 12)
        return Month(year, zero_based_month + 1)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def next_month(value: Month | str) -> str:
    """Return the calendar month after ``value`` as ``YYYY-MM``."""
    return str(Month.parse(value).offset(1))


# ── periods ──────────────────────────────────────────────────────────────────

def infer_period_frequency(dataset_root: str | Path) -> str:
    return "qtr" if "qtr" in Path(dataset_root).name else "mnth"


def validate_period_alignment(start: dt.date, end: dt.date, frequency: str) -> None:
    if start.day != 1:
        raise ValueError(f"date_start ({start}) must be the 1st of a month.")
    last = calendar.monthrange(end.year, end.month)[1]
    if end.day != last:
        raise ValueError(
            f"date_end ({end}) must be the last day of a month "
            f"(expected {end.year}-{end.month:02d}-{last:02d})."
        )
    quarter_starts, quarter_ends = {1, 4, 7, 10}, {3, 6, 9, 12}
    if frequency == "qtr" and start.month not in quarter_starts:
        nearest = min(quarter_starts, key=lambda month: abs(month - start.month))
        raise ValueError(
            f"date_start month ({start.month:02d}) is not quarter-aligned. "
            f"Nearest valid start month: {nearest:02d}."
        )
    if frequency == "qtr" and end.month not in quarter_ends:
        nearest = min(quarter_ends, key=lambda month: abs(month - end.month))
        raise ValueError(
            f"date_end month ({end.month:02d}) is not quarter-aligned. "
            f"Nearest valid end month: {nearest:02d}."
        )


def period_directories(
    dataset_root: str | Path,
    date_start: str | dt.date,
    date_end: str | dt.date,
    *,
    allow_unaligned_dates: bool = False,
) -> list[Path]:
    """Return existing dataset period directories in chronological order.

    ``dataset_root`` may be a Hub spec (see :mod:`stable_finance.dataset.storage.locations`);
    the months in range are then downloaded first and local paths returned.
    """
    start = dt.date.fromisoformat(str(date_start))
    end = dt.date.fromisoformat(str(date_end))
    frequency = infer_period_frequency(dataset_root)
    if not allow_unaligned_dates:
        validate_period_alignment(start, end, frequency)
    root = resolve_root(dataset_root, start.isoformat(), end.isoformat())
    found = []
    cursor, last = Month(start.year, start.month), Month(end.year, end.month)
    while cursor <= last:
        if frequency == "mnth" or cursor.month in {3, 6, 9, 12}:
            candidate = root / str(cursor.year) / f"{cursor.month:02d}"
            if candidate.is_dir():
                found.append(candidate)
        cursor = cursor.offset(1)
    if not found:
        raise ValueError(
            f"No dataset period directories found in {dataset_root} for date "
            f"range [{date_start}, {date_end}]. Expected "
            f"{dataset_root}/{{year}}/{{MM}}/ subdirectories."
        )
    return found


# ── sources ──────────────────────────────────────────────────────────────────

HF_SCHEME = "hf://"


def is_hub(root: str | Path) -> bool:
    return str(root).startswith(HF_SCHEME)


def parse_hub(spec: str) -> tuple[str, str, str | None, str]:
    """``hf://<type>s/<org>/<repo>[@rev]/<subdir>`` -> (repo_type, repo_id, revision, subdir)."""
    parts = spec[len(HF_SCHEME):].strip("/").split("/")
    if len(parts) < 3 or parts[0] not in ("datasets", "models"):
        raise ValueError(
            f"{spec!r}: expected hf://datasets/<org>/<repo>[@revision]/<subdir>")
    repo_type = parts[0][:-1]
    repo, _, revision = parts[2].partition("@")
    return repo_type, f"{parts[1]}/{repo}", revision or None, "/".join(parts[3:])


def cache_root() -> Path:
    return Path(os.environ.get("STABLE_FINANCE_CACHE",
                               Path.home() / ".cache" / "stable-finance"))


def resolve_root(root: str | Path, date_start: str | None = None,
                 date_end: str | None = None) -> Path:
    """A local directory holding ``root``'s months in ``[date_start, date_end]``.

    A local ``root`` is returned as is -- nothing is checked or copied, so
    existing configs behave exactly as before. A Hub spec downloads the months
    in range (every month when no range is given) and returns the local copy.
    """
    if not is_hub(root):
        return Path(root)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Reading a store from the Hub requires `uv sync --extra hf`") from error

    repo_type, repo_id, revision, subdir = parse_hub(str(root))
    prefix = f"{subdir}/" if subdir else ""
    if date_start is not None and date_end is not None:
        cursor, last = Month.parse(str(date_start)[:7]), Month.parse(str(date_end)[:7])
        patterns = []
        while cursor <= last:
            patterns.append(f"{prefix}{cursor.year:04d}/{cursor.month:02d}/*")
            cursor = cursor.offset(1)
    else:
        patterns = [f"{prefix}*"]

    local = cache_root() / repo_id.replace("/", "--")
    if revision:
        local = local.with_name(f"{local.name}@{revision}")
    snapshot_download(repo_id, repo_type=repo_type, revision=revision,
                      allow_patterns=patterns, local_dir=local)
    return local / subdir if subdir else local


# ── month directories ────────────────────────────────────────────────────────

def month_dirs(root: Path, start: str, end: str) -> list[tuple[str, Path]]:
    """Months in range that actually hold samples.

    A SAMPLE-LESS MONTH IS SKIPPED, NOT FAILED. Some month directories carry a
    valid index.json declaring zero shards -- 2007-12 and 2025-01..03 in the
    current dataset, the boundaries of what was ingested. LocalDataset raises
    IndexError deep inside its Spanner on those rather than reporting an empty
    dataset, so they have to be filtered here or a fan-out over months dies on
    its edges.
    """
    import json

    out = []
    for year_dir in sorted(root.glob("[0-9][0-9][0-9][0-9]")):
        for month_dir in sorted(year_dir.glob("[0-9][0-9]")):
            ym = f"{year_dir.name}-{month_dir.name}"
            if not (start <= ym <= end):
                continue
            index = month_dir / "index.json"
            if not index.is_file():
                continue
            shards = json.loads(index.read_text()).get("shards") or []
            if not shards or not sum(s.get("samples", 0) for s in shards):
                print(f"{ym}: no samples, skipping", flush=True)
                continue
            out.append((ym, month_dir))
    return out
