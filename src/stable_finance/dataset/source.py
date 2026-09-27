"""Where a store's bytes come from: a local directory or a Hugging Face repo.

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
"""

from __future__ import annotations

import os
from pathlib import Path

from stable_finance.dataset.months import Month

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
