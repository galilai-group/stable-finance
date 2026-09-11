"""Rewrite a dense mosaic month with its samples ordered by DATE.

WHY THIS EXISTS. The writer shuffles: ``write_mds`` accumulates
``BUFFER_FLUSH_PARTITIONS`` partitions and shuffles before flushing, so a
79-sample shard holds 17-19 distinct dates and roughly four tickers per date.
That is the right layout for an i.i.d. sampler and the wrong one for a
CROSS-SECTIONAL objective.

The supervised head is trained by ``SupervisedModel._within_cell_loss``, which
ranks only among samples sharing a ``(date, anchor)`` -- one cross-section, the
unit the reported rank IC is computed over. Cross-cell pairs are excluded, not
downweighted, because comparing a stock at 13:00 Tuesday against one at 15:30
Thursday is a different question from the one the metric asks.

With the shuffled layout those pairs are scarce: a random batch of 256 rows
yields ~72 usable pairs, so most of the batch produces no gradient.

A CORRECTION IS RECORDED HERE ON PURPOSE, because an earlier version of this
file drew the wrong conclusion from the right measurement. It claimed the
head's rank-IC deficit against a ridge probe on the SAME frozen embeddings
(+0.0767 vs +0.1418) WAS the pair count, on the strength of an offline refit
at ~48 pairs reproducing the shipped head to within 0.0005. That agreement was
a coincidence, and it was persuasive enough to stop the investigation.

The actual cause was that ``_within_cell_loss`` had never executed.
``collate_bucketed`` dropped ``xs_cell``, so ``bucket.get("xs_cell")`` was
None on every step and every supervised head trained on the FLAT cross-cell
surrogate instead. Fixed in market-jepa 00fbb35; collating the key raised
pairs from ~48 to 71.5 on this same unchanged mosaic.

So date ordering is a pair-SUPPLY optimization and nothing more. It does not
explain the historical head-probe gap and must not be quoted as though it
does. Whether more pairs buys accuracy on top of the fix is an open question
this module exists to let us measure, not one it has answered.

WHAT ORDERING BUYS. A ticker-day record holds the whole session, so the anchor
is a free choice at crop time -- a cross-section never has to be materialized
or duplicated, it only needs enough same-date tickers resident in one batch.
Ordering by date makes a batch date-local by sequential read, which lifts the
expected pairs per 256-row batch from ~48 to ~907 even with every sample
drawing its own anchor, and to ~32,640 if the batch shares one.

IT IS A PURE RE-LAYOUT. Every record is copied through byte-identical --
same ticker, date, grid_start and features -- and the count is asserted. No
session is rebuilt, nothing is re-derived from ticks, and the source month is
never modified. That is what makes it safe to run against a mosaic other
results already depend on: point a consumer at the new directory or the old
one and it reads the same data.

THE COST, STATED. Date-local batches are correlated -- same regime, same news,
same market-wide shock. The ranking loss wants that correlation and the
backbone's own learning may not, so a sampler should draw SEVERAL dates per
batch rather than one.

WHICH IS WHY THE SHUFFLE KNOBS MATTER MORE THAN THE ORDERING. Measured with
MosaicML streaming on a resharded month: ``py1s`` is useless here (22.3
dates/batch, i.e. it undoes the layout), while ``py1br`` with
``shuffle_block_size=128`` gives 5.7 dates and 426.7 pairs per batch -- 5.8x
the supply at a date count that still mixes regimes. Ordering alone, with
shuffling off, reaches 1261 pairs at 1.3 dates/batch, which is the degenerate
end of the trade and not recommended.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from stable_finance.dataset.mds import open_shard, shard_list
from stable_finance.dataset.write_mds import DENSE_MDS_COLUMNS

# A month is ~11 GiB dense. Loading one is the simple, single-read path; a
# larger month is refused rather than silently thrashing.
DEFAULT_MAX_GIB = 48.0


def month_bytes(month_dir: Path) -> int:
    """On-disk size of a month, from its index rather than a directory walk."""
    idx = json.loads((Path(month_dir) / "index.json").read_text())
    total = 0
    for s in idx.get("shards", []):
        total += s["raw_data"]["bytes"]
        zd = s.get("zip_data")
        if zd:
            total += zd["bytes"]
    return total


def read_month(month_dir: Path, scratch_dir: Path | str) -> list[dict]:
    """Every record of a dense month, in stored order.

    ``scratch_dir`` MUST be on a volume with room, and one shard is expanded at
    a time. ``open_shard`` decompresses into scratch, so a month-long
    TemporaryDirectory accumulates the whole month UNCOMPRESSED -- ~11 GiB
    against a /tmp that is commonly a small shared volume (9.8 GiB on bll01,
    which is exactly how this failed the first time). densify.py already
    carries the same warning; this is the second place to learn it.
    """
    month_dir = Path(month_dir)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for meta in shard_list(month_dir):
        # Per-shard, so only one expansion is on disk at any moment.
        with tempfile.TemporaryDirectory(prefix="reshard_read_",
                                         dir=str(scratch_dir)) as td:
            reader = open_shard(month_dir, meta, td)
            for i in range(meta["samples"]):
                s = reader.get_item(i)
                out.append({
                    "ticker": str(s["ticker"]),
                    "date": str(s["date"]),
                    "grid_start": int(s["grid_start"]),
                    "features": np.asarray(s["features"], dtype=np.float32),
                })
    return out


def reshard_month(src: Path, dst: Path, *, compression: str = "zstd",
                  size_limit: int | None = None,
                  max_gib: float = DEFAULT_MAX_GIB,
                  verbose: bool = True) -> dict:
    """Copy ``src`` to ``dst`` with records sorted by ``(date, ticker)``.

    Returns a summary dict. ``dst`` must not already hold an index.
    """
    from streaming import MDSWriter

    src, dst = Path(src), Path(dst)
    if (dst / "index.json").exists():
        raise FileExistsError(f"{dst} already holds an MDS index; refusing")
    gib = month_bytes(src) / 2 ** 30
    if gib > max_gib:
        raise MemoryError(
            f"{src} is {gib:.1f} GiB, over max_gib={max_gib}. This path reads "
            "the month into memory; raise the cap only if the box has room.")

    records = read_month(src, scratch_dir=dst.parent)
    n_in = len(records)
    if not n_in:
        raise ValueError(f"{src} holds no samples")
    # (date, ticker) is a total order on a dense month: one record per pair.
    records.sort(key=lambda r: (r["date"], r["ticker"]))

    dst.mkdir(parents=True, exist_ok=True)
    kw = dict(out=str(dst), columns=DENSE_MDS_COLUMNS,
              compression=compression, exist_ok=True)
    if size_limit is not None:
        kw["size_limit"] = int(size_limit)
    with MDSWriter(**kw) as w:
        for r in records:
            w.write(r)

    dates = sorted({r["date"] for r in records})
    per_date = [sum(1 for r in records if r["date"] == d) for d in dates]
    summary = {
        "src": str(src), "dst": str(dst), "samples": n_in,
        "dates": len(dates),
        "tickers_per_date_median": int(np.median(per_date)),
        "gib_in": round(gib, 2),
    }
    if verbose:
        print(f"  {src.parent.name}/{src.name}: {n_in} samples, "
              f"{len(dates)} dates, median {summary['tickers_per_date_median']} "
              f"tickers/date -> {dst}")
    return summary


def verify(src: Path, dst: Path) -> dict:
    """Assert the copy is a permutation of the source, and report date locality.

    Content equality is checked on the (ticker, date) key set and on a sampled
    set of feature arrays -- a full array compare would re-read both months a
    second time for a property the writer cannot violate one record at a time.
    """
    dst = Path(dst)
    a = read_month(Path(src), scratch_dir=dst.parent)
    b = read_month(dst, scratch_dir=dst.parent)
    if len(a) != len(b):
        raise AssertionError(f"sample count {len(a)} -> {len(b)}")
    ka = sorted((r["date"], r["ticker"]) for r in a)
    kb = sorted((r["date"], r["ticker"]) for r in b)
    if ka != kb:
        raise AssertionError("(date, ticker) key sets differ")
    by_key = {(r["date"], r["ticker"]): r for r in a}
    rng = np.random.default_rng(0)
    for i in rng.choice(len(b), size=min(64, len(b)), replace=False):
        r = b[int(i)]
        o = by_key[(r["date"], r["ticker"])]
        if r["grid_start"] != o["grid_start"]:
            raise AssertionError(f"grid_start differs for {r['ticker']} {r['date']}")
        if not np.array_equal(r["features"], o["features"]):
            raise AssertionError(f"features differ for {r['ticker']} {r['date']}")

    def locality(recs, window=79):
        """Mean distinct dates in a sliding window the size of one shard."""
        d = [r["date"] for r in recs]
        return float(np.mean([len(set(d[i:i + window]))
                              for i in range(0, len(d) - window, window)]))
    return {"samples": len(a),
            "dates_per_shard_before": round(locality(a), 1),
            "dates_per_shard_after": round(locality(b), 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--src-root", required=True,
                   help="mosaic root holding <YYYY>/<MM>/ month dirs")
    p.add_argument("--dst-root", required=True)
    p.add_argument("--months", nargs="+", required=True, help="YYYY-MM ...")
    p.add_argument("--size-limit", type=int, default=None)
    p.add_argument("--max-gib", type=float, default=DEFAULT_MAX_GIB)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args(argv)

    src_root, dst_root = Path(a.src_root), Path(a.dst_root)
    for ym in a.months:
        y, m = ym.split("-")
        src, dst = src_root / y / m, dst_root / y / m
        if not (src / "index.json").is_file():
            raise SystemExit(f"no month at {src}")
        reshard_month(src, dst, size_limit=a.size_limit, max_gib=a.max_gib)
        if a.verify:
            print(f"    verify: {verify(src, dst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
