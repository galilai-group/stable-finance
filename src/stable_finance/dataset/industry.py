"""Per-month ticker -> Fama-French 49 industry map, for same-industry pairing.

One row per (month, ticker) with the firm's FF49 code, joined through the
point-in-time gvkey link (industry membership is monthly, so no mosaic scan
is needed). Built over EVERY month the mosaic holds so no sweep, eval or
ad-hoc month can fall outside it -- the map is built once.

Sources (paths are arguments; the files themselves live with the consumer):

  * Compustat North America Fundamentals Annual, columns
    gvkey/datadate/fyear/sich, screened INDL/STD/C. ``sich`` is the
    HISTORICAL (as-reported) SIC, so the label attached to a month is the one
    in force at the time rather than a current header code applied
    retroactively.
  * Ken French's FF49 definition file (Siccodes49.txt): 598 SIC ranges over
    49 industries, from the data library's Detail link.
  * A point-in-time ticker <-> gvkey link with valid_start_date /
    valid_end_date.

This replaced an earlier source (``lab_sample_industry_definitions``) that
started 2015-01 and covered 6,388 gvkeys -- it forced the same-industry
campaign onto 14 post-2015 months and left ~17% of the traded universe
(REITs, Canadian cross-listings, ADRs) unlabelled. On the 2015+ overlap the
two agree on 99.9% of FF49 codes wherever they report the same underlying
SIC; the residual is the sich-vs-header-SIC definitional difference.

Conventions:
  * SIC codes matching no FF49 range are left NULL rather than folded into
    49 ("Other"). Folding them in would inflate "Other" from ~0.5% to ~3% and
    turn a residual bucket into a large pseudo-industry, which for
    same-industry pairing is worse than no label: an unlabelled focal simply
    degrades to an unrestricted cross_stock draw.
  * A ticker held by two gvkeys inside one month (mid-month handoffs; 8-14
    tickers/month, ~0.03%) resolves to the gvkey covering more of the month.
  * Before a firm's first annual observation its earliest sich is carried
    back. Without this the 2007 months are near-empty (183 tickers in
    2007-01, since only firms with a fiscal year ending in early 2007 have
    reported yet) and only reach full density at 2007-12.

Output parquet columns: month (str YYYY-MM), ticker (str), ff49 (int16,
always >= 0 -- unmapped tickers are simply absent).

Usage (from a consumer repo that keeps the source files under data/):
    python -m stable_finance.dataset.industry \\
        --funda data/compustat_funda_sich.csv.gz \\
        --siccodes data/ff_raw/Siccodes49.txt \\
        --gvkey-link /data/lab/market-text-data/gvkey_ticker_history.parquet \\
        --out data/industry_map.parquet
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

FIRST_MONTH = "2007-01"
LAST_MONTH = "2024-12"


def load_ff49_ranges(siccodes: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse Siccodes49.txt -> (lo, hi, ff49) arrays of SIC ranges.

    Industry headers look like ``" 1 Agric  Agriculture"``; range lines are
    indented ``"    0100-0199 Agricultural production - crops"``.
    """
    lo, hi, ff = [], [], []
    current = None
    for line in Path(siccodes).read_text().splitlines():
        rng = re.match(r"\s+(\d{4})-(\d{4})", line)
        if rng:
            if current is None:
                raise ValueError(f"SIC range before any industry header: {line!r}")
            lo.append(int(rng.group(1)))
            hi.append(int(rng.group(2)))
            ff.append(current)
            continue
        head = re.match(r"\s*(\d{1,2})\s+\S+\s+\S", line)
        if head:
            current = int(head.group(1))
    return np.array(lo), np.array(hi), np.array(ff)


def sic_to_ff49(sic: np.ndarray, lo, hi, ff) -> np.ndarray:
    """Map 4-digit SIC codes to FF49; NaN where no range matches."""
    out = np.full(len(sic), np.nan)
    codes = np.asarray(sic).astype(int)
    hits = (codes[:, None] >= lo[None, :]) & (codes[:, None] <= hi[None, :])
    matched = hits.any(1)
    out[matched] = ff[hits.argmax(1)][matched]
    return out


def month_range(first: str, last: str) -> list[str]:
    import pandas as pd

    idx = pd.period_range(first, last, freq="M")
    return [f"{p.year}-{p.month:02d}" for p in idx]


def build_industry_map(funda_path, siccodes_path, gvkey_link_path, *,
                       first_month: str = FIRST_MONTH, last_month: str = LAST_MONTH,
                       log=print):
    """Return the (month, ticker, ff49) DataFrame."""
    import pandas as pd

    lo, hi, ff = load_ff49_ranges(siccodes_path)
    log(f"FF49 definition: {len(lo)} SIC ranges over {len(set(ff))} industries")

    funda = pd.read_csv(funda_path, dtype={"gvkey": str})
    funda = funda.dropna(subset=["sich"])
    funda["gvkey"] = funda.gvkey.astype(int)
    funda["datadate"] = pd.to_datetime(funda.datadate)
    funda["ff49"] = sic_to_ff49(funda.sich.values, lo, hi, ff)
    unmatched = int(funda.ff49.isna().sum())
    log(f"funda: {len(funda)} rows with sich, {funda.gvkey.nunique()} gvkeys; "
        f"{unmatched} ({unmatched / max(1, len(funda)):.2%}) match no FF49 range -> dropped")
    funda = funda.dropna(subset=["ff49"]).sort_values("datadate")

    gv = funda.gvkey.values
    ff49_v = funda.ff49.values.astype(np.int16)
    dates = funda.datadate.values
    # Earliest label per gvkey, used to back-fill months preceding a firm's
    # first fiscal-year observation.
    gv_first = dict(zip(funda.gvkey.values[::-1], ff49_v[::-1]))

    link = pd.read_parquet(gvkey_link_path)
    l_start = link.valid_start_date.values
    l_end = link.valid_end_date.values
    l_tick = link.ticker.values
    l_gv = link.gvkey.values

    months = month_range(first_month, last_month)
    log(f"building {len(months)} months: {months[0]} .. {months[-1]}")

    # Walk months forward, folding in each fiscal-year observation as its
    # datadate is passed. gv_latest holds the point-in-time label.
    gv_latest: dict[int, int] = {}
    ptr = 0
    frames = []
    n_backfilled = 0
    for ym in months:
        m_start = np.datetime64(f"{ym}-01")
        m_end = (pd.Timestamp(f"{ym}-01") + pd.DateOffset(months=1)
                 - pd.Timedelta(days=1)).to_datetime64()
        while ptr < len(dates) and dates[ptr] <= m_end:
            gv_latest[int(gv[ptr])] = int(ff49_v[ptr])
            ptr += 1
        alive = (l_start <= m_end) & (l_end >= m_start)
        df = pd.DataFrame({"ticker": l_tick[alive], "gvkey": l_gv[alive]})
        # Longest-overlap wins for mid-month ticker handoffs.
        df["overlap"] = np.minimum(l_end[alive], m_end) - np.maximum(l_start[alive], m_start)
        df = df.sort_values("overlap", ascending=False).drop_duplicates("ticker")
        lab = df.gvkey.map(gv_latest)
        back = lab.isna() & df.gvkey.isin(gv_first)
        n_backfilled += int(back.sum())
        lab = lab.fillna(df.gvkey.map(gv_first))
        df = df[lab.notna()]
        df["ff49"] = lab[lab.notna()].astype(int)
        df["month"] = ym
        frames.append(df[["month", "ticker", "ff49"]])

    out = pd.concat(frames, ignore_index=True)
    out["ff49"] = out.ff49.astype("int16")
    log(f"  back-filled rows (pre-first-observation): {n_backfilled}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--funda", default=os.environ.get("COMPUSTAT_FUNDA"),
                   required=not os.environ.get("COMPUSTAT_FUNDA"))
    p.add_argument("--siccodes", default=os.environ.get("FF49_SICCODES"),
                   required=not os.environ.get("FF49_SICCODES"))
    p.add_argument("--gvkey-link", default=os.environ.get("GVKEY_LINK"),
                   required=not os.environ.get("GVKEY_LINK"))
    p.add_argument("--out", default=os.environ.get("INDUSTRY_MAP"),
                   required=not os.environ.get("INDUSTRY_MAP"))
    p.add_argument("--first-month", default=FIRST_MONTH)
    p.add_argument("--last-month", default=LAST_MONTH)
    args = p.parse_args(argv)
    out = build_industry_map(args.funda, args.siccodes, args.gvkey_link,
                             first_month=args.first_month, last_month=args.last_month)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    per_month = out.groupby("month").size()
    print(f"wrote {len(out)} rows over {out.month.nunique()} months -> {path}\n"
          f"  tickers/month min/med/max = "
          f"{per_month.min()}/{int(per_month.median())}/{per_month.max()}\n"
          f"  file size = {path.stat().st_size / 1048576:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
