"""The appendix: every configuration in one table, and the IC each needs to pay.

Reviewers ask to see the full set of trials, not the winners; practitioners
ask what forecast quality a construction needs before it covers its costs.
This writes both from the finished runs:

* ``appendix.csv``: every row of every finished spec under ``results/``, with
  a ``spec`` column, the configuration's settings and its scores (one CSV, so
  nothing is selected away);
* ``lookup.csv``: for each construction (weighting and selection) and EFQ, the
  forecast IC at which net Sharpe crosses zero, interpolated in log IC between
  the oracle ICs that bracket it, from the dense break-even run
  (``e1x_breakeven``) when it exists, else the expected-IC run (E1x), else E1.
  The crossing is found on the seed-mean curve and, where the run has several
  seeds, on each seed's own curve, whose range is reported beside it. NaN
  where the construction never or always pays across the ICs run;
* ``lookup_noopen.csv`` and ``open_effect.csv``, when ``e1x_breakeven_noopen``
  has run: the same lookup without the opening decision, and the two side by
  side. The runs are not paired -- a ``skip_open`` run forms its books five
  minutes later and draws fresh oracle noise -- so a shift in a break-even is
  read against the two seed ranges (``ranges_overlap``), not on its own.

    uv run --extra experiments python -m experiments.appendix [--results DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from experiments.run import RESULTS


def appendix(results: Path):
    import pandas as pd

    frames = [pd.read_parquet(p).assign(spec=p.parent.name)
              for p in sorted(results.glob("*/rows.parquet"))]
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True, sort=False)
    first = [c for c in ("spec", "forecast", "eval_horizon", "weighting", "selection",
                         "risk_model", "cost_model", "efq", "rebalance") if c in table]
    return table[first + [c for c in table.columns if c not in first]]


def breakeven_ic(curve) -> float:
    """The IC at which net Sharpe crosses zero, linear in log IC; NaN without a crossing."""
    curve = curve.dropna().sort_index()
    ics, sharpe = curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float)
    for a, b, sa, sb in zip(ics[:-1], ics[1:], sharpe[:-1], sharpe[1:]):
        if sa < 0 <= sb:
            weight = -sa / (sb - sa)
            return float(np.exp(np.log(a) + weight * (np.log(b) - np.log(a))))
    return float("nan")


#: The runs ``lookup.csv`` is read from: the first of them that has finished.
LOOKUP_SOURCES = ("e1x_breakeven", "e1x_expected_ic", "e1_transfer_map")
#: ``e1x_breakeven`` without the opening decision, for ``lookup_noopen.csv``.
NOOPEN_SOURCE = "e1x_breakeven_noopen"


def lookup(results: Path, sources: tuple[str, ...] = LOOKUP_SOURCES):
    import pandas as pd

    for name in sources:
        path = results / name / "rows.parquet"
        if path.is_file():
            rows = pd.read_parquet(path)
            break
    else:
        return None
    rows = rows[(rows["f_noise"] == "random") & (rows["efq"] > 0)]
    out = []
    seeds = sorted(rows["f_seed"].unique()) if "f_seed" in rows else []
    for (weighting, selection, efq), part in rows.groupby(["weighting", "selection", "efq"]):
        curve = part.groupby("f_ic")["sharpe_net"].mean()
        record = {"source": name, "weighting": weighting, "selection": selection, "efq": efq,
                  "breakeven_ic": breakeven_ic(curve),
                  "net_sharpe_at_max_ic": float(curve.iloc[-1])}
        if len(seeds) > 1:
            each = np.array([breakeven_ic(part[part["f_seed"] == s].groupby("f_ic")["sharpe_net"].mean())
                             for s in seeds])
            crossed = each[np.isfinite(each)]
            record.update(n_seeds=len(seeds), seeds_crossing=len(crossed),
                          breakeven_ic_min=float(crossed.min()) if len(crossed) else np.nan,
                          breakeven_ic_max=float(crossed.max()) if len(crossed) else np.nan)
        if "sharpe_daily_net" in part:
            record["daily_net_sharpe_at_max_ic"] = float(
                part.groupby("f_ic")["sharpe_daily_net"].mean().iloc[-1])
        out.append(record)
    return pd.DataFrame(out)


def open_effect(with_open, without_open):
    """The break-evens with and without the opening decision, side by side.

    ``breakeven_ic_shift`` is the seed-mean break-even without the open less
    the one with it. The two runs draw their oracle noise independently and
    form their books at different times of day, so the shift carries the
    noise of both: ``ranges_overlap`` says whether the two seed ranges
    overlap, and a shift inside overlapping ranges is not attributed to the
    open. NaN there when either run has no seed range.
    """
    keys = ["weighting", "selection", "efq"]
    columns = [c for c in ("breakeven_ic", "breakeven_ic_min", "breakeven_ic_max",
                           "seeds_crossing", "n_seeds")
               if c in with_open and c in without_open]
    merged = with_open[keys + columns].merge(without_open[keys + columns], on=keys,
                                             suffixes=("", "_noopen"))
    merged["breakeven_ic_shift"] = merged["breakeven_ic_noopen"] - merged["breakeven_ic"]
    if "breakeven_ic_min" in columns:
        lo, hi = merged["breakeven_ic_min"], merged["breakeven_ic_max"]
        lo2, hi2 = merged["breakeven_ic_min_noopen"], merged["breakeven_ic_max_noopen"]
        known = lo.notna() & hi.notna() & lo2.notna() & hi2.notna()
        merged["ranges_overlap"] = ((lo <= hi2) & (lo2 <= hi)).where(known)
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--out", default=None, help="default <results>/appendix")
    args = parser.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "appendix"
    out.mkdir(parents=True, exist_ok=True)
    table = appendix(results)
    if table is None:
        print("no finished runs")
        return 1
    table.to_csv(out / "appendix.csv", index=False)
    print(f"appendix.csv: {len(table)} rows from {table['spec'].nunique()} specs")
    ics = lookup(results)
    if ics is not None and not ics.empty:
        ics.to_csv(out / "lookup.csv", index=False)
        print(ics.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    noopen = lookup(results, (NOOPEN_SOURCE,))
    if noopen is not None and not noopen.empty:
        noopen.to_csv(out / "lookup_noopen.csv", index=False)
        if ics is not None and not ics.empty and ics["source"].iloc[0] == LOOKUP_SOURCES[0]:
            effect = open_effect(ics, noopen)
            effect.to_csv(out / "open_effect.csv", index=False)
            print(effect.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
