"""The appendix: every configuration in one table, and the IC each needs to pay.

Reviewers ask to see the full set of trials, not the winners; practitioners
ask what forecast quality a construction needs before it covers its costs.
This writes both from the finished runs:

* ``appendix.csv``: every row of every finished spec under ``results/``, with
  a ``spec`` column, the configuration's settings and its scores (one CSV, so
  nothing is selected away);
* ``lookup.csv``: for each construction (weighting and selection) and EFQ, the
  forecast IC at which net Sharpe crosses zero, interpolated in log IC between
  the oracle ICs that bracket it, from the expected-IC run (E1x) when it
  exists, else E1. NaN where the construction never or always pays across the
  ICs run.

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


def lookup(results: Path):
    import pandas as pd

    for name in ("e1x_expected_ic", "e1_transfer_map"):
        path = results / name / "rows.parquet"
        if path.is_file():
            rows = pd.read_parquet(path)
            break
    else:
        return None
    rows = rows[(rows["f_noise"] == "random") & (rows["efq"] > 0)]
    out = []
    for (weighting, selection, efq), part in rows.groupby(["weighting", "selection", "efq"]):
        curve = part.groupby("f_ic")["sharpe_net"].mean()
        out.append({"source": name, "weighting": weighting, "selection": selection, "efq": efq,
                    "breakeven_ic": breakeven_ic(curve),
                    "net_sharpe_at_max_ic": float(curve.iloc[-1])})
    return pd.DataFrame(out)


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
