"""Figures for the research build: uncertainty, the fundamental law, robustness.

Each figure is drawn from finished rows (and, where it says, the inference and
walk-forward outputs) in light and dark variants with the palette and helpers
of :mod:`experiments.figures`, and is skipped, with a note, when its inputs do
not exist yet:

* ``law``: realized per-period IR against ``TC * IC * sqrt(N)`` (the
  fundamental law with the transfer coefficient), one point per row of the
  expected-IC runs, coloured by construction, with the 45-degree line;
* ``daily_ci``: daily net Sharpe with its 95% day-bootstrap interval for each
  construction, at IC 0.03 and EFQ 0.7, per noise placement;
* ``seeds``: the spread of daily Sharpe across oracle seeds, per construction,
  against the gaps between constructions it is meant to be compared with;
* ``breakeven``: a heatmap of break-even EFQ by construction and IC;
* ``walkforward``: per-month small multiples of daily net Sharpe, when any
  ``results/*_wf`` run exists;
* ``mcs``: the share of groups in which each construction is in the 90% model
  confidence set, when ``results/*/inference.parquet`` exists.

    uv run --extra experiments python -m experiments.figures_v2 [--results DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from experiments.figures import INK, LABELS, SEQUENTIAL, WEIGHTING_ORDER, _color, _plt
from experiments.run import RESULTS

MODES = ("light", "dark")


def _rows(results: Path, name: str):
    import pandas as pd

    path = results / name / "rows.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _save(fig, out: Path, name: str, mode: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{mode}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.clf()
    return path


def _weightings(frame) -> list[str]:
    return [w for w in WEIGHTING_ORDER if w in set(frame["weighting"])]


def law(results: Path, out: Path, mode: str) -> list[Path]:
    import pandas as pd

    frames = [f for f in (_rows(results, "e1x_expected_ic"), _rows(results, "e6_sanity_checks"))
              if f is not None]
    if not frames:
        return []
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[(rows["efq"] == 0.0) & (rows["information_coefficient"].abs() > 1e-3)
                & (rows["information_coefficient"] < 0.5)
                & np.isfinite(rows["transfer_coefficient"])]
    if rows.empty:
        return []
    plt = _plt(mode)
    ir = rows["sharpe_mid"] / np.sqrt(rows["periods_per_year"])
    theory = rows["transfer_coefficient"] * rows["information_coefficient"] * np.sqrt(
        rows["mean_names_held"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for w in _weightings(rows):
        part = rows["weighting"] == w
        ax.scatter(theory[part], ir[part], s=14, alpha=0.8, color=_color(mode, w, WEIGHTING_ORDER),
                   label=LABELS[w])
    top = float(np.nanmax([theory.max(), ir.max()]))
    ax.plot([0, top], [0, top], color=INK[mode]["axis"], linewidth=1, label="law")
    ax.set_xlabel("TC x IC x sqrt(N)  (per period)")
    ax.set_ylabel("Realized IR per period (gross)")
    ax.set_title("The fundamental law, expected-IC oracles")
    ax.legend(fontsize=7)
    path = _save(fig, out, "law", mode)
    plt.close(fig)
    return [path]


def daily_ci(results: Path, out: Path, mode: str) -> list[Path]:
    rows = next((f for f in (_rows(results, "e1x_expected_ic"), _rows(results, "e1_transfer_map"))
                 if f is not None and "sharpe_daily_net" in f), None)
    if rows is None:
        return []
    part = rows[np.isclose(rows["f_ic"], 0.03) & np.isclose(rows["efq"], 0.7)
                & (rows["selection"] == "all")]
    if part.empty:
        return []
    plt = _plt(mode)
    noises = [n for n in ("tails", "middle", "random") if n in set(part["f_noise"])]
    weightings = _weightings(part)
    fig, ax = plt.subplots(figsize=(1.3 * len(weightings) + 2.0, 3.0))
    width = 0.8 / max(len(noises), 1)
    for k, noise in enumerate(noises):
        sub = part[part["f_noise"] == noise].groupby("weighting")
        mid = sub["sharpe_daily_net"].mean()
        low, high = sub["sharpe_daily_net_low"].mean(), sub["sharpe_daily_net_high"].mean()
        x = np.arange(len(weightings)) + (k - (len(noises) - 1) / 2) * width
        values = np.array([mid.get(w, np.nan) for w in weightings])
        err = np.array([[values[i] - low.get(w, np.nan), high.get(w, np.nan) - values[i]]
                        for i, w in enumerate(weightings)]).T
        ax.bar(x, values, width=width * 0.9, color=SEQUENTIAL[2 + 2 * k], label=LABELS[noise])
        ax.errorbar(x, values, yerr=err, fmt="none", ecolor=INK[mode]["primary"], linewidth=0.8,
                    capsize=2)
    ax.axhline(0, color=INK[mode]["axis"], linewidth=1)
    ax.set_xticks(range(len(weightings)), [LABELS[w] for w in weightings])
    ax.set_ylabel("Daily net Sharpe (95% CI)")
    ax.set_title("IC 0.03, 15 min, EFQ 70%: daily net Sharpe with day-bootstrap intervals")
    ax.legend(fontsize=7)
    path = _save(fig, out, "daily_ci", mode)
    plt.close(fig)
    return [path]


def seeds(results: Path, out: Path, mode: str) -> list[Path]:
    rows = _rows(results, "e6_sanity_checks")
    if rows is None:
        return []
    part = rows[(rows["forecast"] == "oracle") & np.isclose(rows["f_ic"], 0.03)
                & (rows["efq"] == 0.0) & (rows["rebalance"] == "horizon")]
    if part.empty:
        return []
    plt = _plt(mode)
    weightings = _weightings(part)
    fig, ax = plt.subplots(figsize=(1.3 * len(weightings) + 2.0, 3.0))
    for i, w in enumerate(weightings):
        values = part[part["weighting"] == w]["sharpe_daily_mid"].to_numpy()
        jitter = (np.arange(len(values)) - (len(values) - 1) / 2) * 0.03
        ax.scatter(np.full(len(values), i) + jitter, values, s=16,
                   color=_color(mode, w, WEIGHTING_ORDER))
        ax.plot([i - 0.25, i + 0.25], [values.mean()] * 2, color=INK[mode]["primary"],
                linewidth=1.2)
    ax.set_xticks(range(len(weightings)), [LABELS[w] for w in weightings])
    ax.set_ylabel("Daily gross Sharpe")
    ax.set_title("One forecast quality (IC 0.03), ten seeds: the spread one draw hides")
    path = _save(fig, out, "seeds", mode)
    plt.close(fig)
    return [path]


def breakeven(results: Path, out: Path, mode: str) -> list[Path]:
    rows = next((f for f in (_rows(results, "e1x_expected_ic"), _rows(results, "e1_transfer_map"))
                 if f is not None and "breakeven_efq" in f), None)
    if rows is None:
        return []
    part = rows[(rows["f_noise"] == "random") & (rows["selection"] == "all")
                & (rows["efq"] > 0) & np.isfinite(rows["breakeven_efq"])]
    if part.empty:
        return []
    plt = _plt(mode)
    weightings = _weightings(part)
    ics = sorted(part["f_ic"].unique())
    grid = np.array([[part[(part["weighting"] == w) & np.isclose(part["f_ic"], ic)]
                      ["breakeven_efq"].median() for ic in ics] for w in weightings])
    fig, ax = plt.subplots(figsize=(0.9 * len(ics) + 2.5, 0.45 * len(weightings) + 1.6))
    image = ax.imshow(np.clip(grid, 0, 2), cmap="Blues", vmin=0, vmax=2, aspect="auto")
    for i in range(len(weightings)):
        for j in range(len(ics)):
            value = grid[i, j]
            if np.isfinite(value):
                ink = "#ffffff" if value > 1.2 else INK["light"]["primary"]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=ink)
    ax.set_xticks(range(len(ics)), [f"{ic:g}" for ic in ics])
    ax.set_yticks(range(len(weightings)), [LABELS[w] for w in weightings])
    ax.set_xlabel("Forecast IC")
    ax.set_title("Break-even EFQ (random-error oracle, whole cross-section)")
    fig.colorbar(image, ax=ax, label="EFQ (clipped at 2)")
    path = _save(fig, out, "breakeven", mode)
    plt.close(fig)
    return [path]


def walkforward(results: Path, out: Path, mode: str) -> list[Path]:
    import pandas as pd

    paths = []
    for run in sorted(results.glob("*_wf")):
        rows_path = run / "rows.parquet"
        if not rows_path.is_file():
            continue
        rows = pd.read_parquet(rows_path)
        if "eval_month" not in rows or "sharpe_daily_net" not in rows:
            continue
        plt = _plt(mode)
        months = sorted(rows["eval_month"].unique())
        fig, axes = plt.subplots(1, len(months), figsize=(2.6 * len(months), 2.6), sharey=True,
                                 squeeze=False)
        for ax, month in zip(axes[0], months):
            part = rows[rows["eval_month"] == month].groupby("weighting")["sharpe_daily_net"]
            weightings = [w for w in WEIGHTING_ORDER if w in part.groups]
            ax.bar(range(len(weightings)), [part.get_group(w).mean() for w in weightings],
                   color=[_color(mode, w, WEIGHTING_ORDER) for w in weightings])
            ax.set_xticks(range(len(weightings)), [LABELS[w] for w in weightings], rotation=60,
                          fontsize=6)
            ax.set_title(str(month))
        axes[0, 0].set_ylabel("Daily net Sharpe")
        fig.suptitle(f"{run.name}: by evaluation month")
        paths.append(_save(fig, out, f"walkforward_{run.name}", mode))
        plt.close(fig)
    return paths


def mcs(results: Path, out: Path, mode: str) -> list[Path]:
    import pandas as pd

    frames = []
    for path in sorted(results.glob("*/inference.parquet")):
        frame = pd.read_parquet(path)
        if {"in_mcs", "weighting"} <= set(frame.columns):
            frames.append(frame.assign(run=path.parent.name))
    if not frames:
        return []
    rows = pd.concat(frames, ignore_index=True)
    plt = _plt(mode)
    share = rows.groupby("weighting")["in_mcs"].mean()
    weightings = [w for w in WEIGHTING_ORDER if w in share.index]
    fig, ax = plt.subplots(figsize=(1.2 * len(weightings) + 2.0, 2.8))
    ax.bar(range(len(weightings)), [share[w] for w in weightings],
           color=[_color(mode, w, WEIGHTING_ORDER) for w in weightings])
    ax.set_xticks(range(len(weightings)), [LABELS[w] for w in weightings])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of groups in the 90% MCS")
    ax.set_title("Which constructions cannot be told apart from the best")
    path = _save(fig, out, "mcs", mode)
    plt.close(fig)
    return [path]


FIGURES = (law, daily_ci, seeds, breakeven, walkforward, mcs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--out", default=None, help="default <results>/figures")
    args = parser.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "figures"
    for figure in FIGURES:
        made = [p for mode in MODES for p in figure(results, out, mode)]
        print(f"{figure.__name__}: " + (", ".join(p.name for p in made) if made
                                        else "skipped (inputs not found)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
