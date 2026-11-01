"""Figures for the experiment results, in light and dark variants.

Every figure is drawn from the result rows ``run.py`` wrote, so a figure and
the table beside it cannot disagree. Colours follow one validated categorical
order (identity: a construction keeps its colour in every figure), a single
blue ramp for magnitude, and a blue-gray-red diverging ramp for signed
quantities. Text is always in ink colours, never a series colour, and every
multi-series panel carries a legend.

    uv run --extra experiments python -m experiments.figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = RESULTS / "figures"

#: Categorical slots in fixed order, light and dark steps of the same hues.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
              "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
             "#e66767"],
}
INK = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

#: A construction keeps its slot in every figure.
WEIGHTING_ORDER = ["equal", "rank", "signal", "hrp", "min_var_qp", "mean_variance"]
NOISE_ORDER = ["tails", "middle", "random"]
LABELS = {"equal": "Equal", "rank": "Rank", "signal": "Signal", "hrp": "HRP",
          "min_var_qp": "Min-var (QP)", "mean_variance": "Mean-variance",
          "tails": "Errors in the tails", "middle": "Errors in the middle",
          "random": "Errors at random"}


def _plt(mode: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = INK[mode]
    plt.rcParams.update({
        "figure.facecolor": ink["surface"], "axes.facecolor": ink["surface"],
        "savefig.facecolor": ink["surface"], "text.color": ink["primary"],
        "axes.labelcolor": ink["secondary"], "axes.edgecolor": ink["axis"],
        "xtick.color": ink["muted"], "ytick.color": ink["muted"],
        "axes.grid": True, "grid.color": ink["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": ink["primary"],
        "legend.frameon": False, "legend.labelcolor": ink["secondary"],
        "lines.linewidth": 2.0, "lines.markersize": 5,
    })
    return plt


#: Categorical slot per construction. Mean-variance takes violet rather than the
#: next slot's green, which sits too close to signal's aqua when both appear.
SLOT = {"equal": 0, "rank": 1, "signal": 2, "hrp": 3, "min_var_qp": 4, "mean_variance": 6}


def _color(mode: str, key: str, order: list[str]) -> str:
    index = SLOT.get(key, order.index(key))
    return SERIES[mode][index % len(SERIES[mode])]


def _save(fig, name: str, mode: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}-{mode}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.clf()
    return path


def _rows(name: str):
    import pandas as pd

    path = RESULTS / name / "rows.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _held(frame, horizon):
    """Rows scored with each book held for ``horizon``: ``rebalance="horizon"``,
    which ``run.py`` runs as ``every_decision`` at the 300 s decision spacing."""
    at = frame[frame["eval_horizon"] == horizon]
    return at[at["rebalance"] == ("every_decision" if horizon <= 300 else "horizon")]


# ── E1: IC to Sharpe ────────────────────────────────────────────────────────

def transfer_map(mode: str) -> list[Path]:
    frame = _rows("e1_transfer_map")
    if frame is None:
        return []
    plt = _plt(mode)
    out = []
    for metric, efq, title in (("sharpe_mid", 0.0, "Gross Sharpe (mid)"),
                               ("sharpe_net", 0.7, "Net Sharpe at EFQ 70%")):
        horizons = sorted(frame["eval_horizon"].unique())
        noises = [n for n in NOISE_ORDER if n in set(frame["f_noise"])]
        fig, axes = plt.subplots(len(horizons), len(noises), figsize=(3.2 * len(noises),
                                 2.6 * len(horizons)), sharex=True, squeeze=False)
        for i, h in enumerate(horizons):
            for j, noise in enumerate(noises):
                ax = axes[i, j]
                part = _held(frame, h)
                part = part[(part["f_noise"] == noise) & (part["selection"] == "all")
                            & np.isclose(part["efq"], efq)]
                for w in WEIGHTING_ORDER:
                    line = part[part["weighting"] == w].sort_values("f_ic")
                    if len(line):
                        ax.plot(line["f_ic"], line[metric], marker="o",
                                color=_color(mode, w, WEIGHTING_ORDER), label=LABELS[w])
                ax.axhline(0, color=INK[mode]["axis"], linewidth=1)
                ax.set_xscale("log")
                if i == 0:
                    ax.set_title(LABELS[noise])
                if j == 0:
                    ax.set_ylabel(f"{h // 60} min horizon\n{title}")
                if i == len(horizons) - 1:
                    ax.set_xlabel("Forecast rank IC")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{title} against IC, whole cross-section, held books", y=1.07)
        out.append(_save(fig, f"e1_{metric}", mode))
        plt.close(fig)
    return out


def breakeven(mode: str) -> list[Path]:
    frame = _rows("e1_transfer_map")
    if frame is None:
        return []
    plt = _plt(mode)
    horizons = sorted(frame["eval_horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.4 * len(horizons), 2.8),
                             sharey=True, squeeze=False)
    for j, h in enumerate(horizons):
        ax = axes[0, j]
        part = _held(frame, h)
        part = part[(part["f_noise"] == "random") & (part["selection"] == "all")
                    & np.isclose(part["efq"], 1.0)]
        for w in WEIGHTING_ORDER:
            line = part[part["weighting"] == w].sort_values("f_ic")
            if len(line):
                ax.plot(line["f_ic"], line["breakeven_efq"] * 100, marker="o",
                        color=_color(mode, w, WEIGHTING_ORDER), label=LABELS[w])
        ax.set_xscale("log")
        for level, name in ((55, "direct access"), (88, "NYSE")):
            ax.axhline(level, color=INK[mode]["muted"], linewidth=1, linestyle="--")
            if j == 0:
                ax.text(0.0045, level + 2, name, color=INK[mode]["muted"], fontsize=8)
        ax.set_title(f"{h // 60} min horizon")
        ax.set_xlabel("Forecast rank IC")
        if j == 0:
            ax.set_ylabel("Break-even EFQ (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("The execution quality at which the edge is used up (random errors)", y=1.16)
    path = _save(fig, "e1_breakeven", mode)
    plt.close(fig)
    return [path]


def transfer_coefficients(mode: str) -> list[Path]:
    import pandas as pd

    frame = _rows("e1_transfer_map")
    if frame is None:
        return []
    plt = _plt(mode)
    part = _held(frame, 900)
    part = part[(part["f_noise"] == "random") & np.isclose(part["f_ic"], 0.03)
                & np.isclose(part["efq"], 0.7)]
    e1b = _rows("e1b_mean_variance")
    if e1b is not None:
        extra = _held(e1b, 900)
        extra = extra[(extra["f_noise"] == "random") & np.isclose(extra["f_ic"], 0.03)
                      & np.isclose(extra["efq"], 0.7)]
        part = pd.concat([part, extra], ignore_index=True)
    weightings = [w for w in WEIGHTING_ORDER if w in set(part["weighting"])]
    fig, ax = plt.subplots(figsize=(1.25 * len(weightings) + 1.0, 2.8))
    x = np.arange(len(weightings))
    for k, (selection, shade) in enumerate((("all", 0), ("decile", 1))):
        values = [part[(part["weighting"] == w) & (part["selection"] == selection)]
                  ["transfer_coefficient"].mean() for w in weightings]
        colors = [_color(mode, w, WEIGHTING_ORDER) for w in weightings]
        ax.bar(x + (k - 0.5) * 0.38, values, width=0.36, color=colors,
               alpha=1.0 if shade == 0 else 0.55, edgecolor=INK[mode]["surface"], linewidth=2,
               label="whole cross-section" if shade == 0 else "deciles only")
        for xi, v in zip(x + (k - 0.5) * 0.38, values):
            if np.isfinite(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7,
                        color=INK[mode]["secondary"])
    ax.set_xticks(x, [LABELS[w] for w in weightings])
    ax.set_ylabel("Transfer coefficient")
    top = np.nanmax(part["transfer_coefficient"]) if len(part) else 1.0
    ax.set_ylim(0, top * 1.3)
    ax.set_title("How much of the forecast the held book expresses\n"
                 "IC 0.03, 15 min, EFQ 70%", pad=10)
    ax.legend(loc="upper left", ncol=2)
    path = _save(fig, "e1_transfer_coefficient", mode)
    plt.close(fig)
    return [path]


# ── E5: real forecasts on the oracle's map ─────────────────────────────────

def real_on_map(mode: str) -> list[Path]:
    oracle, real = _rows("e1_transfer_map"), _rows("e5_real_signals")
    if oracle is None or real is None:
        return []
    plt = _plt(mode)
    horizons = sorted(set(oracle["eval_horizon"]) & set(real["eval_horizon"]))
    markers = {"lasso": "D", "ridge": "s", "reversal_15m": "^"}
    names = {"lasso": "LASSO", "ridge": "Ridge", "reversal_15m": "15-min reversal"}
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.0 * len(horizons), 3.0), squeeze=False)
    for j, h in enumerate(horizons):
        ax = axes[0, j]
        for w in ("signal", "hrp"):
            color = _color(mode, w, WEIGHTING_ORDER)
            curve = _held(oracle, h)
            curve = curve[(curve["f_noise"] == "random") & (curve["weighting"] == w)
                          & (curve["selection"] == "all") & np.isclose(curve["efq"], 0.0)]
            curve = curve.sort_values("f_ic")
            ax.plot(curve["f_ic"], curve["sharpe_mid"], color=color, alpha=0.55,
                    label=f"{LABELS[w]}, oracle (random errors)")
            points = _held(real, h)
            points = points[(points["weighting"] == w) & (points["selection"] == "all")
                            & np.isclose(points["efq"], 0.0)]
            for _, row in points.iterrows():
                ax.scatter(row["information_coefficient"], row["sharpe_mid"], s=46,
                           marker=markers.get(row["f_name"], "o"), color=color,
                           edgecolor=INK[mode]["surface"], linewidth=1.5, zorder=3,
                           label=f"{LABELS[w]}, {names.get(row['f_name'], row['f_name'])}")
        ax.axhline(0, color=INK[mode]["axis"], linewidth=1)
        ax.set_xscale("log")
        ax.set_title(f"{h // 60} min horizon")
        ax.set_xlabel("Rank IC (oracle: target; real: measured)")
        if j == 0:
            ax.set_ylabel("Gross Sharpe (mid)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.16), fontsize=8)
    fig.suptitle("Real forecasts earn less than an oracle of the same IC", y=1.24)
    path = _save(fig, "e5_real_on_map", mode)
    plt.close(fig)
    return [path]


# ── E2: where errors sit ───────────────────────────────────────────────────

def where_errors_sit(mode: str) -> list[Path]:
    frame = _rows("e2_where_errors_sit")
    if frame is None:
        return []
    plt = _plt(mode)
    out = []
    for selection in ("all", "decile"):
        part = _held(frame, 900)
        part = part[(part["selection"] == selection) & np.isclose(part["f_ic"], 0.03)]
        weightings = [w for w in WEIGHTING_ORDER if w in set(part["weighting"])]
        noises = [n for n in NOISE_ORDER if n in set(part["f_noise"])]
        grid = np.array([[part[(part["weighting"] == w) & (part["f_noise"] == n)]
                          ["sharpe_mid"].mean() for n in noises] for w in weightings])
        fig, ax = plt.subplots(figsize=(4.8, 0.5 * len(weightings) + 1.2))
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

        # Signed values: blue for gains, red for losses, the neutral gray at zero.
        middle = "#f0efec" if mode == "light" else "#383835"
        cmap = LinearSegmentedColormap.from_list(
            "div", ["#a82828", "#e34948", "#f4b4b3", middle, "#9ec5f4", "#2a78d6", "#0d366b"])
        bound = float(np.nanmax(np.abs(grid))) or 1.0
        image = ax.imshow(grid, cmap=cmap, aspect="auto",
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound))
        ax.grid(False)
        ax.set_xticks(range(len(noises)), [LABELS[n].replace("Errors ", "") for n in noises])
        ax.set_yticks(range(len(weightings)), [LABELS[w] for w in weightings])
        for i in range(len(weightings)):
            for j in range(len(noises)):
                v = grid[i, j]
                r, g, b_, _ = cmap(image.norm(v))
                light_cell = 0.2126 * r + 0.7152 * g + 0.0722 * b_ > 0.5
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="#0b0b0b" if light_cell else "#ffffff")
        cbar = fig.colorbar(image, ax=ax, fraction=0.05)
        cbar.set_label("Gross Sharpe (mid)")
        ax.set_title(f"Same IC, same dispersion, different errors ({selection}, IC 0.03, 15 min)")
        out.append(_save(fig, f"e2_heatmap_{selection}", mode))
        plt.close(fig)
    return out


# ── E3: horizon and overlap ────────────────────────────────────────────────

def overlap(mode: str) -> list[Path]:
    frame = _rows("e3_horizon_overlap")
    if frame is None:
        return []
    plt = _plt(mode)
    part = frame[(frame["weighting"] == "equal") & np.isclose(frame["efq"], 0.0)]
    horizons = sorted(part["eval_horizon"].unique())
    # A half-life of 0 stands for independent errors (half_life=None).
    half_life = part["f_half_life"].fillna(0)
    half_lives = sorted(half_life.unique())
    label = lambda h: "independent" if h == 0 else f"half-life {int(h)}"
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0))
    ax = axes[0]
    width = 0.8 / max(len(half_lives), 1)
    for k, p in enumerate(half_lives):
        ratios = []
        for h in horizons:
            sub = part[(part["eval_horizon"] == h) & (half_life == p)]
            legacy = sub[sub["rebalance"] == "every_decision"]["mean_gross_return_bps"].mean()
            held_rows = sub[sub["rebalance"] == "horizon"]
            # A held book's period is the horizon, h / 300 decisions long.
            held = (held_rows["mean_gross_return_bps"].mean() / max(h // 300, 1)
                    if len(held_rows) else legacy)
            ratios.append(legacy / held if held else np.nan)
        ax.bar(np.arange(len(horizons)) + (k - (len(half_lives) - 1) / 2) * width, ratios,
               width=width * 0.92, color=SERIES[mode][k], label=label(p))
    ax.axhline(1, color=INK[mode]["axis"], linewidth=1)
    ax.set_xticks(range(len(horizons)), [f"{h // 60}m" for h in horizons])
    ax.set_xlabel("Forecast horizon (decisions every 5 min)")
    ax.set_ylabel("Booked / earned return per decision")
    ax.set_title("Overlapping returns inflate the booked P&L")
    ax.legend()
    ax = axes[1]
    for k, p in enumerate(half_lives):
        sub = part[(half_life == p) & (part["rebalance"] == "every_decision")]
        turnover = [sub[sub["eval_horizon"] == h]["mean_turnover"].mean() for h in horizons]
        ax.plot(range(len(horizons)), turnover, marker="o", color=SERIES[mode][k],
                label=label(p))
    ax.set_xticks(range(len(horizons)), [f"{h // 60}m" for h in horizons])
    ax.set_ylabel("Turnover per decision (x gross)")
    ax.set_title("Persistent errors trade less")
    ax.legend()
    fig.tight_layout()
    path = _save(fig, "e3_overlap", mode)
    plt.close(fig)
    return [path]


# ── E4: harness sensitivity ────────────────────────────────────────────────

def harness(mode: str) -> list[Path]:
    frame = _rows("e4_harness_sensitivity")
    if frame is None:
        return []
    plt = _plt(mode)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ics = sorted(frame["f_ic"].unique())
    for k, ic in enumerate(ics):
        part = frame[np.isclose(frame["f_ic"], ic)]
        for w in WEIGHTING_ORDER:
            sub = part[part["weighting"] == w]
            if not len(sub):
                continue
            jitter = (np.arange(len(sub)) - len(sub) / 2) / max(len(sub), 1) * 0.25
            ax.scatter(k + jitter, sub["sharpe_net"], s=22,
                       color=_color(mode, w, WEIGHTING_ORDER),
                       edgecolor=INK[mode]["surface"], linewidth=1,
                       label=LABELS[w] if k == 0 else None)
    ax.axhline(0, color=INK[mode]["axis"], linewidth=1)
    ax.set_xticks(range(len(ics)), [f"IC {ic:g}" for ic in ics])
    ax.set_ylabel("Net Sharpe at EFQ 70%")
    ax.set_title("One forecast, many harnesses (15 min, held books)", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    path = _save(fig, "e4_harness_spread", mode)
    plt.close(fig)
    return [path]


# ── stage timings ──────────────────────────────────────────────────────────

def timings(mode: str) -> list[Path]:
    path = RESULTS / "bench" / "bench.json"
    if not path.is_file():
        return []
    bench = json.loads(path.read_text())
    plt = _plt(mode)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.8))
    names = list(bench["seconds"])
    for ax, key, label in ((axes[0], "seconds", "Seconds per month (one horizon)"),
                           (axes[1], "peak_mb", "Peak allocated memory (MB)")):
        values = [bench[key][n] for n in names]
        colors = [_color(mode, n, WEIGHTING_ORDER) if n in WEIGHTING_ORDER
                  else INK[mode]["muted"] for n in names]
        ax.barh(range(len(names)), values, color=colors, edgecolor=INK[mode]["surface"],
                linewidth=2)
        ax.set_yticks(range(len(names)), [LABELS.get(n, n) for n in names])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=7, color=INK[mode]["secondary"])
    fig.suptitle(f"Cost of each stage on one month "
                 f"({bench['decisions']} decisions x {bench['assets']} names, one thread)")
    fig.tight_layout()
    out = _save(fig, "bench", mode)
    plt.close(fig)
    return [out]


def main() -> int:
    made = []
    for mode in ("light", "dark"):
        for draw in (transfer_map, breakeven, transfer_coefficients, where_errors_sit, overlap,
                     real_on_map, harness, timings):
            made += draw(mode)
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
