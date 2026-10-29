"""Turn TFWM encoder panels into forecast files the harness can score.

The TFWM encoders (Towards Financial World Modeling, arXiv 2610.09048; 18
pretrained ~22M-parameter transformers at huggingface.co/fin-ai-lab) need a
GPU to embed a month, so the expensive step runs on a cluster with TFWM's own
script, and only its output comes here:

    # on a GPU node, in a clone of github.com/fin-ai-lab/tfwm
    uv run scripts/eval/portfolio_ic_sharpe.py embed \\
        --ckpt-dirs <released-encoder>/<eval-month> --panel-dir panels/

Each panel holds a training month's and an evaluation month's embeddings with
their row keys (date, anchor, ticker) and raw forward returns. This script
fits the paper's probe on the training month -- ridge on the embeddings,
against the cross-sectionally demeaned raw return, so the forecast is in
return units -- applies it to the evaluation month, and writes one forecast
file per panel on the harness's ``(decision, asset, horizon)`` grid, keyed by
the same ``"date|anchor"`` labels the Market-1T readers use. ``run.py`` then
scores it like any other forecast (``kind: "file"``), restricted to the
decisions the encoder was evaluated at.

    uv run --extra experiments python -m experiments.tfwm_panels panels/*.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import stable_finance as sf
from stable_finance.dataset.datasets.market1t.daystore import decision_labels

OUT = Path(__file__).resolve().parent / "results" / "tfwm"


def grid(rows_value: np.ndarray, labels: np.ndarray, tickers: np.ndarray):
    """Scatter row-keyed values onto a ``(decision, asset, ...)`` cube."""
    decisions = np.unique(labels)
    assets = np.unique(tickers)
    d = np.searchsorted(decisions, labels)
    a = np.searchsorted(assets, tickers)
    cube = np.full((len(decisions), len(assets), *rows_value.shape[1:]), np.nan)
    cube[d, a] = rows_value
    return cube, decisions, assets


def demean_by_decision(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for label in np.unique(labels):
        rows = labels == label
        out[rows] -= np.nanmean(out[rows], axis=0)
    return out


def forecast_from_panel(path: Path, alpha: float = 10.0) -> Path:
    panel = np.load(path, allow_pickle=False)
    horizons = panel["horizons"].astype(np.int64)
    train_labels = decision_labels(panel["train_date"], panel["train_anchor"])
    eval_labels = decision_labels(panel["eval_date"], panel["eval_anchor"])
    target = demean_by_decision(panel["train_raw"], train_labels)
    probe = sf.StreamingRidge(alpha=alpha).partial_fit(panel["train_X"], target).finalize()
    prediction = probe.predict(panel["eval_X"])
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    cube, decisions, assets = grid(prediction, eval_labels, panel["eval_ticker"].astype(str))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{path.stem}.npz"
    np.savez(out, forecast=cube, decisions=decisions, assets=assets, horizons=horizons)
    realized, _, _ = grid(panel["eval_raw"].astype(np.float64), eval_labels,
                          panel["eval_ticker"].astype(str))
    ic = [float(sf.grouped_rank_ic(cube[:, :, h], realized[:, :, h], min_assets=20).mean)
          for h in range(len(horizons))]
    print(f"{path.stem}: {len(decisions)} decisions x {len(assets)} names; "
          f"probe rank IC by horizon {dict(zip(horizons.tolist(), np.round(ic, 4)))}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("panels", nargs="+", type=Path)
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="ridge penalty; the paper's probe uses 10")
    args = parser.parse_args(argv)
    for path in args.panels:
        forecast_from_panel(path, alpha=args.alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
