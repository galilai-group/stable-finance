"""Dynamic trading rules against trading to the target, out of sample.

Every book in the main studies is traded to its target at every decision. This
script asks how much of the cost of that a dynamic rule saves, using the rules
in :mod:`stable_finance.evaluation.trades.dynamic`:

* ``target``: the aim portfolio with no cost aversion -- the frictionless
  mean-variance portfolio of the per-decision sum of the multi-horizon
  forecasts -- traded to at every decision (the baseline);
* ``partial``: :func:`~stable_finance.partial_adjustment` toward it, at the
  rate that maximised net Sharpe on the training month;
* ``band``: :func:`~stable_finance.no_trade_band` around it, with the band a
  multiple of each name's cost, the multiple chosen on the training month;
* ``aim``: the Garleanu-Pedersen aim portfolio, whose cost aversion shrinks
  the fast-decaying short-horizon signal, traded partway at the tuned rate;
  cost aversion and rate chosen together on the training month.

Every parameter is chosen on July 2020 and the rules are then scored once on
August 2020, so the comparison is out of sample. Forecasts are multi-horizon
oracles at 5, 15 and 60 minutes with ICs 0.03, 0.02 and 0.01: the copula
oracle (errors persist per name, half-life 6 decisions) and the random-swap
oracle (errors independent across decisions). Books are held for one decision
(five minutes) and scored by the harness at the weights level with
cross-spread fills at EFQ 0.7. The risk model is a shrunk sample covariance of
the previous month's five-minute returns (for July, July's own: the tuning
month has no earlier month cached, which only affects the tuning).

    uv run --extra experiments python -m experiments.dynamic

Writes ``results/dynamic/dynamic.json``.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

import stable_finance as sf
from experiments.month_cache import load_month
from experiments.policy import (
    EFQ,
    ORACLE_HORIZONS,
    ORACLE_ICS,
    month_inputs,
)
from experiments.run import _inference

OUT = Path(__file__).resolve().parent / "results" / "dynamic"
TRAIN, EVALUATE = "2020-07", "2020-08"
GROSS = 2.0
RATES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
BANDS = (0.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
COST_AVERSIONS = (0.0, 5.0, 20.0, 100.0, 500.0)


def common_assets() -> np.ndarray:
    """Names quoted on at least 95% of the evaluation decisions and present in both months."""
    train, test = load_month(TRAIN), load_month(EVALUATE)
    quoted = np.isfinite(test["half_spread"]).mean(axis=0) >= 0.95
    keep = set(test["realized"].assets[quoted].astype(str)) & set(
        train["realized"].assets.astype(str))
    return np.asarray(sorted(keep))


def forecasts(month: dict) -> dict[str, sf.ForwardReturns]:
    """The two multi-horizon oracles on one month's panel."""
    realized = month["realized"]
    panel = sf.ForwardReturns(month["horizon_panel"], realized.decisions, realized.assets,
                              np.asarray(ORACLE_HORIZONS))
    random = sf.oracle_forecast(panel, np.asarray(ORACLE_ICS), noise="random", seed=0)
    copula = sf.ForwardReturns(month["oracle"], realized.decisions, realized.assets,
                               np.asarray(ORACLE_HORIZONS))
    return {"copula": copula, "random": random}


def load(month_name: str, assets: np.ndarray, sigma_from: dict | None = None) -> dict:
    month = month_inputs(month_name, assets)
    data = load_month(month_name)
    index = {a: i for i, a in enumerate(data["realized"].assets.astype(str))}
    take = [index[a] for a in assets]
    hz = [int(np.flatnonzero(data["realized"].horizons == h)[0]) for h in ORACLE_HORIZONS]
    month["horizon_panel"] = data["realized"].values[:, take][:, :, hz]
    source = month if sigma_from is None else sigma_from
    month["sigma"] = sf.estimate_risk("shrunk_sample", source["realized"])[0]
    return month


def aim(forecast: sf.ForwardReturns, month: dict, cost_aversion: float) -> sf.PortfolioWeights:
    """The aim portfolio, scaled by a causal risk aversion to a mean gross of GROSS.

    The scale follows the harness: the risk aversion at each decision makes
    the books up to it average the requested gross, so no decision is sized
    with a later one's forecast.
    """
    unit = sf.aim_portfolio(forecast, month["sigma"], risk_aversion=1.0,
                            cost_aversion=cost_aversion)
    gross = np.nansum(np.abs(unit.values[:, :, 0]), axis=1)
    aversion = np.maximum(np.cumsum(gross) / np.arange(1, len(gross) + 1) / GROSS, 1e-12)
    return sf.PortfolioWeights(np.nan_to_num(unit.values / aversion[:, None, None]),
                               unit.decisions, unit.assets, unit.horizons)


def evaluate(held: sf.PortfolioWeights, month: dict) -> dict:
    config = sf.DEFAULT_BACKTEST.replace(cost_model="cross", efq=EFQ, annualization="decisions")
    net = sf.run_backtest(held, month["realized"], market=month["market"], days=month["days"],
                          config=config, with_series=True)
    series = net.series
    out = {"sharpe_mid": float(net.sharpe_mid), "sharpe_net": float(net.sharpe_net),
           "mean_turnover": float(net.mean_turnover),
           "holding_decisions": float(net.mean_gross_exposure / net.mean_turnover)
           if net.mean_turnover > 0 else float("inf"),
           "breakeven_efq": float(sf.breakeven_efq(series.gross, series.cost / EFQ)),
           **_inference(series)}
    return out


def policies(target: sf.PortfolioWeights, month: dict, *, rate: float, band: float
             ) -> dict[str, sf.PortfolioWeights]:
    return {"target": target,
            "partial": sf.partial_adjustment(target, rate),
            "band": sf.no_trade_band(target, band, cost=month["cost"])}


def tune(forecast: sf.ForwardReturns, month: dict) -> dict:
    """Each rule's parameter by net Sharpe on the training month."""
    target = aim(forecast, month, 0.0)
    by_rate = {r: evaluate(sf.partial_adjustment(target, r), month)["sharpe_net"] for r in RATES}
    by_band = {b: evaluate(sf.no_trade_band(target, b, cost=month["cost"]), month)["sharpe_net"]
               for b in BANDS}
    rate = max(by_rate, key=by_rate.get)
    by_aim = {k: evaluate(sf.partial_adjustment(aim(forecast, month, k), rate),
                          month)["sharpe_net"] for k in COST_AVERSIONS}
    # A parameter tuned onto the edge of its grid is a sign the grid is too
    # narrow; the report records it so a reader can see.
    edges = [name for name, grid, best in (("rate", RATES, rate),
                                           ("band", BANDS, max(by_band, key=by_band.get)),
                                           ("cost_aversion", COST_AVERSIONS,
                                            max(by_aim, key=by_aim.get)))
             if best in (grid[0], grid[-1]) and best != 0.0]
    return {"rate": rate, "band": max(by_band, key=by_band.get), "at_grid_edge": edges,
            "cost_aversion": max(by_aim, key=by_aim.get),
            "train_sharpe_net": {"rate": {str(k): v for k, v in by_rate.items()},
                                 "band": {str(k): v for k, v in by_band.items()},
                                 "cost_aversion": {str(k): v for k, v in by_aim.items()}}}


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assets = common_assets()
    train = load(TRAIN, assets)
    test = load(EVALUATE, assets, sigma_from=train)
    report = {"train": TRAIN, "evaluate": EVALUATE, "assets": len(assets), "efq": EFQ,
              "horizons": list(ORACLE_HORIZONS), "ics": list(ORACLE_ICS), "forecasts": {}}
    train_forecasts, test_forecasts = forecasts(train), forecasts(test)
    for name in test_forecasts:
        chosen = tune(train_forecasts[name], train)
        target = aim(test_forecasts[name], test, 0.0)
        rules = policies(target, test, rate=chosen["rate"], band=chosen["band"])
        rules["aim"] = sf.partial_adjustment(aim(test_forecasts[name], test,
                                                 chosen["cost_aversion"]), chosen["rate"])
        scores = {rule: evaluate(held, test) for rule, held in rules.items()}
        report["forecasts"][name] = {"tuned_on_train": chosen, "test": scores}
        print(f"{name}: tuned rate {chosen['rate']}, band {chosen['band']}, "
              f"cost aversion {chosen['cost_aversion']}")
        for rule, s in scores.items():
            print(f"  {rule:<8} net Sharpe {s['sharpe_net']:8.2f}  daily net "
                  f"{s['sharpe_daily_net']:8.2f} [{s['sharpe_daily_net_low']:.1f}, "
                  f"{s['sharpe_daily_net_high']:.1f}]  turnover {s['mean_turnover']:.3f}  "
                  f"holding {s['holding_decisions']:.1f}  break-even EFQ {s['breakeven_efq']:.2f}")
    report["seconds"] = time.perf_counter() - t0
    (OUT / "dynamic.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"-> {OUT / 'dynamic.json'} ({report['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
