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
oracle (errors independent across decisions), each under seeds 0-4. Books are
held for one decision (five minutes) and scored by the harness at the weights
level. The risk model is a shrunk sample covariance of the previous month's
five-minute returns (for July, July's own: the tuning month has no earlier
month cached, which only affects the tuning). The universe is the names quoted
on at least 95% of July's decisions that are also present in August, so
nothing about August chooses it.

COSTS. Each rule is tuned and scored under two cost settings: ``spread``,
cross-spread fills at EFQ 0.7, and ``fees``, the same plus 2020 fees (the Reg
NMS take-fee cap of 0.003 USD/share and SEC Section 31 at 22.10e-6 of sales),
so a band is not rewarded for trading less than a fee-paying book could. The
rules tuned under ``fees`` are also scored, untuned, under the impact model
(``impact``: $1M, filled at the decision, with a 3 bp cost floor; adv from the
v2 cache). The rate and the cost aversion of ``aim`` are tuned JOINTLY, on
grids that extend past where the earlier optima sat.

The 3 bp floor is the impact model's fourth coefficient, which
``BacktestConfig`` accepts only from the runner fixes on (branch
``fix-runner``, "Impact model: an optional cost floor"), so that branch has to
be merged before this one. On a library without it the config refuses the
floor; the impact block is then skipped, and the report records why
(``impact_skipped``) rather than failing.

``holding_decisions`` is the conventional average holding period,
``2 x gross / turnover`` with turnover the sum of absolute weight changes (a
round trip trades the position twice). ``--skip-open`` drops each day's first
decision (09:30:00, the opening cross) from both months.

    uv run --extra experiments python -m experiments.dynamic [--skip-open] [--seeds N] [--quick]

Writes ``results/dynamic/dynamic.json`` (``dynamic_noopen.json`` with
``--skip-open``).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import functools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import stable_finance as sf
from experiments.month_cache import load_month
from experiments.policy import (
    EFQ,
    HALF_LIFE,
    ORACLE_HORIZONS,
    ORACLE_ICS,
    month_inputs,
)
from experiments.run import _inference, trailing_adv

OUT = Path(__file__).resolve().parent / "results" / "dynamic"
TRAIN, EVALUATE = "2020-07", "2020-08"
GROSS = 2.0
RATES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
BANDS = (0.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)
COST_AVERSIONS = (0.0, 5.0, 20.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0)
SEEDS = (0, 1, 2, 3, 4)
#: 2020 fees, as in e8_costs: the Reg NMS take-fee cap and the SEC Section 31 rate.
FEES = {"fee_per_share": 0.003, "sec_fee_rate": 22.10e-6}
#: The impact variant: $1M filled at the decision, a 3 bp floor on the cost per dollar.
IMPACT = {"cost_model": "impact", "equity": 1e6, "execution_days": 0.0,
          "impact": (0.0, -0.53, 11.21, 3.0)}


def common_assets() -> np.ndarray:
    """Names quoted on at least 95% of the TRAINING decisions and present in both months.

    The evaluation month only has to contain a name for it to be traded; its
    quotes do not choose the universe, so nothing about August selects it.
    """
    train, test = load_month(TRAIN), load_month(EVALUATE)
    quoted = np.isfinite(train["half_spread"]).mean(axis=0) >= 0.95
    keep = set(train["realized"].assets[quoted].astype(str)) & set(
        test["realized"].assets.astype(str))
    return np.asarray(sorted(keep))


def forecasts(month: dict, seed: int) -> dict[str, sf.ForwardReturns]:
    """The two multi-horizon oracles on one month's panel, under one seed."""
    realized = month["realized"]
    panel = sf.ForwardReturns(month["horizon_panel"], realized.decisions, realized.assets,
                              np.asarray(ORACLE_HORIZONS))
    random = sf.oracle_forecast(panel, np.asarray(ORACLE_ICS), noise="random", seed=seed)
    copula = sf.copula_forecast(panel, np.asarray(ORACLE_ICS), half_life=HALF_LIFE,
                                days=month["days"], seed=seed)
    return {"copula": copula, "random": random}


def _drop_open(month: dict) -> dict:
    """The month without each day's first decision, on every per-decision array."""
    days = np.asarray(month["days"]).astype(str)
    keep = np.flatnonzero(np.r_[False, days[1:] == days[:-1]])
    realized, market = month["realized"], month["market"]
    out = dict(month)
    out["realized"] = sf.ForwardReturns(realized.values[keep], realized.decisions[keep],
                                        realized.assets, realized.horizons)
    out["market"] = sf.MarketData({q: v[keep] for q, v in market.columns.items()},
                                  realized.decisions[keep], realized.assets)
    for key in ("one_step", "oracle", "features", "cost", "horizon_panel"):
        out[key] = month[key][keep]
    out["days"] = month["days"][keep]
    return out


def load(month_name: str, assets: np.ndarray, sigma_from: dict | None = None, *,
         skip_open: bool = False) -> dict:
    month = month_inputs(month_name, assets)
    data = load_month(month_name)
    index = {a: i for i, a in enumerate(data["realized"].assets.astype(str))}
    take = [index[a] for a in assets]
    hz = [int(np.flatnonzero(data["realized"].horizons == h)[0]) for h in ORACLE_HORIZONS]
    month["horizon_panel"] = data["realized"].values[:, take][:, :, hz]
    if skip_open:
        month = _drop_open(month)
    source = month if sigma_from is None else sigma_from
    month["sigma"] = sf.estimate_risk("shrunk_sample", source["realized"])[0]
    return month


def with_adv(month: dict, month_name: str, history_name: str) -> dict:
    """The month's market with ``adv`` from the v2 caches, for the impact variant."""
    from experiments.month_cache_v2 import load_month_v2

    evaluation, history = load_month_v2(month_name), load_month_v2(history_name)
    adv = trailing_adv(evaluation, history, month["realized"].assets, 20)
    row = {str(d): i for i, d in enumerate(evaluation["realized"].decisions.astype(str))}
    adv = adv[[row[str(d)] for d in month["realized"].decisions.astype(str)]]
    market = month["market"]
    return {**month, "market": sf.MarketData({**market.columns, "adv": adv},
                                             market.decisions, market.assets)}


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


def impact_unavailable() -> str | None:
    """Why the impact variant cannot be scored with this library, or None when it can.

    The variant's 3 bp floor is a fourth impact coefficient, which only a
    library with the runner fixes accepts; without them ``config_for`` raises
    and this returns the error's message.
    """
    try:
        config_for("impact")
    except ValueError as error:
        return str(error)
    return None


def holding_decisions(mean_gross: float, mean_turnover: float) -> float:
    """The conventional average holding period in decisions, ``2 x gross / turnover``.

    Turnover is the sum of absolute weight changes, and a round trip trades a
    position twice, once in and once out. A book that never trades holds
    forever.
    """
    return float(2 * mean_gross / mean_turnover) if mean_turnover > 0 else float("inf")


def config_for(costs: str) -> sf.BacktestConfig:
    base = sf.DEFAULT_BACKTEST.replace(cost_model="cross", efq=EFQ, annualization="decisions")
    if costs == "spread":
        return base
    if costs == "fees":
        return base.replace(**FEES)
    if costs == "impact":
        return base.replace(efq=1.0, **IMPACT, **FEES)
    raise ValueError(f"unknown cost setting {costs!r}")


def evaluate(held: sf.PortfolioWeights, month: dict, costs: str = "spread") -> dict:
    config = config_for(costs)
    net = sf.run_backtest(held, month["realized"], market=month["market"], days=month["days"],
                          config=config, with_series=True)
    series = net.series
    if config.cost_model == "cross" and series.fee is None:
        breakeven = sf.breakeven_efq(series.gross, series.cost / EFQ)
    elif config.cost_model == "cross":
        breakeven = sf.breakeven_efq(series.gross, (series.cost - series.fee) / EFQ, fees=series.fee)
    else:
        breakeven = float("nan")
    out = {"sharpe_mid": float(net.sharpe_mid), "sharpe_net": float(net.sharpe_net),
           "mean_turnover": float(net.mean_turnover),
           "holding_decisions": holding_decisions(net.mean_gross_exposure, net.mean_turnover),
           "breakeven_efq": float(breakeven), **_inference(series)}
    return out


def policies(target: sf.PortfolioWeights, month: dict, *, rate: float, band: float
             ) -> dict[str, sf.PortfolioWeights]:
    return {"target": target,
            "partial": sf.partial_adjustment(target, rate),
            "band": sf.no_trade_band(target, band, cost=month["cost"])}


def tune(forecast: sf.ForwardReturns, month: dict, costs: str, *, quick: bool = False) -> dict:
    """Each rule's parameters by net Sharpe on the training month, under ``costs``.

    ``partial`` takes the best rate, ``band`` the best multiple of cost, and
    ``aim`` the best (cost aversion, rate) pair over the full grid of both.
    """
    rates, bands, aversions = ((RATES[2::3], BANDS[::3], COST_AVERSIONS[::3]) if quick
                               else (RATES, BANDS, COST_AVERSIONS))
    target = aim(forecast, month, 0.0)
    score = lambda held: evaluate(held, month, costs)["sharpe_net"]
    by_rate = {r: score(sf.partial_adjustment(target, r)) for r in rates}
    by_band = {b: score(sf.no_trade_band(target, b, cost=month["cost"])) for b in bands}
    by_aim = {}
    for k in aversions:
        book = aim(forecast, month, k)
        for r in rates:
            by_aim[(k, r)] = score(sf.partial_adjustment(book, r))
    rate, band = max(by_rate, key=by_rate.get), max(by_band, key=by_band.get)
    aversion, aim_rate = max(by_aim, key=by_aim.get)
    # A parameter tuned onto the edge of its grid is a sign the grid is too
    # narrow; the report records it so a reader can see.
    edges = [name for name, grid, best in (("rate", rates, rate), ("band", bands, band),
                                           ("aim_rate", rates, aim_rate),
                                           ("cost_aversion", aversions, aversion))
             if best in (grid[0], grid[-1]) and best != 0.0]
    return {"rate": rate, "band": band, "cost_aversion": aversion, "aim_rate": aim_rate,
            "at_grid_edge": edges,
            "train_sharpe_net": {"rate": {str(k): v for k, v in by_rate.items()},
                                 "band": {str(k): v for k, v in by_band.items()},
                                 "aim": {f"{k}|{r}": v for (k, r), v in by_aim.items()}}}


def one_seed(seed: int, *, skip_open: bool, quick: bool) -> dict:
    """Tune on July and score on August, for both forecasts and every cost setting.

    The impact block is skipped, with the reason recorded, when the library
    refuses the impact config (see :func:`impact_unavailable`).
    """
    assets = common_assets()
    train = load(TRAIN, assets, skip_open=skip_open)
    test = load(EVALUATE, assets, sigma_from=train, skip_open=skip_open)
    impact_skipped = impact_unavailable()
    test_impact = None if impact_skipped else with_adv(test, EVALUATE, TRAIN)
    train_forecasts, test_forecasts = forecasts(train, seed), forecasts(test, seed)
    out = {}
    for name in test_forecasts:
        out[name] = {}
        for costs in ("spread", "fees"):
            chosen = tune(train_forecasts[name], train, costs, quick=quick)
            target = aim(test_forecasts[name], test, 0.0)
            rules = policies(target, test, rate=chosen["rate"], band=chosen["band"])
            rules["aim"] = sf.partial_adjustment(
                aim(test_forecasts[name], test, chosen["cost_aversion"]), chosen["aim_rate"])
            scores = {rule: evaluate(held, test, costs) for rule, held in rules.items()}
            out[name][costs] = {"tuned_on_train": chosen, "test": scores}
            if costs == "fees" and impact_skipped:
                out[name]["impact"] = {"tuned_on_train": "fees", "skipped": impact_skipped}
            elif costs == "fees":
                out[name]["impact"] = {"tuned_on_train": "fees", "test": {
                    rule: evaluate(held, test_impact, "impact") for rule, held in rules.items()}}
    return {"seed": seed, "assets": len(assets), "forecasts": out}


def summarize(seeds: list[dict]) -> dict:
    """Per forecast, cost setting and rule: the seed mean, sd and range of each score.

    A cost setting that was skipped carries its reason instead of rules.
    """
    out = {}
    first = seeds[0]["forecasts"]
    for name, settings in first.items():
        out[name] = {}
        for costs, block in settings.items():
            if "test" not in block:
                out[name][costs] = {"skipped": block["skipped"]}
                continue
            out[name][costs] = {}
            for rule in block["test"]:
                row = {}
                for metric in ("sharpe_net", "sharpe_daily_net", "sharpe_daily_mid",
                               "mean_turnover", "holding_decisions"):
                    values = np.array([s["forecasts"][name][costs]["test"][rule][metric]
                                       for s in seeds], dtype=float)
                    row[metric] = {"mean": float(np.mean(values)),
                                   "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                   "min": float(np.min(values)), "max": float(np.max(values))}
                row["seeds_net_positive"] = int(sum(
                    s["forecasts"][name][costs]["test"][rule]["sharpe_daily_net"] > 0 for s in seeds))
                out[name][costs][rule] = row
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-open", action="store_true",
                        help="drop each day's first decision (09:30:00) from both months")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="seeds 0..N-1 (default 5)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="coarse grids, for a smoke test")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    # one_seed's options are keyword-only, so they are bound here rather than
    # passed to map as extra positional iterables.
    run = functools.partial(one_seed, skip_open=args.skip_open, quick=args.quick)
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(seeds)))) as pool:
        results = list(pool.map(run, seeds))
    report = {"train": TRAIN, "evaluate": EVALUATE, "assets": results[0]["assets"], "efq": EFQ,
              "horizons": list(ORACLE_HORIZONS), "ics": list(ORACLE_ICS),
              "skip_open": args.skip_open, "quick": args.quick, "seeds": seeds,
              "fees": FEES, "impact": IMPACT, "impact_skipped": impact_unavailable(),
              "summary": summarize(results), "by_seed": results}
    for name, settings in report["summary"].items():
        for costs, rules in settings.items():
            if "skipped" in rules:
                print(f"{name}, {costs}: skipped ({rules['skipped']})")
                continue
            print(f"{name}, {costs}:")
            for rule, s in rules.items():
                d = s["sharpe_daily_net"]
                print(f"  {rule:<8} daily net {d['mean']:8.2f} ± {d['sd']:.2f} "
                      f"[{d['min']:.1f}, {d['max']:.1f}]  {s['seeds_net_positive']}/{len(seeds)} > 0  "
                      f"holding {s['holding_decisions']['mean']:.1f}")
    report["seconds"] = time.perf_counter() - t0
    name = "dynamic_noopen.json" if args.skip_open else "dynamic.json"
    (OUT / name).write_text(json.dumps(report, indent=2, default=float))
    print(f"-> {OUT / name} ({report['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
