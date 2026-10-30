"""Add a multi-horizon target mix to the existing portfolio pipeline.

Run: python examples/multi_horizon.py --output-dir /absolute/local/output

The external forecasts are fixed analytical inputs. Existing portfolio
construction and backtesting are reused; no predictor is trained and no
parameters are selected. One-step outcomes score the single net target,
not the overlapping source forecast horizons.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from stable_finance import (
    BacktestConfig, ForwardReturns, PortfolioExposureLimits, combine_horizon_targets,
    mean_variance_target, portfolio_exposure_report, run_backtest,
)


DEFAULT_OUTPUT = Path.home()/".cache/stable-finance/peer-multihorizon-extension/example"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def external_inputs():
    """Two fixed horizon forecasts with exactly opposite native targets."""
    decisions, assets = np.arange(4)*300, np.asarray(["A", "B"])
    values = np.tile([[[.004, -.004], [-.004, .004]]], (4, 1, 1))
    forecasts = ForwardReturns(values, decisions, assets, [300, 900])
    covariance = np.tile(np.eye(2)[None]*.04, (2, 1, 1))
    # Each row covers its own next 300 seconds. These are separate evaluation
    # inputs and never enter construction or the capital allocation.
    realized = ForwardReturns(np.array([
        [.002, -.001], [-.001, .001], [.003, -.002], [.001, -.001],
    ])[:, :, None], decisions, assets, [300])
    return forecasts, covariance, realized, np.full((4, 2), .0002)


def run_example(output_dir):
    output = Path(output_dir)
    forecasts, covariance, realized, half_spread = external_inputs()
    components = mean_variance_target(forecasts, covariance, risk_aversion=2.)
    # Explicit target mix only: no vintage schedule or holding policy is
    # inferred from the source horizons.
    mixed = combine_horizon_targets(components, capital_weights=[.5, .5],
                                    target_horizon_seconds=300)
    control = combine_horizon_targets(components, capital_weights=[1., 0.],
                                      target_horizon_seconds=300)
    config = BacktestConfig(cost_model="cross", efq=.7,
                           rebalance="every_decision", annualization="decisions")
    reports = {
        "equal_capital_mix": run_backtest(mixed, realized,
            half_spread=half_spread, config=config, with_series=True),
        "single_horizon_control": run_backtest(control, realized,
            half_spread=half_spread, config=config, with_series=True),
    }
    limits = PortfolioExposureLimits(gross_limit=1., max_weight=.1,
        max_abs_net=0., max_abs_beta=.01, max_abs_industry=.03, max_hhi=.4)
    # Diagnose the completed target weights, not the forecast used to size
    # them. The zero net book must also have zero beta/industry exposure.
    exposures = {name: portfolio_exposure_report(book, beta=[1.2, .8],
        industry=["Technology", "Finance"], limits=limits).as_dict()
        for name, book in (("components", components),
                           ("equal_capital_mix", mixed),
                           ("single_horizon_control", control))}
    np.testing.assert_allclose(mixed.values, 0., atol=1e-14)
    assert reports["equal_capital_mix"].mean_cost_bps == 0.
    assert reports["equal_capital_mix"].mean_turnover == 0.
    assert reports["single_horizon_control"].mean_cost_bps > 0.
    for name, targets in (("equal_capital_mix", mixed),
                          ("single_horizon_control", control)):
        expected = (targets.values[:, :, 0]*realized.values[:, :, 0]).sum(axis=1)
        np.testing.assert_allclose(reports[name].series.gross, expected, atol=1e-14)

    output.mkdir(parents=True, exist_ok=True)
    fixture = output/"external_inputs.npz"
    np.savez_compressed(fixture, forecasts=forecasts.values,
        decisions=forecasts.decisions, assets=forecasts.assets,
        source_horizons=forecasts.horizons, covariance=covariance,
        realized_one_step=realized.values, target_horizon=realized.horizons,
        half_spread=half_spread)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"src/stable_finance/evaluation/portfolio/multihorizon.py",
             root/"src/stable_finance/evaluation/portfolio/exposure.py",
             root/"src/stable_finance/evaluation/portfolio/construction.py",
             root/"src/stable_finance/evaluation/backtest.py"]
    result = {
        "status": "passed",
        "purpose": "Native pipeline integration, not empirical strategy performance.",
        "predictor_fits": 0,
        "source_horizons_seconds": forecasts.horizons.tolist(),
        "evaluation_interval_seconds": 300,
        "first_component_targets": components.values[0].tolist(),
        "first_net_target": mixed.values[0, :, 0].tolist(),
        "target_exposures": exposures,
        "reports": {name: {
            "mean_gross_bps": report.mean_gross_return_bps,
            "mean_cost_bps": report.mean_cost_bps,
            "mean_turnover": report.mean_turnover,
            "gross_series": report.series.gross.tolist(),
            "net_series": report.series.net.tolist(),
        } for name, report in reports.items()},
        "input_sha256": file_hash(fixture),
        "source_sha256": {str(path.relative_to(root)): file_hash(path) for path in paths},
    }
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/"result.txt").write_text(
        "Multi-horizon extension: passed.\n\n"
        "Existing mean_variance_target constructed both horizon portfolios.\n"
        "The new helper combined them before existing run_backtest execution.\n"
        "Equal and opposite targets cancelled: zero net exposure, turnover and cost.\n"
        "The single-horizon control retained exposure and paid the native EFQ cost.\n"
        "Gross P&L matched the net target times nonoverlapping one-step outcomes.\n\n"
        "Final target beta, industry, concentration and constraint diagnostics are\n"
        "included for the component books, single horizon and combined net book.\n\n"
        "No predictor was fitted. These analytical inputs establish integration\n"
        "and accounting identities, not alpha or investment performance.\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_example(args.output_dir)
    print(json.dumps({"status": result["status"], "predictor_fits": result["predictor_fits"],
        "first_net_target": result["first_net_target"], "reports": result["reports"]}, indent=2))


if __name__ == "__main__":
    main()
