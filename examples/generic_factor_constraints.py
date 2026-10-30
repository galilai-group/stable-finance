"""One declared factor budget across single/joint targets, trades and replay.

Run: PYTHONPATH=src python examples/generic_factor_constraints.py
All inputs are deterministic and supplied; the example fits no model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import stable_finance as sf


DEFAULT_OUTPUT = Path.home()/".cache/stable-finance/peer-multihorizon-extension/generic-factor-example"


def analytical_inputs():
    assets = np.asarray(list("ABCDEF"))
    decisions, horizons = np.asarray([0, 5, 10]), np.asarray([5, 10])
    first = np.asarray([[.003, -.003, .002, -.002, .001, -.001],
                        [-.002, .002, .001, -.001, .003, -.003],
                        [.001, -.001, -.003, .003, .002, -.002]])
    later = -.6*first
    cumulative = np.stack([first, first+later], axis=2)
    eye = np.eye(len(assets))
    covariance = np.block([[.01*eye, .004*eye], [.004*eye, .02*eye]])
    times = np.arange(17)
    marks = 100.*np.exp(times[:, None]*np.asarray([.0003, -.0002, .0001, -.0001, .0002, 0.]))
    prices = sf.PricePanel(marks, times, assets)
    market = sf.MarketData({"mid":marks, "bid_price":marks*(1-.0001),
                           "ask_price":marks*(1+.0001)}, times, assets)
    # Group membership and style are synthetic inputs, not inferred industries.
    loadings = np.column_stack([[1.2, .8, 1.1, .9, 1.3, .7],
                                [1., 1., 0., 0., 1., 0.],
                                [-1., 1., .5, -.5, .25, -.25]])
    factors = sf.FactorExposureConstraints(loadings, [-.01, 0., -.02], [.01, 0., .02],
        assets, ["proxy_beta", "synthetic_group", "style"], available_at=-1)
    return assets, decisions, horizons, cumulative, covariance, prices, market, factors


def run_example(output_dir=DEFAULT_OUTPUT):
    assets, decisions, endpoints, cumulative, covariance, prices, market, factors = analytical_inputs()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results, arrays = {}, {}
    config = sf.BacktestConfig(cost_model="cross", efq=.7, annualization="decisions")
    for label, k in (("single", 1), ("joint", 2)):
        forecast = sf.JointReturnPanel(cumulative[:, :, :k], decisions, assets, endpoints[:k],
            entry_delay_seconds=1)
        risk = sf.JointRiskModel(covariance[:6*k, :6*k], assets, forecast.horizons,
            information_cutoff=-1, entry_delay_seconds=1)
        target = sf.construct_joint_portfolio(forecast, risk=risk, factor_constraints=factors)
        target_report = sf.factor_exposure_report(target.current_targets, factors)
        np.testing.assert_array_equal(target_report.status, "pass")
        assert np.all(target.factor_lower_slack >= -1e-8)
        assert np.all(target.factor_upper_slack >= -1e-8)
        report = sf.run_backtest(forecast, prices, market=market,
            objective=sf.ReturnRiskBacktestSpec(risk, factor_constraints=factors,
                stage_counts=np.minimum([2, 2, 1], k), strategy_end=15),
            execution_state=sf.BacktestExecutionState(equity=100_000.),
            config=config, with_series=True)
        trace = report.objective_trace
        planned = sf.PortfolioWeights(trace.ledger.planned_weights[:, :, None],
            decisions, assets, endpoints[:1])
        held = sf.PortfolioWeights(trace.ledger.actual_decision_weights[:, :, None],
            decisions, assets, endpoints[:1])
        planned_report = sf.factor_exposure_report(planned, factors)
        held_report = sf.factor_exposure_report(held, factors)
        np.testing.assert_array_equal(planned_report.status, "pass")
        for certificate in trace.certificates:
            assert certificate.status == "solved"
            assert certificate.max_constraint_violation <= 1e-8
            assert np.all(certificate.factor_lower_slack >= -1e-8)
            assert np.all(certificate.factor_upper_slack >= -1e-8)
        np.testing.assert_allclose(report.series.gross-report.series.cost, report.series.net, atol=1e-14)
        results[label] = dict(targets_passed=True, all_reference_stages_passed=True,
            planned_exposures=planned_report.exposures.tolist(),
            actual_decision_exposures=held_report.exposures.tolist(),
            actual_decision_violations=held_report.violations.tolist(),
            actual_decision_failures=int(np.count_nonzero(held_report.status == "fail")),
            total_net_return=float(trace.ledger.total_return),
            note="Actual pretrade holdings can drift or await execution; future reference stages are not orders.")
        arrays[label+"_targets"] = target.current_targets.values
        arrays[label+"_plans"] = trace.reference_plans
        arrays[label+"_actual"] = trace.ledger.actual_decision_weights
        arrays[label+"_nav"] = trace.ledger.nav
    # Coupling changes frictionless targets. Costs may rationally leave a
    # current executed book in cash; no trading is required by this example.
    assert not np.allclose(arrays["single_targets"][0], arrays["joint_targets"][0], atol=1e-8)
    # The old target-tracking policy consumes the same declared matrix/budget.
    tracking = sf.solve_constrained_rebalance(arrays["single_targets"][0, :, 0],
        covariance[:6, :6], previous=np.zeros(6), cost=.00007,
        factor_loadings=factors.loadings, factor_lower=factors.lower,
        factor_upper=factors.upper, factor_names=factors.factor_names)
    assert tracking.status == "solved"
    assert np.all(tracking.factor_lower_slack >= -1e-8)
    assert np.all(tracking.factor_upper_slack >= -1e-8)
    np.savez_compressed(output/"traces.npz", **arrays, assets=assets,
        decisions=decisions, factor_names=factors.factor_names, factor_loadings=factors.loadings)
    result = dict(status="passed", factor_names=factors.factor_names.tolist(),
        lower=factors.lower.tolist(), upper=factors.upper.tolist(),
        source="Deterministic synthetic inputs; no empirical profitability claim.",
        reports=results, tracking_constraints_passed=True,
        cross_interval_coupling_verified=True)
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_example(args.output_dir)
    print(json.dumps(dict(status=result["status"], output_dir=str(args.output_dir))), flush=True)


if __name__ == "__main__":
    main()
