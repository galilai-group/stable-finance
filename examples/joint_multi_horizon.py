"""Optional joint target alongside native single and fixed-mix portfolios.

Supplied analytical forecasts test algebra and the existing weights injection
point. This example trains nothing and makes no empirical performance claim.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from stable_finance import (
    BacktestConfig, ForwardReturns, JointPortfolioConfig, JointReturnPanel,
    PortfolioExposureLimits,
    JointRiskModel, combine_horizon_targets, construct_joint_portfolio,
    mean_variance_target, portfolio_exposure_report, run_backtest,
)


def supplied_problem(second=.006, cross=.01):
    spread = np.array([[.04, cross], [cross, .09]])
    contrast, common = np.array([1., -1.]), np.ones(2)
    covariance = (np.kron(spread, np.outer(contrast, contrast)/4)
                  + np.kron(np.eye(2)*.02, np.outer(common, common)/4))
    increments = np.array([[.004, -.004], [second, -second]])/2
    values = np.tile(np.cumsum(increments, axis=0).T[None], (3, 1, 1))
    forecast = JointReturnPanel(values, [0, 300, 600], ["A", "B"], [300, 900])
    risk = JointRiskModel(covariance, forecast.assets, forecast.horizons, -1)
    return forecast, risk


def run_example(output_dir):
    forecast, risk = supplied_problem()
    config = JointPortfolioConfig(risk_aversion=2.)
    joint = construct_joint_portfolio(forecast, risk=risk, config=config)
    changed, _ = supplied_problem(second=-.006)
    changed_joint = construct_joint_portfolio(changed, risk=risk, config=config)
    no_cross_forecast, no_cross_risk = supplied_problem(cross=0.)
    independent = construct_joint_portfolio(no_cross_forecast,
        risk=no_cross_risk, config=config)
    changed_independent = construct_joint_portfolio(changed,
        risk=no_cross_risk, config=config)
    expected = np.linalg.solve([[.04, .01], [.01, .09]], [.004, .006])/2
    np.testing.assert_allclose(joint.current_targets.values[:, 0, 0], expected[0])
    np.testing.assert_allclose(changed_joint.current_targets.values[:, 0, 0], .06)
    np.testing.assert_allclose(independent.current_targets.values,
                               changed_independent.current_targets.values)

    native = ForwardReturns(forecast.values, forecast.decisions,
                            forecast.assets, forecast.horizons)
    # Exact cumulative covariance from the same supplied increment model.
    cumulative_risk = np.stack([risk.blocks[:h+1, :, :h+1, :].sum(axis=(0, 2))
                               for h in range(2)])
    components = mean_variance_target(native, cumulative_risk, risk_aversion=2.)
    books = {
        "native_single": combine_horizon_targets(components,
            capital_weights=[1., 0.], target_horizon_seconds=300),
        "native_fixed_mix": combine_horizon_targets(components,
            capital_weights=[.5, .5], target_horizon_seconds=300),
        "joint_full": joint.current_targets,
        "joint_no_cross": independent.current_targets,
    }
    realized = ForwardReturns(np.array([[.002, -.001], [-.001, .001], [.003, -.002]])[:, :, None],
        forecast.decisions, forecast.assets, [300])
    reports = {}
    limits = PortfolioExposureLimits(gross_limit=1., max_weight=.1,
        max_abs_net=0., max_abs_beta=.01, max_abs_industry=.03, max_hhi=.4)
    for name, target in books.items():
        report = run_backtest(target, realized, half_spread=np.full((3, 2), .0002),
            config=BacktestConfig(cost_model="cross", efq=.7,
                rebalance="every_decision", annualization="decisions"), with_series=True)
        expected_gross = (target.values[:, :, 0]*realized.values[:, :, 0]).sum(axis=1)
        np.testing.assert_allclose(report.series.gross, expected_gross)
        reports[name] = dict(first_target=target.values[0, :, 0].tolist(),
            mean_cost_bps=report.mean_cost_bps,
            gross_series=report.series.gross.tolist(),
            target_exposure=portfolio_exposure_report(target, beta=[1.2, .8],
                industry=["Technology", "Finance"], limits=limits).as_dict())

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output/"external_inputs.npz"
    np.savez_compressed(fixture, forecasts=forecast.values, decisions=forecast.decisions,
        assets=forecast.assets, horizons=forecast.horizons, covariance=risk.covariance,
        realized_one_step=realized.values)
    result = dict(status="passed", predictor_fits=0,
        purpose="Analytical and native-interface verification only.",
        current_long_weight=float(joint.current_targets.values[0, 0, 0]),
        after_later_forecast_change=float(changed_joint.current_targets.values[0, 0, 0]),
        block_diagonal_decoupling=True, only_first_stage_scored=True,
        reports=reports, input_sha256=sha256(fixture.read_bytes()).hexdigest())
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
        default=Path.home()/".cache/stable-finance/peer-multihorizon-extension/joint-demo")
    args = parser.parse_args(argv)
    print(json.dumps(run_example(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
