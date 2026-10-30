"""Diagnose completed single- and multi-horizon portfolio weights.

This analytical example supplies forecasts, factor exposures and covariance.
It trains no predictor, estimates no factors and changes no target weights.
Constraint failures are intentional diagnostics, not a failed example run.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from stable_finance import (
    ForwardReturns, JointPortfolioConfig, PortfolioExposureLimits,
    combine_horizon_targets, construct_joint_portfolio, mean_variance_target,
    portfolio_exposure_report,
)


DEFAULT_OUTPUT = Path.home()/".cache/stable-finance/peer-multihorizon-extension/exposure-demo"


def supplied_inputs():
    """A supplied alpha neutral to the intercept, beta and both industries."""
    alpha = np.array([.002, -.002, -.002, .002])
    beta = np.array([.5, 1., 1.5, 2.])
    industry = np.array(["Tech", "Tech", "Finance", "Finance"])
    values = np.tile(np.column_stack([alpha, -alpha])[None], (3, 1, 1))
    forecast = ForwardReturns(values, [0, 300, 600], ["A", "B", "C", "D"], [300, 900])
    covariance = np.stack([np.diag([.01, .02, .04, .08]), np.diag([.02, .04, .08, .16])])
    return forecast, covariance, beta, industry


def _first_metrics(report):
    """One labeled row for the compact human-readable example table."""
    return dict(gross=float(report.gross[0, 0]), net=float(report.net[0, 0]),
        beta=float(report.beta_exposure[0, 0]), hhi=float(report.concentration_hhi[0, 0]),
        effective_names=float(report.effective_names[0, 0]),
        max_abs_weight=float(report.max_abs_weight[0, 0]),
        constraint_status=str(report.constraint_status[0, 0]))


def run_example(output_dir):
    forecast, covariance, beta, industry = supplied_inputs()
    alpha = forecast.values[0, :, 0]
    np.testing.assert_allclose(alpha.sum(), 0., atol=1e-16)
    np.testing.assert_allclose(alpha@beta, 0., atol=1e-16)
    for label in np.unique(industry):
        np.testing.assert_allclose(alpha[industry == label].sum(), 0., atol=1e-16)

    # Reuse the peer's existing constructor and our target-only mixer.
    components = mean_variance_target(forecast, covariance, risk_aversion=2.)
    single = combine_horizon_targets(components, capital_weights=[1., 0.],
                                    target_horizon_seconds=300)
    mixed = combine_horizon_targets(components, capital_weights=[.5, .5],
                                   target_horizon_seconds=300)
    limits = PortfolioExposureLimits(gross_limit=.2, max_weight=.08,
        max_abs_net=.01, max_abs_beta=.01, max_abs_industry=.04, max_hhi=.35)
    books = {"per_horizon_targets": components, "single_horizon_target": single,
             "fixed_multi_horizon_target": mixed}
    snapshots = {name:weights.values.copy() for name, weights in books.items()}
    reports = {name:portfolio_exposure_report(weights, beta=beta, industry=industry,
                                             limits=limits) for name, weights in books.items()}

    # Alpha neutrality does not survive arbitrary risk-based weighting.
    np.testing.assert_allclose(single.values[0, :, 0], [.1, -.05, -.025, .0125])
    np.testing.assert_allclose(mixed.values, .25*single.values)
    first = reports["single_horizon_target"]
    np.testing.assert_allclose(first.gross, .1875)
    np.testing.assert_allclose(first.net, .0375)
    np.testing.assert_allclose(first.beta_exposure, -.0125)
    np.testing.assert_allclose(first.concentration_hhi, 17/45)
    np.testing.assert_allclose(first.effective_names, 45/17)
    expected_sectors = {"Tech":.05, "Finance":-.0125}
    for index, label in enumerate(first.industry_labels):
        np.testing.assert_allclose(first.industry_net[:, :, index], expected_sectors[str(label)])
    # Shrinking the book changes absolute exposures, not gross-normalized HHI.
    np.testing.assert_allclose(reports["fixed_multi_horizon_target"].concentration_hhi,
                               first.concentration_hhi)
    for name, weights in books.items():
        np.testing.assert_array_equal(weights.values, snapshots[name])

    # The joint result's first-stage output uses the same native contract.
    try:
        from examples.joint_multi_horizon import supplied_problem
    except ModuleNotFoundError:
        from joint_multi_horizon import supplied_problem
    joint_forecast, joint_risk = supplied_problem()
    joint = construct_joint_portfolio(joint_forecast, risk=joint_risk,
        config=JointPortfolioConfig(risk_aversion=2.))
    joint_before = joint.current_targets.values.copy()
    joint_report = portfolio_exposure_report(joint.current_targets,
        beta=[1.2, .8], industry=["Tech", "Finance"], limits=limits)
    np.testing.assert_allclose(joint_report.net, 0., atol=1e-14)
    np.testing.assert_allclose(joint_report.beta_exposure, .4*.04285714285714286)
    np.testing.assert_array_equal(joint.current_targets.values, joint_before)
    reports["joint_current_target"] = joint_report

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output/"external_inputs.npz"
    np.savez_compressed(fixture, forecasts=forecast.values,
        decisions=forecast.decisions, assets=forecast.assets, horizons=forecast.horizons,
        covariance=covariance, beta=beta, industry=industry)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/"src/stable_finance/evaluation/portfolio/exposure.py",
        root/"src/stable_finance/evaluation/portfolio/construction.py",
        root/"src/stable_finance/evaluation/portfolio/multihorizon.py",
        root/"src/stable_finance/evaluation/portfolio/joint.py",
        root/"src/stable_finance/evaluation/portfolio/joint_risk.py",
        root/"examples/joint_multi_horizon.py"]
    result = dict(status="passed", predictor_fits=0, factor_estimates=0,
        purpose="Diagnostics of supplied target weights; not observed post-fill holdings or empirical performance.",
        alpha_is_dollar_beta_and_industry_neutral=True, weights_unchanged=True,
        diagnostic_checks_passed=True,
        reports={name:report.as_dict() for name, report in reports.items()},
        first_rows={name:_first_metrics(report) for name, report in reports.items()},
        input_sha256=sha256(fixture.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(root)):sha256(path.read_bytes()).hexdigest()
                       for path in sources})
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Portfolio exposure diagnostics", "", "Diagnostic checks: passed.", "",
        "The supplied alpha is dollar, beta and industry neutral. Native mean-variance",
        "weighting nevertheless produces nonzero portfolio exposures. The report",
        "measures the completed weights and never repairs or normalizes them.", "",
        "| Target | Gross | Net | Beta | Max name | HHI | Effective names | Constraints |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, row in result["first_rows"].items():
        lines.append(f"| {name} | {row['gross']:.6f} | {row['net']:.6f} | {row['beta']:.6f} | {row['max_abs_weight']:.6f} | {row['hhi']:.6f} | {row['effective_names']:.6f} | {row['constraint_status']} |")
    lines += ["", "Constraint failures are intentional and remain visible. The single-target",
        "industry net exposures are Tech +0.05 and Finance -0.0125. Equal mixing",
        "reduces absolute positions to one quarter of the single target, while HHI",
        "is unchanged. The joint current target is dollar neutral but beta exposed.", "",
        "The JSON includes every decision/horizon, industry net/gross exposure,",
        "factor coverage and requested constraint residual. No predictor was trained",
        "and no factor was estimated. These are target-weight diagnostics; supply",
        "observed post-fill weights to diagnose actual executed holdings.", ""]
    (output/"exposure_report.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_example(args.output_dir)
    print(json.dumps({key:result[key] for key in ("status", "weights_unchanged", "first_rows")}, indent=2))


if __name__ == "__main__":
    main()
