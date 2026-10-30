"""Native single/joint original-objective backtests on supplied analytical inputs.

No training, dataset download or preserved execution checkout is required.
Run: PYTHONPATH=src python examples/native_return_risk_backtest.py
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import stable_finance as sf


DEFAULT_OUTPUT = Path.home()/".cache/stable-finance/peer-multihorizon-extension/native-backtest-example"


def analytical_inputs():
    """Current forecasts and risk are external inputs, not fitted predictions."""
    assets = np.asarray(["A", "B", "C", "D"])
    decisions = np.asarray([0, 1, 5, 10])
    times = np.arange(18)
    marks = 100.*np.exp(times[:, None]*np.array([.0002, -.0001, .00005, 0.])[None])
    bid, ask = marks*(1-.0001), marks*(1+.0001)
    bid[2, 0] = ask[2, 0] = np.nan  # The first A order misses its arrival fill.
    bid[5, 3] = ask[5, 3] = np.nan  # Current missing quote freezes an extra D trade.
    prices = sf.PricePanel(marks, times, assets)
    market = sf.MarketData({"mid":marks, "bid_price":bid, "ask_price":ask}, times, assets)
    first = np.array([[.003, -.003, .001, -.001], [.0024, -.0024, .001, -.001],
                      [-.002, .002, .001, -.001], [.002, -.002, -.001, .001]])
    future_increment = np.array([[-.002, .002, -.001, .001], [-.001, .001, -.001, .001],
                                 [.002, -.002, -.001, .001], [-.001, .001, .001, -.001]])
    cumulative = np.stack([first, first+future_increment], axis=2)
    eye = np.eye(4)
    covariance = np.block([[.02*eye, .008*eye], [.008*eye, .03*eye]])
    beta = np.array([1.2, .8, 1.1, .9])
    return dict(prices=prices, market=market, decisions=decisions, assets=assets,
        cumulative=cumulative, covariance=covariance, beta=beta)


def json_safe(value):
    if isinstance(value, dict):
        return {key:json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_example(output_dir=DEFAULT_OUTPUT):
    data = analytical_inputs()
    config = sf.BacktestConfig(cost_model="cross", efq=.7, annualization="decisions")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports, arrays = {}, {}
    for label, k in (("single", 1), ("joint", 2)):
        forecast = sf.JointReturnPanel(data["cumulative"][:, :, :k], data["decisions"],
            data["assets"], np.asarray([5, 10])[:k], entry_delay_seconds=2)
        risk = sf.JointRiskModel(data["covariance"][:4*k, :4*k], data["assets"],
            forecast.horizons, information_cutoff=-1, entry_delay_seconds=2)
        report = sf.run_backtest(forecast, data["prices"], market=data["market"],
            objective=sf.ReturnRiskBacktestSpec(risk=risk, risk_aversion=5.,
                constraints=sf.JointPortfolioConstraints(), beta=data["beta"],
                max_abs_beta=.10, beta_available_at=-1,
                stage_counts=np.minimum([2, 2, 2, 1], k), strategy_end=15),
            execution_state=sf.BacktestExecutionState(equity=100_000.),
            config=config, with_series=True)
        trace, ledger = report.objective_trace, report.objective_trace.ledger
        assert len(trace.certificates) == len(data["decisions"])
        assert np.any(ledger.pending_decision_weights[1] != 0)
        assert ledger.unfilled_shares[2, 0] != 0
        assert ledger.filled_shares[2, 0] == 0
        assert ledger.liquidated
        np.testing.assert_allclose(ledger.nav[1:], ledger.nav[:-1]*(1+report.series.net),
                                   rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(ledger.cash+(ledger.shares*data["prices"].values).sum(axis=1),
                                   ledger.posttrade_nav, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(report.series.gross-report.series.cost, report.series.net, atol=1e-14)
        if any(row.max_constraint_violation > 1e-8 for row in trace.certificates):
            raise AssertionError("every reference-stage plan must satisfy its constraints")
        arrays[label+"_planned"] = ledger.planned_weights
        arrays[label+"_actual"] = ledger.actual_decision_weights
        arrays[label+"_pending"] = ledger.pending_decision_weights
        arrays[label+"_reference_plans"] = trace.reference_plans
        arrays[label+"_nav"] = ledger.nav
        reports[label] = dict(report=report.as_dict(),
            total_net_return=ledger.total_return,
            planned_first=ledger.planned_weights[0].tolist(),
            pending_feedback_verified=True, missing_arrival_fill_verified=True,
            physical_nav_reconciled=True, terminal_liquidation_verified=True,
            all_reference_constraints_passed=True, plans=len(trace.certificates),
            notes=list(trace.notes))
    if np.allclose(arrays["single_planned"][0], arrays["joint_planned"][0], atol=1e-8):
        raise AssertionError("the example must exercise cross-interval-risk coupling")
    path = output/"native_traces.npz"
    np.savez_compressed(path, **arrays, times=data["prices"].times,
        assets=data["assets"], decisions=data["decisions"])
    result = dict(status="passed", purpose="Native integration mechanics, not an empirical trading claim.",
        predictor_fits=0, dataset_downloads=0, preserved_checkout_required=False,
        single_joint_current_plans_differ=True, reports=reports,
        artifact_path=str(path), artifact_sha256=sha256(path.read_bytes()).hexdigest())
    (output/"result.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_example(args.output_dir)
    print(json.dumps(dict(status=result["status"], artifact_path=result["artifact_path"])), flush=True)


if __name__ == "__main__":
    main()
