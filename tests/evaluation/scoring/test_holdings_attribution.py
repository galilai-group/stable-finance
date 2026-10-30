"""Physical-share accounting identities and paired exposure attribution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stable_finance.evaluation.scoring.holdings_attribution import (
    execution_attribution, paired_execution_attribution,
)


def ledger_case(fills=None, *, times=None, beta_targets=None, spread=.1, rate=.002):
    """A tiny cash/share ledger, including cost-bearing terminal liquidation."""
    times = np.array([0., 5., 10., 15.] if times is None else times)
    prices = np.array([[10., 20.], [11., 19.], [12., 18.], [13., 17.]])
    filled = np.array([[2., -1.], [-1., 0.], [0., 0.], [-1., 1.]] if fills is None else fills)
    rates = np.full_like(prices, rate)
    paid = np.where(filled != 0, prices+spread*np.sign(filled), np.nan)
    fill_dollars = filled*np.nan_to_num(paid)
    commission = (rates*np.abs(fill_dollars)).sum(axis=1)
    slippage = (filled*np.where(filled != 0, paid-prices, 0.)).sum(axis=1)
    shares, cash = np.cumsum(filled, axis=0), 100.-np.cumsum(fill_dollars.sum(axis=1)+commission)
    post_nav = cash+(shares*prices).sum(axis=1)
    previous_shares = np.vstack([np.zeros(2), shares[:-1]])
    previous_cash = np.r_[100., cash[:-1]]
    pre_nav = previous_cash+(previous_shares*prices).sum(axis=1)
    gross = (shares[:-1]*np.diff(prices, axis=0)).sum(axis=1)/pre_nav[:-1]
    costs = (slippage[:-1]+commission[:-1])/pre_nav[:-1]
    costs[-1] += (slippage[-1]+commission[-1])/pre_nav[-2]
    net = gross-costs
    nav = pre_nav.copy()
    nav[-1] = post_nav[-1]
    records = []
    for i in range(4):
        if np.any(filled[i]):
            records.append(SimpleNamespace(arrival_time=times[i], decision_time=times[i],
                terminal=i == 3, filled_shares=filled[i].copy()))
    targets = np.array([[.1, 0.], [.05, .025]] if beta_targets is None else beta_targets)
    ledger = SimpleNamespace(times=times, assets=np.array(["A", "B"]), initial_equity=100.,
        shares=shares, filled_shares=filled, fill_prices=paid, commission_rates=rates,
        trade_notional=fill_dollars, posttrade_weights=shares*prices/post_nav[:, None],
        posttrade_nav=post_nav, pretrade_nav=pre_nav, nav=nav,
        arrival_slippage=slippage, commissions=commission, gross_returns=gross,
        net_returns=net, total_return=post_nav[-1]/100.-1.,
        target_decisions=times[:2], target_weights=targets, order_records=tuple(records))
    return ledger, prices


def test_currency_returns_and_turnover_reconcile_per_asset():
    ledger, prices = ledger_case()
    report = execution_attribution(ledger, prices, beta=[1., 2.])
    rows, total = report["asset_rows"], report["totals"]
    # A holds two shares for the first interval, then one for two intervals.
    # B remains short one share through all three falling-price intervals.
    np.testing.assert_allclose([r["gross_pnl"] for r in rows], [4., 3.])
    np.testing.assert_allclose(total["arrival_slippage"], .6)
    np.testing.assert_allclose(total["commission"], np.abs(ledger.trade_notional).sum()*.002)
    np.testing.assert_allclose(total["net_return"], ledger.total_return)
    np.testing.assert_allclose(total["turnover"], np.abs(ledger.trade_notional).sum()/100.)
    assert max(report["reconciliation_max_abs_errors"].values()) < 1e-12
    json.dumps(report, allow_nan=False)


def test_terminal_fills_charged_once_and_excluded_from_reversals():
    ledger, prices = ledger_case()
    report = execution_attribution(ledger, prices, beta=[1., 2.], beta_bound=.1)
    a, b = report["asset_rows"]
    terminal = (ledger.arrival_slippage[-1]+ledger.commissions[-1])/100.
    assert report["totals"]["terminal_cost_return"] == pytest.approx(terminal)
    assert a["trade_sign_reversals"] == 1
    assert b["trade_sign_reversals"] == 0
    assert report["totals"]["terminal_order_events"] == 1
    assert report["totals"]["boundary_target_decisions"] == 2
    assert report["totals"]["boundary_fill_events"] == 3
    assert (report["totals"]["boundary_cost_return"]+terminal
            == pytest.approx(report["totals"]["cost_return"]))


def test_beta_means_weight_actual_intervals_and_exclude_terminal_flat_row():
    ledger, prices = ledger_case(times=[0., 2., 5., 15.])
    report = execution_attribution(ledger, prices, beta=[1., 2.])
    exposure = ledger.posttrade_weights[:-1]@np.array([1., 2.])
    expected = np.average(np.abs(exposure), weights=[2., 3., 10.])
    assert report["totals"]["beta_abs_mean"] == pytest.approx(expected)
    assert sum(r["beta_abs_mean_contribution"] for r in report["asset_rows"]) == pytest.approx(expected)
    assert report["duration_seconds"] == 15.


def test_beta_optional_and_missing_bound_rejected():
    ledger, prices = ledger_case()
    report = execution_attribution(ledger, prices)
    assert report["totals"]["beta_mean"] is None
    assert report["asset_rows"][0]["beta_abs_mean_contribution"] is None
    assert report["totals"]["boundary_cost_return"] == 0.
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="requires supplied beta"):
        execution_attribution(ledger, prices, beta_bound=.1)


def test_paired_deltas_and_beta_reduction_are_additive():
    reference, prices = ledger_case()
    candidate, _ = ledger_case(fills=np.array([[1., -1.], [0., 0.], [0., 0.], [-1., 1.]]))
    report = paired_execution_attribution(reference, candidate, prices, beta=[1., 2.])
    for key in ("net_return", "gross_return", "cost_return", "turnover"):
        assert sum(row[key] for row in report["asset_rows"]) == pytest.approx(report["totals"][key])
    allocation = sum(r["beta_abs_mean_reduction_contribution"] for r in report["asset_rows"])
    assert allocation == pytest.approx(report["totals"]["beta_abs_mean_reduction"])
    # Candidate has a larger absolute negative beta despite lower A exposure.
    assert report["totals"]["beta_abs_mean_reduction"] < 0
    assert report["beta_allocation_max_abs_error"] < 1e-15
    json.dumps(report, allow_nan=False)


def test_beta_secant_handles_sign_crossing_and_identical_total_exposures():
    reference, prices = ledger_case(fills=np.array([[2., 0.], [0., 0.], [0., 0.], [-2., 0.]]), spread=0., rate=0.)
    candidate, _ = ledger_case(fills=np.array([[-1., 0.], [0., 0.], [0., 0.], [1., 0.]]), spread=0., rate=0.)
    report = paired_execution_attribution(reference, candidate, prices, beta=[1., 2.])
    assert report["totals"]["beta_abs_mean_reduction"] > 0
    assert sum(row["beta_abs_mean_reduction_contribution"] for row in report["asset_rows"]) == pytest.approx(report["totals"]["beta_abs_mean_reduction"])
    same = paired_execution_attribution(reference, reference, prices, beta=[1., 2.])
    assert all(row["beta_abs_mean_reduction_contribution"] == 0. for row in same["asset_rows"])


def test_direct_module_load_has_no_execution_package_dependency():
    source = Path(__file__).resolve().parents[3]/"src/stable_finance/evaluation/scoring/holdings_attribution.py"
    # The module has no package-relative imports or class identities to couple
    # it to whichever stable_finance execution version owns the supplied ledger.
    spec = importlib.util.spec_from_file_location("isolated_attribution", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ledger, prices = ledger_case()
    assert module.execution_attribution(ledger, prices)["totals"]["gross_pnl"] == 7.


@pytest.mark.parametrize("field", ["arrival_slippage", "commissions", "trade_notional", "nav", "net_returns", "gross_returns"])
def test_accounting_mismatch_rejected(field):
    ledger, prices = ledger_case()
    setattr(ledger, field, getattr(ledger, field).copy())
    getattr(ledger, field).flat[0] += 1.
    with pytest.raises(ValueError, match="reconcile"):
        execution_attribution(ledger, prices, beta=[1., 2.])


@pytest.mark.parametrize("change,match", [
    (lambda ledger: setattr(ledger, "assets", np.array(["A", "A"])), "unique"),
    (lambda ledger: setattr(ledger, "times", np.array([0., 5., 5., 15.])), "increasing"),
    (lambda ledger: setattr(ledger, "commission_rates", -np.ones((4, 2))), "negative"),
    (lambda ledger: setattr(ledger, "order_records", ()), "order records"),
])
def test_bad_axes_rates_and_records_rejected(change, match):
    ledger, prices = ledger_case()
    change(ledger)
    with pytest.raises(ValueError, match=match):
        execution_attribution(ledger, prices, beta=[1., 2.])


def test_bad_prices_beta_and_pair_axes_rejected():
    ledger, prices = ledger_case()
    with pytest.raises(ValueError, match="beta must be finite"):
        execution_attribution(ledger, prices, beta=[1., np.nan])
    with pytest.raises(ValueError, match="positive"):
        execution_attribution(ledger, -prices)
    candidate = deepcopy(ledger)
    candidate.assets = np.array(["B", "A"])
    with pytest.raises(ValueError, match="paired ledgers"):
        paired_execution_attribution(ledger, candidate, prices, beta=[1., 2.])


def test_price_panel_axes_are_checked():
    ledger, prices = ledger_case()
    panel = SimpleNamespace(values=prices, times=ledger.times, assets=ledger.assets)
    assert execution_attribution(ledger, panel)["totals"]["gross_pnl"] == 7.
    panel.assets = np.array(["B", "A"])
    with pytest.raises(ValueError, match="price-panel axes"):
        execution_attribution(ledger, panel)


def test_boundary_classification_uses_send_target_not_arrival_exposure():
    ledger, prices = ledger_case()
    # Both target beta exposures hit 0.1, while actual filled exposures differ.
    report = execution_attribution(ledger, prices, beta=[1., 2.], beta_bound=.1)
    assert report["totals"]["boundary_target_decisions"] == 2
    assert report["totals"]["boundary_turnover"] > 0
    assert report["totals"]["beta_abs_mean"] != pytest.approx(.1)
