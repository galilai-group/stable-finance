"""Fees charged on top of a cost model: commissions, access fees and the SEC's fee on sales.

Three properties carry the feature. A fee adds exactly ``fee x |dw|`` to each
period's cost (the SEC fee only on what is sold), so the accounting identity
``net = gross - cost`` still splits into the model's charge and the fees. The
allocator is charged the fees it is then billed, so a book is never planned
against one cost and scored under another. And a configuration that sets no
fee is the configuration it always was: same label, same row, same numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

import stable_finance as sf
from stable_finance import (
    ACCESS_FEE_CAP_2020,
    DEFAULT_BACKTEST,
    SEC_FEE_RATE_FY2020,
    BacktestConfig,
    ForwardReturns,
    MarketData,
    MissingRolesError,
    PortfolioWeights,
    Trades,
    breakeven_efq,
    check_config,
    enumerate_configs,
    evaluate_fills,
    event_backtest,
    fee_cost,
    fee_per_unit,
    fill_path,
    run_backtest,
    sharpe_vs_execution_quality,
    sweep_configs,
    trading_cost,
)

D, A = 40, 5
DECISIONS = np.arange(D)
ASSETS = np.array([f"S{i}" for i in range(A)])
AXIS = np.array([300])
DAYS = np.repeat(np.arange(D // 8), 8)
CROSS = DEFAULT_BACKTEST.replace(cost_model="cross", annualization="decisions")


def _panel(values, cls=ForwardReturns):
    return cls(np.asarray(values, dtype=float)[:, :, None], DECISIONS, ASSETS, AXIS)


def _world(seed=0):
    """A quoted market (bid and ask, so mid and half-spread are derived), a path of weights, returns."""
    rng = np.random.default_rng(seed)
    mid = 20.0 + 30.0 * rng.random(A) * np.exp(rng.normal(0, 0.01, (D, A)).cumsum(axis=0))
    half = mid * rng.uniform(1e-4, 4e-4, (D, A))
    market = MarketData({"bid_price": mid - half, "ask_price": mid + half}, DECISIONS, ASSETS)
    weights = rng.normal(0, 0.2, (D, A))
    weights[::7] = weights[np.maximum(np.arange(0, D, 7) - 1, 0)]     # some decisions do not trade
    realized = rng.normal(0, 0.002, (D, A))
    return {"weights": _panel(weights, PortfolioWeights), "realized": _panel(realized),
            "market": market, "mid": mid, "path": weights}


def _moves(path):
    return np.diff(np.vstack([np.zeros((1, A)), path]), axis=0)


# ── configuration ───────────────────────────────────────────────────────────

def test_unset_fees_leave_the_default_label_and_row_exactly_as_they_were():
    assert DEFAULT_BACKTEST.label == (
        "universe=all|selection=all|weighting=mean_variance|risk_model=shrunk_sample|"
        "cost_model=cross|efq=1.0|rebalance=every_decision|annualization=daily|"
        "gross_exposure=2.0|shrinkage=0.3|max_iter=400")
    assert DEFAULT_BACKTEST.fees == {} and DEFAULT_BACKTEST.requires == {
        "cost_model": ("half_spread",)}
    w = _world()
    row = run_backtest(w["weights"], w["realized"], market=w["market"], config=CROSS).as_dict()
    assert not {"fee_bps", "fee_per_share", "sec_fee_rate", "mean_fee_bps"} & set(row)
    # ...and the default space is untouched by settings it does not sweep.
    assert all(c.fees == {} for c in enumerate_configs())


def test_fee_settings_are_validated_and_stored_as_floats():
    for name in ("fee_bps", "fee_per_share", "sec_fee_rate"):
        for bad in (-1e-9, np.inf, np.nan, True, "1"):
            with pytest.raises(ValueError, match=name):
                BacktestConfig(**{name: bad})
    config = BacktestConfig(fee_bps=1, fee_per_share=np.float64(0.003), sec_fee_rate=0)
    assert config.fees == {"fee_bps": 1.0, "fee_per_share": 0.003, "sec_fee_rate": 0.0}
    assert all(type(v) is float for v in config.fees.values())
    assert "fee_bps=1.0" in config.label and config == BacktestConfig(
        fee_bps=1.0, fee_per_share=0.003, sec_fee_rate=0.0)


def test_the_frictionless_benchmark_refuses_fees():
    with pytest.raises(ValueError, match="frictionless"):
        BacktestConfig(cost_model="mid", fee_bps=0.5)
    with pytest.raises(ValueError, match="frictionless"):
        trading_cost("mid", np.array([[1e-4]]), fee_bps=0.5)
    # A midpoint fill that pays its fees is cross at zero execution quality.
    BacktestConfig(cost_model="cross", efq=0.0, fee_bps=0.5)


def test_a_per_share_fee_needs_a_price_and_says_so():
    config = CROSS.replace(fee_per_share=ACCESS_FEE_CAP_2020)
    assert config.requires == {"cost_model": ("half_spread",), "fee_per_share": ("mid",)}
    assert CROSS.replace(fee_bps=1.0, sec_fee_rate=SEC_FEE_RATE_FY2020).requires == CROSS.requires
    spread_only = MarketData({"half_spread": np.full((D, A), 1e-4)}, DECISIONS, ASSETS)
    with pytest.raises(MissingRolesError, match="fee_per_share=0.003.*share price"):
        check_config(config, spread_only)
    w = _world()
    check_config(config, w["market"])                     # mid derived from bid and ask
    with pytest.raises(MissingRolesError, match="fee_per_share"):
        run_backtest(w["weights"], w["realized"], market=spread_only, config=config)
    # A space built for a market keeps only what it can charge.
    space = enumerate_configs(market=spread_only, universe="all", selection="all",
                              weighting="equal", risk_model="diagonal",
                              efq=1.0, rebalance="every_decision",
                              fee_per_share=(None, 0.003), fee_bps=(None, 1.0))
    assert {(c.fee_per_share, c.fee_bps) for c in space} == {(None, None), (None, 1.0)}


# ── the functions ───────────────────────────────────────────────────────────

def test_fee_cost_charges_commission_and_access_fee_both_ways_and_the_sec_fee_on_sales():
    q = np.array([[0.1, -0.2, 0.0], [0.0, 0.05, -0.3]])
    price = np.array([[20.0, 50.0, np.nan], [10.0, 25.0, 40.0]])
    paid = fee_cost(q, price, fee_bps=0.5, fee_per_share=0.003, sec_fee_rate=22.10e-6)
    expected = (np.abs(q) * (0.5e-4 + 0.003 / np.where(q != 0, price, 1.0))
                + np.maximum(-q, 0) * 22.10e-6)
    np.testing.assert_allclose(paid, expected, rtol=1e-13, atol=0)
    assert paid[0, 2] == 0.0 and paid[1, 0] == 0.0       # no trade, no fee, even unpriced
    assert np.all(fee_cost(q) == 0.0)                     # nothing set charges nothing
    with pytest.raises(ValueError, match="positive price"):
        fee_cost(np.array([[0.1]]), np.array([[np.nan]]), fee_per_share=0.003)
    with pytest.raises(ValueError, match="needs each name's price"):
        fee_cost(q, fee_per_share=0.003)


def test_the_allocator_charge_is_a_round_trip_split_evenly_between_its_legs():
    """The SEC fee is paid once per round trip; the allocator charges half on each leg."""
    price = np.array([[20.0, np.nan]])
    per_unit = fee_per_unit(price, fee_bps=0.5, fee_per_share=0.003, sec_fee_rate=22.10e-6)
    assert per_unit[0, 0] == pytest.approx(0.5e-4 + 0.003 / 20.0 + 22.10e-6 / 2, rel=1e-14)
    assert np.isinf(per_unit[0, 1])                       # no price: cannot be charged, so held
    trip = fee_cost(np.array([[0.1], [-0.1]]), 20.0, fee_bps=0.5, fee_per_share=0.003,
                    sec_fee_rate=22.10e-6).sum()
    assert trip == pytest.approx(2 * 0.1 * per_unit[0, 0], rel=1e-13)
    # The 2020 values, as the docstring shows them on a $20 stock.
    assert (ACCESS_FEE_CAP_2020, SEC_FEE_RATE_FY2020) == (0.003, 22.10e-6)
    assert fee_per_unit(20.0, fee_per_share=ACCESS_FEE_CAP_2020,
                        sec_fee_rate=SEC_FEE_RATE_FY2020) == pytest.approx(1.6105e-4)
    assert fee_per_unit(fee_bps=2.0) == pytest.approx(2e-4)


def test_trading_cost_adds_the_fees_and_keeps_unquoted_names_untradable():
    spread = np.array([[1e-4, 2e-4, np.nan]])
    price = np.array([[20.0, 40.0, 30.0]])
    fees = {"fee_bps": 0.3, "fee_per_share": 0.003, "sec_fee_rate": 22.10e-6}
    charged = trading_cost("cross", spread, multiple=0.7, price=price, **fees)
    np.testing.assert_allclose(charged[0, :2], trading_cost("cross", spread, multiple=0.7)[0, :2]
                               + fee_per_unit(price, **fees)[0, :2], rtol=1e-14)
    assert np.isinf(charged[0, 2])


# ── scoring: exactly fee x |dw| per period ──────────────────────────────────

@pytest.mark.parametrize("fees", [
    {"fee_bps": 0.7},
    {"fee_per_share": ACCESS_FEE_CAP_2020},
    {"sec_fee_rate": SEC_FEE_RATE_FY2020},
    {"fee_bps": 0.7, "fee_per_share": ACCESS_FEE_CAP_2020, "sec_fee_rate": SEC_FEE_RATE_FY2020},
])
def test_fees_add_exactly_fee_times_the_weight_traded_to_each_periods_cost(fees):
    w = _world()
    run = lambda config: run_backtest(w["weights"], w["realized"], market=w["market"],
                                      config=config, with_series=True)
    plain, charged = run(CROSS), run(CROSS.replace(**fees))
    dw = _moves(w["path"])
    expected = (np.abs(dw) * (fees.get("fee_bps", 0.0) / 1e4
                              + fees.get("fee_per_share", 0.0) / w["mid"])
                + np.maximum(-dw, 0) * fees.get("sec_fee_rate", 0.0)).sum(axis=1)
    np.testing.assert_allclose(charged.series.cost - plain.series.cost, expected,
                               rtol=1e-9, atol=1e-16)
    np.testing.assert_allclose(charged.series.fee, expected, rtol=1e-12, atol=0)
    # The fees are inside the cost, and net is still gross minus cost.
    np.testing.assert_array_equal(charged.series.gross, plain.series.gross)
    np.testing.assert_allclose(charged.series.net, charged.series.gross - charged.series.cost,
                               rtol=1e-12, atol=1e-18)
    assert charged.mean_fee_bps == pytest.approx(expected.mean() * 1e4, rel=1e-12)
    assert charged.mean_cost_bps == pytest.approx(plain.mean_cost_bps + charged.mean_fee_bps)
    assert charged.sharpe_mid == plain.sharpe_mid and charged.sharpe_net < plain.sharpe_net
    assert charged.mean_turnover == plain.mean_turnover
    assert charged.as_dict()["mean_fee_bps"] == charged.mean_fee_bps


def test_fees_fold_into_days_with_the_rest_of_the_cost():
    w = _world()
    config = CROSS.replace(annualization="daily", fee_bps=1.0)
    daily = run_backtest(w["weights"], w["realized"], market=w["market"], days=DAYS,
                         config=config, with_series=True)
    per_decision = run_backtest(w["weights"], w["realized"], market=w["market"], days=DAYS,
                                config=config.replace(annualization="decisions"),
                                with_series=True)
    np.testing.assert_allclose(daily.series.fee, per_decision.series.fee.reshape(-1, 8).sum(axis=1),
                               rtol=1e-12)
    assert daily.series.cost == pytest.approx(per_decision.series.cost.reshape(-1, 8).sum(axis=1))


def test_the_impact_model_and_drift_holding_pay_fees_on_their_own_fills():
    w = _world()
    adv = MarketData({**w["market"].columns, "adv": np.full((D, A), 5e7)}, DECISIONS, ASSETS)
    impact = DEFAULT_BACKTEST.replace(cost_model="impact", equity=1e6, execution_days=0,
                                      annualization="decisions")
    drift = CROSS.replace(holding="drift")
    for config, market in ((impact, adv), (drift, w["market"])):
        plain = run_backtest(w["weights"], w["realized"], market=market, days=DAYS,
                             config=config, with_series=True)
        charged = run_backtest(w["weights"], w["realized"], market=market, days=DAYS,
                               config=config.replace(fee_bps=2.0), with_series=True)
        traded = plain.mean_turnover * D
        assert (charged.series.cost - plain.series.cost).sum() == pytest.approx(
            2e-4 * traded, rel=1e-9)


# ── the allocator is charged what it is billed ──────────────────────────────

def _flat_market(spread):
    mid = np.full((D, A), 25.0)
    return MarketData({"half_spread": np.full((D, A), spread), "mid": mid}, DECISIONS, ASSETS)


def _forecast_world(seed=3):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 0.003, (D, A))
    return {"forecast": _panel(truth + rng.normal(0, 0.003, (D, A))),
            "realized": _panel(truth + rng.normal(0, 0.006, (D, A))),
            "history": _panel(rng.normal(0, 0.006, (D, A)))}


def test_a_flat_fee_on_a_flat_spread_is_the_same_book_and_bill_as_a_higher_efq():
    """The allocator plans against spread x efq + fee, and the scoring bills the same.

    On a market whose half-spread is one number ``s``, a fee of ``f`` per unit
    costs what ``f / s`` more EFQ would, in the allocator and in the scoring
    alike. So the two configurations must hold the same book and pay the same
    bill -- which they would not if the fee reached only one of the two.
    """
    s, f_bps, efq = 2e-4, 0.8, 0.5
    w, market = _forecast_world(), _flat_market(s)
    fee = run_backtest(**w, market=market, config=CROSS.replace(efq=efq, fee_bps=f_bps))
    same = run_backtest(**w, market=market, config=CROSS.replace(efq=efq + f_bps / 1e4 / s))
    free = run_backtest(**w, market=market, config=CROSS.replace(efq=efq))
    for name in ("mean_turnover", "mean_cost_bps", "sharpe_net", "transfer_coefficient"):
        assert getattr(fee, name) == pytest.approx(getattr(same, name), rel=1e-7), name
    assert fee.mean_turnover < free.mean_turnover           # the fee changed the book
    # A per-share fee on a flat price is a flat fee too: f / p per unit.
    share = run_backtest(**w, market=market,
                         config=CROSS.replace(efq=efq, fee_per_share=0.002))
    alike = run_backtest(**w, market=market, config=CROSS.replace(efq=efq, fee_bps=0.002 / 25 * 1e4))
    assert share.mean_turnover == pytest.approx(alike.mean_turnover, rel=1e-7)
    assert share.sharpe_net == pytest.approx(alike.sharpe_net, rel=1e-7)


def test_a_path_independent_book_ignores_fees_and_only_the_bill_moves():
    w, market = _forecast_world(), _flat_market(2e-4)
    flat = CROSS.replace(weighting="equal", risk_model="diagonal")
    a = run_backtest(**w, market=market, config=flat)
    b = run_backtest(**w, market=market, config=flat.replace(fee_bps=3.0))
    assert a.mean_turnover == b.mean_turnover and a.sharpe_mid == b.sharpe_mid
    assert b.mean_cost_bps == pytest.approx(a.mean_cost_bps + 3.0 * a.mean_turnover)


def test_a_sweep_with_fees_agrees_with_running_each_alone():
    w, market = _forecast_world(), _flat_market(2e-4)
    space = enumerate_configs(universe="all", selection="all", risk_model="diagonal",
                              weighting=("mean_variance", "equal"),
                              efq=(0.5, 1.0), rebalance="every_decision",
                              fee_bps=(None, 1.0), sec_fee_rate=(None, SEC_FEE_RATE_FY2020),
                              fee_per_share=(None, 0.003))
    assert len(space) == 2 * 2 * 2 * 2 * 2
    swept = sweep_configs(w["forecast"], w["realized"], space, market=market,
                          history=w["history"], book_cache_size=2)
    for config, report in zip(space, swept):
        one = run_backtest(**w, market=market, config=config)
        assert report.as_dict() == pytest.approx(one.as_dict(), nan_ok=True)


# ── the break-even identity, after fees ─────────────────────────────────────

def test_cost_is_linear_in_efq_with_the_fees_as_its_intercept_and_the_break_even_holds():
    """net = gross - fees - efq x S, so the break-even EFQ is mean(gross - fees) / mean(S)."""
    w = _world(seed=5)
    # A book that peeks at its own return, so there is an edge to break even on.
    path = w["path"] * 0.2 + 0.02 * np.sign(w["realized"].values[:, :, 0])
    weights = _panel(path, PortfolioWeights)
    fees = {"fee_bps": 0.2, "fee_per_share": 0.001, "sec_fee_rate": SEC_FEE_RATE_FY2020}
    run = lambda efq: run_backtest(weights, w["realized"], market=w["market"],
                                   config=CROSS.replace(efq=efq, **fees), with_series=True)
    one, half = run(1.0), run(0.5)
    spread_cost = one.series.cost - one.series.fee
    np.testing.assert_allclose(half.series.fee, one.series.fee, rtol=1e-12)
    np.testing.assert_allclose(half.series.cost, half.series.fee + 0.5 * spread_cost,
                               rtol=1e-9, atol=1e-16)
    gross = one.series.gross
    breakeven = breakeven_efq(gross, spread_cost, fees=one.series.fee)
    assert breakeven == pytest.approx((gross - one.series.fee).mean() / spread_cost.mean())
    assert 0 < breakeven < breakeven_efq(gross, spread_cost)          # fees move it down
    at = run(breakeven)
    assert at.series.net.mean() == pytest.approx(0.0, abs=1e-12)
    curve = sharpe_vs_execution_quality(gross, spread_cost, fees=one.series.fee,
                                        efq_grid=[0.0, 0.5, 1.0],
                                        periods_per_year=one.periods_per_year)
    assert curve.breakeven == pytest.approx(breakeven)
    assert curve.at(0.5) == pytest.approx(half.sharpe_net, rel=1e-9)
    assert curve.at(1.0) == pytest.approx(one.sharpe_net, rel=1e-9)


# ── unset is the configuration it always was ────────────────────────────────

def test_zero_fees_reproduce_the_unset_report_bit_for_bit():
    """``x + 0.0 == x``: a zero fee runs the fee path and must land on the same numbers.

    The unset path does not run the fee code at all, so this pins that adding
    the fee code changed nothing it added nothing to -- for the cost-aware
    allocator, whose book the fee enters, and for every score.
    """
    w, market = _forecast_world(), _flat_market(2e-4)
    for config in (CROSS, CROSS.replace(efq=0.7, annualization="daily")):
        unset = run_backtest(**w, market=market, days=DAYS, config=config, with_series=True)
        zero = run_backtest(**w, market=market, days=DAYS, with_series=True,
                            config=config.replace(fee_bps=0.0, fee_per_share=0.0,
                                                  sec_fee_rate=0.0))
        a, b = unset.as_dict(), zero.as_dict()
        assert b.pop("mean_fee_bps") == 0.0
        for name in ("fee_bps", "fee_per_share", "sec_fee_rate"):
            assert b.pop(name) == 0.0
        assert a == b
        for name in ("gross", "net", "cost"):
            np.testing.assert_array_equal(getattr(unset.series, name), getattr(zero.series, name))
        assert unset.series.fee is None and unset.mean_fee_bps is None


# ── the other places a fill's cost is read ──────────────────────────────────

def test_evaluate_fills_adds_the_fees_it_is_given():
    w = _world()
    fills = fill_path(w["weights"], w["market"], model="cross", efq=1.0)
    paid = fee_cost(fills.quantity, w["mid"][:, :, None], fee_per_share=0.003)
    plain, = evaluate_fills(fills, w["realized"])
    charged, = evaluate_fills(fills, w["realized"], fees=paid)
    np.testing.assert_allclose(charged.costs - plain.costs, paid[:, :, 0].sum(axis=1), rtol=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_fills(fills, w["realized"], fees=-paid)


def test_an_event_ledger_charges_each_trade_its_own_fees():
    t0 = 1_700_000_000
    tape = {"ticker": np.array(["A"] * 4), "timestamp": t0 + np.array([0, 5, 100, 105]),
            "bid_price": np.array([99.0, 99.0, 109.0, 109.0]),
            "ask_price": np.array([101.0, 101.0, 111.0, 111.0])}
    trades = Trades(["A"], [t0], [t0 + 100], [0.01])
    config = DEFAULT_BACKTEST.replace(cost_model="cross", efq=0.5)
    plain = event_backtest(trades, tape, config=config)
    charged = event_backtest(trades, tape, config=config.replace(
        fee_per_share=ACCESS_FEE_CAP_2020, sec_fee_rate=SEC_FEE_RATE_FY2020))
    exit_size = 0.01 * 110 / 100                                   # the position drifted with the price
    fees = 0.01 * 0.003 / 100 + exit_size * (0.003 / 110 + 22.10e-6)
    assert charged.ledger.cost[0] - plain.ledger.cost[0] == pytest.approx(fees, rel=1e-9)
    assert charged.report.mean_fee_bps * charged.report.n_periods == pytest.approx(
        1e4 * fees, rel=1e-9)


def test_the_fee_api_is_exported():
    for name in ("ACCESS_FEE_CAP_2020", "SEC_FEE_RATE_FY2020", "FEE_SETTINGS", "FEE_REQUIRES",
                 "fee_cost", "fee_per_unit"):
        assert name in sf.__all__
    assert sf.FEE_SETTINGS == ("fee_bps", "fee_per_share", "sec_fee_rate")
    assert set(sf.FEE_SETTINGS) <= set(sf.OPTIONAL_SETTINGS)
