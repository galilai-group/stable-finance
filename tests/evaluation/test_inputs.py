"""Your table in, evaluation inputs out: the interface any dataset needs."""
from __future__ import annotations

import numpy as np
import pytest

import stable_finance as sf
from stable_finance import DEFAULT_BACKTEST, MissingRolesError, evaluation_inputs
from stable_finance.dataset import FeatureSchema

T0 = 1_700_000_000
MID = DEFAULT_BACKTEST.replace(cost_model="mid")


def _frame(n=600, assets=("A", "B", "C"), quotes=False, seed=0):
    rng = np.random.default_rng(seed)
    t, sym, px, bid, ask, vol = [], [], [], [], [], []
    for k, name in enumerate(assets):
        p = 50 + k + rng.normal(0, 0.02, n).cumsum()
        t.append(T0 + 60 * np.arange(1, n + 1))          # stamped at the end of each minute
        sym.append(np.full(n, name)); px.append(p)
        bid.append(p - 0.01); ask.append(p + 0.01); vol.append(rng.poisson(5, n) + 1.0)
    frame = {"timestamp": np.concatenate(t), "ticker": np.concatenate(sym),
             "px": np.concatenate(px)}
    if quotes:
        frame |= {"bid": np.concatenate(bid), "ask": np.concatenate(ask),
                  "vol": np.concatenate(vol)}
    return frame


def test_a_univariate_table_gives_returns_and_only_a_midpoint():
    frame = _frame()
    decisions = T0 + 60 * np.arange(10, 400, 30)
    realized, market = evaluation_inputs(frame, FeatureSchema.univariate("px"), decisions, [300, 900])
    assert realized.values.shape == (len(decisions), 3, 2)
    assert market.available == ("mid", "window_best_buy", "window_best_sell")
    assert market.level == "mid"
    # The base price is the minute after the decision, never the bar ending at it.
    a = frame["ticker"] == "A"
    ts, px = frame["timestamp"][a], frame["px"][a]
    t = decisions[0]
    base = px[(ts > t) & (ts <= t + 60)].mean()
    forward = px[(ts > t + 300) & (ts <= t + 360)].mean()
    np.testing.assert_allclose(realized.values[0, 0, 0], forward / base - 1)
    # The market is what was known by the decision.
    np.testing.assert_allclose(market["mid"][0, 0], px[ts <= t][-1])


def test_the_univariate_inputs_run_a_frictionless_backtest_and_refuse_costs():
    frame = _frame(assets=tuple("ABCDEFGH"))
    decisions = T0 + 60 * np.arange(10, 500, 15)
    realized, market = evaluation_inputs(frame, FeatureSchema.univariate("px"), decisions, [300])
    forecast = sf.ForwardReturns(np.nan_to_num(realized.values), realized.decisions,
                                 realized.assets, realized.horizons)
    report = sf.run_backtest(forecast, realized, market=market, config=MID)
    assert np.isfinite(report.sharpe_mid)
    with pytest.raises(MissingRolesError, match="cost_model='cross'"):
        sf.run_backtest(forecast, realized, market=market)


def test_quotes_and_volume_give_a_vwap_return_and_a_spread():
    schema = FeatureSchema.from_roles(vwap="px", bid_price="bid", ask_price="ask", volume="vol")
    frame = _frame(quotes=True)
    decisions = T0 + 60 * np.arange(10, 400, 30)
    realized, market = evaluation_inputs(frame, schema, decisions, [300])
    assert market.has("half_spread", "bid_price", "ask_price", "vwap")
    np.testing.assert_allclose(market["half_spread"], 0.01 / market["mid"], rtol=1e-12)
    a = frame["ticker"] == "A"
    ts, px, vol = frame["timestamp"][a], frame["px"][a], frame["vol"][a]
    t = decisions[1]
    w0, w1 = (ts > t) & (ts <= t + 60), (ts > t + 300) & (ts <= t + 360)
    vwap = lambda m: (px[m] * vol[m]).sum() / vol[m].sum()  # noqa: E731
    np.testing.assert_allclose(realized.values[1, 0, 0], vwap(w1) / vwap(w0) - 1)


def test_a_horizon_past_the_data_is_missing_not_guessed():
    frame = _frame(n=100)
    realized, _ = evaluation_inputs(frame, FeatureSchema.univariate("px"),
                                    [T0 + 60 * 90], [3600])
    assert np.isnan(realized.values).all()


def test_a_single_series_needs_no_asset_column():
    frame = _frame(assets=("A",))
    del frame["ticker"]
    realized, market = evaluation_inputs(frame, FeatureSchema.univariate("px"),
                                         [T0 + 600], [300], asset=None)
    assert realized.assets.tolist() == ["series"] and np.isfinite(realized.values).all()
    with pytest.raises(KeyError):
        evaluation_inputs(frame, FeatureSchema.univariate("price"), [T0 + 600], [300])


def test_quotes_are_read_at_arrival_and_sizes_over_the_depth_window():
    from stable_finance import evaluation_inputs
    from stable_finance.dataset import FeatureSchema

    t0 = 1_700_000_000
    i = np.arange(100)
    table = {"timestamp": t0 + i, "ticker": np.full(100, "A"), "bid": 100.0 + i,
             "ask": 101.0 + i, "bsz": i.astype(float), "asz": 2.0 * i}
    schema = FeatureSchema.from_roles(bid_price="bid", ask_price="ask", bid_size="bsz",
                                      ask_size="asz")
    _, market = evaluation_inputs(table, schema, [t0 + 10], [30], delay_seconds=5,
                                  depth_seconds=3)
    assert market.level == "book"
    assert market.get("bid_price")[0, 0] == 110.0                 # the decision's quote
    assert market.get("exec_bid_price")[0, 0] == 115.0            # five seconds later
    assert market.get("book_ask_price")[0, 0, 0] == 116.0
    assert market.get("book_bid_size")[0, 0, 0] == pytest.approx(np.mean([13, 14, 15]))
    assert market.get("book_ask_size")[0, 0, 0] == pytest.approx(2 * np.mean([13, 14, 15]))
    _, prompt = evaluation_inputs(table, schema, [t0 + 10], [30], delay_seconds=0)
    assert prompt.get("exec_bid_price")[0, 0] == 110.0
