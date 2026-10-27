import numpy as np
import pytest

from stable_finance import DEFAULT_BACKTEST, Trades, event_backtest
from stable_finance.dataset.anchors import TO_CLOSE

T0 = 1_700_000_000          # epoch seconds
CROSS = DEFAULT_BACKTEST.replace(cost_model="cross", efq=0.5)


def tape(rows):
    """rows: (ticker, seconds after T0, bid, ask)."""
    t, s, b, a = zip(*rows)
    return {"ticker": np.array(t), "timestamp": T0 + np.array(s), "bid_price": np.array(b, float),
            "ask_price": np.array(a, float)}


def test_a_round_trip_earns_its_price_change_and_pays_its_own_fills():
    q = tape([("A", 0, 99.0, 101.0), ("A", 5, 99.0, 101.0), ("A", 100, 109.0, 111.0), ("A", 105, 109.0, 111.0)])
    trades = Trades(["A"], [T0], [T0 + 100], [0.01])
    r = event_backtest(trades, q, config=CROSS)
    assert r.ledger.gross[0] == pytest.approx(0.01 * (110 / 100 - 1))
    # entry 0.5 x 1% half-spread, exit 0.5 x (1/110) half-spread, on the position each time
    assert r.ledger.cost[0] == pytest.approx(0.01 * 0.005 + 0.01 * 1.1 * 0.5 / 110)
    assert r.report.mean_gross_return_bps == pytest.approx(1e4 * r.ledger.gross[0])


def test_an_auction_exit_fills_at_its_price_with_no_spread():
    q = tape([("A", 0, 99.0, 101.0), ("A", 50, 104.0, 106.0)])
    trades = Trades(["A"], [T0], [T0 + 100], [-0.02], exit_price=[90.0])
    r = event_backtest(trades, q, config=CROSS)
    assert r.ledger.exit_price[0] == pytest.approx(90.0)
    assert r.ledger.gross[0] == pytest.approx(-0.02 * (90 / 100 - 1))
    assert r.ledger.cost[0] == pytest.approx(0.02 * 0.5 * 0.01)          # entry only
    assert r.report.horizon == TO_CLOSE


def test_overlapping_trades_on_one_name_are_separate_positions_and_slots_are_reused():
    q = tape([("A", 0, 100.0, 100.0), ("A", 30, 102.0, 102.0), ("A", 60, 104.0, 104.0),
              ("A", 90, 106.0, 106.0), ("A", 200, 110.0, 110.0)])
    trades = Trades(["A", "A", "A"], [T0, T0 + 30, T0 + 90], [T0 + 60, T0 + 90, T0 + 200], [0.01, -0.01, 0.01])
    r = event_backtest(trades, q, config=CROSS)
    assert r.ledger.slot[0] != r.ledger.slot[1]
    assert r.ledger.slot[2] == r.ledger.slot[0]                     # free again after the first closed
    expect = [0.01 * (104 / 100 - 1), -0.01 * (106 / 102 - 1), 0.01 * (110 / 106 - 1)]
    assert r.ledger.gross == pytest.approx(expect)
    assert r.ledger.cost == pytest.approx([0, 0, 0])                 # locked quotes: no spread


def test_a_touch_exit_without_a_quote_waits_for_the_next_one():
    q = tape([("A", 0, 99.0, 101.0), ("A", 100, np.nan, np.nan), ("A", 150, 109.0, 111.0),
              ("B", 0, 50.0, 50.0), ("B", 160, 50.0, 50.0)])
    trades = Trades(["A", "B"], [T0, T0], [T0 + 100, T0 + 160], [0.01, 0.01])
    r = event_backtest(trades, q, config=CROSS)
    assert r.ledger.exit_fill_time[0] == T0 + 160                    # first decision with a quote
    assert r.ledger.gross[0] == pytest.approx(0.01 * (110 / 100 - 1))


def test_a_per_trade_efq_replaces_the_configured_one():
    q = tape([("A", 0, 99.0, 101.0), ("A", 100, 99.0, 101.0)])
    free = event_backtest(Trades(["A"], [T0], [T0 + 100], [0.01], efq=[0.0]), q, config=CROSS)
    full = event_backtest(Trades(["A"], [T0], [T0 + 100], [0.01], efq=[1.0]), q, config=CROSS)
    assert free.ledger.cost[0] == 0.0
    assert full.ledger.cost[0] == pytest.approx(2 * 0.01 * 0.01)


def test_trades_are_validated():
    with pytest.raises(ValueError, match="exit after"):
        Trades(["A"], [T0], [T0], [0.01])
    with pytest.raises(ValueError, match="one value per trade"):
        Trades(["A", "B"], [T0], [T0 + 1, T0 + 2], [0.01, 0.01])
