"""Every input level runs against every level of market data its execution allows.

The grid is two axes: what you bring (embeddings, a forecast, weights,
orders) and what market data you have (a bare price, bars, quotes, a book).
Each cell either runs, or is refused with MissingRolesError naming the
column it lacks -- never a silent NaN or a free trade.
"""
from __future__ import annotations

import numpy as np
import pytest

import stable_finance as sf
from stable_finance import (
    DEFAULT_BACKTEST,
    Embeddings,
    MarketData,
    MissingRolesError,
    Orders,
    PortfolioWeights,
    evaluation_inputs,
    execute,
    fill_path,
    run_backtest,
)
from stable_finance.dataset import FeatureSchema, bar_half_spread

T0 = 1_700_000_000
N_MINUTES, N_ASSETS, TICK = 900, 12, 0.01
HALF = 0.01          # half the quoted spread, in price units
EQUITY = 1e5


def _table(seed=0):
    """Minute bars and end-of-minute quotes built from trades at bid and ask."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("timestamp", "ticker", "px", "open", "high", "low", "close",
                            "vol", "bid", "ask")}
    for a in range(N_ASSETS):
        p = 50 + a
        for m in range(N_MINUTES):
            path = p + rng.normal(0, 0.01, 8).cumsum()
            trades = path + HALF * rng.choice([-1, 1], 8)
            p = path[-1]
            for key, value in (("timestamp", T0 + 60 * (m + 1)), ("ticker", f"S{a:02d}"),
                               ("px", p), ("open", trades[0]), ("high", trades.max()),
                               ("low", trades.min()), ("close", trades[-1]), ("vol", 8000.0),
                               ("bid", p - HALF), ("ask", p + HALF)):
                cols[key].append(value)
    return {k: np.asarray(v) for k, v in cols.items()}


SCHEMAS = {
    "mid": FeatureSchema.univariate("px"),
    "bars": FeatureSchema.from_roles(open="open", high="high", low="low", close="close",
                                     volume="vol"),
    "quotes": FeatureSchema.from_roles(mid="px", bid_price="bid", ask_price="ask",
                                       volume="vol"),
}


def _book(market: MarketData) -> MarketData:
    """Three levels a tick apart, deeper further from the touch."""
    bid, ask = market.get("bid_price"), market.get("ask_price")
    steps = np.arange(3) * TICK
    sizes = np.broadcast_to(np.array([100.0, 200.0, 400.0]), (*bid.shape, 3))
    return MarketData({**market.columns,
                       "book_bid_price": bid[..., None] - steps,
                       "book_ask_price": ask[..., None] + steps,
                       "book_bid_size": sizes, "book_ask_size": sizes},
                      market.decisions, market.assets)


@pytest.fixture(scope="module")
def world():
    table = _table()
    decisions = T0 + 60 * np.arange(60, N_MINUTES - 60, 15)
    out = {}
    for level, schema in SCHEMAS.items():
        realized, market = evaluation_inputs(
            table, schema, decisions, [600], price="close" if level == "bars" else None,
            spread_lookback=1800 if level == "bars" else None,
            adv_lookback=None if level == "mid" else 3600)
        out[level] = (realized, market)
    realized, quotes = out["quotes"]
    out["book"] = (realized, _book(quotes))
    return out


def _inputs(realized):
    forecast = sf.oracle_forecast(realized, 0.3, seed=1)
    rng = np.random.default_rng(2)
    features = np.concatenate([np.nan_to_num(forecast.values),
                               rng.normal(size=forecast.values.shape[:2] + (3,))], axis=2)
    embeddings = Embeddings(features, realized.decisions, realized.assets)
    weights = PortfolioWeights(np.sign(np.nan_to_num(forecast.values)) / N_ASSETS,
                               realized.decisions, realized.assets, realized.horizons)
    return {"embeddings": embeddings, "forecast": forecast, "weights": weights,
            "orders": sf.orders_from_weights(weights)}


SUPPORTED = {"mid": {"mid"}, "bars": {"mid", "cross", "impact"},
             "quotes": {"mid", "cross", "impact"}, "book": {"mid", "cross", "book", "impact"}}


@pytest.mark.parametrize("level", ["mid", "bars", "quotes", "book"])
@pytest.mark.parametrize("cost_model", ["mid", "cross", "book", "impact", "impact_stock"])
@pytest.mark.parametrize("kind", ["embeddings", "forecast", "weights", "orders"])
def test_every_cell_runs_or_is_refused_for_what_it_lacks(world, level, cost_model, kind):
    realized, market = world[level]
    source = _inputs(realized)[kind]
    config = DEFAULT_BACKTEST.replace(
        cost_model=cost_model, equity=EQUITY if cost_model.startswith(("book", "impact")) else None)
    days = np.arange(len(realized.decisions)) // 13
    run = lambda: run_backtest(source, realized, market=market, config=config,  # noqa: E731
                               days=days,
                               train=(source, realized) if kind == "embeddings" else None)
    if cost_model not in SUPPORTED[level]:
        with pytest.raises(MissingRolesError):
            run()
        return
    report = run()
    assert np.isfinite(report.sharpe_mid) and np.isfinite(report.sharpe_net)
    assert report.mean_cost_bps >= 0
    if cost_model == "mid":
        assert report.mean_cost_bps == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(report.information_coefficient) == (kind in ("embeddings", "forecast"))


def test_market_levels_derive_downward(world):
    _, book = world["book"]
    assert world["mid"][1].level == "mid"
    assert world["bars"][1].levels == ("mid", "bars")
    assert world["bars"][1].has("adv") and not world["mid"][1].has("adv")
    assert world["quotes"][1].level == "quotes"
    only_depth = book.select(*sf.BOOK_QUANTITIES)
    assert only_depth.level == "book" and only_depth.has("bid_price", "mid", "half_spread")
    np.testing.assert_allclose(only_depth.get("mid"), book.get("mid"))


def test_the_bar_spread_estimate_recovers_the_true_spread(world):
    _, bars = world["bars"]
    assert bars.estimated == frozenset({"half_spread"})
    relative = HALF / bars.get("close")
    ratio = np.nanmedian(bars.get("half_spread") / relative)
    assert 0.7 < ratio < 1.3
    # One pair alone is a noisy term, not an estimate, and may be negative.
    assert bar_half_spread([2.0, 2.0], [1.0, 1.0], [1.0, 2.0]).shape == (1,)


def test_a_bigger_order_walks_further_into_the_book(world):
    _, book = world["book"]
    d, a = book.shape
    one = slice(0, 1)
    small = np.zeros((1, a, 1)); small[0, 0, 0] = 1e-4     # 10 dollars: the touch
    large = np.zeros((1, a, 1)); large[0, 0, 0] = 0.5      # 50,000 dollars: past all three levels
    market = MarketData({k: v[one] for k, v in book.columns.items()}, book.decisions[one],
                        book.assets)
    def fill(q):
        return execute(Orders(q, book.decisions[one], book.assets, [600]), market,
                       model="book", equity=EQUITY)
    s, big = fill(small), fill(large)
    ask = market.get("ask_price")[0, 0]
    assert s.price[0, 0, 0] == pytest.approx(ask)
    assert big.price[0, 0, 0] > ask
    # 700 shares is all the depth; the rest is left unfilled.
    assert big.quantity[0, 0, 0] == pytest.approx(700 * market.get("mid")[0, 0] / EQUITY)
    with pytest.raises(ValueError, match="equity"):
        execute(Orders(small, book.decisions[one], book.assets, [600]), market, model="book")


def test_a_limit_rests_and_fills_only_if_the_window_trades_through(world):
    realized, quotes = world["quotes"]
    rng = np.random.default_rng(5)
    q = rng.choice([-0.05, 0.05], size=realized.values.shape)
    market_orders = Orders(q, realized.decisions, realized.assets, realized.horizons)
    crossing = execute(market_orders, quotes)
    # A limit through the touch is marketable: it fills exactly as a market order.
    generous = Orders(q, realized.decisions, realized.assets, realized.horizons, limit=0.01)
    np.testing.assert_allclose(execute(generous, quotes).price, crossing.price)
    # A limit at the mid cannot cross a spread; it rests, and fills at the mid only
    # where the window's best price went through it.
    passive = Orders(q, realized.decisions, realized.assets, realized.horizons, limit=0.0)
    fills = execute(passive, quotes)
    filled = fills.quantity != 0
    assert 0 < filled.mean() < 1
    np.testing.assert_allclose(fills.price[filled], fills.mid_price[filled])
    best = np.where(q > 0, quotes.get("window_best_buy")[..., None],
                    quotes.get("window_best_sell")[..., None])
    mid = quotes.get("mid")[..., None]
    assert ((np.sign(q) * (mid - best))[filled] > 0).all()
    # Without the window, a limit has nothing to be filled against.
    with pytest.raises(MissingRolesError, match="window_best_buy"):
        execute(passive, quotes.select("bid_price", "ask_price"))


def test_a_limit_works_on_a_bare_price_series(world):
    realized, mid = world["mid"]
    q = np.full(realized.values.shape, 0.01)
    fills = execute(Orders(q, realized.decisions, realized.assets, realized.horizons,
                           limit=-0.0005), mid, model="mid")
    filled = fills.quantity != 0
    assert 0 < filled.mean() < 1
    np.testing.assert_allclose(fills.price[filled], fills.mid_price[filled] * (1 - 0.0005))


def test_the_fill_path_chases_a_target_the_book_could_not_reach(world):
    realized, book = world["book"]
    target = np.zeros(realized.values.shape)
    target[:, 0, 0] = 0.5                                  # far more than 700 shares
    weights = PortfolioWeights(target, realized.decisions, realized.assets, realized.horizons)
    path = fill_path(weights, book, model="book", equity=EQUITY)
    held = sf.positions_from_fills(path).values[:, 0, 0]
    assert (np.diff(held) >= -1e-12).all() and held[1] > held[0]
    one_shot = execute(sf.orders_from_weights(weights), book, model="book", equity=EQUITY)
    assert sf.positions_from_fills(one_shot).values[-1, 0, 0] < held[-1]
