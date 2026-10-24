import numpy as np
import pytest

from stable_finance import (
    Fills,
    ForwardReturns,
    PortfolioWeights,
    cross_spread_sharpe,
    evaluate_fills,
    mid_price_sharpe,
    orders_from_weights,
    positions_from_fills,
    simulate_fills,
)


def _panel():
    axes = ([1, 2], ["A", "B"], [300])
    weights = PortfolioWeights(
        np.array([[[0.5], [-0.5]], [[0.0], [1.0]]]), *axes
    )
    realized = ForwardReturns(
        np.array([[[0.10], [-0.10]], [[0.20], [0.03]]]), *axes
    )
    return weights, realized


def test_orders_are_weight_changes_from_cash_or_from_an_initial_portfolio():
    weights, _ = _panel()
    from_cash = orders_from_weights(weights)
    np.testing.assert_allclose(
        from_cash.values[:, :, 0], [[0.5, -0.5], [-0.5, 1.5]]
    )
    held = orders_from_weights(weights, initial=[[0.5], [0.0]])
    np.testing.assert_allclose(held.values[0, :, 0], [0.0, -0.5])


def test_orders_treat_missing_targets_as_flat():
    weights = PortfolioWeights([[[np.nan]], [[0.2]]], [1, 2], ["A"], [300])
    np.testing.assert_allclose(orders_from_weights(weights).values[:, 0, 0], [0.0, 0.2])


def test_fills_execute_buys_at_ask_and_sells_at_bid():
    weights, _ = _panel()
    fills = simulate_fills(
        orders_from_weights(weights), np.full((2, 2), 99.0), np.full((2, 2), 101.0)
    )
    np.testing.assert_allclose(fills.quantity[:, :, 0], [[0.5, -0.5], [-0.5, 1.5]])
    np.testing.assert_allclose(fills.price[:, :, 0], [[101.0, 99.0], [99.0, 101.0]])
    np.testing.assert_allclose(fills.mid_price, 100.0)


def test_orders_without_a_quote_are_left_unfilled_and_positions_lag():
    weights, realized = _panel()
    bid = np.array([[99.0, np.nan], [99.0, 99.0]])
    ask = np.array([[101.0, np.nan], [101.0, 101.0]])
    fills = simulate_fills(orders_from_weights(weights), bid, ask)
    assert fills.quantity[0, 1, 0] == 0.0
    assert np.isnan(fills.price[0, 1, 0])
    positions = positions_from_fills(fills)
    # B was never sold short, so the later +1.5 buy leaves it at 1.5, not 1.0.
    np.testing.assert_allclose(positions.values[:, :, 0], [[0.5, 0.0], [0.0, 1.5]])
    result, = evaluate_fills(fills, realized, periods_per_year=2)
    np.testing.assert_allclose(result.gross_returns, [0.05, 0.045])
    np.testing.assert_allclose(result.costs, [0.005, 0.02])


def test_fully_filled_orders_reproduce_the_quoted_spread_sharpe():
    weights, realized = _panel()
    bid, ask = np.full((2, 2), 99.0), np.full((2, 2), 101.0)
    fills = simulate_fills(orders_from_weights(weights), bid, ask)
    result, = evaluate_fills(fills, realized, periods_per_year=2)
    np.testing.assert_allclose(result.net_returns, [0.09, 0.01])
    assert result.sharpe == pytest.approx(
        cross_spread_sharpe(weights, realized, bid, ask, periods_per_year=2)[0]
    )
    realized_positions = positions_from_fills(fills)
    np.testing.assert_allclose(realized_positions.values, weights.values)
    assert mid_price_sharpe(realized_positions, realized, periods_per_year=2)[0] == (
        pytest.approx(mid_price_sharpe(weights, realized, periods_per_year=2)[0])
    )


def test_fill_contract_requires_prices_on_every_trade():
    axes = ([1], ["A"], [300])
    with pytest.raises(ValueError, match="positive price"):
        Fills([[[0.1]]], [[[np.nan]]], [[[100.0]]], *axes)
    with pytest.raises(ValueError, match="quantity must be finite"):
        Fills([[[np.nan]]], [[[100.0]]], [[[100.0]]], *axes)
    # Zero quantity needs no price.
    Fills([[[0.0]]], [[[np.nan]]], [[[np.nan]]], *axes)


def test_simulate_fills_validates_quotes():
    weights, _ = _panel()
    orders = orders_from_weights(weights)
    with pytest.raises(ValueError, match="below best_bid"):
        simulate_fills(orders, np.full((2, 2), 101.0), np.full((2, 2), 99.0))
    with pytest.raises(ValueError, match="must have shape"):
        simulate_fills(orders, np.full((3, 2), 99.0), np.full((3, 2), 101.0))


def test_a_crossing_fill_meets_the_touch_at_arrival_and_is_measured_from_the_decision():
    from stable_finance import MarketData, Orders, execute

    decisions, assets = np.arange(2), ["A", "B"]
    bid, ask = np.full((2, 2), 99.0), np.full((2, 2), 101.0)
    # By the time the order arrives the market has moved up a dollar.
    market = MarketData({"bid_price": bid, "ask_price": ask,
                         "exec_bid_price": bid + 1, "exec_ask_price": ask + 1}, decisions, assets)
    orders = Orders(np.array([[[0.01], [-0.01]]] * 2), decisions, assets, [600])
    touch = execute(orders, market)
    assert touch.price[0, 0, 0] == pytest.approx(102.0)        # the ask on arrival
    assert touch.price[0, 1, 0] == pytest.approx(100.0)        # the bid on arrival
    np.testing.assert_allclose(touch.mid_price, 100.0)          # cost is from the decision mid
    half = execute(orders, market, efq=0.5)
    assert half.price[0, 0, 0] == pytest.approx(101.0 * (1 + 0.5 / 101.0))
    # A midpoint fill is worked at the price the return is measured from.
    free = execute(orders, market, efq=0.0)
    np.testing.assert_allclose(free.price, 100.0)
    # Without arrival quotes the decision's touch stands in.
    same = execute(orders, market.select("bid_price", "ask_price"))
    assert same.price[0, 0, 0] == pytest.approx(101.0)
