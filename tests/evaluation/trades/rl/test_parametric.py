import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation.trades.rl.parametric import (
    ParametricPolicy,
    _characteristics,
    _costs,
    _objective_and_gradient,
    to_weights,
)


def panel(n_decisions=300, n_assets=20, persistence=0.0, seed=1):
    """Returns driven by a latent signal, with one informative and one noise column.

    With no persistence the informative column is the realized return plus
    noise; with persistence it is a noisy read of a slowly moving signal, the
    setting where trading only part of the way to the target pays.
    """
    rng = np.random.default_rng(seed)
    signal = np.empty((n_decisions, n_assets))
    state = rng.normal(size=n_assets)
    for t in range(n_decisions):
        state = persistence * state + np.sqrt(1 - persistence ** 2) * rng.normal(size=n_assets)
        signal[t] = state
    returns = 0.01 * signal + 0.02 * rng.normal(size=signal.shape)
    if persistence == 0:
        informative = returns + 0.02 * rng.normal(size=signal.shape)
    else:
        informative = 0.01 * signal + 0.005 * rng.normal(size=signal.shape)
    noise = rng.normal(size=signal.shape)
    return np.stack([informative, noise], axis=2), returns


def turnover(held):
    previous = np.vstack([np.zeros(held.shape[1]), held[:-1]])
    return np.abs(held - previous).sum(axis=1).mean()


def test_the_analytic_gradient_matches_central_finite_differences():
    rng = np.random.default_rng(0)
    n_decisions, n_assets = 8, 5
    raw = rng.normal(size=(n_decisions, n_assets, 3))
    raw[3, 2, 1] = np.nan
    X, active = _characteristics(raw, standardize=True)
    returns = rng.normal(scale=0.02, size=(n_decisions, n_assets))
    cost = rng.uniform(0, 0.01, size=(n_decisions, n_assets))
    cost[2, 1] = np.inf
    charge, tradable = _costs(cost, (n_decisions, n_assets))
    initial = rng.normal(scale=0.1, size=n_assets)
    settings = {"gross": 2.0, "risk_aversion": 5.0, "l2": 0.1, "epsilon": 1e-6}

    def utility(theta, rho):
        return _objective_and_gradient(theta, rho, X, active, returns, charge,
                                       tradable, initial, **settings)

    theta, rho = np.array([0.3, -0.2, 0.5]), 0.4
    _, grad_theta, grad_rho = utility(theta, rho)
    analytic = np.append(grad_theta, grad_rho)
    step = 1e-6
    numeric = []
    for i in range(3):
        bump = np.zeros(3)
        bump[i] = step
        numeric.append((utility(theta + bump, rho)[0] - utility(theta - bump, rho)[0])
                       / (2 * step))
    numeric.append((utility(theta, rho + step)[0] - utility(theta, rho - step)[0])
                   / (2 * step))
    numeric = np.array(numeric)
    assert np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric) < 1e-5


def test_a_small_learning_rate_never_lowers_the_training_objective():
    X, returns = panel(persistence=0.9)
    policy = ParametricPolicy(learning_rate=1e-3, max_iter=100).fit(
        X, returns, cost=np.full(returns.shape, 0.002))
    path = policy.objective_path_
    assert path.shape == (101,)
    assert np.all(np.diff(path) >= -1e-12)
    assert path[-1] > path[0]


def test_free_trading_learns_to_trade_fully_on_the_informative_characteristic():
    X, returns = panel()
    policy = ParametricPolicy().fit(X, returns)
    assert policy.eta_ > 0.95
    assert abs(policy.theta_[0]) > 10 * abs(policy.theta_[1])
    assert policy.theta_[0] > 0
    assert policy.objective_path_[-1] > policy.objective_path_[0]


def test_a_high_cost_slows_trading_and_cuts_turnover():
    X, returns = panel(persistence=0.9)
    free = ParametricPolicy().fit(X, returns, cost=np.zeros(returns.shape))
    costly = ParametricPolicy().fit(X, returns, cost=np.full(returns.shape, 0.01))
    assert costly.eta_ < free.eta_ - 0.3
    assert turnover(costly.predict(X)) < 0.5 * turnover(free.predict(X))


def test_an_untradable_name_holds_its_previous_weight():
    X, returns = panel(n_decisions=60)
    policy = ParametricPolicy(max_iter=50).fit(X, returns)
    cost = np.zeros(returns.shape)
    cost[10:20, 3] = np.inf
    cost[30:35, 7] = np.nan
    held = policy.predict(X, cost=cost)
    np.testing.assert_array_equal(held[10:20, 3], held[9, 3])
    np.testing.assert_array_equal(held[30:35, 7], held[29, 7])
    frozen = policy.predict(X, cost=np.full(returns.shape, np.inf), initial=0.05)
    np.testing.assert_array_equal(frozen, 0.05)


def test_held_weights_are_dollar_neutral_within_the_requested_gross():
    X, returns = panel(persistence=0.9)
    for gross in (1.0, 2.0):
        policy = ParametricPolicy(gross, max_iter=100).fit(
            X, returns, cost=np.full(returns.shape, 0.002))
        held = policy.predict(X)
        np.testing.assert_allclose(held.sum(axis=1), 0.0, atol=1e-12)
        assert np.all(np.abs(held).sum(axis=1) <= gross + 1e-12)
        assert np.abs(held).sum(axis=1).max() > 0.5 * gross


def test_two_fits_on_the_same_data_are_bitwise_identical():
    X, returns = panel(persistence=0.9)
    cost = np.full(returns.shape, 0.002)
    first = ParametricPolicy(max_iter=100).fit(X, returns, cost=cost)
    second = ParametricPolicy(max_iter=100).fit(X, returns, cost=cost)
    np.testing.assert_array_equal(first.theta_, second.theta_)
    assert first.eta_ == second.eta_
    np.testing.assert_array_equal(first.objective_path_, second.objective_path_)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_the_policy_clones_and_reports_its_hyperparameters_like_an_estimator():
    policy = ParametricPolicy(1.5, risk_aversion=3.0, l2=0.1, max_iter=20)
    params = policy.get_params()
    assert params["gross_exposure"] == 1.5 and params["risk_aversion"] == 3.0
    copy = clone(policy)
    assert copy.get_params() == params
    assert not hasattr(copy, "theta_")
    with pytest.raises(NotFittedError):
        copy.predict(np.zeros((2, 3, 2)))
    copy.set_params(max_iter=5)
    X, returns = panel(n_decisions=20)
    assert copy.fit(X, returns).objective_path_.shape == (6,)


def test_forecast_contracts_go_in_and_portfolio_weights_come_out():
    X, returns = panel(n_decisions=40, n_assets=6)
    decisions, assets = np.arange(40), [f"A{i}" for i in range(6)]
    characteristics = ForwardReturns(X, decisions, assets, [300, 900])
    realized = ForwardReturns(returns[:, :, None], decisions, assets, [300])
    policy = ParametricPolicy(max_iter=20).fit(characteristics, realized)
    weights = policy.predict(characteristics)
    assert isinstance(weights, PortfolioWeights)
    np.testing.assert_array_equal(weights.horizons, [300])
    np.testing.assert_array_equal(weights.values[:, :, 0], policy.predict(X))
    wrapped = to_weights(policy.predict(X), decisions, assets, 300)
    np.testing.assert_array_equal(wrapped.values, weights.values)


def test_invalid_hyperparameters_and_inputs_are_refused():
    X, returns = panel(n_decisions=10)
    with pytest.raises(ValueError, match="gross_exposure"):
        ParametricPolicy(0.0).fit(X, returns)
    with pytest.raises(ValueError, match="max_iter"):
        ParametricPolicy(max_iter=0).fit(X, returns)
    with pytest.raises(ValueError, match="negative"):
        ParametricPolicy().fit(X, returns, cost=-0.01)
    with pytest.raises(ValueError, match="returns must have shape"):
        ParametricPolicy().fit(X, returns[:, :3])
    policy = ParametricPolicy(max_iter=5).fit(X, returns)
    with pytest.raises(ValueError, match="characteristics"):
        policy.predict(X[:, :, :1])
