"""Weighting rules: the risk-model and forecast-driven ways to size a leg."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from stable_finance import ForwardReturns, mean_variance_target, weight_book
from stable_finance.evaluation.portfolio.construction import (
    WEIGHTING_REQUIRES,
    WEIGHTING_RULES,
    _hrp,
    _min_var,
    _min_var_qp,
)

NEW_RULES = ("signal", "min_var_qp", "hrp")


def correlated_sigma(n=5, seed=3):
    """A positive definite covariance whose clipped min-var differs from the QP."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    return factors @ factors.T * 1e-4 + np.diag(rng.uniform(1e-5, 3e-4, n))


def variance(weights, sigma):
    weights = np.asarray(weights) / np.sum(weights)
    return float(weights @ sigma @ weights)


def factor_sigma(n, seed, k=3):
    """A k-factor covariance with a market-like common loading, so the
    unconstrained minimum-variance book shorts names and the constraint binds."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n, k)) * rng.uniform(0.5, 2.0, size=(1, k)) + rng.normal(size=(1, k))
    return loadings @ loadings.T * 1e-4 + np.diag(rng.uniform(1e-5, 4e-4, n))


def assert_kkt(w, sigma, rtol=1e-6):
    """Feasible, equal marginal variance on the names held, none cheaper left out."""
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    gradient = sigma @ w
    support = w > 1e-9
    level = float(w @ gradient)
    np.testing.assert_allclose(gradient[support], level, rtol=rtol)
    assert (gradient[~support] >= level * (1 - rtol)).all()


def brute_force_min_var(sigma):
    """The long-only minimum-variance book by enumerating every support.

    On the optimal support the problem is the budget-constrained one, so the
    optimum is the lowest-variance feasible solution among the ``2^n - 1``
    equality-constrained subproblems -- exact, and independent of any
    active-set logic.
    """
    n = len(sigma)
    best, best_variance = None, np.inf
    for size in range(1, n + 1):
        for support in itertools.combinations(range(n), size):
            support = list(support)
            raw = np.linalg.solve(sigma[np.ix_(support, support)], np.ones(size))
            if raw.sum() <= 0 or (raw < 0).any():
                continue
            w = np.zeros(n)
            w[support] = raw / raw.sum()
            if w @ sigma @ w < best_variance:
                best, best_variance = w, float(w @ sigma @ w)
    return best


# ── registration ────────────────────────────────────────────────────────────

def test_the_new_rules_are_named_and_need_nothing_of_the_market():
    for name in NEW_RULES:
        assert name in WEIGHTING_RULES
        assert WEIGHTING_REQUIRES[name] == ()


# ── diagonal risk model ─────────────────────────────────────────────────────

@pytest.mark.parametrize("rule", [_hrp, _min_var_qp])
def test_hrp_and_min_var_qp_are_inverse_variance_under_a_diagonal_model(rule):
    variances = np.array([1e-4, 4e-4, 9e-4, 2e-4])
    raw = rule(np.diag(variances), np.zeros(4))
    expected = (1.0 / variances) / np.sum(1.0 / variances)
    np.testing.assert_allclose(raw / raw.sum(), expected, rtol=1e-9)


# ── long-only minimum variance ──────────────────────────────────────────────

def binding_sigma():
    """Names 0 and 1 are highly correlated with unequal volatility, so the
    unconstrained minimum-variance book shorts name 1 and the constraint binds."""
    correlation = np.full((4, 4), 0.3)
    np.fill_diagonal(correlation, 1.0)
    correlation[0, 1] = correlation[1, 0] = 0.9
    correlation[2, 3] = correlation[3, 2] = 0.6
    volatility = np.array([0.01, 0.02, 0.015, 0.012])
    return correlation * np.outer(volatility, volatility)


def test_min_var_qp_satisfies_the_kkt_conditions_on_a_correlated_model():
    sigma = binding_sigma()
    assert np.linalg.solve(sigma, np.ones(4)).min() < 0
    w = _min_var_qp(sigma, np.zeros(4))
    assert_kkt(w, sigma)
    assert (w <= 1e-9).any(), "the test covariance should make the constraint bind"


def test_min_var_qp_is_at_least_as_low_variance_as_the_clipped_closed_form():
    sigma = binding_sigma()
    score = np.zeros(4)
    # Strictly lower where clipping actually removed a short.
    assert variance(_min_var_qp(sigma, score), sigma) < variance(_min_var(sigma, score), sigma)
    for seed in range(5):
        sigma = correlated_sigma(seed=seed)
        score = np.zeros(len(sigma))
        assert (variance(_min_var_qp(sigma, score), sigma)
                <= variance(_min_var(sigma, score), sigma) * (1 + 1e-12))


def test_min_var_qp_matches_a_brute_force_search_over_supports():
    cases = [binding_sigma()] + [factor_sigma(n, seed, k=2)
                                 for n in (3, 5, 6, 8) for seed in range(4)]
    bound = 0
    for sigma in cases:
        w = _min_var_qp(sigma, np.zeros(len(sigma)))
        reference = brute_force_min_var(sigma)
        np.testing.assert_allclose(w, reference, atol=1e-10)
        assert variance(w, sigma) <= variance(reference, sigma) * (1 + 1e-12)
        bound += int((reference == 0).any())
    assert bound > len(cases) // 2, "most cases should have names the constraint excludes"


def test_min_var_qp_terminates_and_satisfies_kkt_on_a_60_name_factor_model():
    sigma = factor_sigma(60, seed=11)
    w = _min_var_qp(sigma, np.zeros(60))
    assert_kkt(w, sigma)
    assert 0 < (w > 1e-9).sum() < 60
    np.testing.assert_array_equal(w, _min_var_qp(sigma.copy(), np.zeros(60)))


def test_min_var_qp_solves_the_covariance_that_cycles_the_drop_most_negative_shortcut():
    # Jumping to each working set's solution and dropping its most negative
    # weight readmits and re-drops the same names forever on this two-factor
    # model; the ratio test settles it, and on the brute-force optimum.
    rng = np.random.default_rng(54)
    factors = rng.normal(size=(12, 2)) * rng.uniform(0.1, 3, size=(1, 2)) + rng.normal(size=(1, 2))
    sigma = factors @ factors.T + np.diag(rng.uniform(1e-3, 1e-1, 12))
    w = _min_var_qp(sigma, np.zeros(12))
    assert_kkt(w, sigma)
    np.testing.assert_allclose(w, brute_force_min_var(sigma), atol=1e-10)


@pytest.mark.parametrize("sigma", [
    np.full((3, 3), 1e-4),                                   # one name, three times
    np.array([[1e-4, 1e-4, 0.0], [1e-4, 1e-4, 0.0], [0.0, 0.0, 2e-4]]),  # a duplicate
    np.zeros((2, 2)),                                        # no variance at all
    np.array([[1e-4, 2e-4], [2e-4, 1e-4]]),                  # indefinite
])
def test_min_var_qp_refuses_a_singular_or_indefinite_leg_covariance(sigma):
    with pytest.raises(ValueError, match="positive definite"):
        _min_var_qp(sigma, np.zeros(len(sigma)))


def test_min_var_qp_refuses_a_non_finite_leg_covariance_but_not_a_one_name_leg():
    with pytest.raises(ValueError, match="not finite"):
        _min_var_qp(np.array([[1e-4, np.nan], [np.nan, 1e-4]]), np.zeros(2))
    np.testing.assert_array_equal(_min_var_qp(np.zeros((1, 1)), np.zeros(1)), [1.0])


# ── hierarchical risk parity ────────────────────────────────────────────────

def test_hrp_is_deterministic_and_handles_one_and_two_name_legs():
    sigma = correlated_sigma(n=6)
    first = _hrp(sigma, np.zeros(6))
    np.testing.assert_array_equal(first, _hrp(sigma.copy(), np.zeros(6)))
    assert first.sum() == pytest.approx(1.0) and (first > 0).all()
    np.testing.assert_array_equal(_hrp(np.array([[4e-4]]), np.zeros(1)), [1.0])
    pair = _hrp(np.array([[1e-4, 5e-5], [5e-5, 4e-4]]), np.zeros(2))
    # Two names split by inverse variance whatever their correlation.
    np.testing.assert_allclose(pair, [0.8, 0.2])


def test_hrp_keeps_correlated_names_together_in_the_bisection():
    # Names 0 and 2 are near-duplicates, 1 and 3 another pair: quasi-diagonal
    # ordering must put each pair in the same half, so each pair shares a
    # half of the budget instead of crowding out the other.
    correlation = np.array([[1.0, 0.0, 0.95, 0.0],
                            [0.0, 1.0, 0.0, 0.95],
                            [0.95, 0.0, 1.0, 0.0],
                            [0.0, 0.95, 0.0, 1.0]])
    w = _hrp(correlation * 1e-4, np.zeros(4))
    np.testing.assert_allclose(w, 0.25, rtol=1e-12)


# ── signal weighting and the centred score ──────────────────────────────────

def test_signal_weights_are_proportional_to_distance_from_the_book_median():
    signs = np.array([[1, 1, 1, -1, -1, -1]], dtype=np.int8)
    forecast = np.array([[7.0, 4.0, 5.0, 1.0, 2.5, 3.0]])
    weights = weight_book("signal", signs, forecast, np.eye(6), gross_exposure=2.0)
    centre = np.median(forecast)
    long_distance = forecast[0, :3] - centre
    short_distance = centre - forecast[0, 3:]
    np.testing.assert_allclose(weights[0, :3], long_distance / long_distance.sum())
    np.testing.assert_allclose(weights[0, 3:], -short_distance / short_distance.sum())


def test_the_signal_book_ignores_the_forecasts_offset():
    signs = np.array([[1, 1, -1, -1, 0, 0]], dtype=np.int8)
    forecast = np.array([[3.0, 2.0, 1.0, 0.5, 0.0, 0.0]])
    base = weight_book("signal", signs, forecast, np.eye(6))
    shifted = weight_book("signal", signs, forecast + 100.0, np.eye(6))
    np.testing.assert_allclose(shifted, base, rtol=1e-12)


def test_rank_weights_are_unchanged_by_centring_the_score():
    signs = np.array([[1, 1, 1, -1, -1, 0]], dtype=np.int8)
    forecast = np.array([[0.9, 0.2, 0.5, -0.3, 0.1, 9.0]])
    weights = weight_book("rank", signs, forecast, np.eye(6), gross_exposure=2.0)
    # The rank formula on the raw, leg-oriented forecast.
    long_rank = np.array([3.0, 1.0, 2.0])
    short_rank = np.array([2.0, 1.0])  # -0.3 is the short leg's best name
    np.testing.assert_allclose(weights[0, :3], long_rank / long_rank.sum())
    np.testing.assert_allclose(weights[0, 3:5], -short_rank / short_rank.sum())
    assert weights[0, 5] == 0.0


@pytest.mark.parametrize("name", NEW_RULES)
def test_every_new_rule_keeps_the_book_dollar_neutral_at_the_requested_gross(name):
    sigma = np.eye(8) * 1e-6
    sigma[:4, :4] = correlated_sigma(n=4, seed=1)
    sigma[4:, 4:] = correlated_sigma(n=4, seed=2)
    signs = np.array([[1, 1, 1, 1, -1, -1, -1, -1],
                      [1, -1, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    forecast = np.array([[5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0],
                         [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    weights = weight_book(name, signs, forecast, sigma, gross_exposure=1.5)
    np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(weights).sum(axis=1), 1.5)
    assert (np.sign(weights) == signs).all()


# ── per-decision risk aversion ──────────────────────────────────────────────

def mv_panel(seed=0, n_decisions=4, n_assets=3):
    rng = np.random.default_rng(seed)
    forecast = ForwardReturns(rng.normal(0, 1e-3, (n_decisions, n_assets, 1)),
                              np.arange(n_decisions), [f"A{i}" for i in range(n_assets)],
                              [300])
    return forecast, correlated_sigma(n=n_assets, seed=seed)[None]


def test_a_constant_risk_aversion_array_equals_the_scalar_exactly():
    forecast, sigma = mv_panel()
    scalar = mean_variance_target(forecast, sigma, risk_aversion=2.5).values
    array = mean_variance_target(forecast, sigma, risk_aversion=np.full(4, 2.5)).values
    np.testing.assert_array_equal(array, scalar)


def test_each_risk_aversion_scales_only_its_own_decisions_target():
    forecast, sigma = mv_panel()
    aversion = np.array([1.0, 2.0, 4.0, 8.0])
    target = mean_variance_target(forecast, sigma, risk_aversion=aversion).values
    unit = mean_variance_target(forecast, sigma, risk_aversion=1.0).values
    np.testing.assert_allclose(target, unit / aversion[:, None, None], rtol=1e-12)
    bumped = aversion.copy()
    bumped[2] = 16.0
    moved = mean_variance_target(forecast, sigma, risk_aversion=bumped).values
    np.testing.assert_array_equal(np.delete(moved, 2, axis=0), np.delete(target, 2, axis=0))
    assert not np.allclose(moved[2], target[2])


@pytest.mark.parametrize("bad, message", [
    (np.ones(3), "one value per decision"),
    (np.ones((4, 1)), "one value per decision"),
    (np.array([1.0, 0.0, 1.0, 1.0]), "positive"),
    (-1.0, "positive"),
    (np.array([1.0, np.nan, 1.0, 1.0]), "finite"),
])
def test_mean_variance_target_rejects_a_malformed_risk_aversion(bad, message):
    forecast, sigma = mv_panel()
    with pytest.raises(ValueError, match=message):
        mean_variance_target(forecast, sigma, risk_aversion=bad)
