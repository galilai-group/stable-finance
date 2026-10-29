"""Weighting rules: the risk-model and forecast-driven ways to size a leg."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from stable_finance import ForwardReturns, mean_variance_target, weight_book
from stable_finance.evaluation.portfolio import construction
from stable_finance.evaluation.portfolio.construction import (
    WEIGHTING_REQUIRES,
    WEIGHTING_RULES,
    _bisection_plan,
    _hrp,
    _hrp_distance,
    _min_var,
    _min_var_qp,
    _min_var_qp_book,
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


def cycling_sigma():
    """A two-factor model on which jumping to each working set's solution and
    dropping its most negative weight readmits and re-drops the same names
    forever."""
    rng = np.random.default_rng(54)
    factors = rng.normal(size=(12, 2)) * rng.uniform(0.1, 3, size=(1, 2)) + rng.normal(size=(1, 2))
    return factors @ factors.T + np.diag(rng.uniform(1e-3, 1e-1, 12))


def test_min_var_qp_solves_the_covariance_that_cycles_the_drop_most_negative_shortcut():
    # The ratio test settles it, and on the brute-force optimum.
    sigma = cycling_sigma()
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


def reference_min_var_qp(sigma):
    """The solver as first written: an eigenvalue check, then LU for every solve.

    Kept here so the solver can be held to it bit for bit. The warm start's
    Cholesky factorization and weight_book's one check of a whole book change
    how fast the book is found, and must never change its digits.
    """
    n = len(sigma)
    if n == 1:
        return np.ones(1)
    sigma = (sigma + sigma.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(sigma)
    assert eigenvalues[0] > eigenvalues[-1] / 1e12
    active = np.ones(n, dtype=bool)
    while True:
        index = np.flatnonzero(active)
        raw = np.linalg.solve(sigma[np.ix_(index, index)], np.ones(len(index)))
        if not (raw < 0).any():
            break
        active[index[raw < 0]] = False
    weights = np.zeros(n)
    weights[index] = raw / raw.sum()
    for _ in range(2 * n):
        index = np.flatnonzero(active)
        raw = np.linalg.solve(sigma[np.ix_(index, index)], np.ones(len(index)))
        target = np.zeros(n)
        target[index] = raw / raw.sum()
        step = target - weights
        falling = index[step[index] < 0]
        if len(falling):
            ratios = weights[falling] / -step[falling]
            first = int(np.argmin(ratios))
            if ratios[first] < 1.0:
                weights = np.maximum(weights + ratios[first] * step, 0.0)
                weights[falling[first]] = 0.0
                active[falling[first]] = False
                continue
        weights = target
        marginal = sigma @ weights
        level = float(weights @ marginal)
        slack = marginal - level
        violated = np.flatnonzero(~active & (slack < -1e-10 * level))
        if len(violated) == 0:
            return weights
        active[violated[np.argmin(slack[violated])]] = True
    raise AssertionError("the reference did not settle")


def sample_sigma(n, observations, seed):
    """A shrunk sample covariance of fat-tailed returns with a common factor."""
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(4, size=(observations, n)) * rng.uniform(0.005, 0.03, size=n)
    returns += rng.normal(size=(observations, 1)) * 0.01
    sample = np.cov(returns, rowvar=False)
    return 0.8 * sample + 0.2 * np.diag(np.diag(sample))


def test_min_var_qp_matches_the_straightforward_solver_bit_for_bit():
    two = np.array([[1e-4, 1.9e-4], [1.9e-4, 4e-4]])
    exchangeable = np.full((8, 8), 5e-5) + np.eye(8) * 5e-5     # every ratio and slack tied
    asymmetric = factor_sigma(30, seed=5)
    asymmetric[3, 7] += 1e-19
    cases = [binding_sigma(), cycling_sigma(), two, exchangeable, asymmetric,
             factor_sigma(150, seed=6), sample_sigma(120, 100, seed=7),
             factor_sigma(40, seed=8) + np.diag(np.full(40, 1e-9))]
    cases += [factor_sigma(n, seed, k=1 + seed % 3) for seed, n in enumerate(range(2, 40, 3))]
    for sigma in cases:
        np.testing.assert_array_equal(_min_var_qp(sigma, np.zeros(len(sigma))),
                                      reference_min_var_qp(sigma))


def test_min_var_qp_falls_back_to_lu_where_the_cholesky_factorization_breaks_down(monkeypatch):
    import scipy.linalg.lapack

    sigma = factor_sigma(40, seed=9)
    # A breakdown reported, beside a solution that would short every name if
    # it were believed.
    monkeypatch.setattr(scipy.linalg.lapack, "dposv", lambda a, b, lower: (a, -b, 1))
    np.testing.assert_array_equal(_min_var_qp(sigma, np.zeros(40)), reference_min_var_qp(sigma))


def long_short_book(n_assets=30, n_decisions=12, seed=0):
    """Every name held, the upper half of each decision's forecast long."""
    forecast = np.random.default_rng(seed).normal(size=(n_decisions, n_assets))
    upper = forecast > np.median(forecast, axis=1, keepdims=True)
    return np.where(upper, 1, -1).astype(np.int8), forecast


def assert_sized_decision_by_decision(signs, forecast, sigma):
    """weight_book on the whole book equals it one decision at a time, where
    the book is too small for the one check and every leg checks itself."""
    book = weight_book("min_var_qp", signs, forecast, sigma)
    for decision in range(len(signs)):
        alone = slice(decision, decision + 1)
        assert _min_var_qp_book(sigma, signs[alone])[0] is _min_var_qp
        np.testing.assert_array_equal(book[alone],
                                      weight_book("min_var_qp", signs[alone], forecast[alone], sigma))
    return book


def test_weight_book_checks_a_min_var_qp_book_once_and_sizes_each_leg_as_alone():
    sigma = factor_sigma(30, seed=10)
    sigma[3, 7] += 1e-19      # symmetrized once for the book, and per leg alone
    signs, forecast = long_short_book()
    assert _min_var_qp_book(sigma, signs)[0] is not _min_var_qp, "should be checked once"
    book = assert_sized_decision_by_decision(signs, forecast, sigma)
    for decision in range(len(signs)):
        for leg in (1, -1):
            members = np.flatnonzero(signs[decision] == leg)
            reference = reference_min_var_qp(sigma[np.ix_(members, members)])
            np.testing.assert_array_equal(book[decision, members], leg * reference / reference.sum())


def test_a_min_var_qp_book_the_union_cannot_vouch_for_is_checked_leg_by_leg():
    signs, forecast = long_short_book()
    # One name and its duplicate, held on opposite sides: the union is
    # singular, yet no leg is.
    duplicated = factor_sigma(30, seed=11)
    duplicated[0], duplicated[:, 0] = duplicated[1], duplicated[:, 1]
    signs[:, 0], signs[:, 1] = 1, -1
    assert _min_var_qp_book(duplicated, signs)[0] is _min_var_qp
    assert_sized_decision_by_decision(signs, forecast, duplicated)
    # A NaN where a long name meets a short one lies in no leg's block.
    holed = factor_sigma(30, seed=12)
    holed[0, 1] = holed[1, 0] = np.nan
    assert _min_var_qp_book(holed, signs)[0] is _min_var_qp
    assert_sized_decision_by_decision(signs, forecast, holed)
    # The duplicate inside one leg is refused, as that leg alone would be.
    signs[:, 1] = 1
    with pytest.raises(ValueError, match="positive definite"):
        weight_book("min_var_qp", signs, forecast, duplicated)


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


def reference_variance(sigma):
    variance = np.diag(sigma).astype(np.float64)
    positive = variance[np.isfinite(variance) & (variance > 0)]
    floor = positive.min() * 1e-12 if len(positive) else 1.0
    return np.where(np.isfinite(variance) & (variance > 0), variance, floor)


def reference_distance(sigma):
    """HRP's condensed distance as first written, on fresh arrays."""
    from scipy.spatial.distance import squareform

    volatility = np.sqrt(reference_variance(sigma))
    correlation = np.clip(sigma / np.outer(volatility, volatility), -1.0, 1.0)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.sqrt(np.maximum((1.0 - correlation) / 2.0, 0.0))
    distance = np.maximum((distance + distance.T) / 2.0, 0.0)
    np.fill_diagonal(distance, 0.0)
    return squareform(distance, checks=False)


def reference_hrp(sigma):
    """HRP as first written: the bisection one cluster at a time.

    The definition :func:`_hrp` must reproduce -- the recursion of the
    docstring, a FIFO of leaf-order slices, one gathered quadratic form per
    half -- slow, and transparently right.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    n = len(sigma)
    if n == 1:
        return np.ones(1)
    variance = reference_variance(sigma)
    order = leaves_list(linkage(reference_distance(sigma), method="single"))

    def cluster_variance(members):
        inverse = 1.0 / variance[members]
        weights = inverse / inverse.sum()
        return float(weights @ sigma[np.ix_(members, members)] @ weights)

    weights = np.ones(n)
    pending = [order]
    while pending:
        members = pending.pop(0)
        if len(members) < 2:
            continue
        half = len(members) // 2
        left, right = members[:half], members[half:]
        left_var, right_var = cluster_variance(left), cluster_variance(right)
        total = left_var + right_var
        share = right_var / total if total > 0 and np.isfinite(total) else 0.5
        weights[left] *= share
        weights[right] *= 1.0 - share
        pending.extend((left, right))
    return weights


def awkward_covariances():
    """Leg covariances that stress the tree and the cluster variances."""
    cases = {f"factor-{n}": factor_sigma(n, seed=n) for n in (3, 4, 5, 7, 8, 16, 31, 33, 97, 128)}
    duplicated = factor_sigma(9, seed=1)
    index = np.r_[np.arange(9), [0, 3, 3, 5]]
    cases["duplicate names"] = duplicated[np.ix_(index, index)]
    cases["equal correlations"] = (np.full((12, 12), 0.3) + 0.7 * np.eye(12)) * 1e-4
    loadings = np.random.default_rng(2).normal(size=(40, 1)) + 1.0
    cases["one factor, almost no idiosyncratic risk"] = loadings @ loadings.T * 1e-4 + 1e-14 * np.eye(40)
    asymmetric = factor_sigma(50, seed=3)
    cases["symmetric only to rounding"] = asymmetric * (
        1.0 + 1e-15 * np.random.default_rng(3).normal(size=asymmetric.shape))
    degenerate = factor_sigma(20, seed=4)
    degenerate[0, 0], degenerate[1, 1], degenerate[2, 2] = 0.0, -1e-6, np.nan
    cases["zero, negative and NaN variances"] = degenerate
    undefined = factor_sigma(10, seed=5)
    undefined[0, 3] = undefined[3, 0] = np.nan
    cases["an undefined covariance"] = undefined
    indefinite = factor_sigma(12, seed=6)
    indefinite[0, 1] = indefinite[1, 0] = 5 * np.sqrt(indefinite[0, 0] * indefinite[1, 1])
    cases["indefinite"] = indefinite
    cases["tiny scale"] = factor_sigma(30, seed=7) * 1e-200
    cases["huge scale"] = factor_sigma(30, seed=8) * 1e200
    return cases


AWKWARD = awkward_covariances()


@pytest.mark.parametrize("label", AWKWARD)
def test_hrp_distance_is_the_reference_distance_bit_for_bit(label):
    # The distance decides the tree: any bit that moved could reorder a tie.
    sigma = AWKWARD[label]
    volatility = np.sqrt(reference_variance(sigma))
    np.testing.assert_array_equal(_hrp_distance(sigma, volatility), reference_distance(sigma))


@pytest.mark.parametrize("label", AWKWARD)
def test_hrp_matches_the_cluster_at_a_time_bisection(label):
    sigma = AWKWARD[label]
    np.testing.assert_allclose(_hrp(sigma, np.zeros(len(sigma))), reference_hrp(sigma),
                               rtol=1e-12, atol=0)


def test_hrp_one_and_two_name_legs_match_the_reference_bit_for_bit():
    for seed in range(20):
        sigma = factor_sigma(2, seed=seed)
        np.testing.assert_array_equal(_hrp(sigma, np.zeros(2)), reference_hrp(sigma))
    np.testing.assert_array_equal(_hrp(np.full((2, 2), np.nan), np.zeros(2)), [0.5, 0.5])


@pytest.mark.parametrize("n", range(1, 70))
def test_bisection_plan_lists_the_clusters_the_recursion_visits(n):
    expected, frontier = [], [(0, n)]
    while any(length > 1 for _, length in frontier):
        splits = [(start, length) for start, length in frontier if length > 1]
        frontier = sorted(
            [(start, 1) for start, length in frontier if length == 1]
            + [(start, length // 2) for start, length in splits]
            + [(start + length // 2, length - length // 2) for start, length in splits])
        expected.append((splits, frontier))
    plan = _bisection_plan(n)
    assert len(plan) == len(expected)
    for (left, right, starts, owner, singles), (splits, frontier) in zip(plan, expected, strict=True):
        assert left.tolist() == [start for start, _ in splits]
        assert right.tolist() == [start + length // 2 for start, length in splits]
        assert starts.tolist() == [start for start, _ in frontier]
        assert owner.tolist() == [start for start, length in frontier for _ in range(length)]
        assert singles.tolist() == [start for start, length in frontier if length == 1]
        assert not any(array.flags.writeable for array in (left, right, starts, owner, singles))


def test_weight_book_hrp_matches_the_reference_with_unheld_names_and_nan_forecasts(monkeypatch):
    rng = np.random.default_rng(11)
    n_assets = 60
    sigma = factor_sigma(n_assets, seed=11)
    signs = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(6, n_assets), p=[0.4, 0.2, 0.4])
    signs[4, :] = 0
    signs[5, :] = 0
    signs[5, [7, 20]] = 1                                # a one-leg, two-name book
    forecast = rng.normal(size=signs.shape)
    forecast[rng.random(signs.shape) < 0.1] = np.nan
    fast = weight_book("hrp", signs, forecast, sigma)
    monkeypatch.setitem(construction._RULES, "hrp", lambda s, score: reference_hrp(s))
    np.testing.assert_allclose(fast, weight_book("hrp", signs, forecast, sigma), rtol=1e-12, atol=0)


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
