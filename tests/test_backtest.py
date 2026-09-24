import numpy as np
import pytest

from stable_finance import (
    BacktestConfig,
    CONFIG_SPACE,
    DEFAULT_BACKTEST,
    Embeddings,
    ForwardReturns,
    diagonal_covariance,
    embedding_covariance,
    enumerate_configs,
    estimate_risk,
    ledoit_wolf_covariance,
    liquidity_screen,
    presence_screen,
    run_backtest,
    select_book,
    tail_selection,
    trading_cost,
    weight_book,
)

ASSETS = [f"A{i}" for i in range(6)]
AXIS = np.array([900])


def panel(values):
    values = np.asarray(values, dtype=float)
    return ForwardReturns(values[:, :, None], np.arange(len(values)), ASSETS, AXIS)


def world(seed=0, n_decisions=80, n_assets=6):
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(n_decisions // 8), 8)
    truth = rng.normal(scale=0.004, size=(n_decisions, n_assets))
    realized = truth + rng.normal(scale=0.008, size=(n_decisions, n_assets))
    forecast = truth + rng.normal(scale=0.004, size=(n_decisions, n_assets))
    history = rng.normal(scale=0.008, size=(n_decisions, n_assets))
    spread = np.abs(rng.normal(2e-4, 5e-5, size=(n_decisions, n_assets)))
    return dict(forecast=panel(forecast), realized=panel(realized),
                history=panel(history), half_spread=spread, days=days)


# ── configuration ───────────────────────────────────────────────────────────

def test_default_config_is_valid_and_labelled():
    assert DEFAULT_BACKTEST.universe == "all"
    assert "weighting=mean_variance" in DEFAULT_BACKTEST.label


@pytest.mark.parametrize("axis", sorted(CONFIG_SPACE))
def test_every_axis_rejects_an_unknown_option(axis):
    with pytest.raises(ValueError, match=axis):
        BacktestConfig(**{axis: "not-an-option"})


def test_scalar_axes_are_validated():
    with pytest.raises(ValueError, match="gross_exposure"):
        BacktestConfig(gross_exposure=0.0)
    with pytest.raises(ValueError, match="shrinkage"):
        BacktestConfig(shrinkage=1.5)


def test_enumerate_drops_configs_that_duplicate_another_portfolio():
    space = enumerate_configs()
    assert len(space) == len(set(config.label for config in space))
    # Equal weighting ignores the covariance, so it appears under exactly one
    # risk model rather than once per model.
    equal = {c.risk_model for c in space if c.weighting == "equal"}
    assert equal == {"diagonal"}


def test_enumerate_accepts_a_scalar_or_a_list_for_an_axis():
    assert {c.efq for c in enumerate_configs(efq=0.70)} == {0.70}
    sliced = enumerate_configs(efq=(0.0, 1.0), universe="all")
    assert {c.universe for c in sliced} == {"all"}
    assert {c.efq for c in sliced} == {0.0, 1.0}


def test_enumerate_rejects_an_unknown_keyword():
    with pytest.raises(ValueError, match="not a configuration axis"):
        enumerate_configs(nonsense=("a",))


# ── universe ────────────────────────────────────────────────────────────────

def test_liquidity_screen_keeps_the_tightest_fraction_of_each_decision():
    spread = np.array([[1.0, 2.0, 3.0, 4.0]]) * 1e-4
    kept = liquidity_screen(spread, keep=0.5)
    np.testing.assert_array_equal(kept, [[True, True, False, False]])


def test_liquidity_screen_never_empties_a_non_empty_cross_section():
    spread = np.array([[1e-4, 2e-4]])
    assert liquidity_screen(spread, keep=0.01).sum() == 1


def test_screens_exclude_names_without_a_two_sided_market():
    spread = np.array([[1e-4, np.nan, 2e-4]])
    np.testing.assert_array_equal(
        liquidity_screen(spread, keep=1.0), [[True, False, True]]
    )


def test_presence_screen_is_an_asset_mask():
    observed = np.array([[True, True], [True, False], [True, False]])
    np.testing.assert_array_equal(presence_screen(observed, minimum=0.6), [True, False])


# ── selection ───────────────────────────────────────────────────────────────

def test_tail_selection_legs_do_not_overlap_and_respect_the_fraction():
    forecast = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    eligible = np.ones_like(forecast, dtype=bool)
    signs = tail_selection(forecast, eligible, fraction=0.5)
    assert (signs == 1).sum() == 3 and (signs == -1).sum() == 3
    np.testing.assert_array_equal(signs, [[-1, -1, -1, 1, 1, 1]])


def test_long_only_selection_takes_no_shorts():
    forecast = np.array([[1.0, 2.0, 3.0, 4.0]])
    eligible = np.ones_like(forecast, dtype=bool)
    signs = select_book("decile_long_only", forecast, eligible)
    assert (signs == -1).sum() == 0 and (signs == 1).sum() == 1


def test_selection_ignores_screened_out_names():
    forecast = np.array([[1.0, 5.0, 2.0, 4.0]])
    eligible = np.array([[True, False, True, True]])
    signs = select_book("all", forecast, eligible)
    assert signs[0, 1] == 0


# ── weighting ───────────────────────────────────────────────────────────────

def test_long_short_book_is_dollar_neutral_at_the_requested_gross():
    signs = np.array([[1, 1, -1, -1, 0, 0]], dtype=np.int8)
    forecast = np.array([[4.0, 3.0, 2.0, 1.0, 0.0, 0.0]])
    weights = weight_book("equal", signs, forecast, np.eye(6), gross_exposure=2.0)
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(weights).sum() == pytest.approx(2.0)


def test_a_one_legged_book_carries_the_whole_exposure():
    signs = np.array([[1, 1, 0, 0, 0, 0]], dtype=np.int8)
    forecast = np.zeros((1, 6))
    weights = weight_book("equal", signs, forecast, np.eye(6), gross_exposure=1.0)
    assert weights.sum() == pytest.approx(1.0)


def test_inverse_vol_and_erc_agree_under_a_diagonal_risk_model():
    """The degeneracy enumerate_configs relies on when it drops diagonal+erc."""
    sigma = np.diag([1e-4, 4e-4, 9e-4, 1e-4, 4e-4, 9e-4])
    signs = np.array([[1, 1, 1, -1, -1, -1]], dtype=np.int8)
    forecast = np.arange(6, dtype=float)[None]
    a = weight_book("inverse_vol", signs, forecast, sigma)
    b = weight_book("erc", signs, forecast, sigma)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_min_var_on_a_diagonal_model_is_inverse_variance():
    sigma = np.diag([1e-4, 4e-4, 1.0, 1.0, 1.0, 1.0])
    signs = np.array([[1, 1, 0, 0, 0, 0]], dtype=np.int8)
    weights = weight_book("min_var", signs, np.zeros((1, 6)), sigma,
                          gross_exposure=1.0)
    assert weights[0, 0] / weights[0, 1] == pytest.approx(4.0)


def test_rank_weighting_orients_the_short_leg_by_its_own_best_name():
    signs = np.array([[0, 0, 0, -1, -1, -1]], dtype=np.int8)
    forecast = np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])
    weights = weight_book("rank", signs, forecast, np.eye(6), gross_exposure=1.0)
    # The lowest forecast is the short leg's strongest conviction.
    assert abs(weights[0, 3]) > abs(weights[0, 5])


# ── costs ───────────────────────────────────────────────────────────────────

def test_mid_charges_nothing_and_cross_charges_the_half_spread():
    spread = np.array([[1e-4, 2e-4]])
    np.testing.assert_array_equal(trading_cost("mid", spread), [[0.0, 0.0]])
    np.testing.assert_allclose(trading_cost("cross", spread), spread)


def test_the_add_on_is_charged_in_basis_points_on_top_of_the_spread():
    spread = np.array([[1e-4]])
    np.testing.assert_allclose(trading_cost("cross+2bps", spread), [[1e-4 + 2e-4]])


def test_an_unquoted_name_costs_infinity_rather_than_nothing():
    assert np.isinf(trading_cost("cross", np.array([[np.nan]]))[0, 0])


def test_unknown_cost_model_is_rejected():
    with pytest.raises(ValueError, match="unknown cost model"):
        trading_cost("free", np.array([[1e-4]]))


# ── risk models ─────────────────────────────────────────────────────────────

def test_diagonal_model_has_no_correlations():
    sigma = diagonal_covariance(world()["history"])[0]
    assert np.allclose(sigma - np.diag(np.diag(sigma)), 0.0)


@pytest.mark.parametrize("name", ["diagonal", "shrunk_sample", "ledoit_wolf"])
def test_every_return_only_risk_model_is_positive_semidefinite(name):
    sigma = estimate_risk(name, world()["history"])[0]
    assert np.linalg.eigvalsh(sigma).min() >= 0


def test_ledoit_wolf_shrinks_a_short_history_toward_the_identity():
    short = panel(np.random.default_rng(1).normal(scale=0.01, size=(8, 6)))
    sigma = ledoit_wolf_covariance(short)[0]
    off = np.abs(sigma - np.diag(np.diag(sigma))).max()
    assert np.linalg.eigvalsh(sigma).min() > 0
    assert off < np.diag(sigma).max()


def test_identical_embeddings_forecast_a_perfect_correlation():
    history = world()["history"]
    repeated = np.repeat(np.ones((80, 1, 4)), 6, axis=1)
    sigma = embedding_covariance(
        history, Embeddings(repeated, np.arange(80), ASSETS), shrinkage=0.0
    )[0]
    correlation = sigma / np.outer(np.sqrt(np.diag(sigma)), np.sqrt(np.diag(sigma)))
    np.testing.assert_allclose(correlation, np.ones((6, 6)), atol=1e-8)


def test_embedding_risk_model_requires_embeddings():
    with pytest.raises(ValueError, match="needs embeddings"):
        estimate_risk("embedding", world()["history"])


def test_unknown_risk_model_is_rejected():
    with pytest.raises(ValueError, match="unknown risk model"):
        estimate_risk("vibes", world()["history"])


# ── the wiring ──────────────────────────────────────────────────────────────

def test_the_default_configuration_runs_and_reports_both_sharpes():
    report = run_backtest(**world())
    assert report.config == DEFAULT_BACKTEST
    assert report.horizon == 900
    assert np.isfinite(report.sharpe_mid) and np.isfinite(report.sharpe_net)


def test_charging_a_cost_never_improves_the_net_result():
    setup = world()
    free = run_backtest(**setup, config=BacktestConfig(cost_model="mid"))
    paid = run_backtest(**setup, config=BacktestConfig(cost_model="cross+2bps"))
    assert paid.mean_cost_bps >= free.mean_cost_bps
    assert free.mean_cost_bps == pytest.approx(0.0)


def test_the_information_coefficient_does_not_depend_on_the_cost_model():
    """IC is a property of the forecast; only the portfolio sees the cost."""
    setup = world()
    a = run_backtest(**setup, config=BacktestConfig(cost_model="mid"))
    b = run_backtest(**setup, config=BacktestConfig(cost_model="cross+2bps"))
    assert a.information_coefficient == pytest.approx(b.information_coefficient)


def test_holding_the_book_for_a_day_cuts_turnover():
    setup = world()
    every = run_backtest(**setup, config=BacktestConfig(rebalance="every_decision"))
    daily = run_backtest(**setup, config=BacktestConfig(rebalance="daily"))
    assert daily.mean_turnover < every.mean_turnover
    # Holding does not change how much is held, only how often it is retraded.
    assert daily.mean_gross_exposure == pytest.approx(every.mean_gross_exposure,
                                                      rel=0.35)


def test_annualization_changes_the_period_count_not_the_book():
    setup = world()
    per_decision = run_backtest(**setup, config=BacktestConfig(annualization="decisions"))
    per_day = run_backtest(**setup, config=BacktestConfig(annualization="daily"))
    assert per_decision.periods_per_year == 8 * 252
    assert per_day.periods_per_year == 252
    assert per_day.n_periods == 10 and per_decision.n_periods == 80
    assert per_day.mean_turnover == pytest.approx(per_decision.mean_turnover)


def test_a_tighter_universe_holds_fewer_names():
    setup = world()
    wide = run_backtest(**setup, config=BacktestConfig(universe="all"))
    tight = run_backtest(**setup, config=BacktestConfig(universe="tight_25pct"))
    assert tight.mean_names_held < wide.mean_names_held


def test_half_spread_shape_is_validated():
    setup = world()
    setup["half_spread"] = setup["half_spread"][:, :3]
    with pytest.raises(ValueError, match="half_spread must have shape"):
        run_backtest(**setup)


def test_days_length_is_validated():
    setup = world()
    setup["days"] = setup["days"][:-1]
    with pytest.raises(ValueError, match="days must have length"):
        run_backtest(**setup)


def test_report_flattens_to_a_row_carrying_its_configuration():
    row = run_backtest(**world()).as_dict()
    assert row["weighting"] == "mean_variance" and "sharpe_net" in row


def test_erc_equalizes_risk_contributions_on_a_correlated_covariance():
    """The defining property, which the diagonal case cannot distinguish."""
    correlation = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, 0.4], [0.2, 0.4, 1.0]])
    volatility = np.array([0.01, 0.02, 0.03])
    sigma = correlation * np.outer(volatility, volatility)
    padded = np.eye(6) * 1e-6
    padded[:3, :3] = sigma
    signs = np.array([[1, 1, 1, 0, 0, 0]], dtype=np.int8)
    weights = weight_book("erc", signs, np.zeros((1, 6)), padded,
                          gross_exposure=1.0)[0, :3]
    contributions = weights * (sigma @ weights)
    np.testing.assert_allclose(contributions, contributions.mean(), rtol=1e-4)


def test_selection_changes_the_mean_variance_book_too():
    """Otherwise four selection options would name one identical portfolio.

    The count of names held does NOT fall: a name dropped by the selection
    keeps a zero expected return but stays in the risk problem, so the
    allocator may still hold it as a hedge. What changes is the sizing, and
    that is what the scores below are asserting.
    """
    setup = world()
    whole = run_backtest(**setup, config=BacktestConfig(weighting="mean_variance",
                                                       selection="all"))
    tails = run_backtest(**setup, config=BacktestConfig(weighting="mean_variance",
                                                        selection="decile_long_only"))
    assert whole.sharpe_mid != pytest.approx(tails.sharpe_mid)
    assert whole.mean_turnover != pytest.approx(tails.mean_turnover)


def test_a_name_dropped_by_selection_can_still_be_exited():
    """It is priced out of the book, not frozen by an infinite cost."""
    setup = world()
    report = run_backtest(**setup, config=BacktestConfig(weighting="mean_variance",
                                                         selection="decile"))
    assert report.mean_turnover > 0
    assert np.isfinite(report.sharpe_net)


def test_sweeping_agrees_exactly_with_running_one_config_at_a_time():
    """The cache is an optimization; it must not be a different answer."""
    from stable_finance import sweep_configs

    setup = world()
    space = enumerate_configs(risk_model=("diagonal", "shrunk_sample"),
                              efq=(0.0, 1.0))
    swept = sweep_configs(setup["forecast"], setup["realized"], space,
                          half_spread=setup["half_spread"],
                          history=setup["history"], days=setup["days"])
    assert len(swept) == len(space)
    for config, report in zip(space, swept):
        one = run_backtest(**setup, config=config)
        assert report.config == config == one.config
        assert report.sharpe_net == pytest.approx(one.sharpe_net, nan_ok=True)
        assert report.sharpe_mid == pytest.approx(one.sharpe_mid, nan_ok=True)
        assert report.mean_turnover == pytest.approx(one.mean_turnover)


def test_sweeping_can_skip_a_configuration_that_raises():
    from stable_finance import sweep_configs

    setup = world()
    setup["embeddings"] = None
    space = enumerate_configs(risk_model=("diagonal", "embedding"),
                              cost_model="mid", weighting="min_var")
    with pytest.raises(ValueError):
        sweep_configs(setup["forecast"], setup["realized"], space,
                      half_spread=setup["half_spread"], history=setup["history"],
                      days=setup["days"])
    kept = sweep_configs(setup["forecast"], setup["realized"], space,
                         half_spread=setup["half_spread"],
                         history=setup["history"], days=setup["days"],
                         on_error="skip")
    assert 0 < len(kept) < len(space)
    assert {r.config.risk_model for r in kept} == {"diagonal"}


# ── the profit path and the scalar scores must be the same number ───────────

def test_the_series_reproduces_the_reported_sharpe_and_means():
    """A chart drawn from the series must agree with the table's scalars.

    THE POINT OF THE TEST. A figure and a table computed by two code paths can
    disagree without either looking wrong, and a reviewer reconciling a plotted
    cumulative return against a reported Sharpe is exactly the person who finds
    it. Recomputing the report's own statistics from the series it carries is
    what makes that reconciliation a test rather than a hope.
    """
    setup = world(seed=11)
    for annualization in ("decisions", "daily"):
        config = BacktestConfig(annualization=annualization,
                                cost_model="cross+1bps")
        report = run_backtest(**setup, config=config, with_series=True)
        series = report.series
        assert series is not None
        assert len(series.net) == report.n_periods
        assert len(series.periods) == report.n_periods

        scale = np.sqrt(report.periods_per_year)
        assert (series.net.mean() / series.net.std(ddof=1) * scale
                == pytest.approx(report.sharpe_net, rel=1e-9))
        assert (series.gross.mean() / series.gross.std(ddof=1) * scale
                == pytest.approx(report.sharpe_mid, rel=1e-9))
        assert series.cost.mean() * 1e4 == pytest.approx(
            report.mean_cost_bps, rel=1e-9)
        # net is gross minus cost period by period, not merely on average
        assert series.net == pytest.approx(series.gross - series.cost, rel=1e-9)


def test_the_series_is_absent_unless_asked_for_and_never_serialized():
    """as_dict stays small: a sweep writes one row per configuration."""
    setup = world(seed=12)
    plain = run_backtest(**setup)
    assert plain.series is None
    withs = run_backtest(**setup, with_series=True)
    assert withs.series is not None
    assert "series" not in withs.as_dict()
    assert plain.as_dict() == withs.as_dict()


def test_a_cost_multiple_scales_the_charge_and_keeps_unquoted_names_untradable():
    from stable_finance.costs import trading_cost

    spread = np.array([[1e-4, 2e-4, np.nan]])
    base = trading_cost("cross", spread)
    assert trading_cost("cross", spread, multiple=2.0)[0, :2] == pytest.approx(
        base[0, :2] * 2.0)
    assert trading_cost("cross", spread, multiple=0.25)[0, :2] == pytest.approx(
        base[0, :2] * 0.25)
    # A multiple is a claim about execution quality, not about whether a market
    # that was not quoting can be traded in.
    for multiple in (0.0, 0.5, 3.0):
        assert np.isinf(trading_cost("cross", spread, multiple=multiple)[0, 2])
    # ...which is what keeps multiple=0 distinct from the mid model.
    assert trading_cost("mid", spread)[0, 2] == 0.0


def test_the_add_on_scales_with_the_multiple_too():
    from stable_finance.costs import trading_cost

    spread = np.array([[1e-4]])
    assert trading_cost("cross+1bps", spread, multiple=2.0)[0, 0] == pytest.approx(
        (1e-4 + 1e-4) * 2.0)


def test_a_negative_cost_multiple_is_refused():
    from stable_finance.costs import trading_cost

    with pytest.raises(ValueError):
        trading_cost("cross", np.array([[1e-4]]), multiple=-1.0)


# ── execution quality as a swept axis ───────────────────────────────────────

def test_worse_execution_never_improves_the_net_result():
    setup = world()
    previous = None
    for efq in (0.0, 0.25, 0.70, 1.0, 1.5):
        report = run_backtest(**setup, config=BacktestConfig(efq=efq))
        if previous is not None:
            assert report.mean_cost_bps >= previous - 1e-12
        previous = report.mean_cost_bps


def test_a_midpoint_fill_is_charged_nothing():
    report = run_backtest(**world(), config=BacktestConfig(efq=0.0))
    assert report.mean_cost_bps == pytest.approx(0.0)
    assert report.sharpe_net == pytest.approx(report.sharpe_mid)


def test_the_information_coefficient_does_not_depend_on_execution_quality():
    setup = world()
    a = run_backtest(**setup, config=BacktestConfig(efq=0.0))
    b = run_backtest(**setup, config=BacktestConfig(efq=1.0))
    assert a.information_coefficient == pytest.approx(b.information_coefficient)


def test_the_allocator_is_billed_the_execution_quality_it_optimised_against():
    """The reason EFQ is swept and not applied afterwards.

    The mean-variance allocator is cost-aware, so the BOOK depends on the EFQ
    it was told about, not only the bill. A cheap-execution book scored at an
    expensive EFQ is a portfolio nobody would have held, which is exactly what
    rescaling a stored net return post hoc would produce -- so a higher EFQ
    must change turnover, not just multiply the charge.
    """
    setup = world()
    cheap = run_backtest(**setup, config=BacktestConfig(efq=0.1))
    dear = run_backtest(**setup, config=BacktestConfig(efq=2.0))
    assert dear.mean_turnover < cheap.mean_turnover

    # A path-independent rule ignores cost when sizing, so ITS book is
    # unchanged and only the bill moves. The contrast is the point.
    flat = dict(weighting="equal", risk_model="diagonal")
    a = run_backtest(**setup, config=BacktestConfig(efq=0.1, **flat))
    b = run_backtest(**setup, config=BacktestConfig(efq=2.0, **flat))
    assert a.mean_turnover == pytest.approx(b.mean_turnover)
    assert b.mean_cost_bps == pytest.approx(20.0 * a.mean_cost_bps, rel=1e-6)


def test_execution_quality_is_an_axis_of_the_swept_space():
    from stable_finance.config import EFQ_GRID
    configs = enumerate_configs()
    assert {c.efq for c in configs} == set(EFQ_GRID)
    # Every EFQ carries the same harness, so a per-EFQ summary compares like
    # with like rather than a different mix of portfolios at each point.
    counts = {efq: sum(c.efq == efq for c in configs) for efq in EFQ_GRID}
    assert len(set(counts.values())) == 1


def test_annualization_is_pinned_to_daily_and_out_of_the_sweep():
    # It is pure accounting -- the same trades folded into different periods
    # -- so leaving it in the space inflates the reported spread with a
    # difference no model has an opinion about.
    from stable_finance.config import CONFIG_SPACE
    assert "annualization" not in CONFIG_SPACE
    assert {c.annualization for c in enumerate_configs()} == {"daily"}
    assert run_backtest(**world()).periods_per_year == pytest.approx(252.0)
