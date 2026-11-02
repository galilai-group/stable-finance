"""experiments/sanity.py: the checks' statistics, on synthetic panels with known answers.

The checks themselves read finished runs and a cached month; what is tested
here is the arithmetic each one rests on, where a wrong unit or a pooled
sample would pass silently.
"""

import numpy as np
import pytest

import stable_finance as sf
from experiments import run, sanity
from stable_finance.evaluation.scoring.ic import grouped_rank_ic

# ── the fundamental law ─────────────────────────────────────────────────────


def _panel(seed=0, n_periods=400, n_assets=300, ic=0.1):
    """Forecasts with a target value IC, outcomes whose dispersion varies by period."""
    rng = np.random.default_rng(seed)
    dispersion = np.exp(rng.normal(0.0, 0.4, n_periods))[:, None] * 1e-3
    z = rng.standard_normal((n_periods, n_assets))
    outcome = dispersion * z
    mu = ic * z + np.sqrt(1 - ic**2) * rng.standard_normal((n_periods, n_assets))
    return mu, outcome


def test_row_correlation_matches_numpy_on_each_row_and_needs_three_names():
    rng = np.random.default_rng(0)
    left, right = rng.standard_normal((2, 5, 40))
    mask = rng.random((5, 40)) > 0.3
    mask[4] = False
    mask[4, :2] = True
    got = sanity._row_correlation(left, right, mask)
    for t in range(4):
        m = mask[t]
        assert got[t] == pytest.approx(np.corrcoef(left[t, m], right[t, m])[0, 1], abs=1e-12)
    assert np.isnan(got[4])


def test_formation_statistics_score_the_harness_rank_ic_and_skip_an_unknown_outcome():
    mu, outcome = _panel(n_periods=30)
    outcome[-1] = np.nan                       # a formation whose forward return is never known
    screened = np.ones(mu.shape, dtype=bool)
    screened[:, :10] = False
    stats = sanity.formation_statistics(mu, outcome, screened)
    for t in range(29):
        expected = grouped_rank_ic(np.where(screened, mu, np.nan)[t:t + 1], outcome[t:t + 1])
        assert stats["ic_rank"][t] == pytest.approx(expected.mean, abs=1e-12)
        assert stats["dispersion"][t] == pytest.approx(outcome[t, 10:].std(), rel=1e-12)
    assert np.isnan(stats["ic_rank"][-1]) and np.isnan(stats["ic_value"][-1])
    assert stats["breadth"][0] == 290 and stats["breadth"][-1] == 0


def test_identity_tc_is_one_for_a_signal_book_and_sqrt_two_over_pi_for_signs():
    rng = np.random.default_rng(1)
    alpha = rng.standard_normal((50, 2000))
    screened = np.ones(alpha.shape, dtype=bool)
    signal = alpha - alpha.mean(axis=1, keepdims=True)
    assert np.allclose(sanity.identity_tc(signal, alpha, screened), 1.0)
    signs = np.sign(alpha) + 1e-13 * (alpha > 0)   # roundoff residue is not a position
    assert np.mean(sanity.identity_tc(signs, alpha, screened)) == pytest.approx(
        np.sqrt(2 / np.pi), abs=0.01)


def test_day_draws_start_from_the_full_sample_and_resample_run_days():
    day = np.repeat(np.arange(5), 3)
    weights = sanity.day_draws(day, n_draws=200)
    assert weights.shape == (201, 15)
    assert np.all(weights[0] == 1.0)
    assert np.all(weights[1:].sum(axis=1) == 15)                 # five days of three periods
    assert np.all(weights[:, ::3] == weights[:, 1::3])           # a day is drawn whole
    draws = np.random.default_rng(0).integers(0, 5, size=(200, 5))
    assert weights[1, ::3].tolist() == np.bincount(draws[0], minlength=5).tolist()


def test_moments_weight_periods_and_ignore_unknown_values():
    values = np.array([1.0, 2.0, np.nan, 4.0])
    weights = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 5.0, 1.0]])
    mean, sd = sanity._moments(weights, values)
    assert mean[0] == pytest.approx(7 / 3) and sd[0] == pytest.approx(np.std([1, 2, 4], ddof=1))
    assert mean[1] == pytest.approx(2.0) and sd[1] == pytest.approx(np.std([1, 1, 4], ddof=1))


def test_law_draws_explain_a_signal_book_by_value_ic_times_dispersion():
    mu, outcome = _panel(ic=0.1)
    screened = np.ones(mu.shape, dtype=bool)
    book = mu - mu.mean(axis=1, keepdims=True)
    book *= 2.0 / np.abs(book).sum(axis=1, keepdims=True)
    gross = (book * outcome).sum(axis=1)
    stats = sanity.formation_statistics(mu, outcome, screened)
    tc = sanity.identity_tc(book, mu, screened)
    weights = sanity.day_draws(np.repeat(np.arange(20), 20))
    result = sanity.law_draws(gross, tc, stats, weights)
    # P&L is the value IC times the dispersion, up to the forecast's shape.
    assert result["ir"][0] / result["value_dispersion"][0] == pytest.approx(1.0, abs=0.02)
    # A varying dispersion costs IR the law does not see.
    assert result["ir"][0] < result["law"][0]
    assert result["breadth"][0] == 300 and result["n_periods"][0] == 400
    assert np.all(np.isfinite(result["ir"]))


def test_law_draws_leave_a_formation_without_a_known_return_out_of_every_factor():
    mu, outcome = _panel(n_periods=60)
    screened = np.ones(mu.shape, dtype=bool)
    book = mu - mu.mean(axis=1, keepdims=True)
    gross = (book * outcome).sum(axis=1)
    weights = sanity.day_draws(np.repeat(np.arange(6), 10), n_draws=50)
    kept = sanity.law_draws(gross, sanity.identity_tc(book, mu, screened),
                            sanity.formation_statistics(mu, outcome, screened), weights)
    dead = outcome.copy()
    dead[9::10] = np.nan                       # the last formation of each day books nothing
    gross_dead = np.where(np.isnan(dead).all(axis=1), 0.0, gross)
    cut = sanity.law_draws(gross_dead, sanity.identity_tc(book, mu, screened),
                           sanity.formation_statistics(mu, dead, screened), weights)
    keep = ~np.isnan(dead).all(axis=1)
    assert cut["n_periods"][0] == 54
    assert cut["ir"][0] == pytest.approx(gross[keep].mean() / gross[keep].std(ddof=1), rel=1e-9)
    assert cut["ir"][0] != pytest.approx(kept["ir"][0], rel=1e-6)


def test_row_config_reads_back_the_configuration_a_row_was_written_from():
    config = sf.BacktestConfig(weighting="rank", selection="decile", cost_model="cross", efq=0.7,
                               risk_model="shrunk_sample", universe="all",
                               annualization="decisions", rebalance="horizon")
    row = {**config.__dict__, "sharpe_mid": 1.0, "equity": float("nan"), "alpha": None}
    row = {k: (np.float64(v) if isinstance(v, float) else v) for k, v in row.items()}
    assert sanity._row_config(row) == config


def test_row_forecast_reads_back_an_oracle_spec():
    pd = pytest.importorskip("pandas")
    row = pd.Series({"forecast": "oracle", "f_ic": np.float64(0.03), "f_noise": "random",
                     "f_scatter": None, "f_persistence": np.nan, "f_seed": np.int64(2),
                     "weighting": "signal"})
    spec = sanity._row_forecast(row)
    assert spec == {"kind": "oracle", "ic": 0.03, "noise": "random", "seed": 2}
    assert type(spec["seed"]) is int
    assert run.forecast_specs({"forecasts": {"kind": "oracle", "ic": 0.03, "noise": "random",
                                             "seed": 2}})[0]["seed"] == spec["seed"]


# ── latency ─────────────────────────────────────────────────────────────────


def test_first_of_day_marks_each_days_opening_decision():
    labels = ["2020-08-03|0000", "2020-08-03|0300", "2020-08-04|0000", "2020-08-04|0300",
              "2020-08-04|0600"]
    assert sanity._first_of_day(labels).tolist() == [True, False, True, False, False]


def test_cost_and_sharpe_leave_the_open_out_of_every_number():
    from types import SimpleNamespace

    from experiments.run import _sharpe

    rng = np.random.default_rng(0)
    periods = np.array([f"2020-08-{d:02d}|{s:04d}" for d in range(3, 9) for s in (0, 300, 600)])
    opening = sanity._first_of_day(periods)
    cost = np.where(opening, 10e-4, 1e-4)
    gross = rng.normal(2e-4, 1e-4, len(periods))
    series = SimpleNamespace(periods=periods, gross=gross, net=gross - cost, cost=cost)
    every = sanity._cost_and_sharpe(series, np.ones_like(opening))
    rest = sanity._cost_and_sharpe(series, ~opening)
    assert every["mean_cost_bps"] == pytest.approx(4.0)
    assert rest["mean_cost_bps"] == pytest.approx(1.0)
    daily = (gross - cost)[~opening].reshape(6, 2).sum(axis=1)
    assert rest["sharpe_daily_net"] == pytest.approx(float(_sharpe(daily)))
    daily = gross.reshape(6, 3).sum(axis=1)
    assert every["sharpe_daily_gross"] == pytest.approx(float(_sharpe(daily)))


# ── checks on finished runs ─────────────────────────────────────────────────


def _write_rows(results, name, rows):
    pd = pytest.importorskip("pandas")
    (results / name).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(results / name / "rows.parquet", index=False)


def test_exact_ic_invariance_keeps_placements_and_horizons_apart(tmp_path):
    rows = []
    for noise, sign in (("tails", -1.0), ("middle", 1.0)):
        for horizon, scale in ((900, 125.0), (3600, 53.0)):
            for k, ic in enumerate((0.005, 0.03, 0.1)):
                rows.append({"f_noise": noise, "weighting": "signal", "selection": "all",
                             "efq": 0.0, "eval_horizon": horizon, "rebalance": "horizon",
                             "f_ic": ic, "sharpe_mid": sign * scale * (1 + 0.01 * k)})
    _write_rows(tmp_path, "e1_transfer_map", rows)
    result = sanity.exact_ic_invariance(tmp_path)
    assert result["status"] == "INFO" and result["n_curves"] == 4
    curve = result["curves"]["tails/900s/horizon"]
    assert curve["sharpe_by_ic"] == pytest.approx({"0.005": -125.0, "0.03": -126.25,
                                                   "0.1": -127.5})
    assert curve["relative_range"] == pytest.approx(2.5 / 126.25)
    assert result["max_relative_range"] == pytest.approx(0.02 / 1.01)


def _cost_run(results, breakeven_error=0.0):
    rng = np.random.default_rng(3)
    gross = rng.normal(1e-4, 1e-4, 50)
    cost = np.abs(rng.normal(0.7e-4, 0.1e-4, 50))
    breakeven = float(sf.breakeven_efq(gross, cost / 0.7))
    nan = float("nan")
    rows = [  # mid; crossing without fees (fee column NaN); with fees; no finite break-even
        {"row_id": "mid", "cost_model": "mid", "efq": 0.0, "breakeven_efq": nan,
         "mean_fee_bps": nan},
        {"row_id": "cross", "cost_model": "cross", "efq": 0.7,
         "breakeven_efq": breakeven + breakeven_error, "mean_fee_bps": nan},
        {"row_id": "fees", "cost_model": "cross", "efq": 0.7, "breakeven_efq": nan,
         "mean_fee_bps": 2.5},
        {"row_id": "flat", "cost_model": "cross", "efq": 0.7, "breakeven_efq": nan,
         "mean_fee_bps": nan},
        {"row_id": "unsaved", "cost_model": "cross", "efq": 0.7, "breakeven_efq": nan,
         "mean_fee_bps": nan},
    ]
    _write_rows(results, "run", rows)
    series = {r["row_id"]: np.stack([gross, gross - cost, cost]) for r in rows[:4]}
    np.savez(results / "run" / "series.npz", **series)


def test_cost_identity_checks_a_row_whose_fee_column_is_nan_and_reports_coverage(tmp_path):
    _cost_run(tmp_path)
    result = sanity.cost_identity(tmp_path)
    assert result["status"] == "PASS"
    assert (result["rows"], result["with_series"], result["breakeven_rows"]) == (5, 4, 3)
    assert result["breakeven_checked"] == 1
    assert result["breakeven_skipped_fees"] == 1 and result["breakeven_skipped_not_finite"] == 1


def test_cost_identity_fails_a_break_even_its_series_does_not_reproduce(tmp_path):
    _cost_run(tmp_path, breakeven_error=1e-6)
    result = sanity.cost_identity(tmp_path)
    assert result["status"] == "FAIL" and result["worst_breakeven_error"] == pytest.approx(1e-6)
