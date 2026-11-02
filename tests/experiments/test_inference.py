"""experiments/inference.py: the joint tests' bookkeeping, on synthetic daily P&L.

The tests that run a confidence set or an SPA need ``arch`` (the
``experiments`` extra) and are skipped without it; the counting rules around
them are plain numpy and always run.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from experiments import inference

N_DAYS = 100


def _group(columns: dict[str, np.ndarray]):
    """A group frame and its daily dict: each column is one configuration's net P&L."""
    ids = list(columns)
    group = pd.DataFrame({"row_id": ids, "weighting": ["signal"] * len(ids),
                          "selection": ["all"] * len(ids)})
    daily = {r: np.stack([v, v, np.zeros_like(v)]) for r, v in columns.items()}
    return group, daily


def _statistics(columns, **options):
    pytest.importorskip("arch")
    group, daily = _group(columns)
    settings = {"benchmark": "weighting=rank,selection=all", "which": 1, "reps": 200,
                "seed": 0, "block": 1, "pbo_n": None, "block_data": None, **options}
    return inference.group_statistics(group, daily, **settings)


# ── counting distinct configurations ────────────────────────────────────────

def test_distinct_columns_names_the_first_copy_as_the_representative():
    a, b = np.arange(5.0), np.arange(5.0)[::-1]
    first, representative = inference.distinct_columns(np.stack([a, b, a, b, a + 1], axis=1))
    assert sorted(first.tolist()) == [0, 1, 4]
    assert representative.tolist() == [0, 1, 0, 1, 4]


def test_scaled_losses_average_to_minus_the_daily_sharpe_and_zero_for_a_flat_book():
    rng = np.random.default_rng(0)
    pnl = np.column_stack([rng.normal(0.3, 2.0, 50), rng.normal(-0.1, 0.5, 50), np.zeros(50)])
    loss = inference.scaled_losses(pnl)
    sharpe = pnl[:, :2].mean(axis=0) / pnl[:, :2].std(axis=0, ddof=1)
    np.testing.assert_allclose(loss[:, :2].mean(axis=0), -sharpe, rtol=1e-12)
    assert np.all(loss[:, 2] == 0.0)


def test_duplicates_are_kept_out_of_every_mcs_count():
    rng = np.random.default_rng(1)
    good = rng.normal(1.0, 1.0, N_DAYS)
    summary, per_row = _statistics({"a": good, "b": good.copy(),
                                    "c": rng.normal(-1.0, 1.0, N_DAYS),
                                    "d": rng.normal(0.9, 1.0, N_DAYS)})
    assert summary["n_configs"] == 4 and summary["distinct_models"] == 3
    assert per_row["b"]["duplicate_of"] == "a"
    assert all(per_row[r]["duplicate_of"] is None for r in "acd")
    for kind in ("pnl", "sharpe"):
        found = summary[f"mcs_{kind}"]
        assert "b" not in found["included"] and "a" in found["included"]
        assert found["distinct_models"] == 3
        assert found["included_distinct"] == len(found["included"])
        assert found["share_distinct"] == found["included_distinct"] / 3
        assert found["loss"].startswith("minus the day's net P&L")
        # A duplicate mirrors its representative's membership and p-value.
        assert per_row["b"][f"in_mcs_{kind}"] == per_row["a"][f"in_mcs_{kind}"]
        assert per_row["b"][f"mcs_pvalue_{kind}"] == per_row["a"][f"mcs_pvalue_{kind}"]
        assert not per_row["c"][f"in_mcs_{kind}"]


def test_a_group_of_copies_is_one_model():
    pnl = np.random.default_rng(2).normal(0.0, 1.0, N_DAYS)
    summary, per_row = _statistics({"a": pnl, "b": pnl.copy(), "c": pnl.copy()})
    assert summary["distinct_models"] == 1
    assert summary["mcs_pnl"]["included"] == ["a"]
    assert summary["mcs_pnl"]["included_distinct"] == 1
    assert all(per_row[r]["in_mcs_pnl"] for r in "abc")


def test_the_two_losses_rank_a_volatile_book_and_a_steady_one_differently():
    rng = np.random.default_rng(3)
    steady = rng.normal(0.5, 0.1, N_DAYS)            # daily Sharpe about 5
    volatile = rng.normal(2.0, 3.0, N_DAYS)          # more P&L, Sharpe about 0.7
    summary, per_row = _statistics({"steady": steady, "volatile": volatile})
    assert per_row["volatile"]["in_mcs_pnl"] and not per_row["steady"]["in_mcs_pnl"]
    assert per_row["steady"]["in_mcs_sharpe"] and not per_row["volatile"]["in_mcs_sharpe"]
    assert "Sharpe" in summary["mcs_sharpe"]["loss"]


def test_series_the_scaled_loss_cannot_tell_apart_share_one_p_value():
    rng = np.random.default_rng(4)
    base = rng.normal(0.2, 1.0, N_DAYS)
    summary, per_row = _statistics({"a": base, "b": 2.0 * base,
                                    "c": rng.normal(0.1, 1.0, N_DAYS)})
    assert summary["distinct_models"] == 3            # different P&L, so distinct
    assert summary["mcs_sharpe"]["tied_by_loss"] == 1
    assert per_row["a"]["mcs_pvalue_sharpe"] == per_row["b"]["mcs_pvalue_sharpe"]
    assert per_row["b"]["duplicate_of"] is None


def test_stepm_counts_distinct_superior_configurations():
    pytest.importorskip("arch")
    rng = np.random.default_rng(5)
    bench = rng.normal(0.0, 1.0, N_DAYS)
    better = bench + 2.0 + rng.normal(0.0, 0.2, N_DAYS)
    group, daily = _group({"rank": bench, "x": better, "y": better.copy()})
    group.loc[0, "weighting"] = "rank"
    summary, _ = inference.group_statistics(
        group, daily, benchmark="weighting=rank,selection=all", which=1, reps=200, seed=0,
        block=1, pbo_n=None, block_data=None)
    assert summary["stepm"]["superior"] == ["x", "y"]
    assert summary["stepm"]["superior_distinct"] == 1


# ── the deflated Sharpe's null ──────────────────────────────────────────────

def test_null_sharpe_variance_is_about_one_over_days_for_iid_pnl():
    rng = np.random.default_rng(6)
    pnl = np.column_stack([rng.normal(0.5, 1.0, N_DAYS), rng.normal(-3.0, 0.1, N_DAYS),
                           np.zeros(N_DAYS)])
    index = rng.integers(0, N_DAYS, size=(4000, N_DAYS))
    variance = inference.null_sharpe_variance(pnl, index, chunk=2)
    # Demeaned first, so neither column's own (large) Sharpe enters.
    np.testing.assert_allclose(variance[:2], 1.0 / N_DAYS, rtol=0.15)
    assert np.isnan(variance[2])


def test_the_dsr_hurdle_comes_from_the_null_not_the_spread_of_costs():
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 1.0, (N_DAYS, 4))
    # Four books whose means differ by their cost: a wide cross-sectional
    # spread of Sharpe ratios that says nothing about selection noise.
    columns = {name: noise[:, k] - cost for k, (name, cost)
               in enumerate(zip("abcd", (0.0, 0.5, 1.0, 1.5)))}
    summary, per_row = _statistics(columns)
    dsr = summary["dsr"]
    assert dsr["sharpe_var_cross_per_day"] > 10 * dsr["sharpe_var_per_day"]
    np.testing.assert_allclose(dsr["sharpe_var_per_day"], 1.0 / N_DAYS, rtol=0.3)
    assert dsr["sharpe_var_iid_per_day"] == 1.0 / (N_DAYS - 1)
    assert dsr["hurdle_per_day"] == inference.sf.expected_max_sharpe(
        dsr["n_trials"], dsr["sharpe_var_per_day"])
    scale = np.sqrt(inference.DAYS_A_YEAR)
    assert dsr["hurdle_annual"] == dsr["hurdle_per_day"] * scale
    assert dsr["best_sharpe_daily"] == dsr["best_sharpe_per_day"] * scale
    assert dsr["best_dsr"] == per_row[dsr["best_row_id"]]["dsr_daily"]


def test_dsr_and_pbo_count_a_copy_once():
    rng = np.random.default_rng(8)
    a, b = rng.normal(0.1, 1.0, N_DAYS), rng.normal(0.0, 1.0, N_DAYS)
    summary, _ = _statistics({"a": a, "b": b, "a2": a.copy(), "a3": a.copy()})
    assert summary["dsr"]["n_trials_all"] == 2
    assert summary["dsr"]["n_eff_day"] <= 2.0


# ── Lo's ratio at several lag counts ────────────────────────────────────────

def test_lo_lags_span_one_horizon_two_and_a_session_in_the_rows_periods():
    every = {"rebalance": "every_decision", "annualization": "decisions",
             "eval_horizon": 900, "periods_per_year": 78 * 252.0}
    held = {**every, "rebalance": "horizon", "periods_per_year": 26 * 252.0}
    daily = {**every, "annualization": "daily", "periods_per_year": 252.0}
    assert inference.lo_lags(every, 1638, 300.0) == {"hold": 3, "hold2": 6, "day": 78}
    assert inference.lo_lags(held, 546, 300.0) == {"hold": 1, "hold2": 2, "day": 26}
    assert inference.lo_lags(daily, 21, 300.0) == {"hold": 1, "hold2": 2, "day": 1}
    assert inference.lo_lags(every, 1638, None)["hold"] == 1
    assert inference.lo_lags(every, 30, 300.0)["day"] == 28       # capped at n - 2


def test_lo_ratio_under_corrects_an_overlapping_book_until_the_lags_cover_it():
    rng = np.random.default_rng(9)
    k = 6                       # each period's P&L sums six one-step returns
    x = np.convolve(rng.normal(0.01, 1.0, 40_000), np.ones(k), mode="valid")
    q = 10_000.0
    found = inference.lo_at_lags(x, q=q, lags={"short": 1, "cover": k - 1, "long": 4 * k})
    np.testing.assert_allclose(found["lo_ratio_cover"], 1.0 / np.sqrt(k), rtol=0.1)
    np.testing.assert_allclose(found["lo_ratio_long"], 1.0 / np.sqrt(k), rtol=0.1)
    assert found["lo_ratio_short"] > 1.3 * found["lo_ratio_cover"]
    stats = inference.series_statistics(x, q=q, lags=k - 1)
    assert found["lo_ratio_cover"] == stats["lo_ratio"]
    assert found["sharpe_lo_cover"] == stats["sharpe_lo"]


# ── grouping ────────────────────────────────────────────────────────────────

def _grid_rows(**columns):
    base = {"row_id": ["a", "b", "c", "d"], "forecast": ["oracle"] * 4, "f_ic": [0.03] * 4,
            "eval_horizon": [900] * 4, "efq": [0.7] * 4, "weighting": ["signal", "hrp"] * 2,
            "cost_model": ["cross"] * 4}
    return pd.DataFrame({**base, **columns})


def test_default_group_splits_by_a_scoring_setting_only_when_it_varies():
    plain = _grid_rows(equity=[1e6] * 4)
    assert inference.default_group(plain) == ["forecast", "f_ic", "eval_horizon", "efq"]
    sizes = _grid_rows(equity=[1e5, 1e5, 1e6, 1e6])
    assert inference.default_group(sizes)[-1] == "equity"
    fees = _grid_rows(fee_per_share=[None, None, 0.003, 0.003])   # unset counts as a value
    assert inference.default_group(fees)[-1] == "fee_per_share"


def test_group_keys_take_the_cli_then_the_spec_then_the_default_and_append_additions():
    rows = _grid_rows(universe=["all", "all", "tight", "tight"])
    default = inference.default_group(rows)
    assert inference.group_keys(rows, None) == default
    spec = {"inference": {"group_by": ["forecast", "efq"], "group_add": ["universe"]}}
    assert inference.group_keys(rows, spec) == ["forecast", "efq", "universe"]
    assert inference.group_keys(rows, spec, ["f_ic"]) == ["f_ic", "universe"]
    assert inference.group_keys(rows, None, group_add=["universe"]) == [*default, "universe"]
    assert inference.group_keys(rows, None, ["all"]) == ["all"]
    assert inference.group_keys(rows, None, ["all"], ["universe"]) == ["universe"]
    with pytest.raises(ValueError, match="cannot group by"):
        inference.group_keys(rows, None, group_add=["equity"])


# ── the placebo size check ──────────────────────────────────────────────────

def test_wilson_interval_brackets_the_rate_and_stays_in_the_unit_interval():
    low, high = inference.wilson_interval(1, 20)
    assert 0.0 < low < 0.05 < high < 0.3
    assert inference.wilson_interval(0, 10)[0] == 0.0
    assert inference.wilson_interval(10, 10)[1] == 1.0
    assert all(np.isnan(inference.wilson_interval(0, 0)))


def _placebo_run(path, *, seeds=6, days=30, per_day=4, spec=None):
    """A synthetic IC-0 run on disk: three weightings under ``seeds`` seeds at EFQ 0 and 0.7.

    Gross P&L is pure noise, identical at both EFQs; the cost is charged at
    EFQ 0.7 only, so net differs. Labels are saved, as run.py saves them.
    """
    import json

    rng = np.random.default_rng(10)
    labels = np.array([f"d{d:02d}|{34200 + 300 * k}" for d in range(days) for k in range(per_day)])
    q = per_day * inference.DAYS_A_YEAR
    rows, series = [], {}
    for seed in range(seeds):
        for weighting in ("rank", "signal", "equal"):
            gross = rng.normal(0.0, 1e-3, len(labels))
            cost = np.abs(rng.normal(1e-4, 2e-5, len(labels)))
            for efq in (0.0, 0.7):
                row_id = f"s{seed}{weighting}{efq}"
                net = gross - efq * cost
                series[row_id] = np.stack([gross, net, efq * cost])
                rows.append({"row_id": row_id, "forecast": "copula", "f_ic": 0.0, "f_seed": seed,
                             "eval_horizon": 300, "selection": "all", "weighting": weighting,
                             "cost_model": "cross", "efq": efq, "rebalance": "every_decision",
                             "annualization": "decisions", "periods_per_year": q,
                             "sharpe_mid": gross.mean() / gross.std(ddof=1) * np.sqrt(q),
                             "sharpe_net": net.mean() / net.std(ddof=1) * np.sqrt(q)})
    path.mkdir()
    pd.DataFrame(rows).to_parquet(path / "rows.parquet", index=False)
    np.savez(path / "series.npz", **series)
    np.savez(path / "periods.npz", row_id=np.array(list(series)),
             set=np.zeros(len(series), dtype=int), set_0=labels)
    if spec is not None:
        (path / "manifest.json").write_text(json.dumps({"spec": spec}))
    return path


def test_size_check_runs_the_joint_tests_on_the_placebo_at_its_lowest_efq(tmp_path):
    pytest.importorskip("arch")
    out = _placebo_run(tmp_path / "placebo")
    report = inference.analyze(str(out), reps=50, size=True)
    size = report["placebo_size"]
    assert size["efq"] == 0.0 and size["pnl"] == "mid"
    assert size["n_rows"] == 18 and size["n_groups"] == 6 and size["n_skipped"] == 0
    tests = size["tests"]
    assert tests["daily_t"]["n"] == 18 and tests["daily_t"]["unit"] == "distinct configuration"
    assert {name: t["n"] for name, t in tests.items() if name != "daily_t"} == dict.fromkeys(
        ("daily_t_any", "spa", "stepm", "mcs_pnl", "mcs_sharpe", "dsr"), 6)
    assert tests["mcs_pnl"]["level"] == inference.MCS_SIZE
    assert tests["spa"]["level"] == tests["dsr"]["level"] == inference.TEST_SIZE
    for t in tests.values():
        assert 0.0 <= t["low"] <= t["rate"] <= t["high"] <= 1.0
    # Every group, skipped or not, says how many distinct configurations it had.
    assert all("distinct_models" in g for g in report["groups"])
    assert report["checks"]["distinct_models"] == sum(g["distinct_models"]
                                                      for g in report["groups"])
    assert "placebo_size" not in inference.analyze(str(out), reps=50)


def test_a_spec_can_set_the_grouping(tmp_path):
    pytest.importorskip("arch")
    out = _placebo_run(tmp_path / "grouped", seeds=2, spec={"inference": {"group_by": ["f_seed"]}})
    report = inference.analyze(str(out), reps=50)
    assert report["settings"]["group_by"] == ["f_seed"]
    assert [g["n_configs"] for g in report["groups"]] == [6, 6]
    # Gross is shared across EFQ but net is not, so all six are distinct.
    assert [g["distinct_models"] for g in report["groups"]] == [6, 6]


def test_clustered_rate_widens_the_interval_when_rejections_come_together():
    spread = np.tile([True] + [False] * 9, 10)                  # one hit in each cluster
    bunched = np.r_[[True] * 10, [False] * 90]                  # all hits in one cluster
    clusters = np.repeat(np.arange(10), 10)
    rate, low, high = inference.clustered_rate(spread, clusters)
    assert rate == 0.1 and (low, high) == inference.wilson_interval(10, 100)
    rate, low_b, high_b = inference.clustered_rate(bunched, clusters)
    assert rate == 0.1 and low_b < low and high_b > high + 0.15
    # No rejection: each cluster counts as one trial, not a zero-width interval.
    assert inference.clustered_rate(np.zeros(100, bool), clusters)[1:] == (
        0.0, inference.wilson_interval(0, 10)[1])
    assert np.isnan(inference.clustered_rate([True, False], [0, 0])[1])


def test_size_check_counts_the_lowest_efq_once_and_clusters_by_group():
    t = [3.0, 0.1, 0.2, 0.3, 3.0, 0.1, 0.2, 0.3]
    table = pd.DataFrame({"row_id": list("abcdefgh"), "f_ic": [0.0] * 8,
                          "efq": [0.0] * 4 + [0.7] * 4,
                          "group": ["s0", "s0", "s1", "s1", "s0b", "s0b", "s1b", "s1b"],
                          "sharpe_t_hac_mid": t, "sharpe_t_iid_mid": t,
                          "sharpe_daily_t_hac_mid": t})
    size = inference.size_check(table)
    assert size["n_rows"] == 4 and size["efq"] == 0.0 and size["n_clusters"] == 2
    assert size["reject_sharpe_daily_t_hac_mid"] == 0.25
    low, high = size["interval_sharpe_daily_t_hac_mid"]
    assert 0.0 < low < 0.25 and high > 0.5        # wider than Wilson on the four rows
    assert high > inference.wilson_interval(1, 4)[1]
