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
