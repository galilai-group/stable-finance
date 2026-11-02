"""experiments/dynamic.py: the driver, the cost settings and the bookkeeping, on synthetic data.

No month cache is read: ``main`` runs with ``one_seed`` replaced by a stub of
the same signature, so the process pool's argument passing is exercised
without tuning anything.
"""

import inspect
import json

import numpy as np
import pytest

import stable_finance as sf
from experiments import dynamic

D, A = 24, 4
DAYS = np.repeat(np.array(["2020-08-03", "2020-08-04", "2020-08-05"]), D // 3)
OFFSETS = np.tile(np.arange(D // 3) * 300, 3)
DECISIONS = np.array([f"{d}|{o:04d}" for d, o in zip(DAYS, OFFSETS)])
ASSETS = np.array([f"S{i}" for i in range(A)])
METRICS = ("sharpe_net", "sharpe_daily_net", "sharpe_daily_mid", "mean_turnover",
           "holding_decisions")


def _month(seed: int = 0) -> dict:
    """The arrays a loaded month carries, on a small quoted market."""
    rng = np.random.default_rng(seed)
    mid = 20.0 + 30.0 * rng.random(A) * np.exp(rng.normal(0, 0.01, (D, A)).cumsum(axis=0))
    half = mid * rng.uniform(1e-4, 4e-4, (D, A))
    market = sf.MarketData({"bid_price": mid - half, "ask_price": mid + half}, DECISIONS, ASSETS)
    returns = rng.normal(0, 0.002, (D, A, 1))
    stamp = np.arange(D, dtype=float)[:, None] * np.ones(A)
    return {"realized": sf.ForwardReturns(returns, DECISIONS, ASSETS, np.array([300])),
            "market": market, "days": DAYS.copy(), "cost": 0.7 * half / mid,
            "one_step": stamp, "oracle": stamp[:, :, None].repeat(3, axis=2),
            "features": stamp[:, :, None].repeat(2, axis=2),
            "horizon_panel": stamp[:, :, None].repeat(3, axis=2)}


# ── main: seeds through the process pool ────────────────────────────────────

def _stub_seed(seed: int, *, skip_open: bool, quick: bool) -> dict:
    """What one_seed returns, with the options it was called with written into the scores."""
    scores = {metric: float(seed) + 0.5 for metric in METRICS}
    scores["mean_turnover"] = 1.0 + skip_open + 2 * quick
    return {"seed": seed, "assets": 7, "forecasts": {"copula": {
        "fees": {"tuned_on_train": {}, "test": {"target": scores, "band": dict(scores)}},
        "impact": {"tuned_on_train": "fees", "skipped": "stub"}}}}


def test_the_stub_has_one_seeds_signature():
    assert inspect.signature(_stub_seed).parameters.keys() == \
        inspect.signature(dynamic.one_seed).parameters.keys()
    for name, parameter in inspect.signature(dynamic.one_seed).parameters.items():
        assert inspect.signature(_stub_seed).parameters[name].kind == parameter.kind


@pytest.mark.parametrize("flags, turnover", [([], 1.0), (["--skip-open", "--quick"], 4.0)])
def test_main_hands_every_seed_its_options_through_the_pool(monkeypatch, tmp_path, flags,
                                                            turnover):
    monkeypatch.setattr(dynamic, "one_seed", _stub_seed)
    monkeypatch.setattr(dynamic, "OUT", tmp_path)
    assert dynamic.main(["--seeds", "3", "--workers", "2", *flags]) == 0
    name = "dynamic_noopen.json" if "--skip-open" in flags else "dynamic.json"
    report = json.loads((tmp_path / name).read_text())
    assert report["seeds"] == [0, 1, 2] and [s["seed"] for s in report["by_seed"]] == [0, 1, 2]
    assert report["skip_open"] == ("--skip-open" in flags)
    assert report["quick"] == ("--quick" in flags)
    row = report["summary"]["copula"]["fees"]["target"]
    assert row["mean_turnover"]["mean"] == turnover
    assert row["sharpe_net"] == {"mean": 1.5, "sd": 1.0, "min": 0.5, "max": 2.5}
    assert report["summary"]["copula"]["impact"] == {"skipped": "stub"}


# ── summaries ───────────────────────────────────────────────────────────────

def test_summarize_with_one_seed_has_zero_spread_and_counts_a_positive_daily_net():
    summary = dynamic.summarize([_stub_seed(0, skip_open=False, quick=False)])
    row = summary["copula"]["fees"]["band"]
    assert row["sharpe_daily_net"] == {"mean": 0.5, "sd": 0.0, "min": 0.5, "max": 0.5}
    assert row["seeds_net_positive"] == 1
    assert summary["copula"]["impact"] == {"skipped": "stub"}


def test_summarize_counts_only_the_seeds_with_a_positive_daily_net():
    seeds = [_stub_seed(s, skip_open=False, quick=False) for s in range(3)]
    seeds[1]["forecasts"]["copula"]["fees"]["test"]["band"]["sharpe_daily_net"] = -1.0
    assert dynamic.summarize(seeds)["copula"]["fees"]["band"]["seeds_net_positive"] == 2


# ── cost settings and the holding period ────────────────────────────────────

def test_config_for_each_cost_setting():
    spread, fees = dynamic.config_for("spread"), dynamic.config_for("fees")
    assert (spread.cost_model, spread.efq, spread.annualization) == ("cross", dynamic.EFQ,
                                                                     "decisions")
    assert spread.fees == {}
    assert fees.fees == {"fee_per_share": 0.003, "sec_fee_rate": 22.10e-6}
    assert (fees.cost_model, fees.efq) == ("cross", dynamic.EFQ)
    with pytest.raises(ValueError, match="unknown cost setting"):
        dynamic.config_for("free")


def test_the_impact_setting_either_builds_with_its_floor_or_says_why_not():
    reason = dynamic.impact_unavailable()
    if reason is None:
        impact = dynamic.config_for("impact")
        assert impact.cost_model == "impact" and impact.efq == 1.0
        assert tuple(impact.impact) == dynamic.IMPACT["impact"]
        assert impact.fees == dynamic.FEES
    else:
        # A library without the runner fixes takes three coefficients only.
        with pytest.raises(ValueError) as error:
            dynamic.config_for("impact")
        assert str(error.value) == reason


def test_holding_decisions_is_twice_gross_over_turnover():
    assert dynamic.holding_decisions(2.0, 0.5) == 8.0
    assert dynamic.holding_decisions(2.0, 0.0) == float("inf")


def test_evaluate_reports_the_conventional_holding_period():
    month = _month()
    rng = np.random.default_rng(1)
    weights = rng.normal(0, 0.3, (D, A))
    weights[1::2] = weights[::2]                     # half the decisions do not trade
    held = sf.PortfolioWeights(weights[:, :, None], DECISIONS, ASSETS, np.array([300]))
    for costs in ("spread", "fees"):
        out = dynamic.evaluate(held, month, costs)
        report = sf.run_backtest(held, month["realized"], market=month["market"],
                                 days=month["days"], config=dynamic.config_for(costs))
        assert out["mean_turnover"] == pytest.approx(report.mean_turnover)
        assert out["holding_decisions"] == pytest.approx(
            2 * report.mean_gross_exposure / report.mean_turnover)
        assert np.isfinite(out["breakeven_efq"]) and out["n_days"] == 3


# ── the opening decision ────────────────────────────────────────────────────

def test_drop_open_removes_each_days_first_decision_from_every_array():
    month = _month()
    out = dynamic._drop_open(month)
    keep = np.flatnonzero(OFFSETS != 0)
    assert len(keep) == D - 3
    np.testing.assert_array_equal(out["realized"].decisions, DECISIONS[keep])
    np.testing.assert_array_equal(out["realized"].values, month["realized"].values[keep])
    np.testing.assert_array_equal(out["market"].decisions, DECISIONS[keep])
    np.testing.assert_array_equal(out["market"].get("bid_price"),
                                  month["market"].get("bid_price")[keep])
    for key in ("one_step", "oracle", "features", "cost", "horizon_panel", "days"):
        np.testing.assert_array_equal(out[key], month[key][keep])
    # The input month is left as it was.
    assert len(month["days"]) == D and month["realized"].values.shape[0] == D
