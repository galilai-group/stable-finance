"""Joint payoff covariance, observation timing and residual-risk references."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from stable_finance.evaluation.portfolio.joint_risk import (
    JointReturnPanel, JointRiskModel, estimate_joint_risk,
)
from stable_finance.evaluation.returns.contract import ForwardReturns


def history(seed=2, d=20, n=2, horizons=(60, 600), delay=0):
    rng = np.random.default_rng(seed)
    increments = rng.normal(size=(d, n, len(horizons)))
    # Different interval lengths retain visibly different marginal scales.
    increments *= np.arange(1, len(horizons)+1)[None, None, :]
    panel = JointReturnPanel(np.cumsum(increments, axis=2), np.arange(d)*1000,
                             [f"A{i}" for i in range(n)], horizons,
                             entry_delay_seconds=delay)
    available = panel.decisions[:, None] + delay + np.asarray(horizons)[None]
    return panel, available, increments


def flat(values):
    d, n, k = values.shape
    return values.transpose(0, 2, 1).reshape(d, k*n)


def test_panel_encodes_known_decision_price_payoffs_and_copies_all_axes():
    values, decisions, assets, horizons = np.zeros((2, 2, 2)), [0, 1], ["A", "B"], [60, 600]
    panel = JointReturnPanel(values, decisions, assets, horizons, entry_delay_seconds=5)
    values[:] = 100
    decisions[0], assets[0], horizons[0] = -100, "C", 1
    np.testing.assert_array_equal(panel.values, 0)
    np.testing.assert_array_equal(panel.decisions, [0, 1])
    np.testing.assert_array_equal(panel.assets, ["A", "B"])
    np.testing.assert_array_equal(panel.horizons, [60, 600])
    assert panel.return_basis == "decision_price_payoff"
    assert panel.entry_delay_seconds == 5
    with pytest.raises(ValueError, match="read-only"):
        panel.values[0, 0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        panel.entry_delay_seconds = 0


@pytest.mark.parametrize("kwargs", [
    {"horizons": [600, 60]}, {"horizons": [60, 60]}, {"horizons": [60, np.nan]},
    {"horizons": [60, 60.5]}, {"horizons": [60, np.inf]},
    {"assets": ["A", "A"]}, {"assets": [None, "A"]},
    {"decisions": [1, 0]}, {"decisions": [0, 0]},
    {"entry_delay_seconds": -1}, {"entry_delay_seconds": True},
    {"return_basis": "future_entry_simple"},
])
def test_panel_rejects_ambiguous_axes_or_unknown_units(kwargs):
    inputs = dict(values=np.zeros((2, 2, 2)), decisions=[0, 1], assets=["A", "B"], horizons=[60, 600])
    with pytest.raises(ValueError):
        JointReturnPanel(**{**inputs, **kwargs})


def test_supplied_tensor_risk_has_interval_major_order_and_immutable_storage():
    matrix = np.arange(16).reshape(4, 4)
    covariance = matrix @ matrix.T
    tensor = covariance.reshape(2, 2, 2, 2)
    risk = JointRiskModel(tensor, ["A", "B"], [60, 600], information_cutoff=10,
                          entry_delay_seconds=5)
    tensor[:] = 0
    np.testing.assert_allclose(risk.covariance, matrix @ matrix.T, atol=1e-10)
    assert risk.blocks.shape == (2, 2, 2, 2)
    assert risk.blocks[1, 0, 0, 1] == pytest.approx(risk.covariance[2, 1])
    with pytest.raises(ValueError, match="read-only"):
        risk.blocks[0, 0, 0, 0] = 1
    assert risk.return_basis == "decision_price_payoff_increments"
    risk.assert_available(10)
    risk.assert_available(11)
    with pytest.raises(ValueError, match="later"):
        risk.assert_available(9)


@pytest.mark.parametrize("kwargs", [
    {"covariance": [[1, .1], [.2, 1]]}, {"covariance": [[1, 0], [0, -1]]},
    {"covariance": [[1, 0], [0, np.inf]]}, {"covariance": [[1]]},
    {"source_observations": True}, {"source_observations": -1},
    {"shrinkage": -0.1}, {"shrinkage": np.inf}, {"shrinkage": True},
    {"information_cutoff": np.nan}, {"return_basis": "simple_returns"},
    {"risk_source": "invented"}, {"entry_delay_seconds": 1.5},
])
def test_supplied_risk_rejects_invalid_covariance_and_provenance(kwargs):
    inputs = dict(covariance=np.eye(2), assets=["A"], horizons=[60, 600], information_cutoff=10)
    with pytest.raises(ValueError):
        JointRiskModel(**{**inputs, **kwargs})


def test_supplied_zero_psd_model_is_explicitly_allowed():
    risk = JointRiskModel(np.zeros((2, 2)), ["A"], [60, 600], information_cutoff=10)
    np.testing.assert_array_equal(risk.covariance, 0)


def test_fortran_ordered_covariance_still_provides_a_readonly_block_view():
    risk = JointRiskModel(np.asfortranarray(np.eye(4)), ["A", "B"], [60, 600], 0)
    assert risk.covariance.flags.c_contiguous
    assert not risk.blocks.flags.writeable


def test_float_horizon_at_signed_integer_overflow_is_rejected():
    with pytest.raises(ValueError, match="integer-second range"):
        JointReturnPanel(np.zeros((1, 1, 1)), [0], ["A"], [float(2**63)])


def test_unshrunk_joint_covariance_matches_exact_increment_sample_covariance():
    panel, available, increments = history()
    risk = estimate_joint_risk(panel, available_at=available, as_of=available.max(), shrinkage=0)
    np.testing.assert_allclose(risk.covariance, np.cov(flat(increments), rowvar=False), rtol=1e-12, atol=1e-12)
    assert risk.source_observations == len(panel.decisions)
    assert risk.information_cutoff == available.max()
    assert risk.risk_source == "unconditional_payoff_covariance"
    assert risk.shrinkage == 0
    assert risk.estimator == "fixed_correlation_shrinkage"


def test_cumulative_to_disjoint_covariance_is_the_full_linear_difference_transform():
    panel, available, _ = history()
    risk = estimate_joint_risk(panel, available_at=available, as_of=available.max(), shrinkage=0)
    difference = np.array([[1, 0], [-1, 1]])
    transform = np.kron(difference, np.eye(2))
    cumulative_covariance = np.cov(flat(panel.values), rowvar=False)
    expected = transform @ cumulative_covariance @ transform.T
    np.testing.assert_allclose(risk.covariance, expected, atol=1e-12)
    assert np.abs(risk.blocks[0, :, 1, :]).max() > .01


def test_full_correlation_shrinkage_preserves_each_interval_marginal_variance():
    panel, available, increments = history()
    risk = estimate_joint_risk(panel, available_at=available, as_of=available.max(), shrinkage=1)
    variances = np.var(flat(increments), axis=0, ddof=1)
    np.testing.assert_allclose(risk.covariance, np.diag(variances), atol=1e-12)
    assert variances[2:].mean() > 2*variances[:2].mean()


def test_automatic_intensity_is_estimated_on_standardized_complete_paths():
    panel, available, increments = history(seed=4)
    risk = estimate_joint_risk(panel, available_at=available, as_of=available.max())
    paths = flat(increments)
    sd = paths.std(axis=0, ddof=1)
    standardized = (paths-paths.mean(axis=0))/sd
    intensity = LedoitWolf(assume_centered=True).fit(standardized).shrinkage_
    correlation = standardized.T @ standardized/(len(paths)-1)
    expected = ((1-intensity)*correlation+intensity*np.eye(paths.shape[1]))*np.outer(sd, sd)
    np.testing.assert_allclose(risk.covariance, expected, atol=1e-12)
    assert risk.shrinkage == pytest.approx(intensity)
    np.testing.assert_allclose(np.diag(risk.covariance), sd**2, atol=1e-12)
    assert np.linalg.eigvalsh(risk.covariance).min() >= -1e-12


def test_rows_are_selected_as_complete_mature_paths_not_pairwise_fragments():
    panel, available, increments = history(d=6)
    values = panel.values.copy()
    values[1, 0, 0] = np.nan
    values[3, 1, 1] = np.nan
    panel = replace(panel, values=values)
    cutoff = available[4, 0]  # Path four's short label is mature; its long one is not.
    risk = estimate_joint_risk(panel, available_at=available, as_of=cutoff, shrinkage=0)
    np.testing.assert_allclose(risk.covariance, np.cov(flat(increments[[0, 2]]), rowvar=False), atol=1e-12)
    assert risk.source_observations == 2
    assert risk.information_cutoff == available[2, 1]


def test_future_values_including_infinities_cannot_change_the_earlier_risk_estimate():
    panel, available, _ = history(d=8)
    cutoff = available[3, -1]
    expected = estimate_joint_risk(panel, available_at=available, as_of=cutoff)
    changed = panel.values.copy()
    changed[4:] = np.inf
    got = estimate_joint_risk(replace(panel, values=changed), available_at=available, as_of=cutoff)
    np.testing.assert_array_equal(got.covariance, expected.covariance)
    assert got.source_observations == expected.source_observations == 4
    assert got.information_cutoff == expected.information_cutoff


def test_mature_infinities_and_zero_variance_are_not_silently_filled():
    panel, available, _ = history(d=4)
    changed = panel.values.copy()
    changed[1, 0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        estimate_joint_risk(replace(panel, values=changed), available_at=available, as_of=available.max())
    changed = panel.values.copy()
    changed[:, 0, 0] = 0
    with pytest.raises(ValueError, match="positive finite observed variance"):
        estimate_joint_risk(replace(panel, values=changed), available_at=available, as_of=available.max())


@pytest.mark.parametrize("kwargs", [{"shrinkage": "other"}, {"shrinkage": -1},
                                     {"shrinkage": np.nan}, {"shrinkage": True},
                                     {"min_samples": 1}, {"min_samples": True}])
def test_estimation_parameters_are_explicit_and_valid(kwargs):
    panel, available, _ = history()
    with pytest.raises(ValueError):
        estimate_joint_risk(panel, available_at=available, as_of=available.max(), **kwargs)


def test_ambiguous_forward_returns_cannot_enter_the_joint_risk_estimator():
    panel, available, _ = history()
    ordinary = ForwardReturns(panel.values, panel.decisions, panel.assets, panel.horizons)
    with pytest.raises(TypeError, match="JointReturnPanel"):
        estimate_joint_risk(ordinary, available_at=available, as_of=available.max())


def test_availability_requires_every_horizon_to_mature_and_respects_the_declared_delay():
    panel, available, _ = history(d=4, delay=5)
    with pytest.raises(ValueError, match="complete mature paths"):
        estimate_joint_risk(panel, available_at=available, as_of=available[1, -1]-1)
    risk = estimate_joint_risk(panel, available_at=available, as_of=available[1, -1])
    assert risk.source_observations == 2
    assert risk.entry_delay_seconds == 5
    wrong = available-1
    with pytest.raises(ValueError, match="anchor plus delay"):
        estimate_joint_risk(panel, available_at=wrong, as_of=available.max())


def test_availability_axes_and_clocks_cannot_be_paired_by_position():
    panel, available, _ = history()
    with pytest.raises(ValueError, match="shape"):
        estimate_joint_risk(panel, available_at=available[:, 0], as_of=available.max())
    wrong = available.copy()
    wrong[:, 1] = wrong[:, 0]-1
    with pytest.raises(ValueError, match="nondecreasing"):
        estimate_joint_risk(panel, available_at=wrong, as_of=available.max())
    with pytest.raises(ValueError, match="same time basis"):
        estimate_joint_risk(panel, available_at=available, as_of=np.datetime64("2020-01-01"))


def test_asset_permutation_preserves_the_corresponding_joint_covariance():
    panel, available, _ = history(n=3)
    order = np.array([2, 0, 1])
    expected = estimate_joint_risk(panel, available_at=available, as_of=available.max())
    shuffled = replace(panel, values=panel.values[:, order], assets=panel.assets[order])
    got = estimate_joint_risk(shuffled, available_at=available, as_of=available.max())
    coordinates = np.r_[order, order+3]
    np.testing.assert_allclose(got.covariance, expected.covariance[np.ix_(coordinates, coordinates)], atol=1e-12)


def test_datetime64_nanosecond_cutoffs_are_not_rounded_to_seconds():
    panel, _, _ = history(d=4)
    base = np.datetime64("2020-07-01T10:00:00", "ns")
    decisions = base+np.arange(4)*np.timedelta64(1000, "s")
    panel = replace(panel, decisions=decisions)
    available = decisions[:, None] + panel.horizons[None].astype("timedelta64[s]")
    cutoff = available[2, -1]-np.timedelta64(1, "ns")
    risk = estimate_joint_risk(panel, available_at=available, as_of=cutoff)
    assert risk.source_observations == 2
    risk.assert_available(available[1, -1])
    with pytest.raises(ValueError, match="later"):
        risk.assert_available(available[1, -1]-np.timedelta64(1, "ns"))


def test_aware_native_clocks_compare_instants_across_timezones():
    panel, _, _ = history(d=4)
    ny = ZoneInfo("America/New_York")
    start = datetime(2020, 7, 1, 10, tzinfo=ny)
    decisions = np.asarray([start+timedelta(seconds=1000*i) for i in range(4)], dtype=object)
    panel = replace(panel, decisions=decisions)
    available = np.asarray([[value+timedelta(seconds=int(h)) for h in panel.horizons]
                             for value in decisions], dtype=object)
    cutoff = available[2, -1].astimezone(timezone.utc)
    risk = estimate_joint_risk(panel, available_at=available, as_of=cutoff)
    assert risk.source_observations == 3
    assert risk.information_cutoff == cutoff
    risk.assert_available(cutoff.astimezone(ny))
    with pytest.raises(ValueError, match="same time basis"):
        risk.assert_available(cutoff.replace(tzinfo=None))


def test_dst_fold_cutoff_is_compared_by_instant_not_equal_wall_time():
    ny = ZoneInfo("America/New_York")
    earlier = datetime(2020, 11, 1, 1, 30, tzinfo=ny, fold=0)
    later = datetime(2020, 11, 1, 1, 30, tzinfo=ny, fold=1)
    risk = JointRiskModel([[1]], ["A"], [60], information_cutoff=later)
    with pytest.raises(ValueError, match="later"):
        risk.assert_available(earlier)
    risk.assert_available(later)


def test_residual_covariance_uses_only_issued_historical_forecasts():
    panel, available, _ = history(seed=3)
    rng = np.random.default_rng(5)
    predicted = panel.values*.6+rng.normal(scale=.1, size=panel.values.shape)
    forecasts = replace(panel, values=predicted)
    issued = np.broadcast_to(panel.decisions[:, None], available.shape)
    risk = estimate_joint_risk(panel, available_at=available, as_of=available.max(),
        historical_forecasts=forecasts, forecast_available_at=issued, shrinkage=0)
    residual = panel.values-predicted
    increments = np.diff(residual, axis=2, prepend=np.zeros((*residual.shape[:2], 1)))
    np.testing.assert_allclose(risk.covariance, np.cov(flat(increments), rowvar=False), atol=1e-12)
    assert risk.risk_source == "historical_forecast_residual_covariance"
    with pytest.raises(ValueError, match="no later than each anchor"):
        estimate_joint_risk(panel, available_at=available, as_of=available.max(),
            historical_forecasts=forecasts, forecast_available_at=issued+1)


def test_residual_risk_drops_an_incomplete_forecast_as_one_path_and_ignores_future_values():
    panel, available, _ = history(seed=7, d=8)
    predicted = .5*panel.values
    predicted[1, 0, 0] = np.nan
    predicted[5:] = np.inf
    forecasts = replace(panel, values=predicted)
    issued = np.broadcast_to(panel.decisions[:, None], available.shape)
    risk = estimate_joint_risk(panel, available_at=available, as_of=available[4, -1],
        historical_forecasts=forecasts, forecast_available_at=issued, shrinkage=0)
    selected = panel.values[[0, 2, 3, 4]]*.5
    increments = np.diff(selected, axis=2, prepend=np.zeros((4, 2, 1)))
    np.testing.assert_allclose(risk.covariance, np.cov(flat(increments), rowvar=False), atol=1e-12)
    assert risk.source_observations == 4
    assert risk.information_cutoff == available[4, -1]


def test_perfect_historical_forecasts_do_not_receive_an_invented_risk_floor():
    panel, available, _ = history()
    issued = np.broadcast_to(panel.decisions[:, None], available.shape)
    with pytest.raises(ValueError, match="positive finite observed variance"):
        estimate_joint_risk(panel, available_at=available, as_of=available.max(),
                            historical_forecasts=panel, forecast_available_at=issued)


@pytest.mark.parametrize("mismatch", ["assets", "decisions", "delay", "missing_availability", "extra_availability"])
def test_historical_forecasts_require_exact_semantics_and_availability(mismatch):
    panel, available, _ = history()
    forecasts = panel
    issued = np.broadcast_to(panel.decisions[:, None], available.shape)
    if mismatch == "assets":
        forecasts = replace(panel, assets=panel.assets[::-1])
    elif mismatch == "decisions":
        forecasts = replace(panel, decisions=panel.decisions-1)
    elif mismatch == "delay":
        forecasts = replace(panel, entry_delay_seconds=5)
    elif mismatch == "missing_availability":
        issued = None
    elif mismatch == "extra_availability":
        forecasts = None
    with pytest.raises(ValueError):
        estimate_joint_risk(panel, available_at=available, as_of=available.max(),
                            historical_forecasts=forecasts, forecast_available_at=issued)
