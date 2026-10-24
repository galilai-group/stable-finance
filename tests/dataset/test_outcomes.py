"""Target families by role: returns, spread and volatility changes, and measuring to the close."""
from __future__ import annotations

import numpy as np
import pytest

from stable_finance.dataset import (
    MARKET_SCHEMA,
    AnchorSpec,
    FeatureSchema,
    MissingRolesError,
    anchor_targets,
    check_targets,
    compute_pair_targets,
    forward_price,
    forward_vwap,
    supported_targets,
)
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES,
    anchor_indices,
    get_target_names,
    window_realized_volatility,
    window_spread,
)



def _features(length: int = 1_200) -> np.ndarray:
    features = np.zeros((length, 9), dtype=np.float64)
    price = 100.0 + np.arange(length) * 0.01
    features[:, 0] = price - 0.05
    features[:, 1] = price
    features[:, 4] = price + 0.05
    features[:, 7] = 1.0
    return features


def test_anchor_spec_validates_injectable_geometry():
    spec = AnchorSpec(
        session_seconds=600,
        step_seconds=60,
        measurement_window_seconds=10,
    )
    assert spec.anchors_per_session == 10
    np.testing.assert_array_equal(anchor_indices(spec=spec), np.arange(0, 600, 60))
    with pytest.raises(ValueError, match="step_seconds"):
        AnchorSpec(step_seconds=0)


def test_window_outcomes_follow_observable_columns():
    features = _features()
    assert forward_vwap(features, 10, 4) == pytest.approx(
        features[10:14, 1].mean()
    )
    assert window_spread(features, 10, 4) == pytest.approx(0.1)
    assert np.isfinite(window_realized_volatility(features, 10, 60))


def test_anchor_targets_have_stable_type_order_and_no_close_clamp():
    features = _features(1_000)
    anchors = np.array([100, 900])
    targets = anchor_targets(features, [60], anchors)

    assert targets.shape == (2, len(ANCHOR_TARGET_TYPES), 1)
    expected = forward_vwap(features, 160) / forward_vwap(features, 100) - 1.0
    assert targets[0, ANCHOR_TARGET_TYPES.index("return"), 0] == pytest.approx(
        expected
    )
    assert np.isnan(targets[1]).all()


def test_pair_targets_match_anchor_targets_for_all_tasks_and_horizons():
    features = _features(2_000)
    horizons = [60, 300, 900]
    index = 500
    anchor = anchor_targets(features, horizons, np.array([index]))[0]
    pair = compute_pair_targets(
        features, index, horizons, list(ANCHOR_TARGET_TYPES)
    ).reshape(len(ANCHOR_TARGET_TYPES), len(horizons))
    np.testing.assert_allclose(pair, anchor, rtol=1e-6, equal_nan=True)


def test_pair_targets_only_emit_requested_work_in_requested_order():
    features = _features(1_000)
    selected = compute_pair_targets(
        features, 100, [60, 300], ["spread_change"]
    )
    assert selected.shape == (2,)
    assert get_target_names([60, 300], ["spread_change"]) == [
        "spread_change_060",
        "spread_change_300",
    ]
    with pytest.raises(ValueError, match="unknown target"):
        compute_pair_targets(features, 100, [60], ["not_a_target"])


def _univariate_rows(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return (100 * np.exp(rng.normal(0, 1e-3, n).cumsum()))[:, None]


def test_univariate_return_is_the_windowed_mean_midpoint():
    schema = FeatureSchema.univariate()
    prices = _univariate_rows()
    out = anchor_targets(prices, [120], np.asarray([0, 300]), types=("return",),
                         spec=AnchorSpec(session_seconds=600), schema=schema)
    window = 60
    for k, t in enumerate((0, 300)):
        expect = prices[t + 120:t + 120 + window, 0].mean() / prices[t:t + window, 0].mean() - 1
        np.testing.assert_allclose(out[k, 0, 0], expect, rtol=1e-12)


def test_univariate_refuses_spread_change_naming_the_missing_roles():
    schema = FeatureSchema.univariate()
    assert supported_targets(schema) == ("return", "volatility_change")
    with pytest.raises(MissingRolesError, match="bid_price, ask_price"):
        anchor_targets(_univariate_rows(), [60], np.asarray([0]), types=("spread_change",),
                       spec=AnchorSpec(session_seconds=600), schema=schema)
    with pytest.raises(MissingRolesError, match="vwap"):
        check_targets(schema, ["return"], price="vwap")


def test_market1t_default_price_is_the_one_minute_vwap_and_mid_is_selectable():
    rng = np.random.default_rng(3)
    f = np.empty((400, 9))
    mid = 50 + rng.normal(0, 0.01, 400).cumsum()
    f[:, 0], f[:, 4] = mid - 0.01, mid + 0.01
    f[:, 1] = mid + rng.normal(0, 0.002, 400)
    f[:, 2], f[:, 3] = f[:, 1] + 0.01, f[:, 1] - 0.01
    f[:, 5:7] = 100
    f[:, 7] = rng.poisson(5, 400)
    f[:, 8] = 1
    assert forward_price(f, 10) == forward_vwap(f, 10)
    np.testing.assert_allclose(forward_price(f, 10, price="mid"), mid[10:70].mean(), rtol=1e-12)
    vwap = anchor_targets(f, [120], np.asarray([0]), types=("return",))
    by_mid = anchor_targets(f, [120], np.asarray([0]), types=("return",), price="mid")
    np.testing.assert_allclose(by_mid[0, 0, 0], mid[120:180].mean() / mid[:60].mean() - 1,
                               rtol=1e-9)
    assert vwap[0, 0, 0] != by_mid[0, 0, 0]


def test_minute_bars_convert_horizons_and_windows_to_rows():
    schema = FeatureSchema.univariate()
    prices = _univariate_rows(390)                     # one row per minute
    spec = AnchorSpec(session_seconds=390 * 60, step_seconds=300,
                      measurement_window_seconds=60, bar_seconds=60)
    out = anchor_targets(prices, [300, 3600], np.asarray([0, 5]), types=("return",),
                         spec=spec, schema=schema)
    for k, t in enumerate((0, 5)):
        np.testing.assert_allclose(out[k, 0], [prices[t + 5, 0] / prices[t, 0] - 1,
                                               prices[t + 60, 0] / prices[t, 0] - 1])
    with pytest.raises(ValueError, match="whole"):
        anchor_targets(prices, [90], np.asarray([0]), types=("return",), spec=spec, schema=schema)
    single = compute_pair_targets(prices, 5, [300], ["return"], schema=schema, bar_seconds=60)
    np.testing.assert_allclose(single, out[1, 0, :1], rtol=1e-6)


def test_reordered_columns_move_every_target_with_them():
    """Targets used to read Market-1T's fixed indices whatever the schema said."""
    rng = np.random.default_rng(5)
    f = np.abs(rng.normal(50, 1, (300, 9))) + 1
    order = list(reversed(range(9)))
    columns = tuple(MARKET_SCHEMA.columns[i] for i in order)
    reordered = FeatureSchema(
        columns=columns, forward_fill=MARKET_SCHEMA.forward_fill,
        zero_fill=MARKET_SCHEMA.zero_fill,
        aggregation=tuple((c, MARKET_SCHEMA.aggregation_rules[c]) for c in columns),
        normalization=MARKET_SCHEMA.normalization, roles=MARKET_SCHEMA.roles)
    a = anchor_targets(f, [60, 120], np.asarray([0, 50]))
    b = anchor_targets(f[:, order], [60, 120], np.asarray([0, 50]), schema=reordered)
    np.testing.assert_array_equal(a, b)
