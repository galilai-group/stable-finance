from __future__ import annotations

import numpy as np
import pytest

from stable_finance.dataset.anchors import AnchorSpec
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES,
    anchor_targets,
    anchor_indices,
    forward_vwap,
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
