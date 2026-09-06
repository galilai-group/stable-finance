from __future__ import annotations

import numpy as np

from stable_finance.dataset import (
    AnchorTargetStats,
    TARGET_TRANSFORMS,
    build_cross_section_metadata,
)


def _stats(tmp_path):
    values = np.array(
        [-2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
         7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        dtype=np.float32,
    )
    quantiles = np.linspace(values[0], values[-1], 64, dtype=np.float32)
    path = tmp_path / "2023-01.npz"
    np.savez(
        path,
        dates=np.array(["2023-01-03"]),
        anchors=np.array([300], dtype=np.int32),
        types=np.array(["return"]),
        horizons=np.array([900], dtype=np.int32),
        mu=np.zeros((1, 1, 1, 1), dtype=np.float32),
        sigma=np.full((1, 1, 1, 1), 2.0, dtype=np.float32),
        count=np.full((1, 1, 1, 1), len(values), dtype=np.int32),
        quantiles=quantiles.reshape(1, 1, 1, 1, -1),
        quantile_levels=np.linspace(0, 1, 64, dtype=np.float32),
        sorted_values=values.reshape(1, 1, 1, 1, -1),
    )
    return AnchorTargetStats(path)


def test_transform_returns_every_representation_and_selects_by_name(tmp_path):
    stats = _stats(tmp_path)
    raw = np.array([[-1.0]])
    bundle = stats.transform(raw, "2023-01-03", 300, ["return"], [900])

    assert tuple(bundle.as_dict()) == TARGET_TRANSFORMS
    np.testing.assert_array_equal(bundle.raw, raw)
    assert bundle.zscore.item() == -0.5
    assert bundle.uniform.item() == 2.5 / 21
    for name in TARGET_TRANSFORMS:
        np.testing.assert_array_equal(bundle.select(name), bundle.as_dict()[name])


def test_uniform_uses_open_interval_endpoints(tmp_path):
    stats = _stats(tmp_path)

    def score(value):
        return stats.uniform_score(
            np.array([[value]]), "2023-01-03", 300, ["return"], [900]
        ).item()

    assert score(-2.0) == 1 / 21
    assert score(16.0) == 20 / 21


def test_uniform_refuses_approximate_legacy_tables(tmp_path):
    stats = _stats(tmp_path)
    stats.sorted_values = None
    with np.testing.assert_raises_regex(ValueError, "exact order statistics"):
        stats.uniform_score(
            np.array([[0.0]]), "2023-01-03", 300, ["return"], [900]
        )


def test_table_builder_owns_moments_quantiles_and_exact_order_statistics():
    values = np.array([[[1.0]], [[3.0]], [[2.0]]], dtype=np.float32)
    sums = values.sum(axis=0, keepdims=True)
    squares = (values * values).sum(axis=0, keepdims=True)
    counts = np.full((1, 1, 1), 3, dtype=np.int64)

    table = build_cross_section_metadata(
        sums, squares, counts, [values], np.array([0.0, 0.5, 1.0])
    )

    np.testing.assert_array_equal(table["sorted_values"].ravel(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(table["quantiles"].ravel(), [1.0, 2.0, 3.0])
    # The cell is intentionally thinner than the production minimum.
    assert np.isnan(table["mu"]).all()
