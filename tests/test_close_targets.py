"""Measuring to the closing auction rather than to a fixed offset."""

import numpy as np
import pytest

from stable_finance.dataset.anchors import (
    CLOSING_AUCTION_SPEC,
    DEFAULT_ANCHOR_SPEC,
    TO_CLOSE,
    AnchorSpec,
)
from stable_finance.dataset.outcomes import anchor_targets, resolve_future
from stable_finance.dataset.schema import MARKET_SCHEMA

VWAP = MARKET_SCHEMA.index("vwap_all")
VOLUME = MARKET_SCHEMA.index("volume")


def session(prices, *, spec=CLOSING_AUCTION_SPEC, volume=100.0):
    """A session whose VWAP is exactly ``prices`` in every second."""
    features = np.zeros((spec.rows_retained, len(MARKET_SCHEMA.columns)))
    features[:, VWAP] = prices
    features[:, VOLUME] = volume
    return features


class TestGeometry:
    def test_the_tail_does_not_move_the_session(self):
        # Anchors, crops and views index off session_seconds, so retaining a
        # tail must leave every one of them exactly where it was.
        assert CLOSING_AUCTION_SPEC.session_seconds == \
            DEFAULT_ANCHOR_SPEC.session_seconds
        assert CLOSING_AUCTION_SPEC.anchors_per_session == \
            DEFAULT_ANCHOR_SPEC.anchors_per_session
        assert CLOSING_AUCTION_SPEC.rows_retained == \
            DEFAULT_ANCHOR_SPEC.rows_retained + 60

    def test_the_auction_is_the_first_row_after_the_session(self):
        assert CLOSING_AUCTION_SPEC.close_index == 23_400

    def test_a_tail_shorter_than_the_window_cannot_measure_the_close(self):
        assert not DEFAULT_ANCHOR_SPEC.measures_close
        assert not AnchorSpec(post_close_seconds=59).measures_close
        assert AnchorSpec(post_close_seconds=60).measures_close


class TestResolveFuture:
    def test_a_fixed_horizon_measures_at_anchor_plus_horizon(self):
        future = resolve_future(np.array([0, 100]), np.array([30, 60]), None)
        assert future.tolist() == [[30, 60], [130, 160]]

    def test_to_close_measures_at_the_same_row_from_every_anchor(self):
        future = resolve_future(np.array([0, 100, 200]),
                                np.array([30, TO_CLOSE]), close_index=1000)
        assert future.tolist() == [[30, 1000], [130, 1000], [230, 1000]]

    def test_to_close_without_a_close_index_is_refused_not_guessed(self):
        with pytest.raises(ValueError, match="close_index"):
            resolve_future(np.array([0]), np.array([TO_CLOSE]), None)


class TestToCloseTargets:
    def test_it_is_the_return_from_the_anchor_to_the_auction(self):
        spec = CLOSING_AUCTION_SPEC
        prices = np.full(spec.rows_retained, 100.0)
        prices[spec.close_index:] = 101.0        # the auction prints 1% up
        anchors = np.array([0, 5_000, 20_000])
        out = anchor_targets(session(prices), [TO_CLOSE], anchors,
                             types=("return",), spec=spec)
        assert out[:, 0, 0] == pytest.approx(0.01)

    def test_every_anchor_can_answer_it_where_a_fixed_horizon_cannot(self):
        # The point of the sentinel: a 2-hour horizon is unmeasurable from the
        # late session, so a fixed-horizon study silently keeps only the early
        # anchors and becomes a time-of-day study.
        spec = CLOSING_AUCTION_SPEC
        prices = np.full(spec.rows_retained, 100.0)
        prices[spec.close_index:] = 102.0
        anchors = np.arange(0, spec.session_seconds, spec.step_seconds)

        fixed = anchor_targets(session(prices), [7_200], anchors,
                               types=("return",), spec=spec)[:, 0, 0]
        to_close = anchor_targets(session(prices), [TO_CLOSE], anchors,
                                  types=("return",), spec=spec)[:, 0, 0]
        assert np.isnan(fixed).sum() > 0
        # Only the final anchors, which have no room left to hold anything.
        assert np.isfinite(to_close).sum() > np.isfinite(fixed).sum()
        assert np.isfinite(to_close[:-1]).all()

    def test_an_anchor_at_or_past_the_close_has_no_holding_period(self):
        spec = CLOSING_AUCTION_SPEC
        prices = np.full(spec.rows_retained, 100.0)
        prices[spec.close_index:] = 101.0
        out = anchor_targets(session(prices), [TO_CLOSE],
                             np.array([spec.close_index - 1, spec.close_index]),
                             types=("return",), spec=spec)
        assert np.isfinite(out[0, 0, 0])
        assert np.isnan(out[1, 0, 0])

    def test_it_travels_beside_fixed_horizons_as_one_more_column(self):
        spec = CLOSING_AUCTION_SPEC
        prices = np.full(spec.rows_retained, 100.0)
        prices[600:] = 100.5
        prices[spec.close_index:] = 103.0
        out = anchor_targets(session(prices), [300, TO_CLOSE, 900],
                             np.array([0]), types=("return",), spec=spec)
        assert out.shape == (1, 1, 3)
        assert out[0, 0, 0] == pytest.approx(0.0)      # 300s: still 100.0
        assert out[0, 0, 1] == pytest.approx(0.03)     # to the auction
        assert out[0, 0, 2] == pytest.approx(0.005)    # 900s: 100.5

    def test_a_spec_without_the_tail_refuses_rather_than_returning_nan(self):
        prices = np.full(DEFAULT_ANCHOR_SPEC.rows_retained, 100.0)
        with pytest.raises(ValueError, match="retains the auction window"):
            anchor_targets(session(prices, spec=DEFAULT_ANCHOR_SPEC),
                           [TO_CLOSE], np.array([0]),
                           types=("return",), spec=DEFAULT_ANCHOR_SPEC)

    def test_the_tail_changes_no_value_it_only_adds_boundary_coverage(self):
        # The regression that matters: retaining post-close rows must not move
        # any target that was already measurable, or every cached number
        # silently changes. It DOES make new cells measurable at the boundary,
        # where the forward window previously ran off the end of the session.
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(scale=0.01, size=23_460))
        anchors = np.arange(0, 23_400, 300)
        without = anchor_targets(
            session(prices[:23_400], spec=DEFAULT_ANCHOR_SPEC),
            [300, 600, 3600], anchors, types=("return",),
            spec=DEFAULT_ANCHOR_SPEC)
        with_tail = anchor_targets(session(prices), [300, 600, 3600], anchors,
                                   types=("return",), spec=CLOSING_AUCTION_SPEC)
        was = np.isfinite(without)
        assert without[was] == pytest.approx(with_tail[was], rel=1e-12)
        gained = np.isfinite(with_tail) & ~was
        assert gained.sum() > 0

    def test_a_boundary_cell_the_tail_unlocks_measures_the_auction(self):
        # Anchor 22800 at a 600s horizon lands exactly on the close, so the
        # cell the tail unlocks is the to-close return by another name. This
        # is why the tail is a fix for the horizon grid and not just an extra
        # target: the grid's missing corners were always the auction.
        spec = CLOSING_AUCTION_SPEC
        prices = np.full(spec.rows_retained, 100.0)
        prices[spec.close_index:] = 104.0
        anchor = np.array([22_800])
        fixed = anchor_targets(session(prices), [600], anchor,
                               types=("return",), spec=spec)[0, 0, 0]
        to_close = anchor_targets(session(prices), [TO_CLOSE], anchor,
                                  types=("return",), spec=spec)[0, 0, 0]
        assert fixed == pytest.approx(0.04)
        assert fixed == pytest.approx(to_close)

        # ...and without the tail that cell is simply absent, which is the
        # state the cached targets are in today.
        absent = anchor_targets(
            session(prices[:23_400], spec=DEFAULT_ANCHOR_SPEC), [600], anchor,
            types=("return",), spec=DEFAULT_ANCHOR_SPEC)[0, 0, 0]
        assert np.isnan(absent)
