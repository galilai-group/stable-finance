"""Decision-anchor geometry for forward-outcome construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorSpec:
    """Geometry shared by target tables, training crops, and evaluation.

    These defaults describe the current regular-session recipe. Keeping them
    in a value object lets callers inject a different grid without treating
    dataset geometry as model configuration.
    """

    session_seconds: int = 23_400
    step_seconds: int = 300
    measurement_window_seconds: int = 60
    post_close_seconds: int = 0

    def __post_init__(self) -> None:
        if self.session_seconds < 1:
            raise ValueError("session_seconds must be positive")
        if self.step_seconds < 1:
            raise ValueError("step_seconds must be positive")
        if self.step_seconds > self.session_seconds:
            raise ValueError("step_seconds cannot exceed session_seconds")
        if self.measurement_window_seconds < 1:
            raise ValueError("measurement_window_seconds must be positive")
        if self.post_close_seconds < 0:
            raise ValueError("post_close_seconds cannot be negative")

    @property
    def anchors_per_session(self) -> int:
        return self.session_seconds // self.step_seconds

    @property
    def close_index(self) -> int:
        """Row of the closing auction: the first second after the session.

        Rows 0..session_seconds-1 are the regular session, so the auction
        prints at ``session_seconds`` itself. It is not part of the session
        and never enters a view.
        """
        return self.session_seconds

    @property
    def rows_retained(self) -> int:
        """Rows a stored session holds, regular session plus the tail."""
        return self.session_seconds + self.post_close_seconds

    @property
    def measures_close(self) -> bool:
        """Is enough of the tail retained to measure the auction window?

        The auction is read as a VWAP over ``[close_index, +window)`` like
        every other forward measurement, so the tail has to cover the window
        or the target cannot be formed.
        """
        return self.post_close_seconds >= self.measurement_window_seconds


DEFAULT_ANCHOR_SPEC = AnchorSpec()

# Named compatibility constants for concise numerical code and downstream
# scripts. New APIs should accept an AnchorSpec where geometry is configurable.
SESSION_LEN = DEFAULT_ANCHOR_SPEC.session_seconds
ANCHOR_STEP = DEFAULT_ANCHOR_SPEC.step_seconds
RETURN_VWAP_WINDOW = DEFAULT_ANCHOR_SPEC.measurement_window_seconds
ANCHORS_PER_DAY = DEFAULT_ANCHOR_SPEC.anchors_per_session
DEFAULT_TARGET_HORIZONS = (300, 600, 900, 1800, 3600, 7200)

#: Horizon sentinel meaning "measure to the closing auction" rather than to a
#: fixed offset. A fixed horizon asks the same question at every anchor and so
#: answers a different one as the session runs out: from 15:00 a two-hour
#: return is unmeasurable, and the anchors where it IS measurable are exactly
#: the early ones, which silently turns a horizon study into a time-of-day
#: study. Measuring to the close instead asks one question -- what happens
#: between here and the auction -- that every anchor can answer, with a
#: holding period that shortens through the day as a real position's does.
#:
#: It travels inside the ordinary ``horizons`` tuple so that target tables,
#: the day store and the panel cache carry it as one more column without any
#: of them learning about it. Negative because no real offset can collide.
TO_CLOSE = -1

#: The recipe that can measure it: the stored session keeps one measurement
#: window past the close. The regular session is unchanged at 23,400 rows, so
#: anchors, crops and views are bit-identical to the default spec -- the tail
#: exists only for the target builder to read.
CLOSING_AUCTION_SPEC = AnchorSpec(post_close_seconds=RETURN_VWAP_WINDOW)
