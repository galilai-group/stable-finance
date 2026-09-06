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

    def __post_init__(self) -> None:
        if self.session_seconds < 1:
            raise ValueError("session_seconds must be positive")
        if self.step_seconds < 1:
            raise ValueError("step_seconds must be positive")
        if self.step_seconds > self.session_seconds:
            raise ValueError("step_seconds cannot exceed session_seconds")
        if self.measurement_window_seconds < 1:
            raise ValueError("measurement_window_seconds must be positive")

    @property
    def anchors_per_session(self) -> int:
        return self.session_seconds // self.step_seconds


DEFAULT_ANCHOR_SPEC = AnchorSpec()

# Named compatibility constants for concise numerical code and downstream
# scripts. New APIs should accept an AnchorSpec where geometry is configurable.
SESSION_LEN = DEFAULT_ANCHOR_SPEC.session_seconds
ANCHOR_STEP = DEFAULT_ANCHOR_SPEC.step_seconds
RETURN_VWAP_WINDOW = DEFAULT_ANCHOR_SPEC.measurement_window_seconds
ANCHORS_PER_DAY = DEFAULT_ANCHOR_SPEC.anchors_per_session
