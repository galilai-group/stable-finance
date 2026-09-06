"""Model-independent configuration for extracting a view from a session."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewSpec:
    """Geometry of a single-session model view.

    These values are hyperparameters, not properties of the dataset. The
    current research recipe uses a 2,048-token view spanning 50–100% of a
    session, but callers must opt into that recipe explicitly through this
    object. ``aggregation_seconds`` takes precedence over ``scale_range`` when
    supplied, matching the established extended-hours behavior.
    """

    sequence_length: int = 2048
    scale_range: tuple[float, float] = (0.5, 1.0)
    aggregation_seconds: tuple[int, int] | None = None
    end_grid_seconds: int | None = None
    min_future_seconds: int = 0

    def __post_init__(self) -> None:
        low, high = self.scale_range
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if not 0 < low <= high <= 1:
            raise ValueError("scale_range must satisfy 0 < low <= high <= 1")
        if self.aggregation_seconds is not None:
            agg_low, agg_high = self.aggregation_seconds
            if not 1 <= agg_low <= agg_high:
                raise ValueError("aggregation_seconds must be positive and ordered")
        if self.end_grid_seconds is not None and self.end_grid_seconds < 1:
            raise ValueError("end_grid_seconds must be positive")
        if self.min_future_seconds < 0:
            raise ValueError("min_future_seconds cannot be negative")

