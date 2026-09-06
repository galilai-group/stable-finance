"""Calendar-month identifiers used by fitting and evaluation APIs."""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Month:
    """A validated calendar month with inclusive date bounds."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if self.year < 1 or not 1 <= self.month <= 12:
            raise ValueError(f"invalid calendar month: {self.year}-{self.month}")

    @classmethod
    def parse(cls, value: "Month | str") -> "Month":
        if isinstance(value, cls):
            return value
        try:
            year, month = (int(part) for part in str(value)[:7].split("-"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"month must be YYYY-MM, got {value!r}") from error
        return cls(year, month)

    @property
    def first_day(self) -> dt.date:
        return dt.date(self.year, self.month, 1)

    @property
    def last_day(self) -> dt.date:
        return dt.date(
            self.year, self.month, calendar.monthrange(self.year, self.month)[1]
        )

    def offset(self, count: int) -> "Month":
        ordinal = self.year * 12 + self.month - 1 + count
        year, zero_based_month = divmod(ordinal, 12)
        return Month(year, zero_based_month + 1)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def next_month(value: Month | str) -> str:
    """Return the calendar month after ``value`` as ``YYYY-MM``."""
    return str(Month.parse(value).offset(1))

