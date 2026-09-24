"""Every decision between a forward-return forecast and a Sharpe ratio.

A Sharpe ratio is not a property of a forecast. It is a property of a forecast
plus a trading harness, and the harness contributes decisions the model being
evaluated has no opinion about: which names are tradable, how much of the
cross-section to hold, how risk is forecast, how weights follow from it, how
often to rebalance, what a trade costs, and how a periodic return is
annualized. This module makes that harness an explicit, enumerable object.

There is a DEFAULT so that the common case is one call and the library has an
opinion. There is a config SPACE so that the sensitivity of a result to the
harness can be measured rather than assumed, which is the only way to tell a
finding about a model from a finding about a backtest.

    from stable_finance import DEFAULT_BACKTEST, enumerate_configs

    report = run_backtest(forecast, realized, half_spread=spread)   # default
    space = enumerate_configs()                                     # all of it
    space = enumerate_configs(efq=0.70)                             # a slice
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields, replace

import numpy as np

from stable_finance.costs import COST_MODELS
from stable_finance.risk import RISK_MODELS
from stable_finance.selection import SELECTION_RULES
from stable_finance.universe import UNIVERSE_SCREENS
from stable_finance.weighting import WEIGHTING_RULES

__all__ = [
    "ANNUALIZATIONS",
    "EFQ_GRID",
    "BacktestConfig",
    "CONFIG_SPACE",
    "DEFAULT_BACKTEST",
    "REBALANCE_RULES",
    "enumerate_configs",
]

#: The execution qualities swept, as EFQ (effective / quoted half-spread).
#: 0 is a midpoint fill and 1 pays the whole quoted spread; 0.55 and 0.88 are
#: Levy (2022)'s measured direct-market-access and NYSE figures, and 0.70 sits
#: between them. EFQ REPLACES the old ``cost_model`` sweep as the cost axis,
#: because it is one continuous measured quantity rather than five named
#: models, and because a result reported against it can be read by anyone who
#: knows their own execution.
EFQ_GRID = (0.0, 0.25, 0.50, 0.70, 0.88, 1.00)

#: How a periodic return series is annualized. ``decisions`` treats every
#: decision as an independent period, which for an intraday anchor grid
#: annualizes a signal that is in the market for part of the session as though
#: it were in the market all year; ``daily`` first sums each day's decisions
#: into one daily return and annualizes at 252. Neither is wrong, they answer
#: different questions, and the gap between them is large enough that leaving
#: it implicit would be the mistake.
ANNUALIZATIONS = ("decisions", "daily")

#: When the book is allowed to change. ``every_decision`` re-solves at every
#: anchor; ``daily`` sets the book at each day's first decision and holds it,
#: which cuts turnover by roughly the number of anchors in a day and is the
#: difference between a signal that pays for its spread and one that does not.
REBALANCE_RULES = ("every_decision", "daily")

#: ``mean_variance`` is the cost-aware, path-dependent allocator; the rest are
#: the path-independent rules that size a selected book directly.
ALLOCATORS = ("mean_variance", *WEIGHTING_RULES)


@dataclass(frozen=True)
class BacktestConfig:
    """One point in the harness configuration space.

    Defaults reproduce the library's standard pipeline: hold the whole
    cross-section long-short, forecast risk with a shrunk sample covariance,
    size with the cost-aware mean-variance allocator, rebalance every
    decision, and charge the observable quoted half-spread. The frictionless
    ``mid`` result is always reported alongside, so the default never hides
    what the spread cost.
    """

    universe: str = "all"
    selection: str = "all"
    weighting: str = "mean_variance"
    risk_model: str = "shrunk_sample"
    cost_model: str = "cross"
    #: Execution quality: the multiple of the quoted half-spread actually
    #: paid. Charged BOTH to the cost-aware allocator and to the scoring, so
    #: a book is always optimised against the cost it is then billed for.
    efq: float = 1.0
    rebalance: str = "every_decision"
    annualization: str = "daily"
    gross_exposure: float = 2.0
    shrinkage: float = 0.3
    #: Iteration budget for the cost-aware allocator's proximal solver. A
    #: NUMERICAL setting, not a modelling one: it is not swept, and a value too
    #: low changes the answer rather than approximating it. 400 reproduces the
    #: 1000-iteration solution to floating point on panels of a few hundred
    #: names while costing a fifth of the time.
    max_iter: int = 400

    def __post_init__(self) -> None:
        for field, allowed in (
            ("universe", UNIVERSE_SCREENS), ("selection", SELECTION_RULES),
            ("weighting", ALLOCATORS), ("risk_model", RISK_MODELS),
            ("cost_model", COST_MODELS), ("rebalance", REBALANCE_RULES),
            ("annualization", ANNUALIZATIONS),
        ):
            value = getattr(self, field)
            if value not in allowed:
                raise ValueError(
                    f"{field}={value!r} is not one of {allowed}"
                )
        # Type-checked before the numeric test: EFQ is the one axis whose
        # options are numbers, so a caller passing a name here (as they would
        # for every other axis) must get the same ValueError and not a
        # TypeError out of numpy.
        if (isinstance(self.efq, bool)
                or not isinstance(self.efq, (int, float))
                or not np.isfinite(self.efq) or self.efq < 0):
            raise ValueError("efq must be a finite, non-negative number")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be positive")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")

    @property
    def label(self) -> str:
        """A stable, filename-safe identifier for this configuration."""
        return "|".join(
            f"{field.name}={getattr(self, field.name)}" for field in fields(self)
        )

    def replace(self, **changes) -> "BacktestConfig":
        """A copy with some axes changed, validated on construction."""
        return replace(self, **changes)


#: The library's opinion when the caller does not have one.
DEFAULT_BACKTEST = BacktestConfig()

#: The axes swept by :func:`enumerate_configs`, and their option lists. Scalar
#: settings (``gross_exposure``, ``shrinkage``, ``max_iter``) are deliberately
#: absent: the first two are continuous, so a study should choose its own grid
#: rather than inherit one, and the third is numerical, not a modelling choice.
#: ``annualization`` is deliberately ABSENT. It is pure accounting -- the same
#: trades folded into different periods -- and holding it at ``daily`` makes
#: every Sharpe here one non-overlapping observation per day annualized at
#: 252, which is what the execution-quality figure reports. ``cost_model`` is
#: absent because ``efq`` is now the cost axis.
CONFIG_SPACE = {
    "universe": UNIVERSE_SCREENS,
    "selection": SELECTION_RULES,
    "weighting": ALLOCATORS,
    "risk_model": RISK_MODELS,
    "efq": EFQ_GRID,
    "rebalance": REBALANCE_RULES,
}


def _is_degenerate(config: BacktestConfig) -> bool:
    """Drop configurations that are duplicates of another point in the space.

    Two collapses are real rather than cosmetic. A book whose weights ignore
    the covariance entirely is the same book under every risk model, so only
    one risk model is kept for it. And under a diagonal risk model,
    inverse-volatility and equal-risk-contribution both reduce to ``w ~
    1/sigma``, so keeping all three would triple-count one portfolio and
    silently weight the summary toward it.
    """
    if config.weighting in ("equal", "rank") and config.risk_model != "diagonal":
        return True
    if config.risk_model == "diagonal" and config.weighting == "erc":
        return True
    return False


def enumerate_configs(**overrides) -> tuple[BacktestConfig, ...]:
    """The configuration space, or a named slice of it.

    Each keyword replaces one axis's option list; a scalar is accepted and
    treated as a one-element list, so ``enumerate_configs(cost_model="mid")``
    sweeps everything else at zero friction. Configurations that duplicate
    another point are removed, so every element is a distinct portfolio.
    """
    axes = dict(CONFIG_SPACE)
    scalars = {}
    for name, value in overrides.items():
        if name in axes:
            # A scalar is a one-element list, so a caller can pin an axis with
            # ``efq=0.7`` or ``selection="all"`` as readily as passing a tuple.
            axes[name] = ((value,) if isinstance(value, (str, float, int))
                          else tuple(value))
        elif name in {f.name for f in fields(BacktestConfig)}:
            scalars[name] = value
        else:
            raise ValueError(f"{name!r} is not a configuration axis")

    names = list(axes)
    out = []
    for combination in itertools.product(*(axes[name] for name in names)):
        config = BacktestConfig(**dict(zip(names, combination)), **scalars)
        if not _is_degenerate(config):
            out.append(config)
    return tuple(out)
