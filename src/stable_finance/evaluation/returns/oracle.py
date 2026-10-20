"""Forecasts with a known information coefficient, built from the outcomes.

Testing a portfolio or execution stage against a learned forecast confounds two
unknowns: how good the forecast was and how well the stage used it. Sorting the
realized returns gives a forecast whose rank IC is exactly 1; degrading that
ranking by a chosen amount gives a forecast whose IC is whatever the test
wants, with the errors placed where the test wants them. Everything downstream
of it is then measured against a forecast that is known exactly.

The degradation is a sequence of swaps of forecast ranks. Spearman correlation
is Pearson correlation on ranks, so with forecast ranks ``f`` and realized
average ranks ``r`` it moves only through the centred sum
``C = sum((f - mean f) (r - mean r))``, and exchanging the forecast ranks of
names ``a`` and ``b`` lowers ``C`` by exactly ``(f_a - f_b) * (r_a - r_b)``.
Each swap is therefore costed in constant time, ties in the realized returns
included, rather than through ``1 - 6 sum(d**2) / (n (n**2 - 1))``, which
holds only without ties.

Three placements of the error are provided. Two are deterministic and hit the
target at every decision; the third is random and hits it on average:

``tails``
    Exchange the most extreme names first -- best with worst, then the
    second-best with the second-worst -- so the middle of the ranking stays
    correct and the tails are wrong. At a low IC the extremes are *inverted*,
    not scrambled: a book that holds the tails loses.
``middle``
    Exchange the names on either side of the median first, moving outwards, so
    the tails stay correct and the middle is wrong. A book that holds the tails
    sees an IC-1 forecast.
``random``
    Random transpositions over the whole cross-section, so errors are spread
    evenly over the ranking: the generic noisy forecast. Summed over all
    ``n (n - 1)`` ordered pairs, the cost of a swap is ``2 n C``, so one random
    transposition multiplies the expected ``C`` by ``1 - 2 / (n - 1)`` whatever
    the current ranking, and ``k`` of them leave an expected IC of
    ``(1 - 2 / (n - 1)) ** k``: about ``(n - 1) / 2 * ln(1 / ic)`` swaps reach
    ``ic``, with ``n`` the finite names in that cross-section. The swap count
    is fixed from that formula and no decision is steered to the target, so
    each decision's IC
    scatters around ``ic`` with standard deviation near ``1 / sqrt(n - 1)``,
    which is precisely the sampling noise a real forecast of that IC shows on
    a cross-section of that size. Swaps are applied in rounds of disjoint
    pairs, a round of ``m`` multiplying the expected ``C`` by exactly
    ``1 - 2 m / (n - 1)``, and the last round's size is Bernoulli-rounded so
    the expectation is exact rather than off by part of a swap.

``tails`` and ``middle`` can be given the same per-decision scatter with
``scatter="sampling"``: each decision's target is drawn from a Gaussian with
that ``1 / sqrt(n - 1)`` standard deviation and then hit exactly. The draw is
of the IC rather than of a swap count because the count is far too coarse a
variable: the outermost tail pair moves the IC by about ``12 / n``, and the
middle pair that brings it down to ``ic`` by about
``3 * 16^(1/3) (1 - ic)^(2/3) / n``, so a spread of ``1 / sqrt(n - 1)`` in the
IC is a spread of only a few pairs in the count whatever ``n`` is, and with
ties the count-to-IC map is not even fixed.

Mirror pairs cannot hit a target on their own, since the outermost exchange
alone moves the IC by about ``12 / n``. ``tails`` and ``middle`` therefore apply
pairs in their priority order until the next would overshoot and then spend the
remaining budget on whichever other pairs fit, which leaves the placement
intact and lands within about 1e-6 of the target -- or 1e-4 when the realized
returns are heavily tied, because exchanging two tied names moves nothing and
the budget becomes coarser.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stable_finance.evaluation.returns.contract import ForwardReturns
from stable_finance.evaluation._grid import _rankdata

__all__ = ["NOISE_MODES", "oracle_forecast"]

NOISE_MODES = ("tails", "middle", "random")


def oracle_forecast(
    realized: ForwardReturns,
    ic: float,
    *,
    noise: str = "random",
    scale: float | None = None,
    scatter: float | str | None = None,
    seed: int | np.random.Generator | None = None,
) -> ForwardReturns:
    """Degrade the realized ranking to a forecast with rank IC ``ic``.

    Every ``(decision, horizon)`` cross-section is degraded independently.
    Under ``tails`` and ``middle`` the per-decision rank IC that
    :func:`grouped_rank_ic` reports is ``ic`` at every decision; under
    ``random`` it is ``ic`` in expectation and scatters around it by about
    ``1 / sqrt(n - 1)`` per decision, as a real forecast of that IC would. A
    cross-section whose realized returns are all tied has no ranking to degrade
    and is left NaN, as are names whose realized return is NaN.

    The forecast takes its values from the realized cross-section, reassigned
    by the degraded ranking, so it is in return units and its ranking is the
    only thing that was changed. Those values are then shrunk towards the
    cross-sectional mean by ``scale``. An unshrunk oracle has the dispersion of
    the realized returns themselves, which no forecast with an IC of 0.02
    would claim, and a cost-aware allocator or an edge-to-cost ratio would
    read that as an edge many times too large. The default ``scale=|ic|`` is
    Grinold's ``alpha = IC * volatility * score``: the size of forecast a
    forecaster who believed its own IC would produce. Pass ``scale=1.0`` for
    the raw oracle magnitude. Any positive scale leaves the IC unchanged.

    Args:
        realized: Realized forward returns, the outcomes the forecast is
            scored against.
        ic: Target rank IC in ``[-1, 1]``. A target above what ties in the
            realized returns allow returns the perfect ranking. ``tails`` and
            ``middle`` reach ``-1`` by full reversal; ``random`` needs a
            positive target, since random swaps only ever decay a ranking
            towards an uninformative one.
        noise: Where the ranking errors go; one of :data:`NOISE_MODES`.
        scale: Shrinkage of the forecast towards the cross-sectional mean.
            Defaults to ``abs(ic)``, so ``ic=0`` needs an explicit scale.
        scatter: Per-decision spread of the IC under ``tails`` and
            ``middle``. ``None`` hits ``ic`` at every decision. ``"sampling"``
            draws each decision's target from a Gaussian with standard
            deviation ``1 / sqrt(n - 1)``, which is the distribution ``random``
            produces, so the three modes then differ only in where the errors
            sit. A float is that standard deviation directly. Draws are
            clipped to the attainable range, which matters only when ``ic``
            is within a few standard deviations of ``1``. ``random`` accepts
            ``None`` or ``"sampling"``, both meaning its inherent scatter.
        seed: Seed or generator for ``noise="random"`` and for ``scatter``;
            unused otherwise.
    """
    if not -1.0 <= ic <= 1.0:
        raise ValueError("ic must lie in [-1, 1]")
    if noise not in NOISE_MODES:
        raise ValueError(f"noise must be one of {NOISE_MODES}, got {noise!r}")
    if scale is None:
        if ic == 0.0:
            raise ValueError("ic=0 has no natural scale; pass scale explicitly")
        scale = abs(ic)
    if not scale > 0.0:
        raise ValueError("scale must be positive")
    if noise == "random" and ic <= 0.0:
        raise ValueError(
            "noise='random' decays the ranking towards an uninformative one "
            "and needs a positive ic; use 'tails' or 'middle' for ic <= 0"
        )
    if isinstance(scatter, str):
        if scatter != "sampling":
            raise ValueError(f"scatter must be None, 'sampling' or a float, got {scatter!r}")
    elif scatter is not None:
        if noise == "random":
            raise ValueError(
                "noise='random' has its own sampling scatter; a numeric "
                "scatter applies only to 'tails' and 'middle'"
            )
        if not scatter >= 0.0:
            raise ValueError("scatter must be non-negative")
    rng = np.random.default_rng(seed)

    values = realized.values
    forecast = np.full(values.shape, np.nan)
    for decision in range(values.shape[0]):
        for horizon in range(values.shape[2]):
            outcome = values[decision, :, horizon]
            names = np.flatnonzero(np.isfinite(outcome))
            if len(names) < 2:
                continue
            forecast[decision, names, horizon] = _cross_section(
                outcome[names], ic, noise, scale, scatter, rng
            )
    return ForwardReturns(
        forecast, realized.decisions, realized.assets, realized.horizons
    )


def _cross_section(outcome: NDArray, ic: float, noise: str, scale: float,
                   scatter: float | str | None,
                   rng: np.random.Generator) -> NDArray:
    """Forecast values for one cross-section of finite realized returns."""
    n = len(outcome)
    order = np.argsort(outcome, kind="stable")
    ordered = outcome[order]
    if ordered[0] == ordered[-1]:
        return np.full(n, np.nan)
    # Names are worked on in realized order: position i holds the i-th
    # smallest outcome, r its average rank, f its forecast rank.
    r = _rankdata(ordered)
    f = np.arange(n, dtype=np.float64)
    centred_f = f - f.mean()
    centred_r = r - r.mean()
    attainable = float(centred_f @ centred_r)
    norm = float(np.sqrt((centred_f @ centred_f) * (centred_r @ centred_r)))
    target = ic
    if noise != "random" and scatter is not None:
        # The IC of a random permutation has variance 1 / (n - 1) and is
        # Gaussian for all practical n, so this is what ``random`` produces.
        sd = 1.0 / np.sqrt(n - 1) if scatter == "sampling" else scatter
        bound = attainable / norm
        target = float(np.clip(ic + sd * rng.standard_normal(), -bound, bound))
    # How far C = sum(centred f * centred r) must fall from the perfect
    # ranking, as an amount and as a fraction of itself.
    need = attainable - target * norm
    if need > 0.0:
        if noise == "random":
            _random_swaps(f, ic * norm / attainable, rng)
        else:
            _mirror_pairs(f, r, need, middle_first=noise == "middle")

    # Strictly increasing values so the forecast carries no ties of its own:
    # a tie would average two forecast ranks and move the IC off target.
    spread = ordered[-1] - ordered[0]
    levels = ordered + np.arange(n) * (spread * 1e-9 / n)
    centre = outcome.mean()
    values = np.empty(n)
    values[order] = centre + scale * (levels[f.astype(np.intp)] - centre)
    return values


def _mirror_pairs(f: NDArray, r: NDArray, need: float, *,
                  middle_first: bool) -> None:
    """Exchange mirror-image positions, tails or middle first, in place."""
    n = len(f)
    low = np.arange(n // 2)
    high = n - 1 - low
    if middle_first:
        low, high = low[::-1], high[::-1]
    # Pairs are disjoint and start from the perfect ranking, so their costs
    # are fixed in advance and add.
    costs = (high - low) * (r[high] - r[low])
    chosen = _budget_pairs(costs, need)
    f[low[chosen]], f[high[chosen]] = f[high[chosen]], f[low[chosen]]


def _budget_pairs(costs: NDArray, need: float) -> NDArray[np.bool_]:
    """Take pairs in priority order, then fit the remainder of the budget.

    The prefix that fits is always taken. The remainder is then spent one of
    two ways, whichever lands closer: by filling it with later pairs that
    still fit (the natural move when later pairs are smaller, as in
    ``tails``), or by taking the overshooting pair and handing back earlier
    pairs, latest first (the natural move when they are larger, as in
    ``middle``).
    """
    cumulative = np.cumsum(costs)
    taken = int(np.searchsorted(cumulative, need, side="right"))
    chosen = np.zeros(len(costs), dtype=bool)
    chosen[:taken] = True
    if taken == len(costs):
        return chosen

    filled = chosen.copy()
    short = need - (cumulative[taken - 1] if taken else 0.0)
    for index in range(taken + 1, len(costs)):
        if costs[index] <= short:
            filled[index] = True
            short -= costs[index]

    returned = chosen.copy()
    returned[taken] = True
    over = cumulative[taken] - need
    for index in range(taken - 1, -1, -1):
        if costs[index] <= over:
            returned[index] = False
            over -= costs[index]
    return filled if short <= over else returned


def _random_swaps(f: NDArray, factor: float, rng: np.random.Generator) -> None:
    """Random transpositions leaving ``E[C] = factor * C``, in place.

    A random pair's expected cost is ``2 C / (n - 1)`` whatever the current
    ranking, so a round of ``m`` disjoint random pairs -- whose costs are
    evaluated on the same ranking and add -- multiplies ``E[C]`` by exactly
    ``1 - 2 m / (n - 1)``. Rounds of a quarter of the names halve it; the
    last round takes whatever fraction remains, its size Bernoulli-rounded so
    that the expectation is exact. No cost is ever evaluated.
    """
    n = len(f)
    per_pair = 2.0 / (n - 1)
    quarter = max(n // 4, 1)
    sizes = []
    while factor < 1.0 - quarter * per_pair:
        sizes.append(quarter)
        factor /= 1.0 - quarter * per_pair
    last = (1.0 - factor) / per_pair
    last = int(last) + int(rng.random() < last - int(last))
    sizes.append(last)
    for m in sizes:
        if m == 0:
            continue
        positions = rng.permutation(n)[: 2 * m]
        a, b = positions[:m], positions[m:]
        f[a], f[b] = f[b], f[a]
