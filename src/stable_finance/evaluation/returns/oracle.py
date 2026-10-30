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

Four placements of the error are provided. Two are deterministic and hit the
target at every decision; the other two are random and hit it on average:

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

Two properties of a real forecast are not captured by one IC applied to every
cross-section independently, and both can be asked for:

Per-horizon IC
    A real forecast's IC decays with horizon -- a signal that ranks the next
    five minutes well ranks the next day barely at all -- and a study of how
    horizons are blended is meaningless if every horizon is equally good.
    ``ic`` may therefore be one value per horizon. Each horizon is degraded
    exactly as a scalar ``ic`` would degrade it, from the same random stream
    in the same order, so a constant array reproduces the scalar result bit
    for bit.
Persistent errors
    Degrading each decision independently makes the forecast's errors
    independent across decisions: at one decision a name is wrongly ranked
    high, at the next wrongly low. A portfolio built on such a forecast chases
    fresh noise every decision and turns over far more than one built on a
    real forecast, whose errors -- a stale view, a mis-specified factor --
    move slowly. ``persistence=k`` draws the random part of the degradation
    once per block of ``k`` consecutive decisions and re-applies it to every
    decision in the block: the same transpositions in rank space under
    ``random``, the same standardized scatter draw under ``tails`` and
    ``middle``. Each decision's IC is drawn from exactly the distribution it
    had before; what changes is that the same rank positions are wrong
    decision after decision while the truth underneath them moves. The
    errors persist in *rank* space, not name space: a name whose realized
    rank drifts by a few places lands on an unrelated forecast rank under
    ``random``, so the forecast itself becomes sticky only as far as the
    realized ranking holds still (see ``docs/oracle_forecast_math.md``).

Errors that stay with the NAME are a different model, and :func:`copula_forecast`
generates them. Its forecast is a Gaussian copula score, the normal score of
each name's realized rank plus that name's own noise, and the noise is an
AR(1) across decisions with a chosen half-life: a name that is wrongly ranked
high stays wrongly ranked high while its noise decays, whatever the realized
ranking does underneath it. That is the slowly-moving forecast whose
turnover a real signal shows. It shares neither the mechanism nor the
parameters of the rank swaps above -- a half-life rather than a block length,
an IC hit in expectation rather than steered -- so it is a function of its
own rather than a fourth ``noise``.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.evaluation._grid import _rankdata
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = ["NOISE_MODES", "copula_forecast", "oracle_forecast"]

NOISE_MODES = ("tails", "middle", "random")


def oracle_forecast(
    realized: ForwardReturns,
    ic: float | ArrayLike,
    *,
    noise: str = "random",
    scale: float | None = None,
    scatter: float | str | None = None,
    seed: int | np.random.Generator | None = None,
    persistence: int | None = None,
    days: ArrayLike | None = None,
) -> ForwardReturns:
    """Degrade the realized ranking to a forecast with rank IC ``ic``.

    By default every ``(decision, horizon)`` cross-section is degraded
    independently. Under ``tails`` and ``middle`` the per-decision rank IC
    that :func:`grouped_rank_ic` reports is ``ic`` at every decision; under
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

    **Per-horizon IC.** A real forecast's IC decays with horizon, so ``ic``
    may be a 1-D array with one target per horizon of ``realized``. Every
    element is validated as a scalar ``ic`` would be, and horizon ``h`` is
    degraded with ``ic[h]`` and, by default, ``scale=|ic[h]|``. Randomness is
    drawn from the same single stream, in the same decision-major,
    horizon-minor order, as for a scalar, so a constant array gives a result
    bitwise identical to the scalar it repeats.

    **Persistence.** Independent degradation makes the forecast's errors
    independent across decisions, so a portfolio built on it re-trades fresh
    noise every decision and turns over far more than one built on a real,
    slowly-moving forecast. With ``persistence=k`` the decisions are cut into
    blocks of ``k`` consecutive decisions -- ``decision // k``, or, when
    ``days`` is given, blocks that restart at every change of day label so
    that no block straddles a night -- and every random draw is taken from a
    generator private to the block and horizon,
    ``np.random.default_rng([root, horizon_index, block_index])``, recreated
    for every decision of the block. ``root`` is ``seed`` itself when it is a
    non-negative integer, and otherwise one integer drawn from
    ``np.random.default_rng(seed)``. Every decision in a block therefore
    replays the same draws:

    * under ``random`` the same rounds of transpositions are applied in rank
      space, so the map from a name's realized rank to its forecast rank is
      the same throughout the block. This holds exactly when the decisions
      have the same number of finite names and an untied outcome; otherwise
      the swap counts differ and the draws, though seeded alike, land
      differently;
    * under ``tails`` and ``middle`` the swap pattern is already a fixed
      function of rank position, so persistence matters only for
      ``scatter``: the block shares one standardized Gaussian draw, hence
      (for equal ``n``) one target IC. Without ``scatter`` it has no effect.

    What persistence does **not** change is the distribution of any single
    decision's IC: each decision is still a cross-section degraded by draws
    with the same law, so its IC has the same mean and spread as without
    persistence. What it does change is the dependence *across* decisions.
    The same rank positions are wrong decision after decision while the truth
    underneath them moves, so the error pattern by realized rank repeats
    within a block, and so -- since without ties the IC of a fixed rank
    permutation does not depend on the values it permutes -- does the IC
    itself: the decision-to-decision scatter no longer averages away inside
    a block. The forecast's own autocorrelation across decisions rises only
    in proportion to how still the realized ranking is. Under ``random`` the
    shared permutation is not smooth in rank, so a name whose realized rank
    moves by a few places draws an unrelated forecast rank: with realized
    returns whose ranking is fully frozen within a block, the forecast's
    rank autocorrelation goes from about ``ic**2`` to about ``1``, but with a
    rank autocorrelation of 0.99 in the realized returns it rises only from
    about 0.01 to 0.08 (200 names, ``ic=0.05``, ``persistence=10``).
    Persistence is therefore a model of persistent *errors in rank space*,
    not by itself of a slowly-moving forecast; :func:`copula_forecast` is
    the generator whose errors persist by name. Errors are independent
    across blocks and across horizons.
    ``persistence=None`` and ``persistence=1`` both reproduce the default,
    independent behaviour bit for bit, single shared stream included.

    Args:
        realized: Realized forward returns, the outcomes the forecast is
            scored against.
        ic: Target rank IC in ``[-1, 1]``, or a 1-D array of them with one
            per horizon. A target above what ties in the realized returns
            allow returns the perfect ranking. ``tails`` and ``middle`` reach
            ``-1`` by full reversal; ``random`` needs a positive target, since
            random swaps only ever decay a ranking towards an uninformative
            one.
        noise: Where the ranking errors go; one of :data:`NOISE_MODES`.
        scale: Shrinkage of the forecast towards the cross-sectional mean.
            Defaults to ``abs(ic)`` per horizon, so an ``ic`` of ``0`` at any
            horizon needs an explicit scale.
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
        persistence: Number of consecutive decisions that share one draw of
            the random degradation. ``None`` or ``1`` draws afresh for every
            decision.
        days: Optional day label per decision, one per decision of
            ``realized``. A change of label starts a new persistence block.
            Ignored when ``persistence`` is ``None``.
    """
    values = realized.values
    n_decisions, _, n_horizons = values.shape
    ics = _horizon_ics(ic, n_horizons)
    if not np.all((ics >= -1.0) & (ics <= 1.0)):
        raise ValueError("ic must lie in [-1, 1]")
    if noise not in NOISE_MODES:
        raise ValueError(f"noise must be one of {NOISE_MODES}, got {noise!r}")
    scales = _horizon_scales(ics, scale)
    if noise == "random" and np.any(ics <= 0.0):
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
    if persistence is not None and (
        isinstance(persistence, bool)
        or not isinstance(persistence, (int, np.integer))
        or persistence < 1
    ):
        raise ValueError(f"persistence must be a positive integer, got {persistence!r}")
    days = _day_labels(days, n_decisions)
    rng = np.random.default_rng(seed)

    # persistence=1 is the independent case and keeps its single stream, so
    # it is bitwise the default rather than merely equal in distribution.
    blocks = root = None
    if persistence is not None and persistence > 1:
        blocks = _persistence_blocks(n_decisions, int(persistence), days)
        root = _root_seed(seed, rng)
    draws_randomness = noise == "random" or scatter is not None

    forecast = np.full(values.shape, np.nan)
    for decision in range(n_decisions):
        for horizon in range(n_horizons):
            outcome = values[decision, :, horizon]
            names = np.flatnonzero(np.isfinite(outcome))
            if len(names) < 2:
                continue
            draw = rng
            if blocks is not None and draws_randomness:
                draw = np.random.default_rng([root, horizon, int(blocks[decision])])
            forecast[decision, names, horizon] = _cross_section(
                outcome[names], float(ics[horizon]), noise,
                float(scales[horizon]), scatter, draw
            )
    return ForwardReturns(
        forecast, realized.decisions, realized.assets, realized.horizons
    )


def copula_forecast(
    realized: ForwardReturns,
    ic: float | ArrayLike,
    *,
    half_life: float | None = None,
    days: ArrayLike | None = None,
    scale: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> ForwardReturns:
    """A forecast of rank IC ``ic`` whose errors belong to the names.

    Each name's forecast score is

        ``score = rho * z + sqrt(1 - rho**2) * e``

    where ``z`` is the normal score ``Phi^-1(rank / (n + 1))`` of the name's
    realized rank (ties at their average rank), ``e`` is the name's own
    standard-normal noise and ``rho = 2 sin(pi ic / 6)``, the Pearson
    correlation at which a bivariate normal has Spearman correlation ``ic``.
    The forecast is that Gaussian copula's ranking of the cross-section.

    **Why this is not a mode of** :func:`oracle_forecast`. That function
    degrades the realized ranking by swapping rank positions, so its errors
    live in rank space: even with ``persistence`` the thing that repeats is a
    permutation of rank positions, and a name whose realized rank moves by a
    few places lands on an unrelated forecast rank. Here the error belongs
    to the NAME, and ``half_life`` makes it persist: ``e`` is an AR(1) across
    decisions, so a name that is wrongly ranked high stays wrongly ranked
    high while its noise decays, whatever the realized ranking does
    underneath it. That is the slowly-moving forecast, with the low turnover
    a real signal shows, that rank swaps cannot produce. The two are
    different models with different parameters -- a half-life here, a block
    length there; an IC hit in expectation here, steered to the target
    there -- so they are different functions.

    **IC.** No decision is steered to the target: each decision's rank IC
    is ``ic`` in expectation and scatters around it by about
    ``1 / sqrt(n - 1)`` (more exactly ``(1 - ic**2) / sqrt(n - 1)``), with
    ``n`` the finite names in that cross-section, which is the sampling noise
    a real forecast of that IC shows on a cross-section of that size. The
    expectation falls short of ``ic`` by a relative amount of order ``1 / n``
    -- about 2% at 120 names, half a percent at 400 -- far inside that
    scatter. ``ic`` may be negative; ``|ic| = 1`` is excluded, since
    ``rho = +-1`` leaves no noise at all (``oracle_forecast(noise="tails")``
    gives the perfect and the fully reversed ranking).

    **Half-life.** With ``half_life=k`` the noise follows
    ``e_d = phi * e_{d-1} + sqrt(1 - phi**2) * eta_d`` with
    ``phi = 0.5 ** (1 / k)``: the correlation of a name's noise ``j``
    decisions apart is ``phi**j``, which halves every ``k`` decisions. The
    innovation is scaled so that ``e`` stays standard normal, so a single
    decision's IC has the same law whatever the half-life; what the
    half-life sets is how fast the forecast moves. On realized returns that
    are independent across decisions, the forecast's lag-one rank
    autocorrelation is about ``(1 - rho**2) * phi`` -- above 0.9 at
    ``half_life=12`` and small ``ic``, against about zero with
    ``half_life=None``, which draws every decision's noise afresh. When
    ``days`` is given the noise restarts from a fresh standard-normal draw at
    every change of day label, so no error is carried over a night.
    ``half_life=math.inf`` freezes each name's noise for the day.

    **Values.** The forecast takes its values from the realized
    cross-section: the realized returns, sorted, are handed out in the order
    of ``score``, so the name with the ``i``-th smallest score receives the
    ``i``-th smallest realized return. Ties in the realized returns
    therefore give tied forecast values, which :func:`grouped_rank_ic` ranks
    by their average rank just as it ranks the tied outcomes. Those values
    are then shrunk towards the cross-sectional mean by ``scale``, defaulting
    to ``|ic|`` per horizon, Grinold's ``alpha = IC * volatility * score``,
    for the reason :func:`oracle_forecast` gives. Names whose realized return
    is NaN are left NaN (their noise keeps evolving, so it is still theirs
    when they return), as is a cross-section whose realized returns are all
    tied.

    **Randomness.** Horizon ``h`` draws all of its noise from a generator of
    its own, ``np.random.default_rng([root, h])``, where ``root`` is ``seed``
    itself when it is a non-negative integer and otherwise one integer drawn
    from ``np.random.default_rng(seed)``. Errors are therefore independent
    across horizons, and a horizon's forecast does not depend on how many
    horizons ``realized`` has.

    Args:
        realized: Realized forward returns, the outcomes the forecast is
            scored against.
        ic: Target rank IC in ``(-1, 1)``, or a 1-D array of them with one
            per horizon.
        half_life: Half-life of each name's noise, in decisions; positive.
            ``None`` draws the noise independently at every decision.
        days: Optional day label per decision, one per decision of
            ``realized``. A change of label restarts every name's noise.
        scale: Shrinkage of the forecast towards the cross-sectional mean.
            Defaults to ``abs(ic)`` per horizon, so an ``ic`` of ``0`` at
            any horizon needs an explicit scale.
        seed: Seed or generator for the noise.
    """
    values = realized.values
    n_decisions, n_assets, n_horizons = values.shape
    ics = _horizon_ics(ic, n_horizons)
    if not np.all(np.abs(ics) < 1.0):
        raise ValueError(
            "ic must lie strictly inside (-1, 1): at |ic| = 1 the copula has no "
            "noise; use oracle_forecast(noise='tails') for a perfect ranking"
        )
    scales = _horizon_scales(ics, scale)
    if half_life is not None and (isinstance(half_life, bool) or not half_life > 0.0):
        raise ValueError(f"half_life must be positive, got {half_life!r}")
    days = _day_labels(days, n_decisions)
    root = _root_seed(seed, np.random.default_rng(seed))

    phi = 0.0 if half_life is None else 0.5 ** (1.0 / float(half_life))
    innovation = np.sqrt(1.0 - phi * phi)
    forecast = np.full(values.shape, np.nan)
    for horizon in range(n_horizons):
        draw = np.random.default_rng([root, horizon])
        rho = 2.0 * np.sin(np.pi * float(ics[horizon]) / 6.0)
        # The stationary start, then one innovation per decision; the first
        # decision's goes unused, since the start already is its noise.
        noise = draw.standard_normal(n_assets)
        for decision in range(n_decisions):
            fresh = draw.standard_normal(n_assets)
            if decision > 0:
                new_day = days is not None and days[decision] != days[decision - 1]
                noise = fresh if new_day else phi * noise + innovation * fresh
            outcome = values[decision, :, horizon]
            names = np.flatnonzero(np.isfinite(outcome))
            if len(names) < 2:
                continue
            finite = outcome[names]
            if finite.min() == finite.max():
                continue
            score = (rho * _normal_scores(finite)
                     + np.sqrt(1.0 - rho * rho) * noise[names])
            reordered = np.empty(len(names))
            reordered[np.argsort(score, kind="stable")] = np.sort(finite)
            centre = finite.mean()
            forecast[decision, names, horizon] = (
                centre + float(scales[horizon]) * (reordered - centre)
            )
    return ForwardReturns(forecast, realized.decisions, realized.assets, realized.horizons)


def _horizon_ics(ic: float | ArrayLike, n_horizons: int) -> NDArray:
    """``ic`` as one target per horizon, a scalar repeated."""
    if np.ndim(ic) == 0:
        return np.full(n_horizons, float(ic))
    ics = np.asarray(ic, dtype=np.float64)
    if ics.ndim != 1 or len(ics) != n_horizons:
        raise ValueError(
            f"ic must be a scalar or a 1-D array with one value per "
            f"horizon ({n_horizons}), got shape {ics.shape}"
        )
    return ics


def _horizon_scales(ics: NDArray, scale: float | None) -> NDArray:
    """The value shrinkage per horizon: ``scale``, or ``|ic|`` by default."""
    if scale is None:
        if np.any(ics == 0.0):
            raise ValueError("ic=0 has no natural scale; pass scale explicitly")
        return np.abs(ics)
    if not scale > 0.0:
        raise ValueError("scale must be positive")
    return np.full(len(ics), float(scale))


def _day_labels(days: ArrayLike | None, n_decisions: int) -> NDArray | None:
    """``days`` as a validated 1-D array with one label per decision."""
    if days is None:
        return None
    days = np.asarray(days)
    if days.ndim != 1 or len(days) != n_decisions:
        raise ValueError(
            f"days must be 1-D with one label per decision ({n_decisions}), "
            f"got shape {days.shape}"
        )
    return days


def _root_seed(seed: int | np.random.Generator | None, rng: np.random.Generator) -> int:
    """``seed`` when it is a non-negative integer, else one draw from ``rng``."""
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0:
        return int(seed)
    return int(rng.integers(2**63))


def _normal_scores(outcome: NDArray) -> NDArray:
    """``Phi^-1(rank / (n + 1))`` of each value, ties at their average rank."""
    ranks = _rankdata(outcome)
    unique, inverse = np.unique(ranks, return_inverse=True)
    normal = NormalDist()
    return np.array([normal.inv_cdf(r / (len(outcome) + 1)) for r in unique])[inverse]


def _persistence_blocks(n_decisions: int, persistence: int,
                        days: NDArray | None) -> NDArray[np.intp]:
    """Block index of every decision: runs of ``persistence``, cut at days.

    A block closes when it holds ``persistence`` decisions or when the day
    label changes, whichever comes first, so without ``days`` the index is
    ``decision // persistence`` and with them every day starts a fresh block
    (the last block of a day may be short). Indices count up across the
    whole panel, so no two blocks share a generator.
    """
    blocks = np.empty(n_decisions, dtype=np.intp)
    block, filled = -1, persistence
    for decision in range(n_decisions):
        new_day = (days is not None and decision > 0
                   and days[decision] != days[decision - 1])
        if filled == persistence or new_day:
            block, filled = block + 1, 0
        blocks[decision] = block
        filled += 1
    return blocks


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
