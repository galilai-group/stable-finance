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

The copula's errors are still independent ACROSS names and its quality is
the same at every decision and for every name, and a real forecast has
neither property: its errors are bets on factors -- it is wrong about all
high-beta names at once -- its IC drifts through good and bad stretches, and
it is more accurate in some names than in others. :func:`factor_forecast`
adds the three to the copula, each opt-in by a parameter: errors that load
on the principal components of past returns, a per-decision IC that follows
an AR(1), and accuracy that rises with a name's spread or volatility. With
all three switched off it is the copula, draw for draw.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stable_finance.dataset.market import MarketData, require_market_aligned
from stable_finance.evaluation._grid import _rankdata
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = [
    "NOISE_MODES",
    "QUALITY_VARIABLES",
    "copula_forecast",
    "factor_forecast",
    "oracle_forecast",
]

NOISE_MODES = ("tails", "middle", "random")

#: What a name's accuracy can be made to rise with in :func:`factor_forecast`.
QUALITY_VARIABLES = ("spread", "vol")

#: The largest per-name, per-decision rank IC :func:`factor_forecast` will
#: use. A time-varying IC or a quality weight can push a target past
#: ``+-1``, where the copula has no noise left; clipping there keeps a sliver.
_IC_CAP = 0.999


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


def factor_forecast(
    realized: ForwardReturns,
    ic: float | ArrayLike,
    *,
    history: ForwardReturns | None = None,
    n_factors: int = 3,
    factor_share: float = 0.3,
    ic_persistence: float | None = None,
    ic_dispersion: float = 0.0,
    quality: str | None = None,
    quality_strength: float = 1.0,
    market: MarketData | None = None,
    half_life: float | None = None,
    days: ArrayLike | None = None,
    scale: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> ForwardReturns:
    """A copula forecast whose errors are factor bets and whose IC drifts.

    :func:`copula_forecast` scores each name as
    ``rho * z + sqrt(1 - rho**2) * e`` with ``z`` the normal score of its
    realized rank and ``e`` noise of its own. Its errors are independent
    across names, its IC is the same at every decision, and every name is
    forecast equally well. None of that is how a real forecast is wrong, and
    each of the three biases a study in a known direction:

    * independent errors diversify away across a few hundred names, so
      breadth looks larger than it is, and neutralizing a factor or using a
      better risk model can only remove true signal -- the measured value of
      both is biased towards zero or below. A real forecast's errors are
      bets: it is wrong about all high-beta names, or all of one industry,
      at once, and that is precisely what neutralization and a risk model
      exist to take out;
    * a fixed IC makes every decision's P&L a draw from the same law, so the
      only strategy risk left is sampling noise. A real forecast has good
      and bad stretches, and ``IR = mean(IC) / sd(IC)`` (Qian and Hua, 2004;
      Ding and Martin, 2017) is set by them;
    * equal accuracy everywhere misplaces the alpha. Short-horizon
      predictability concentrates in wide-spread, volatile names, which is
      where a cost-blind construction pays most to trade.

    This generator adds the three, each switched off by default except the
    factor errors. The score of name ``i`` at decision ``t`` is

        ``s = c_it * z + sqrt(1 - c_it**2) * e``,
        ``e = g_t + sqrt(1 - share_i) * u``,

    and the forecast is the ranking of ``s``, valued as in
    :func:`copula_forecast`; ``u`` is each name's own noise, the copula's
    ``e``.

    **Factor errors.** ``g_t`` is the factor part of the error: each name's
    loadings ``B_i`` on ``n_factors`` factors, dotted with a standard-normal
    factor shock ``f_t`` and scaled so that the name's factor part has
    variance ``share_i`` over time, ``sqrt(share_i) * B_i . f_t / |B_i|``.
    ``B`` comes from a principal-component analysis of the NORMAL SCORES of
    past returns: each cross-section of the first horizon of ``history`` (or
    of ``realized`` when no history is given) is replaced by the normal
    scores of its ranks, each name is centred over time, and ``B`` is the
    first ``n_factors`` right singular vectors scaled by their singular
    values. Normal scores rather than returns because the forecast lives in
    normal-score space, because they are already centred across names -- a
    shift common to every name changes no ranking, so the market's level is
    not a factor here -- and because one outlier or split cannot capture a
    component. A name's factor share is proportional to its squared loading,
    capped at one, and scaled so that the shares average ``factor_share``
    over the names: a name that moves with the factors is wrong with them,
    and one the factors barely touch keeps a mostly idiosyncratic error. Each
    name's error therefore has unit variance, and two names' errors have
    correlation ``sqrt(share_i * share_j)`` times the cosine of their
    loadings. At every decision ``g_t`` is then centred across the names of
    the cross-section and rescaled to their average share, so that the
    factors carry exactly that share of the error's cross-sectional variance
    at every decision (up to the sample variance of ``u``, which is the
    copula's). Unscaled, that variance would swing with the size of
    ``f_t``, and the IC, a concave function of it, would move off the
    copula's. At ``factor_share=1`` every name with a loading has a share of
    one, and a name without history has no loading and an idiosyncratic
    error whatever the share. With no ``history`` the loadings are
    in-sample, and the realized cross-sections correlate with loadings
    estimated on themselves a little more than with out-of-sample ones, so
    the IC's spread widens slightly even on pure noise.

    **What factor errors do to the IC.** They widen its per-decision spread.
    When the realized returns load on the same factors -- they do: on August
    2020 the 5-minute cross-section correlates with the first three loading
    vectors by 0.26, 0.17 and 0.13 in absolute value on average -- an error
    that bets on a factor is right or wrong about many names at once. The
    IC then moves with ``rho_t``, the cross-sectional correlation of the
    error with the realized normal scores, whose variance is about
    ``factor_share * E[r_t**2] + (1 - factor_share) / (n - 1)``, with
    ``r_t`` the correlation of ``g_t`` with the realized cross-section, and
    the IC's standard deviation is about ``sqrt(1 - c**2)`` times its root.
    On August 2020 with three factors ``E[r_t**2]`` is 0.068 at
    ``factor_share=0.3``, which predicts 0.146 and measures 0.151, against
    the copula's 0.039; real forecasts show 0.12 (ridge, LASSO) to 0.15
    (15-minute reversal) on the same month. The same nonlinearity that gives
    the copula its ``O(1 / n)`` shortfall then lowers the mean: the expected
    cross-sectional correlation is about ``c (1 - Var(rho_t))``, so the mean
    IC falls short of the copula's by a relative amount of about
    ``factor_share * (E[r_t**2] - 1 / (n - 1))``: 2% at
    ``factor_share=0.3`` and 4% at 0.6 on August 2020, 2.5% and 2% measured
    over four seeds, each within its 2% seed error. That is inside one
    month's standard error of the mean IC, which the wider spread has
    itself grown to about 0.0037. It is left in, not calibrated away: it is
    a property of a forecast whose errors are bets, and every scored row
    reports the IC it actually had. On realized returns with no factor
    structure the factor errors barely change the IC's law: the spread is
    the copula's, and the mean is within about a percent of the copula's at
    ``factor_share=0.3`` and a few percent at ``factor_share=1``, where the
    error is an ``n_factors``-dimensional pattern whose spread across names
    is far from Gaussian. See ``docs/oracle_forecast_math.md`` (section 8).

    **Time-varying IC.** With ``ic_dispersion > 0`` the target is no longer
    ``ic`` at every decision but ``kappa_t = ic + ic_dispersion * a_t``, with
    ``a_t`` a stationary standard-normal AR(1) of lag-one autocorrelation
    ``ic_persistence`` (``None`` draws it afresh at every decision). The
    expected per-decision IC is therefore itself an AR(1) with mean ``ic``,
    standard deviation ``ic_dispersion`` and autocorrelation
    ``ic_persistence ** j`` at lag ``j``; the IC a scorer measures adds the
    sampling scatter (and the factor term above) to it, so its
    autocovariance at lag ``j >= 1`` is ``ic_dispersion**2 *
    ic_persistence**j`` and its variance ``ic_dispersion**2`` plus the
    scatter. ``kappa_t`` goes into the copula's ``rho = 2 sin(pi kappa /
    6)`` per decision, clipped to ``+-0.999``, which matters only when a
    draw comes within reach of ``+-1``. ``ic_persistence=1`` freezes
    ``a_t``: one IC per day when ``days`` is given, one for the whole panel
    otherwise.

    **Quality.** With ``quality="spread"`` or ``"vol"`` each name's IC is
    ``kappa_t * q_it``, with ``q`` rising linearly in the name's rank by the
    variable among the cross-section's ``m`` ranked names,
    ``q = 1 + quality_strength * (2 (rank - 1/2) / m - 1)``. At the default
    strength the narrowest-spread (or calmest) name carries almost no
    information and the widest (or most volatile) about twice the average.
    It is the name's IC that varies, not the size of its noise alone: every
    score keeps unit variance, so the tails of the forecast are not filled
    with the noisiest names, and a book that holds the tails finds the
    accuracy where the quality variable says it is. The weights are then
    rescaled at each decision so that the cross-section's rank IC stays
    ``kappa_t``. To first order in ``rho`` the rank IC is proportional to
    ``sum(rho_i z_i (Phi(z_i) - 1/2))``, so the rescaling keeps
    ``sum(q_i z_i (Phi(z_i) - 1/2))`` equal to ``sum(z_i (Phi(z_i) - 1/2))``.
    A plain mean of one is not enough when the variable tracks ``|z|``, as
    volatility does -- volatile names are the ones that land in the tails
    of a cross-section -- and with log-normal volatilities across names the
    unrescaled weights raised the IC by about a quarter. Where the variable
    is unrelated to ``|z|`` the rescaling is close to one, and the average
    weight is one. ``"spread"`` reads each decision's half-spread from
    ``market``, so the ranking moves with the quotes; names without a quote
    get ``q = 1`` before the rescaling. ``"vol"`` ranks names by the
    standard deviation of the same first-horizon returns the loadings come
    from, once for the panel. On heteroskedastic returns even the copula
    forecasts volatile names' RANKS better, since their realized ranks sit
    in the tails where a given ``rho`` places them most reliably; ``"vol"``
    adds to that.

    **Persistence.** ``half_life`` makes ``u`` the copula's AR(1) by name,
    and the factor shock ``f_t`` an AR(1) with the same half-life, so a
    persistent forecast is persistently wrong about the same factor; the
    normalized ``g_t`` follows ``f_t``'s direction, whose lag correlation is
    a little below ``f_t``'s own (with one factor, ``(2 / pi) arcsin`` of
    it). Given ``days``, the noise, the factor shock and the IC process all
    restart from their stationary laws at each change of day label.

    **Randomness.** Horizon ``h`` draws ``u`` from
    ``np.random.default_rng([root, h])`` exactly as :func:`copula_forecast`
    does, the factor shocks from ``default_rng([root, h, 1])`` and the IC
    process from ``default_rng([root, h, 2])``, with ``root`` as there.
    Errors and IC paths are therefore independent across horizons, a
    horizon's forecast does not depend on how many horizons there are, and
    with ``factor_share=0``, ``ic_dispersion=0`` and ``quality=None`` the
    forecast is bitwise :func:`copula_forecast`'s, whatever ``n_factors`` or
    ``ic_persistence`` say and whichever (aligned) ``history`` is given.

    Args:
        realized: Realized forward returns, the outcomes the forecast is
            scored against.
        ic: Target rank IC in ``(-1, 1)``, or a 1-D array of them with one
            per horizon; the mean of the per-decision target.
        history: Past returns on ``realized``'s assets, in their order, whose
            first horizon gives the factor loadings and, for
            ``quality="vol"``, the volatilities. ``None`` uses ``realized``.
        n_factors: Number of principal components the errors load on.
        factor_share: Share of the error variance carried by the factors,
            averaged over the names, in ``[0, 1]``. ``0`` gives the copula's
            independent errors.
        ic_persistence: Lag-one autocorrelation of the IC process, in
            ``[0, 1]``. ``None`` draws it independently at every decision.
        ic_dispersion: Standard deviation of the per-decision target IC
            around ``ic``; non-negative. ``0`` fixes it at ``ic``.
        quality: ``None``, or what a name's accuracy rises with; one of
            :data:`QUALITY_VARIABLES`.
        quality_strength: How steeply accuracy rises with the quality rank,
            in ``[0, 1]``: ``0`` is uniform, ``1`` runs from about zero to
            about twice the average.
        market: The observable market on ``realized``'s grid; needed for
            ``quality="spread"`` (its half-spread, or the bid and ask it is
            derived from), and otherwise unused.
        half_life: Half-life of each name's noise and of the factor shocks,
            in decisions; positive. ``None`` draws them afresh at every
            decision.
        days: Optional day label per decision, one per decision of
            ``realized``. A change of label restarts the noise, the factor
            shocks and the IC process.
        scale: Shrinkage of the forecast towards the cross-sectional mean.
            Defaults to ``abs(ic)`` per horizon, so an ``ic`` of ``0`` at
            any horizon needs an explicit scale.
        seed: Seed or generator for every draw.
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
    if (isinstance(n_factors, bool) or not isinstance(n_factors, (int, np.integer))
            or not 1 <= n_factors < n_assets):
        raise ValueError(
            f"n_factors must be a positive integer below the number of assets "
            f"({n_assets}), got {n_factors!r}")
    if not 0.0 <= factor_share <= 1.0:
        raise ValueError(f"factor_share must lie in [0, 1], got {factor_share!r}")
    if ic_persistence is not None and not 0.0 <= ic_persistence <= 1.0:
        raise ValueError(f"ic_persistence must lie in [0, 1], got {ic_persistence!r}")
    if not 0.0 <= ic_dispersion < np.inf:
        raise ValueError(f"ic_dispersion must be non-negative and finite, got {ic_dispersion!r}")
    if quality is not None and quality not in QUALITY_VARIABLES:
        raise ValueError(f"quality must be None or one of {QUALITY_VARIABLES}, got {quality!r}")
    if not 0.0 <= quality_strength <= 1.0:
        raise ValueError(f"quality_strength must lie in [0, 1], got {quality_strength!r}")
    if half_life is not None and (isinstance(half_life, bool) or not half_life > 0.0):
        raise ValueError(f"half_life must be positive, got {half_life!r}")
    days = _day_labels(days, n_decisions)
    past = _first_horizon(history, realized)
    spread = None
    if quality == "spread":
        if market is None:
            raise ValueError(
                "quality='spread' needs market, with the half-spread or the bid and "
                "ask it is derived from")
        require_market_aligned(market, realized)
        market.require("half_spread", purpose="factor_forecast(quality='spread')")
        spread = market.get("half_spread")
    root = _root_seed(seed, np.random.default_rng(seed))

    loadings = directions = shares = own_weight = None
    if factor_share > 0.0:
        loadings = _factor_loadings(past, int(n_factors))
        shares = _factor_shares(loadings, factor_share)
        size = np.linalg.norm(loadings, axis=1, keepdims=True)
        directions = np.divide(np.sqrt(shares)[:, None] * loadings, size,
                               out=np.zeros_like(loadings), where=size > 0)
        own_weight = np.sqrt(1.0 - shares)
    vol = _volatility(past) if quality == "vol" else None
    phi = 0.0 if half_life is None else 0.5 ** (1.0 / float(half_life))
    innovation = np.sqrt(1.0 - phi * phi)
    ic_phi = 0.0 if ic_persistence is None else float(ic_persistence)
    ic_innovation = np.sqrt(1.0 - ic_phi * ic_phi)
    forecast = np.full(values.shape, np.nan)
    for horizon in range(n_horizons):
        # The copula's stream for the names' own noise, consumed exactly as
        # copula_forecast consumes it; the factor shocks and the IC process
        # have streams of their own, so switching either on leaves u alone.
        draw = np.random.default_rng([root, horizon])
        factor_draw = np.random.default_rng([root, horizon, 1])
        ic_draw = np.random.default_rng([root, horizon, 2])
        noise = draw.standard_normal(n_assets)
        shocks = factor_draw.standard_normal(n_factors) if loadings is not None else None
        level = float(ic_draw.standard_normal()) if ic_dispersion > 0.0 else 0.0
        for decision in range(n_decisions):
            fresh = draw.standard_normal(n_assets)
            fresh_shocks = (factor_draw.standard_normal(n_factors)
                            if loadings is not None else None)
            fresh_level = float(ic_draw.standard_normal()) if ic_dispersion > 0.0 else 0.0
            if decision > 0:
                new_day = days is not None and days[decision] != days[decision - 1]
                if new_day:
                    noise, shocks, level = fresh, fresh_shocks, fresh_level
                else:
                    noise = phi * noise + innovation * fresh
                    if shocks is not None:
                        shocks = phi * shocks + innovation * fresh_shocks
                    level = ic_phi * level + ic_innovation * fresh_level
            outcome = values[decision, :, horizon]
            names = np.flatnonzero(np.isfinite(outcome))
            if len(names) < 2:
                continue
            finite = outcome[names]
            if finite.min() == finite.max():
                continue
            error = noise[names]
            if directions is not None:
                # Each name's factor part has its share as variance over time;
                # across the names of this cross-section it is centred and
                # scaled to their average share, so that the share is exact
                # at every decision rather than swinging with |f_t|.
                common = directions[names] @ shocks
                common = common - common.mean()
                spread_of_common = common.std()
                if spread_of_common > 0.0:
                    error = (common * (np.sqrt(shares[names].mean()) / spread_of_common)
                             + own_weight[names] * error)
            z = _normal_scores(finite)
            target = float(ics[horizon])
            if ic_dispersion > 0.0:
                target = float(np.clip(target + ic_dispersion * level, -_IC_CAP, _IC_CAP))
            if quality is not None:
                variable = (spread[decision, names] if spread is not None else vol[names])
                weights = _quality_weights(variable, quality_strength)
                # To first order in rho the rank IC is proportional to
                # sum(rho_i z_i (Phi(z_i) - 1/2)): weights that track |z| --
                # volatile names land in the tails -- would raise the IC along
                # with the accuracy unless rescaled to keep that sum.
                # z is Phi^-1(rank / (n + 1)), so Phi(z) is that quantile.
                leverage = z * (_rankdata(finite) / (len(finite) + 1) - 0.5)
                weights *= leverage.sum() / (weights @ leverage)
                target = np.clip(target * weights, -_IC_CAP, _IC_CAP)
            rho = 2.0 * np.sin(np.pi * target / 6.0)
            score = rho * z + np.sqrt(1.0 - rho * rho) * error
            reordered = np.empty(len(names))
            reordered[np.argsort(score, kind="stable")] = np.sort(finite)
            centre = finite.mean()
            forecast[decision, names, horizon] = (
                centre + float(scales[horizon]) * (reordered - centre)
            )
    return ForwardReturns(forecast, realized.decisions, realized.assets, realized.horizons)


def _first_horizon(history: ForwardReturns | None, realized: ForwardReturns) -> NDArray:
    """The returns loadings and volatilities come from: ``(observation, asset)``."""
    source = realized if history is None else history
    if not isinstance(source, ForwardReturns):
        raise TypeError("history must be ForwardReturns on the realized returns' assets")
    if not np.array_equal(source.assets, realized.assets):
        raise ValueError("assets of the history and the realized returns are not aligned")
    return source.values[:, :, 0]


def _factor_loadings(returns: NDArray, n_factors: int) -> NDArray:
    """Each name's loadings on the first principal components, mean square one.

    The components are those of the normal scores of each cross-section,
    centred by name over time; the loadings are the right singular vectors
    scaled by their singular values, so a dominant component weighs more,
    and then scaled together so that a name's squared loading averages one
    over the names that have any. Each component's sign is fixed so that its
    largest loading is positive: a singular vector's sign is arbitrary, and
    the forecast for a given seed should not depend on which one the
    linear-algebra library returns.
    """
    scores = np.zeros(returns.shape)
    for row, outcome in enumerate(returns):
        names = np.flatnonzero(np.isfinite(outcome))
        if len(names) < 2 or outcome[names].min() == outcome[names].max():
            continue
        scores[row, names] = _normal_scores(outcome[names])
    scores -= scores.mean(axis=0)
    if min(scores.shape) <= n_factors or not np.any(scores):
        raise ValueError(
            f"the history has too few informative cross-sections for {n_factors} factors")
    _, singular, components = np.linalg.svd(scores, full_matrices=False)
    loadings = components[:n_factors].T * singular[:n_factors]
    largest = np.argmax(np.abs(loadings), axis=0)
    loadings *= np.sign(loadings[largest, np.arange(n_factors)])
    size = (loadings**2).sum(axis=1)
    return loadings / np.sqrt(size[size > 0].mean())


def _factor_shares(loadings: NDArray, factor_share: float) -> NDArray:
    """Each name's factor share: its squared loading scaled, capped at one.

    The scale is the one that makes the shares average ``factor_share`` over
    all names -- water-filling: the names whose scaled squared loading would
    pass one are capped there, and the rest are scaled up to make up the
    average. A share above what the names with any loading can carry gives
    every such name a share of one.
    """
    size = (loadings**2).sum(axis=1)
    target = factor_share * len(size)
    if target >= np.count_nonzero(size):
        return (size > 0).astype(np.float64)
    descending = np.sort(size)[::-1]
    # rest[c] is the total squared loading left when the c largest are capped.
    rest = np.r_[np.cumsum(descending[::-1])[::-1], 0.0]
    for capped in range(len(size)):
        scale = (target - capped) / rest[capped]
        if scale * descending[capped] <= 1.0:
            return np.minimum(scale * size, 1.0)
    return (size > 0).astype(np.float64)


def _volatility(returns: NDArray) -> NDArray:
    """Each name's standard deviation over time; NaN with fewer than two values."""
    finite = np.isfinite(returns)
    count = finite.sum(axis=0)
    filled = np.where(finite, returns, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = filled.sum(axis=0) / count
        squares = (np.where(finite, returns - mean, 0.0) ** 2).sum(axis=0)
        return np.where(count >= 2, np.sqrt(squares / (count - 1)), np.nan)


def _quality_weights(variable: NDArray, strength: float) -> NDArray:
    """Weights rising linearly in rank by ``variable``, averaging exactly one.

    Names whose variable is missing get weight one, so the average over the
    whole cross-section is one as well.
    """
    weights = np.ones(len(variable))
    known = np.isfinite(variable)
    m = int(known.sum())
    if m >= 2:
        ranks = _rankdata(variable[known])
        weights[known] = 1.0 + strength * (2.0 * (ranks - 0.5) / m - 1.0)
    return weights


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
