"""How far to trust a Sharpe ratio picked as the best of many configurations.

A harness sweep compares thousands of configurations -- signals, transforms,
horizons, cost assumptions -- and then reports the one that did best. The
best of thousands of Sharpe ratios is not an estimate of anything: it is the
maximum of a noisy sample, and A SWEEP LARGE ENOUGH WILL ALWAYS CONTAIN AN
IMPRESSIVE SHARPE RATIO EVEN WHEN EVERY CONFIGURATION IS PURE NOISE. The
statistics here put that selection back into the number.

``probabilistic_sharpe`` (Bailey & Lopez de Prado, 2012) is the probability
that the true Sharpe exceeds a benchmark, given the sample length and the
skewness and kurtosis of the returns, which widen the Sharpe's sampling
error well beyond the normal-theory value for fat-tailed P&L.
``min_track_record`` inverts it into the number of periods needed before a
Sharpe of this size is distinguishable from the benchmark.

``deflated_sharpe`` (Bailey & Lopez de Prado, 2014) is the same probability
with the benchmark raised to ``expected_max_sharpe``: the Sharpe the best of
``n_trials`` skill-less configurations would be expected to reach given how
dispersed the trials' Sharpes are. It answers "is the winner better than the
winner of a sweep of noise?" rather than "is the winner better than zero?".

``pbo`` (Bailey, Borwein, Lopez de Prado & Zhu, 2017) asks the selection
question directly, without distributional assumptions: across every split of
the sample into equal in-sample and out-of-sample halves, how often does the
in-sample winner land in the bottom half out of sample? A sweep whose
selection procedure has real information scores near 0; one that is fitting
noise scores near 0.5 or worse.

``sharpe_difference`` (Ledoit & Wolf, 2008) tests whether two paired return
series have equal Sharpe ratios, with a heteroskedasticity- and
autocorrelation-consistent standard error, so "configuration A beats B" can be
stated with a p-value that survives fat tails and serial dependence.

UNITS. Every Sharpe ratio in this module is PER PERIOD -- the mean return per
period over its standard deviation, NOT annualized. Benchmarks and trial
Sharpe variances must be given in the same per-period units; to convert an
annualized Sharpe ``S`` at ``P`` periods a year use ``S / sqrt(P)`` (and
divide a variance by ``P``). Everything here is deterministic: nothing draws
random numbers, and the PBO enumerates every combination rather than sampling.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EULER_MASCHERONI",
    "MAX_PBO_COMBINATIONS",
    "PBOResult",
    "SharpeDifference",
    "deflated_sharpe",
    "deflated_sharpe_from_trials",
    "expected_max_sharpe",
    "min_track_record",
    "pbo",
    "probabilistic_sharpe",
    "sharpe_difference",
]

#: The Euler--Mascheroni constant, which weights the two normal quantiles in
#: the extreme-value approximation to the expected maximum of N Gaussians.
EULER_MASCHERONI = 0.5772156649015329

#: The most in-sample/out-of-sample splits ``pbo`` will enumerate. C(16, 8) is
#: 12,870; C(20, 10) is 184,756, past which exhaustive CSCV stops being cheap
#: and a coarser ``n_blocks`` gives essentially the same estimate.
MAX_PBO_COMBINATIONS = 100_000

#: Combinations scored per vectorized batch inside ``pbo``, bounding memory
#: at roughly this many times ``n_trials`` floats per intermediate array.
_PBO_BATCH = 512

_NORMAL = NormalDist()


def _finite(returns: ArrayLike) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _moments(values: np.ndarray) -> tuple[float, float, float, int]:
    """Per-period Sharpe, skewness and (non-excess) kurtosis of finite values.

    The Sharpe uses the ``ddof=1`` standard deviation, matching
    :func:`~stable_finance.evaluation.scoring.sharpe.sharpe_ratio` at one
    period per year. Skewness and kurtosis are the population (moment)
    estimators, as in Bailey & Lopez de Prado. NaNs when undefined.
    """
    n = len(values)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    deviation = values - values.mean()
    m2 = float(np.mean(deviation ** 2))
    if m2 <= 0:
        return np.nan, np.nan, np.nan, n
    sharpe = float(values.mean() / values.std(ddof=1))
    skewness = float(np.mean(deviation ** 3)) / m2 ** 1.5
    kurtosis = float(np.mean(deviation ** 4)) / m2 ** 2
    return sharpe, skewness, kurtosis, n


def _sharpe_variance_factor(sharpe: float, skewness: float, kurtosis: float) -> float:
    """``1 - g3*SR + (g4 - 1)/4 * SR^2``, the non-normal Sharpe variance scale.

    For sample moments ``g4 >= 1 + g3^2``, so this is at least
    ``(1 - g3*SR/2)^2`` and never negative.
    """
    return 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2


def probabilistic_sharpe(returns: ArrayLike, *, benchmark: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds ``benchmark``.

    Bailey & Lopez de Prado (2012), "The Sharpe ratio efficient frontier",
    *Journal of Risk* 15(2)::

        PSR = Phi( (SR - SR*) * sqrt(n - 1)
                   / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    with ``SR`` the sample per-period Sharpe, ``g3`` the skewness, ``g4`` the
    kurtosis (3 for a normal, NOT excess kurtosis) and ``n`` the number of
    finite observations. ``benchmark`` (``SR*``) is per period too.

    NaN when fewer than two finite returns remain or they have no dispersion.
    """
    if not math.isfinite(benchmark):
        return np.nan
    sharpe, skewness, kurtosis, n = _moments(_finite(returns))
    if not math.isfinite(sharpe):
        return np.nan
    scale = _sharpe_variance_factor(sharpe, skewness, kurtosis)
    if scale <= 0:
        return np.nan
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(scale)
    return float(_NORMAL.cdf(z))


def min_track_record(returns: ArrayLike, *, benchmark: float = 0.0,
                     confidence: float = 0.95) -> float:
    """Periods needed for this Sharpe to beat ``benchmark`` at ``confidence``.

    Bailey & Lopez de Prado (2012), minimum track record length::

        MinTRL = 1 + (1 - g3*SR + (g4 - 1)/4 * SR^2)
                     * (Phi^-1(confidence) / (SR - SR*))^2

    in the same periods as ``returns``; compare it with the number of finite
    observations actually available. ``inf`` when ``SR <= SR*``, since no
    track record makes a Sharpe at or below the benchmark significant above
    it. NaN when the Sharpe is undefined.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if not math.isfinite(benchmark):
        return np.nan
    sharpe, skewness, kurtosis, _ = _moments(_finite(returns))
    if not math.isfinite(sharpe):
        return np.nan
    if sharpe <= benchmark:
        return float("inf")
    scale = _sharpe_variance_factor(sharpe, skewness, kurtosis)
    quantile = _NORMAL.inv_cdf(confidence)
    return float(1.0 + scale * (quantile / (sharpe - benchmark)) ** 2)


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """Expected best per-period Sharpe among ``n_trials`` skill-less trials.

    Bailey & Lopez de Prado (2014), via the extreme-value approximation to the
    maximum of N standard normals::

        E[max SR] ~ sqrt(V) * ( (1 - gamma) * Phi^-1(1 - 1/N)
                                + gamma * Phi^-1(1 - 1/(N e)) )

    with ``gamma`` the Euler--Mascheroni constant and ``V`` =
    ``trial_sharpe_var``, the variance across trials of their per-period
    Sharpe ratios. It grows with both N and V, and is 0 for a single trial
    (there is no selection) or for zero variance. NaN in, NaN out.
    """
    if isinstance(n_trials, bool) or int(n_trials) != n_trials or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not math.isfinite(trial_sharpe_var):
        return np.nan
    if trial_sharpe_var < 0:
        raise ValueError("trial_sharpe_var cannot be negative")
    n_trials = int(n_trials)
    if n_trials == 1 or trial_sharpe_var == 0:
        return 0.0
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(trial_sharpe_var)
                 * ((1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second))


def deflated_sharpe(returns: ArrayLike, *, n_trials: int,
                    trial_sharpe_var: float) -> float:
    """Probabilistic Sharpe against the best Sharpe a sweep of noise would show.

    Bailey & Lopez de Prado (2014), "The deflated Sharpe ratio", *Journal of
    Portfolio Management* 40(5): ``probabilistic_sharpe`` with ``benchmark =
    expected_max_sharpe(n_trials, trial_sharpe_var)``. ``n_trials`` is the
    number of configurations the reported one was selected from -- all of
    them, not only the ones that were written down -- and
    ``trial_sharpe_var`` the variance of their per-period Sharpes.
    """
    hurdle = expected_max_sharpe(n_trials, trial_sharpe_var)
    return probabilistic_sharpe(returns, benchmark=hurdle)


def deflated_sharpe_from_trials(returns_matrix: ArrayLike) -> NDArray[np.float64]:
    """Each trial's deflated Sharpe, deflated by the sweep it came from.

    ``returns_matrix`` is ``(n_periods, n_trials)`` per-period returns, one
    column per configuration. ``N`` is the number of columns and ``V`` the
    ``ddof=1`` variance of the columns' finite per-period Sharpes; each column
    is then scored with its own skewness, kurtosis and finite length (NaNs
    are dropped column by column). Columns with no defined Sharpe are NaN.
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be a 2-D (n_periods, n_trials) array")
    n_trials = matrix.shape[1]
    if n_trials < 1:
        raise ValueError("returns_matrix needs at least one trial column")
    sharpes = np.array([_moments(_finite(column))[0] for column in matrix.T])
    finite = sharpes[np.isfinite(sharpes)]
    variance = float(finite.var(ddof=1)) if len(finite) >= 2 else 0.0
    hurdle = expected_max_sharpe(n_trials, variance)
    return np.array([probabilistic_sharpe(column, benchmark=hurdle)
                     for column in matrix.T], dtype=np.float64)


@dataclass(frozen=True)
class PBOResult:
    """Probability of backtest overfitting and the CSCV splits behind it.

    ``logits`` holds one ``lambda = log(w / (1 - w))`` per split, ``w`` being
    the in-sample winner's relative out-of-sample rank; ``probability`` is the
    share of splits with ``lambda <= 0``. ``in_sample_sharpe`` and
    ``out_of_sample_sharpe`` are the selected trial's per-period Sharpe in
    each half, the pairs for a performance-degradation scatter.
    """

    probability: float
    logits: np.ndarray
    n_combinations: int
    in_sample_sharpe: np.ndarray
    out_of_sample_sharpe: np.ndarray
    n_blocks: int
    block_length: int

    @property
    def probability_of_loss(self) -> float:
        """Share of splits where the in-sample winner loses money out of sample."""
        return float(np.mean(self.out_of_sample_sharpe < 0))


def _average_rank_of_first(values: np.ndarray, column: np.ndarray) -> np.ndarray:
    """Ascending rank (1..N, ties averaged) of ``values[i, column[i]]`` per row."""
    chosen = values[np.arange(len(values)), column][:, None]
    below = np.sum(values < chosen, axis=1)
    ties = np.sum(values == chosen, axis=1)
    return below + (ties + 1) / 2.0


def _block_sharpe(sums: np.ndarray, squares: np.ndarray, length: int) -> np.ndarray:
    """Per-period ``ddof=1`` Sharpe from summed values and squares; NaN if flat."""
    mean = sums / length
    variance = (squares - length * mean ** 2) / (length - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = mean / np.sqrt(variance)
    return np.where(variance > 0, sharpe, np.nan)


def pbo(performance: ArrayLike, *, n_blocks: int = 16) -> PBOResult:
    """Probability of backtest overfitting by combinatorially symmetric CV.

    Bailey, Borwein, Lopez de Prado & Zhu (2017), "The probability of backtest
    overfitting", *Journal of Computational Finance* 20(4). ``performance``
    is ``(n_periods, n_trials)`` per-period returns, one column per
    configuration. Rows containing any NaN are dropped first, so every trial
    is judged on the same periods. The remaining rows are cut into
    ``n_blocks`` contiguous equal blocks; the ``n_periods % n_blocks``
    leftover rows at the END are discarded.

    Every choice of ``n_blocks / 2`` blocks is taken as in-sample, in
    ``itertools.combinations`` order, with the complement out of sample. In
    each split the trial with the highest in-sample per-period Sharpe is
    selected (the first on ties), its out-of-sample Sharpe is ranked among all
    trials' (ascending, ties averaged, a NaN Sharpe ranking as worst), and
    ``w = rank / (n_trials + 1)`` gives ``lambda = log(w / (1 - w))``. PBO is
    the fraction of splits with ``lambda <= 0``: the in-sample winner no
    better than the out-of-sample median.
    """
    matrix = np.asarray(performance, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("performance must be a 2-D (n_periods, n_trials) array")
    if isinstance(n_blocks, bool) or int(n_blocks) != n_blocks:
        raise ValueError("n_blocks must be an integer")
    n_blocks = int(n_blocks)
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer of at least 2")
    n_combinations = math.comb(n_blocks, n_blocks // 2)
    if n_combinations > MAX_PBO_COMBINATIONS:
        raise ValueError(
            f"n_blocks={n_blocks} gives {n_combinations} splits, more than "
            f"{MAX_PBO_COMBINATIONS}; use fewer blocks"
        )
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    n_periods, n_trials = matrix.shape
    if n_trials < 2:
        raise ValueError("pbo needs at least two trials to select among")
    block_length = n_periods // n_blocks
    half_length = block_length * (n_blocks // 2)
    if half_length < 2:
        raise ValueError(
            f"{n_periods} complete periods are too few for {n_blocks} blocks"
        )
    matrix = matrix[: block_length * n_blocks]
    # Centering each trial keeps the sum-of-squares variance numerically
    # stable; the shift is undone exactly when each half's Sharpe is formed.
    shift = matrix.mean(axis=0)
    centered = (matrix - shift).reshape(n_blocks, block_length, n_trials)
    block_sums = centered.sum(axis=1)
    block_squares = (centered ** 2).sum(axis=1)
    total_sums, total_squares = block_sums.sum(axis=0), block_squares.sum(axis=0)

    combos = np.array(list(itertools.combinations(range(n_blocks), n_blocks // 2)),
                      dtype=np.intp)
    logits = np.empty(n_combinations)
    in_sample = np.empty(n_combinations)
    out_of_sample = np.empty(n_combinations)
    for start in range(0, n_combinations, _PBO_BATCH):
        chosen = combos[start:start + _PBO_BATCH]
        sums = block_sums[chosen].sum(axis=1)
        squares = block_squares[chosen].sum(axis=1)

        def sharpe(s: np.ndarray, q: np.ndarray) -> np.ndarray:
            # Undo the centering: sum(x + c) and sum((x + c)^2) over the half.
            raw_sums = s + half_length * shift
            raw_squares = q + 2 * shift * s + half_length * shift ** 2
            return _block_sharpe(raw_sums, raw_squares, half_length)

        is_sharpe = sharpe(sums, squares)
        oos_sharpe = sharpe(total_sums - sums, total_squares - squares)
        winner = np.argmax(np.where(np.isnan(is_sharpe), -np.inf, is_sharpe), axis=1)
        rank = _average_rank_of_first(
            np.where(np.isnan(oos_sharpe), -np.inf, oos_sharpe), winner
        )
        omega = rank / (n_trials + 1)
        rows = np.arange(len(chosen))
        stop = start + len(chosen)
        logits[start:stop] = np.log(omega / (1.0 - omega))
        in_sample[start:stop] = is_sharpe[rows, winner]
        out_of_sample[start:stop] = oos_sharpe[rows, winner]

    return PBOResult(
        probability=float(np.mean(logits <= 0)),
        logits=logits,
        n_combinations=n_combinations,
        in_sample_sharpe=in_sample,
        out_of_sample_sharpe=out_of_sample,
        n_blocks=n_blocks,
        block_length=block_length,
    )


@dataclass(frozen=True)
class SharpeDifference:
    """Ledoit--Wolf test of equal Sharpe ratios for two paired series.

    ``difference`` is ``SR(a) - SR(b)`` per period, from population
    (``ddof=0``) moments as in the test's delta-method derivation;
    ``p_value`` is two-sided against equal Sharpe ratios.
    """

    difference: float
    standard_error: float
    z: float
    p_value: float
    observations: int
    lags: int


def _bartlett_long_run_variance(series: np.ndarray, lags: int) -> float:
    """Newey--West (Bartlett-kernel) long-run variance of a mean-zero series."""
    n = len(series)
    variance = float(np.dot(series, series)) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(np.dot(series[lag:], series[:-lag])) / n
    return variance


def sharpe_difference(a: ArrayLike, b: ArrayLike, *,
                      lags: int | None = None) -> SharpeDifference:
    """Test whether two paired return series have equal Sharpe ratios.

    Ledoit & Wolf (2008), "Robust performance hypothesis testing with the
    Sharpe ratio", *Journal of Empirical Finance* 15(5), HAC inference
    (their Section 3.1; NOT their bootstrap). With ``v = (mu_a, mu_b,
    E[a^2], E[b^2])`` and ``f(v) = mu_a / sqrt(E[a^2] - mu_a^2) - mu_b /
    sqrt(E[b^2] - mu_b^2)``, the delta method gives ``se = sqrt(grad' Psi
    grad / T)``, where ``Psi`` is the long-run covariance of ``y_t = (a_t -
    mu_a, b_t - mu_b, a_t^2 - E[a^2], b_t^2 - E[b^2])``. It is computed here
    as the long-run variance of the scalar ``grad' y_t``, which is identical.

    Ledoit & Wolf use a prewhitened quadratic-spectral kernel; this uses the
    Bartlett (Newey--West) kernel with ``lags`` lags, defaulting to the
    Newey--West (1994) rule ``floor(4 * (T / 100) ** (2 / 9))``. ``lags=0``
    is the iid case. Periods where either series is non-finite are dropped
    as a pair.
    """
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError("a and b must be paired series of the same length")
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    n = len(left)
    if n < 3:
        raise ValueError("need at least three paired finite periods")
    if lags is None:
        lags = math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))
    elif isinstance(lags, bool) or int(lags) != lags or not 0 <= lags < n:
        raise ValueError("lags must be an integer in [0, number of periods)")
    lags = int(lags)

    mean_a, mean_b = float(left.mean()), float(right.mean())
    second_a, second_b = float(np.mean(left ** 2)), float(np.mean(right ** 2))
    var_a, var_b = second_a - mean_a ** 2, second_b - mean_b ** 2
    if var_a <= 0 or var_b <= 0:
        raise ValueError("both series need nonzero dispersion")
    difference = mean_a / math.sqrt(var_a) - mean_b / math.sqrt(var_b)
    gradient = (
        second_a / var_a ** 1.5,
        -second_b / var_b ** 1.5,
        -mean_a / (2.0 * var_a ** 1.5),
        mean_b / (2.0 * var_b ** 1.5),
    )
    projected = (gradient[0] * (left - mean_a) + gradient[1] * (right - mean_b)
                 + gradient[2] * (left ** 2 - second_a)
                 + gradient[3] * (right ** 2 - second_b))
    long_run = max(_bartlett_long_run_variance(projected, lags), 0.0)
    standard_error = math.sqrt(long_run / n)
    if standard_error > 0:
        z = difference / standard_error
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    elif difference == 0:
        z, p_value = 0.0, 1.0
    else:
        z, p_value = math.copysign(math.inf, difference), 0.0
    return SharpeDifference(
        difference=float(difference),
        standard_error=float(standard_error),
        z=float(z),
        p_value=float(p_value),
        observations=n,
        lags=lags,
    )
