"""A trading policy learned for its net-of-cost performance, not hand-written.

Every other policy in this stage is forecast-then-optimize: a forecast is
estimated to minimise prediction error, a risk model is estimated, and an
optimizer turns the two into trades under a cost the forecast never saw.
Each step is sensible on its own and the whole is not, because the forecast
was fitted to a loss that does not price turnover. A signal that is accurate
but fast-decaying wins the prediction contest and loses money after costs;
a slower, noisier one can be the better thing to trade. Jensen, Kelly,
Malamud and Pedersen ("Machine learning and the implementable efficient
frontier", Review of Financial Studies, forthcoming) make the point
empirically: choosing the portfolio rule DIRECTLY for its net-of-cost
utility dominates estimating returns first and optimizing second, and the
gap is largest exactly where costs are, among the small and illiquid names.

:class:`ParametricPolicy` is the smallest model that takes that seriously.
It follows Brandt, Santa-Clara and Valkanov (2009, "Parametric portfolio
policies: exploiting characteristics in the cross-section of equity
returns", RFS 22(9)): rather than modelling the joint distribution of
returns, write the portfolio weight as a function of observable
characteristics with a handful of coefficients, and choose the coefficients
to maximise the investor's realised utility on the training sample. With
characteristics ``X_t`` of shape ``(n_assets, k)`` -- forecasts at several
horizons, say, or a forecast beside its own cost -- the score is

    s_t = demean(X_t @ theta),

demeaned over the names with every characteristic observed so the book is
dollar neutral, and the TARGET is that score scaled to the requested gross:

    target_t = gross_exposure * s_t / sum_i |s_t,i|.

``theta`` therefore learns how to BLEND the characteristics, and because the
blend is chosen against net returns rather than prediction error, a
characteristic that predicts well but turns over fast is down-weighted in
favour of a persistent one when trading is costly. Only theta's direction
matters -- the target is unchanged by rescaling it -- and the reported
``theta_`` is read as relative loadings.

Brandt, Santa-Clara and Valkanov trade to the target every period. DeMiguel,
Martin-Utrera, Nogales and Uppal (2020, "A transaction-cost perspective on
the multitude of firm characteristics", RFS 33(5)) show that once costs are
charged the parametric policy must also learn HOW FAST to move: they add a
partial-adjustment rule, and it is what turns characteristics that are
individually useless net of cost into a profitable combination, since
trading toward a blended target nets opposing trades before they are paid
for. The held weights here follow

    w_t = w_{t-1} + eta * (target_t - w_{t-1}),        w_{-1} = initial,

which is the trading rule Garleanu and Pedersen (2013, "Dynamic trading
with predictable returns and transaction costs", JF 68(6)) derive as
optimal under quadratic costs: trade a constant fraction ``eta`` of the way
toward an aim portfolio each period. They derive ``eta`` from the cost and
risk aversion; here it is LEARNED, as ``eta = sigmoid(rho)`` so it stays in
``(0, 1)``, so the rule adapts to whatever the actual cost schedule is. A
name whose cost is NaN or infinite cannot be traded at that decision and
holds its previous weight, as in :func:`~.rebalance.rebalance`.

The training objective is the investor's mean-variance utility of the NET
return per period,

    net_t = w_t . R_t - sum_i c_t,i * |w_t,i - w_{t-1},i|
    J     = mean(net_t) - (risk_aversion / 2) var(net_t)
            - (l2 / 2) ||theta - theta_0||^2,

with ``|x|`` smoothed to ``sqrt(x^2 + epsilon^2) - epsilon`` so the
objective is differentiable at a zero trade. ``R_t`` is the return EARNED
BY HOLDING ``w_t`` from decision ``t`` to decision ``t + 1``: it must be the
one-period, NON-OVERLAPPING holding-period return on the same clock the
decisions are made on. Passing a forward return over a horizon longer than
the decision spacing credits each period with returns the next periods also
claim, and the fitted policy will trade far too little because it believes
every position earns several periods' return. ``c_t,i`` is the cost per unit
of weight traded, such as the half-spread times an effective-fill quantity.
The ``l2`` penalty shrinks theta toward the equal blend ``theta_0 = 1/k``,
the forecast-combination benchmark that is famously hard to beat out of
sample (Smith and Wallis, 2009, "A simple explanation of the forecast
combination puzzle", OBES 71(3)).

The objective is optimised by full-batch gradient ascent with Adam moment
estimates (Kingma and Ba, 2015), DETERMINISTICALLY: theta starts at the
equal blend, ``rho`` at zero (``eta = 1/2``), the learning rate is fixed and
so is the number of iterations, with no random initialisation, no
mini-batches and no early stopping. The same data therefore gives the same
``theta_`` bit for bit, which is what lets a backtest built on it be
re-run and audited. The gradient is exact, by backpropagation through the
weight recursion in a reverse pass over the decisions, not a numerical
approximation -- which also means the policy has no hidden dependence on a
finite-difference step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from stable_finance.dataset.market import MarketData, market_quantity, requires
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns

__all__ = ["ParametricPolicy", "to_weights"]

_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8
# A score whose cross-sectional spread is this small relative to its level is
# rounding noise left by demeaning equal values, not a ranking worth trading.
_FLAT_SCORE = 1e-12


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    z = np.exp(x)
    return z / (1.0 + z)


def _characteristics(
    X: ArrayLike, *, standardize: bool
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Characteristics with unobserved names zeroed, and which names are active.

    A name is ACTIVE at a decision when every characteristic is finite there;
    only active names receive a score, and the rest are aimed at zero. With
    ``standardize`` each characteristic is z-scored per decision over the
    active names, so a characteristic in basis points and one in ranks enter
    the blend on the same footing and theta reads as relative loadings.
    """
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            f"characteristics must have shape (n_decisions, n_assets, k), got {values.shape}")
    if values.shape[2] < 1:
        raise ValueError("characteristics need at least one column")
    active = np.isfinite(values).all(axis=2)
    filled = np.where(active[:, :, None], values, 0.0)
    if standardize:
        counts = active.sum(axis=1)[:, None]
        safe = np.maximum(counts, 1)
        mean = filled.sum(axis=1) / safe
        centred = np.where(active[:, :, None], filled - mean[:, None, :], 0.0)
        std = np.sqrt((centred ** 2).sum(axis=1) / safe)
        with np.errstate(invalid="ignore", divide="ignore"):
            filled = np.where(std[:, None, :] > 0, centred / std[:, None, :], 0.0)
    return filled, active


def _returns(returns: ArrayLike, shape: tuple[int, int]) -> NDArray[np.float64]:
    array = np.asarray(returns, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"returns must have shape {shape}, got {array.shape}")
    # A return that was not observed contributes nothing; a caller who cannot
    # observe a name's return should also mark it untradable through its cost.
    return np.where(np.isfinite(array), array, 0.0)


def _costs(
    cost: ArrayLike | None, shape: tuple[int, int]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Finite per-unit costs, zero where untradable, and the tradable mask."""
    if cost is None:
        return np.zeros(shape), np.ones(shape, dtype=bool)
    array = np.asarray(cost, dtype=np.float64)
    try:
        array = np.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError(
            f"cost must broadcast to {shape}, got shape {array.shape}") from error
    if np.any(np.isfinite(array) & (array < 0)):
        raise ValueError("cost cannot be negative")
    tradable = np.isfinite(array)
    return np.where(tradable, array, 0.0), tradable


def _initial(initial: ArrayLike | None, n_assets: int) -> NDArray[np.float64]:
    if initial is None:
        return np.zeros(n_assets)
    array = np.asarray(initial, dtype=np.float64)
    try:
        array = np.broadcast_to(array, (n_assets,)).astype(np.float64)
    except ValueError as error:
        raise ValueError(
            f"initial must broadcast to ({n_assets},), got shape {array.shape}") from error
    if not np.isfinite(array).all():
        raise ValueError("initial weights must be finite")
    return array


def _targets(
    X: NDArray[np.float64], active: NDArray[np.bool_], theta: NDArray[np.float64],
    gross: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Dollar-neutral targets at ``gross``; also the score and its gross for backprop."""
    raw = X @ theta
    counts = np.maximum(active.sum(axis=1), 1)
    mean = raw.sum(axis=1) / counts
    score = np.where(active, raw - mean[:, None], 0.0)
    size = np.abs(score).sum(axis=1)
    level = np.abs(raw).sum(axis=1)
    flat = size <= _FLAT_SCORE * np.maximum(level, 1.0)
    size = np.where(flat, 0.0, size)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(flat, 0.0, gross / np.where(flat, 1.0, size))
    return score * scale[:, None], score, size


def _hold(
    targets: NDArray[np.float64], tradable: NDArray[np.bool_], eta: float,
    initial: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The partial-adjustment path, with the starting portfolio as row 0."""
    path = np.empty((targets.shape[0] + 1, targets.shape[1]))
    path[0] = initial
    for t in range(targets.shape[0]):
        path[t + 1] = path[t] + eta * np.where(tradable[t], targets[t] - path[t], 0.0)
    return path


def _objective_and_gradient(
    theta: NDArray[np.float64],
    rho: float,
    X: NDArray[np.float64],
    active: NDArray[np.bool_],
    returns: NDArray[np.float64],
    cost: NDArray[np.float64],
    tradable: NDArray[np.bool_],
    initial: NDArray[np.float64],
    *,
    gross: float,
    risk_aversion: float,
    l2: float,
    epsilon: float,
) -> tuple[float, NDArray[np.float64], float]:
    """Training utility and its exact gradient in ``theta`` and ``rho``.

    The reverse pass carries ``G_t``, the total derivative of the utility in
    the held weights ``w_t``. It collects three terms: ``w_t`` earns ``R_t``
    and pays for the trade into it at ``t``; it sets the size of the trade
    out of it at ``t + 1``; and it is carried into ``w_{t+1}`` with weight
    ``1 - eta`` on every tradable name. From ``G_t`` the target receives
    ``eta * G_t`` on the tradable names, and that flows back through the
    gross scaling and the demeaning to theta.
    """
    n_decisions = X.shape[0]
    theta_0 = np.full(theta.shape, 1.0 / theta.size)
    eta = _sigmoid(rho)
    targets, score, size = _targets(X, active, theta, gross)
    path = _hold(targets, tradable, eta, initial)
    held, previous = path[1:], path[:-1]
    trade = held - previous
    smooth = np.sqrt(trade ** 2 + epsilon ** 2)
    paid = (cost * (smooth - epsilon)).sum(axis=1)
    net = (held * returns).sum(axis=1) - paid
    centred = net - net.mean()
    objective = (net.mean() - 0.5 * risk_aversion * np.mean(centred ** 2)
                 - 0.5 * l2 * float(np.sum((theta - theta_0) ** 2)))

    weight = (1.0 - risk_aversion * centred) / n_decisions
    marginal_cost = cost * (trade / smooth)
    keep = 1.0 - eta * tradable
    grad_target = np.zeros_like(targets)
    grad_eta = 0.0
    carried = np.zeros(X.shape[1])
    for t in range(n_decisions - 1, -1, -1):
        total = weight[t] * (returns[t] - marginal_cost[t])
        if t + 1 < n_decisions:
            total += weight[t + 1] * marginal_cost[t + 1] + keep[t + 1] * carried
        grad_target[t] = eta * tradable[t] * total
        grad_eta += float(np.sum(tradable[t] * (targets[t] - previous[t]) * total))
        carried = total

    # target = gross * s / sum|s|: d/ds = (gross / A) (I - s sign(s)' / A).
    safe = np.where(size > 0, size, 1.0)
    scale = np.where(size > 0, gross / safe, 0.0)
    along = (grad_target * score).sum(axis=1) / safe
    grad_score = scale[:, None] * (grad_target - np.sign(score) * along[:, None])
    # s = demean(raw) over the active names; the projection is symmetric.
    counts = np.maximum(active.sum(axis=1), 1)
    grad_raw = np.where(
        active, grad_score - (grad_score.sum(axis=1) / counts)[:, None], 0.0)
    grad_theta = np.einsum("dnk,dn->k", X, grad_raw) - l2 * (theta - theta_0)
    grad_rho = grad_eta * eta * (1.0 - eta)
    return float(objective), grad_theta, float(grad_rho)


def to_weights(
    held: ArrayLike, decisions: ArrayLike, assets: ArrayLike, horizon
) -> PortfolioWeights:
    """Wrap a ``(n_decisions, n_assets)`` held path as :class:`PortfolioWeights`.

    The policy holds ONE portfolio per decision, for one holding period, so
    the contract's horizon axis has length one and carries that period. This
    is the form the backtest harness and the execution models score, as the
    :mod:`~stable_finance.evaluation.trades.rl` contract asks of a learned
    policy.
    """
    values = np.asarray(held, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"held weights must be 2-D, got shape {values.shape}")
    return PortfolioWeights(values[:, :, None], decisions, assets, [horizon])


@requires("half_spread")
class ParametricPolicy(BaseEstimator):
    """Characteristic-blend target with a learned trading rate, fitted net of cost.

    See the module docstring for the policy and its objective. The
    constructor arguments are hyperparameters and nothing is learned until
    :meth:`fit`, which sets ``theta_`` (one loading per characteristic),
    ``rho_`` and ``eta_ = sigmoid(rho_)`` (the fraction of the gap to the
    target traded each period), and ``objective_path_``, the training
    utility before each iteration and after the last, ``max_iter + 1``
    values.

    ``characteristics`` is ``(n_decisions, n_assets, k)``, or a
    :class:`ForwardReturns` whose horizon axis is read as the ``k``
    characteristics -- forecasts at several horizons being the natural
    input. ``returns`` is ``(n_decisions, n_assets)``, or a one-horizon
    :class:`ForwardReturns`, holding the non-overlapping one-period return
    earned by the weights chosen at each decision; a fit on the latter
    remembers that horizon so :meth:`predict` on a :class:`ForwardReturns`
    hands back :class:`PortfolioWeights` on it. ``cost`` broadcasts to
    ``(n_decisions, n_assets)``, is the cost per unit of weight traded, and
    is NaN or infinite where a name cannot be traded; :class:`MarketData` in
    its place charges the quoted half-spread (EFQ 1), so scale it yourself
    for any other execution model.

    ``gross_exposure`` is the target's gross, and because the held weights
    are a convex combination of targets and the starting book, the held
    gross never exceeds it from a start within it. ``risk_aversion`` is the
    penalty on the variance of the per-period net return; it multiplies
    returns in the units supplied, so it is per period, not annualized.
    ``learning_rate`` and ``max_iter`` fix the optimizer completely.
    """

    def __init__(
        self,
        gross_exposure: float = 2.0,
        *,
        risk_aversion: float = 5.0,
        l2: float = 0.0,
        standardize: bool = True,
        learning_rate: float = 0.05,
        max_iter: int = 500,
        epsilon: float = 1e-6,
    ) -> None:
        self.gross_exposure = gross_exposure
        self.risk_aversion = risk_aversion
        self.l2 = l2
        self.standardize = standardize
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon

    def _validate(self) -> None:
        checks = (
            ("gross_exposure", self.gross_exposure, lambda v: v > 0, "positive"),
            ("risk_aversion", self.risk_aversion, lambda v: v >= 0, "non-negative"),
            ("l2", self.l2, lambda v: v >= 0, "non-negative"),
            ("learning_rate", self.learning_rate, lambda v: v > 0, "positive"),
            ("epsilon", self.epsilon, lambda v: v > 0, "positive"),
        )
        for name, value, ok, word in checks:
            if not np.isfinite(value) or not ok(value):
                raise ValueError(f"{name} must be finite and {word}")
        if int(self.max_iter) != self.max_iter or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

    def _inputs(self, characteristics, cost, initial):
        if isinstance(characteristics, ForwardReturns):
            values = characteristics.values
        else:
            values = characteristics
        X, active = _characteristics(values, standardize=self.standardize)
        cost = market_quantity(
            cost, "half_spread", stage=ParametricPolicy,
            like=characteristics if isinstance(characteristics, ForwardReturns) else None)
        charge, tradable = _costs(cost, X.shape[:2])
        return X, active, charge, tradable, _initial(initial, X.shape[1])

    def fit(
        self,
        characteristics: ArrayLike | ForwardReturns,
        returns: ArrayLike | ForwardReturns,
        *,
        cost: ArrayLike | MarketData | None = None,
        initial: ArrayLike | None = None,
    ) -> ParametricPolicy:
        """Choose theta and the trading rate to maximise net utility on this panel."""
        self._validate()
        X, active, charge, tradable, start = self._inputs(characteristics, cost, initial)
        n_decisions, n_assets, k = X.shape
        if n_decisions < 2:
            raise ValueError("fitting needs at least two decisions")
        horizon = None
        if isinstance(returns, ForwardReturns):
            if returns.values.shape[2] != 1:
                raise ValueError("returns must hold exactly one holding-period horizon")
            if isinstance(characteristics, ForwardReturns):
                for axis in ("decisions", "assets"):
                    if not np.array_equal(getattr(returns, axis),
                                          getattr(characteristics, axis)):
                        raise ValueError(f"returns {axis} are not aligned")
            horizon = returns.horizons[0]
            returns = returns.values[:, :, 0]
        realized = _returns(returns, (n_decisions, n_assets))

        params = np.concatenate([np.full(k, 1.0 / k), [0.0]])
        first = np.zeros_like(params)
        second = np.zeros_like(params)
        path = np.empty(int(self.max_iter) + 1)
        settings = {"gross": float(self.gross_exposure),
                        "risk_aversion": float(self.risk_aversion),
                        "l2": float(self.l2), "epsilon": float(self.epsilon)}
        for step in range(int(self.max_iter) + 1):
            value, grad_theta, grad_rho = _objective_and_gradient(
                params[:k], float(params[k]), X, active, realized, charge, tradable,
                start, **settings)
            path[step] = value
            if step == self.max_iter:
                break
            gradient = np.concatenate([grad_theta, [grad_rho]])
            first = _ADAM_BETA1 * first + (1.0 - _ADAM_BETA1) * gradient
            second = _ADAM_BETA2 * second + (1.0 - _ADAM_BETA2) * gradient ** 2
            unbiased_first = first / (1.0 - _ADAM_BETA1 ** (step + 1))
            unbiased_second = second / (1.0 - _ADAM_BETA2 ** (step + 1))
            params = params + self.learning_rate * unbiased_first / (
                np.sqrt(unbiased_second) + _ADAM_EPS)

        self.theta_ = params[:k].copy()
        self.rho_ = float(params[k])
        self.eta_ = _sigmoid(self.rho_)
        self.objective_path_ = path
        self.n_features_in_ = k
        self.characteristics_ = (characteristics.horizons.copy()
                                 if isinstance(characteristics, ForwardReturns)
                                 else np.arange(k))
        self.holding_horizon_ = horizon
        return self

    def predict(
        self,
        characteristics: ArrayLike | ForwardReturns,
        *,
        cost: ArrayLike | MarketData | None = None,
        initial: ArrayLike | None = None,
    ) -> NDArray[np.float64] | PortfolioWeights:
        """The held weights, decision by decision, under the fitted rule.

        An array of characteristics gives a ``(n_decisions, n_assets)``
        array; wrap it with :func:`to_weights`. A :class:`ForwardReturns`
        gives :class:`PortfolioWeights` on its decisions and assets, at the
        holding horizon the policy was fitted on. ``cost`` decides only
        which names can trade -- the fitted rule does not re-optimise to a
        new cost -- and defaults to every name tradable. ``initial`` is the
        book held before the first decision and defaults to cash.
        """
        if not hasattr(self, "theta_"):
            raise NotFittedError("fit the ParametricPolicy before calling predict")
        X, active, _, tradable, start = self._inputs(characteristics, cost, initial)
        if X.shape[2] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} characteristics, got {X.shape[2]}")
        targets, _, _ = _targets(X, active, self.theta_, float(self.gross_exposure))
        held = _hold(targets, tradable, self.eta_, start)[1:]
        if not isinstance(characteristics, ForwardReturns):
            return held
        if not np.array_equal(characteristics.horizons, self.characteristics_):
            raise ValueError("characteristics are not aligned with the fitted ones")
        if self.holding_horizon_ is None:
            raise ValueError(
                "fit on ForwardReturns returns to predict PortfolioWeights, or "
                "wrap the array prediction with to_weights")
        return to_weights(held, characteristics.decisions, characteristics.assets,
                          self.holding_horizon_)
