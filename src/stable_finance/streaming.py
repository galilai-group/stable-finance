"""A ridge probe fit from accumulated moments, for a pool too big to hold.

A ridge on d features needs only the second moments of the data, never the
data: ``X'X`` is d x d whatever the row count, so a fit pool of millions of
rows can be consumed one shard at a time in megabytes. That is the difference
between subsampling a pool and using it.

THIS IS EXACT, NOT APPROXIMATE. It solves the same normal equations
:class:`~stable_finance.probes.ColumnwiseRidge` solves and agrees with it to
floating-point precision -- there is a test that asserts exactly that. The only
thing it gives up is a second pass over the rows, which is why prediction
requires the caller to hand the rows back.

    probe = StreamingRidge(alpha=10.0)
    for shard in shards:                       # never all in memory at once
        probe.partial_fit(shard["X"], shard["y"])
    probe.finalize()
    forecast = probe.predict(eval_X)

Numerically this is the textbook trade: forming ``X'X`` squares the condition
number relative to a QR on the raw matrix. It is the right call here anyway --
the features are standardized before the solve, the ridge penalty floors the
spectrum, and float64 accumulation over a few million rows of order-one
numbers leaves far more precision than the fourth decimal an IC is read at.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

__all__ = ["StreamingRidge"]


class StreamingRidge(BaseEstimator):
    """Standardize and ridge-regress targets, accumulating moments only.

    Mirrors :class:`~stable_finance.probes.ColumnwiseRidge`, including how it
    handles missing values: the standardizer is estimated over every row with
    a finite feature vector, and each regression over the subset of those
    whose own label is also finite.

    ``y`` may be a single target or a ``(n_rows, n_targets)`` block. The block
    form exists because the costly moment is ``X'X`` and a pool streamed once
    can fill several targets' moments on the way past -- fitting six forecast
    horizons off one pass rather than six. Targets do NOT share a gram: a long
    horizon runs off the end of the session and is missing on rows a short one
    keeps, so each target accumulates over its own labelled subset and the
    result is identical to fitting it alone. ``alpha`` is a scalar applied to
    every target, or one value per target.
    """

    def __init__(self, alpha: float | ArrayLike = 1.0, *,
                 min_samples: int = 20) -> None:
        self.alpha = alpha
        self.min_samples = min_samples

    def _start(self, n_features: int, n_targets: int) -> None:
        self.n_features_in_ = n_features
        self.n_targets_in_ = n_targets
        # Scaler moments: every row with a finite feature vector. Shared,
        # because standardization does not depend on any label.
        self._scaler_count = 0
        self._scaler_sum = np.zeros(n_features)
        self._scaler_square = np.zeros(n_features)
        # Regression moments: each target's own labelled subset.
        self._count = np.zeros(n_targets, dtype=np.int64)
        self._gram = np.zeros((n_targets, n_features, n_features))
        self._feature_sum = np.zeros((n_targets, n_features))
        self._cross = np.zeros((n_targets, n_features))
        self._label_sum = np.zeros(n_targets)

    def partial_fit(self, X: ArrayLike, y: ArrayLike) -> "StreamingRidge":
        """Fold one block of rows into the accumulated moments."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        single = y.ndim == 1
        if single:
            y = y[:, None]
        if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row of labels per row")
        if not hasattr(self, "n_features_in_"):
            self._start(X.shape[1], y.shape[1])
            self._single_target = single
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features, got {X.shape[1]}"
            )
        elif y.shape[1] != self.n_targets_in_:
            raise ValueError(
                f"y must have {self.n_targets_in_} targets, got {y.shape[1]}"
            )
        if hasattr(self, "coefficients_"):
            raise RuntimeError("this StreamingRidge is already finalized")

        usable = np.isfinite(X).all(axis=1)
        if usable.any():
            block = X[usable]
            self._scaler_count += len(block)
            self._scaler_sum += block.sum(axis=0)
            self._scaler_square += np.einsum("ij,ij->j", block, block)
            rows = y[usable]
            for target in range(self.n_targets_in_):
                labelled = np.isfinite(rows[:, target])
                if not labelled.any():
                    continue
                fit_block = block[labelled]
                labels = rows[labelled, target]
                self._count[target] += len(fit_block)
                self._gram[target] += fit_block.T @ fit_block
                self._feature_sum[target] += fit_block.sum(axis=0)
                self._cross[target] += fit_block.T @ labels
                self._label_sum[target] += float(labels.sum())
        return self

    def finalize(self) -> "StreamingRidge":
        """Solve each target's ridge from the accumulated moments."""
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError("call partial_fit before finalize")
        short = np.flatnonzero(self._count < self.min_samples)
        if len(short):
            raise ValueError(
                f"target {int(short[0])} has {int(self._count[short[0]])} "
                f"labelled rows; need {self.min_samples}"
            )
        mean = self._scaler_sum / self._scaler_count
        variance = self._scaler_square / self._scaler_count - mean ** 2
        scale = np.sqrt(np.maximum(variance, 0.0))
        # A constant feature has no scale; sklearn's StandardScaler leaves it
        # at zero rather than dividing by zero, and so does this.
        scale[scale <= 0] = 1.0

        alphas = np.broadcast_to(
            np.asarray(self.alpha, dtype=np.float64), (self.n_targets_in_,)
        )
        if not (np.isfinite(alphas).all() and (alphas >= 0).all()):
            raise ValueError("alpha must be finite and non-negative")

        coefficients = np.empty((self.n_targets_in_, self.n_features_in_))
        intercepts = np.empty(self.n_targets_in_)
        for target in range(self.n_targets_in_):
            count = int(self._count[target])
            # Second moments of the STANDARDIZED features on this target's
            # labelled rows.
            outer = np.outer(mean, self._feature_sum[target])
            centred = (self._gram[target] - outer - outer.T
                       + count * np.outer(mean, mean))
            gram = centred / np.outer(scale, scale)
            cross = (self._cross[target] - mean * self._label_sum[target]) / scale

            # Ridge with an intercept centres both sides on the fitted rows, so
            # the penalty never touches the intercept -- matching sklearn.
            feature_mean = (self._feature_sum[target] / count - mean) / scale
            label_mean = self._label_sum[target] / count
            gram -= count * np.outer(feature_mean, feature_mean)
            cross -= count * feature_mean * label_mean

            gram.flat[:: self.n_features_in_ + 1] += float(alphas[target])
            coefficients[target] = np.linalg.solve(gram, cross)
            intercepts[target] = label_mean - float(feature_mean @ coefficients[target])

        self.mean_, self.scale_ = mean, scale
        self.n_samples_ = (int(self._count[0]) if self._single_target
                           else self._count.astype(int).copy())
        self.coefficients_ = (coefficients[0] if self._single_target
                              else coefficients)
        self.intercept_ = (float(intercepts[0]) if self._single_target
                           else intercepts)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict, leaving rows with a non-finite feature vector as NaN.

        Shape follows what was fit: ``(n_rows,)`` for a single target,
        ``(n_rows, n_targets)`` for a block.
        """
        if not hasattr(self, "coefficients_"):
            raise NotFittedError("call finalize before predict")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_rows, {self.n_features_in_})")
        shape = (len(X),) if self._single_target else (len(X), self.n_targets_in_)
        out = np.full(shape, np.nan)
        usable = np.isfinite(X).all(axis=1)
        if usable.any():
            scaled = (X[usable] - self.mean_) / self.scale_
            if self._single_target:
                out[usable] = scaled @ self.coefficients_ + self.intercept_
            else:
                out[usable] = scaled @ self.coefficients_.T + self.intercept_
        return out
