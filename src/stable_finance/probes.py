"""Textbook probes that turn model embeddings into return forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stable_finance.data import (
    Embeddings,
    ForwardReturns,
    require_same_observations,
)
from stable_finance.dataset.months import Month


class MonthlyProbeData(Protocol):
    """Model-specific provider used by :meth:`RidgeProbe.fit_month`."""

    def embeddings(self, month: Month) -> Embeddings: ...

    def forward_returns(self, month: Month) -> ForwardReturns: ...


class RidgeProbe(BaseEstimator):
    """One standardized ridge regression per forward-return horizon.

    This preserves the originating market-jepa recipe: fit one
    :class:`StandardScaler` across embedding rows, then independently fit a
    ridge for each target column after removing that column's missing labels.
    A scalar ``alpha`` applies to every horizon; a sequence configures one
    penalty per horizon.
    """

    def __init__(self, alpha: float | Sequence[float] = 1.0, *,
                 min_samples: int = 20) -> None:
        self.alpha = alpha
        self.min_samples = min_samples

    def _alphas(self, n_horizons: int) -> np.ndarray:
        values = np.asarray(self.alpha, dtype=np.float64)
        if values.ndim == 0:
            values = np.repeat(values.item(), n_horizons)
        if values.shape != (n_horizons,):
            raise ValueError(
                f"alpha must be a scalar or have length {n_horizons}, "
                f"got shape {values.shape}"
            )
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("alpha values must be finite and non-negative")
        return values

    def fit(self, embeddings: Embeddings, targets: ForwardReturns) -> "RidgeProbe":
        """Fit the scaler and horizon-specific regressions."""
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        require_same_observations(embeddings, targets)
        n_features = embeddings.values.shape[2]
        X = embeddings.values.reshape(-1, n_features)
        y = targets.values.reshape(-1, len(targets.horizons))
        usable_features = np.isfinite(X).all(axis=1)
        if usable_features.sum() < self.min_samples:
            raise ValueError(
                "not enough rows with finite embeddings: "
                f"need {self.min_samples}, got {usable_features.sum()}"
            )

        scaler = StandardScaler().fit(X[usable_features])
        scaled = scaler.transform(X[usable_features])
        usable_rows = np.flatnonzero(usable_features)
        models: list[Ridge] = []
        alphas = self._alphas(len(targets.horizons))
        for horizon, alpha in enumerate(alphas):
            finite_target = np.isfinite(y[usable_rows, horizon])
            count = int(finite_target.sum())
            if count < self.min_samples:
                raise ValueError(
                    f"horizon {targets.horizons[horizon]!r} has {count} usable "
                    f"rows; need at least {self.min_samples}"
                )
            models.append(
                Ridge(alpha=float(alpha)).fit(
                    scaled[finite_target], y[usable_rows[finite_target], horizon]
                )
            )

        self.scaler_ = scaler
        self.models_ = tuple(models)
        self.horizons_ = targets.horizons.copy()
        self.n_features_in_ = n_features
        return self

    def fit_month(
        self, month: Month | str, source: MonthlyProbeData
    ) -> "RidgeProbe":
        """Load one fit month through a model-specific provider and fit.

        Stable-finance owns what a month and return panel mean; the provider
        owns how a particular model produces embeddings. This keeps checkpoint
        loading out of the evaluation library while giving orchestration code a
        month-level API.
        """
        fit_month = Month.parse(month)
        self.fit(source.embeddings(fit_month), source.forward_returns(fit_month))
        self.fit_month_ = fit_month
        return self

    def predict(self, embeddings: Embeddings) -> ForwardReturns:
        """Predict every fitted horizon, preserving the input observation grid."""
        if not hasattr(self, "models_"):
            raise NotFittedError("fit the RidgeProbe before calling predict")
        if embeddings.values.shape[2] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} embedding features, "
                f"got {embeddings.values.shape[2]}"
            )

        shape = embeddings.values.shape[:2]
        X = embeddings.values.reshape(-1, self.n_features_in_)
        usable = np.isfinite(X).all(axis=1)
        predictions = np.full((len(X), len(self.models_)), np.nan)
        if usable.any():
            scaled = self.scaler_.transform(X[usable])
            for horizon, model in enumerate(self.models_):
                predictions[usable, horizon] = model.predict(scaled)
        return ForwardReturns(
            predictions.reshape(*shape, len(self.models_)),
            embeddings.decisions,
            embeddings.assets,
            self.horizons_,
        )


def fit(
    embeddings: Embeddings,
    targets: ForwardReturns,
    *,
    alpha: float | Sequence[float] = 1.0,
    min_samples: int = 20,
) -> RidgeProbe:
    """Fit the default embeddings-to-forecast adapter.

    This is the concise entry point for users entering the pipeline with
    embeddings. It intentionally performs no target transformation: raw
    returns, cross-sectional z-scores, or another target are fit exactly as
    supplied. Instantiate :class:`RidgeProbe` directly when composing through
    scikit-learn estimator tooling.
    """
    return RidgeProbe(alpha=alpha, min_samples=min_samples).fit(embeddings, targets)
