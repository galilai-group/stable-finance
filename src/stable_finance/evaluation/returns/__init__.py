"""Stage 2, embeddings to forward returns: ridge probes, in memory or streamed, and oracle forecasts."""

from stable_finance.evaluation.returns.contract import (
    ForwardReturns,
)
from stable_finance.evaluation.returns.oracle import (
    NOISE_MODES,
    QUALITY_VARIABLES,
    copula_forecast,
    factor_forecast,
    oracle_forecast,
)
from stable_finance.evaluation.returns.probes import (
    ColumnwiseRidge,
    ProbeData,
    RidgeProbe,
    StreamingRidge,
    fit,
)

__all__ = [
    "ColumnwiseRidge",
    "ForwardReturns",
    "NOISE_MODES",
    "ProbeData",
    "QUALITY_VARIABLES",
    "RidgeProbe",
    "StreamingRidge",
    "copula_forecast",
    "factor_forecast",
    "fit",
    "oracle_forecast",
]
