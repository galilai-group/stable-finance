"""Stage 2, embeddings to forward returns: ridge probes, in memory or streamed, and oracle forecasts."""

from stable_finance.evaluation.returns.contract import (
    ForwardReturns,
)
from stable_finance.evaluation.returns.oracle import (
    NOISE_MODES,
    oracle_forecast,
)
from stable_finance.evaluation.returns.probes import (
    ColumnwiseRidge,
    MonthlyProbeData,
    RidgeProbe,
    StreamingRidge,
    fit,
)

__all__ = [
    "ColumnwiseRidge",
    "ForwardReturns",
    "MonthlyProbeData",
    "NOISE_MODES",
    "RidgeProbe",
    "StreamingRidge",
    "fit",
    "oracle_forecast",
]
