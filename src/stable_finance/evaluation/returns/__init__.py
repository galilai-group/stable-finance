"""Stage 2, embeddings to forward returns: probes, oracle forecasts, and the IC."""

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
    fit,
)
from stable_finance.evaluation.returns.scoring import (
    InformationCoefficient,
    evaluate_forward_returns,
    grouped_rank_ic,
    grouped_rank_ic_by_label,
    paired_difference,
    pooled_estimates,
)
from stable_finance.evaluation.returns.streaming import (
    StreamingRidge,
)

__all__ = [
    "ColumnwiseRidge",
    "ForwardReturns",
    "InformationCoefficient",
    "MonthlyProbeData",
    "NOISE_MODES",
    "RidgeProbe",
    "StreamingRidge",
    "evaluate_forward_returns",
    "fit",
    "grouped_rank_ic",
    "grouped_rank_ic_by_label",
    "oracle_forecast",
    "paired_difference",
    "pooled_estimates",
]
