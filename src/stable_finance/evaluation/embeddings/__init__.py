"""Stage 1, what the model brings: embeddings on a decision and asset grid."""

from stable_finance.evaluation.embeddings.contract import (
    Embeddings,
    require_same_observations,
)

__all__ = [
    "Embeddings",
    "require_same_observations",
]
