"""Evaluation building blocks for financial models and portfolios.

``stable_finance.dataset`` turns market time series into sessions, views and
labels; ``stable_finance.evaluation`` runs the pipeline from embeddings to
trades. Everything public in the evaluation pipeline is re-exported here, so
``import stable_finance as sf; sf.run_backtest(...)`` keeps working.
"""

from stable_finance import _moved

_moved.install()

from stable_finance.dataset.spec.schema import MissingRolesError
from stable_finance.dataset.labels.tables import (
    TARGET_TRANSFORMS,
    AnchorTargetStats,
    CrossSectionalTargetBundle,
)
from stable_finance.evaluation import *  # noqa: E402,F401,F403
from stable_finance.evaluation import __all__ as _evaluation  # noqa: E402

__all__ = sorted({*_evaluation, "AnchorTargetStats", "CrossSectionalTargetBundle",
                  "MissingRolesError", "TARGET_TRANSFORMS"})
