"""Evaluation building blocks for financial models and portfolios.

``stable_finance.dataset`` turns market time series into sessions, views and
labels; ``stable_finance.evaluation`` runs the pipeline from embeddings to
trades. Everything public in the evaluation pipeline is re-exported here, so
``import stable_finance as sf; sf.run_backtest(...)`` keeps working.
"""

from stable_finance import _moved

_moved.install()

from stable_finance.dataset.schema import MissingRolesError
from stable_finance.dataset.datasets.market1t.tables import (
    TARGET_TRANSFORMS,
    AnchorTargetStats,
    CrossSectionalTargetBundle,
)
from stable_finance.evaluation import *  # noqa: E402,F401,F403
from stable_finance.evaluation import __all__ as _evaluation  # noqa: E402

# Market-1T's evaluation readers: market data and returns from its stores.
from stable_finance.dataset.datasets.market1t.daystore import (  # noqa: E402
    decision_labels,
    forward_returns_from_days,
    market_from_days,
)
from stable_finance.dataset.datasets.market1t.views import (  # noqa: E402
    MarketPanel,
    find_cached_month,
)

__all__ = sorted({*_evaluation, "AnchorTargetStats", "CrossSectionalTargetBundle",
                  "MissingRolesError", "TARGET_TRANSFORMS", "MarketPanel", "decision_labels",
                  "find_cached_month", "forward_returns_from_days", "market_from_days"})
