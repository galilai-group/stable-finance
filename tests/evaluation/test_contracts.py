import numpy as np
import pytest

from stable_finance.evaluation._grid import require_aligned
from stable_finance.evaluation.embeddings.contract import Embeddings
from stable_finance.evaluation.portfolio.contract import PortfolioWeights
from stable_finance.evaluation.returns.contract import ForwardReturns


def contract(values=None, *, assets=("A", "B")):
    if values is None:
        values = np.zeros((2, 2, 1))
    return ForwardReturns(values, [1, 2], assets, [300])


def test_contract_rejects_a_shape_that_does_not_match_its_axes():
    with pytest.raises(ValueError, match="values must have shape"):
        contract(np.zeros((2, 3, 1)))


def test_a_horizon_is_a_positive_duration_or_to_close():
    from stable_finance.dataset.anchors import TO_CLOSE

    held_to_close = ForwardReturns(np.zeros((2, 2, 2)), [1, 2], ["A", "B"], [300, TO_CLOSE])
    assert list(held_to_close.horizons) == [300, TO_CLOSE]
    for bad in (0, -2):
        with pytest.raises(ValueError, match="positive durations or TO_CLOSE"):
            ForwardReturns(np.zeros((2, 2, 1)), [1, 2], ["A", "B"], [bad])


def test_alignment_checks_axis_values_not_only_shape():
    returns = contract()
    weights = PortfolioWeights(np.zeros((2, 2, 1)), [1, 2], ["B", "A"], [300])
    with pytest.raises(ValueError, match="assets are not aligned"):
        require_aligned(weights, returns)


def test_embeddings_validate_the_observation_axes():
    with pytest.raises(ValueError, match="values must have shape"):
        Embeddings(np.zeros((3, 2, 4)), [1, 2], ["A", "B"])
