import numpy as np
import pytest

from stable_finance.data import ForwardReturns, PortfolioWeights, require_aligned


def contract(values=None, *, assets=("A", "B")):
    if values is None:
        values = np.zeros((2, 2, 1))
    return ForwardReturns(values, [1, 2], assets, [300])


def test_contract_rejects_a_shape_that_does_not_match_its_axes():
    with pytest.raises(ValueError, match="values must have shape"):
        contract(np.zeros((2, 3, 1)))


def test_alignment_checks_axis_values_not_only_shape():
    returns = contract()
    weights = PortfolioWeights(np.zeros((2, 2, 1)), [1, 2], ["B", "A"], [300])
    with pytest.raises(ValueError, match="assets are not aligned"):
        require_aligned(weights, returns)

