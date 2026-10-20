"""An objective is a label definition, resolved against a schema and keyed."""
from __future__ import annotations

import pytest

from stable_finance.dataset import MARKET_SCHEMA, TO_CLOSE, FeatureSchema, Objective



def test_an_objective_resolves_against_the_schema_and_keys_its_definition():
    univariate = FeatureSchema.univariate()
    resolved = Objective().resolve(univariate)
    assert resolved.types == ("return", "volatility_change") and resolved.price == "mid"
    assert resolved.key == Objective().resolve(univariate).key           # stable
    assert resolved.key != Objective(window_seconds=300).resolve(univariate).key
    assert resolved.key != Objective().resolve(MARKET_SCHEMA).key        # vwap, three families
    with pytest.raises(ValueError, match="resolve"):
        Objective().key
    with pytest.raises(Exception, match="bid_price"):
        Objective(types=("spread_change",)).resolve(univariate)


def test_names_carry_the_window_and_lookahead_is_horizon_plus_window():
    five = Objective(types=("return",), horizons=(900,), window_seconds=300)
    assert five.resolve(MARKET_SCHEMA).names == ["return_900@300"]
    assert Objective(types=("return",), horizons=(900,)).resolve(MARKET_SCHEMA).names == [
        "return_900"]
    assert five.lookahead_seconds == 1200
    assert Objective(horizons=(300, TO_CLOSE)).lookahead_seconds is None
    with pytest.raises(ValueError):
        Objective(horizons=(0,))
    with pytest.raises(ValueError):
        Objective(cross_section="batch")
