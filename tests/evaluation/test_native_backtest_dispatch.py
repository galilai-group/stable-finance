"""The optional physical objective must not change legacy backtest behavior."""

from dataclasses import replace

import numpy as np
import pytest

from stable_finance import BacktestConfig, ForwardReturns, run_backtest


def legacy_inputs():
    decisions = np.arange(4)*5
    forecast = ForwardReturns(np.array([[.01, -.01], [.02, -.02], [-.01, .01], [.01, -.01]])[:, :, None],
                              decisions, ["A", "B"], [5])
    realized = ForwardReturns(np.array([[.002, -.001], [-.003, .001], [.001, -.001], [.002, .001]])[:, :, None],
                              decisions, ["A", "B"], [5])
    config = BacktestConfig(weighting="equal", cost_model="cross", efq=.7,
                            annualization="decisions")
    return forecast, realized, config


def test_omitting_objective_or_passing_none_keeps_legacy_rows_and_series(monkeypatch):
    from stable_finance.evaluation import return_risk_backtest

    def unexpected(*args, **kwargs):
        raise AssertionError("the legacy path must not call the physical objective")

    monkeypatch.setattr(return_risk_backtest, "run_return_risk_backtest", unexpected)
    forecast, realized, config = legacy_inputs()
    spread = np.full((4, 2), .0001)
    old = run_backtest(forecast, realized, half_spread=spread, config=config, with_series=True)
    explicit = run_backtest(forecast, realized, half_spread=spread, config=config, with_series=True,
                            objective=None, execution_state=None)
    assert old.as_dict() == explicit.as_dict()
    assert old.objective_trace is None
    assert "objective_trace" not in old.as_dict()
    for field in ("gross", "net", "cost", "periods"):
        np.testing.assert_array_equal(getattr(old.series, field), getattr(explicit.series, field))


def test_flat_report_does_not_copy_or_serialize_an_execution_trace():
    class NoDeepcopy:
        def __deepcopy__(self, memo):
            raise AssertionError("a flat report must not copy the full physical trace")

    forecast, realized, config = legacy_inputs()
    report = run_backtest(forecast, realized, half_spread=np.full((4, 2), .0001), config=config)
    with_trace = replace(report, objective_trace=NoDeepcopy())
    assert report.as_dict() == with_trace.as_dict()
    assert "objective_trace" not in with_trace.as_dict()


@pytest.mark.parametrize("extra", ["history", "embeddings", "train", "probe"])
def test_supplied_objective_cannot_silently_refit_or_override_risk(extra):
    forecast, realized, config = legacy_inputs()
    with pytest.raises(ValueError, match="supplied forecasts and risk"):
        run_backtest(forecast, realized, config=config, objective=object(), **{extra:object()})
