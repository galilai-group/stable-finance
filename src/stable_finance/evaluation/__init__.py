"""The evaluation pipeline, one package per stage.

    embeddings  ->  returns  ->  portfolio  ->  trades     ->  scoring
    (you bring)     probes,      universe,      policies:       IC, P&L,
                    oracle       selection,     rebalance,      Sharpe,
                                 risk, sizing   trade rules     execution-
                                 (no costs)                     quality curves

                       execution: orders, fills, costs at an EFQ

Each stage package holds its stage's contract (the object it produces) and
the code that produces it from the stage before. The portfolio stage solves
for what to hold as if trading were free. The trades stage holds the policies
that decide how far to move there given what trading costs. execution is the
model of what a trade costs and where it fills, set by one number, the EFQ.
scoring measures each stage's output: the IC of a forecast, and the P&L and
Sharpe of what a portfolio actually held. config and backtest are the harness
that runs all of them under one named configuration. market_data is what
every stage may read of the observable market, with the declarations by which
a stage says which market columns it needs. inputs turns your own table into
realized returns and market data.
"""

from stable_finance.evaluation._grid import (
    require_aligned,
)
from stable_finance.evaluation.backtest import (
    BacktestReport,
    PnLSeries,
    run_backtest,
    sweep_configs,
)
from stable_finance.evaluation.config import (
    ALLOCATORS,
    ANNUALIZATIONS,
    AXIS_REQUIRES,
    CONFIG_SPACE,
    DEFAULT_BACKTEST,
    EFQ_GRID,
    REBALANCE_RULES,
    BacktestConfig,
    check_config,
    enumerate_configs,
)
from stable_finance.evaluation.embeddings import (
    Embeddings,
    require_same_observations,
)
from stable_finance.evaluation.execution import (
    COST_MODEL_REQUIRES,
    COST_MODELS,
    EFQ_DIRECT_ACCESS,
    EFQ_INFORMED_RANGE,
    EFQ_LIT_EXCHANGE,
    Fills,
    Orders,
    effective_half_spread,
    fill_requires,
    orders_from_weights,
    simulate_fills,
    trading_cost,
)
from stable_finance.evaluation.inputs import (
    evaluation_inputs,
    to_grid,
)
from stable_finance.evaluation.market_data import (
    MARKET_QUANTITIES,
    MarketData,
    requires,
)
from stable_finance.evaluation.portfolio import (
    RISK_MODEL_REQUIRES,
    RISK_MODELS,
    SELECTION_REQUIRES,
    SELECTION_RULES,
    UNIVERSE_REQUIRES,
    UNIVERSE_SCREENS,
    WEIGHTING_REQUIRES,
    WEIGHTING_RULES,
    PortfolioWeights,
    apply_screen,
    diagonal_covariance,
    embedding_covariance,
    estimate_covariance,
    estimate_risk,
    ledoit_wolf_covariance,
    liquidity_screen,
    mean_variance_target,
    presence_screen,
    select_book,
    tail_selection,
    tradable,
    weight_book,
)
from stable_finance.evaluation.returns import (
    NOISE_MODES,
    ColumnwiseRidge,
    ForwardReturns,
    MonthlyProbeData,
    RidgeProbe,
    StreamingRidge,
    fit,
    oracle_forecast,
)
from stable_finance.evaluation.scoring import (
    BacktestResult,
    ExecutionQualityCurve,
    InformationCoefficient,
    breakeven_efq,
    cross_spread_sharpe,
    evaluate_fills,
    evaluate_forward_returns,
    evaluate_weights,
    grouped_rank_ic,
    grouped_rank_ic_by_label,
    mid_price_sharpe,
    paired_difference,
    pooled_estimates,
    positions_from_fills,
    sharpe_ratio,
    sharpe_vs_execution_quality,
)
from stable_finance.evaluation.trades import (
    MeanVarianceAllocator,
    allocate,
    edge_to_cost,
    persistent_selection,
    rebalance,
    threshold_selection,
)

__all__ = [
    "ALLOCATORS",
    "ANNUALIZATIONS",
    "AXIS_REQUIRES",
    "BacktestConfig",
    "BacktestReport",
    "BacktestResult",
    "CONFIG_SPACE",
    "COST_MODELS",
    "COST_MODEL_REQUIRES",
    "ColumnwiseRidge",
    "DEFAULT_BACKTEST",
    "EFQ_DIRECT_ACCESS",
    "EFQ_GRID",
    "EFQ_INFORMED_RANGE",
    "EFQ_LIT_EXCHANGE",
    "Embeddings",
    "ExecutionQualityCurve",
    "Fills",
    "ForwardReturns",
    "InformationCoefficient",
    "MARKET_QUANTITIES",
    "MarketData",
    "MeanVarianceAllocator",
    "MonthlyProbeData",
    "NOISE_MODES",
    "Orders",
    "PnLSeries",
    "PortfolioWeights",
    "REBALANCE_RULES",
    "RISK_MODELS",
    "RISK_MODEL_REQUIRES",
    "RidgeProbe",
    "SELECTION_REQUIRES",
    "SELECTION_RULES",
    "StreamingRidge",
    "UNIVERSE_REQUIRES",
    "UNIVERSE_SCREENS",
    "WEIGHTING_REQUIRES",
    "WEIGHTING_RULES",
    "allocate",
    "apply_screen",
    "breakeven_efq",
    "check_config",
    "cross_spread_sharpe",
    "diagonal_covariance",
    "edge_to_cost",
    "effective_half_spread",
    "embedding_covariance",
    "enumerate_configs",
    "estimate_covariance",
    "estimate_risk",
    "evaluate_fills",
    "evaluate_forward_returns",
    "evaluate_weights",
    "evaluation_inputs",
    "fill_requires",
    "fit",
    "grouped_rank_ic",
    "grouped_rank_ic_by_label",
    "ledoit_wolf_covariance",
    "liquidity_screen",
    "mean_variance_target",
    "mid_price_sharpe",
    "oracle_forecast",
    "orders_from_weights",
    "paired_difference",
    "persistent_selection",
    "pooled_estimates",
    "positions_from_fills",
    "presence_screen",
    "rebalance",
    "require_aligned",
    "require_same_observations",
    "requires",
    "run_backtest",
    "select_book",
    "sharpe_ratio",
    "sharpe_vs_execution_quality",
    "simulate_fills",
    "sweep_configs",
    "tail_selection",
    "threshold_selection",
    "to_grid",
    "tradable",
    "trading_cost",
    "weight_book",
]
