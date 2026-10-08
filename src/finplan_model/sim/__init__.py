"""The common paper execution simulator (paper only, never live)."""

from .config import ConstraintSet, SimulationConfig
from .engine import Fill, SimulationResult, Simulator, rebalance_sessions
from .market import Bar, HoldingsView, MarketData, PointInTimeView, synthetic_market
from .strategy import ReplayStrategy, Strategy, TargetWeights, aggregate_solution_status, coerce_target

__all__ = [
    "Bar",
    "ConstraintSet",
    "Fill",
    "HoldingsView",
    "MarketData",
    "PointInTimeView",
    "ReplayStrategy",
    "SimulationConfig",
    "SimulationResult",
    "Simulator",
    "Strategy",
    "TargetWeights",
    "aggregate_solution_status",
    "coerce_target",
    "rebalance_sessions",
    "synthetic_market",
]
