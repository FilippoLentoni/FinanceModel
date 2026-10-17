"""Common-simulator adapter for the exact optimizer used by the classical MCP service."""

from __future__ import annotations

from finplan_model.sim.strategy import TargetWeights
from finplan_model.strategies.estimators import (
    estimate_covariance,
    estimate_mean,
    pit_returns,
)

from .math import settings, solve


class ClassicalStrategy:
    def __init__(self, algorithm, *, lookback=60, max_weight=0.6, risk_aversion=2.0):
        self.name = algorithm
        self.algorithm = "cvar" if algorithm == "scenario_cvar" else algorithm
        self.settings = settings(
            {
                "lookback_days": lookback,
                "max_weight": max_weight,
                "risk_aversion": risk_aversion,
            }
        )

    def decide(self, view, holdings):
        returns = pit_returns(view, self.settings["lookback_days"])
        current = [float(holdings.weights.get(i, 0.0)) for i in view.instruments]
        if len(returns) < self.settings["lookback_days"]:
            return TargetWeights(
                dict(zip(view.instruments, current)),
                float(holdings.cash_weight),
                "no_effect",
                {"reason": "insufficient_history"},
            )
        inputs = {
            "algorithm": self.algorithm,
            "instruments": list(view.instruments),
            "settings": self.settings,
            "current_weights": current,
            "expected_returns": estimate_mean(returns, "historical_mean").tolist(),
            "covariance": estimate_covariance(returns, "ledoit_wolf").tolist(),
            "scenarios": returns.tolist(),
        }
        result = solve(inputs)
        if result["status"] != "optimal":
            return TargetWeights(
                dict(zip(view.instruments, current)),
                float(holdings.cash_weight),
                "infeasible",
                {"reason": result["reason"]},
            )
        return TargetWeights(
            dict(zip(view.instruments, result["weights"])),
            result["cash_weight"],
            "optimal",
            {
                "classical_implementation": "finplan-classical/1",
                "settings": self.settings,
            },
        )
