"""Released, bounded PPO ablations; provider text cannot alter executable configuration."""

from __future__ import annotations

import copy

from finplan_model.core.errors import FinplanError

PROFILES = ("recursive_ppo_features", "recursive_ppo_turnover", "recursive_ppo_horizon")


def profile_protocol(base, profile):
    if profile not in PROFILES:
        raise FinplanError.validation("unknown released recursive PPO profile", pointer="/configuration/payload/objective")
    p = copy.deepcopy(base)
    p["runtime"]["reserve_seconds"] = 180
    p["rl"]["algorithms"] = ["ppo"]
    p["rl"]["seeds"] = [0, 1, 2]
    p["rl"]["grid"] = {"risk_penalty": [1.]}
    p["rl"]["policy_selection"] = "ensemble"
    p["rl"]["ppo"].update(total_timesteps=8000, eval_every=2048, patience=3)
    p["traditional"] = {
        "min_variance": {"grid": {"lookback": [60]}, "fixed": {}},
        "mean_variance": {"grid": {"lookback": [60], "risk_aversion": [2.]}, "fixed": {}},
        "scenario_cvar": {"grid": {"lookback": [60], "alpha": [.9]}, "fixed": {"scenario_method": "historical"}},
    }
    if profile == "recursive_ppo_features":
        p["rl"]["env"]["features"] = ["log_return_window", "market_summary", "current_weights", "portfolio_drawdown"]
    elif profile == "recursive_ppo_turnover":
        p["rl"]["env"]["reward"]["turnover_penalty"] = .01
        p["rl"]["env"]["rebalance_fraction"] = .10
    else:
        p["rl"]["env"]["episode_sessions"] = 126
        p["rl"]["ppo"]["gamma"] = .995
    return p
