"""Released, bounded PPO ablations; provider text cannot alter executable configuration."""

from __future__ import annotations

import copy

from finplan_model.core.errors import FinplanError

STATE_PROFILES = {
    "recursive_ppo_state_baseline": (),
    "recursive_ppo_state_risk": ("cross_asset_risk",),
    "recursive_ppo_state_regime": ("market_regime",),
    "recursive_ppo_state_sequence": ("recent_returns",),
    "recursive_ppo_state_combined": ("cross_asset_risk", "market_regime", "recent_returns"),
}
PROFILES = ("recursive_ppo_features", "recursive_ppo_turnover", "recursive_ppo_horizon", *STATE_PROFILES)


def profile_protocol(base, profile):
    if profile not in PROFILES:
        raise FinplanError.validation("unknown released recursive PPO profile", pointer="/configuration/payload/objective")
    p = copy.deepcopy(base)
    if profile in STATE_PROFILES:
        # Exactly one independent state change against the same released baseline.
        p["runtime"]["reserve_seconds"] = 180
        p["rl"]["algorithms"] = ["ppo"]
        p["rl"]["seeds"] = [0, 1, 2, 3, 4]
        p["rl"]["grid"] = {"risk_penalty": [.5]}
        p["rl"]["policy_selection"] = "ensemble"
        p["rl"]["env"].update(window=60, features=["market_summary", "current_weights", "portfolio_drawdown", *STATE_PROFILES[profile]],
                                action_scale=1., episode_sessions=64, rebalance_fraction=.25, step_sessions=1, decision_frequency="daily")
        p["rl"]["env"]["reward"].update(risk_penalty=.5, drawdown_penalty=.5, turnover_penalty=.001, reward_scale=100.)
        p["rl"]["ppo"].update(total_timesteps=30000, eval_every=2048, patience=5, gamma=.99)
        return p
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
