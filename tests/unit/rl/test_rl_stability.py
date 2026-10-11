"""Synthetic checks of beta's observable state, episode boundaries and ensemble evaluation."""
import numpy as np
import pytest

from finplan_model.evaluate import evaluate
from finplan_model.rl.arrays import PriceArrays
from finplan_model.rl.env import PortfolioEnv
from finplan_model.rl.policy import RLPolicyStrategy
from finplan_model.rl.spec import EnvSpec, build_observation
from finplan_model.selection.job import select_models
from finplan_model.sim.config import SimulationConfig
from tests.unit.rl.support import market, sim_config, tiny_protocol


def daily_config():
    return SimulationConfig.from_dict({**sim_config().to_dict(), "rebalance_frequency": "daily"})


def spec():
    return EnvSpec.from_dict({"window": 20, "features": ["market_summary", "current_weights", "portfolio_drawdown"],
                              "episode_sessions": 32, "rebalance_fraction": 0.25, "action_scale": 1.0})


def test_state_and_smoothed_actions_match_common_evaluator_with_drawdown():
    mkt, cfg, s = market(), daily_config(), spec()
    arrays = PriceArrays.from_market(mkt)
    observations = []

    def predict(obs):
        observations.append(obs.copy())
        # The action depends on both market summaries and the portfolio drawdown.
        return np.tanh(obs[:6] + obs[-1] * np.arange(6))

    result = evaluate(RLPolicyStrategy("ppo", predict, s, seed=0, configuration_id="cfg_test"), mkt, cfg,
                      start=mkt.sessions[30], end=mkt.sessions[120])
    common_obs = list(observations)
    observations.clear()
    out = PortfolioEnv(arrays, s, cfg, i0=30, i1=120, mode="eval").evaluate(predict)
    assert np.allclose(common_obs, observations, atol=1e-6)
    assert np.max(np.asarray(common_obs)[:, -1]) > 0  # peak-dependent observation was exercised
    assert np.allclose([r["value"] for r in result.simulation.nav], out["values"], atol=1e-6)


def test_summary_features_match_visible_prices_and_need_observed_drawdown():
    s, mkt = spec(), market()
    _, visible = mkt.view(mkt.sessions[50]).price_matrix("close", lookback=s.window + 1)
    obs = build_observation(s, visible, np.zeros(5), 1.0, drawdown=0.2)
    assert obs.shape == (37,) and obs[-1] == pytest.approx(0.2)
    assert np.allclose(obs[:5], np.log(visible[1:] / visible[:-1])[-5:].mean(axis=0) * s.return_scale)


def test_randomized_episodes_stay_in_training_and_bootstrap_at_artificial_boundary():
    arrays = PriceArrays.from_market(market())
    env = PortfolioEnv(arrays, spec(), daily_config(), i0=0, i1=100, mode="train")
    starts = set()
    for seed in range(10):
        env.reset(seed=seed)
        starts.add(env._schedule[0])
        assert env._schedule[0] >= 20 and env._episode_end <= 100
    assert len(starts) > 1
    env.reset(seed=0)
    assert env._episode_end < 100
    for _ in range(32):
        obs, _, terminated, truncated, _ = env.step(np.zeros(6))
    assert truncated and not terminated
    assert np.any(obs != 0) and obs.shape == env.observation_space.shape
    env.reset(seed=0)
    assert env._schedule[0] in starts


def test_seed_ensemble_is_evaluated_and_reused_test_cannot_promote():
    p = tiny_protocol()
    p["rl"]["algorithms"] = ["ppo"]
    p["rl"]["policy_selection"] = "ensemble"
    out = select_models(market(), daily_config(), protocol=p, prior_test_accesses=1)
    block = out.summary["rl"]["ppo"]
    assert block["chosen"]["seed"] is None
    assert block["chosen"]["policy_selection"] == "ensemble"
    assert not any(s["selected"] for s in block["seeds"])
    assert len(block["seeds"]) == 3
    assert out.summary["promotion_check"]["result"] == "research_only"
    assert "test_reuse" in {c["kind"] for c in out.summary["caveats"]}
    candidate = next(r for r in out.summary["comparison"]["test"]["strategies"] if r["strategy"] == "ppo")
    assert candidate["params"]["seeds"] == [0, 1, 2]


def test_partial_seed_grid_is_excluded(monkeypatch):
    import finplan_model.rl.train as learner

    p = tiny_protocol()
    p["rl"]["algorithms"] = ["ppo"]
    p["rl"]["policy_selection"] = "ensemble"
    calls = []
    original = learner.train_policy

    def first_only(*args, **kwargs):
        kwargs["deadline"] = None
        result = original(*args, **kwargs)
        calls.append(1)
        return result

    monkeypatch.setattr(learner, "train_policy", first_only)
    out = select_models(market(), daily_config(), protocol=p, deadline=1.0, clock=lambda: 2.0 if calls else 0.0)
    assert len(calls) == 1
    assert out.summary["rl"]["ppo"]["status"] == "not_trained_time_budget"
    assert "ppo" not in {r["strategy"] for r in out.summary["comparison"]["test"]["strategies"]}
