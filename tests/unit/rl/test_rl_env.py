"""RL-01 (versioned specification in the configuration_id; undeclared feature), RL-02 (environment and
common evaluator agree for the same decisions), RL-03 (softmax transform: long-only, weights sum to 1
with cash), the gymnasium API contract and deterministic policy evaluation. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.evaluate import evaluate
from finplan_model.rl.arrays import PriceArrays, calendar_schedule
from finplan_model.rl.env import PortfolioEnv, run_schedule
from finplan_model.rl.policy import RLPolicyStrategy
from finplan_model.rl.spec import (
    EnvSpec,
    build_observation,
    policy_configuration,
    softmax_weights,
)
from tests.unit.rl.support import market, sim_config


def _arrays():
    return PriceArrays.from_market(market())


# ----------------------------------------------------------------- RL-01
def test_rl01_reward_coefficient_change_gives_a_new_configuration_id():
    base = EnvSpec.from_dict({"window": 5})
    changed = base.with_reward(turnover_penalty=0.01)
    assert base.configuration_id != changed.configuration_id
    args = dict(simulation_configuration_id=sim_config().configuration_id, training_range={"start": "2025-01-01", "end": "2025-12-31"}, instruments=("A", "B"))
    from finplan_model.core.ids import configuration_id

    a = configuration_id(policy_configuration("ppo", {"total_timesteps": 10}, base, **args))
    b = configuration_id(policy_configuration("ppo", {"total_timesteps": 10}, changed, **args))
    c = configuration_id(policy_configuration("ppo", {"total_timesteps": 10}, base, **{**args, "training_range": {"start": "2025-01-01", "end": "2025-06-30"}}))
    assert len({a, b, c}) == 3  # reward and training range both change the policy configuration (RL-08)
    assert "reward_formula" in base.document() and base.document()["reward"]["risk_penalty"] == 1.0


def test_rl01_undeclared_or_unknown_features_fail_validation():
    with pytest.raises(FinplanError) as unknown:
        EnvSpec.from_dict({"features": ["log_return_window", "news_sentiment"]})
    assert unknown.value.code == "VALIDATION_FAILED"
    spec = EnvSpec.from_dict({"window": 3, "features": ["log_return_window"]})
    with pytest.raises(FinplanError) as undeclared:
        spec.require("current_weights")
    assert undeclared.value.code == "VALIDATION_FAILED" and undeclared.value.details["feature"] == "current_weights"
    # observation size follows the declared features only
    assert spec.observation_size(5) == 15 and EnvSpec.from_dict({"window": 3}).observation_size(5) == 21


def test_rl01_specification_fields_are_validated():
    for bad in ({"window": 0}, {"action_transform": "argmax"}, {"reward": {"risk_penalty": -1}}, {"reward": {"sharpe_bonus": 1}}, {"surprise": 1}):
        with pytest.raises(FinplanError) as exc:
            EnvSpec.from_dict(bad)
        assert exc.value.code == "VALIDATION_FAILED", bad


# ----------------------------------------------------------------- RL-03
@pytest.mark.parametrize("action", [np.zeros(6), np.ones(6), -np.ones(6), np.array([5.0, -3.0, 0.2, 0.0, 1.0, -9.0]), np.array([1, -1, -1, -1, -1, -1.0])])
def test_rl03_softmax_transform_gives_long_only_weights_summing_to_one_with_cash(action):
    w, cash = softmax_weights(action, 5, 5.0)
    assert np.all(w >= 0) and cash >= 0 and np.all(w <= 1.0)
    assert abs(w.sum() + cash - 1.0) < 1e-12


def test_rl03_wrong_action_size_or_non_finite_action_is_refused():
    for bad in (np.zeros(5), np.array([np.nan] * 6)):
        with pytest.raises(FinplanError):
            softmax_weights(bad, 5, 5.0)


def test_rl03_simulator_constraint_policy_projects_and_records():
    """A tighter max_weight than the transform guarantees is projected by the simulator and recorded."""
    cfg = sim_config()
    raw = cfg.to_dict()
    raw["constraints"] = {**raw["constraints"], "max_weight": 0.3}
    from finplan_model.sim.config import SimulationConfig

    tight = SimulationConfig.from_dict(raw)
    spec = EnvSpec.from_dict({"window": 5})
    strat = RLPolicyStrategy("sac", lambda obs: np.array([1, -1, -1, -1, -1, -1.0]), spec, seed=0, configuration_id="cfg_test")
    res = evaluate(strat, market(), tight, start=market().sessions[30], end=market().sessions[90])
    projected = [d for d in res.simulation.decisions if d.get("action") == "projected"]
    assert projected and all(max(d["constraints"]["final_weights"].values()) <= 0.3 + 1e-9 for d in projected)


# ----------------------------------------------------------------- RL-02
def test_rl02_environment_nav_matches_the_common_evaluator_for_the_same_policy():
    mkt, cfg = market(), sim_config()
    arrays = PriceArrays.from_market(mkt)
    spec = EnvSpec.from_dict({"window": 5, "decision_frequency": cfg.rebalance_frequency, "step_sessions": 21})
    rng = np.random.default_rng(0)
    bias = rng.uniform(-1, 1, size=6)

    def predict(obs: np.ndarray) -> np.ndarray:  # a scripted, state-dependent "policy"
        return np.tanh(bias + 0.1 * obs[:6])

    start, end = mkt.sessions[25], mkt.sessions[120]
    res = evaluate(RLPolicyStrategy("ppo", predict, spec, seed=0, configuration_id="cfg_test"), mkt, cfg, start=start, end=end)
    i0, i1 = arrays.window_indices(start, end)
    env = PortfolioEnv(arrays, spec, cfg, i0=i0, i1=i1, mode="eval")
    out = env.evaluate(predict)
    nav = [row["value"] for row in res.simulation.nav]
    assert len(nav) == len(out["values"])
    assert max(abs(a - b) for a, b in zip(nav, out["values"])) < 1e-6 * cfg.initial_cash
    assert out["sharpe"] == pytest.approx(res.metrics["sharpe_ratio"], rel=1e-9)
    assert res.simulation.summary["total_costs"] > 0  # costs were paid in both


def test_rl02_scripted_buy_and_hold_parity_with_costs_and_cash_interest():
    mkt = market()
    raw = sim_config().to_dict()
    raw["cash_rate_annual"] = 0.04
    from finplan_model.sim.config import SimulationConfig

    cfg = SimulationConfig.from_dict(raw)
    arrays = PriceArrays.from_market(mkt)
    spec = EnvSpec.from_dict({"window": 5})
    i0, i1 = 10, 100
    sched = calendar_schedule(arrays, i0, i1, "monthly")
    target = (np.array([0.3, 0.1, 0.2, 0.1, 0.1]), 0.2)
    values = run_schedule(arrays, cfg, spec, i0, i1, lambda book, k: target, sched)

    class Fixed:
        name = "fixed"

        def decide(self, view, holdings):
            return {"weights": dict(zip(arrays.instruments, target[0].tolist())), "cash": target[1]}

    res = evaluate(Fixed(), mkt, cfg, start=arrays.sessions[i0], end=arrays.sessions[i1])
    assert max(abs(a - b["value"]) for a, b in zip(values, res.simulation.nav)) < 1e-6 * cfg.initial_cash


def test_observation_is_point_in_time_and_matches_the_view():
    mkt = market()
    arrays = PriceArrays.from_market(mkt)
    spec = EnvSpec.from_dict({"window": 5})
    k = 40
    view = mkt.view(mkt.sessions[k])
    _, px = view.price_matrix("close", lookback=spec.window + 1)
    a = build_observation(spec, px, np.zeros(5), 1.0)
    b = build_observation(spec, arrays.close[k - spec.window : k + 1], np.zeros(5), 1.0)
    assert np.allclose(a, b) and a.dtype == np.float32 and a.shape == (spec.observation_size(5),)
    assert np.all(np.abs(a) <= spec.obs_clip)


def test_policy_holds_without_enough_history():
    mkt = market()
    spec = EnvSpec.from_dict({"window": 20})
    strat = RLPolicyStrategy("ppo", lambda obs: np.zeros(6), spec, seed=0, configuration_id="cfg_test")
    res = evaluate(strat, mkt, sim_config(), start=mkt.sessions[0], end=mkt.sessions[60])
    first = res.simulation.decisions[0]
    assert first["solution_status"] == "no_effect" and first["strategy_output"]["diagnostics"]["reason"] == "insufficient_history"
    assert any(d["solution_status"] == "feasible" for d in res.simulation.decisions[1:])


# ----------------------------------------------------------------- gymnasium contract
def test_environment_follows_the_gymnasium_api_and_reports_reward_terms():
    from gymnasium.utils.env_checker import check_env

    arrays = _arrays()
    spec = EnvSpec.from_dict({"window": 5, "step_sessions": 5})
    env = PortfolioEnv(arrays, spec, sim_config(), i0=0, i1=80, mode="train")
    check_env(env, skip_render_check=True)
    env.reset(seed=3)
    total, steps, done, info = 0.0, 0, False, {}
    while not done:
        _, r, done, _, info = env.step(env.action_space.sample())
        total += r
        steps += 1
        assert set(info["reward_terms"]) == {"log_return", "sum_sq_log_return", "drawdown_increase", "turnover"}
    assert info["episode_reward"] == pytest.approx(total) and 10 <= steps <= 16


def test_training_episodes_start_at_a_seeded_random_offset():
    arrays = _arrays()
    spec = EnvSpec.from_dict({"window": 5, "step_sessions": 7})
    env = PortfolioEnv(arrays, spec, sim_config(), i0=0, i1=80, mode="train")
    firsts = set()
    for seed in range(12):
        env.reset(seed=seed)
        firsts.add(env._schedule[0])
    assert len(firsts) > 1 and min(firsts) >= 5 and max(firsts) < 12
    env.reset(seed=4)
    a = list(env._schedule)
    env.reset(seed=4)
    assert env._schedule == a


def test_reward_penalizes_risk_and_drawdown_as_declared():
    arrays = _arrays()
    cfg = sim_config()
    plain = EnvSpec.from_dict({"window": 5, "step_sessions": 10, "reward": {"risk_penalty": 0.0, "drawdown_penalty": 0.0, "reward_scale": 1.0}})
    risky = plain.with_reward(risk_penalty=10.0)
    act = np.array([1, 1, 1, 1, 1, -1.0])
    rewards = {}
    for name, spec in (("plain", plain), ("risky", risky)):
        env = PortfolioEnv(arrays, spec, cfg, i0=0, i1=60, mode="train")
        env.reset(seed=1)
        _, r, _, _, info = env.step(act)
        rewards[name] = (r, info["reward_terms"])
    r0, t0 = rewards["plain"]
    r1, t1 = rewards["risky"]
    assert r0 == pytest.approx(t0["log_return"])
    assert r1 == pytest.approx(t1["log_return"] - 10.0 * t1["sum_sq_log_return"]) and r1 < r0


def test_documented_default_specification_validates_and_is_the_configured_one():
    """Task 1.4: docs/rl-environment.md documents the specification the configuration uses."""
    import json
    import re

    from finplan_model.core.config import load_config
    from tests.unit.rl.support import ROOT

    text = (ROOT / "docs" / "rl-environment.md").read_text(encoding="utf-8")
    assert "configuration pending user review" in text
    (block,) = re.findall(r"```json\n(.*?)\n```", text, re.S)
    documented = EnvSpec.from_dict(json.loads(block))
    configured = EnvSpec.from_dict(load_config("beta").raw["job_types"]["model_selection"]["protocol"]["rl"]["env"])
    assert configured == documented
    for env in ("gamma", "prod"):
        legacy = EnvSpec.from_dict(load_config(env).raw["job_types"]["model_selection"]["protocol"]["rl"]["env"])
        assert legacy.window == 20 and legacy.features == ("log_return_window", "current_weights")
        assert legacy.episode_sessions == 0 and legacy.rebalance_fraction == 1
