"""PPO and SAC on tiny synthetic data (fixture step counts only): seeded and reproducible, deterministic
evaluation of the stored policy, validation checkpoint selection, the time budget, the shaped reward
kept apart from portfolio metrics (RL-06) and the build-stage learner guard (RL-07)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.rl.arrays import PriceArrays
from finplan_model.rl.spec import EnvSpec
from finplan_model.rl.train import (
    STEP_LIMIT_ENV,
    check_step_limit,
    load_predictor,
    train_policy,
)
from finplan_model.rl.train_params import check_hyperparameters
from tests.unit.rl.support import market, sim_config, tiny_protocol

SPEC = EnvSpec.from_dict({"window": 5, "step_sessions": 5})


def _train(algo: str, seed: int, **over):
    arrays = PriceArrays.from_market(market())
    hp = {**tiny_protocol()["rl"][algo], **over}
    return train_policy(arrays, SPEC, sim_config(), algorithm=algo, seed=seed, hyperparameters=hp, train_window=(0, 84), validation_window=(85, 126), configuration_id="cfg_" + "0" * 64)


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_training_is_seeded_and_the_stored_policy_evaluates_deterministically(algo):
    a = _train(algo, 1)
    b = _train(algo, 1)
    assert a.checksum.startswith("sha256:") and a.model_bytes[:2] == b"PK"  # SB3 zip
    assert a.validation_curve == b.validation_curve  # same seed, same learning path
    assert a.timesteps_trained >= 1 and a.stopped_reason in ("completed", "early_stopping")
    assert a.validation_curve[0]["step"] == 0  # the untrained policy is the step-0 checkpoint
    assert a.best_step in [c["step"] for c in a.validation_curve]
    predict = load_predictor(algo, a.model_bytes)
    obs = np.linspace(-1, 1, SPEC.observation_size(5)).astype(np.float32)
    assert np.array_equal(predict(obs), predict(obs))  # deterministic=True
    assert np.array_equal(predict(obs), load_predictor(algo, b.model_bytes)(obs))
    tr = a.training_reward
    assert "not portfolio performance" in tr["note"] and tr["episodes"] >= 1
    assert not {"total_return", "sharpe", "max_drawdown"} & set(tr)  # RL-06: reward section only
    if algo == "ppo":
        assert tr["optimizer_updates"] and "explained_variance" in tr["optimizer_updates"][0]
        assert tr["checkpoint_improved_over_initial"] == (a.best_step > 0)


def test_different_seeds_give_different_policies():
    obs = np.linspace(-1, 1, SPEC.observation_size(5)).astype(np.float32)
    a, b = (load_predictor("ppo", _train("ppo", s).model_bytes)(obs) for s in (0, 3))
    assert not np.allclose(a, b)


def test_rl07_learner_runs_beyond_the_fixture_step_count_are_refused_in_the_build(monkeypatch):
    import os

    assert os.environ.get(STEP_LIMIT_ENV) == "512"  # set by the offline harness for every test
    with pytest.raises(FinplanError) as exc:
        _train("ppo", 0, total_timesteps=100_000)  # a planted "long training" test
    assert exc.value.code == "OPERATION_NOT_PERMITTED"
    check_step_limit(512)
    monkeypatch.delenv(STEP_LIMIT_ENV)
    check_step_limit(10_000_000)  # outside the build (the SageMaker training job) there is no limit


def test_deadline_stops_training_and_keeps_the_best_checkpoint():
    arrays = PriceArrays.from_market(market())
    hp = {**tiny_protocol()["rl"]["ppo"], "total_timesteps": 256}
    out = train_policy(arrays, SPEC, sim_config(), algorithm="ppo", seed=0, hyperparameters=hp, train_window=(0, 84), validation_window=(85, 126), configuration_id="cfg_" + "0" * 64, deadline=time.monotonic() - 1)
    assert out.stopped_reason == "time_budget" and out.timesteps_trained < 256 and out.validation_curve


def test_hyperparameters_are_validated_without_the_learners():
    check_hyperparameters("sac", tiny_protocol()["rl"]["sac"])
    for algo, bad in (("ppo", {"total_timesteps": 10, "policy": "CnnPolicy"}), ("ppo", {"total_timesteps": 0}), ("sac", {"total_timesteps": 10, "learning_starts": 10}), ("ppo", {"n_steps": 8}), ("dqn", {"total_timesteps": 1})):
        with pytest.raises(FinplanError) as exc:
            check_hyperparameters(algo, bad)
        assert exc.value.code == "VALIDATION_FAILED", (algo, bad)
