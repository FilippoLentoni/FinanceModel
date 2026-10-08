"""PPO and SAC training with validation checkpoint selection (spec rl-strategies; tasks 2.1-2.5).

:func:`train_policy` trains one ``(algorithm, configuration, seed)`` with stable-baselines3 on CPU
torch (one thread, seeded) in the training window of :class:`~finplan_model.rl.env.PortfolioEnv`, and
every ``eval_every`` steps runs the **deterministic** policy over the validation window with the
evaluator's calendar rebalance schedule. The checkpoint with the best validation metric is kept
(early stopping after ``patience`` evaluations without improvement); the untrained policy is the
step-0 checkpoint, so a policy that never beats it says so. The training window and the validation
window never overlap; the test window is never touched here.

Guards:

* ``FINPLAN_LEARNER_STEP_LIMIT`` (set by the offline test harness, so the build stage's unit suite):
  a learner run asking for more steps is refused with ``OPERATION_NOT_PERMITTED`` (RL-07: no
  training in CodeBuild beyond the fixture step count);
* ``deadline`` (``time.monotonic()`` value): training stops at the deadline and keeps the best
  checkpoint so far (the job's runtime budget), recorded as ``stopped_reason: time_budget``.

The returned :class:`TrainedPolicy` carries the policy bytes (SB3 zip), their SHA-256 checksum, the
seed, the policy ``configuration_id`` and the training-reward statistics (reported separately from
portfolio metrics, RL-06).
"""

from __future__ import annotations

import copy
import io
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from finplan_model.core.artifacts import sha256_checksum
from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.sim.config import SimulationConfig

from .arrays import PriceArrays, calendar_schedule
from .env import PortfolioEnv
from .spec import EnvSpec
from .train_params import ALGORITHM_PARAMS, LOOP_PARAMS, check_hyperparameters

__all__ = ["ALGORITHM_PARAMS", "STEP_LIMIT_ENV", "TrainedPolicy", "check_step_limit", "load_predictor", "train_policy"]

STEP_LIMIT_ENV = "FINPLAN_LEARNER_STEP_LIMIT"
SELECTION_METRICS = ("sharpe", "net_return")


@dataclass
class TrainedPolicy:
    algorithm: str
    seed: int
    configuration_id: str
    model_bytes: bytes
    checksum: str
    best_step: int
    timesteps_trained: int
    stopped_reason: str
    validation_curve: list[dict[str, Any]]
    training_reward: dict[str, Any]
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "configuration_id": self.configuration_id,
            "checksum": self.checksum,
            "best_step": self.best_step,
            "timesteps_trained": self.timesteps_trained,
            "stopped_reason": self.stopped_reason,
            "seconds": round(self.seconds, 3),
        }


def check_step_limit(total_timesteps: int) -> None:
    raw = os.environ.get(STEP_LIMIT_ENV)
    if raw is None or raw == "":
        return
    limit = int(raw)
    if int(total_timesteps) > limit:
        raise FinplanError(ErrorCode.OPERATION_NOT_PERMITTED, "learner runs beyond the fixture step count are not allowed outside the SageMaker training job", details={"total_timesteps": int(total_timesteps), "limit": limit})


def _algo(name: str) -> Any:
    from stable_baselines3 import PPO, SAC

    return {"ppo": PPO, "sac": SAC}[name]


def _kwargs(algo: str, hp: Mapping[str, Any]) -> dict[str, Any]:
    check_hyperparameters(algo, hp)
    kw = {k: v for k, v in hp.items() if k not in LOOP_PARAMS and k != "net_arch"}
    kw["policy_kwargs"] = {"net_arch": list(hp.get("net_arch", [64, 64]))}
    if algo == "sac" and "train_freq" in kw:
        kw["train_freq"] = int(kw["train_freq"])
    return kw


def _metric(res: Mapping[str, Any], metric: str) -> float:
    v = res.get(metric)
    return -math.inf if v is None or not math.isfinite(float(v)) else float(v)


def load_predictor(algorithm: str, model_bytes: bytes) -> Callable[[np.ndarray], np.ndarray]:
    """Deterministic ``obs -> action`` from stored policy bytes (``deterministic=True``)."""
    import torch

    torch.set_num_threads(1)
    model = _algo(algorithm).load(io.BytesIO(model_bytes), device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def train_policy(
    arrays: PriceArrays,
    spec: EnvSpec,
    config: SimulationConfig,
    *,
    algorithm: str,
    seed: int,
    hyperparameters: Mapping[str, Any],
    train_window: tuple[int, int],
    validation_window: tuple[int, int],
    configuration_id: str,
    metric: str = "sharpe",
    deadline: float | None = None,
) -> TrainedPolicy:
    if algorithm not in ALGORITHM_PARAMS:
        raise FinplanError.validation("unknown RL algorithm", pointer="/rl/algorithms", algorithm=algorithm)
    if metric not in SELECTION_METRICS:
        raise FinplanError.validation("unknown validation metric", pointer="/selection/metric")
    hp = dict(hyperparameters)
    total = int(hp.get("total_timesteps", 10000))
    check_step_limit(total)
    eval_every = max(1, int(hp.get("eval_every", 1000)))
    patience = max(1, int(hp.get("patience", 5)))
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)
    started = time.monotonic()
    t0, t1 = train_window
    v0, v1 = validation_window
    train_env = PortfolioEnv(arrays, spec, config, i0=t0, i1=t1, mode="train")
    val_env = PortfolioEnv(arrays, spec, config, i0=v0, i1=v1, mode="eval", schedule=calendar_schedule(arrays, v0, v1, spec.decision_frequency))
    model = _algo(algorithm)("MlpPolicy", train_env, seed=int(seed), device="cpu", verbose=0, **_kwargs(algorithm, hp))

    class Select(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.curve: list[dict[str, Any]] = []
            self.best = -math.inf
            self.best_step = 0
            self.best_state: Any = None
            self.since = 0
            self.last_eval = -1
            self.stopped = "completed"
            self.episode_rewards: list[float] = []
            self.updates: list[dict[str, Any]] = []

        def _on_rollout_start(self) -> None:
            # SB3 exposes the previous optimizer update here, before the next rollout.
            keys = ("explained_variance", "approx_kl", "clip_fraction", "entropy_loss", "value_loss", "policy_gradient_loss", "std", "n_updates")
            row = {key: float(self.model.logger.name_to_value["train/" + key])
                   for key in keys if "train/" + key in self.model.logger.name_to_value}
            if row and len(self.updates) < 2000:
                self.updates.append({"step": int(self.num_timesteps), **{k: v if math.isfinite(v) else None for k, v in row.items()}})

        def _evaluate(self) -> bool:
            res = val_env.evaluate(lambda obs: self.model.predict(obs, deterministic=True)[0])
            score = _metric(res, metric)
            self.curve.append({"step": int(self.num_timesteps), "validation_" + metric: None if not math.isfinite(score) else round(score, 6), "validation_net_return": round(float(res["net_return"]), 6)})
            self.last_eval = self.num_timesteps
            if score > self.best + 1e-12 or self.best_state is None:
                self.best, self.best_step, self.since = score, int(self.num_timesteps), 0
                self.best_state = copy.deepcopy(self.model.policy.state_dict())
                return True
            self.since += 1
            return self.since < patience

        def _on_training_start(self) -> None:
            self._evaluate()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos") or []:
                if "episode_reward" in info:
                    self.episode_rewards.append(float(info["episode_reward"]))
            if deadline is not None and time.monotonic() > deadline:
                self.stopped = "time_budget"
                return False
            if self.num_timesteps - max(self.last_eval, 0) >= eval_every and not self._evaluate():
                self.stopped = "early_stopping"
                return False
            return True

    cb = Select()
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    if cb.last_eval != model.num_timesteps and cb.stopped != "time_budget":
        cb._evaluate()
    if cb.best_state is not None:
        model.policy.load_state_dict(cb.best_state)
    buf = io.BytesIO()
    model.save(buf)
    data = buf.getvalue()
    rewards = cb.episode_rewards
    k = max(1, len(rewards) // 10)
    training_reward = {
        "episodes": len(rewards),
        "mean_episode_reward": None if not rewards else round(float(np.mean(rewards)), 6),
        "first_decile_mean_episode_reward": None if not rewards else round(float(np.mean(rewards[:k])), 6),
        "last_decile_mean_episode_reward": None if not rewards else round(float(np.mean(rewards[-k:])), 6),
        "best_validation_" + metric: None if not math.isfinite(cb.best) else round(cb.best, 6),
        "note": "shaped training reward (reward_scale x (log return - risk, drawdown and turnover penalties)); not portfolio performance",
        "optimizer_updates": cb.updates,
        "checkpoint_improved_over_initial": cb.best_step > 0,
    }
    return TrainedPolicy(
        algorithm=algorithm,
        seed=int(seed),
        configuration_id=configuration_id,
        model_bytes=data,
        checksum=sha256_checksum(data),
        best_step=cb.best_step,
        timesteps_trained=int(model.num_timesteps),
        stopped_reason=cb.stopped,
        validation_curve=cb.curve,
        training_reward=training_reward,
        seconds=time.monotonic() - started,
    )
