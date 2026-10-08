"""The versioned RL environment specification (spec rl-strategies, "Versioned state, action and reward
specification"; RL-01, RL-03).

Everything that defines what a policy sees, does and is rewarded for is declared here and enters the
policy's ``configuration_id`` (:func:`policy_configuration`): a changed coefficient, window or
transform is a new configuration, and a policy trained under one configuration is never reused for
another (RL-08).

State (point in time, at the close of the decision session ``d``)
    ``log_return_window``: the trailing ``window`` daily log close-to-close returns of every
    instrument (``window x n``, oldest first), multiplied by ``return_scale`` and clipped to
    ``[-obs_clip, obs_clip]``; ``current_weights``: the current weight of every instrument plus the
    cash weight (``n + 1``, marked at the decision close). Observation size ``window * n + n + 1``.
    Reading any other feature fails with ``VALIDATION_FAILED`` (:meth:`EnvSpec.require`).

Action and transform (``softmax``)
    a real vector ``a`` in ``[-1, 1]^(n+1)``; target weights ``w = softmax(action_scale * a)`` over
    the ``n`` instruments plus cash, so weights are long-only, at most 1 and sum to 1 with cash. The
    simulator's constraint policy is applied afterwards (and records any projection).

Reward (per decision step ``t``, from the decision close ``d_t`` to the next decision close ``d_{t+1}``)::

    reward_t = reward_scale * ( ln(V[d_{t+1}] / V[d_t])                       # net-of-cost log return
                                - risk_penalty      * sum_{s in (d_t, d_{t+1}]} ln(V[s] / V[s-1])^2
                                - drawdown_penalty  * max(0, DD[d_{t+1}] - DD[d_t])
                                - turnover_penalty  * turnover_t )

    V = simulated net asset value (fees, spread and slippage paid), DD[s] = 1 - V[s] / max_{u <= s} V[u]
    within the episode, turnover_t = traded notional at the execution session / value at the decision close.

The reward is a training signal only: it is reported in the training-reward section and never as
portfolio performance (RL-06); portfolio metrics come from the common evaluator.

Episode
    training: one pass over the training window, all cash at the start, first decision at a random
    offset in ``[window, window + step_sessions)`` sessions into the window (seeded), then a decision
    every ``step_sessions`` sessions (about one month); evaluation (validation checkpoints): the
    calendar rebalance sessions of the window (``decision_frequency``), exactly as the common evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import configuration_id

__all__ = [
    "ENV_SPEC_VERSION",
    "STATE_FEATURES",
    "EnvSpec",
    "RewardSpec",
    "build_observation",
    "policy_configuration",
    "softmax_weights",
]

ENV_SPEC_VERSION = "finplan-rl-env/1"
#: Features the environment implements; a specification may declare only these.
STATE_FEATURES = ("log_return_window", "current_weights")
ACTION_TRANSFORMS = ("softmax",)
REWARD_FORMULA = (
    "reward_t = reward_scale * (ln(V[d_t+1]/V[d_t]) - risk_penalty * sum_s ln(V[s]/V[s-1])^2 "
    "- drawdown_penalty * max(0, DD[d_t+1] - DD[d_t]) - turnover_penalty * turnover_t)"
)


def _num(d: Mapping[str, Any], key: str, default: float, *, lo: float, hi: float, ptr: str) -> float:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= float(v) <= hi:
        raise FinplanError.validation(f"environment specification {key} must be a number in [{lo}, {hi}]", pointer=f"{ptr}/{key}")
    return float(v)


def _int(d: Mapping[str, Any], key: str, default: int, *, lo: int, hi: int, ptr: str) -> int:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise FinplanError.validation(f"environment specification {key} must be an integer in [{lo}, {hi}]", pointer=f"{ptr}/{key}")
    return v


@dataclass(frozen=True)
class RewardSpec:
    risk_penalty: float = 1.0
    drawdown_penalty: float = 0.5
    turnover_penalty: float = 0.0
    reward_scale: float = 100.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None, ptr: str = "/reward") -> RewardSpec:
        d = dict(d or {})
        unknown = sorted(set(d) - set(cls.__dataclass_fields__))
        if unknown:
            raise FinplanError.validation("unknown reward term", pointer=f"{ptr}/{unknown[0]}")
        return cls(
            risk_penalty=_num(d, "risk_penalty", 1.0, lo=0.0, hi=1e4, ptr=ptr),
            drawdown_penalty=_num(d, "drawdown_penalty", 0.5, lo=0.0, hi=1e4, ptr=ptr),
            turnover_penalty=_num(d, "turnover_penalty", 0.0, lo=0.0, hi=1e4, ptr=ptr),
            reward_scale=_num(d, "reward_scale", 100.0, lo=1e-6, hi=1e6, ptr=ptr),
        )


@dataclass(frozen=True)
class EnvSpec:
    window: int = 20
    features: tuple[str, ...] = STATE_FEATURES
    return_scale: float = 50.0
    obs_clip: float = 5.0
    action_transform: str = "softmax"
    action_scale: float = 5.0
    step_sessions: int = 1
    decision_frequency: str = "daily"
    reward: RewardSpec = field(default_factory=RewardSpec)
    version: str = ENV_SPEC_VERSION

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None, ptr: str = "/rl/env") -> EnvSpec:
        d = dict(d or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise FinplanError.validation("unknown environment specification field", pointer=f"{ptr}/{unknown[0]}")
        if d.get("version", ENV_SPEC_VERSION) != ENV_SPEC_VERSION:
            raise FinplanError.validation("unsupported environment specification version", pointer=f"{ptr}/version")
        features = tuple(d.get("features", STATE_FEATURES))
        bad = [f for f in features if f not in STATE_FEATURES]
        if bad or not features or len(set(features)) != len(features):
            raise FinplanError.validation("state features must be distinct implemented features", pointer=f"{ptr}/features", known=list(STATE_FEATURES))
        transform = d.get("action_transform", "softmax")
        if transform not in ACTION_TRANSFORMS:
            raise FinplanError.validation("PPO and SAC map actions to weights by a declared continuous transform (softmax)", pointer=f"{ptr}/action_transform")
        freq = d.get("decision_frequency", "daily")
        if freq not in ("daily", "weekly", "monthly", "quarterly"):
            raise FinplanError.validation("decision_frequency must be a rebalance frequency", pointer=f"{ptr}/decision_frequency")
        return cls(
            window=_int(d, "window", 20, lo=2, hi=252, ptr=ptr),
            features=features,
            return_scale=_num(d, "return_scale", 50.0, lo=1e-6, hi=1e6, ptr=ptr),
            obs_clip=_num(d, "obs_clip", 5.0, lo=0.1, hi=1e6, ptr=ptr),
            action_transform=str(transform),
            action_scale=_num(d, "action_scale", 5.0, lo=0.1, hi=50.0, ptr=ptr),
            step_sessions=_int(d, "step_sessions", 1, lo=1, hi=252, ptr=ptr),
            decision_frequency=str(freq),
            reward=RewardSpec.from_dict(d.get("reward"), ptr=f"{ptr}/reward"),
        )

    def with_reward(self, **changes: float) -> EnvSpec:
        return EnvSpec(**{**self._fields(), "reward": RewardSpec(**{**asdict(self.reward), **changes})})

    def _fields(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def document(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = list(self.features)
        d["reward_formula"] = REWARD_FORMULA
        return d

    @property
    def configuration_id(self) -> str:
        return configuration_id(self.document())

    def require(self, feature: str) -> None:
        """Every feature the environment reads must be declared (RL-01 "Undeclared feature")."""
        if feature not in self.features:
            raise FinplanError.validation("the environment read a feature that the state specification does not declare", pointer="/rl/env/features", feature=feature)

    def observation_size(self, n_instruments: int) -> int:
        size = 0
        if "log_return_window" in self.features:
            size += self.window * n_instruments
        if "current_weights" in self.features:
            size += n_instruments + 1
        return size


def softmax_weights(action: Any, n_instruments: int, action_scale: float) -> tuple[np.ndarray, float]:
    """The declared transform (RL-03): ``softmax(action_scale * a)`` over the instruments plus cash."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != n_instruments + 1 or not np.all(np.isfinite(a)):
        raise FinplanError.validation("the policy action has the wrong size or non-finite values", pointer="/action", expected=n_instruments + 1)
    z = action_scale * np.clip(a, -1.0, 1.0)
    z = z - z.max()
    e = np.exp(z)
    w = e / e.sum()
    weights = w[:n_instruments]
    cash = float(max(0.0, 1.0 - float(weights.sum())))
    return weights, cash


def build_observation(spec: EnvSpec, closes: np.ndarray, weights: np.ndarray, cash_weight: float) -> np.ndarray:
    """Observation at a decision from ``window + 1`` aligned closes (oldest first) and current weights."""
    parts: list[np.ndarray] = []
    if "log_return_window" in spec.features:
        spec.require("log_return_window")
        px = np.asarray(closes, dtype=np.float64)
        if px.shape[0] != spec.window + 1:
            raise FinplanError.validation("the observation needs window + 1 closes", pointer="/rl/env/window", window=spec.window)
        r = np.log(px[1:] / px[:-1]) * spec.return_scale
        parts.append(np.clip(r, -spec.obs_clip, spec.obs_clip).reshape(-1))
    if "current_weights" in spec.features:
        spec.require("current_weights")
        parts.append(np.append(np.asarray(weights, dtype=np.float64), float(cash_weight)))
    return np.concatenate(parts).astype(np.float32)


def policy_configuration(algo: str, hyperparameters: Mapping[str, Any], env_spec: EnvSpec, *, simulation_configuration_id: str, training_range: Mapping[str, str], instruments: tuple[str, ...]) -> dict[str, Any]:
    """The canonical configuration of one trained policy (its ``configuration_id`` covers all of it)."""
    return {
        "strategy": algo,
        "family": "reinforcement_learning",
        "algorithm": algo,
        "hyperparameters": {k: hyperparameters[k] for k in sorted(hyperparameters)},
        "environment": env_spec.document(),
        "simulation_configuration_id": simulation_configuration_id,
        "training_range": dict(training_range),
        "instruments": list(instruments),
    }
