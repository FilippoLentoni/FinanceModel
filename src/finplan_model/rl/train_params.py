"""Hyperparameters PPO and SAC accept (pure Python: validated by the config gate and the control plane,
which never import the learners)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from finplan_model.core.errors import FinplanError

__all__ = ["ALGORITHM_PARAMS", "LOOP_PARAMS", "check_hyperparameters"]

#: Training-loop keys shared by both algorithms.
LOOP_PARAMS = frozenset({"total_timesteps", "eval_every", "patience"})
ALGORITHM_PARAMS: dict[str, frozenset[str]] = {
    "ppo": frozenset({"learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "net_arch"}) | LOOP_PARAMS,
    "sac": frozenset({"learning_rate", "buffer_size", "learning_starts", "batch_size", "tau", "gamma", "train_freq", "gradient_steps", "net_arch", "ent_coef"}) | LOOP_PARAMS,
}
_INTS = {"total_timesteps": (1, 10_000_000), "eval_every": (1, 10_000_000), "patience": (1, 1000), "n_steps": (2, 100_000), "batch_size": (2, 100_000), "n_epochs": (1, 1000), "buffer_size": (10, 10_000_000), "learning_starts": (0, 10_000_000), "train_freq": (1, 100_000), "gradient_steps": (1, 1000)}
_FLOATS = {"learning_rate": (1e-8, 1.0), "gamma": (0.0, 1.0), "gae_lambda": (0.0, 1.0), "clip_range": (1e-6, 10.0), "tau": (1e-6, 1.0)}


def check_hyperparameters(algorithm: str, hp: Mapping[str, Any]) -> None:
    if algorithm not in ALGORITHM_PARAMS:
        raise FinplanError.validation("unknown RL algorithm", pointer="/rl/algorithms")
    ptr = f"/rl/{algorithm}"
    unknown = sorted(set(hp) - ALGORITHM_PARAMS[algorithm])
    if unknown:
        raise FinplanError.validation(f"unknown {algorithm} hyperparameter", pointer=f"{ptr}/{unknown[0]}")
    if "total_timesteps" not in hp:
        raise FinplanError.validation("total_timesteps is required (it bounds the training run)", pointer=f"{ptr}/total_timesteps")
    for k, v in hp.items():
        if k in _INTS:
            lo, hi = _INTS[k]
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise FinplanError.validation(f"{k} must be an integer in [{lo}, {hi}]", pointer=f"{ptr}/{k}")
        elif k in _FLOATS:
            lo, hi = _FLOATS[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= float(v) <= hi:
                raise FinplanError.validation(f"{k} must be a number in [{lo}, {hi}]", pointer=f"{ptr}/{k}")
        elif k == "ent_coef":
            if not (v == "auto" or (isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= float(v) <= 10)):
                raise FinplanError.validation("ent_coef must be a non-negative number or 'auto'", pointer=f"{ptr}/{k}")
        elif k == "net_arch":
            if not isinstance(v, list) or not v or not all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 4096 for x in v):
                raise FinplanError.validation("net_arch lists hidden layer sizes", pointer=f"{ptr}/{k}")
    if algorithm == "ppo" and "batch_size" in hp and "n_steps" in hp and hp["batch_size"] > hp["n_steps"]:
        raise FinplanError.validation("ppo batch_size must not exceed n_steps", pointer=f"{ptr}/batch_size")
    if algorithm == "sac" and hp.get("learning_starts", 0) >= hp["total_timesteps"]:
        raise FinplanError.validation("sac learning_starts must be below total_timesteps", pointer=f"{ptr}/learning_starts")
