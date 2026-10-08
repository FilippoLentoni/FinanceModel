"""The model-selection protocol (decision 27, 2026-10-08; ``docs/model-selection.md``).

A protocol is configuration (``config/<env>.json`` ``job_types.model_selection.protocol``), validated
at build time (config gate) and frozen into the run at submission, so a run is reproducible from its
spec alone. Default (decision 27): **only 2025-2026 data**; train/calibrate 2025-01-01..2025-12-31,
validate 2026-01-01..2026-06-30, untouched test 2026-07-01..latest session, evaluated once at the end.

Families, all through the common evaluator with the same costs, monthly rebalance, long-only weights
of at most 1 and the same universe plus cash:

* controls: ``cash``, ``buy_and_hold``, ``equal_weight`` (no parameters);
* traditional: ``min_variance``, ``mean_variance``, ``scenario_cvar`` (grid over lookback and risk
  aversion / CVaR level, chosen on validation);
* RL: ``ppo``, ``sac`` (grid over the reward risk penalty, every seed of the seed list; checkpoints,
  configuration and seed chosen on validation).

Selection rule (:func:`validate_selection_rule`): a declared metric (``sharpe`` or ``net_return``,
net of costs) on the **validation** split. A rule naming the test split or a holdout fails with
``OPERATION_NOT_PERMITTED`` (RL-05 "Selection on test data attempted").
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from datetime import date
from typing import Any

from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.ids import configuration_id

__all__ = [
    "CONTROL_NAMES",
    "DEFAULT_PROTOCOL",
    "PROTOCOL_VERSION",
    "TRADITIONAL_NAMES",
    "grid_points",
    "protocol_id",
    "validate_protocol",
    "validate_selection_rule",
]

PROTOCOL_VERSION = "finplan-model-selection/1"
CONTROL_NAMES = ("cash", "buy_and_hold", "equal_weight")
TRADITIONAL_NAMES = ("min_variance", "mean_variance", "scenario_cvar")
RL_NAMES = ("ppo", "sac")
SELECTION_METRICS = ("sharpe", "net_return")
SPLITS = ("train", "validation", "test")
_FORBIDDEN_SELECTION_WORDS = ("test", "holdout")

DEFAULT_PROTOCOL: dict[str, Any] = {
    "protocol_version": PROTOCOL_VERSION,
    "data_start": "2025-01-01",
    "splits": {
        "train": {"start": "2025-01-01", "end": "2025-12-31"},
        "validation": {"start": "2026-01-01", "end": "2026-06-30"},
        "test": {"start": "2026-07-01", "end": None},
    },
    "selection": {"metric": "sharpe", "split": "validation"},
    "traditional": {
        "min_variance": {"grid": {"lookback": [20, 60, 120]}, "fixed": {}},
        "mean_variance": {"grid": {"lookback": [20, 60, 120], "risk_aversion": [1.0, 5.0, 20.0]}, "fixed": {}},
        "scenario_cvar": {"grid": {"lookback": [60, 120], "alpha": [0.9, 0.95]}, "fixed": {"n_scenarios": 500, "scenario_method": "bootstrap", "seed": 0}},
    },
    "rl": {
        "algorithms": ["ppo", "sac"],
        "seeds": [0, 1, 2, 3, 4],
        "env": {"window": 20, "return_scale": 50.0, "obs_clip": 5.0, "action_scale": 5.0, "step_sessions": 21, "decision_frequency": "monthly", "reward": {"risk_penalty": 1.0, "drawdown_penalty": 0.5, "turnover_penalty": 0.0, "reward_scale": 100.0}},
        "grid": {"risk_penalty": [0.5, 2.0]},
        "ppo": {"total_timesteps": 30000, "learning_rate": 0.0003, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.9, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "net_arch": [64, 64], "eval_every": 2048, "patience": 5},
        "sac": {"total_timesteps": 8000, "learning_rate": 0.0003, "buffer_size": 50000, "learning_starts": 500, "batch_size": 128, "tau": 0.005, "gamma": 0.9, "train_freq": 1, "gradient_steps": 1, "net_arch": [64, 64], "eval_every": 1000, "patience": 4},
    },
    "runtime": {"reserve_seconds": 600},
    "incumbent_fallback": "buy_and_hold",
}


def _date(v: Any, ptr: str, *, allow_none: bool = False) -> date | None:
    if v is None and allow_none:
        return None
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        raise FinplanError.validation("protocol dates are ISO dates", pointer=ptr) from None


def validate_selection_rule(rule: Mapping[str, Any] | None) -> dict[str, str]:
    """Validation-only selection; any reference to the test split or a holdout is refused."""
    rule = dict(rule or {})
    text = " ".join(str(v).lower() for v in rule.values())
    if any(w in text for w in _FORBIDDEN_SELECTION_WORDS) or any(w in str(k).lower() for k in rule for w in _FORBIDDEN_SELECTION_WORDS):
        raise FinplanError(ErrorCode.OPERATION_NOT_PERMITTED, "selection rules may use validation metrics only; test and holdout metrics are never used for selection", details={"pointer": "/selection"})
    unknown = sorted(set(rule) - {"metric", "split"})
    if unknown:
        raise FinplanError.validation("unknown selection rule field", pointer=f"/selection/{unknown[0]}")
    metric = str(rule.get("metric", "sharpe"))
    split = str(rule.get("split", "validation"))
    if split != "validation":
        raise FinplanError(ErrorCode.OPERATION_NOT_PERMITTED, "selection uses the validation split only", details={"pointer": "/selection/split"})
    if metric not in SELECTION_METRICS:
        raise FinplanError.validation("selection metric must be sharpe or net_return (net of costs)", pointer="/selection/metric")
    return {"metric": metric, "split": split}


def grid_points(grid: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product in declared key order (the tie-break order)."""
    points: list[dict[str, Any]] = [{}]
    for key, values in grid.items():
        points = [{**p, key: v} for p in points for v in values]
    return points


def _check_grid(grid: Any, ptr: str) -> None:
    if not isinstance(grid, Mapping) or not grid:
        raise FinplanError.validation("a grid maps parameter names to non-empty value lists", pointer=ptr)
    for k, vals in grid.items():
        if not isinstance(vals, list) or not vals:
            raise FinplanError.validation("grid values must be non-empty lists", pointer=f"{ptr}/{k}")


def validate_protocol(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """The protocol with defaults applied; ``VALIDATION_FAILED`` (or ``OPERATION_NOT_PERMITTED`` for a
    test-referencing selection rule) otherwise. Pure Python: imports no learner."""
    p = copy.deepcopy(dict(raw)) if raw is not None else copy.deepcopy(DEFAULT_PROTOCOL)
    if p.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise FinplanError.validation("unsupported model-selection protocol version", pointer="/protocol_version")
    p["protocol_version"] = PROTOCOL_VERSION
    unknown = sorted(set(p) - set(DEFAULT_PROTOCOL))
    if unknown:
        raise FinplanError.validation("unknown protocol field", pointer=f"/{unknown[0]}")
    for key in DEFAULT_PROTOCOL:
        p.setdefault(key, copy.deepcopy(DEFAULT_PROTOCOL[key]))
    data_start = _date(p["data_start"], "/data_start")
    splits = p["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise FinplanError.validation("splits are exactly train, validation and test", pointer="/splits")
    prev_end: date | None = None
    for name in SPLITS:
        s = splits[name]
        start = _date(s.get("start"), f"/splits/{name}/start")
        end = _date(s.get("end"), f"/splits/{name}/end", allow_none=name == "test")
        assert start is not None and data_start is not None
        if start < data_start:
            raise FinplanError.validation("splits start on or after data_start", pointer=f"/splits/{name}/start")
        if end is not None and end < start:
            raise FinplanError.validation("a split ends before it starts", pointer=f"/splits/{name}/end")
        if prev_end is not None and start <= prev_end:
            raise FinplanError.validation("splits are chronological and do not overlap (train < validation < test)", pointer=f"/splits/{name}/start")
        prev_end = end
    p["selection"] = validate_selection_rule(p["selection"])
    trad = p["traditional"]
    if not isinstance(trad, Mapping) or not set(trad) <= set(TRADITIONAL_NAMES):
        raise FinplanError.validation("traditional optimizers are min_variance, mean_variance and scenario_cvar", pointer="/traditional")
    from finplan_model.strategies import build_strategy

    for name, block in trad.items():
        _check_grid(block.get("grid"), f"/traditional/{name}/grid")
        for point in grid_points(block["grid"]):
            build_strategy(name, {**dict(block.get("fixed") or {}), **point}, pointer=f"/traditional/{name}")
    rl = p["rl"]
    algos = list(rl.get("algorithms") or [])
    if not algos or not set(algos) <= set(RL_NAMES) or len(set(algos)) != len(algos):
        raise FinplanError.validation("rl.algorithms lists ppo and/or sac", pointer="/rl/algorithms")
    seeds = rl.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds):
        raise FinplanError.validation("rl.seeds lists at least 3 distinct non-negative integer seeds", pointer="/rl/seeds")
    from finplan_model.rl.spec import EnvSpec, RewardSpec

    env = EnvSpec.from_dict(rl.get("env"))
    grid = rl.get("grid") or {}
    if grid:
        _check_grid(grid, "/rl/grid")
        bad = sorted(set(grid) - set(RewardSpec.__dataclass_fields__))
        if bad:
            raise FinplanError.validation("the RL grid ranges over reward coefficients only", pointer=f"/rl/grid/{bad[0]}")
        for point in grid_points(grid):
            env.with_reward(**point)
            RewardSpec.from_dict({**env.reward.__dict__, **point})
    for algo in algos:
        hp = rl.get(algo)
        if not isinstance(hp, Mapping):
            raise FinplanError.validation("hyperparameters missing for an RL algorithm", pointer=f"/rl/{algo}")
        from finplan_model.rl.train_params import check_hyperparameters

        check_hyperparameters(algo, hp)
    runtime = p["runtime"]
    rs = runtime.get("reserve_seconds")
    if isinstance(rs, bool) or not isinstance(rs, (int, float)) or not math.isfinite(float(rs)) or rs < 0:
        raise FinplanError.validation("runtime.reserve_seconds must be a non-negative number", pointer="/runtime/reserve_seconds")
    if p["incumbent_fallback"] not in CONTROL_NAMES:
        raise FinplanError.validation("the incumbent fallback is a control", pointer="/incumbent_fallback")
    return p


def protocol_id(protocol: Mapping[str, Any]) -> str:
    return configuration_id(dict(protocol))
