"""Deterministic exported actors and frozen baseline allocation inference."""
from __future__ import annotations

import io
import math
import zipfile
from collections.abc import Mapping
from typing import Any

import numpy as np

from finplan_model.core.artifacts import sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.constraints import apply_constraints
from .spec import EnvSpec, allocation_action, build_observation

FORMAT = "finplan-actor/1"


def implementation_identity():
    """Identify the inference transforms as well as the separately frozen actor weights."""
    from pathlib import Path

    from finplan_model.sim import constraints
    from . import spec

    return {
        "version": "finplan-policy-inference/1",
        "numpy_version": np.__version__,
        "source_checksums": {
            name: sha256_checksum(Path(module_path).read_bytes())
            for name, module_path in (
                ("inference", __file__),
                ("observation_and_actions", spec.__file__),
                ("constraints", constraints.__file__),
            )
        },
    }


def export_actor(data: bytes, algorithm: str = "ppo") -> dict[str, Any]:
    """Read tensors only; never deserialize SB3's cloudpickled Python objects."""
    import torch

    if algorithm not in ("ppo", "sac"):
        raise FinplanError.validation("unsupported actor algorithm", pointer="/algorithm")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.getinfo("policy.pth")
        if info.file_size > 8 * 1024 * 1024:
            raise FinplanError.validation("policy tensor payload exceeds the export bound", pointer="/policy")
        tensors = torch.load(io.BytesIO(archive.read(info)), weights_only=True, map_location="cpu")
    prefix = "mlp_extractor.policy_net." if algorithm == "ppo" else "actor.latent_pi."
    keys = sorted((k for k in tensors if k.startswith(prefix) and k.endswith(".weight")), key=lambda k: int(k.split(".")[-2]))
    if len(keys) != 2 or keys != [prefix + "0.weight", prefix + "2.weight"]:
        raise FinplanError.validation("unsupported actor architecture", pointer="/policy")
    final = "action_net.weight" if algorithm == "ppo" else "actor.mu.weight"
    layers = []
    for key in [*keys, final]:
        layers.append({"weight": tensors[key].detach().numpy().tolist(), "bias": tensors[key[:-6] + "bias"].detach().numpy().tolist(),
                       "activation": "linear" if key == final else ("tanh" if algorithm == "ppo" else "relu")})
    return {"format": FORMAT, "algorithm": algorithm, "layers": layers, "output_transform": "clip" if algorithm == "ppo" else "tanh"}


def actor_action(actor: Mapping[str, Any], observation: np.ndarray) -> np.ndarray:
    if actor.get("format") not in (FORMAT, "finplan-ppo-actor/1") or len(actor.get("layers", [])) != 3:
        raise FinplanError.precondition("unsupported policy export", reason="policy_format_invalid")
    x = np.asarray(observation, dtype=np.float32)
    for layer in actor["layers"]:
        weight, bias = np.asarray(layer["weight"], dtype=np.float32), np.asarray(layer["bias"], dtype=np.float32)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],) or weight.shape[1] != x.size or not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise FinplanError.precondition("invalid policy tensor shape or value", reason="policy_format_invalid")
        x = weight @ x + bias
        activation = layer["activation"]
        if activation == "tanh":
            x = np.tanh(x)
        elif activation == "relu":
            x = np.maximum(x, 0)
        elif activation != "linear":
            raise FinplanError.precondition("unsupported actor activation", reason="policy_format_invalid")
    transform = actor.get("output_transform", "clip")
    if transform == "tanh":
        return np.tanh(x)
    if transform == "clip":
        return np.clip(x, -1, 1)
    raise FinplanError.precondition("unsupported actor output transform", reason="policy_format_invalid")


def portfolio_state(instruments, holdings):
    rows = holdings["weights"]
    by_id = {r["instrument_id"]: float(r["weight"]) for r in rows}
    if len(by_id) != len(rows) or set(by_id) - set(instruments):
        raise FinplanError.validation("holdings contain duplicate or unsupported instruments", pointer="/holdings/weights")
    current = np.array([by_id.get(i, 0.) for i in instruments])
    cash = float(holdings["cash_weight"])
    value, peak = float(holdings["portfolio_value"]), float(holdings["high_watermark"])
    if not np.isfinite(current).all() or np.any(current < 0) or np.any(current > 1) or not all(math.isfinite(v) for v in (cash, value, peak)) or not 0 <= cash <= 1 or value <= 0 or peak < value or abs(float(current.sum()) + cash - 1) > 1e-6:
        raise FinplanError.validation("holdings must be finite, sum to one and have a high watermark covering positive current value", pointer="/holdings")
    return current, cash, value, peak


def allocation_response(bundle, holdings, proposed, proposed_cash, *, solution_status="feasible", diagnostics=None):
    instruments = bundle["instruments"]
    current, cash, value, _ = portfolio_state(instruments, holdings)
    constraints = SimulationConfig.from_dict({"constraints": bundle.get("constraints")}).constraints
    outcome = apply_constraints(dict(zip(instruments, np.asarray(proposed).tolist())), float(proposed_cash), constraints,
                                current=dict(zip(instruments, current.tolist())), current_cash=cash, policy=bundle.get("constraint_policy", "project"), tol=1e-9)
    if outcome.weights is None:
        # The offline evaluator rejects the proposed trade and holds existing positions.
        # Preserve that behavior at serving time and expose the rejection explicitly.
        weights, target_cash = current, cash
        solution_status = "rejected_hold_current"
    else:
        weights = np.array([outcome.weights[i] for i in instruments])
        target_cash = float(outcome.cash)
    deltas = weights - current
    return {"mode": "advisory_paper", "strategy": bundle.get("strategy_id", "ppo"), "policy_source_run_id": bundle["source_run_id"], "configuration_id": bundle["configuration_id"],
            "policy_seeds": [m["seed"] for m in bundle.get("members", [])], "aggregation": "mean_target_weights" if bundle.get("members") else "single_strategy",
            "target_weights": [{"instrument_id": i, "weight": float(weights[k])} for k, i in enumerate(instruments)], "cash_weight": target_cash,
            "decisions": [{"instrument_id": i, "action": "hold" if abs(float(deltas[k])) < 1e-6 else ("buy" if deltas[k] > 0 else "sell"), "delta_weight": float(deltas[k]), "indicative_notional": float(deltas[k]) * value} for k, i in enumerate(instruments)],
            "estimated_turnover": float(np.abs(deltas).sum()), "constraint_outcome": outcome.to_dict(), "solution_status": solution_status, "diagnostics": diagnostics or {},
            "forecast": {"status": "not_available", "reason": "this allocation artifact does not publish a calibrated return forecast"},
            "limitations": ["research_only", "hindsight_selected_universe", "no_trade_execution", "uses_completed_close_next_session_decision"]}


def recommend(bundle: Mapping[str, Any], prices: np.ndarray, holdings: Mapping[str, Any]) -> dict[str, Any]:
    spec = EnvSpec.from_dict(bundle["environment"])
    instruments = list(bundle["instruments"])
    current, cash, value, peak = portfolio_state(instruments, holdings)
    if prices.shape != (spec.window + 1, len(instruments)) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise FinplanError.precondition("the approved snapshot lacks aligned completed history", reason="insufficient_history")
    if not bundle.get("members"):
        raise FinplanError.precondition("the exported policy has no members", reason="policy_members_missing")
    obs = build_observation(spec, prices, current, cash, drawdown=1 - value / peak)
    targets = [allocation_action(spec, actor_action(m["actor"], obs), current, cash) for m in bundle["members"]]
    return allocation_response(
        bundle, holdings, np.mean([t[0] for t in targets], axis=0),
        float(np.mean([t[1] for t in targets])),
        diagnostics={
            "input_features": list(spec.features),
            "observation_dimension": int(obs.size),
            "observation_checksum": sha256_checksum(obs.tobytes()),
            "feature_window_sessions": spec.window,
            "observed_drawdown": 1 - value / peak,
            "member_target_weights": [
                {"seed": m["seed"], "weights": target[0].tolist(), "cash_weight": float(target[1])}
                for m, target in zip(bundle["members"], targets)
            ],
            "interpretation": "Actual frozen actor outputs and constraint transform; not causal attribution or a return forecast",
        },
    )


def recommend_baseline(bundle, market, as_of, holdings):
    from finplan_model.sim.market import HoldingsView
    from finplan_model.sim.strategy import coerce_target
    from finplan_model.strategies import build_strategy

    instruments = tuple(bundle["instruments"])
    current, cash, value, _ = portfolio_state(instruments, holdings)
    view = market.view(as_of, instruments)
    prices = {i: view.latest(i) for i in instruments}
    if any(p is None for p in prices.values()):
        raise FinplanError.precondition("the snapshot lacks current instrument prices", reason="insufficient_history")
    observed = HoldingsView({i: float(current[k]) * value / prices[i] for k, i in enumerate(instruments)}, cash * value, prices, value)
    constraints = SimulationConfig.from_dict({"constraints": bundle.get("constraints")}).constraints
    strategy = build_strategy(bundle["strategy_id"], bundle["parameters"], constraints=constraints)
    target = coerce_target(strategy.decide(view, observed), instruments)
    if target.solution_status in ("infeasible", "unbounded") or target.diagnostics.get("reason") == "insufficient_history":
        raise FinplanError.precondition("the selected strategy cannot produce a valid target", reason=target.diagnostics.get("reason", target.solution_status))
    return allocation_response(bundle, holdings, [target.weights[i] for i in instruments], target.cash, solution_status=target.solution_status, diagnostics=dict(target.diagnostics))
