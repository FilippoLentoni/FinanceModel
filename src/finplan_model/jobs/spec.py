"""The run spec: everything a job container needs, resolved once by the control plane at start.

The spec is complete and self-describing so a run is reproducible from it alone: the contract
job submission fields, the minted ``run_id`` and ``correlation_id``, the approved runtime, the
cost estimate, ``model_version`` (when the registry knows the strategy implementation) and the
**full** simulation configuration (environment defaults from ``config/<env>.json`` merged with the
experiment configuration's fees, constraints and rebalance frequency). It never contains a storage
location or a secret.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from finplan_model.core.errors import FinplanError
from finplan_model.sim.config import SimulationConfig

from .strategy_resolver import strategy_params

__all__ = ["SPEC_VERSION", "build_run_spec", "simulation_config_for", "universe_for", "validate_run_spec"]

SPEC_VERSION = "run-spec-v1"
#: Pseudo-instrument some configurations list for the cash sleeve (contract fixtures use ``CASH``).
CASH_IDS = frozenset({"CASH", "USD_CASH"})
_REQUIRED = ("spec_version", "run_id", "environment", "job_type", "purpose", "input_snapshot_id", "configuration", "configuration_id", "correlation_id", "contract_version", "simulation", "max_runtime_seconds")


def simulation_config_for(defaults: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Environment simulation defaults overridden by the experiment configuration (finance payload)."""
    sim: dict[str, Any] = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in defaults.items() if not str(k).startswith("$")}
    if payload.get("rebalance_frequency"):
        sim["rebalance_frequency"] = payload["rebalance_frequency"]
    fees = payload.get("fees") or {}
    if "transaction_cost_bps" in fees:
        sim["fees"] = {**dict(sim.get("fees") or {}), "proportional_bps": float(fees["transaction_cost_bps"])}
    cons = payload.get("constraints") or {}
    if cons:
        merged = dict(sim.get("constraints") or {})
        for key in ("long_only", "max_weight", "min_weight", "max_turnover"):
            if key in cons:
                merged[key] = cons[key]
        sim["constraints"] = merged
    # Validate now: a bad combination fails the submission, not the job.
    return SimulationConfig.from_dict(sim).to_dict()


def universe_for(payload: Mapping[str, Any]) -> list[str]:
    return [str(i) for i in payload.get("universe", []) if str(i) not in CASH_IDS]


def build_run_spec(run: Mapping[str, Any], *, simulation: Mapping[str, Any], image_digest: str | None) -> dict[str, Any]:
    payload = run["configuration"]["payload"]
    spec: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "run_id": run["run_id"],
        "environment": run["environment"],
        "job_type": run["job_type"],
        "purpose": run["purpose"],
        "input_snapshot_id": run["input_snapshot_id"],
        "configuration": run["configuration"],
        "configuration_id": run["configuration_id"],
        "domain": run["configuration"]["domain"],
        "domain_schema_version": run["configuration"]["domain_schema_version"],
        "correlation_id": run["correlation_id"],
        "contract_version": run["contract_version"],
        "simulation": dict(simulation),
        "universe": universe_for(payload),
        "strategy": payload.get("strategy"),
        "strategy_params": strategy_params(str(payload.get("strategy") or ""), payload),
        "max_runtime_seconds": int(run["max_runtime_seconds"]),
        "cost_estimate": run.get("cost_estimate"),
        "synthetic": bool(run.get("synthetic")),
    }
    if run.get("evaluation_window"):
        spec["evaluation_window"] = dict(run["evaluation_window"])
    if run.get("purpose") == "production_candidate" and run.get("plan_id"):
        spec["staging"] = {"plan_id": run["plan_id"]}  # contracts 1.1.0 job-submission plan_id
    if run.get("dataset_id"):
        spec["dataset_id"] = run["dataset_id"]
    if run.get("policy_source"):
        spec["policy_source"] = dict(run["policy_source"])
    if run.get("production_strategy"):
        spec["production_strategy"] = dict(run["production_strategy"])  # frozen at submission (M2)
    if run.get("model_version"):
        spec["model_version"] = run["model_version"]
    if run.get("selection_protocol"):
        # model_selection: the protocol frozen at submission (config/<env>.json), never from the request
        spec["selection_protocol"] = dict(run["selection_protocol"])
        spec["test_period_prior_accesses"] = int(run.get("test_period_prior_accesses") or 0)
        if run.get("incumbent_strategy"):
            spec["incumbent_strategy"] = str(run["incumbent_strategy"])
        spec["compute"] = {"instance_type": str(run.get("instance_type") or ""), "instance_count": int(run.get("instance_count") or 1), "sagemaker_job": str(run.get("sagemaker_job") or "processing")}
    if image_digest:
        spec["image_digest"] = image_digest
    return validate_run_spec(spec)


_STORAGE_RE = re.compile(r"(?i)^(s3|s3a|s3n|gs|file|hdfs)://|arn:aws[a-z-]*:s3:|\.s3[.-]")


def _no_storage(node: Any, ptr: str = "") -> None:
    if isinstance(node, str) and _STORAGE_RE.search(node):
        raise FinplanError.validation("run specs never carry storage locations", pointer=ptr)
    if isinstance(node, Mapping):
        for k, v in node.items():
            _no_storage(v, f"{ptr}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _no_storage(v, f"{ptr}/{i}")


def validate_run_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    for key in _REQUIRED:
        if key not in spec:
            raise FinplanError.validation("run spec is incomplete", pointer=f"/{key}")
    if spec["spec_version"] != SPEC_VERSION:
        raise FinplanError.validation("unsupported run spec version", pointer="/spec_version")
    _no_storage(spec)
    SimulationConfig.from_dict(spec["simulation"])
    return dict(spec)
