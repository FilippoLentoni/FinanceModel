"""Lambda ``finplan-beta-financemodel-forward-runner``: scheduled forward competition slots.

Invoked synchronously by FinancialPlanning's Step Functions state machine (no FinanceModel
schedule). Events:

``{"action": "run", "experiment": "beta-competition-v1", "slot": "2026-10-19T13:45:00Z"}``
    Runs the experiment's fixed approach list for that slot (a weekday 09:45 or 15:45
    America/New_York instant), immediately after FinancialPlanning stored the slot's
    ``valuation_quotes``. Returns ``{"status", "slot", "experiment", "approaches":
    [{"approach_id", "status", "reason", "execution_status"}]}``. ``status`` is ``complete``
    or ``incomplete`` (another invocation holds a lease). Approach failures are data, never
    a Lambda error; infrastructure failures (DynamoDB, S3, SSM, platform API) raise.

``{"action": "prepare_llm_prompt", "experiment": "beta-competition-v1", "research_slot": "2026-10-19T13:00:00Z"}``
    Persists the direct_claude_news reviewer prompt for a 09:00/15:00 New York research slot
    (:func:`finplan_model.forward.llm_review.prepare_llm_prompt`). Idempotent.

``{"action": "preflight"}``
    Imports every module the runner uses, resolves the evidence backend (SSM + bucket
    encryption, read only) and reads SSM references and the experiment registry. No writes.
"""
from __future__ import annotations

import json
import os
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

REGION = "us-east-2"
EXPERIMENT_TRIGGER = "financialplanning_step_functions"
#: SSM references the run path reads (all must exist for a healthy preflight).
REQUIRED_PARAMETERS = ("/finplan/beta/financialplanning/config/bucket-reports", "/finplan/beta/financemodel/config/research-storage-ref",
                       "/finplan/beta/financemodel/config/advisory-policy", "/finplan/beta/financialplanning/api/plan-endpoint")
#: Agent-approach references (FinanceAgent-owned); missing ones make only those books unavailable.
AGENT_PARAMETERS = ("/finplan/beta/financeagent/agent/authorizer-metadata-ref", "/finplan/beta/financeagent/agent/runtime-ref",
                    "/finplan/beta/financeagent/secret-ref/modeling-runner-client")
MODULES = ("finplan_platform.core.modeling_evidence", "finplan_model.forward.hourly", "finplan_model.forward.runtime", "finplan_model.forward.research",
           "finplan_model.forward.llm_review", "finplan_model.forward.agent_client", "finplan_model.forward.execution_reconcile", "finplan_model.forward.paper",
           "finplan_model.evaluate.evaluator", "finplan_model.rl.inference", "finplan_model.classical.strategy", "finplan_model.strategies", "finplan_model.forward.competition")


class EventError(ValueError):
    """The event is not a valid runner request (not retryable)."""


def _require_beta() -> None:
    env = os.environ.get("FINPLAN_ENVIRONMENT", "beta")
    if env != "beta":
        raise EventError(f"the forward runner is beta only (FINPLAN_ENVIRONMENT={env})")


def _backend():
    from .hourly import evidence_backend
    return evidence_backend()


def _remaining(context: Any):
    if context is None or not hasattr(context, "get_remaining_time_in_millis"):
        return None
    return lambda: context.get_remaining_time_in_millis() / 1000.0


def run_slot(event: dict[str, Any], context: Any = None, *, backend: Any = None, run=None) -> dict[str, Any]:
    from .competition import competition
    from .hourly import run_hourly
    from .schedule import validate_slot

    spec = competition(str(event.get("experiment", "")))
    try:
        slot = validate_slot(str(event.get("slot", "")), spec.comparison_policy["decision_opportunity"])
    except ValueError as exc:
        raise EventError(str(exc)) from None
    output = Path(tempfile.mkdtemp(prefix="forward-", dir=os.environ.get("FINPLAN_RUNNER_TMP", tempfile.gettempdir())))
    result = (run or run_hourly)(slot, spec.experiment_id, output, backend=backend or _backend(), cohort=spec.cohort,
                                 comparison_policy=spec.comparison_policy, approach_ids=list(spec.approach_ids), remaining_seconds=_remaining(context))
    approaches = []
    for row in result["approaches"]:
        execution = row.get("execution") or {}
        approaches.append({"approach_id": row["approach_id"], "status": row["status"], "reason": row.get("reason") or execution.get("reason"),
                           "execution_status": execution.get("status")})
    return {"status": result["status"], "experiment": spec.experiment_id, "slot": result["slot"], "approaches": approaches,
            "shared_input_id": result.get("shared_input_id")}


def prepare_prompt(event: dict[str, Any], *, backend: Any = None, prepare=None) -> dict[str, Any]:
    from .competition import competition
    from .llm_review import APPROACH_ID, prepare_llm_prompt
    from .schedule import validate_research_slot

    spec = competition(str(event.get("experiment", "")))
    if APPROACH_ID not in spec.approach_ids:
        raise EventError(f"{spec.experiment_id} has no {APPROACH_ID} book")
    try:
        slot = validate_research_slot(str(event.get("research_slot", "")))
    except ValueError as exc:
        raise EventError(str(exc)) from None
    return (prepare or prepare_llm_prompt)(backend or _backend(), spec.experiment_id, slot, comparison_policy=spec.comparison_policy, cohort=spec.cohort)


def preflight(*, backend_factory=None, ssm: Any = None) -> dict[str, Any]:
    from .competition import COMPETITIONS

    modules = {name: import_module(name).__name__ for name in MODULES}
    if ssm is None:
        import boto3
        ssm = boto3.client("ssm", region_name=REGION)
    missing, present = [], []
    for name in REQUIRED_PARAMETERS + AGENT_PARAMETERS:
        try:
            ssm.get_parameter(Name=name)
            present.append(name)
        except Exception as exc:  # noqa: BLE001 - reported, then judged below
            if getattr(exc, "response", {}).get("Error", {}).get("Code") != "ParameterNotFound":
                raise
            missing.append(name)
    backend = (backend_factory or _backend)()
    registries = {}
    for experiment in COMPETITIONS:
        registry = backend.read_registry(experiment)
        registries[experiment] = {"registered": registry is not None, "approaches": [a["approach_id"] for a in (registry or {}).get("approaches", [])]}
    required_missing = [n for n in missing if n in REQUIRED_PARAMETERS]
    return {"status": "failed" if required_missing else "ok", "modules": sorted(modules), "parameters_present": present,
            "required_parameters_missing": required_missing, "agent_parameters_missing": [n for n in missing if n in AGENT_PARAMETERS],
            "evidence_backend": {"tables": sorted(getattr(getattr(backend, "repo", None), "names", {}) or {}), "mirror_content": getattr(backend, "mirror_content", None)},
            "experiments": registries}


def handler(event: Any, context: Any = None) -> dict[str, Any]:
    """Lambda entry point (``finplan_model.forward.runner.handler``)."""
    _require_beta()
    os.environ.setdefault("FINPLAN_EXPERIMENT_TRIGGER", EXPERIMENT_TRIGGER)
    if isinstance(event, str):
        event = json.loads(event)
    if not isinstance(event, dict):
        raise EventError("event must be a JSON object")
    action = event.get("action")
    if action == "run":
        result = run_slot(event, context)
    elif action == "prepare_llm_prompt":
        result = prepare_prompt(event)
    elif action == "preflight":
        result = preflight()
    else:
        raise EventError(f"unknown action {action!r}; expected run, prepare_llm_prompt or preflight")
    print(json.dumps({"action": action, "status": result.get("status"), "experiment": event.get("experiment"),
                      "slot": event.get("slot", event.get("research_slot"))}, sort_keys=True))
    return result
