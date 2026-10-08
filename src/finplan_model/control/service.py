"""The job control plane: submit, status, result, cancel, list, approve, the dispatcher and the
SageMaker state-change handler (design D1 to D4; specs experiment-job-interface and
job-execution-controls).

The service only validates, records, estimates cost, takes leases and calls
``CreateProcessingJob`` / ``StopProcessingJob`` / ``DescribeProcessingJob`` (or the
``*TrainingJob`` operations for job types configured with ``sagemaker_job: training``, such as
``model_selection``). It never runs strategy
code (spec job-deployment-pipeline, "Job API Lambda"). Every state change is a conditional write
(revision check) plus an append-only event, so late or duplicate SageMaker events after a terminal
state change nothing (JOB-05).

Dependencies are injected (:class:`ServiceDeps`): the run store, runtime settings, a SageMaker
client (tests use ``tests/fakes/sagemaker.py``; real and moto SageMaker are blocked in tests), the
run hand-off IO, a clock and the ID minter, and optionally the platform client (approved-snapshot
check at submission), a ``model_version`` resolver (registry), a dispatcher ``kick`` and the
dispatcher schedule (``wakeup``, :mod:`finplan_model.control.wakeup`): the dispatcher runs only
while runs are queued, active or awaiting an approval deadline (design D1).
"""

from __future__ import annotations

import base64
import binascii
import copy
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from finplan_contracts import gpu_rule

from finplan_model.core.clock import Clock, parse_utc, utc_iso
from finplan_model.core.config import EnvConfig
from finplan_model.core.context import RunContext
from finplan_model.core.errors import ErrorCode, FinplanError, as_finplan_error
from finplan_model.core.ids import IdMinter, request_hash, require_id
from finplan_model.core.outcome import require_valid
from finplan_model.core.platform import PlatformClient, SnapshotReader
from finplan_model.jobs.runio import RunIO
from finplan_model.jobs.spec import build_run_spec, simulation_config_for

from . import costs
from . import production_strategy as ps
from .auth import Principal
from .leases import LeaseManager
from .sagemaker import JOB_KINDS, SAGEMAKER_TERMINAL, build_job_request, classify_start_error, job_kind, job_status, run_id_from_job_name, terminal_outcome
from .settings import SettingsProvider
from .states import ACTIVE_STATES, STATES, WAITING_STATES, check_transition, is_terminal
from .store import ConditionFailed, RunStore, idempotency_scope_key, iter_runs
from .validation import find_storage_location, validate_submission
from .wakeup import TICK, DispatchSchedule, WakePlan, plan_wakeup, stronger

__all__ = ["JobService", "ServiceDeps", "status_document"]

_IDEM_KEY = re.compile(r"^[A-Za-z0-9_-]{1,128}\Z")
_START_STALE = timedelta(minutes=5)
_LIMIT_SLACK_SECONDS = 5


@dataclass
class ServiceDeps:
    cfg: EnvConfig
    store: RunStore
    settings: SettingsProvider
    sagemaker: Any
    run_io: RunIO
    clock: Clock
    ids: IdMinter
    platform: PlatformClient | None = None
    model_version_resolver: Callable[[str | None, str | None], str | None] | None = None
    kick: Callable[[str], None] | None = None
    wakeup: DispatchSchedule | None = None
    result_hooks: list[Callable[[Mapping[str, Any], dict[str, Any]], dict[str, Any]]] = field(default_factory=list)


# ===================================================================== documents
def _elapsed_seconds(run: Mapping[str, Any], now: datetime) -> float:
    started = next((t["at"] for t in run.get("transitions", []) if t["state"] == "running"), None)
    if started is None:
        return 0.0
    end = run.get("completed_at") if is_terminal(run["state"]) else None
    end_dt = parse_utc(end) if end else now
    return max(0.0, (end_dt - parse_utc(started)).total_seconds())


def status_document(run: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """The contract ``job-status`` document of a run (no storage locations, no principal ARNs)."""
    doc: dict[str, Any] = {
        "run_id": run["run_id"],
        "state": run["state"],
        "purpose": run["purpose"],
        "dry_run": False,
        "compute_class": run["compute_class"],
        "cost_estimate": dict(run["cost_estimate"]),
        "transitions": [dict(t) for t in run.get("transitions", [])],
        "elapsed_seconds": _elapsed_seconds(run, now),
        "configuration_id": run["configuration_id"],
        "input_snapshot_id": run["input_snapshot_id"],
        "domain": run["domain"],
        "submitted_at": run["submitted_at"],
        "updated_at": run["updated_at"],
        "job_type": run["job_type"],
        "max_runtime_seconds": run["max_runtime_seconds"],
    }
    if is_terminal(run["state"]):
        doc["completion_status"] = run["state"]
    if run.get("approval"):
        doc["approval"] = dict(run["approval"])
    if run.get("model_version"):
        doc["model_version"] = run["model_version"]
    if run["state"] == "failed" and run.get("error"):
        doc["error"] = dict(run["error"])
    if run.get("wait_reason") and run["state"] in WAITING_STATES:
        doc["wait_reason"] = run["wait_reason"]
    if run.get("cancel_reason") and run["state"] in ("cancelled", "stopping"):
        doc["cancel_reason"] = run["cancel_reason"]
    if run.get("solution_status") and run["state"] == "succeeded":
        doc["solution_status"] = run["solution_status"]
    if run.get("synthetic"):
        doc["synthetic"] = True
    return require_valid(doc, "job-status")


def _encode_token(cursor: str) -> str:
    return base64.urlsafe_b64encode(cursor.encode()).decode().rstrip("=")


def _decode_token(token: str) -> str:
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise FinplanError.validation("next_token is not valid", pointer="/next_token") from None
    if not re.fullmatch(r"[0-9TZ:.\-]+#run_[0-9A-HJKMNP-TV-Z]{26}", raw):
        raise FinplanError.validation("next_token is not valid", pointer="/next_token")
    return raw


# ===================================================================== service
class JobService:
    def __init__(self, deps: ServiceDeps) -> None:
        self.d = deps
        self.cfg = deps.cfg
        self.env = deps.cfg.env
        self.store = deps.store
        self.leases = LeaseManager(deps.store, self.env, clock=deps.clock, ttl_seconds=int(deps.cfg.lease["ttl_seconds"]), limits=deps.settings.lease_limits)

    # ------------------------------------------------------------------ helpers
    def now(self) -> datetime:
        return self.d.clock.now()

    def ts(self) -> str:
        return utc_iso(self.now())

    def _ctx(self, correlation_id: str, run_id: str | None = None, synthetic: bool = True) -> RunContext:
        return RunContext(environment=self.env, run_id=run_id, correlation_id=correlation_id, clock=self.d.clock, ids=self.d.ids, synthetic=synthetic, logger_name="finplan_model.control")

    def _log(self, event: str, run: Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
        override = fields.pop("correlation_id", None)
        cid = (run or {}).get("correlation_id") or override or "corr-control-plane"
        return self._ctx(cid, (run or {}).get("run_id")).log(event, **fields)

    def _envelope(self, run: Mapping[str, Any], err: FinplanError) -> dict[str, Any]:
        return err.to_envelope(run["correlation_id"], synthetic=True if run.get("synthetic") else None)

    def _get(self, run_id: Any) -> dict[str, Any]:
        require_id("run_id", run_id)
        run = self.store.get_run(run_id)
        if run is None:
            raise FinplanError(ErrorCode.NOT_FOUND, "run not found", details={"record_type": "run"})
        return run

    def _transition(self, run: dict[str, Any], new_state: str, *, reason: str | None = None, **changes: Any) -> dict[str, Any]:
        check_transition(run["state"], new_state)
        if new_state not in ("cancelled", "failed") and not gpu_rule.may_transition({**run, **changes}, new_state):
            raise FinplanError.precondition("GPU runs cannot leave awaiting_approval without a recorded approval", reason="gpu_approval_required")
        now = self.ts()
        new = copy.deepcopy(run)
        new.update(changes)
        entry: dict[str, Any] = {"state": new_state, "at": now}
        if reason:
            entry["reason"] = reason
        new["transitions"] = [*run.get("transitions", []), entry]
        new.update(state=new_state, updated_at=now, revision=int(run["revision"]) + 1)
        if is_terminal(new_state):
            new["completion_status"] = new_state
            new["completed_at"] = now
        event = {"run_id": run["run_id"], "seq": len(new["transitions"]) - 1, "from_state": run["state"], "state": new_state, "at": now, **({"reason": reason} if reason else {})}
        self.store.update_run(new, int(run["revision"]), [event])
        self._log("run_transition", new, from_state=run["state"], to_state=new_state, reason=reason)
        return new

    def _update(self, run: dict[str, Any], **changes: Any) -> dict[str, Any]:
        new = copy.deepcopy(run)
        new.update(changes)
        new.update(updated_at=self.ts(), revision=int(run["revision"]) + 1)
        self.store.update_run(new, int(run["revision"]), [])
        return new

    def _release(self, run: Mapping[str, Any]) -> None:
        if self.leases.release(run["compute_class"], run["run_id"]):
            self._log("lease_released", run)

    # ================================================================== submit_job
    def submit_job(self, principal: Principal, body: Any, *, correlation_id: str) -> tuple[int, dict[str, Any]]:
        if not isinstance(body, Mapping):
            raise FinplanError.validation("the request body must be a JSON object", pointer="")
        dry_run = body.get("dry_run") is True
        key = body.get("idempotency_key")
        scope_key = None
        if not dry_run and isinstance(key, str) and _IDEM_KEY.match(key):
            scope_key = idempotency_scope_key(principal.arn, self.env, "submit_job", key)
            replay = self._replay(scope_key, body)
            if replay is not None:
                return replay
        sub = validate_submission(body, self.cfg, job_definition_published=lambda jt: self.d.settings.job_definition(jt) is not None)
        daily = sub.job_type.name == ps.DAILY_JOB_TYPE
        trigger = principal.role_name == ps.daily_trigger_role_name(self.env)
        if daily and not trigger:
            raise FinplanError(ErrorCode.FORBIDDEN, "only the platform daily trigger submits daily_recommendation", details={"job_type": ps.DAILY_JOB_TYPE})
        if trigger and not daily:
            raise FinplanError(ErrorCode.FORBIDDEN, "the platform daily trigger submits daily_recommendation only; experiments are user-initiated", details={"job_type": sub.job_type.name})
        all_runs: list[dict[str, Any]] | None = None
        production: dict[str, Any] | None = None
        if daily:
            if sub.purpose != "production_candidate":
                raise FinplanError.validation("daily_recommendation runs with purpose production_candidate only", pointer="/purpose")
            if not body.get("plan_id"):
                raise FinplanError.validation("daily_recommendation needs the plan_id it stages against", pointer="/plan_id")
            all_runs = list(iter_runs(self.store))
            production = self._production_strategy(body, all_runs)
        elif sub.purpose == "production_candidate":
            principal.require_production_candidate_grant(self.d.settings.production_candidate_principals())
        synthetic = sub.synthetic
        dataset_id: str | None = None
        disclosures: list[dict[str, Any]] | None = None
        if self.d.platform is not None:
            snap = SnapshotReader(self.d.platform).resolve(str(body["input_snapshot_id"]))
            synthetic = synthetic or snap.synthetic
            dataset_id = snap.dataset_id or None
            disclosures = list(snap.record.get("bias_disclosures") or []) or None
            universe = ps.is_universe_dataset(dataset_id)
            if daily and not universe:
                raise FinplanError.validation("daily_recommendation needs an approved equity-etf-daily snapshot", pointer="/input_snapshot_id")
            if universe and not disclosures:
                raise FinplanError.validation("the universe snapshot carries no bias disclosures", pointer="/input_snapshot_id", reason="bias_disclosures_missing")
        payload = body["configuration"]["payload"]
        simulation = simulation_config_for(self.cfg.simulation_defaults, payload)
        selection: dict[str, Any] = {}
        if sub.job_type.name == "model_selection":
            selection = self._selection_fields(sub, dataset_id, all_runs)
        est = costs.estimate(self.cfg, sub.job_type, instance_type=sub.instance_type, instance_count=sub.instance_count, max_runtime_seconds=sub.max_runtime_seconds, prices=self.d.settings.instance_prices(), now=self.now())
        if all_runs is None:
            all_runs = list(iter_runs(self.store))
        remaining = costs.check_budget(est, allocation=self.d.settings.budget_allocation(), budget_state=self.d.settings.budget_state(), runs=all_runs, correlation_id=correlation_id)
        block = est.block(remaining, synthetic=synthetic)
        require_valid(block, "cost-estimate")
        if dry_run:
            resp = {"run_id": None, "configuration_id": sub.configuration_id, "state": None, "dry_run": True, "cost_estimate": block, "message": "Dry run: the request is valid and estimated; no run was recorded and no job started."}
            if synthetic:
                resp["synthetic"] = True
            self._log("submit_dry_run", correlation_id=correlation_id, job_type=sub.job_type.name, estimate=est.estimated_usd_upper_bound)
            return 200, require_valid(resp, "tools/submit-experiment-response")
        waiting = sum(1 for r in all_runs if r["state"] in WAITING_STATES)
        if waiting >= int(self.cfg.queue["max_depth"]):
            raise FinplanError(ErrorCode.RATE_LIMITED, "the job queue is full; retry later", details={"max_depth": int(self.cfg.queue["max_depth"])})
        gpu = sub.job_type.compute_class == "gpu" or est.budget_category == "gpu"
        threshold = sub.job_type.auto_approve_usd if sub.job_type.auto_approve_usd is not None else self.d.settings.auto_approve_usd()
        needs_approval = gpu or est.estimated_usd_upper_bound > threshold + 1e-12
        state = "awaiting_approval" if needs_approval else "queued"
        now = self.ts()
        run_id = self.d.ids.run_id()
        model_version = None
        if self.d.model_version_resolver is not None:
            jd = self.d.settings.job_definition(sub.job_type.name) or {}
            uri = str(jd.get("image_uri") or "")
            model_version = self.d.model_version_resolver(payload.get("strategy"), uri.rsplit("@", 1)[1] if "@" in uri else None)
        if production is not None:
            production = {**production, "configuration_id": sub.configuration_id, "model_version": model_version or production.get("model_version")}
            production = {k: v for k, v in production.items() if v is not None}
        run: dict[str, Any] = {
            "run_id": run_id,
            "environment": self.env,
            "state": state,
            "revision": 0,
            "purpose": sub.purpose,
            "job_type": sub.job_type.name,
            "compute_class": sub.job_type.compute_class,
            "instance_type": sub.instance_type,
            "instance_count": sub.instance_count,
            "max_runtime_seconds": sub.max_runtime_seconds,
            "runtime_ceiling_seconds": sub.job_type.max_runtime_seconds,
            "principal": principal.arn,
            "principal_name": principal.display_name,
            "idempotency_key": body["idempotency_key"],
            "configuration": copy.deepcopy(body["configuration"]),
            "configuration_id": sub.configuration_id,
            "input_snapshot_id": body["input_snapshot_id"],
            "evaluation_window": copy.deepcopy(body.get("evaluation_window")),
            "domain": body["domain"],
            "domain_schema_version": body["domain_schema_version"],
            "contract_version": body["contract_version"],
            "simulation": simulation,
            "cost_estimate": block,
            "budget_category": est.budget_category,
            "approval_required": needs_approval,
            "approval": None,
            "submitted_at": now,
            "updated_at": now,
            "transitions": [{"state": state, "at": now}],
            "correlation_id": correlation_id,
            "synthetic": synthetic,
            "model_version": model_version,
            "dry_run": False,
            "job_started": False,
            "job_name": None,
            "attempt": 0,
            "start_retries": 0,
            "quota_waits": 0,
            "quota_wait_started_at": None,
            "not_before": None,
            "wait_reason": "awaiting_human_approval" if needs_approval else None,
            "cancel_requested": False,
            "cancel_reason": None,
            "estimated_cost_usd": est.estimated_usd_upper_bound,
            "actual_cost_usd": None,
            "dataset_id": dataset_id,
            "plan_id": body.get("plan_id"),
            "production_strategy": production,
            "bias_disclosures": disclosures,
            "sagemaker_job": sub.job_type.sagemaker_job,
            **selection,
        }
        message = "A human approver must approve this run before it starts." if needs_approval else "Queued; the run starts when a concurrency lease is free."
        resp = {"run_id": run_id, "configuration_id": sub.configuration_id, "state": state, "dry_run": False, "cost_estimate": block, "message": message}
        if synthetic:
            resp["synthetic"] = True
        require_valid(resp, "tools/submit-experiment-response")
        idem = self._idem_record(scope_key, principal, "submit_job", str(body["idempotency_key"]), body, resp, 202) if scope_key else None
        event = {"run_id": run_id, "seq": 0, "from_state": None, "state": state, "at": now}
        try:
            self.store.create_run(run, [event], idem)
        except ConditionFailed:
            replay = self._replay(scope_key, body) if scope_key else None
            if replay is not None:
                return replay
            raise FinplanError(ErrorCode.CONFLICT, "a concurrent submission won; retry with the same idempotency key") from None
        self._log("run_submitted", run, state=state, job_type=run["job_type"], estimate=est.estimated_usd_upper_bound, budget_category=est.budget_category)
        self._arm_for(run)
        if state == "queued":
            self._kick(run_id)
        return 202, resp

    def _selection_fields(self, sub: Any, dataset_id: str | None, runs: list[dict[str, Any]] | None) -> dict[str, Any]:
        """``model_selection`` (decision 27): the protocol frozen from configuration (never from the
        request), the incumbent for the promotion check (the production strategy when one is set,
        else the protocol's fallback in the job) and how often this test period was already used."""
        from finplan_model.selection.protocol import validate_protocol

        protocol = validate_protocol(sub.job_type.protocol or {})
        prior = sum(1 for r in (runs if runs is not None else iter_runs(self.store)) if r.get("job_type") == "model_selection" and r.get("state") == "succeeded" and r.get("dataset_id") == dataset_id)
        incumbent = None
        raw = self.d.settings.production_strategy()
        if raw:
            try:
                import json

                incumbent = str(json.loads(raw).get("strategy_id") or "") or None
            except (ValueError, AttributeError):
                incumbent = None
        out: dict[str, Any] = {"selection_protocol": protocol, "test_period_prior_accesses": prior}
        if incumbent:
            out["incumbent_strategy"] = incumbent
        return out

    def _production_strategy(self, body: Mapping[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
        """M2: resolve the strategy from the SSM key at submission and re-validate it (nothing starts otherwise)."""
        getter = getattr(self.d.settings, "production_strategy", None)
        doc = ps.parse_document(getter() if callable(getter) else None)
        if doc is None:
            raise FinplanError.precondition("no production strategy is selected in this environment", reason="no_production_strategy")
        strategy_id = str(doc["strategy_id"])
        verdict = ps.eligibility(strategy_id, cfg=self.cfg, runs=runs, model_version=doc.get("model_version"))
        if not verdict.ok:
            raise FinplanError.precondition("the selected production strategy is no longer eligible", reason="strategy_not_eligible", rule=verdict.rule)
        requested = body["configuration"]["payload"].get("strategy")
        if requested not in (None, strategy_id):
            raise FinplanError.validation("the request names a different strategy than the production strategy", pointer="/configuration/payload/strategy", field="strategy_id")
        if requested is None:
            raise FinplanError.validation("the configuration must name the production strategy", pointer="/configuration/payload/strategy", field="strategy_id")
        return {"strategy_id": strategy_id, "model_version": doc.get("model_version"), "evidence_run_id": verdict.evidence_run_id}

    def _idem_record(self, scope_key: str, principal: Principal, operation: str, key: str, body: Mapping[str, Any], resp: Mapping[str, Any], status: int) -> dict[str, Any]:
        now = self.now()
        retain = now + timedelta(days=int(self.cfg.idempotency["retention_days"]))
        record = {
            "scope": {"principal": principal.arn, "environment": self.env, "operation": operation},
            "idempotency_key": key,
            "request_hash": request_hash(dict(body)),
            "response": dict(resp),
            "recorded_at": utc_iso(now),
            "retain_until": utc_iso(retain),
        }
        require_valid(record, "idempotency")
        return {**record, "status_code": status, "scope_key": scope_key, "ttl": int(retain.timestamp())}

    def _replay(self, scope_key: str | None, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]] | None:
        if scope_key is None:
            return None
        rec = self.store.get_idempotency(scope_key)
        if rec is None or parse_utc(rec["retain_until"]) < self.now():
            return None
        if rec["request_hash"] != request_hash(dict(body)):
            raise FinplanError(ErrorCode.IDEMPOTENCY_KEY_REUSED, "the idempotency key was used with a different request")
        self._log("idempotent_replay", correlation_id=None, operation=rec["scope"]["operation"])
        return int(rec.get("status_code", 200)), dict(rec["response"])

    # ================================================================== reads
    def get_job_status(self, principal: Principal, run_id: Any) -> dict[str, Any]:
        return status_document(self._get(run_id), self.now())

    def get_job_result(self, principal: Principal, run_id: Any) -> dict[str, Any]:
        run = self._get(run_id)
        if not is_terminal(run["state"]):
            raise FinplanError.precondition("the run has not finished; no result yet", reason="run_not_terminal", state=run["state"])
        result = copy.deepcopy(run.get("result") or self._result_without_job(run, run["state"], run.get("error")))
        for hook in self.d.result_hooks:
            result = hook(run, result)
        if find_storage_location(result) is not None:
            raise FinplanError.internal("result failed the storage-location check")
        return require_valid(result, "job-result")

    def list_jobs(self, principal: Principal, query: Mapping[str, Any]) -> dict[str, Any]:
        state = query.get("state")
        if state is not None and state not in STATES:
            raise FinplanError.validation("unknown state filter", pointer="/state")
        size = query.get("page_size", 20)
        if isinstance(size, str):
            size = int(size) if size.isdigit() else -1
        if not isinstance(size, int) or not 1 <= size <= 100:
            raise FinplanError.validation("page_size must be between 1 and 100", pointer="/page_size")
        after = _decode_token(str(query["next_token"])) if query.get("next_token") else None
        runs, nxt = self.store.query_runs(state, limit=size, after=after)
        now = self.now()
        out: dict[str, Any] = {"jobs": [status_document(r, now) for r in runs]}
        if nxt:
            out["next_token"] = _encode_token(nxt)
        return out

    # ================================================================== cancel_job
    def cancel_job(self, principal: Principal, run_id: Any, body: Any) -> tuple[int, dict[str, Any]]:
        body = dict(body or {}) if isinstance(body, Mapping) else body
        if not isinstance(body, dict):
            raise FinplanError.validation("the request body must be a JSON object", pointer="")
        key = body.get("idempotency_key")
        if not isinstance(key, str) or not _IDEM_KEY.match(key):
            raise FinplanError.validation("idempotency_key is required on cancel_job", pointer="/idempotency_key")
        unknown = set(body) - {"idempotency_key", "reason"}
        if unknown:
            raise FinplanError.validation("unknown field in cancel request", pointer="/" + sorted(unknown)[0] if re.fullmatch(r"[a-z_]{1,64}", sorted(unknown)[0]) else "")
        hashed = {"run_id": run_id, **body}
        scope_key = idempotency_scope_key(principal.arn, self.env, "cancel_job", key)
        replay = self._replay(scope_key, hashed)
        if replay is not None:
            return replay
        run = self._get(run_id)
        approver = self.d.settings.approver_role_name()
        if principal.arn != run["principal"] and (not approver or principal.role_name != approver):
            raise FinplanError(ErrorCode.FORBIDDEN, "only the submitting principal or the approver may cancel this run")
        run = self._cancel(run, reason="cancelled_by_caller")
        resp = status_document(run, self.now())
        try:
            rec = self._idem_record(scope_key, principal, "cancel_job", key, hashed, resp, 200)
            self.store.put_idempotency(scope_key, rec, rec["ttl"])
        except ConditionFailed:
            replay = self._replay(scope_key, hashed)
            if replay is not None:
                return replay
        return 200, resp

    def _cancel(self, run: dict[str, Any], *, reason: str) -> dict[str, Any]:
        for _ in range(5):
            try:
                if is_terminal(run["state"]):
                    return run  # terminal: reported unchanged
                if run["state"] == "stopping":
                    return run if run.get("cancel_requested") else self._update(run, cancel_requested=True, cancel_reason=reason)
                if run["state"] in WAITING_STATES or (run["state"] == "starting" and not run.get("job_name")):
                    new = self._transition(run, "cancelled", reason=reason, cancel_requested=True, cancel_reason=reason, wait_reason=None)
                    self._release(new)
                    return new
                # starting with a job, or running: stop the SageMaker job and wait for confirmation
                new = self._transition(run, "stopping", reason=reason, cancel_requested=True, cancel_reason=reason)
                self._stop_job(new)
                return new
            except ConditionFailed:
                run = self._get(run["run_id"])
        raise FinplanError(ErrorCode.CONFLICT, "the run changed concurrently; retry")

    def _stop_job(self, run: Mapping[str, Any]) -> None:
        try:
            kind = job_kind(run)
            getattr(self.d.sagemaker, JOB_KINDS[kind]["stop"])(**{JOB_KINDS[kind]["name"]: run["job_name"]})
            self._log("stop_requested", run, job_name=run["job_name"])
        except Exception as exc:  # noqa: BLE001 - the job may already be terminal; reconcile
            self._log("stop_request_failed", run, error_code=getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else type(exc).__name__)
            self._reconcile(dict(run))

    # ================================================================== approve_run
    def approve_run(self, principal: Principal, run_id: Any, body: Any) -> dict[str, Any]:
        body = dict(body or {}) if isinstance(body, Mapping) else body
        if not isinstance(body, dict):
            raise FinplanError.validation("the request body must be a JSON object", pointer="")
        unknown = set(body) - {"idempotency_key", "approved_estimate_usd"}
        if unknown:
            raise FinplanError.validation("unknown field in approve request", pointer="")
        run = self._get(run_id)
        principal.require_approver(self.d.settings.approver_role_name(), submitter=run["principal"])
        if run.get("approval") and run["state"] != "awaiting_approval":
            return status_document(run, self.now())  # already approved (idempotent)
        if run["state"] != "awaiting_approval":
            raise FinplanError.precondition("the run is not awaiting approval", reason="not_awaiting_approval", state=run["state"])
        if self._approval_expired(run):
            self._transition(run, "cancelled", reason="approval_expired", cancel_reason="approval_expired", wait_reason=None)
            raise FinplanError.precondition("the approval window has expired; the run was cancelled", reason="approval_expired", state="cancelled")
        estimate = float(run["cost_estimate"]["estimated_usd_upper_bound"])
        approved_usd = body.get("approved_estimate_usd", estimate)
        if isinstance(approved_usd, bool) or not isinstance(approved_usd, (int, float)) or not math.isfinite(float(approved_usd)) or approved_usd < 0:
            raise FinplanError.validation("approved_estimate_usd must be a non-negative number", pointer="/approved_estimate_usd")
        if estimate > float(approved_usd) + 1e-9:
            raise FinplanError.precondition("the cost estimate exceeds the approved amount", reason="estimate_above_approval")
        from finplan_contracts.budget import _state_enforced  # contract interpretation of budget-state

        if _state_enforced(self.d.settings.budget_state()):
            raise FinplanError(ErrorCode.BUDGET_EXCEEDED, "the project budget enforcement action is active", details={"budget_category": run["budget_category"], "budget_state": "enforced"})
        approval = {"approved_by": principal.display_name, "approved_at": self.ts(), "approved_estimate_usd": float(approved_usd)}
        try:
            new = self._transition(run, "queued", reason="approved", approval=approval, wait_reason=None)
        except ConditionFailed:
            raise FinplanError(ErrorCode.CONFLICT, "the run changed concurrently; retry") from None
        self._log("run_approved", new, approved_by=approval["approved_by"])
        self._arm_for(new)
        self._kick(new["run_id"])
        return status_document(new, self.now())

    def _approval_deadline(self, run: Mapping[str, Any]) -> datetime:
        return parse_utc(run["submitted_at"]) + timedelta(hours=float(self.cfg.approval["approval_window_hours"]))

    def _approval_expired(self, run: Mapping[str, Any]) -> bool:
        return self.now() > self._approval_deadline(run)

    # ------------------------------------------------------------------ dispatcher wake-up (design D1)
    def _kick(self, run_id: str) -> None:
        if self.d.kick is not None:
            self.d.kick(run_id)

    def _plan_for(self, run: Mapping[str, Any] | None) -> WakePlan | None:
        """What one non-terminal run needs from the dispatcher (``None``: nothing)."""
        if run is None or is_terminal(run["state"]):
            return None
        if run["state"] == "awaiting_approval":
            return plan_wakeup(queued=0, active=0, approval_deadlines=[self._approval_deadline(run)], now=self.now())
        return WakePlan(TICK, f"run {run['state']}")

    def _arm_for(self, run: Mapping[str, Any]) -> None:
        """Strengthen the schedule so ``run`` is served (never weakens it). Failures are logged: the
        asynchronous kick and the next state event still reach the dispatcher."""
        plan = self._plan_for(run)
        if plan is None or self.d.wakeup is None:
            return
        try:
            if self.d.wakeup.ensure(plan):
                self._log("dispatch_schedule_armed", run, schedule=plan.label)
        except Exception as exc:  # noqa: BLE001 - never fail a submission over the schedule
            self._log("dispatch_schedule_unavailable", run, error=type(exc).__name__)

    def pending_plan(self) -> WakePlan:
        """The schedule this environment's pending runs need (:func:`~finplan_model.control.wakeup.plan_wakeup`)."""
        queued = sum(1 for _ in iter_runs(self.store, "queued"))
        active = sum(1 for st in ACTIVE_STATES for _ in iter_runs(self.store, st))
        deadlines = [self._approval_deadline(r) for r in iter_runs(self.store, "awaiting_approval")]
        return plan_wakeup(queued=queued, active=active, approval_deadlines=deadlines, now=self.now())

    def rearm(self, hint_run_id: str | None = None) -> WakePlan:
        """Set the schedule to exactly what the pending runs need (end of every dispatcher tick).

        ``hint_run_id`` (the run a submission or approval kicked the dispatcher for) is read with a
        strongly consistent read, so a run the state index does not show yet still keeps the
        dispatcher armed. Before disarming, the pending runs are read once more: a submission that
        raced with this tick re-arms it.
        """
        plan = self.pending_plan()
        if hint_run_id:
            hint = self.store.get_run(str(hint_run_id))
            plan = stronger(plan, self._plan_for(hint))
        if self.d.wakeup is None:
            return plan
        try:
            if self.d.wakeup.apply(plan):
                self._log("dispatch_schedule_set", correlation_id="corr-dispatch-schedule", schedule=plan.label, reason=plan.reason)
            if not plan.enabled:
                again = self.pending_plan()
                if again.enabled:
                    self.d.wakeup.apply(again)
                    self._log("dispatch_schedule_set", correlation_id="corr-dispatch-schedule", schedule=again.label, reason="raced: " + again.reason)
                    plan = again
        except Exception as exc:  # noqa: BLE001 - the tick's work is done; an unchanged schedule only costs ticks
            self._log("dispatch_schedule_unavailable", correlation_id="corr-dispatch-schedule", error=type(exc).__name__)
        return plan

    # ================================================================== dispatcher
    def dispatch(self, hint_run_id: str | None = None) -> dict[str, list[str]]:
        """One dispatcher tick: approval expiry, reconcile active runs, reclaim stale leases, start
        queued runs; then set the schedule for what is still pending (:meth:`rearm`)."""
        summary: dict[str, list[str]] = {"expired": [], "reconciled": [], "reclaimed": [], "started": [], "requeued": [], "failed": [], "waiting": []}
        for run in iter_runs(self.store, "awaiting_approval"):
            if self._approval_expired(run):
                try:
                    self._transition(run, "cancelled", reason="approval_expired", cancel_reason="approval_expired", wait_reason=None)
                    summary["expired"].append(run["run_id"])
                except ConditionFailed:
                    pass
        for state in ACTIVE_STATES:
            for run in iter_runs(self.store, state):
                self._reconcile(run)
                summary["reconciled"].append(run["run_id"])
        for cls in sorted(self.d.settings.lease_limits()):
            for slot in self.leases.expired(cls):
                if self._maybe_reclaim(cls, slot):
                    summary["reclaimed"].append(str(slot["holder"]))
        queued = list(iter_runs(self.store, "queued"))
        for run in queued:
            if run.get("not_before") and parse_utc(run["not_before"]) > self.now():
                summary["waiting"].append(run["run_id"])
                continue
            outcome = self._start(run)
            if outcome == "no_lease":
                # Runs without a lease stay queued (CTL-02); later runs wait behind this one.
                pos = queued.index(run)
                summary["waiting"] += [r["run_id"] for r in queued[pos:]]
                break
            summary.setdefault(outcome, []).append(run["run_id"])
        summary["schedule"] = [self.rearm(hint_run_id).label]
        return summary

    def _fail(self, run: dict[str, Any], err: FinplanError, *, reason: str) -> dict[str, Any]:
        envelope = self._envelope(run, err)
        result = self._result_without_job(run, "failed", envelope)
        new = self._transition(run, "failed", reason=reason, error=envelope, result=result, wait_reason=None)
        self._release(new)
        return new

    def _start(self, run: dict[str, Any]) -> str:
        from finplan_contracts.budget import _state_enforced

        if _state_enforced(self.d.settings.budget_state()):
            self._fail(run, FinplanError(ErrorCode.BUDGET_EXCEEDED, "the project budget enforcement action is active", details={"budget_category": run["budget_category"], "budget_state": "enforced"}), reason="budget_enforced")
            return "failed"
        cls = run["compute_class"]
        slot = self.leases.acquire(cls, run["run_id"])
        if slot is None:
            self._log("lease_unavailable", run, instance_class=cls)
            return "no_lease"
        attempt = int(run.get("attempt", 0)) + 1
        try:
            run = self._transition(run, "starting", reason="lease_acquired", attempt=attempt, lease_slot=slot, wait_reason=None, not_before=None)
        except ConditionFailed:
            self.leases.release(cls, run["run_id"])
            return "conflict"
        try:
            jd = self.d.settings.job_definition(run["job_type"])
            if not jd:
                raise FinplanError.dependency_unavailable("the job definition is not published", retryable=False, job_type=run["job_type"])
            image_uri = str(jd.get("image_uri") or "")
            digest = image_uri.rsplit("@", 1)[1] if "@" in image_uri else None
            spec = build_run_spec(run, simulation=run["simulation"], image_digest=digest)
            kind = job_kind(run)
            kind, request = build_job_request(run, attempt=attempt, image_uri=image_uri, role_arn=str(self.d.settings.job_role_arn() or ""), run_spec_checksum="pending", output_bucket=self.d.settings.research_storage() if kind == "training" else None)
            checksum = self.d.run_io.put_spec(run["run_id"], spec)
            request["Environment"]["FINPLAN_RUN_SPEC_SHA256"] = checksum
        except FinplanError as err:
            self._fail(run, err, reason="start_precondition")
            return "failed"
        try:
            getattr(self.d.sagemaker, JOB_KINDS[kind]["create"])(**request)
        except Exception as exc:  # noqa: BLE001 - classified
            kind = classify_start_error(exc)
            if kind != "exists":
                return self._start_refused(run, kind)
        name = request[JOB_KINDS[kind]["name"]]
        try:
            self._update(run, job_name=name, job_started=True, start_requested_at=self.ts())
        except ConditionFailed:
            # Cancelled while the job was being created: record the job (it costs money) and stop it.
            latest = self._get(run["run_id"])
            try:
                latest = self._update(latest, job_name=name, job_started=True)
            except ConditionFailed:
                latest = {**self._get(run["run_id"]), "job_name": name}
            if latest.get("cancel_requested") or is_terminal(latest["state"]):
                self._stop_job(latest)
            return "started"
        self._log("job_started", run, job_name=name, attempt=attempt, sagemaker_job=kind, max_runtime_seconds=request["StoppingCondition"]["MaxRuntimeInSeconds"])
        return "started"

    def _backoff(self, n: int) -> timedelta:
        q = self.cfg.queue
        return timedelta(seconds=min(float(q["quota_backoff_max_seconds"]), float(q["quota_backoff_initial_seconds"]) * (2 ** max(0, n - 1))))

    def _start_refused(self, run: dict[str, Any], kind: str) -> str:
        now = self.now()
        q = self.cfg.queue
        if kind == "quota":
            started = parse_utc(run["quota_wait_started_at"]) if run.get("quota_wait_started_at") else now
            waits = int(run.get("quota_waits", 0)) + 1
            if (now - started).total_seconds() >= float(q["quota_max_wait_seconds"]):
                self._fail(run, FinplanError.dependency_unavailable("the account-level instance quota stayed exhausted beyond the maximum wait", retryable=True, reason="quota_wait_exceeded"), reason="quota_wait_exceeded")
                return "failed"
            new = self._transition(run, "queued", reason="account_quota_exhausted", wait_reason="account_quota_exhausted", quota_waits=waits, quota_wait_started_at=utc_iso(started), not_before=utc_iso(now + self._backoff(waits)))
            self._release(new)
            return "requeued"
        if kind == "throttled":
            retries = int(run.get("start_retries", 0)) + 1
            if retries > int(q["start_retry_limit"]):
                self._fail(run, FinplanError.dependency_unavailable("SageMaker kept throttling the job start; start retries are exhausted", retryable=True, reason="start_retries_exhausted"), reason="start_retries_exhausted")
                return "failed"
            new = self._transition(run, "queued", reason="start_retry", wait_reason="start_throttled", start_retries=retries, not_before=utc_iso(now + self._backoff(retries)))
            self._release(new)
            return "requeued"
        if kind == "denied":
            self._fail(run, FinplanError.dependency_unavailable("SageMaker refused the job start for this role", retryable=False, reason="start_denied"), reason="start_denied")
            return "failed"
        self._fail(run, FinplanError.internal("SageMaker rejected the job request", reason="start_rejected"), reason="start_rejected")
        return "failed"

    # ------------------------------------------------------------------ reconcile
    def _describe(self, job_name: str, kind: str = "processing") -> dict[str, Any] | None:
        try:
            return dict(getattr(self.d.sagemaker, JOB_KINDS[kind]["describe"])(**{JOB_KINDS[kind]["name"]: job_name}))
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if code in ("ResourceNotFound", "ValidationException"):
                return None
            raise

    def _reconcile(self, run: dict[str, Any]) -> None:
        if is_terminal(run["state"]):
            return
        if not run.get("job_name"):
            if run["state"] == "starting" and self.now() - parse_utc(run["updated_at"]) > _START_STALE:
                try:
                    new = self._transition(run, "queued", reason="start_incomplete", wait_reason="start_incomplete")
                    self._release(new)
                except ConditionFailed:
                    pass
            return
        kind = job_kind(run)
        try:
            desc = self._describe(run["job_name"], kind)
        except Exception:  # noqa: BLE001 - transient; the next tick retries
            self._log("describe_failed", run)
            return
        if desc is None:
            requested = run.get("start_requested_at") or run["updated_at"]
            if self.now() - parse_utc(requested) < _START_STALE:
                return  # not visible yet (eventual consistency)
            desc = {JOB_KINDS[kind]["status"]: "Failed", "FailureReason": "job not found"}
        self._apply_status(run, str(job_status(desc, kind)), desc, source="dispatcher")

    def _limit_reached(self, run: Mapping[str, Any], desc: Mapping[str, Any]) -> bool:
        fields = JOB_KINDS[job_kind(run)]
        start, end = desc.get(fields["start"]), desc.get(fields["end"])
        if isinstance(start, datetime) and isinstance(end, datetime):
            elapsed = (end - start).total_seconds()
        else:
            running_at = next((t["at"] for t in run.get("transitions", []) if t["state"] == "running"), None)
            elapsed = (self.now() - parse_utc(running_at)).total_seconds() if running_at else 0.0
        return elapsed >= int(run["max_runtime_seconds"]) - _LIMIT_SLACK_SECONDS

    def _apply_status(self, run: dict[str, Any], status: str, desc: Mapping[str, Any], *, source: str) -> dict[str, Any]:
        for _ in range(5):
            try:
                return self._apply_status_once(run, status, desc, source=source)
            except ConditionFailed:
                run = self._get(run["run_id"])
        raise FinplanError(ErrorCode.CONFLICT, "the run changed concurrently; retry")

    def _apply_status_once(self, run: dict[str, Any], status: str, desc: Mapping[str, Any], *, source: str) -> dict[str, Any]:
        if is_terminal(run["state"]):
            self._log("late_event_ignored", run, sagemaker_status=status, source=source, state=run["state"])
            return run
        cls = run["compute_class"]
        if status == "InProgress":
            if run["state"] == "starting":
                run = self._transition(run, "running", reason="sagemaker_in_progress")
            self.leases.renew(cls, run["run_id"])
            return run
        if status == "Stopping":
            self.leases.renew(cls, run["run_id"])
            return run
        if status not in SAGEMAKER_TERMINAL:
            return run
        d = dict(desc)
        d["_elapsed_reached_limit"] = self._limit_reached(run, desc)
        outcome = str(terminal_outcome(status, d, cancel_requested=bool(run.get("cancel_requested"))))
        job_doc = None
        try:
            job_doc = self.d.run_io.get_result(run["run_id"])
        except FinplanError:
            job_doc = None
        outcome, result, error = self._final_result(run, outcome, job_doc)
        changes: dict[str, Any] = {"result": result, "error": error, "wait_reason": None}
        if outcome == "succeeded":
            changes["solution_status"] = result.get("solution_status")
        new = self._transition(run, outcome, reason=f"sagemaker_{status.lower()}", **changes)
        self._release(new)
        if outcome == "failed":
            self._log("run_failed", new, code=(error or {}).get("code"))
        return new

    def _maybe_reclaim(self, cls: str, slot: Mapping[str, Any]) -> bool:
        holder = str(slot["holder"])
        run = self.store.get_run(holder)
        desc: dict[str, Any] | None = None
        if run is not None and run.get("job_name"):
            try:
                desc = self._describe(run["job_name"], job_kind(run))
            except Exception:  # noqa: BLE001 - cannot confirm: keep the lease
                self._log("lease_stale_unconfirmed", run, slot=slot["slot"])
                return False
            terminal = desc is None or job_status(desc, job_kind(run)) in SAGEMAKER_TERMINAL
        else:
            terminal = run is None or run["state"] not in ACTIVE_STATES or (run["state"] == "starting" and self.now() - parse_utc(run["updated_at"]) > _START_STALE)
        if not terminal:
            self.leases.renew(cls, holder, int(slot["slot"]))
            self._log("lease_stale_holder_alive", run, slot=slot["slot"])
            return False
        if not self.leases.reclaim(cls, dict(slot)):
            return False
        self._log("lease_reclaimed", run, correlation_id="corr-lease-reclaim", slot=slot["slot"], holder=holder, instance_class=cls)
        if run is not None and not is_terminal(run["state"]):
            if desc is not None or run.get("job_name"):
                status = str(job_status(desc, job_kind(run)) or "Failed")
                self._apply_status(run, status, desc or {"FailureReason": "job not found"}, source="lease_reclaim")
            elif run["state"] == "starting":
                try:
                    self._transition(run, "queued", reason="start_incomplete", wait_reason="start_incomplete")
                except ConditionFailed:
                    pass
        return True

    # ------------------------------------------------------------------ results
    def _base_result(self, run: Mapping[str, Any], completion: str) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "run_id": run["run_id"],
            "completion_status": completion,
            "artifacts": [],
            "artifacts_complete": False,
            "configuration_id": run["configuration_id"],
            "input_snapshot_id": run["input_snapshot_id"],
            "domain": run["domain"],
            "domain_schema_version": run["domain_schema_version"],
            "completed_at": self.ts(),
        }
        if run.get("model_version"):
            doc["model_version"] = run["model_version"]
        if run.get("synthetic"):
            doc["synthetic"] = True
        return doc

    def _result_without_job(self, run: Mapping[str, Any], completion: str, error: Mapping[str, Any] | None) -> dict[str, Any]:
        doc = self._base_result(run, completion)
        if completion == "failed":
            doc["error"] = dict(error) if error else self._envelope(run, FinplanError.internal("the run failed"))
        return doc

    def _final_result(self, run: Mapping[str, Any], outcome: str, job_doc: Mapping[str, Any] | None) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        """``(completion_status, job-result document, error envelope)`` for a terminal SageMaker job."""
        valid_doc: dict[str, Any] | None = None
        if job_doc is not None and job_doc.get("run_id") == run["run_id"] and find_storage_location(job_doc) is None:
            try:
                valid_doc = require_valid(dict(job_doc), "job-result")
            except FinplanError:
                valid_doc = None
        if outcome == "succeeded":
            if valid_doc is None or valid_doc.get("completion_status") != "succeeded":
                if valid_doc is not None and valid_doc.get("completion_status") == "failed":
                    return "failed", valid_doc, valid_doc.get("error")
                err = self._envelope(run, FinplanError.internal("the job finished without a valid result document", reason="result_missing_or_invalid"))
                return "failed", self._result_without_job(run, "failed", err), err
            doc = copy.deepcopy(valid_doc)
            doc.update({k: v for k, v in self._base_result(run, "succeeded").items() if k not in doc or k in ("configuration_id", "input_snapshot_id", "run_id")})
            cost = (doc.get("payload") or {}).get("compute_cost")
            if isinstance(cost, dict):
                cost.setdefault("estimated_usd", float(run["cost_estimate"]["estimated_usd_upper_bound"]))
            return "succeeded", doc, None
        if outcome == "failed":
            if valid_doc is not None and valid_doc.get("completion_status") == "failed" and valid_doc.get("error"):
                return "failed", valid_doc, valid_doc["error"]
            err = self._envelope(run, FinplanError.internal("job container exited abnormally", exit="abnormal"))
            return "failed", self._result_without_job(run, "failed", err), err
        # cancelled or timed_out: outputs are incomplete and never staged
        doc = self._base_result(run, outcome)
        if valid_doc is not None:
            doc["artifacts"] = list(valid_doc.get("artifacts") or [])
            for k in ("evaluator_version", "dataset_checksum"):
                if valid_doc.get(k):
                    doc[k] = valid_doc[k]
        doc["artifacts_complete"] = False
        return outcome, doc, None

    # ================================================================== state-change events
    def handle_sagemaker_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """EventBridge ``SageMaker Processing Job State Change`` or ``SageMaker Training Job State
        Change`` -> run transition (late events ignored)."""
        detail = dict(event.get("detail") or {})
        kind = "training" if "TrainingJobName" in detail else "processing"
        name = str(detail.get(JOB_KINDS[kind]["name"]) or "")
        parsed = run_id_from_job_name(name)
        if parsed is None or parsed[0] != self.env:
            return {"ignored": "not_this_environment"}
        _, run_id, attempt = parsed
        run = self.store.get_run(run_id)
        if run is None:
            self._log("event_for_unknown_run", correlation_id="corr-state-change", job_name=name)
            return {"ignored": "unknown_run"}
        if run.get("job_name") != name and not is_terminal(run["state"]):
            if int(run.get("attempt", 0)) > attempt:
                self._log("stale_attempt_event_ignored", run, job_name=name)
                return {"ignored": "stale_attempt"}
            if run["state"] == "starting" and not run.get("job_name"):
                try:
                    run = self._update(run, job_name=name, job_started=True)
                except ConditionFailed:
                    run = self._get(run_id)
        if job_kind(run) != kind:
            self._log("event_kind_mismatch_ignored", run, job_name=name, sagemaker_job=kind)
            return {"ignored": "job_kind_mismatch"}
        status = str(job_status(detail, kind) or "")
        new = self._apply_status(run, status, detail, source="event")
        # A run that is still active needs lease heartbeats: re-arm the tick (a deploy may have reset
        # the schedule to its deployed DISABLED state while the job was running).
        self._arm_for(new)
        return {"run_id": run_id, "state": new["state"]}

    # ================================================================== failures as results
    def failure_response(self, exc: BaseException, correlation_id: str) -> dict[str, Any]:
        return as_finplan_error(exc).to_envelope(correlation_id)
