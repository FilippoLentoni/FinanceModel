"""Per-environment production-strategy setting and the ``daily_recommendation`` rules (change
add-daily-recommendation-and-on-demand-experiments; specs production-strategy-setting,
daily-recommendation-job, universe-research-input; design M1 to M3).

* The setting lives at ``/finplan/<env>/financemodel/config/production-strategy`` as a
  ``core/v1/production-strategy.json`` document (contracts 1.1.0). Its single writer is the
  ``strategy-selection`` Lambda (:class:`StrategySelectionService`, role
  ``finplan-<env>-financemodel-strategy-selection-role``); the job API only reads it. An absent key,
  an empty value, unparseable JSON or a document without ``strategy_id`` all mean "no strategy".
* Selection (``PUT /v1/production-strategy``, body ``{"action": "get"|"set"|"clear", ...}`` as in
  ``core/v1/tools/production-strategy-request.json`` plus ``confirmed_by_user`` and an optional
  ``on_behalf_of``; ``GET /v1/production-strategy`` reads). ``set`` and ``clear`` need
  ``confirmed_by_user: true`` (else ``PRECONDITION_FAILED`` ``confirmation_required``), an
  ``idempotency_key`` and a caller that is the tool plan-writer role or the platform operator role.
  ``set`` passes :func:`eligibility` (registry rules, M3) or fails ``VALIDATION_FAILED`` naming the
  rule. Every change appends an audit record (old value, new value, user, channel).
* The daily job (M1, M2): only the platform trigger role submits ``daily_recommendation``
  (:func:`daily_trigger_role_name`), and the trigger role submits nothing else.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from finplan_contracts import ssm as contract_ssm

from finplan_model.core.clock import Clock, utc_iso
from finplan_model.core.config import EnvConfig
from finplan_model.core.errors import ErrorCode, FinplanError, contract_version
from finplan_model.core.ids import IdMinter, request_hash
from finplan_model.core.outcome import require_valid

from .auth import Principal
from .store import ConditionFailed, RunStore, idempotency_scope_key, iter_runs

__all__ = [
    "DAILY_JOB_TYPE",
    "EVIDENCE_JOB_TYPES",
    "LEARNING_FAMILIES",
    "RETIRED_STRATEGIES",
    "UNIVERSE_DATASET_KIND",
    "InMemoryStrategyParameter",
    "SsmStrategyParameter",
    "StrategyParameter",
    "StrategySelectionService",
    "daily_trigger_role_name",
    "eligibility",
    "evidence_run",
    "is_universe_dataset",
    "parse_document",
    "selection_caller_patterns",
    "strategy_parameter",
    "supported_datasets",
]

DAILY_JOB_TYPE = "daily_recommendation"
#: The research-universe dataset kind (``finance/equity-etf-daily/research-universe``).
UNIVERSE_DATASET_KIND = "equity-etf-daily"
#: Job types whose succeeded research runs count as evaluation evidence (M3).
EVIDENCE_JOB_TYPES = ("run_backtest", "run_benchmark")
EVIDENCE_PURPOSES = ("research", "holdout_evaluation")
#: Strategy families that must be ``promoted`` in the model registry before selection.
LEARNING_FAMILIES = ("learning", "rl", "llm", "swarm")
#: Strategies retired from selection (none yet).
RETIRED_STRATEGIES: frozenset[str] = frozenset()
_ACTIONS = ("get", "set", "clear")
_IDEM_KEY = re.compile(r"^[A-Za-z0-9_-]{1,128}\Z")
_EXTRA_FIELDS = ("confirmed_by_user", "on_behalf_of")
_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}\Z")


def strategy_parameter(env: str) -> str:
    return contract_ssm.production_strategy_parameter(env)


def daily_trigger_role_name(env: str) -> str:
    """The platform's daily trigger step role, the only ``daily_recommendation`` submitter (M1)."""
    return f"finplan-{env}-financialplanning-daily-trigger-step-role"


def selection_caller_patterns(env: str) -> tuple[re.Pattern[str], ...]:
    """Role names allowed to set or clear: the tool plan-writer role and the platform operator roles."""
    return (
        re.compile(rf"^finplan-{env}-financelambdastool-.*plan-writer.*$"),
        re.compile(rf"^finplan-{env}-financialplanning-operator.*$"),
    )


def is_universe_dataset(dataset_id: str | None) -> bool:
    return bool(dataset_id) and f"/{UNIVERSE_DATASET_KIND}/" in f"{dataset_id}/"


def supported_datasets(strategy_id: str) -> tuple[str, ...]:
    """Dataset kinds a strategy supports: every CPU baseline handles one or many instruments plus cash."""
    from finplan_model.strategies.registry import BASELINES

    return ("etf-daily", UNIVERSE_DATASET_KIND) if strategy_id in BASELINES else ()


def _family(strategy_id: str) -> str | None:
    from finplan_model.strategies.registry import BASELINES

    cls = BASELINES.get(strategy_id)
    return getattr(cls, "family", None) if cls is not None else None


def parse_document(raw: str | None) -> dict[str, Any] | None:
    """The stored document, or ``None`` for absent, empty, unparseable or strategy-less values."""
    if raw is None or not str(raw).strip():
        return None
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(doc, dict) or not doc.get("strategy_id"):
        return None
    return doc


def evidence_run(strategy_id: str, runs: Iterable[Mapping[str, Any]], env: str) -> str | None:
    """``run_id`` of the latest succeeded research run that evaluated ``strategy_id`` on the universe."""
    from finplan_model.jobs.strategy_resolver import CONTROLS

    best: tuple[str, str] | None = None
    for r in runs:
        if r.get("environment", env) != env or r.get("state") != "succeeded" or r.get("job_type") not in EVIDENCE_JOB_TYPES:
            continue
        if r.get("purpose") not in EVIDENCE_PURPOSES or not is_universe_dataset(r.get("dataset_id")):
            continue
        main = str(((r.get("configuration") or {}).get("payload") or {}).get("strategy") or "")
        evaluated = {main} | (set(CONTROLS) if r.get("job_type") == "run_benchmark" else set())
        if strategy_id in evaluated:
            key = (str(r.get("completed_at") or r.get("updated_at") or ""), str(r["run_id"]))
            best = key if best is None or key > best else best
    return best[1] if best else None


@dataclass(frozen=True)
class Eligibility:
    ok: bool
    rule: str | None = None
    evidence_run_id: str | None = None


def eligibility(
    strategy_id: str,
    *,
    cfg: EnvConfig,
    runs: Iterable[Mapping[str, Any]],
    model_version: str | None = None,
    registry_status: Callable[[str], str | None] | None = None,
) -> Eligibility:
    """Registry validation (M3); the first failing rule is named."""
    from finplan_model.jobs.strategy_resolver import known_strategies

    if strategy_id not in known_strategies():
        return Eligibility(False, "strategy_not_registered")
    if not any(cfg.job_type(jt) is not None and cfg.job_type(jt).deployed for jt in (DAILY_JOB_TYPE, *EVIDENCE_JOB_TYPES)):  # type: ignore[union-attr]
        return Eligibility(False, "strategy_not_deployed")
    if strategy_id in RETIRED_STRATEGIES:
        return Eligibility(False, "strategy_retired")
    if UNIVERSE_DATASET_KIND not in supported_datasets(strategy_id):
        return Eligibility(False, "dataset_not_supported")
    if (_family(strategy_id) or "") in LEARNING_FAMILIES:
        status = registry_status(model_version) if (registry_status and model_version) else None
        if status != "promoted":
            return Eligibility(False, "strategy_not_promoted")
    run_id = evidence_run(strategy_id, runs, cfg.env)
    if run_id is None:
        return Eligibility(False, "no_evaluation_evidence")
    return Eligibility(True, None, run_id)


# ===================================================================== the SSM key
class StrategyParameter(Protocol):
    def read(self) -> str | None: ...

    def write(self, value: str) -> None: ...

    def delete(self) -> None: ...


class InMemoryStrategyParameter:
    def __init__(self, value: str | None = None) -> None:
        self.value = value
        self.writes = 0

    def read(self) -> str | None:
        return self.value

    def write(self, value: str) -> None:
        self.value = value
        self.writes += 1

    def delete(self) -> None:
        self.value = None
        self.writes += 1


class SsmStrategyParameter:
    """Uncached reads (the daily job must see a cleared key at once); writes only from the selection role."""

    def __init__(self, ssm_client: Any, name: str) -> None:
        self.ssm = ssm_client
        self.name = name

    def read(self) -> str | None:
        try:
            return str(self.ssm.get_parameter(Name=self.name)["Parameter"]["Value"])
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if code == "ParameterNotFound":
                return None
            raise FinplanError.dependency_unavailable("the production-strategy setting could not be read; retry", retryable=True) from None

    def write(self, value: str) -> None:
        self.ssm.put_parameter(Name=self.name, Value=value, Type="String", Overwrite=True)

    def delete(self) -> None:
        try:
            self.ssm.delete_parameter(Name=self.name)
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if code != "ParameterNotFound":
                raise


# ===================================================================== selection service
class StrategySelectionService:
    """get/set/clear of the production strategy (the ``strategy-selection`` operation)."""

    def __init__(
        self,
        cfg: EnvConfig,
        parameter: StrategyParameter,
        store: RunStore,
        clock: Clock,
        ids: IdMinter,
        *,
        model_version_resolver: Callable[[str | None, str | None], str | None] | None = None,
        registry_status: Callable[[str], str | None] | None = None,
        audit_log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = cfg.env
        self.parameter = parameter
        self.store = store
        self.clock = clock
        self.ids = ids
        self.model_version_resolver = model_version_resolver
        self.registry_status = registry_status
        self.audit_log = audit_log

    def _response(self, action: str, doc: Mapping[str, Any] | None, changed: bool) -> dict[str, Any]:
        resp: dict[str, Any] = {"environment": self.env, "action": action, "strategy": dict(doc) if doc else None, "changed": changed, "contract_version": contract_version()}
        return require_valid(resp, "tools/production-strategy-response")

    def get(self) -> dict[str, Any]:
        doc = parse_document(self.parameter.read())
        if doc is not None:
            try:
                require_valid(doc, "production-strategy")
            except FinplanError:
                doc = None  # an unreadable document means "no strategy"
        return self._response("get", doc, False)

    def handle(self, principal: Principal, body: Any) -> tuple[int, dict[str, Any]]:
        if not isinstance(body, Mapping):
            raise FinplanError.validation("the request body must be a JSON object", pointer="")
        action = body.get("action")
        if action not in _ACTIONS:
            raise FinplanError.validation("action must be get, set or clear", pointer="/action")
        request = {k: v for k, v in body.items() if k not in _EXTRA_FIELDS}
        require_valid(request, "tools/production-strategy-request")
        if action == "get":
            return 200, self.get()
        name = principal.role_name or ""
        if not any(p.match(name) for p in selection_caller_patterns(self.env)):
            raise FinplanError(ErrorCode.FORBIDDEN, "only the plan-writer tool role or the operator may change the production strategy", details={"reason": "caller_not_allowed"})
        if body.get("confirmed_by_user") is not True:
            raise FinplanError.precondition("the user must confirm this change explicitly (confirmed_by_user: true)", reason="confirmation_required")
        user = body.get("on_behalf_of")
        if user is not None and (not isinstance(user, str) or not _USER.match(user)):
            raise FinplanError.validation("on_behalf_of is not a valid user name", pointer="/on_behalf_of")
        key = str(body["idempotency_key"])
        if not _IDEM_KEY.match(key):
            raise FinplanError.validation("idempotency_key is not valid", pointer="/idempotency_key")
        scope_key = idempotency_scope_key(principal.arn, self.env, f"{action}_production_strategy", key)
        replay = self.store.get_idempotency(scope_key)
        if replay is not None:
            if replay["request_hash"] != request_hash(dict(body)):
                raise FinplanError(ErrorCode.IDEMPOTENCY_KEY_REUSED, "the idempotency key was used with a different request")
            return int(replay.get("status_code", 200)), dict(replay["response"])
        old = parse_document(self.parameter.read())
        if action == "set":
            status, resp, new = self._set(principal, str(body["strategy_id"]), old, body)
        else:
            new = None
            changed = old is not None
            if changed:
                self.parameter.delete()
            status, resp = 200, self._response("clear", None, changed)
        if resp["changed"]:
            self._audit(principal, action, old, new, user)
        self._remember(scope_key, principal, action, key, body, resp, status)
        return status, resp

    def _set(self, principal: Principal, strategy_id: str, old: Mapping[str, Any] | None, body: Mapping[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        model_version = self.model_version_resolver(strategy_id, None) if self.model_version_resolver else None
        verdict = eligibility(strategy_id, cfg=self.cfg, runs=iter_runs(self.store), model_version=model_version, registry_status=self.registry_status)
        if not verdict.ok:
            raise FinplanError.validation("the strategy is not eligible for production", pointer="/strategy_id", rule=verdict.rule)
        if old is not None and old.get("strategy_id") == strategy_id and old.get("model_version") == model_version:
            return 200, self._response("set", old, False), dict(old)
        doc: dict[str, Any] = {
            "strategy_id": strategy_id,
            "environment": self.env,
            "selected_at": utc_iso(self.clock.now()),
            "selected_by": (principal.role_name or "unknown-principal")[:128],
            "contract_version": contract_version(),
            "note": f"evidence run {verdict.evidence_run_id}",
        }
        if model_version:
            doc["model_version"] = model_version
        require_valid(doc, "production-strategy")
        self.parameter.write(json.dumps(doc, sort_keys=True, separators=(",", ":")))
        return 200, self._response("set", doc, True), doc

    def _audit(self, principal: Principal, action: str, old: Mapping[str, Any] | None, new: Mapping[str, Any] | None, user: str | None) -> None:
        record = {
            "event": "production_strategy_changed",
            "audit_id": self.ids.correlation_id(),
            "environment": self.env,
            "action": action,
            "old_value": dict(old) if old else None,
            "new_value": dict(new) if new else None,
            "user": user or principal.display_name,
            "channel": "tool" if "financelambdastool" in (principal.role_name or "") else "operator",
            "principal_name": principal.display_name,
            "at": utc_iso(self.clock.now()),
        }
        append = getattr(self.store, "append_audit", None)
        if callable(append):
            append(record)
        if self.audit_log is not None:
            self.audit_log(record)

    def _remember(self, scope_key: str, principal: Principal, action: str, key: str, body: Mapping[str, Any], resp: Mapping[str, Any], status: int) -> None:
        now = self.clock.now()
        retain = now + timedelta(days=int(self.cfg.idempotency["retention_days"]))
        record = {
            "scope": {"principal": principal.arn, "environment": self.env, "operation": f"{action}_production_strategy"},
            "idempotency_key": key,
            "request_hash": request_hash(dict(body)),
            "response": dict(resp),
            "recorded_at": utc_iso(now),
            "retain_until": utc_iso(retain),
            "status_code": status,
        }
        try:
            self.store.put_idempotency(scope_key, record, int(retain.timestamp()))
        except ConditionFailed:
            pass
