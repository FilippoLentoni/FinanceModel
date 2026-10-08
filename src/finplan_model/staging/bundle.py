"""Run-output staging: the hand-off of production-candidate outputs to the platform (spec
run-output-staging; task 9.1; RST-01, RST-02, RST-03, RST-04).

Protocol (contracts D10, platform ``core/staging.py``):

* **Who stages** (RST-01): only a ``production_candidate`` run that ended ``succeeded`` with
  ``solution_status`` ``optimal``, ``feasible`` or ``no_effect``. Research, tuning and holdout runs
  write only to research storage; an infeasible or unbounded candidate stages nothing and its
  result says why (:func:`staging_decision`).
* **What** (RST-02): ``plan-content.json`` (finance ``plan-content``, also inlined as
  ``payload.plan_content``), ``metrics/summary.json`` and the manifest ``manifest.json``, which
  validates against the pinned ``core/v1/staged-output-manifest.json`` plus its finance payload and
  lists the SHA-256 and size of every data file. The manifest is validated **before anything is
  written**: a schema failure ends the run ``failed`` with ``VALIDATION_FAILED`` and nothing is
  staged.
* **Order** (RST-03): data files first, the manifest **last**, under
  ``<run-staging-ref><run_id>/``. The manifest is the only completion marker (no marker file); a
  job that dies before the manifest leaves an incomplete bundle the platform refuses
  (``PRECONDITION_FAILED`` ``staged_output_incomplete``).
* **Write-only, write-once** (RST-04): :class:`S3StagingStore` only ever calls ``PutObject`` with
  ``If-None-Match: *`` under its own run's prefix; it never reads, lists or deletes. The job role
  may only ``PutObject`` under ``staging/run_*/`` of this environment's staging bucket and is denied
  any put without ``If-None-Match`` (so no existing object, of this or another run, can be
  overwritten), and any read, list or delete there (:func:`infra.stacks.policies.job_execution_policy`;
  the platform bucket policy adds the same denies).
* **Commitment** stays with the platform: FinanceModel reports the run as ``staged`` (result block
  ``staging``), never claims a ``plan_version_id`` and never calls the accept route
  (:mod:`finplan_model.staging.outcome` links the platform's outcome later).

CONTRACT GAP (reported, not invented): the pinned ``core/v1/job-submission.json`` has no
``plan_id`` / ``parent_plan_version_id`` (``additionalProperties: false``), but the staged-output
manifest requires ``plan_id``. A production-candidate run therefore takes its target from the run
spec's ``staging`` block (``{"plan_id", "parent_plan_version_id"}``) when the control plane provides
one, and otherwise stages nothing (``not_staged``, reason ``no_plan_target``).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.clock import Clock, SystemClock, utc_iso
from finplan_model.core.errors import ErrorCode, FinplanError, contract_version
from finplan_model.core.ids import require_id
from finplan_model.core.outcome import require_valid

__all__ = [
    "COMMITTABLE_SOLUTIONS",
    "MANIFEST_NAME",
    "METRICS_FILE",
    "PLAN_CONTENT_FILE",
    "InMemoryStagingStore",
    "S3StagingStore",
    "StagedBundle",
    "StagingStore",
    "StagingTarget",
    "build_bundle",
    "parse_staging_ref",
    "plan_content_from_result",
    "proposed_allocation",
    "stage_run_output",
    "staging_decision",
]

MANIFEST_NAME = "manifest.json"
PLAN_CONTENT_FILE = "plan-content.json"
METRICS_FILE = "metrics/summary.json"
COMMITTABLE_SOLUTIONS = frozenset({"optimal", "feasible", "no_effect"})
PRODUCTION_CANDIDATE = "production_candidate"
_PLAN_ID = re.compile(r"^pl_[0-9A-HJKMNP-TV-Z]{26}\Z")
_PLAN_VERSION_ID = re.compile(r"^pv_[0-9A-HJKMNP-TV-Z]{26}\Z")


# ===================================================================== targets and stores
@dataclass(frozen=True)
class StagingTarget:
    """The plan a production candidate is staged for."""

    plan_id: str
    parent_plan_version_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not _PLAN_ID.match(self.plan_id):
            raise FinplanError(ErrorCode.INVALID_IDENTIFIER, "plan_id has the wrong prefix or format", field="plan_id")
        if self.parent_plan_version_id is not None and not _PLAN_VERSION_ID.match(str(self.parent_plan_version_id)):
            raise FinplanError(ErrorCode.INVALID_IDENTIFIER, "parent_plan_version_id has the wrong prefix or format", field="parent_plan_version_id")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> StagingTarget | None:
        block = spec.get("staging") or {}
        if not isinstance(block, Mapping) or not block.get("plan_id"):
            return None
        return cls(str(block["plan_id"]), block.get("parent_plan_version_id") or None)


@runtime_checkable
class StagingStore(Protocol):
    def put_new(self, run_id: str, name: str, data: bytes, content_type: str) -> None:
        """Write ``<run_id>/<name>`` once; never read, list, overwrite or delete."""
        ...


class InMemoryStagingStore:
    """Test store: records the write order; ``fail_before`` simulates a job dying before a file."""

    def __init__(self, *, fail_before: str | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.order: list[str] = []
        self.fail_before = fail_before
        self._lock = threading.Lock()

    def put_new(self, run_id: str, name: str, data: bytes, content_type: str) -> None:
        if self.fail_before == name:
            raise RuntimeError(f"simulated crash before writing {name}")
        key = f"{run_id}/{name}"
        with self._lock:
            if key in self.objects:
                raise FinplanError.not_permitted("staged objects are written once", reason="staging_object_exists")
            self.objects[key] = data
            self.order.append(key)

    def bundle(self, run_id: str) -> dict[str, bytes]:
        prefix = f"{run_id}/"
        return {k[len(prefix):]: v for k, v in self.objects.items() if k.startswith(prefix)}

    def complete(self, run_id: str) -> bool:
        """The platform's view: a bundle is complete only when its manifest exists."""
        return f"{run_id}/{MANIFEST_NAME}" in self.objects


def parse_staging_ref(ref: str) -> tuple[str, str]:
    """``s3://<bucket>/<prefix>/`` (the platform's ``config/run-staging-ref``) -> (bucket, prefix)."""
    if not isinstance(ref, str) or not ref.startswith("s3://"):
        raise FinplanError.precondition("the run-staging reference is not an s3:// reference", reason="staging_ref_invalid")
    rest = ref[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise FinplanError.precondition("the run-staging reference names no bucket", reason="staging_ref_invalid")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


class S3StagingStore:
    """Write-only, write-once staging writer. Bucket and keys never leave this class."""

    def __init__(self, client: Any, staging_ref: str) -> None:
        self._s3 = client
        self._bucket, self._prefix = parse_staging_ref(staging_ref)

    def put_new(self, run_id: str, name: str, data: bytes, content_type: str) -> None:
        require_id("run_id", run_id)
        try:
            self._s3.put_object(Bucket=self._bucket, Key=f"{self._prefix}{run_id}/{name}", Body=data, ContentType=content_type, IfNoneMatch="*")
        except Exception as exc:  # noqa: BLE001 - botocore ClientError
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", "")) if hasattr(exc, "response") else ""
            if code in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                raise FinplanError.not_permitted("staged objects are written once", reason="staging_object_exists") from None
            if code in ("AccessDenied", "403"):
                raise FinplanError.not_permitted("the staging write was denied", reason="staging_write_denied") from None
            raise FinplanError.dependency_unavailable("the platform staging area could not be written", retryable=True) from None


# ===================================================================== decision and bundle
def staging_decision(purpose: str | None, result: Mapping[str, Any]) -> tuple[bool, str]:
    """(stage?, reason) for one run (RST-01)."""
    if purpose != PRODUCTION_CANDIDATE:
        return False, "purpose_not_production_candidate"
    status = result.get("completion_status")
    if status != "succeeded":
        return False, f"completion_status_{status}"
    solution = result.get("solution_status")
    if solution not in COMMITTABLE_SOLUTIONS:
        return False, f"solution_status_{solution}"
    return True, "staged"


def proposed_allocation(evaluation: Any) -> dict[str, Any] | None:
    """The finance ``allocation`` a production candidate proposes: the constrained target weights of
    the evaluation's last executed (or projected) decision; None when no decision produced one."""
    for rec in reversed(list(getattr(evaluation.simulation, "decisions", []) or [])):
        cons = rec.get("constraints") or {}
        weights = cons.get("final_weights")
        if rec.get("action") in ("executed", "projected") and weights is not None:
            rows = [{"instrument_id": k, "weight": round(max(0.0, min(1.0, float(v))), 12)} for k, v in sorted(weights.items()) if float(v) > 0.0]
            cash = round(max(0.0, min(1.0, float(cons.get("final_cash") or 0.0))), 12)
            if not rows:
                return None
            return {"weights": rows, "cash_weight": cash}
    return None


def plan_content_from_result(result: Mapping[str, Any], *, base_currency: str = "USD", constraints: Mapping[str, Any] | None = None, fees: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Finance ``plan-content`` from the run result's ``payload.proposed_allocation``."""
    allocation = ((result.get("payload") or {}).get("proposed_allocation")) or None
    if not allocation:
        raise FinplanError.validation("a production candidate must propose an allocation", pointer="/payload/proposed_allocation")
    content: dict[str, Any] = {"base_currency": base_currency, "allocation": dict(allocation)}
    if constraints:
        content["constraints"] = dict(constraints)
    if fees:
        content["fees"] = dict(fees)
    if result.get("synthetic"):
        content["synthetic"] = True
    return content


@dataclass
class StagedBundle:
    files: dict[str, tuple[bytes, str]]  # name -> (bytes, content type), data files only
    manifest: dict[str, Any]
    manifest_bytes: bytes = field(default=b"")

    @property
    def manifest_checksum(self) -> str:
        return sha256_checksum(self.manifest_bytes)


def build_bundle(result: Mapping[str, Any], *, target: StagingTarget, plan_content: Mapping[str, Any], written_at: str, metrics: Mapping[str, Any] | None = None) -> StagedBundle:
    """Build and validate the bundle; raises ``VALIDATION_FAILED`` before anything is written (RST-02)."""
    require_valid(dict(plan_content), "plan-content")
    perf = metrics if metrics is not None else ((result.get("payload") or {}).get("performance") or {})
    numeric = {k: float(v) for k, v in sorted(perf.items()) if isinstance(v, (int, float)) and not isinstance(v, bool)}
    files: dict[str, tuple[bytes, str]] = {
        PLAN_CONTENT_FILE: (canonical_json_bytes(dict(plan_content)), "application/json"),
        METRICS_FILE: (canonical_json_bytes({"run_id": result.get("run_id"), "metrics": numeric}), "application/json"),
    }
    manifest: dict[str, Any] = {
        "run_id": result.get("run_id"),
        "model_version": result.get("model_version"),
        "configuration_id": result.get("configuration_id"),
        "input_snapshot_id": result.get("input_snapshot_id"),
        "plan_id": target.plan_id,
        "parent_plan_version_id": target.parent_plan_version_id,
        "completion_status": result.get("completion_status"),
        "solution_status": result.get("solution_status"),
        "evaluator_version": result.get("evaluator_version"),
        "contract_version": contract_version(),
        "files": [{"name": name, "checksum": sha256_checksum(data), "size_bytes": len(data), "content_type": ct} for name, (data, ct) in sorted(files.items())],
        "domain": result.get("domain", "finance"),
        "domain_schema_version": result.get("domain_schema_version", "1.0"),
        "payload": {"plan_content": dict(plan_content), "metrics": numeric},
        "written_at": written_at,
    }
    if result.get("synthetic"):
        manifest["synthetic"] = True
        manifest["payload"]["synthetic"] = True
    manifest = {k: v for k, v in manifest.items() if v is not None or k == "parent_plan_version_id"}
    require_valid(manifest, "staged-output-manifest")
    return StagedBundle(files, manifest, canonical_json_bytes(manifest))


def stage_run_output(
    store: StagingStore,
    result: Mapping[str, Any],
    *,
    purpose: str | None,
    target: StagingTarget | None,
    plan_content: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
    registry: Any = None,
    spec: Mapping[str, Any] | None = None,
    actor: str = "financemodel-job",
) -> dict[str, Any]:
    """Stage one run's output (or not) and return the result's ``staging`` block.

    Raises ``VALIDATION_FAILED`` (nothing written) when the bundle does not validate; any failure
    while writing leaves a bundle without manifest, which the platform treats as incomplete.
    """
    stage, reason = staging_decision(purpose, result)
    if not stage:
        return {"status": "not_staged", "reason": reason}
    if target is None:
        return {"status": "not_staged", "reason": "no_plan_target"}
    clock = clock or SystemClock()
    content = dict(plan_content) if plan_content is not None else plan_content_from_result(result)
    bundle = build_bundle(result, target=target, plan_content=content, written_at=utc_iso(clock.now()))
    if registry is not None and spec is not None:
        from finplan_model.registry.resolver import record_spec_lineage

        record_spec_lineage(registry, spec, actor=actor)  # the platform checks lineage before it commits
    run_id = str(result["run_id"])
    for name, (data, ct) in sorted(bundle.files.items()):
        store.put_new(run_id, name, data, ct)
    store.put_new(run_id, MANIFEST_NAME, bundle.manifest_bytes, "application/json")  # LAST: the completion marker
    return {"status": "staged", "plan_id": target.plan_id, "manifest_checksum": bundle.manifest_checksum, "files": len(bundle.files), "platform_validation": "pending"}
