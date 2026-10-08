"""Model registry: ``model_version`` minting, immutable records, lifecycle events and run lineage
(spec model-registry; tasks 9.2 and 9.3; REG-01 to REG-04).

Storage layout (one registry per environment; :class:`S3RegistryStore` in the
``finplan-<env>-financemodel-model-registry-<account>`` bucket, :class:`InMemoryRegistryStore` in
tests). Every object is written **once** (``If-None-Match: *``); nothing is ever overwritten or
deleted (IAM denies both, :mod:`infra.stacks.policies`):

======================================  ============================================================
key                                     content
======================================  ============================================================
``identity/<identity_key>.json``        the record of the ``model_version`` minted for one identity
                                        (strategy, container image digest, parameter schema version,
                                        trained-artifact checksum); the uniqueness item that makes
                                        registration idempotent
``versions/<model_version>.json``       the same record, addressed by ``model_version``
``events/<model_version>/<seq>.json``   append-only lifecycle events (``registered`` ->
                                        ``candidate`` -> ``promoted``; any -> ``retired``) with actor
                                        and time; the current status is the last event
``runs/<run_id>.json``                  run lineage: the ``model_version`` a run used (write-once)
======================================  ============================================================

``identity_key`` is the SHA-256 of the RFC 8785 canonical identity document, so the same strategy,
digest and schema always map to the same key whatever the field order. Registering an identical
combination returns the existing record (REG-01); a changed digest mints a new ``model_version``.
Records are immutable except for the lifecycle status, which only moves by appending an event
(REG-02); an attempt to change anything else fails with ``IMMUTABLE_RECORD``.

The platform verifies a staged bundle's ``(run_id, model_version)`` pair through
:meth:`ModelRegistry.verify_lineage` (REG-03), exposed by the job API route
``GET /v1/registry/lineage/{run_id}?model_version=...`` (:mod:`finplan_model.registry.api`) and
published at ``/finplan/<env>/financemodel/model/registry-ref``.

Design D9 named a DynamoDB table; the pinned contract matrix row ``model-registry`` lists
``AWS::S3::Bucket`` (not ``AWS::DynamoDB::Table``), so the registry uses S3 conditional writes,
which give the same idempotency (create-if-absent) and immutability guarantees.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.clock import Clock, SystemClock, utc_iso
from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.ids import IdMinter, require_id

__all__ = [
    "LIFECYCLE",
    "STATUSES",
    "InMemoryRegistryStore",
    "ModelIdentity",
    "ModelRegistry",
    "RegistryStore",
    "S3RegistryStore",
]

STATUSES = ("registered", "candidate", "promoted", "retired")
#: Allowed lifecycle moves (every status may be retired; retired is final).
LIFECYCLE: dict[str, frozenset[str]] = {
    "registered": frozenset({"candidate", "retired"}),
    "candidate": frozenset({"promoted", "retired"}),
    "promoted": frozenset({"retired"}),
    "retired": frozenset(),
}
_STRATEGY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}\Z")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")
_SCHEMA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,31}\Z")
_ACTOR_RE = re.compile(r"^[\w+=,.@:/-]{1,256}\Z")


# ===================================================================== storage
@runtime_checkable
class RegistryStore(Protocol):
    def put_new(self, key: str, doc: Mapping[str, Any]) -> bool:
        """Create ``key`` with ``doc``; False (and nothing written) when it already exists."""
        ...

    def get(self, key: str) -> dict[str, Any] | None: ...

    def list(self, prefix: str) -> list[str]: ...


class InMemoryRegistryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []
        self._lock = threading.Lock()

    def put_new(self, key: str, doc: Mapping[str, Any]) -> bool:
        data = canonical_json_bytes(dict(doc))
        with self._lock:
            if key in self.objects:
                return False
            self.objects[key] = data
            self.writes.append(key)
            return True

    def get(self, key: str) -> dict[str, Any] | None:
        data = self.objects.get(key)
        return None if data is None else json.loads(data)

    def list(self, prefix: str) -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))


def _code(exc: BaseException) -> str:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return str(resp.get("Error", {}).get("Code", ""))
    return ""


class S3RegistryStore:
    """Registry objects in the environment's registry bucket. Bucket name and keys never leave this class."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._s3 = client
        self._bucket = bucket

    def put_new(self, key: str, doc: Mapping[str, Any]) -> bool:
        try:
            self._s3.put_object(Bucket=self._bucket, Key=key, Body=canonical_json_bytes(dict(doc)), ContentType="application/json", IfNoneMatch="*")
        except Exception as exc:  # noqa: BLE001 - botocore ClientError
            if _code(exc) in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                return False
            raise FinplanError.dependency_unavailable("the model registry could not be written; retry", retryable=True) from None
        return True

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _code(exc) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise FinplanError.dependency_unavailable("the model registry could not be read; retry", retryable=True) from None
        return json.loads(obj["Body"].read())

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token = None
        while True:
            kw: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            try:
                resp = self._s3.list_objects_v2(**kw)
            except Exception:  # noqa: BLE001
                raise FinplanError.dependency_unavailable("the model registry could not be listed; retry", retryable=True) from None
            keys += [o["Key"] for o in resp.get("Contents") or []]
            if not resp.get("IsTruncated"):
                return sorted(keys)
            token = resp.get("NextContinuationToken")


# ===================================================================== identity
@dataclass(frozen=True)
class ModelIdentity:
    """What makes a strategy implementation distinct (spec model-registry, "model_version minting")."""

    strategy: str
    image_digest: str
    param_schema_version: str
    artifact_checksum: str | None = None

    def __post_init__(self) -> None:
        if not _STRATEGY_RE.match(self.strategy or ""):
            raise FinplanError.validation("strategy must be a lowercase strategy name", pointer="/strategy")
        if not _DIGEST_RE.match(self.image_digest or ""):
            raise FinplanError.validation("image_digest must be a sha256 container digest", pointer="/image_digest")
        if not _SCHEMA_RE.match(str(self.param_schema_version or "")):
            raise FinplanError.validation("param_schema_version is not a version string", pointer="/param_schema_version")
        if self.artifact_checksum is not None and not _DIGEST_RE.match(self.artifact_checksum):
            raise FinplanError.validation("artifact_checksum must be a sha256 checksum", pointer="/artifact_checksum")

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"strategy": self.strategy, "image_digest": self.image_digest, "param_schema_version": str(self.param_schema_version)}
        if self.artifact_checksum:
            doc["artifact_checksum"] = self.artifact_checksum
        return doc

    @property
    def key(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


# ===================================================================== registry
class ModelRegistry:
    def __init__(self, store: RegistryStore, *, clock: Clock | None = None, ids: IdMinter | None = None) -> None:
        self.store = store
        self.clock = clock or SystemClock()
        self.ids = ids or IdMinter(self.clock)

    # ---------------------------------------------------------------- minting (REG-01)
    def register(self, identity: ModelIdentity, *, actor: str, synthetic: bool | None = None) -> tuple[dict[str, Any], bool]:
        """``(record, created)``. An identical identity returns the existing record and creates nothing."""
        _actor(actor)
        ikey = f"identity/{identity.key}.json"
        existing = self.store.get(ikey)
        if existing is not None:
            self._ensure_version_copy(existing)
            return self._with_status(existing), False
        mv = self.ids.model_version()
        record: dict[str, Any] = {"model_version": mv, **identity.document(), "identity_key": identity.key, "registered_at": utc_iso(self.clock.now()), "registered_by": actor}
        if synthetic is not None:
            record["synthetic"] = bool(synthetic)
        record["record_checksum"] = sha256_checksum(canonical_json_bytes(record))
        if not self.store.put_new(ikey, record):  # lost a concurrent registration of the same identity
            winner = self.store.get(ikey)
            if winner is None:  # pragma: no cover - the store reported the key as present
                raise FinplanError(ErrorCode.CONFLICT, "concurrent registration; retry", retryable=True)
            self._ensure_version_copy(winner)
            return self._with_status(winner), False
        self._ensure_version_copy(record)
        self.store.put_new(f"events/{mv}/{1:06d}.json", {"model_version": mv, "seq": 1, "from": None, "to": "registered", "actor": actor, "at": record["registered_at"]})
        return self._with_status(record), True

    def _ensure_version_copy(self, record: Mapping[str, Any]) -> None:
        key = f"versions/{record['model_version']}.json"
        if not self.store.put_new(key, record):
            stored = self.store.get(key)
            if stored is not None and stored.get("record_checksum") != record.get("record_checksum"):
                raise FinplanError(ErrorCode.IMMUTABLE_RECORD, "the stored model_version record differs from its identity record", details={"model_version": record["model_version"]})

    # ---------------------------------------------------------------- reads
    def get(self, model_version: str) -> dict[str, Any]:
        require_id("model_version", model_version)
        record = self.store.get(f"versions/{model_version}.json")
        if record is None:
            raise FinplanError(ErrorCode.NOT_FOUND, "model_version is not registered", details={"record_type": "model_version"})
        return self._with_status(record)

    def events(self, model_version: str) -> list[dict[str, Any]]:
        out = []
        for key in self.store.list(f"events/{model_version}/"):
            doc = self.store.get(key)
            if doc is not None:
                out.append(doc)
        return sorted(out, key=lambda e: int(e["seq"]))

    def status(self, model_version: str) -> str:
        ev = self.events(model_version)
        return str(ev[-1]["to"]) if ev else "registered"

    def _with_status(self, record: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(record)
        out["status"] = self.status(str(record["model_version"]))
        return out

    def find(self, identity: ModelIdentity) -> dict[str, Any] | None:
        rec = self.store.get(f"identity/{identity.key}.json")
        return None if rec is None else self._with_status(rec)

    # ---------------------------------------------------------------- immutability and lifecycle (REG-02)
    def update(self, model_version: str, changes: Mapping[str, Any], *, actor: str) -> dict[str, Any]:
        """Only ``status`` may change (by an appended event); anything else is ``IMMUTABLE_RECORD``."""
        record = self.get(model_version)
        immutable = sorted(k for k in changes if k != "status")
        if immutable:
            raise FinplanError(ErrorCode.IMMUTABLE_RECORD, "registry records are immutable; register a new model_version instead", details={"fields": immutable[:10], "model_version": model_version})
        if "status" in changes and changes["status"] != record["status"]:
            return self.transition(model_version, str(changes["status"]), actor=actor)
        return record

    def transition(self, model_version: str, to: str, *, actor: str, reason: str | None = None) -> dict[str, Any]:
        _actor(actor)
        if to not in STATUSES:
            raise FinplanError.validation("unknown lifecycle status", pointer="/status", allowed=list(STATUSES))
        record = self.get(model_version)
        current = record["status"]
        if to not in LIFECYCLE[current]:
            raise FinplanError.precondition("lifecycle transition not allowed", reason="invalid_lifecycle_transition", current=current, requested=to)
        seq = len(self.events(model_version)) + 1
        event: dict[str, Any] = {"model_version": model_version, "seq": seq, "from": current, "to": to, "actor": actor, "at": utc_iso(self.clock.now())}
        if reason:
            event["reason"] = str(reason)[:256]
        if not self.store.put_new(f"events/{model_version}/{seq:06d}.json", event):
            raise FinplanError(ErrorCode.CONFLICT, "a concurrent lifecycle change was recorded first; retry", retryable=True)
        return self.get(model_version)

    # ---------------------------------------------------------------- lineage (REG-03)
    def record_run(self, run_id: str, model_version: str, *, actor: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Record that ``run_id`` used ``model_version`` (write-once; idempotent for the same pair)."""
        require_id("run_id", run_id)
        self.get(model_version)  # NOT_FOUND for an unregistered model_version
        doc: dict[str, Any] = {"run_id": run_id, "model_version": model_version, "recorded_at": utc_iso(self.clock.now()), "recorded_by": _actor(actor)}
        for k in ("configuration_id", "input_snapshot_id", "purpose", "job_type"):
            if details and details.get(k):
                doc[k] = str(details[k])
        key = f"runs/{run_id}.json"
        if self.store.put_new(key, doc):
            return doc
        existing = self.store.get(key) or {}
        if existing.get("model_version") != model_version:
            raise FinplanError(ErrorCode.IMMUTABLE_RECORD, "the run's lineage is already recorded with another model_version", details={"run_id": run_id})
        return existing

    def verify_lineage(self, run_id: str, model_version: str) -> dict[str, Any]:
        """The platform's lineage check: ``run_id`` used ``model_version``, else ``NOT_FOUND``.

        ``details.record_type`` tells the caller what is missing: ``model_version`` (not registered
        here) or ``run_lineage`` (the run is unknown, or used another model_version).
        """
        require_id("run_id", run_id)
        require_id("model_version", model_version)
        record = self.get(model_version)
        lineage = self.store.get(f"runs/{run_id}.json")
        if lineage is None or lineage.get("model_version") != model_version:
            raise FinplanError(ErrorCode.NOT_FOUND, "no run with that model_version is recorded in this environment's registry", details={"record_type": "run_lineage"})
        return {"run_id": run_id, "model_version": model_version, "matches": True, "strategy": record["strategy"], "image_digest": record["image_digest"], "param_schema_version": record["param_schema_version"], "status": record["status"], "recorded_at": lineage["recorded_at"]}


def _actor(actor: str) -> str:
    if not isinstance(actor, str) or not _ACTOR_RE.match(actor):
        raise FinplanError.validation("actor must identify a principal", pointer="/actor")
    return actor
