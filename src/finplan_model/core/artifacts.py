"""Research artifact storage behind trusted artifact references (contracts ``core/v1/artifact-ref.json``).

Results, reports and dataset records reference stored bytes only through an :class:`ArtifactRef`
(``artifact_id``, ``owner`` = ``financemodel``, ``kind``, ``checksum``, ``content_type``). Bucket
names and object keys never leave the store implementation (spec experiment-job-interface,
"Result retrieval"; research-workspace, "FinanceModel-owned research storage").

Implementations of the :class:`ArtifactStore` protocol:

* :class:`InMemoryArtifactStore` - unit tests;
* :class:`LocalArtifactStore` - offline container runs (a directory, for example the SageMaker
  processing output path);
* :class:`S3ArtifactStore` - research storage; the bucket comes from configuration at run time and
  the boto3 client is injected.

Artifact IDs are content-addressed by default (``<kind>_<first 40 hex of the SHA-256>``), so writing
the same bytes twice is idempotent and deterministic (DS-01, REP-06). Reads re-verify the
checksum; a mismatch is ``PRECONDITION_FAILED`` (``artifact_checksum_mismatch``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from finplan_contracts.canonical import canonicalize

from .errors import ErrorCode, FinplanError

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "InMemoryArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "canonical_json_bytes",
    "sha256_checksum",
    "verify_checksum",
]

OWNER = "financemodel"
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}\Z")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,127}\Z")
_CT_RE = re.compile(r"^[a-z]+/[a-z0-9][a-z0-9.+-]*\Z")


def sha256_checksum(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    """RFC 8785 canonical JSON (contract canonicalizer): byte-identical for equal documents."""
    return canonicalize(obj)


def verify_checksum(data: bytes, expected: str, *, what: str = "artifact") -> None:
    actual = sha256_checksum(data)
    if actual != expected:
        raise FinplanError(ErrorCode.PRECONDITION_FAILED, f"{what} checksum does not match its reference", details={"reason": f"{what.replace(' ', '_')}_checksum_mismatch"})


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    checksum: str
    content_type: str
    size_bytes: int | None = None
    owner: str = OWNER
    domain: str | None = None
    synthetic: bool | None = None

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.artifact_id):
            raise ValueError("artifact_id must match the contract artifact_id format")
        if not _KIND_RE.match(self.kind):
            raise ValueError("kind must be lowercase snake case")
        if not _CT_RE.match(self.content_type):
            raise ValueError("content_type must be a media type")
        if not re.match(r"^sha256:[0-9a-f]{64}\Z", self.checksum):
            raise ValueError("checksum must be sha256:<64 hex>")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"artifact_id": self.artifact_id, "owner": self.owner, "kind": self.kind, "checksum": self.checksum, "content_type": self.content_type}
        if self.size_bytes is not None:
            d["size_bytes"] = self.size_bytes
        if self.domain is not None:
            d["domain"] = self.domain
        if self.synthetic is not None:
            d["synthetic"] = self.synthetic
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=str(d["artifact_id"]),
            kind=str(d["kind"]),
            checksum=str(d["checksum"]),
            content_type=str(d["content_type"]),
            size_bytes=d.get("size_bytes"),
            owner=str(d.get("owner", OWNER)),
            domain=d.get("domain"),
            synthetic=d.get("synthetic"),
        )


@runtime_checkable
class ArtifactStore(Protocol):
    def put(self, data: bytes, *, kind: str, content_type: str = "application/json", artifact_id: str | None = None, synthetic: bool | None = None, domain: str | None = None) -> ArtifactRef: ...

    def get(self, ref: ArtifactRef | Mapping[str, Any]) -> bytes: ...

    def exists(self, ref: ArtifactRef | Mapping[str, Any]) -> bool: ...


class _BaseStore:
    def _ref(self, data: bytes, kind: str, content_type: str, artifact_id: str | None, synthetic: bool | None, domain: str | None) -> ArtifactRef:
        checksum = sha256_checksum(data)
        aid = artifact_id or f"{kind}_{checksum.split(':', 1)[1][:40]}"
        return ArtifactRef(artifact_id=aid, kind=kind, checksum=checksum, content_type=content_type, size_bytes=len(data), synthetic=synthetic, domain=domain)

    def put_json(self, obj: Any, *, kind: str, synthetic: bool | None = None, domain: str | None = None, artifact_id: str | None = None) -> ArtifactRef:
        return self.put(canonical_json_bytes(obj), kind=kind, content_type="application/json", artifact_id=artifact_id, synthetic=synthetic, domain=domain)  # type: ignore[attr-defined]

    def get_json(self, ref: ArtifactRef | Mapping[str, Any]) -> Any:
        return json.loads(self.get(ref))  # type: ignore[attr-defined]

    @staticmethod
    def _as_ref(ref: ArtifactRef | Mapping[str, Any]) -> ArtifactRef:
        return ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_dict(ref)

    @staticmethod
    def _immutable(existing_checksum: str, ref: ArtifactRef) -> None:
        if existing_checksum != ref.checksum:
            raise FinplanError(ErrorCode.IMMUTABLE_RECORD, "artifact already exists with different content", details={"artifact_id": ref.artifact_id})


class InMemoryArtifactStore(_BaseStore):
    def __init__(self) -> None:
        self.objects: dict[str, tuple[ArtifactRef, bytes]] = {}

    def put(self, data: bytes, *, kind: str, content_type: str = "application/json", artifact_id: str | None = None, synthetic: bool | None = None, domain: str | None = None) -> ArtifactRef:
        ref = self._ref(data, kind, content_type, artifact_id, synthetic, domain)
        if ref.artifact_id in self.objects:
            self._immutable(self.objects[ref.artifact_id][0].checksum, ref)
            return self.objects[ref.artifact_id][0]
        self.objects[ref.artifact_id] = (ref, bytes(data))
        return ref

    def get(self, ref: ArtifactRef | Mapping[str, Any]) -> bytes:
        r = self._as_ref(ref)
        if r.artifact_id not in self.objects:
            raise FinplanError(ErrorCode.NOT_FOUND, "artifact not found", details={"artifact_id": r.artifact_id})
        data = self.objects[r.artifact_id][1]
        verify_checksum(data, r.checksum)
        return data

    def exists(self, ref: ArtifactRef | Mapping[str, Any]) -> bool:
        return self._as_ref(ref).artifact_id in self.objects


class LocalArtifactStore(_BaseStore):
    """Artifacts as files ``<root>/<kind>/<artifact_id>`` (offline container runs)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, ref: ArtifactRef) -> Path:
        return self.root / ref.kind / ref.artifact_id

    def put(self, data: bytes, *, kind: str, content_type: str = "application/json", artifact_id: str | None = None, synthetic: bool | None = None, domain: str | None = None) -> ArtifactRef:
        ref = self._ref(data, kind, content_type, artifact_id, synthetic, domain)
        path = self._path(ref)
        if path.exists():
            self._immutable(sha256_checksum(path.read_bytes()), ref)
            return ref
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return ref

    def get(self, ref: ArtifactRef | Mapping[str, Any]) -> bytes:
        r = self._as_ref(ref)
        path = self._path(r)
        if not path.is_file():
            raise FinplanError(ErrorCode.NOT_FOUND, "artifact not found", details={"artifact_id": r.artifact_id})
        data = path.read_bytes()
        verify_checksum(data, r.checksum)
        return data

    def exists(self, ref: ArtifactRef | Mapping[str, Any]) -> bool:
        return self._path(self._as_ref(ref)).is_file()


class S3ArtifactStore(_BaseStore):
    """Research storage. ``bucket`` comes from ``/finplan/<env>/financemodel/config/*`` at run time;
    keys are ``<prefix><kind>/<artifact_id>`` and are never returned to callers."""

    def __init__(self, client: Any, bucket: str, prefix: str = "artifacts/") -> None:
        self._s3 = client
        self._bucket = bucket
        self._prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"

    def _key(self, ref: ArtifactRef) -> str:
        return f"{self._prefix}{ref.kind}/{ref.artifact_id}"

    def put(self, data: bytes, *, kind: str, content_type: str = "application/json", artifact_id: str | None = None, synthetic: bool | None = None, domain: str | None = None) -> ArtifactRef:
        ref = self._ref(data, kind, content_type, artifact_id, synthetic, domain)
        key = self._key(ref)
        try:
            head = self._s3.head_object(Bucket=self._bucket, Key=key)
            self._immutable(head.get("Metadata", {}).get("checksum", ""), ref)
            return ref
        except FinplanError:
            raise
        except Exception as exc:  # noqa: BLE001 - botocore ClientError 404 means "absent"
            if _status(exc) != 404:
                raise
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type, Metadata={"checksum": ref.checksum, "kind": kind})
        return ref

    def get(self, ref: ArtifactRef | Mapping[str, Any]) -> bytes:
        r = self._as_ref(ref)
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(r))
        except Exception as exc:  # noqa: BLE001
            if _status(exc) == 404:
                raise FinplanError(ErrorCode.NOT_FOUND, "artifact not found", details={"artifact_id": r.artifact_id}) from None
            raise
        data = obj["Body"].read()
        verify_checksum(data, r.checksum)
        return data

    def exists(self, ref: ArtifactRef | Mapping[str, Any]) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(self._as_ref(ref)))
            return True
        except Exception as exc:  # noqa: BLE001
            if _status(exc) == 404:
                return False
            raise


def _status(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = resp.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return 404
        return int(resp.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)) or None
    return None
