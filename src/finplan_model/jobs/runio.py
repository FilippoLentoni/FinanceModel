"""Run hand-off documents between the control plane and the job container.

The control plane writes the **run spec** (``runs/<run_id>/spec.json``) before it starts the
SageMaker job and passes only its SHA-256 to the container (``FINPLAN_RUN_SPEC_SHA256``); the
container refuses a spec whose checksum differs. The container writes the **job result**
(``runs/<run_id>/job-result.json``, contract ``core/v1/job-result.json``) last; the state-change
handler reads it when SageMaker reports the job terminal (``solution_status`` comes from this file,
never from the SageMaker status).

Both live in FinanceModel research storage. Bucket names and keys never leave this module: API
responses carry identifiers and trusted artifact references only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum, verify_checksum
from finplan_model.core.errors import ErrorCode, FinplanError

__all__ = ["InMemoryRunIO", "LocalRunIO", "RunIO", "S3RunIO"]

SPEC = "spec.json"
RESULT = "job-result.json"


@runtime_checkable
class RunIO(Protocol):
    def put_spec(self, run_id: str, spec: dict[str, Any]) -> str: ...

    def get_spec(self, run_id: str, expected_checksum: str | None) -> dict[str, Any]: ...

    def put_result(self, run_id: str, doc: dict[str, Any]) -> str: ...

    def get_result(self, run_id: str) -> dict[str, Any] | None: ...


class _Base:
    def _read(self, run_id: str, name: str) -> bytes | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _write(self, run_id: str, name: str, data: bytes) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def put_spec(self, run_id: str, spec: dict[str, Any]) -> str:
        data = canonical_json_bytes(spec)
        self._write(run_id, SPEC, data)
        return sha256_checksum(data)

    def get_spec(self, run_id: str, expected_checksum: str | None) -> dict[str, Any]:
        data = self._read(run_id, SPEC)
        if data is None:
            raise FinplanError.precondition("the run specification is missing", reason="run_spec_missing")
        if expected_checksum:
            verify_checksum(data, expected_checksum, what="run spec")
        return json.loads(data)

    def put_result(self, run_id: str, doc: dict[str, Any]) -> str:
        data = canonical_json_bytes(doc)
        self._write(run_id, RESULT, data)
        return sha256_checksum(data)

    def get_result(self, run_id: str) -> dict[str, Any] | None:
        data = self._read(run_id, RESULT)
        if data is None:
            return None
        try:
            doc = json.loads(data)
        except json.JSONDecodeError:
            return None
        return doc if isinstance(doc, dict) else None


class InMemoryRunIO(_Base):
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}

    def _read(self, run_id: str, name: str) -> bytes | None:
        return self.files.get((run_id, name))

    def _write(self, run_id: str, name: str, data: bytes) -> None:
        self.files[(run_id, name)] = data


class LocalRunIO(_Base):
    """``<root>/runs/<run_id>/{spec.json,job-result.json}`` (offline container runs)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, run_id: str, name: str) -> Path:
        if "/" in run_id or ".." in run_id:
            raise FinplanError(ErrorCode.INVALID_IDENTIFIER, "run_id has the wrong format", field="run_id")
        return self.root / "runs" / run_id / name

    def _read(self, run_id: str, name: str) -> bytes | None:
        p = self._path(run_id, name)
        return p.read_bytes() if p.exists() else None

    def _write(self, run_id: str, name: str, data: bytes) -> None:
        p = self._path(run_id, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)


class S3RunIO(_Base):
    """Research storage: ``runs/<run_id>/...`` in the research bucket (bucket from SSM, never a literal;
    encryption is the bucket default)."""

    def __init__(self, client: Any, bucket: str, prefix: str = "runs/") -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, run_id: str, name: str) -> str:
        return f"{self.prefix}{run_id}/{name}"

    def _read(self, run_id: str, name: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(run_id, name))["Body"].read()
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise FinplanError.dependency_unavailable("research storage could not be read; retry", retryable=True) from None

    def _write(self, run_id: str, name: str, data: bytes) -> None:
        try:
            self.client.put_object(Bucket=self.bucket, Key=self._key(run_id, name), Body=data, ContentType="application/json")
        except Exception:  # noqa: BLE001
            raise FinplanError.dependency_unavailable("research storage could not be written; retry", retryable=True) from None
