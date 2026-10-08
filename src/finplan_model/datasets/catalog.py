"""Dataset catalog: ``dataset_key -> lineage record`` with create-once semantics (DS-01).

The dataset key is a SHA-256 of the sorted input snapshot IDs, the preparation ``configuration_id``,
the container image digest and the preparation code version. Preparing the same inputs again finds
the record here and **reuses** the dataset instead of writing a new one.

Records are immutable: :meth:`DatasetCatalog.create` stores a record only if the key is absent and
otherwise returns the existing record (first writer wins). Each stored record carries a
``record_checksum`` (SHA-256 of its canonical form without that field) that is verified on read.

Implementations: :class:`InMemoryDatasetCatalog` (tests), :class:`LocalDatasetCatalog` (offline
container runs; one JSON file per key, created with ``O_EXCL``) and :class:`S3DatasetCatalog`
(research storage; conditional ``PutObject`` with ``If-None-Match: *``; bucket from configuration,
never returned to callers).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import ErrorCode, FinplanError

__all__ = ["DatasetCatalog", "InMemoryDatasetCatalog", "LocalDatasetCatalog", "S3DatasetCatalog", "seal_record", "verify_record"]


def seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in record.items() if k != "record_checksum"}
    return {**body, "record_checksum": sha256_checksum(canonical_json_bytes(body))}


def verify_record(record: Mapping[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in record.items() if k != "record_checksum"}
    if record.get("record_checksum") != sha256_checksum(canonical_json_bytes(body)):
        raise FinplanError(ErrorCode.PRECONDITION_FAILED, "dataset record checksum does not match its content", details={"reason": "dataset_record_checksum_mismatch"})
    return dict(record)


@runtime_checkable
class DatasetCatalog(Protocol):
    def get(self, dataset_key: str) -> dict[str, Any] | None: ...

    def create(self, dataset_key: str, record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        """Store ``record`` if absent; returns ``(stored_record, created)``."""
        ...


class InMemoryDatasetCatalog:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def get(self, dataset_key: str) -> dict[str, Any] | None:
        rec = self.records.get(dataset_key)
        return None if rec is None else verify_record(json.loads(json.dumps(rec)))

    def create(self, dataset_key: str, record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        if dataset_key in self.records:
            return self.get(dataset_key), False  # type: ignore[return-value]
        sealed = seal_record(record)
        self.records[dataset_key] = json.loads(canonical_json_bytes(sealed))
        return sealed, True


class LocalDatasetCatalog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if not key.isalnum():
            raise ValueError("dataset key must be hex")
        return self.root / "dataset-catalog" / f"{key}.json"

    def get(self, dataset_key: str) -> dict[str, Any] | None:
        p = self._path(dataset_key)
        if not p.is_file():
            return None
        return verify_record(json.loads(p.read_bytes()))

    def create(self, dataset_key: str, record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        p = self._path(dataset_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        sealed = seal_record(record)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return self.get(dataset_key), False  # type: ignore[return-value]
        with os.fdopen(fd, "wb") as fh:
            fh.write(canonical_json_bytes(sealed))
        return sealed, True


class S3DatasetCatalog:
    """Records at ``<prefix>dataset-catalog/<key>.json`` in research storage (conditional create)."""

    def __init__(self, client: Any, bucket: str, prefix: str = "") -> None:
        self._s3 = client
        self._bucket = bucket
        self._prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"

    def _key(self, dataset_key: str) -> str:
        if not dataset_key.isalnum():
            raise ValueError("dataset key must be hex")
        return f"{self._prefix}dataset-catalog/{dataset_key}.json"

    def get(self, dataset_key: str) -> dict[str, Any] | None:
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(dataset_key))
        except Exception as exc:  # noqa: BLE001 - botocore ClientError
            if _code(exc) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        return verify_record(json.loads(obj["Body"].read()))

    def create(self, dataset_key: str, record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        sealed = seal_record(record)
        try:
            self._s3.put_object(Bucket=self._bucket, Key=self._key(dataset_key), Body=canonical_json_bytes(sealed), ContentType="application/json", IfNoneMatch="*")
        except Exception as exc:  # noqa: BLE001
            if _code(exc) in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                existing = self.get(dataset_key)
                if existing is not None:
                    return existing, False
            raise
        return sealed, True


def _code(exc: BaseException) -> str:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return str(resp.get("Error", {}).get("Code", ""))
    return ""
