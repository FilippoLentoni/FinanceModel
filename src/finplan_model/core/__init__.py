"""Shared core interfaces used by every FinanceModel package (FOUNDATION-owned)."""

from .artifacts import ArtifactRef, ArtifactStore, InMemoryArtifactStore, LocalArtifactStore, S3ArtifactStore, canonical_json_bytes, sha256_checksum
from .clock import Clock, FrozenClock, SystemClock, parse_utc, utc_iso
from .context import EVALUATOR_VERSION, PURPOSES, REPO, RunContext
from .errors import ErrorCode, FinplanError, as_finplan_error, contract_version
from .ids import IdMinter, configuration_id, request_hash, require_id
from .platform import FixturePlatformClient, HttpPlatformClient, PlatformClient, SnapshotReader, build_synthetic_snapshot

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "Clock",
    "EVALUATOR_VERSION",
    "ErrorCode",
    "FinplanError",
    "FixturePlatformClient",
    "FrozenClock",
    "HttpPlatformClient",
    "IdMinter",
    "InMemoryArtifactStore",
    "LocalArtifactStore",
    "PURPOSES",
    "PlatformClient",
    "REPO",
    "RunContext",
    "S3ArtifactStore",
    "SnapshotReader",
    "SystemClock",
    "as_finplan_error",
    "build_synthetic_snapshot",
    "canonical_json_bytes",
    "configuration_id",
    "contract_version",
    "parse_utc",
    "request_hash",
    "require_id",
    "sha256_checksum",
    "utc_iso",
]
