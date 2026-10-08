"""Platform API access: approved input snapshots and staged-output outcomes (injected client).

FinanceModel reads market data **only** through approved platform snapshots, by
``input_snapshot_id`` (spec research-workspace, "Read-only access to approved platform snapshots";
research-datasets, "Market data only through approved platform snapshots"). This module defines:

* :class:`PlatformClient` - the protocol of the platform plan-API routes FinanceModel uses:
  ``GET /v1/snapshots/{id}`` (``download=true`` adds time-limited grants),
  ``GET /v1/snapshots/{id}/observations`` and ``GET /v1/staged-outputs/{run_id}``;
* :class:`HttpPlatformClient` - the SigV4 implementation (endpoint resolved from
  ``/finplan/<env>/financialplanning/api/plan-endpoint`` by the caller, never a literal);
* :class:`FixturePlatformClient` - the in-process mock provider serving synthetic snapshots built
  with :func:`build_synthetic_snapshot` (unit tests, CI and offline container runs; no network);
* :class:`SnapshotReader` - resolves a snapshot, refuses anything not ``approved``
  (``PRECONDITION_FAILED`` ``snapshot_not_approved``), downloads its manifest and payload and
  re-verifies every SHA-256 before any strategy code can see the data (WS-04:
  ``PRECONDITION_FAILED`` ``snapshot_checksum_mismatch``).

Snapshot layout (platform ingestion): the record carries ``manifest_checksum`` and two trusted
references, ``snapshot_manifest`` and ``snapshot_payload``; the manifest (canonical JSON) names the
payload's ``artifact_id`` and ``checksum``; the payload validates against
``finance/v1/snapshot-payload.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

from finplan_contracts.validate import validate as contract_validate

from .artifacts import canonical_json_bytes, sha256_checksum
from .clock import Clock, SystemClock, utc_iso
from .errors import ErrorCode, FinplanError
from .ids import require_id

__all__ = [
    "FixturePlatformClient",
    "HttpPlatformClient",
    "PlatformClient",
    "ResolvedSnapshot",
    "SnapshotContent",
    "SnapshotReader",
    "build_synthetic_snapshot",
]

MANIFEST_KIND = "snapshot_manifest"
PAYLOAD_KIND = "snapshot_payload"
DOMAIN = "finance"
DOMAIN_SCHEMA_VERSION = "1.0"


@runtime_checkable
class PlatformClient(Protocol):
    def get_snapshot(self, input_snapshot_id: str, *, download: bool = False) -> dict[str, Any]: ...

    def read_snapshot_observations(self, input_snapshot_id: str, *, instrument_id: str | None = None, start_date: str | None = None, end_date: str | None = None, page_size: int = 100, next_token: str | None = None) -> dict[str, Any]: ...

    def download(self, grant: Mapping[str, Any]) -> bytes: ...

    def get_staged_output(self, run_id: str) -> dict[str, Any]: ...


# ===================================================================== reader
@dataclass(frozen=True)
class ResolvedSnapshot:
    input_snapshot_id: str
    record: dict[str, Any]

    @property
    def synthetic(self) -> bool:
        return bool(self.record.get("synthetic"))

    @property
    def manifest_checksum(self) -> str:
        return str(self.record["manifest_checksum"])

    @property
    def lineage(self) -> dict[str, Any]:
        return dict(self.record.get("lineage") or {})

    @property
    def quality_flags(self) -> list[str]:
        return list(self.record.get("quality_flags") or [])

    @property
    def dataset_id(self) -> str:
        return str((self.record.get("dataset") or {}).get("dataset_id", ""))


@dataclass(frozen=True)
class SnapshotContent:
    snapshot: ResolvedSnapshot
    manifest: dict[str, Any]
    payload: dict[str, Any]
    #: artifact_id -> verified checksum of every artifact read
    verified: dict[str, str] = field(default_factory=dict)


class SnapshotReader:
    """Approved-only, checksum-verified snapshot reads through an injected :class:`PlatformClient`."""

    def __init__(self, client: PlatformClient, *, validate_payload: bool = True) -> None:
        self.client = client
        self.validate_payload = validate_payload

    def resolve(self, input_snapshot_id: str) -> ResolvedSnapshot:
        require_id("input_snapshot_id", input_snapshot_id)
        resp = self.client.get_snapshot(input_snapshot_id)
        record = dict(resp.get("snapshot") or {})
        return self._check_record(input_snapshot_id, record)

    def _check_record(self, input_snapshot_id: str, record: dict[str, Any]) -> ResolvedSnapshot:
        if record.get("input_snapshot_id") != input_snapshot_id:
            raise FinplanError.internal("platform returned a different snapshot than requested")
        result = contract_validate(record, "input-snapshot")
        if not result.valid:
            first = result.issues[0]
            raise FinplanError.dependency_unavailable("platform snapshot record does not validate against the pinned contract", retryable=False, reason="snapshot_record_invalid", pointer=first.pointer)
        if record.get("status") != "approved":
            raise FinplanError.precondition("snapshot is not approved for FinanceModel use", reason="snapshot_not_approved", input_snapshot_id=input_snapshot_id, status=str(record.get("status")))
        return ResolvedSnapshot(input_snapshot_id, record)

    def load(self, snapshot: ResolvedSnapshot | str) -> SnapshotContent:
        """Download manifest and payload through grants and verify every checksum (WS-04)."""
        snap = self.resolve(snapshot) if isinstance(snapshot, str) else snapshot
        resp = self.client.get_snapshot(snap.input_snapshot_id, download=True)
        record = dict(resp.get("snapshot") or {})
        if record.get("manifest_checksum") != snap.manifest_checksum:
            raise FinplanError.precondition("snapshot record changed between reads", reason="snapshot_checksum_mismatch")
        snap = self._check_record(snap.input_snapshot_id, record)
        grants = {str(g["artifact_id"]): g for g in resp.get("download_grants") or []}
        refs = {str(r.get("kind")): r for r in record.get("artifacts") or []}
        if MANIFEST_KIND not in refs or PAYLOAD_KIND not in refs:
            raise FinplanError.precondition("snapshot lacks manifest or payload references", reason="snapshot_incomplete")
        verified: dict[str, str] = {}

        def fetch(ref: Mapping[str, Any], expected: Iterable[str], what: str) -> bytes:
            grant = grants.get(str(ref["artifact_id"]))
            if grant is None:
                raise FinplanError.dependency_unavailable(f"no download grant for the snapshot {what}", reason="snapshot_artifact_unavailable")
            data = self.client.download(grant)
            actual = sha256_checksum(data)
            for exp in expected:
                if actual != exp:
                    raise FinplanError.precondition(f"snapshot {what} checksum does not match the manifest", reason="snapshot_checksum_mismatch", artifact_kind=str(ref.get("kind")))
            verified[str(ref["artifact_id"])] = actual
            return data

        manifest_bytes = fetch(refs[MANIFEST_KIND], (refs[MANIFEST_KIND]["checksum"], snap.manifest_checksum), "manifest")
        manifest = json.loads(manifest_bytes)
        payload_meta = manifest.get("payload") or {}
        payload_ref = refs[PAYLOAD_KIND]
        if payload_meta.get("artifact_id") not in (None, payload_ref["artifact_id"]):
            raise FinplanError.precondition("snapshot manifest names a different payload", reason="snapshot_checksum_mismatch")
        payload_bytes = fetch(payload_ref, (payload_ref["checksum"], *( [payload_meta["checksum"]] if payload_meta.get("checksum") else [])), "payload")
        payload = json.loads(payload_bytes)
        if self.validate_payload:
            res = contract_validate(payload, "snapshot-payload")
            if not res.valid:
                raise FinplanError.validation("snapshot payload does not validate against finance/v1/snapshot-payload", pointer=res.issues[0].pointer)
        return SnapshotContent(snap, manifest, payload, verified)

    def read_observations(self, input_snapshot_id: str, *, instrument_id: str | None = None, start_date: str | None = None, end_date: str | None = None, page_size: int = 100) -> list[dict[str, Any]]:
        """All observations in range through the bounded, paginated observations route."""
        self.resolve(input_snapshot_id)
        out: list[dict[str, Any]] = []
        token: str | None = None
        for _ in range(100_000):
            page = self.client.read_snapshot_observations(input_snapshot_id, instrument_id=instrument_id, start_date=start_date, end_date=end_date, page_size=page_size, next_token=token)
            out.extend(page.get("observations") or [])
            token = page.get("next_token")
            if not token:
                return out
        raise FinplanError.internal("observation pagination did not terminate")


# ===================================================================== synthetic fixtures
def build_synthetic_snapshot(
    payload: Mapping[str, Any],
    *,
    input_snapshot_id: str,
    status: str = "approved",
    provider: str = "fixture",
    retrieved_at: str = "2026-01-05T22:00:00Z",
    created_at: str | None = None,
    lineage_extra: Mapping[str, Any] | None = None,
    quality_flags: Iterable[str] = (),
    quality_details: Mapping[str, Any] | None = None,
    calendar: Mapping[str, Any] | None = None,
    synthetic: bool = True,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """A platform-shaped snapshot record plus its artifact bytes (``artifact_id -> bytes``).

    Mirrors the platform ingestion layout. ``synthetic`` must stay true for anything committed to
    this public repository (DS-13); the flag is a parameter only so tests can model a real-data
    record shape.
    """
    payload = dict(payload)
    if synthetic:
        payload.setdefault("synthetic", True)
    payload_bytes = canonical_json_bytes(payload)
    payload_ck = sha256_checksum(payload_bytes)
    sid_suffix = input_snapshot_id.split("_", 1)[1]
    payload_id = f"art_payload_{sid_suffix}"
    manifest_id = f"art_manifest_{sid_suffix}"
    dates = sorted(str(o["session_date"]) for o in payload.get("observations", []))
    coverage = {"start": dates[0], "end": dates[-1]} if dates else {"start": "2026-01-05", "end": "2026-01-05"}
    dataset_id = str(payload.get("dataset_id", "finance/etf-daily/SPY"))
    lineage = {"provider": provider, "retrieved_at": retrieved_at, **dict(lineage_extra or {})}
    manifest: dict[str, Any] = {
        "manifest_version": "snapshot-manifest-v1",
        "input_snapshot_id": input_snapshot_id,
        "domain": DOMAIN,
        "domain_schema_version": DOMAIN_SCHEMA_VERSION,
        "dataset": {"dataset_id": dataset_id},
        "lineage": lineage,
        "calendar": dict(calendar or {"exchange": payload.get("calendar", "XNYS"), "version": "synthetic-1", "synthetic": synthetic}),
        "coverage": coverage,
        "quality_flags": sorted(set(quality_flags)),
        "payload": {"artifact_id": payload_id, "checksum": payload_ck, "size_bytes": len(payload_bytes)},
    }
    if synthetic:
        manifest["synthetic"] = True
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_ck = sha256_checksum(manifest_bytes)

    def ref(aid: str, kind: str, ck: str, size: int) -> dict[str, Any]:
        r: dict[str, Any] = {"artifact_id": aid, "owner": "financialplanning", "kind": kind, "checksum": ck, "content_type": "application/json", "size_bytes": size}
        if synthetic:
            r["synthetic"] = True
        return r

    record: dict[str, Any] = {
        "input_snapshot_id": input_snapshot_id,
        "domain": DOMAIN,
        "domain_schema_version": DOMAIN_SCHEMA_VERSION,
        "dataset": {"dataset_id": dataset_id},
        "manifest_checksum": manifest_ck,
        "source_timestamps": {"earliest": retrieved_at, "latest": retrieved_at},
        "lineage": lineage,
        "coverage": coverage,
        "quality_flags": sorted(set(quality_flags)),
        "status": status,
        "artifacts": [ref(manifest_id, MANIFEST_KIND, manifest_ck, len(manifest_bytes)), ref(payload_id, PAYLOAD_KIND, payload_ck, len(payload_bytes))],
        "created_at": created_at or retrieved_at,
    }
    if quality_details:
        record["quality_details"] = dict(quality_details)
    if payload.get("bias_disclosures"):
        record["bias_disclosures"] = [dict(d) for d in payload["bias_disclosures"]]  # contracts 1.1.0
    if status == "approved":
        record["approval_rule_version"] = "approval-v2-universe" if payload.get("universe") else "approval-v1"
    if synthetic:
        record["synthetic"] = True
    return record, {manifest_id: manifest_bytes, payload_id: payload_bytes}


class FixturePlatformClient:
    """In-process mock of the platform snapshot and staged-output routes (no network).

    ``enforce_approved`` mimics the platform's own refusal for FinanceModel principals
    (``PRECONDITION_FAILED`` ``snapshot_not_approved``); the :class:`SnapshotReader` checks again.
    """

    GRANT_HOST = "https://fixture-platform.invalid/grants/"

    def __init__(self, *, clock: Clock | None = None, enforce_approved: bool = False) -> None:
        self.clock = clock or SystemClock()
        self.enforce_approved = enforce_approved
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.blobs: dict[str, bytes] = {}
        self.staged_outputs: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []

    def add_snapshot(self, record: Mapping[str, Any], blobs: Mapping[str, bytes]) -> str:
        sid = str(record["input_snapshot_id"])
        self.snapshots[sid] = dict(record)
        self.blobs.update(blobs)
        return sid

    def corrupt(self, artifact_id: str, data: bytes = b"{}") -> None:
        """Replace an artifact's bytes (models storage corruption for WS-04 tests)."""
        self.blobs[artifact_id] = data

    def _require(self, input_snapshot_id: str) -> dict[str, Any]:
        rec = self.snapshots.get(input_snapshot_id)
        if rec is None:
            raise FinplanError(ErrorCode.NOT_FOUND, "snapshot not found", details={"record_type": "snapshot"})
        if self.enforce_approved and rec.get("status") != "approved":
            raise FinplanError.precondition("snapshot is not approved", reason="snapshot_not_approved")
        return rec

    def get_snapshot(self, input_snapshot_id: str, *, download: bool = False) -> dict[str, Any]:
        self.calls.append(("get_snapshot", input_snapshot_id))
        rec = self._require(input_snapshot_id)
        out: dict[str, Any] = {"snapshot": json.loads(json.dumps(rec))}
        if download:
            expires = utc_iso(self.clock.now() + timedelta(seconds=300))
            out["download_grants"] = [{"artifact_id": r["artifact_id"], "checksum": r["checksum"], "url": self.GRANT_HOST + r["artifact_id"], "expires_at": expires} for r in rec.get("artifacts", []) if r["artifact_id"] in self.blobs]
        return out

    def read_snapshot_observations(self, input_snapshot_id: str, *, instrument_id: str | None = None, start_date: str | None = None, end_date: str | None = None, page_size: int = 100, next_token: str | None = None) -> dict[str, Any]:
        self.calls.append(("read_snapshot_observations", input_snapshot_id))
        rec = self._require(input_snapshot_id)
        payload_ref = next(r for r in rec["artifacts"] if r["kind"] == PAYLOAD_KIND)
        obs = json.loads(self.blobs[payload_ref["artifact_id"]]).get("observations", [])
        sel = [o for o in obs if (instrument_id is None or o["instrument_id"] == instrument_id) and (start_date is None or o["session_date"] >= start_date) and (end_date is None or o["session_date"] <= end_date)]
        sel.sort(key=lambda o: (o["session_date"], o["instrument_id"], o.get("kind", "")))
        start = int(next_token or 0)
        page = sel[start : start + page_size]
        out: dict[str, Any] = {"observations": page, "requested_range": {"start": start_date or rec["coverage"]["start"], "end": end_date or rec["coverage"]["end"]}}
        if start + page_size < len(sel):
            out["next_token"] = str(start + page_size)
        return out

    def download(self, grant: Mapping[str, Any]) -> bytes:
        url = str(grant["url"])
        if not url.startswith(self.GRANT_HOST):
            raise FinplanError.internal("fixture client only serves fixture grants")
        self.calls.append(("download", url.rsplit("/", 1)[1]))
        return self.blobs[url.rsplit("/", 1)[1]]

    def get_staged_output(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("get_staged_output", run_id))
        if run_id not in self.staged_outputs:
            raise FinplanError(ErrorCode.NOT_FOUND, "staged output not found", details={"record_type": "staged_output"})
        return dict(self.staged_outputs[run_id])


# ===================================================================== HTTP (SigV4)
class HttpPlatformClient:
    """SigV4-signed calls to the platform plan API (``execute-api``).

    ``endpoint`` is resolved at run time from ``/finplan/<env>/financialplanning/api/plan-endpoint``;
    ``credentials`` is a botocore credentials object (the job or job-API role). ``opener`` is
    injectable (unit tests pass a fake; the test harness blocks real network anyway).
    """

    def __init__(self, endpoint: str, *, region: str, credentials: Any, opener: Callable[..., Any] | None = None, timeout: float = 20.0) -> None:
        if not endpoint.startswith("https://"):
            raise ValueError("platform endpoint must be https")
        self.endpoint = endpoint.rstrip("/")
        self.region = region
        self.credentials = credentials
        self.timeout = timeout
        if opener is None:
            import urllib.request

            opener = urllib.request.urlopen
        self._open = opener

    def _request(self, method: str, path: str, query: Mapping[str, Any] | None = None, *, signed: bool = True, url: str | None = None) -> tuple[int, bytes]:
        import urllib.error
        import urllib.parse
        import urllib.request

        qs = urllib.parse.urlencode({k: v for k, v in (query or {}).items() if v is not None})
        full = url or f"{self.endpoint}{path}" + (f"?{qs}" if qs else "")
        headers: dict[str, str] = {"Accept": "application/json"}
        if signed:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest

            req = AWSRequest(method=method, url=full, headers=headers)
            SigV4Auth(self.credentials, "execute-api", self.region).add_auth(req)
            headers = dict(req.headers.items())
        try:
            with self._open(urllib.request.Request(full, method=method, headers=headers), timeout=self.timeout) as resp:
                return int(resp.status), resp.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()

    def _json(self, method: str, path: str, query: Mapping[str, Any] | None = None) -> dict[str, Any]:
        status, body = self._request(method, path, query)
        doc = json.loads(body or b"{}")
        if 200 <= status < 300:
            return doc
        err = doc.get("error", doc) if isinstance(doc, dict) else {}
        code = str(err.get("code", "DEPENDENCY_UNAVAILABLE"))
        try:
            raise FinplanError(code, str(err.get("message", "platform request failed"))[:500], details={k: v for k, v in dict(err.get("details") or {}).items() if k in ("reason", "pointer", "field", "record_type", "served_contract_majors")})
        except ValueError:
            raise FinplanError.dependency_unavailable("platform request failed", status=status) from None

    def get_snapshot(self, input_snapshot_id: str, *, download: bool = False) -> dict[str, Any]:
        return self._json("GET", f"/v1/snapshots/{input_snapshot_id}", {"download": "true"} if download else None)

    def read_snapshot_observations(self, input_snapshot_id: str, *, instrument_id: str | None = None, start_date: str | None = None, end_date: str | None = None, page_size: int = 100, next_token: str | None = None) -> dict[str, Any]:
        return self._json("GET", f"/v1/snapshots/{input_snapshot_id}/observations", {"instrument_id": instrument_id, "start_date": start_date, "end_date": end_date, "page_size": page_size, "next_token": next_token})

    def download(self, grant: Mapping[str, Any]) -> bytes:
        url = str(grant["url"])
        if not url.startswith("https://"):
            raise FinplanError.internal("download grants must be https")
        status, body = self._request("GET", "", signed=False, url=url)
        if status != 200:
            raise FinplanError.dependency_unavailable("snapshot artifact download failed", status=status)
        return body

    def get_staged_output(self, run_id: str) -> dict[str, Any]:
        require_id("run_id", run_id)
        return self._json("GET", f"/v1/staged-outputs/{run_id}")
