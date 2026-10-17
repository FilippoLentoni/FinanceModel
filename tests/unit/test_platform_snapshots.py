"""Snapshot reader against the platform API through an injected client (approved only, checksums)."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.core.platform import FixturePlatformClient, HttpPlatformClient, SnapshotReader, build_synthetic_snapshot

SID = "snap_01KDVDNAZ83BAMMYCEGWF33DPM"
SID2 = "snap_01KDVDNBYGX5V5HY2JSK5XWKHC"


def _payload(n: int = 5) -> dict[str, Any]:
    obs = [{"instrument_id": "SPY", "session_date": f"2026-01-{5 + k:02d}", "kind": "completed_daily", "open": 100.0 + k, "high": 101.0 + k, "low": 99.0 + k, "close": 100.5 + k, "volume": 1000000, "synthetic": True} for k in range(n)]
    return {"dataset_id": "finance/etf-daily/SPY", "calendar": "XNYS", "instruments": [{"instrument_id": "SPY", "asset_class": "etf", "currency": "USD", "synthetic": True}], "observations": obs, "synthetic": True}


@pytest.fixture
def reader(platform):
    rec, blobs = build_synthetic_snapshot(_payload(), input_snapshot_id=SID)
    platform.add_snapshot(rec, blobs)
    rec2, blobs2 = build_synthetic_snapshot(_payload(), input_snapshot_id=SID2, status="committed")
    platform.add_snapshot(rec2, blobs2)
    return SnapshotReader(platform)


def test_approved_snapshot_loads_with_verified_checksums(reader, platform):
    content = reader.load(SID)
    assert content.snapshot.synthetic is True
    assert content.payload["dataset_id"] == "finance/etf-daily/SPY"
    assert content.manifest["payload"]["checksum"] in content.verified.values()
    assert len(content.verified) == 2
    assert content.snapshot.lineage["provider"] == "fixture"


def test_unapproved_snapshot_is_precondition_failed(reader):
    with pytest.raises(FinplanError) as ei:
        reader.resolve(SID2)
    assert ei.value.code == "PRECONDITION_FAILED"
    assert ei.value.details["reason"] == "snapshot_not_approved"


def test_unknown_snapshot_is_not_found(reader):
    with pytest.raises(FinplanError) as ei:
        reader.resolve("snap_01KES9T7J0GRHJS1P0A0T03EHX")
    assert ei.value.code == "NOT_FOUND"


def test_wrong_identifier_is_invalid_identifier(reader):
    with pytest.raises(FinplanError) as ei:
        reader.resolve("run_01KDVDNAZ83BAMMYCEGWF33DPM")
    assert ei.value.code == "INVALID_IDENTIFIER"


@pytest.mark.parametrize("which", ["payload", "manifest"])
def test_corrupted_artifact_fails_before_use(reader, platform, which):
    platform.corrupt(f"art_{which}_{SID.split('_', 1)[1]}", b'{"tampered": true}')
    with pytest.raises(FinplanError) as ei:
        reader.load(SID)
    assert ei.value.code == "PRECONDITION_FAILED"
    assert ei.value.details["reason"] == "snapshot_checksum_mismatch"


def test_paginated_observation_reads(reader):
    obs = reader.read_observations(SID, instrument_id="SPY", page_size=2)
    assert [o["session_date"] for o in obs] == ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]


def test_no_network_request_through_fixture_client(reader, platform):
    reader.load(SID)
    assert {c[0] for c in platform.calls} <= {"get_snapshot", "download"}


class _Resp(io.BytesIO):
    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(body)
        self.status = status

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *a: Any) -> None:
        return None


def test_http_client_signs_and_maps_errors():
    from botocore.credentials import Credentials

    seen: list[Any] = []

    def opener(req, timeout):  # noqa: ANN001
        seen.append(req)
        body = {"error": {"code": "PRECONDITION_FAILED", "message": "snapshot is not approved", "retryable": False, "details": {"reason": "snapshot_not_approved"}}}
        import urllib.error

        raise urllib.error.HTTPError(req.full_url, 412, "x", {}, io.BytesIO(json.dumps(body).encode()))

    client = HttpPlatformClient("https://plan-api.example.invalid/v1-stage", region="us-east-2", credentials=Credentials("testing", "testing", "testing"), opener=opener)
    with pytest.raises(FinplanError) as ei:
        client.get_snapshot(SID)
    assert ei.value.code == "PRECONDITION_FAILED" and ei.value.details["reason"] == "snapshot_not_approved"
    assert seen[0].full_url.endswith(f"/v1/snapshots/{SID}")
    assert "Authorization" in seen[0].headers and "AWS4-HMAC-SHA256" in seen[0].headers["Authorization"]

    def ok(req, timeout):  # noqa: ANN001
        return _Resp(200, json.dumps({"snapshot": {"input_snapshot_id": SID}}).encode())

    client2 = HttpPlatformClient("https://plan-api.example.invalid", region="us-east-2", credentials=Credentials("testing", "testing"), opener=ok)
    assert client2.get_snapshot(SID)["snapshot"]["input_snapshot_id"] == SID
    with pytest.raises(ValueError):
        HttpPlatformClient("http://plain.example.invalid", region="us-east-2", credentials=None)


def test_saved_context_reads_use_signed_gets_and_validate_identifiers():
    from botocore.credentials import Credentials
    from urllib.parse import parse_qs, urlparse

    seen = []
    def opener(req, timeout):
        seen.append(req)
        return _Resp(200, b'{}')
    client = HttpPlatformClient("https://plan-api.example.invalid", region="us-east-2", credentials=Credentials("testing", "testing"), opener=opener)
    suffix = SID.removeprefix("snap_")
    client.get_plan("pl_" + suffix)
    client.get_portfolio_state("pf_" + suffix)
    client.get_latest_snapshot("finance/equity-etf-daily/research-universe")
    assert [urlparse(r.full_url).path for r in seen] == [f"/v1/plans/pl_{suffix}", f"/v1/portfolios/pf_{suffix}/state", "/v1/snapshots/latest"]
    assert parse_qs(urlparse(seen[-1].full_url).query)["dataset_id"] == ["finance/equity-etf-daily/research-universe"]
    assert all(r.method == "GET" and "AWS4-HMAC-SHA256" in r.headers["Authorization"] for r in seen)
    with pytest.raises(FinplanError):
        client.get_portfolio_state("../../another/path")


def test_lifecycle_client_signs_post_body_and_checks_decision_portfolio():
    from botocore.credentials import Credentials

    seen = []
    suffix = SID.removeprefix("snap_")
    pid, did = "pf_" + suffix, "pd_" + suffix
    def opener(req, timeout):
        seen.append(req)
        return _Resp(200, json.dumps({"decision": {"portfolio_id": pid, "decision_id": did}}).encode())
    client = HttpPlatformClient("https://plan-api.example.invalid", region="us-east-2", credentials=Credentials("testing", "testing"), opener=opener)
    body = {"portfolio_id": pid, "recommendation": {"weights": [.2, .8]}, "idempotency_key": "proposal-1"}
    assert client.create_portfolio_decision(pid, body)["decision"]["decision_id"] == did
    assert seen[-1].method == "POST" and json.loads(seen[-1].data) == body
    assert "AWS4-HMAC-SHA256" in seen[-1].headers["Authorization"]
    assert client.get_portfolio_decision(pid, did)["decision"]["portfolio_id"] == pid
    assert seen[-1].full_url.endswith("/v1/portfolio-decisions/" + did)
    with pytest.raises(FinplanError):
        client.get_portfolio_decision(pid, "../../unsafe")


def test_lifecycle_pagination_uses_platform_page_size_and_preserves_tokens():
    from botocore.credentials import Credentials
    from urllib.parse import parse_qs, urlparse

    seen = []
    def opener(req, timeout):
        seen.append(req)
        return _Resp(200, b'{}')
    client = HttpPlatformClient("https://plan-api.example.invalid", region="us-east-2", credentials=Credentials("testing", "testing"), opener=opener)
    pid = "pf_" + SID.removeprefix("snap_")
    for operation in (client.list_portfolio_decisions, client.get_portfolio_history):
        operation(pid, limit=73, next_token="opaque+token/next=")
        assert parse_qs(urlparse(seen[-1].full_url).query) == {
            "page_size": ["73"], "next_token": ["opaque+token/next="],
        }
