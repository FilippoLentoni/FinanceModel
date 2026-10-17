import copy
import io

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from moto import mock_aws

from finplan_model.classical.storage import S3Store, analysis_id, public
from finplan_model.core.artifacts import InMemoryArtifactStore, canonical_json_bytes, sha256_checksum
from finplan_model.core.clock import FrozenClock
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.jobs.handlers import JobInputs, run_benchmark
from finplan_model.jobs.market_loader import load_market


def claim_client():
    return boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )


@pytest.mark.parametrize("kind", ["claim", "record"])
@pytest.mark.parametrize("prefix_sibling", [False, True])
def test_missing_object_403_uses_exact_bounded_probe(kind, prefix_sibling):
    s3 = claim_client()
    store = S3Store(s3, "example-bucket")
    aid, claim = "ca_" + "a" * 32, "idempotency/submit_portfolio_feedback/" + "a" * 64
    key = "classical/" + ("claims/" + claim if kind == "claim" else "records/" + aid) + ".json"
    with Stubber(s3) as stub:
        stub.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403, expected_params={"Bucket": store.bucket, "Key": key})
        stub.add_response("list_objects_v2", {"Contents": [{"Key": key + ".other"}] if prefix_sibling else []}, {"Bucket": store.bucket, "Prefix": key, "MaxKeys": 1})
        if kind == "claim":
            assert store.get_claim(claim) is None
        else:
            with pytest.raises(FinplanError) as exc:
                store.get(aid)
            assert exc.value.code == "NOT_FOUND"
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("probe_denied", [False, True])
def test_actual_s3_authorization_failure_is_never_claim_absence(probe_denied):
    s3 = claim_client()
    store = S3Store(s3, "example-bucket")
    claim = "weekly/2026_w41"
    key = "classical/claims/" + claim + ".json"
    with Stubber(s3) as stub:
        stub.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403, expected_params={"Bucket": store.bucket, "Key": key})
        if probe_denied:
            stub.add_client_error("list_objects_v2", service_error_code="AccessDenied", http_status_code=403, expected_params={"Bucket": store.bucket, "Prefix": key, "MaxKeys": 1})
        else:
            stub.add_response("list_objects_v2", {"Contents": [{"Key": key}]}, {"Bucket": store.bucket, "Prefix": key, "MaxKeys": 1})
        with pytest.raises(ClientError) as exc:
            store.get_claim(claim)
        assert exc.value.response["Error"]["Code"] == "AccessDenied"
        assert exc.value.operation_name == ("ListObjectsV2" if probe_denied else "GetObject")
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("corrupt", [False, True])
def test_existing_claim_get_verifies_checksum_without_listing(corrupt):
    s3 = claim_client()
    store = S3Store(s3, "example-bucket")
    claim, doc = "weekly/2026_w41", {"analysis_id": "ca_" + "a" * 32}
    data = canonical_json_bytes(doc)
    with Stubber(s3) as stub:
        stub.add_response("get_object", {"Body": io.BytesIO(data), "Metadata": {"checksum": "sha256:" + "0" * 64 if corrupt else sha256_checksum(data)}}, {"Bucket": store.bucket, "Key": "classical/claims/" + claim + ".json"})
        if corrupt:
            with pytest.raises(FinplanError, match="checksum"):
                store.get_claim(claim)
        else:
            assert store.get_claim(claim) == doc
        stub.assert_no_pending_responses()


def test_s3_feedback_claim_replay_and_conflict(context):
    from finplan_model.classical_api import handle

    plan = context.service.recommend({})
    private = context.service.store.get(plan["analysis_id"])
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="example-bucket")
        context.service.store = S3Store(s3, "example-bucket")
        context.service.store.put(private)
        body = {"analysis_id": plan["analysis_id"], "text": "Investigate turnover costs.", "idempotency_key": "feedback-s3-replay"}
        event = {"environment": "beta", "operation": "submit_portfolio_feedback", "request": body}
        issued = handle(event, context.service)
        assert handle(event, context.service) == issued
        assert public(context.service.store.get(issued["analysis_id"])) == issued
        with pytest.raises(FinplanError) as exc:
            handle({**event, "request": {**body, "text": "Changed feedback."}}, context.service)
        assert exc.value.code == "IDEMPOTENCY_KEY_REUSED"


def test_s3_write_once_checksum_and_newest_index(context):
    a = context.service.recommend({})
    doc = context.service.store.get(a["analysis_id"])
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="example-bucket")
        store = S3Store(s3, "example-bucket")
        assert store.put(doc) == doc and store.put(doc) == doc
        assert public(store.get(a["analysis_id"])) == a
        newer = copy.deepcopy(doc)
        newer["analysis_id"] = analysis_id({"new": 1})
        newer["created_at"] = "2026-10-11T00:00:00Z"
        store.put(newer)
        assert store.list(limit=1)[0]["analysis_id"] == newer["analysis_id"]
        assert (
            store.list(kind="recommendation", portfolio_id=context.pid, limit=2)[1][
                "analysis_id"
            ]
            == a["analysis_id"]
        )
        corrupt = copy.deepcopy(doc)
        corrupt["request_fingerprint"] = "changed"
        with pytest.raises(FinplanError, match="conflicts"):
            store.put(corrupt)
        s3.put_object(
            Bucket="example-bucket",
            Key="classical/records/" + a["analysis_id"] + ".json",
            Body=b"{}",
            Metadata={"checksum": "sha256:" + "a" * 64},
        )
        with pytest.raises(FinplanError, match="checksum"):
            store.get(a["analysis_id"])


def test_weekly_sandbox_runs_finite_grid_controls_and_chronological_selection(context):
    market, content = load_market(context.platform, context.sid)
    ctx = RunContext(
        run_id="run_01KDVDP88REHGPBXFX6CHX92KS",
        environment="beta",
        clock=FrozenClock("2026-10-10T00:00:00Z"),
        synthetic=True,
    )
    spec = {
        "run_id": ctx.run_id,
        "configuration_id": "cfg_" + "a" * 64,
        "input_snapshot_id": context.sid,
        "configuration": {"payload": {"objective": "classical_weekly_review"}},
        "strategy": "min_variance",
        "universe": list(market.instruments),
        "simulation": {
            "rebalance_frequency": "weekly",
            "constraints": {"long_only": True, "max_weight": 0.6},
            "fees": {"proportional_bps": 2},
        },
        "evaluation_window": {
            "start": context.dates[60].isoformat(),
            "end": context.dates[-1].isoformat(),
        },
        "purpose": "research",
    }
    result = run_benchmark(
        JobInputs(ctx, spec, market, content, InMemoryArtifactStore())
    )
    evidence = result["payload"]["weekly_research"]
    assert (
        result["completion_status"] == "succeeded" and len(evidence["variants"]) == 12
    )
    assert (
        evidence["selection"]["split"] == "validation"
        and evidence["activation"] == "proposal_only"
    )
    assert (
        evidence["windows"]["validation"]["end"]
        < evidence["windows"]["research_test"]["start"]
    )
    assert evidence["common_warmup"]["required_completed_returns"] == 120
    assert evidence["windows"]["validation"]["start"] == context.dates[120].isoformat()
    assert {v["algorithm"] for v in evidence["variants"]} == {
        "cash",
        "buy_and_hold",
        "equal_weight",
        "min_variance",
        "mean_variance",
        "scenario_cvar",
    }


def test_expanded_lookback_grid_requires_sufficient_common_warmup(context):
    from types import SimpleNamespace

    from finplan_model.classical.benchmark import run_weekly_benchmark

    with pytest.raises(FinplanError, match="common maximum-lookback"):
        run_weekly_benchmark(
            SimpleNamespace(
                market=context.market,
                universe=context.market.instruments,
                window=(None, None),
                spec={"configuration": {"payload": {"lookback_days": 120}}},
            )
        )
