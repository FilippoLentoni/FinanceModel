"""The deployed store (DynamoDB single table, conditional writes, transactions) and the SSM settings
reader, both on moto (offline). Covers JOB-04/JOB-05/CTL-02 against the real store implementation."""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from finplan_model.control.auth import Principal
from finplan_model.control.settings import SsmSettings, job_definition_parameter
from finplan_model.control.store import TABLE_SPEC, ConditionFailed, DynamoRunStore
from finplan_model.core.errors import FinplanError

from .support import APPROVER, READER, SUBMITTER, Harness, env_config

TABLE = "finplan-beta-financemodel-runs-table"


def _create_table(client) -> None:
    attrs = {TABLE_SPEC["partition_key"]["name"], TABLE_SPEC["sort_key"]["name"]}
    gsis = []
    for g in TABLE_SPEC["global_secondary_indexes"]:
        attrs |= {g["partition_key"]["name"], g["sort_key"]["name"]}
        gsis.append({"IndexName": g["name"], "KeySchema": [{"AttributeName": g["partition_key"]["name"], "KeyType": "HASH"}, {"AttributeName": g["sort_key"]["name"], "KeyType": "RANGE"}], "Projection": {"ProjectionType": "ALL"}})
    client.create_table(
        TableName=TABLE,
        AttributeDefinitions=[{"AttributeName": a, "AttributeType": "S"} for a in sorted(attrs)],
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
        GlobalSecondaryIndexes=gsis,
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture
def dynamo():
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-2")
        _create_table(client)
        yield DynamoRunStore(client, TABLE)


def test_lifecycle_on_dynamodb(dynamo):
    h = Harness(store=dynamo, auto_approve=1.0)
    _, first = h.submit(idempotency_key="k-a")
    assert h.submit(idempotency_key="k-a")[1] == first  # replay from the table
    h.clock.advance(seconds=1)
    _, second = h.submit(idempotency_key="k-b")
    s = h.service.dispatch()
    assert s["started"] == [first["run_id"]] and s["waiting"] == [second["run_id"]]
    name = dynamo.get_run(first["run_id"])["job_name"]
    h.event(name, "InProgress")
    h.succeed(first["run_id"])
    assert dynamo.get_run(first["run_id"])["state"] == "succeeded"
    assert [e["state"] for e in dynamo.list_events(first["run_id"])] == ["queued", "starting", "running", "succeeded"]
    h.event(name, "Failed")  # late event
    assert dynamo.get_run(first["run_id"])["state"] == "succeeded"
    assert h.service.dispatch()["started"] == [second["run_id"]]
    _, resp = h.service.cancel_job(Principal.from_arn(SUBMITTER), second["run_id"], {"idempotency_key": "c"})
    assert resp["state"] == "stopping"
    listing = h.service.list_jobs(Principal.from_arn(READER), {"page_size": "1"})
    assert len(listing["jobs"]) == 1 and listing["next_token"]
    page2 = h.service.list_jobs(Principal.from_arn(READER), {"page_size": "1", "next_token": listing["next_token"]})
    assert page2["jobs"][0]["run_id"] == second["run_id"] and "next_token" not in page2
    running = h.service.list_jobs(Principal.from_arn(READER), {"state": "stopping"})
    assert [j["run_id"] for j in running["jobs"]] == [second["run_id"]]


def test_conditional_writes_on_dynamodb(dynamo):
    h = Harness(store=dynamo)
    _, sub = h.submit()
    run = dynamo.get_run(sub["run_id"])
    with pytest.raises(ConditionFailed):
        dynamo.update_run({**run, "revision": 5}, 4, [])
    with pytest.raises(ConditionFailed):  # append-only events
        dynamo.update_run({**run, "revision": 1}, 0, [{"run_id": run["run_id"], "seq": 0, "state": "queued", "at": run["submitted_at"]}])
    assert dynamo.get_run(sub["run_id"])["revision"] == 0
    idem = dynamo.get_idempotency(_idem_key(h))
    assert idem is not None and idem["response"]["run_id"] == sub["run_id"]
    other = "run_01KDVDNAZ83BAMMYCEGWF33DPM"
    with pytest.raises(ConditionFailed):  # run and idempotency record are created atomically
        dynamo.create_run({**run, "run_id": other}, [{"run_id": other, "seq": 0, "state": "queued", "at": run["submitted_at"]}], idem)
    assert dynamo.get_run(other) is None


def _idem_key(h: Harness) -> str:
    from finplan_model.control.store import idempotency_scope_key

    return idempotency_scope_key(Principal.from_arn(SUBMITTER).arn, "beta", "submit_job", "client-key-0001")


def test_lease_slots_on_dynamodb(dynamo):
    assert dynamo.acquire_slot("beta#cpu", 0, "run_A", "2026-01-05T10:00:00Z", "2026-01-05T09:00:00Z") is True
    assert dynamo.acquire_slot("beta#cpu", 0, "run_B", "2026-01-05T10:00:00Z", "2026-01-05T09:00:00Z") is False
    assert dynamo.renew_slot("beta#cpu", 0, "run_B", "2026-01-05T11:00:00Z") is False
    assert dynamo.renew_slot("beta#cpu", 0, "run_A", "2026-01-05T11:00:00Z") is True
    assert dynamo.lease_slots("beta#cpu")[0]["expires_at"] == "2026-01-05T11:00:00Z"
    assert dynamo.release_slot("beta#cpu", 0, "run_B") is False
    assert dynamo.release_slot("beta#cpu", 0, "run_A") is True
    assert dynamo.lease_slots("beta#cpu") == []


def test_approval_on_dynamodb(dynamo):
    h = Harness(store=dynamo)
    _, sub = h.submit()
    status = h.service.approve_run(Principal.from_arn(APPROVER), sub["run_id"], {})
    assert status["state"] == "queued"


# ---------------------------------------------------------------- SSM settings
@pytest.fixture
def ssm():
    with mock_aws():
        yield boto3.client("ssm", region_name="us-east-2")


def test_ssm_settings_read_runtime_values(ssm):
    cfg = env_config()
    put = lambda name, value: ssm.put_parameter(Name=name, Value=value, Type="String")  # noqa: E731
    put(cfg.ssm["instance_prices"], json.dumps({"retrieved_at": "2026-01-01T00:00:00Z", "usd_per_hour": {"ml.m5.xlarge": 0.25}}))
    put(cfg.ssm["auto_approve_usd"], "0.5")
    put(cfg.ssm["lease_limits"], json.dumps({"cpu": 2}))
    put(cfg.ssm["approver_role_ref"], "arn:aws:iam::<account-id>:role/finplan-beta-financemodel-approver-role")
    put(cfg.ssm["production_candidate_principals"], json.dumps(["finplan-beta-financialplanning-candidate-role"]))
    put(cfg.ssm["budget_allocation"], json.dumps({"cpu_research": 7, "gpu": 25}))
    put(cfg.ssm["budget_state"], json.dumps({"state": "ok"}))
    put(cfg.ssm["job_role_ref"], "arn:aws:iam::<account-id>:role/finplan-beta-financemodel-job-role")
    put(cfg.ssm["research_storage_ref"], json.dumps({"bucket": "example-research-bucket"}))
    put(job_definition_parameter("beta", "run_backtest"), json.dumps({"image_uri": "registry.invalid/x@sha256:" + "a" * 64}))
    s = SsmSettings(ssm, cfg)
    assert s.instance_prices()["usd_per_hour"]["ml.m5.xlarge"] == 0.25
    assert s.auto_approve_usd() == 0.5
    assert s.lease_limits() == {"cpu": 2, "gpu": 0}
    assert s.approver_role_name() == "finplan-beta-financemodel-approver-role"
    assert s.production_candidate_principals() == ["finplan-beta-financialplanning-candidate-role"]
    assert s.budget_allocation() == {"cpu_research": 7, "gpu": 25}
    assert json.loads(s.budget_state()) == {"state": "ok"}
    assert s.research_storage() == "example-research-bucket"
    assert s.job_definition("run_backtest")["image_uri"].endswith("a" * 64)
    assert s.job_definition("run_benchmark") is None
    assert job_definition_parameter("beta", "run_backtest") == "/finplan/beta/financemodel/job/run-backtest"


def test_ssm_settings_defaults_and_invalid_values(ssm):
    cfg = env_config()
    s = SsmSettings(ssm, cfg)
    assert s.auto_approve_usd() == 0.0  # FM-OQ-3 default
    assert s.lease_limits() == {"cpu": 1, "gpu": 0}
    assert s.budget_allocation()["cpu_research"] == 7 and s.budget_allocation()["gpu"] == 25
    assert s.budget_state() is None and s.instance_prices() is None
    ssm.put_parameter(Name=cfg.ssm["lease_limits"], Value="not json", Type="String")
    with pytest.raises(FinplanError) as ei:
        SsmSettings(ssm, cfg).lease_limits()
    assert ei.value.code == "PRECONDITION_FAILED"


def test_ssm_read_failure_fails_closed(ssm):
    class Broken:
        def get_parameter(self, **kw):
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "x"}}, "GetParameter")

    with pytest.raises(FinplanError) as ei:
        SsmSettings(Broken(), env_config()).budget_state()
    assert ei.value.code == "DEPENDENCY_UNAVAILABLE" and ei.value.retryable is True
