"""The deployed wiring (Lambda handlers -> boto3 clients -> DynamoDB, SSM, S3) on moto, offline.
SageMaker is never called here (the harness blocks it)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import boto3
import pytest
from moto import mock_aws

from finplan_model.control import handlers
from finplan_model.control.settings import job_definition_parameter
from finplan_model.core.clock import utc_iso
from finplan_model.core.config import load_config

from .support import IMAGE_URI, JOB_ROLE_ARN, SUBMITTER, request
from .test_dynamo_store_and_settings import TABLE, _create_table


@pytest.fixture
def deployed(monkeypatch):
    with mock_aws():
        cfg = load_config("beta")
        _create_table(boto3.client("dynamodb", region_name="us-east-2"))
        boto3.client("s3", region_name="us-east-2").create_bucket(Bucket="example-research-bucket", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        ssm = boto3.client("ssm", region_name="us-east-2")
        ssm.put_parameter(Name=cfg.ssm["research_storage_ref"], Value="example-research-bucket", Type="String")
        ssm.put_parameter(Name=cfg.ssm["instance_prices"], Value=json.dumps({"retrieved_at": utc_iso(datetime.now(UTC)), "usd_per_hour": {"ml.m5.xlarge": 0.25}}), Type="String")
        ssm.put_parameter(Name=cfg.ssm["job_role_ref"], Value=JOB_ROLE_ARN, Type="String")
        for jt in cfg.raw["job_types"]:
            ssm.put_parameter(Name=job_definition_parameter("beta", jt), Value=json.dumps({"image_uri": IMAGE_URI}), Type="String")
        monkeypatch.setenv("FINPLAN_ENVIRONMENT", "beta")
        monkeypatch.setenv("FINPLAN_RUNS_TABLE", TABLE)
        monkeypatch.setattr(handlers, "_SERVICE", None)
        yield
        monkeypatch.setattr(handlers, "_SERVICE", None)


def _event(method: str, path: str, body=None) -> dict:
    return {"httpMethod": method, "path": path, "headers": {}, "body": json.dumps(body) if body is not None else None, "requestContext": {"identity": {"userArn": SUBMITTER}}}


def test_api_handler_end_to_end_on_moto(deployed):
    resp = handlers.api_handler(_event("POST", "/v1/jobs", request(dry_run=True)))
    assert resp["statusCode"] == 200, resp["body"]
    resp = handlers.api_handler(_event("POST", "/v1/jobs", request()))
    assert resp["statusCode"] == 202
    run_id = json.loads(resp["body"])["run_id"]
    status = handlers.api_handler(_event("GET", f"/v1/jobs/{run_id}"))
    assert status["statusCode"] == 200 and json.loads(status["body"])["state"] == "awaiting_approval"
    listing = json.loads(handlers.api_handler(_event("GET", "/v1/jobs"))["body"])
    assert [j["run_id"] for j in listing["jobs"]] == [run_id]
    # nothing to start (awaiting approval), so the dispatcher makes no SageMaker call
    assert handlers.dispatcher_handler({}, None)["started"] == []
    assert handlers.state_change_handler({"detail": {"ProcessingJobName": "unrelated", "ProcessingJobStatus": "Completed"}}, None) == {"ignored": "not_this_environment"}


def test_missing_research_storage_is_dependency_unavailable(deployed):
    boto3.client("ssm", region_name="us-east-2").delete_parameter(Name=load_config("beta").ssm["research_storage_ref"])
    resp = handlers.api_handler(_event("GET", "/v1/jobs"))
    assert resp["statusCode"] == 503 and json.loads(resp["body"])["code"] == "DEPENDENCY_UNAVAILABLE"


def test_dispatch_schedule_is_armed_and_disarmed_on_moto(deployed, monkeypatch):
    """Design D1 wiring: the handlers arm the deployed (DISABLED) schedule through EventBridge
    Scheduler and the dispatcher returns it to its deployed state once nothing is pending."""
    name = "finplan-beta-financemodel-job-dispatcher"
    sched = boto3.client("scheduler", region_name="us-east-2")
    from moto.core import DEFAULT_ACCOUNT_ID as acct  # moto's synthetic account; never a real one

    target = {"Arn": f"arn:aws:lambda:us-east-2:{acct}:function:{name}", "RoleArn": f"arn:aws:iam::{acct}:role/finplan-beta-financemodel-job-dispatcher-schedule-role", "Input": "{}"}
    sched.create_schedule(Name=name, ScheduleExpression="rate(1 minute)", FlexibleTimeWindow={"Mode": "OFF"}, Target=target, State="DISABLED")
    monkeypatch.setenv("FINPLAN_DISPATCH_SCHEDULE", name)
    resp = handlers.api_handler(_event("POST", "/v1/jobs", request()))
    assert resp["statusCode"] == 202
    run_id = json.loads(resp["body"])["run_id"]
    armed = sched.get_schedule(Name=name)
    assert armed["State"] == "ENABLED" and armed["ScheduleExpression"].startswith("at(")  # approval-expiry wake-up only
    assert armed["Target"]["Arn"] == target["Arn"]
    assert handlers.dispatcher_handler({"run_id": run_id}, None)["schedule"] == [armed["ScheduleExpression"]]
    cancel = handlers.api_handler(_event("POST", f"/v1/jobs/{run_id}/cancel", {"idempotency_key": "cancel-1"}))
    assert cancel["statusCode"] == 200, cancel["body"]
    assert handlers.dispatcher_handler({}, None)["schedule"] == ["disabled"]
    idle = sched.get_schedule(Name=name)
    assert idle["State"] == "DISABLED" and idle["ScheduleExpression"] == "rate(1 minute)"
