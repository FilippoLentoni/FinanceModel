"""Synth assertions for the per-environment stacks (tasks 10.1, 10.3, 10.4; WS-01, WS-02, JOB-01,
DEP-04 synth part, CTL-01 wiring)."""

from __future__ import annotations

import json

import pytest

from infra.stacks import naming as n

pytestmark = pytest.mark.synth
ENVS = ("beta", "gamma", "prod")


def _tags(res):
    return {t["Key"]: t["Value"] for t in res["Properties"].get("Tags", [])}


def _bucket(assembly, env, logical_role):
    buckets = assembly.resources(f"finplan-{env}-financemodel-storage", "AWS::S3::Bucket")
    return next((lid, b) for lid, b in buckets.items() if _tags(b).get("logical-role") == logical_role)


def test_every_environment_has_its_stacks(assembly):
    for env in ENVS:
        for part in ("storage", "control"):
            assert f"finplan-{env}-financemodel-{part}" in assembly.templates
    assert "finplan-shared-financemodel-tooling" in assembly.templates
    assert "finplan-shared-financemodel-pipeline-store" in assembly.templates
    assert not any(name.endswith("-skeleton") for name in assembly.templates)


def test_ws01_research_storage_per_environment(assembly):
    names = set()
    for env in ENVS:
        lid, bucket = _bucket(assembly, env, "research-workspace-bucket")
        props = bucket["Properties"]
        name = json.dumps(props["BucketName"])
        assert f"finplan-{env}-financemodel-research-workspace-" in name and "AWS::AccountId" in name
        names.add(name)
        assert props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]["ServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"
        assert props["PublicAccessBlockConfiguration"] == {"BlockPublicAcls": True, "BlockPublicPolicy": True, "IgnorePublicAcls": True, "RestrictPublicBuckets": True}
        tags = _tags(bucket)
        assert tags["environment"] == env and tags["owner-repo"] == "financemodel" and tags["project"] == "finplan"
        assert bucket["DeletionPolicy"] == "Retain"
        policy = next(p for p in assembly.resources(f"finplan-{env}-financemodel-storage", "AWS::S3::BucketPolicy").values() if p["Properties"]["Bucket"] == {"Ref": lid})
        stmts = {s.get("Sid"): s for s in policy["Properties"]["PolicyDocument"]["Statement"]}
        text = json.dumps(stmts)
        assert '"aws:SecureTransport": "false"' in text  # TLS only
        others = json.dumps(stmts["DenyOtherEnvironmentPrincipals"])
        for other in set(ENVS) - {env}:
            assert f"role/finplan-{other}-*" in others  # e.g. gamma principals are denied the prod area
        assert f"role/finplan-{env}-financemodel-*" in json.dumps(stmts["DenyObjectAccessOutsideFinanceModel"])
    assert len(names) == 3  # three distinct research storage areas


def test_ws02_lifecycle_keeps_referenced_artifacts(assembly):
    for env in ENVS:
        _lid, bucket = _bucket(assembly, env, "research-workspace-bucket")
        rules = bucket["Properties"]["LifecycleConfiguration"]["Rules"]
        expiring = [r for r in rules if "ExpirationInDays" in r]
        # only unreferenced scratch output expires; artifacts/, runs/ and the dataset catalog never do
        assert [r["Prefix"] for r in expiring] == ["scratch/"]
        from finplan_model.core.config import load_config

        assert expiring[0]["ExpirationInDays"] == int(load_config(env).retention["scratch_days"])
        assert all("Prefix" in r or "ExpirationInDays" not in r for r in rules)
        assert any(r.get("NoncurrentVersionExpiration") for r in rules)


def test_model_registry_bucket_and_control_table(assembly):
    for env in ENVS:
        _lid, reg = _bucket(assembly, env, "model-registry")
        assert reg["Properties"]["VersioningConfiguration"]["Status"] == "Enabled"
        assert not any("ExpirationInDays" in r for r in reg["Properties"]["LifecycleConfiguration"]["Rules"])
        tables = assembly.resources(f"finplan-{env}-financemodel-storage", "AWS::DynamoDB::Table")
        (table,) = tables.values()
        props = table["Properties"]
        assert props["TableName"] == f"finplan-{env}-financemodel-job-control"
        assert props["BillingMode"] == "PAY_PER_REQUEST" and "StreamSpecification" not in props
        assert props["TimeToLiveSpecification"] == {"AttributeName": "ttl", "Enabled": True}
        assert {g["IndexName"] for g in props["GlobalSecondaryIndexes"]} == {"by_state", "by_submitted"}
        assert props.get("DeletionProtectionEnabled", False) is (env == "prod")
        policy = json.dumps(props["ResourcePolicy"])
        assert "Stream" not in policy  # table resource policies reject stream actions
        assert _tags(table)["logical-role"] == "job-control-table"


def test_control_plane_functions_rules_and_schedule(assembly):
    for env in ENVS:
        name = f"finplan-{env}-financemodel-control"
        fns = assembly.resources(name, "AWS::Lambda::Function")
        by_name = {f["Properties"]["FunctionName"]: f for f in fns.values()}
        for logical in (n.JOB_API_HANDLER, n.DISPATCHER, n.STATE_HANDLER):
            fn = by_name[n.function_name(env, logical)]
            props = fn["Properties"]
            assert props["Handler"] == n.FUNCTIONS[logical]
            assert props["Runtime"] == "python3.12" and props["Architectures"] == ["arm64"]
            assert props["Environment"]["Variables"]["FINPLAN_ENVIRONMENT"] == env
            assert props["Environment"]["Variables"]["FINPLAN_RUNS_TABLE"] == f"finplan-{env}-financemodel-job-control"
        assert by_name[n.function_name(env, n.JOB_API_HANDLER)]["Properties"]["Environment"]["Variables"]["FINPLAN_DISPATCHER_FUNCTION"] == n.function_name(env, n.DISPATCHER)
        groups = assembly.resources(name, "AWS::Logs::LogGroup")
        assert len(groups) == len(fns) and all(g["Properties"]["RetentionInDays"] == 30 for g in groups.values())
        rules = {r["Properties"]["Name"]: r for r in assembly.resources(name, "AWS::Events::Rule").values()}
        assert set(rules) == {n.env_name(env, n.STATE_CHANGE_RULE), n.env_name(env, n.TRAINING_STATE_CHANGE_RULE)}
        pattern = rules[n.env_name(env, n.STATE_CHANGE_RULE)]["Properties"]["EventPattern"]
        assert pattern["source"] == ["aws.sagemaker"] and pattern["detail"]["ProcessingJobName"] == [{"prefix": f"fm-{env}-"}]
        # model_selection runs as a SageMaker Training job: its state changes reach the same handler
        training = rules[n.env_name(env, n.TRAINING_STATE_CHANGE_RULE)]["Properties"]
        assert training["EventPattern"] == {"source": ["aws.sagemaker"], "detail-type": ["SageMaker Training Job State Change"], "detail": {"TrainingJobName": [{"prefix": f"fm-{env}-"}]}}
        assert training["Targets"][0]["RetryPolicy"]["MaximumRetryAttempts"] == 4
        schedules = assembly.resources(name, "AWS::Scheduler::Schedule")
        sched = next(s for s in schedules.values() if s["Properties"]["Name"] == n.env_name(env, n.DISPATCHER))
        assert sched["Properties"]["ScheduleExpression"] == "rate(1 minute)" and sched["Properties"]["State"] == "DISABLED"
        assert sched["Metadata"]["logical-role"] == "job-dispatcher-schedule"


def test_job_api_job_role_and_registry_policy_are_always_synthesized(assembly):
    """Contracts 1.0.0 (D16) carries the rows: no contract-gap switch remains."""
    for env in ENVS:
        control = assembly.stack(f"finplan-{env}-financemodel-control")
        assert {"JobEndpoint", "RegistryRef", "ApproverRoleRef", "JobRoleRef", "JobApiRoleRef"} <= set(control["Outputs"])
        assert assembly.resources(f"finplan-{env}-financemodel-control", "AWS::ApiGateway::RestApi")
        policies = assembly.resources(f"finplan-{env}-financemodel-storage", "AWS::S3::BucketPolicy")
        assert len(policies) == 2  # research workspace and model registry
        joined = json.dumps(list(policies.values()))
        assert "DenyPutWithoutIfNoneMatch" in joined and "DenyDeleteRegistryRecords" in joined


def test_dispatcher_schedule_is_deployed_disarmed(assembly):
    """Design D1: the schedule runs only while runs are pending; the control plane arms it."""
    from infra.stacks.policies import dispatcher_schedule_arn

    for env in ENVS:
        name = f"finplan-{env}-financemodel-control"
        schedules = assembly.resources(name, "AWS::Scheduler::Schedule")
        sched = next(s for s in schedules.values() if s["Properties"]["Name"] == n.env_name(env, n.DISPATCHER))
        props = sched["Properties"]
        assert props["State"] == "DISABLED" and props["ScheduleExpression"] == "rate(1 minute)"
        assert props["Name"] == n.env_name(env, n.DISPATCHER)
        fns = assembly.resources(name, "AWS::Lambda::Function")
        roles = assembly.resources(name, "AWS::IAM::Role")
        for logical in (n.JOB_API_HANDLER, n.DISPATCHER, n.STATE_HANDLER):
            fn = next(f for f in fns.values() if f["Properties"]["FunctionName"] == n.function_name(env, logical))
            assert fn["Properties"]["Environment"]["Variables"]["FINPLAN_DISPATCH_SCHEDULE"] == props["Name"]
            role = next(r for r in roles.values() if r["Properties"].get("RoleName") == n.role_name(env, logical))
            doc = json.dumps(role["Properties"]["Policies"])
            assert "scheduler:UpdateSchedule" in doc and "scheduler:GetSchedule" in doc
            assert f"schedule/default/{props['Name']}" in doc and "scheduler.amazonaws.com" in doc
        assert dispatcher_schedule_arn(env, partition="aws", region="us-east-2", account="<account-id>").endswith(f":schedule/default/{props['Name']}")


def test_job01_job_api_is_iam_authenticated(assembly):
    from finplan_model.control.api import ROUTES
    from infra.stacks.control import OPENAPI_ROUTES

    for env in ENVS:
        (api,) = assembly.resources(f"finplan-{env}-financemodel-control", "AWS::ApiGateway::RestApi").values()
        body = api["Properties"]["Body"]
        for path, ops in body["paths"].items():
            for method, op in ops.items():
                assert op["security"] == [{"sigv4": []}], (path, method)
                assert op["x-amazon-apigateway-integration"]["type"] == "aws_proxy"
        assert body["components"]["securitySchemes"]["sigv4"]["x-amazon-apigateway-authtype"] == "awsSigv4"
        assert {"MISSING_AUTHENTICATION_TOKEN", "ACCESS_DENIED"} <= set(body["x-amazon-apigateway-gateway-responses"])
        policy = json.dumps(api["Properties"]["Policy"])
        assert f"finplan-{env}-financelambdastool-*" in policy and "DenyApproveExceptApproverRole" in policy
        (stage,) = assembly.resources(f"finplan-{env}-financemodel-control", "AWS::ApiGateway::Stage").values()
        assert stage["Properties"]["MethodSettings"][0]["ThrottlingRateLimit"] == 5
    # every handler route has an OpenAPI route
    job_routes = {(m, p) for m, p, logical in OPENAPI_ROUTES if logical == n.JOB_API_HANDLER}
    assert len(job_routes) == len(ROUTES)


def test_job_role_uses_the_research_boundary(assembly):
    for env in ENVS:
        roles = assembly.resources(f"finplan-{env}-financemodel-control", "AWS::IAM::Role")
        job = next(r for r in roles.values() if r["Properties"].get("RoleName") == f"finplan-{env}-financemodel-job-execution-role")
        assert f"finplan-{env}-research-permission-boundary" in json.dumps(job["Properties"]["PermissionsBoundary"])
        assert job["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"] == {"Service": "sagemaker.amazonaws.com"}
        others = [r for r in roles.values() if r is not job]
        assert others and all(f"finplan-{env}-permission-boundary" in json.dumps(r["Properties"]["PermissionsBoundary"]) for r in others)


def test_names_fit_aws_limits():
    for env in ENVS:
        for logical in (*n.FUNCTIONS, n.JOB_EXECUTION, n.APPROVER, n.SCHEDULE_ROLE, "pipeline-stage"):
            assert len(n.role_name(env, logical)) <= 64
            assert len(n.function_name(env, logical)) <= 64
        assert len(n.bucket_name(env, n.RESEARCH_BUCKET, "1" * 12)) <= 63
        assert len(n.bucket_name(env, n.REGISTRY_BUCKET, "1" * 12)) <= 63
        assert len(n.exec_role_name(env)) <= 64
    assert len(n.pipeline_store_bucket_name("1" * 12)) <= 63
