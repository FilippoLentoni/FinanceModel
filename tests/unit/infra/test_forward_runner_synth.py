import json

import pytest


@pytest.mark.synth
def test_forward_runner_lambda_role_and_output_are_beta_only(assembly):
    from infra.stacks import naming as n

    for env in ("beta", "gamma", "prod"):
        name = f"finplan-{env}-financemodel-control"
        functions = {f["Properties"]["FunctionName"]: f["Properties"] for f in assembly.resources(name, "AWS::Lambda::Function").values()}
        roles = {r["Properties"].get("RoleName"): r["Properties"] for r in assembly.resources(name, "AWS::IAM::Role").values()}
        outputs = assembly.stack(name).get("Outputs", {})
        runner = n.function_name(env, n.FORWARD_RUNNER)
        role_name = n.role_name(env, n.FORWARD_RUNNER_ROLE)
        if env != "beta":
            assert runner not in functions and role_name not in roles and "ForwardRunnerFunctionArn" not in outputs
            continue
        assert runner == "finplan-beta-financemodel-forward-runner"
        assert role_name == "finplan-beta-financemodel-job-api-handler-forward-runner-role"  # FP plan API admits job-api-handler*
        fn = functions[runner]
        assert fn["Handler"] == "finplan_model.forward.runner.handler" and fn["Timeout"] == 900 and fn["MemorySize"] == 2048
        assert fn["Architectures"] == ["arm64"] and fn["Runtime"] == "python3.12"
        assert fn["Environment"]["Variables"]["FINPLAN_ENVIRONMENT"] == "beta"
        assert len(fn["Environment"]["Variables"]["FINPLAN_PLATFORM_SOURCE_COMMIT"]) == 40
        assert "ReservedConcurrentExecutions" not in fn
        role = roles[role_name]
        assert "PermissionsBoundary" in role
        text = json.dumps(role["Policies"])
        for needle in ("financialplanning-research-input", "financialplanning-portfolio-observation", "/modeling-evidence/*", "kms:ResourceAliases",
                       "financeagent/modeling-runner-client-*", "artifacts/policy_inference/*", "bucket-reports", "s3:GetEncryptionConfiguration"):
            assert needle in text, needle
        assert "DenyComputeConfigAndDeletes" in text
        groups = [g["Properties"] for g in assembly.resources(name, "AWS::Logs::LogGroup").values() if "ForwardRunner" in json.dumps(g["Properties"]["LogGroupName"])]
        assert len(groups) == 1 and groups[0]["RetentionInDays"] == 30
        permissions = [p["Properties"] for p in assembly.resources(name, "AWS::Lambda::Permission").values() if "ForwardRunner" in json.dumps(p["Properties"]["FunctionName"])]
        principals = json.dumps([p["Principal"] for p in permissions])
        assert len(permissions) == 2 and "pipeline-stage-role" in principals and all(p["Action"] == "lambda:InvokeFunction" for p in permissions)
        assert "ForwardRunnerFunctionArn" in outputs
        schedules = json.dumps(assembly.resources(name, "AWS::Scheduler::Schedule"))
        assert "ForwardRunner" not in schedules   # FinancialPlanning schedules it


def test_release_publishes_the_runner_reference_and_budget_role():
    from scripts import release
    from infra.stacks import naming as n

    class Info:
        image_repository = image_digest = None
    plan = release.planned_parameters("beta", Info(), {"ForwardRunnerFunctionArn": "arn-token"}, account="placeholder-account", deployed_job_types=[])
    assert plan["forward-runner-function-arn"] == ("/finplan/beta/financemodel/api/forward-runner-function-arn", "arn-token")
    assert n.role_name("beta", n.FORWARD_RUNNER_ROLE) in release.enforced_role_names("beta", {"ForwardRunnerFunctionArn": "x"})
    assert n.role_name("gamma", n.FORWARD_RUNNER_ROLE) not in release.enforced_role_names("gamma", {})


def test_runner_may_use_the_platform_key_through_dynamodb_and_s3():
    from infra.stacks.policies import forward_runner_policy
    key = next(s for s in forward_runner_policy("beta")["Statement"] if s["Sid"] == "ExperimentLedgerKey")
    assert set(key["Condition"]["StringEquals"]["kms:ViaService"]) == {"s3.${AWS::Region}.amazonaws.com", "dynamodb.${AWS::Region}.amazonaws.com"}
