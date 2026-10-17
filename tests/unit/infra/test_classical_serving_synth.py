import json

import pytest


@pytest.mark.synth
def test_independent_classical_and_weekly_controller_are_beta_only(assembly):
    from infra.stacks import naming as n

    for env in ("beta", "gamma", "prod"):
        name = f"finplan-{env}-financemodel-control"
        functions = assembly.resources(name, "AWS::Lambda::Function")
        found = {
            f["Properties"]["FunctionName"]: f["Properties"]
            for f in functions.values()
            if f["Properties"]["FunctionName"]
            in (
                n.function_name(env, n.CLASSICAL_INFERENCE),
                n.function_name(env, n.RESEARCH_CONTROLLER),
            )
        }
        if env != "beta":
            assert not found
            continue
        assert len(found) == 2
        assert (
            found[n.function_name(env, n.CLASSICAL_INFERENCE)]["Handler"]
            == "finplan_model.classical_api.handler"
        )
        assert all(
            f["Timeout"] == 270 and f["MemorySize"] == 1024 for f in found.values()
        )
        # Weekly cost safety comes from durable claims and sandbox idempotency,
        # without consuming scarce account-level reserved Lambda concurrency.
        assert all("ReservedConcurrentExecutions" not in f for f in found.values())
        assert "ClassicalFunctionRef" in assembly.stack(name)["Outputs"]
        schedules = assembly.resources(name, "AWS::Scheduler::Schedule")
        weekly = next(
            s["Properties"]
            for s in schedules.values()
            if s["Properties"]["Name"] == n.env_name(env, n.RESEARCH_SCHEDULE)
        )
        assert (
            weekly["State"] == "ENABLED"
            and weekly["ScheduleExpression"] == "cron(0 9 ? * MON *)"
        )
        assert weekly["ScheduleExpressionTimezone"] == "America/New_York"
        roles = assembly.resources(name, "AWS::IAM::Role")
        role = next(
            r
            for r in roles.values()
            if r["Properties"].get("RoleName")
            == n.role_name(env, n.CLASSICAL_INFERENCE)
        )
        text = json.dumps(role["Properties"]["Policies"])
        assert "/classical/*" in text and "s3:if-none-match" in text
        assert "advisory-policy" not in text and "production-strategy" not in text
        assert (
            "budgets:ViewBudget" in text
            and "DenyDirectComputeAndFinancialWrites" in text
        )
