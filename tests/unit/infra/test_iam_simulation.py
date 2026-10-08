"""IAM policy simulation of the FinanceModel roles with their contract permission boundaries
(task 10.2; WS-03, WS-05, RST-04, ENV-03, ENV-04, ENV-05; REG-02 identity-side write-once).

Offline: the role documents of :mod:`infra.stacks.policies` (the same functions the stacks use) are
rendered with concrete placeholders and evaluated with :func:`finplan_contracts.iam.evaluate` together
with the contract boundary each role carries.
"""

from __future__ import annotations

import pytest
from finplan_contracts import boundaries as cb
from finplan_contracts.iam import Request, evaluate

from finplan_model.control.policies import approver_identity_policy
from infra.stacks import naming as n
from infra.stacks import policies as pol

ACCT = "<account-id>"
C = {"partition": "aws", "region": "us-east-2", "account": ACCT}
ENV = "gamma"


def _role_arn(name: str) -> str:
    return f"arn:aws:iam::{ACCT}:role/{name}"


def _s3(bucket: str, key: str = "") -> str:
    return f"arn:aws:s3:::{bucket}" + (f"/{key}" if key else "")


RESEARCH = n.bucket_name(ENV, n.RESEARCH_BUCKET, ACCT)
REGISTRY = n.bucket_name(ENV, n.REGISTRY_BUCKET, ACCT)
STAGING = f"finplan-{ENV}-financialplanning-run-staging-area-{ACCT}"
SNAPSHOTS = f"finplan-{ENV}-financialplanning-snapshots-{ACCT}"
RUN = "run_01KDVDNAZ83BAMMYCEGWF33DPM"
OTHER_RUN = "run_01KDVDNBYGX5V5HY2JSK5XWKHC"


@pytest.fixture(scope="module")
def job():
    return {"identity": [pol.job_execution_policy(ENV, **C)], "boundary": cb.research_permission_boundary(ENV, **C)}


@pytest.fixture(scope="module")
def api():
    return {"identity": [pol.control_role_policy(ENV, "api", **C)], "boundary": cb.env_permission_boundary(ENV, **C)}


def allowed(role, action, resource, **context) -> bool:
    return evaluate(Request(action, resource, context), role["identity"], role["boundary"]).allowed


def decision(role, action, resource, **context) -> str:
    return evaluate(Request(action, resource, context), role["identity"], role["boundary"]).decision


# ----------------------------------------------------------------- WS-03: approved snapshots, read-only
def test_ws03_job_role_reads_snapshots_and_never_writes_them(job):
    assert allowed(job, "s3:GetObject", _s3(SNAPSHOTS, "snapshots/snap_x/payload.json"))
    for action in ("s3:PutObject", "s3:DeleteObject", "s3:PutObjectTagging"):
        assert decision(job, action, _s3(SNAPSHOTS, "snapshots/snap_x/payload.json")) == "explicitDeny"
    for stem in ("raw", "curated", "plans", "reports"):
        assert decision(job, "s3:GetObject", _s3(f"finplan-{ENV}-financialplanning-{stem}-{ACCT}", "any")) == "explicitDeny"
    assert allowed(job, "execute-api:Invoke", f"arn:aws:execute-api:us-east-2:{ACCT}:api1/v1/GET/v1/snapshots/snap_x")


# ----------------------------------------------------------------- WS-05 / ENV-04: no authoritative plan state
def test_ws05_job_and_api_roles_cannot_write_plan_state(job, api):
    table = f"arn:aws:dynamodb:us-east-2:{ACCT}:table/finplan-{ENV}-financialplanning-plan-version"
    for role in (job, api):
        assert decision(role, "dynamodb:PutItem", table) == "explicitDeny"
        assert decision(role, "dynamodb:GetItem", table) == "explicitDeny"
        for method, path in (("POST", "v1/plans/pl_x/versions"), ("POST", "v1/publications"), ("POST", "v1/executions"), ("PUT", "v1/portfolios/pf_x"), ("POST", "v1/plans/pl_x/staged-outputs/run_x/accept")):
            assert decision(role, "execute-api:Invoke", f"arn:aws:execute-api:us-east-2:{ACCT}:api1/v1/{method}/{path}") == "explicitDeny", (method, path)
    assert allowed(api, "execute-api:Invoke", f"arn:aws:execute-api:us-east-2:{ACCT}:api1/v1/GET/v1/staged-outputs/{RUN}")


# ----------------------------------------------------------------- RST-04: write-only, write-once staging
def test_rst04_staging_is_write_only_and_write_once(job):
    own = _s3(STAGING, f"staging/{RUN}/manifest.json")
    assert allowed(job, "s3:PutObject", own, **{"s3:if-none-match": "*"})
    assert decision(job, "s3:PutObject", own) == "explicitDeny"  # no If-None-Match: could overwrite
    other = _s3(STAGING, f"staging/{OTHER_RUN}/plan-content.json")
    assert decision(job, "s3:PutObject", other) == "explicitDeny"  # overwrite attempt of another run's key
    for action in ("s3:GetObject", "s3:DeleteObject", "s3:ListBucket"):
        assert decision(job, action, own) == "explicitDeny"
    assert decision(job, "s3:ListBucket", _s3(STAGING)) == "explicitDeny"
    assert not allowed(job, "s3:PutObject", _s3(STAGING, f"accepted/{RUN}/manifest.json"), **{"s3:if-none-match": "*"})
    assert not allowed(job, "s3:PutObject", _s3(f"finplan-{ENV}-financialplanning-plans-{ACCT}", "plan/x.json"), **{"s3:if-none-match": "*"})


# ----------------------------------------------------------------- ENV-03: never another environment
def test_env03_gamma_roles_are_denied_prod(job, api):
    prod_research = n.bucket_name("prod", n.RESEARCH_BUCKET, ACCT)
    prod_staging = f"finplan-prod-financialplanning-run-staging-area-{ACCT}"
    for role in (job, api):
        assert decision(role, "s3:GetObject", _s3(prod_research, "runs/x/spec.json")) == "explicitDeny"
        assert decision(role, "s3:PutObject", _s3(prod_staging, f"staging/{RUN}/manifest.json"), **{"s3:if-none-match": "*"}) == "explicitDeny"
        assert decision(role, "ssm:GetParameter", f"arn:aws:ssm:us-east-2:{ACCT}:parameter/finplan/prod/financemodel/config/research-storage-ref") == "explicitDeny"
        assert decision(role, "dynamodb:Query", f"arn:aws:dynamodb:us-east-2:{ACCT}:table/finplan-prod-financemodel-job-control") == "explicitDeny"
    # resource side: the prod research bucket policy denies every gamma principal
    prod_policy = {"Version": "2012-10-17", "Statement": pol.research_bucket_policy("prod", _s3(prod_research), partition="aws", account=ACCT)}
    res = evaluate(Request("s3:GetObject", _s3(prod_research, "runs/x/spec.json"), {"aws:PrincipalArn": _role_arn(n.role_name(ENV, n.JOB_EXECUTION))}), [prod_policy])
    assert res.decision == "explicitDeny"


def test_env05_live_financial_actions_denied(job, api):
    for role in (job, api):
        assert decision(role, "payments:CreatePayment", "*") == "explicitDeny"
        assert decision(role, "secretsmanager:GetSecretValue", "*", **{"secretsmanager:SecretId": "prod/coinbase-api-key"}) == "explicitDeny"


# ----------------------------------------------------------------- research storage and registry (job role)
def test_job_role_research_storage_and_registry(job):
    assert allowed(job, "s3:GetObject", _s3(RESEARCH, f"runs/{RUN}/spec.json"))
    assert allowed(job, "s3:PutObject", _s3(RESEARCH, f"runs/{RUN}/job-result.json"))
    assert allowed(job, "s3:PutObject", _s3(RESEARCH, "artifacts/run_artifact/art_x"))
    assert not allowed(job, "s3:PutObject", _s3(RESEARCH, f"runs/{RUN}/spec.json"))  # the spec is the control plane's
    assert allowed(job, "s3:PutObject", _s3(REGISTRY, f"runs/{RUN}.json"), **{"s3:if-none-match": "*"})
    assert decision(job, "s3:PutObject", _s3(REGISTRY, f"runs/{RUN}.json")) == "explicitDeny"
    assert decision(job, "s3:DeleteObject", _s3(REGISTRY, "versions/mv_x.json")) == "explicitDeny"
    assert not allowed(job, "s3:PutObject", _s3(REGISTRY, "versions/mv_x.json"), **{"s3:if-none-match": "*"})
    assert not allowed(job, "sagemaker:CreateProcessingJob", f"arn:aws:sagemaker:us-east-2:{ACCT}:processing-job/fm-{ENV}-x")


# ----------------------------------------------------------------- control-plane roles
def test_control_role_starts_only_tagged_jobs_of_its_environment(api):
    own = f"arn:aws:sagemaker:us-east-2:{ACCT}:processing-job/fm-{ENV}-run-01kdvdnaz83bammycegwf33dpm-a1"
    tags = {"aws:RequestTag/environment": ENV, "aws:RequestTag/owner-repo": "financemodel"}
    assert allowed(api, "sagemaker:CreateProcessingJob", own, **tags)
    assert not allowed(api, "sagemaker:CreateProcessingJob", own)
    assert not allowed(api, "sagemaker:CreateProcessingJob", own, **{**tags, "aws:RequestTag/environment": "prod"})
    assert not allowed(api, "sagemaker:CreateProcessingJob", f"arn:aws:sagemaker:us-east-2:{ACCT}:processing-job/fm-prod-run-x", **tags)
    assert allowed(api, "sagemaker:StopProcessingJob", own)
    job_role = _role_arn(n.role_name(ENV, n.JOB_EXECUTION))
    assert allowed(api, "iam:PassRole", job_role, **{"iam:PassedToService": "sagemaker.amazonaws.com"})
    assert not allowed(api, "iam:PassRole", job_role, **{"iam:PassedToService": "lambda.amazonaws.com"})
    assert allowed(api, "lambda:InvokeFunction", f"arn:aws:lambda:us-east-2:{ACCT}:function:{n.function_name(ENV, n.DISPATCHER)}")
    dispatcher = {"identity": [pol.control_role_policy(ENV, "dispatcher", **C)], "boundary": cb.env_permission_boundary(ENV, **C)}
    assert not allowed(dispatcher, "lambda:InvokeFunction", f"arn:aws:lambda:us-east-2:{ACCT}:function:{n.function_name(ENV, n.DISPATCHER)}")


def test_registry_lookup_is_read_only():
    role = {"identity": [pol.registry_lookup_policy(ENV, **C)], "boundary": cb.env_permission_boundary(ENV, **C)}
    assert allowed(role, "s3:GetObject", _s3(REGISTRY, f"runs/{RUN}.json"))
    assert not allowed(role, "s3:PutObject", _s3(REGISTRY, f"runs/{RUN}.json"), **{"s3:if-none-match": "*"})
    assert not allowed(role, "s3:GetObject", _s3(RESEARCH, f"runs/{RUN}/spec.json"))


def test_approver_may_only_read_approve_and_cancel():
    api_arn = f"arn:aws:execute-api:us-east-2:{ACCT}:api1/api"
    role = {"identity": [approver_identity_policy(api_arn)], "boundary": cb.env_permission_boundary(ENV, **C)}
    assert allowed(role, "execute-api:Invoke", f"{api_arn}/POST/v1/jobs/{RUN}/approve")
    assert allowed(role, "execute-api:Invoke", f"{api_arn}/GET/v1/jobs/{RUN}")
    assert not allowed(role, "execute-api:Invoke", f"{api_arn}/POST/v1/jobs")
    assert not allowed(role, "sagemaker:CreateProcessingJob", "*")


# ----------------------------------------------------------------- pipeline roles (deploy, exec, stage)
def test_deploy_execution_role_is_scoped_to_its_environment():
    store = _s3(n.pipeline_store_bucket_name(ACCT))
    role = {"identity": [{"Version": "2012-10-17", "Statement": pol.deploy_execution_statements(ENV, store, **C)}], "boundary": cb.env_permission_boundary(ENV, **C)}
    target = _role_arn(f"finplan-{ENV}-financemodel-job-api-handler-role")
    for boundary in (cb.boundary_name(ENV), cb.research_boundary_name(ENV)):
        assert allowed(role, "iam:CreateRole", target, **{"iam:PermissionsBoundary": f"arn:aws:iam::{ACCT}:policy/{boundary}"})
    assert not allowed(role, "iam:CreateRole", target)
    assert decision(role, "iam:CreateRole", target, **{"iam:PermissionsBoundary": f"arn:aws:iam::{ACCT}:policy/AdministratorAccess"}) == "explicitDeny"
    assert decision(role, "s3:CreateBucket", _s3(n.bucket_name("prod", n.RESEARCH_BUCKET, ACCT))) == "explicitDeny"
    assert allowed(role, "s3:CreateBucket", _s3(n.bucket_name(ENV, n.RESEARCH_BUCKET, ACCT)))
    assert decision(role, "ssm:PutParameter", f"arn:aws:ssm:us-east-2:{ACCT}:parameter/finplan/{ENV}/financialplanning/api/plan-endpoint") == "explicitDeny"
    assert not allowed(role, "dynamodb:CreateTable", f"arn:aws:dynamodb:us-east-2:{ACCT}:table/finplan-{ENV}-financialplanning-plan")
    assert allowed(role, "dynamodb:CreateTable", f"arn:aws:dynamodb:us-east-2:{ACCT}:table/finplan-{ENV}-financemodel-job-control")


def test_stage_role_publishes_only_its_own_references():
    store = _s3(n.pipeline_store_bucket_name(ACCT))
    role = {"identity": [{"Version": "2012-10-17", "Statement": pol.stage_role_statements(ENV, store, **C)}], "boundary": cb.env_permission_boundary(ENV, **C)}
    param = lambda p: f"arn:aws:ssm:us-east-2:{ACCT}:parameter{p}"  # noqa: E731
    for p in ("release/manifest", "api/job-endpoint", "model/registry-ref", "job/job-role-ref", "job/run-backtest", "config/budget-enforced-role-names"):
        assert allowed(role, "ssm:PutParameter", param(f"/finplan/{ENV}/financemodel/{p}")), p
    assert not allowed(role, "ssm:PutParameter", param(f"/finplan/{ENV}/financemodel/config/instance-prices"))  # operator-owned
    assert not allowed(role, "ssm:PutParameter", param(f"/finplan/{ENV}/financialplanning/config/run-staging-ref"))
    assert decision(role, "ssm:GetParameter", param("/finplan/prod/financemodel/release/manifest")) == "explicitDeny"
    assert allowed(role, "ssm:GetParameter", param("/finplan/shared/financialplanning/config/budget-state"))
    assert decision(role, "execute-api:Invoke", f"arn:aws:execute-api:us-east-2:{ACCT}:api1/v1/POST/v1/plans/pl_x/versions") == "explicitDeny"


def test_build_role_pushes_only_the_financemodel_image():
    store = _s3(n.pipeline_store_bucket_name(ACCT))
    role = {"identity": [{"Version": "2012-10-17", "Statement": pol.build_role_statements(store, **C)}], "boundary": cb.shared_permission_boundary(**C)}
    repo = f"arn:aws:ecr:us-east-2:{ACCT}:repository/{n.ecr_repository_name()}"
    assert allowed(role, "ecr:PutImage", repo)
    assert not allowed(role, "ecr:PutImage", f"arn:aws:ecr:us-east-2:{ACCT}:repository/finplan-shared-financeagent-images")
    assert allowed(role, "s3:PutObject", f"{store}/assets/abc.zip")
    assert not allowed(role, "s3:PutObject", f"{store}/bootstrap/abc.json")


@pytest.mark.parametrize("kind", ["api", "dispatcher", "state"])
def test_control_roles_arm_only_their_own_dispatcher_schedule(kind):
    """Design D1: the control plane arms and disarms its own schedule; never another environment's."""
    role = {"identity": [pol.control_role_policy(ENV, kind, **C)], "boundary": cb.env_permission_boundary(ENV, **C)}
    own = pol.dispatcher_schedule_arn(ENV, **C)
    for action in ("scheduler:GetSchedule", "scheduler:UpdateSchedule"):
        assert allowed(role, action, own), action
        assert not allowed(role, action, pol.dispatcher_schedule_arn("prod", **C)), action
    assert not allowed(role, "scheduler:DeleteSchedule", own)
    assert not allowed(role, "scheduler:CreateSchedule", own)
    sched_role = _role_arn(n.role_name(ENV, n.SCHEDULE_ROLE))
    assert allowed(role, "iam:PassRole", sched_role, **{"iam:PassedToService": "scheduler.amazonaws.com"})
    assert not allowed(role, "iam:PassRole", sched_role, **{"iam:PassedToService": "lambda.amazonaws.com"})
    assert not allowed(role, "iam:PassRole", _role_arn(n.role_name("prod", n.SCHEDULE_ROLE)), **{"iam:PassedToService": "scheduler.amazonaws.com"})


# ----------------------------------------------------------------- PSS-03: single writer of the production-strategy key
def test_pss03_only_the_selection_role_writes_the_production_strategy_key(api):
    from finplan_contracts import ssm as contract_ssm

    key = f"arn:aws:ssm:us-east-2:{ACCT}:parameter{contract_ssm.production_strategy_parameter(ENV)}"
    sel = {"identity": [pol.strategy_selection_policy(ENV, **C)], "boundary": cb.env_permission_boundary(ENV, **C)}
    for action in ("ssm:PutParameter", "ssm:DeleteParameter", "ssm:GetParameter"):
        assert allowed(sel, action, key), action
    assert not allowed(sel, "ssm:PutParameter", f"arn:aws:ssm:us-east-2:{ACCT}:parameter/finplan/{ENV}/financemodel/config/auto-approve-usd")
    assert not allowed(sel, "ssm:PutParameter", f"arn:aws:ssm:us-east-2:{ACCT}:parameter/finplan/prod/financemodel/config/production-strategy")
    assert not allowed(sel, "sagemaker:CreateProcessingJob", "*")
    # the job API, dispatcher, state handler and job roles only read it
    for kind in ("api", "dispatcher", "state"):
        role = {"identity": [pol.control_role_policy(ENV, kind, **C)], "boundary": cb.env_permission_boundary(ENV, **C)}
        assert not allowed(role, "ssm:PutParameter", key) and not allowed(role, "ssm:DeleteParameter", key)
    assert allowed(api, "ssm:GetParameter", key)
    job = {"identity": [pol.job_execution_policy(ENV, **C)], "boundary": cb.research_permission_boundary(ENV, **C)}
    assert not allowed(job, "ssm:PutParameter", key)
    # the contract registers exactly this writer (tool, agent and platform roles are refused)
    from finplan_contracts.ssm import Writer, check_write

    path = contract_ssm.production_strategy_parameter(ENV)
    assert check_write(path, Writer("financemodel", "runtime", principal="strategy-selection")).allowed
    for repo in ("financelambdastool", "financeagent", "financialplanning"):
        assert not check_write(path, Writer(repo, "runtime", principal="strategy-selection")).allowed
