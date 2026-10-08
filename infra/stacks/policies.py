"""IAM documents of every FinanceModel role and resource policy (tasks 10.1-10.3; WS-03, WS-05, RST-04,
ENV-03, ENV-04, ENV-05).

Every document is a plain dict built from ``partition``, ``region`` and ``account``:

* the CDK stacks pass CloudFormation tokens (``Aws.PARTITION`` ...), so no account literal is ever
  written to a file;
* the unit tests pass concrete placeholders and evaluate requests offline with
  :func:`finplan_contracts.iam.evaluate`, together with the contract permission boundary the role
  carries (policy simulation, tasks 10.2 and 9.x).

Roles (names in :mod:`infra.stacks.naming`):

==========================  ===================  ==========================================================
role                        boundary             may
==========================  ===================  ==========================================================
job-execution (SageMaker)   research boundary    read its run spec, write its result and artifacts in
                                                 research storage; read approved snapshots (platform
                                                 bucket policy enforces ``approved``); write-once into its
                                                 run's staging prefix; record run lineage in the registry;
                                                 pull the CPU image; nothing else. Explicitly denied:
                                                 raw/curated/plan/report platform storage, snapshot writes,
                                                 staging reads/deletes/overwrites, platform metadata tables,
                                                 plan/publication/execution API writes
job-api-handler             environment          control table, start/stop/describe ``fm-<env>-*`` jobs
                                                 (tagged), pass the job role to SageMaker only, read SSM
                                                 (own env + shared), run hand-off documents, platform
                                                 ``GET /v1/snapshots/*`` and ``GET /v1/staged-outputs/*``,
                                                 register model versions, kick the dispatcher
job-dispatcher /            environment          as the API handler, without the dispatcher kick
job-state-handler
job-registry-lookup         environment          read the model registry only
job-dispatcher-schedule     environment          invoke the dispatcher only
approver                    environment          read the job API and approve/cancel runs only
==========================  ===================  ==========================================================
"""

from __future__ import annotations

from typing import Any

from finplan_contracts import boundaries as contract_boundaries
from finplan_contracts import iam as contract_iam

from . import naming as n

__all__ = [
    "DYNAMODB_DATA_ACTIONS",
    "PLATFORM_DENIED_BUCKET_STEMS",
    "build_role_statements",
    "control_role_policy",
    "control_table_policy",
    "deploy_execution_statements",
    "dispatcher_schedule_arn",
    "job_api_invoker_patterns",
    "job_execution_policy",
    "registry_bucket_policy",
    "registry_lookup_policy",
    "research_bucket_policy",
    "role_arn",
    "schedule_role_policy",
    "stage_role_statements",
    "staging_object_arn",
]

PARTITION = contract_iam.PARTITION
REGION = contract_iam.REGION
ACCOUNT = contract_iam.ACCOUNT

#: DynamoDB data-plane actions guarded by the control table's resource policy. DynamoDB table
#: resource policies reject stream actions (lesson from the platform deploy), so none is listed.
DYNAMODB_DATA_ACTIONS = (
    "dynamodb:BatchGetItem",
    "dynamodb:BatchWriteItem",
    "dynamodb:ConditionCheckItem",
    "dynamodb:DeleteItem",
    "dynamodb:GetItem",
    "dynamodb:PartiQLDelete",
    "dynamodb:PartiQLInsert",
    "dynamodb:PartiQLSelect",
    "dynamodb:PartiQLUpdate",
    "dynamodb:PutItem",
    "dynamodb:Query",
    "dynamodb:Scan",
    "dynamodb:UpdateItem",
)
#: Platform buckets research roles must never touch (only snapshots, read, and the staging area, write-once).
PLATFORM_DENIED_BUCKET_STEMS = ("raw", "curated", "plans", "reports")
_OBJECT_ACTIONS = ("s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:GetObjectTagging", "s3:PutObjectTagging")
_TABLE_ITEM_ACTIONS = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:ConditionCheckItem"]


# ===================================================================== ARNs
def _arn(service: str, resource: str, *, partition: str, region: str = "", account: str = "") -> str:
    return f"arn:{partition}:{service}:{region}:{account}:{resource}"


def role_arn(name_pattern: str, *, partition: str = PARTITION, account: str = ACCOUNT) -> str:
    return f"arn:{partition}:iam::{account}:role/{name_pattern}"


def _bucket_arn(name: str, partition: str) -> str:
    return f"arn:{partition}:s3:::{name}"


def staging_object_arn(env: str, *, partition: str = PARTITION) -> str:
    """Objects of the platform run-output staging area of ``env`` (the ``staging/run_*/`` prefixes)."""
    return f"arn:{partition}:s3:::finplan-{env}-financialplanning-run-staging-area*/staging/run_*/*"


def _platform_bucket(env: str, stem: str, partition: str) -> list[str]:
    base = f"arn:{partition}:s3:::finplan-{env}-financialplanning-{stem}*"
    return [base, base + "/*"]


def _log_statement(env: str, logical: str, *, partition: str, region: str, account: str) -> dict[str, Any]:
    group = f"/aws/lambda/{n.function_name(env, logical)}"
    return {
        "Sid": "OwnLogStreams",
        "Effect": "Allow",
        "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
        "Resource": [_arn("logs", f"log-group:{group}:*", partition=partition, region=region, account=account)],
    }


def _ssm_param(path: str, *, partition: str, region: str, account: str) -> str:
    return contract_iam.ssm_parameter_arn(path, partition=partition, region=region, account=account)


def _registry_write_once(registry_arn: str) -> list[dict[str, Any]]:
    """Model-registry records are write-once and never deleted (REG-02), for every role that may write them."""
    return [
        {"Sid": "DenyRegistryOverwrite", "Effect": "Deny", "Action": ["s3:PutObject"], "Resource": [f"{registry_arn}/*"], "Condition": {"Null": {"s3:if-none-match": "true"}}},
        {"Sid": "DenyRegistryDelete", "Effect": "Deny", "Action": ["s3:DeleteObject", "s3:DeleteObjectVersion"], "Resource": [f"{registry_arn}/*"]},
    ]


def _platform_reads(region: str, account: str, partition: str, *routes: str) -> list[str]:
    return [_arn("execute-api", f"*/*/GET/{r}", partition=partition, region=region, account=account) for r in routes]


def _plan_api_write_arns(region: str, account: str, partition: str) -> list[str]:
    return [
        _arn("execute-api", f"*/*/{m}/{p}", partition=partition, region=region, account=account)
        for m in contract_boundaries.PLAN_API_WRITE_METHOD_PATTERNS
        for p in contract_boundaries.PLAN_API_WRITE_PATH_PATTERNS
    ]


# ===================================================================== job execution role (research boundary)
def job_execution_policy(env: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> dict[str, Any]:
    """Identity policy of ``finplan-<env>-financemodel-job-execution-role`` (WS-03, WS-05, RST-04)."""
    research = _bucket_arn(n.bucket_name(env, n.RESEARCH_BUCKET, account), partition)
    registry = _bucket_arn(n.bucket_name(env, n.REGISTRY_BUCKET, account), partition)
    staging = staging_object_arn(env, partition=partition)
    staging_bucket = f"arn:{partition}:s3:::finplan-{env}-financialplanning-run-staging-area*"
    snapshots = _platform_bucket(env, "snapshots", partition)
    repo = _arn("ecr", f"repository/{n.ecr_repository_name()}", partition=partition, region=region, account=account)
    denied_platform: list[str] = []
    for stem in PLATFORM_DENIED_BUCKET_STEMS:
        denied_platform += [f"arn:{partition}:s3:::finplan-*-financialplanning-{stem}*", f"arn:{partition}:s3:::finplan-*-financialplanning-{stem}*/*"]
    param = lambda p: _ssm_param(p, partition=partition, region=region, account=account)  # noqa: E731
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "ReadRunSpecs", "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [f"{research}/runs/*/spec.json"]},
            {
                "Sid": "WriteRunOutputs",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject"],
                "Resource": [f"{research}/runs/*/job-result.json", f"{research}/artifacts/*", f"{research}/dataset-catalog/*", f"{research}/scratch/*"],
            },
            {
                "Sid": "ListResearchPrefixes",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": [research],
                "Condition": {"StringLike": {"s3:prefix": ["runs/*", "artifacts/*", "dataset-catalog/*", "scratch/*"]}},
            },
            {"Sid": "RecordRunLineage", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": [f"{registry}/runs/*"]},
            {"Sid": "ReadRegistry", "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [f"{registry}/versions/*", f"{registry}/identity/*", f"{registry}/events/*"]},
            {"Sid": "ListRegistry", "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [registry]},
            {"Sid": "ReadApprovedSnapshots", "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [snapshots[1]]},
            {"Sid": "WriteOwnRunStaging", "Effect": "Allow", "Action": ["s3:PutObject"], "Resource": [staging]},
            {
                "Sid": "PlatformKeyThroughS3",
                "Effect": "Allow",
                "Action": ["kms:Decrypt", "kms:GenerateDataKey"],
                "Resource": [_arn("kms", "key/*", partition=partition, region=region, account=account)],
                "Condition": {
                    "StringEquals": {
                        "kms:ViaService": f"s3.{region}.amazonaws.com",
                        "aws:ResourceTag/environment": env,
                        "aws:ResourceTag/owner-repo": "financialplanning",
                    }
                },
            },
            {"Sid": "PlatformSnapshotReads", "Effect": "Allow", "Action": ["execute-api:Invoke"], "Resource": _platform_reads(region, account, partition, "v1/snapshots/*")},
            {
                "Sid": "ReadOwnReferences",
                "Effect": "Allow",
                "Action": ["ssm:GetParameter"],
                "Resource": [param(f"/finplan/{env}/financemodel/config/*"), param(f"/finplan/{env}/financialplanning/api/plan-endpoint"), param(f"/finplan/{env}/financialplanning/config/run-staging-ref")],
            },
            {"Sid": "PullImage", "Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], "Resource": [repo]},
            {"Sid": "EcrToken", "Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": ["*"]},
            {
                "Sid": "ProcessingJobLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
                "Resource": [
                    _arn("logs", "log-group:/aws/sagemaker/ProcessingJobs", partition=partition, region=region, account=account),
                    _arn("logs", "log-group:/aws/sagemaker/ProcessingJobs:*", partition=partition, region=region, account=account),
                ],
            },
            {"Sid": "ProcessingJobMetrics", "Effect": "Allow", "Action": ["cloudwatch:PutMetricData"], "Resource": ["*"], "Condition": {"StringEquals": {"cloudwatch:namespace": "/aws/sagemaker/ProcessingJobs"}}},
            # ---- explicit denies (they hold even if an allow above were widened)
            {"Sid": "DenyRawCuratedPlanReportStorage", "Effect": "Deny", "Action": ["s3:*"], "Resource": denied_platform},
            {"Sid": "DenySnapshotWrites", "Effect": "Deny", "Action": list(contract_boundaries.S3_WRITE_ACTIONS), "Resource": _platform_bucket("*", "snapshots", partition)},
            {"Sid": "DenyStagingReadListDelete", "Effect": "Deny", "Action": ["s3:GetObject*", "s3:ListBucket*", "s3:DeleteObject*", "s3:RestoreObject"], "Resource": [staging_bucket, staging_bucket + "/*"]},
            {"Sid": "DenyStagingOverwrite", "Effect": "Deny", "Action": ["s3:PutObject"], "Resource": [staging_bucket + "/*"], "Condition": {"Null": {"s3:if-none-match": "true"}}},
            {"Sid": "DenyPlatformMetadata", "Effect": "Deny", "Action": ["dynamodb:*"], "Resource": [_arn("dynamodb", "table/finplan-*-financialplanning-*", partition=partition, region=region, account=account)]},
            {"Sid": "DenyPlanPublicationExecutionWrites", "Effect": "Deny", "Action": ["execute-api:Invoke"], "Resource": _plan_api_write_arns(region, account, partition)},
            *_registry_write_once(registry),
        ],
    }


# ===================================================================== control-plane roles
def dispatcher_schedule_arn(env: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> str:
    """The dispatcher schedule (EventBridge Scheduler, default group)."""
    return _arn("scheduler", f"schedule/default/{n.env_name(env, n.DISPATCHER)}", partition=partition, region=region, account=account)


def control_role_policy(env: str, kind: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> dict[str, Any]:
    """Identity policy of the control-plane Lambdas (``kind``: ``api``, ``dispatcher`` or ``state``).

    Wiring follows docs/job-execution.md ("Wiring needed from the infrastructure task group").
    """
    if kind not in ("api", "dispatcher", "state"):
        raise ValueError(kind)
    logical = {"api": n.JOB_API_HANDLER, "dispatcher": n.DISPATCHER, "state": n.STATE_HANDLER}[kind]
    table = _arn("dynamodb", f"table/{n.env_name(env, n.CONTROL_TABLE)}", partition=partition, region=region, account=account)
    research = _bucket_arn(n.bucket_name(env, n.RESEARCH_BUCKET, account), partition)
    registry = _bucket_arn(n.bucket_name(env, n.REGISTRY_BUCKET, account), partition)
    jobs = _arn("sagemaker", f"processing-job/{n.processing_job_prefix(env)}*", partition=partition, region=region, account=account)
    job_role = role_arn(n.role_name(env, n.JOB_EXECUTION), partition=partition, account=account)
    param = lambda p: _ssm_param(p, partition=partition, region=region, account=account)  # noqa: E731
    st: list[dict[str, Any]] = [
        {"Sid": "ControlTable", "Effect": "Allow", "Action": _TABLE_ITEM_ACTIONS, "Resource": [table, f"{table}/index/*"]},
        {"Sid": "ControlTableTransactions", "Effect": "Allow", "Action": ["dynamodb:ConditionCheckItem"], "Resource": [table]},
        {
            "Sid": "StartTaggedJobs",
            "Effect": "Allow",
            "Action": ["sagemaker:CreateProcessingJob", "sagemaker:AddTags"],
            "Resource": [jobs],
            "Condition": {"StringEquals": {"aws:RequestTag/environment": env, "aws:RequestTag/owner-repo": "financemodel"}},
        },
        {"Sid": "ManageOwnJobs", "Effect": "Allow", "Action": ["sagemaker:StopProcessingJob", "sagemaker:DescribeProcessingJob"], "Resource": [jobs]},
        {"Sid": "PassJobRoleToSageMaker", "Effect": "Allow", "Action": ["iam:PassRole"], "Resource": [job_role], "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
        {"Sid": "ReadSettings", "Effect": "Allow", "Action": ["ssm:GetParameter"], "Resource": [param(f"/finplan/{env}/*"), param("/finplan/shared/*")]},
        {"Sid": "RunHandOff", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": [f"{research}/runs/*"]},
        {"Sid": "ListRunHandOff", "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [research], "Condition": {"StringLike": {"s3:prefix": ["runs/*"]}}},
        {"Sid": "ModelRegistry", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": [f"{registry}/identity/*", f"{registry}/versions/*", f"{registry}/events/*", f"{registry}/runs/*"]},
        {"Sid": "ListModelRegistry", "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [registry]},
        {"Sid": "PlatformReads", "Effect": "Allow", "Action": ["execute-api:Invoke"], "Resource": _platform_reads(region, account, partition, "v1/snapshots/*", "v1/staged-outputs/*")},
        _log_statement(env, logical, partition=partition, region=region, account=account),
    ]
    # Arm and disarm the dispatcher schedule (design D1; finplan_model.control.wakeup). UpdateSchedule
    # re-submits the schedule's target, which passes the schedule role to EventBridge Scheduler.
    st.append({"Sid": "ArmDispatcherSchedule", "Effect": "Allow", "Action": ["scheduler:GetSchedule", "scheduler:UpdateSchedule"], "Resource": [dispatcher_schedule_arn(env, partition=partition, region=region, account=account)]})
    st.append({"Sid": "PassScheduleRoleToScheduler", "Effect": "Allow", "Action": ["iam:PassRole"], "Resource": [role_arn(n.role_name(env, n.SCHEDULE_ROLE), partition=partition, account=account)], "Condition": {"StringEquals": {"iam:PassedToService": "scheduler.amazonaws.com"}}})
    if kind == "api":
        st.append({"Sid": "KickDispatcher", "Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": [_arn("lambda", f"function:{n.function_name(env, n.DISPATCHER)}", partition=partition, region=region, account=account)]})
    st.append({"Sid": "DenyPlatformMetadata", "Effect": "Deny", "Action": ["dynamodb:*"], "Resource": [_arn("dynamodb", "table/finplan-*-financialplanning-*", partition=partition, region=region, account=account)]})
    st.append({"Sid": "DenyPlanPublicationExecutionWrites", "Effect": "Deny", "Action": ["execute-api:Invoke"], "Resource": _plan_api_write_arns(region, account, partition)})
    st += _registry_write_once(registry)
    return {"Version": "2012-10-17", "Statement": st}


def registry_lookup_policy(env: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> dict[str, Any]:
    registry = _bucket_arn(n.bucket_name(env, n.REGISTRY_BUCKET, account), partition)
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "ReadRegistry", "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [f"{registry}/*"]},
            {"Sid": "ListRegistry", "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [registry]},
            _log_statement(env, n.REGISTRY_LOOKUP, partition=partition, region=region, account=account),
        ],
    }


def schedule_role_policy(env: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> dict[str, Any]:
    fn = _arn("lambda", f"function:{n.function_name(env, n.DISPATCHER)}", partition=partition, region=region, account=account)
    return {"Version": "2012-10-17", "Statement": [{"Sid": "InvokeDispatcher", "Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": [fn, f"{fn}:*"]}]}


def strategy_selection_policy(env: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> dict[str, Any]:
    """The ``strategy-selection`` role (design M3): the ONLY principal with ``ssm:PutParameter`` /
    ``ssm:DeleteParameter`` on the production-strategy key; reads the run store (evaluation
    evidence) and appends idempotency and audit items; nothing else."""
    from finplan_contracts import ssm as contract_ssm

    table = _arn("dynamodb", f"table/{n.env_name(env, n.CONTROL_TABLE)}", partition=partition, region=region, account=account)
    key = _ssm_param(contract_ssm.production_strategy_parameter(env), partition=partition, region=region, account=account)
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "ProductionStrategyKey", "Effect": "Allow", "Action": ["ssm:GetParameter", "ssm:PutParameter", "ssm:DeleteParameter"], "Resource": [key]},
            {"Sid": "ReadRunStore", "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:Query"], "Resource": [table, f"{table}/index/*"]},
            {"Sid": "AppendIdempotencyAndAudit", "Effect": "Allow", "Action": ["dynamodb:PutItem"], "Resource": [table], "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["IDEM#*", "AUDIT#*"]}}},
            _log_statement(env, n.STRATEGY_SELECTION, partition=partition, region=region, account=account),
        ],
    }


def daily_trigger_role_pattern(env: str) -> str:
    """The FinancialPlanning daily trigger step role (submits ``daily_recommendation`` only)."""
    return f"finplan-{env}-financialplanning-daily-trigger-step-role"


def job_api_invoker_patterns(env: str) -> list[str]:
    """Role-name patterns the job API admits (JOB-01; FinanceLambdasTool tool roles, platform
    production-candidate callers and the lineage lookup, FinanceModel's own pipeline stage role)."""
    return [
        f"finplan-{env}-financelambdastool-*",
        f"finplan-{env}-financialplanning-plan-api-handler-role",
        f"finplan-{env}-financialplanning-operator*",
        n.stage_role_name(env),
    ]


# ===================================================================== resource policies
def _others(env: str, partition: str, account: str) -> list[str]:
    return [role_arn(f"finplan-{o}-*", partition=partition, account=account) for o in n.ENVIRONMENTS if o != env]


def research_bucket_policy(env: str, bucket_arn: str, *, partition: str = PARTITION, account: str = ACCOUNT) -> list[dict[str, Any]]:
    """Research storage (WS-01): other environments' principals denied; object access only for
    FinanceModel principals of this environment. TLS-only and Block Public Access are set by the stack."""
    objects = f"{bucket_arn}/*"
    return [
        {"Sid": "DenyOtherEnvironmentPrincipals", "Effect": "Deny", "Principal": "*", "Action": "s3:*", "Resource": [bucket_arn, objects], "Condition": {"ArnLike": {"aws:PrincipalArn": _others(env, partition, account)}}},
        {
            "Sid": "DenyObjectAccessOutsideFinanceModel",
            "Effect": "Deny",
            "Principal": "*",
            "Action": list(_OBJECT_ACTIONS),
            "Resource": objects,
            "Condition": {"ArnNotLike": {"aws:PrincipalArn": [role_arn(f"finplan-{env}-financemodel-*", partition=partition, account=account)]}},
        },
    ]


def registry_bucket_policy(env: str, bucket_arn: str, *, partition: str = PARTITION, account: str = ACCOUNT) -> list[dict[str, Any]]:
    """The model registry (REG-02): research-storage isolation plus write-once records (every PutObject
    must carry ``If-None-Match``) and no deletes at all."""
    objects = f"{bucket_arn}/*"
    return [
        *research_bucket_policy(env, bucket_arn, partition=partition, account=account),
        {"Sid": "DenyPutWithoutIfNoneMatch", "Effect": "Deny", "Principal": "*", "Action": "s3:PutObject", "Resource": objects, "Condition": {"Null": {"s3:if-none-match": "true"}}},
        {"Sid": "DenyDeleteRegistryRecords", "Effect": "Deny", "Principal": "*", "Action": ["s3:DeleteObject", "s3:DeleteObjectVersion"], "Resource": objects},
    ]


def control_table_policy(env: str, table_arn: str, *, partition: str = PARTITION, account: str = ACCOUNT) -> dict[str, Any]:
    """Run/lease/idempotency table: data access only for FinanceModel principals of this environment.
    No stream action is listed (DynamoDB table resource policies reject them)."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyDataAccessOutsideFinanceModel",
                "Effect": "Deny",
                "Principal": "*",
                "Action": list(DYNAMODB_DATA_ACTIONS),
                "Resource": [table_arn, f"{table_arn}/index/*"],
                "Condition": {"ArnNotLike": {"aws:PrincipalArn": [role_arn(f"finplan-{env}-financemodel-*", partition=partition, account=account)]}},
            }
        ],
    }


# ===================================================================== pipeline roles
def deploy_execution_statements(env: str, store_bucket_arn: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> list[dict[str, Any]]:
    """CloudFormation execution role of ``env``: only ``finplan-<env>-financemodel-*`` resources; roles it
    creates must carry ``finplan-<env>-permission-boundary`` or the research boundary."""
    prefix = f"finplan-{env}-{n.REPO}-"
    a = lambda service, res, reg=True, acct=True: _arn(service, res, partition=partition, region=region if reg else "", account=account if acct else "")  # noqa: E731
    boundaries = [a("iam", f"policy/{contract_boundaries.boundary_name(env)}", reg=False), a("iam", f"policy/{contract_boundaries.research_boundary_name(env)}", reg=False)]
    role_res = a("iam", f"role/{prefix}*", reg=False)
    stmts: list[dict[str, Any]] = [
        {"Sid": "EnvBuckets", "Effect": "Allow", "Action": ["s3:*"], "Resource": [f"arn:{partition}:s3:::{prefix}*", f"arn:{partition}:s3:::{prefix}*/*"]},
        {"Sid": "ReadPublishedAssets", "Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion"], "Resource": [f"{store_bucket_arn}/assets/*"]},
        {"Sid": "EnvTables", "Effect": "Allow", "Action": ["dynamodb:*"], "Resource": [a("dynamodb", f"table/{prefix}*"), a("dynamodb", f"table/{prefix}*/*")]},
        {"Sid": "EnvFunctions", "Effect": "Allow", "Action": ["lambda:*"], "Resource": [a("lambda", f"function:{prefix}*")]},
        {"Sid": "EnvRolesCreateWithBoundary", "Effect": "Allow", "Action": ["iam:CreateRole", "iam:PutRolePermissionsBoundary"], "Resource": [role_res], "Condition": {"StringEquals": {"iam:PermissionsBoundary": boundaries}}},
        {
            "Sid": "EnvRolesManage",
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "iam:GetRolePolicy",
                "iam:ListRolePolicies",
                "iam:ListAttachedRolePolicies",
                "iam:ListRoleTags",
                "iam:DeleteRole",
                "iam:PutRolePolicy",
                "iam:DeleteRolePolicy",
                "iam:UpdateRole",
                "iam:UpdateRoleDescription",
                "iam:UpdateAssumeRolePolicy",
                "iam:TagRole",
                "iam:UntagRole",
                "iam:PassRole",
            ],
            "Resource": [role_res],
        },
        {"Sid": "RestApis", "Effect": "Allow", "Action": ["apigateway:*"], "Resource": [a("apigateway", p, acct=False) for p in ("/restapis", "/restapis/*", "/tags/*")]},
        {"Sid": "EnvSchedules", "Effect": "Allow", "Action": ["scheduler:*"], "Resource": [a("scheduler", f"schedule/*/{prefix}*")]},
        {"Sid": "EnvRules", "Effect": "Allow", "Action": ["events:*"], "Resource": [a("events", f"rule/{prefix}*")]},
        {"Sid": "EnvLogGroups", "Effect": "Allow", "Action": ["logs:*"], "Resource": [a("logs", f"log-group:/aws/lambda/{prefix}*"), a("logs", f"log-group:/aws/lambda/{prefix}*:*")]},
        {"Sid": "LogGroupsDescribe", "Effect": "Allow", "Action": ["logs:DescribeLogGroups"], "Resource": ["*"]},
    ]
    stmts += list(contract_iam.ssm_access_policy(n.REPO, env, partition=partition, region=region, account=account)["Statement"])
    return stmts


def stage_role_statements(env: str, store_bucket_arn: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> list[dict[str, Any]]:
    """Manifest publisher, reference publisher, registry seeder and test runner of ``env``."""
    own = f"/finplan/{env}/{n.REPO}"
    param = lambda p: _ssm_param(p, partition=partition, region=region, account=account)  # noqa: E731
    registry = _bucket_arn(n.bucket_name(env, n.REGISTRY_BUCKET, account), partition)
    return [
        {
            "Sid": "PublishOwnReferences",
            "Effect": "Allow",
            "Action": ["ssm:PutParameter", "ssm:AddTagsToResource"],
            "Resource": [param(f"{own}/release/*"), param(f"{own}/api/*"), param(f"{own}/model/*"), param(f"{own}/job/*"), param(f"{own}/config/research-storage-ref"), param(f"{own}/config/registry-storage-ref"), param(f"{own}/config/approver-role-ref"), param(f"{own}/config/budget-enforced-role-names")],
        },
        {"Sid": "ReadEnvAndShared", "Effect": "Allow", "Action": list(contract_iam.SSM_READ_ACTIONS), "Resource": [param(f"/finplan/{env}"), param(f"/finplan/{env}/*"), param("/finplan/shared"), param("/finplan/shared/*")]},
        {"Sid": "ReadStackOutputs", "Effect": "Allow", "Action": ["cloudformation:DescribeStacks"], "Resource": [_arn("cloudformation", f"stack/finplan-{env}-{n.REPO}-*/*", partition=partition, region=region, account=account)]},
        {"Sid": "ReleaseLedger", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": [f"{store_bucket_arn}/releases/*"]},
        {"Sid": "SeedModelRegistry", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": [f"{registry}/identity/*", f"{registry}/versions/*", f"{registry}/events/*"]},
        {"Sid": "ListModelRegistry", "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [registry]},
        {"Sid": "ApprovalRecord", "Effect": "Allow", "Action": ["codepipeline:ListActionExecutions", "codepipeline:GetPipelineExecution"], "Resource": [_arn("codepipeline", n.PIPELINE_NAME, partition=partition, region=region, account=account)]},
        {"Sid": "CallOwnJobApi", "Effect": "Allow", "Action": ["execute-api:Invoke"], "Resource": [_arn("execute-api", "*/*/*/v1/*", partition=partition, region=region, account=account)]},
        # deployed suite: one synchronous dispatcher tick proves the Lambda bundle imports (platform lesson L3/L5)
        {"Sid": "InvokeOwnDispatcher", "Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": [_arn("lambda", f"function:{n.function_name(env, n.DISPATCHER)}", partition=partition, region=region, account=account)]},
        {"Sid": "DenyPlanPublicationExecutionWrites", "Effect": "Deny", "Action": ["execute-api:Invoke"], "Resource": _plan_api_write_arns(region, account, partition)},
        *_registry_write_once(registry),
    ]


def build_role_statements(store_bucket_arn: str, *, partition: str = PARTITION, region: str = REGION, account: str = ACCOUNT) -> list[dict[str, Any]]:
    """Build stage: publish assets and the release ledger, push the CPU image (built once)."""
    repo = _arn("ecr", f"repository/{n.ecr_repository_name()}", partition=partition, region=region, account=account)
    return [
        {"Sid": "PublishAssetsAndReleases", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": [f"{store_bucket_arn}/assets/*", f"{store_bucket_arn}/releases/*"]},
        {"Sid": "ListStore", "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [store_bucket_arn], "Condition": {"StringLike": {"s3:prefix": ["assets/*", "releases/*"]}}},
        {
            "Sid": "PushImage",
            "Effect": "Allow",
            "Action": ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:CompleteLayerUpload", "ecr:DescribeImages", "ecr:GetDownloadUrlForLayer", "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart"],
            "Resource": [repo],
        },
        {"Sid": "EcrToken", "Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": ["*"]},
    ]
