"""Names of every FinanceModel resource (task group 10; contracts naming ``finplan-<env>-<repo>-<logical>``).

One place for the names that IaC, the release publisher, the bootstrap, the platform grants and the
tests must agree on. Names only: no account ID, ARN or endpoint is ever written to a file. Values
that need the account (bucket names, ARNs) are built at deploy time from CloudFormation pseudo
parameters (``${AWS::AccountId}``) or at run time from the caller's own account.

Platform grants rely on two of these names (FinancialPlanning ``config/<env>.json``
``consumer_principals``):

* the job-execution role ``finplan-<env>-financemodel-job-execution-role`` matches
  ``finplan-<env>-financemodel-job-execution*`` (approved-snapshot reads, staging writes);
* the job API handler role ``finplan-<env>-financemodel-job-api-handler-role`` matches
  ``finplan-<env>-financemodel-job-api-handler*`` (``GET /v1/staged-outputs/*``).
"""

from __future__ import annotations

from finplan_contracts import boundaries as contract_boundaries
from finplan_contracts import ssm as contract_ssm

__all__ = [
    "APPROVER",
    "CONTROL_TABLE",
    "DISPATCHER",
    "ECR_REPOSITORY",
    "ENVIRONMENTS",
    "FUNCTIONS",
    "IMAGE_NAME",
    "JOB_API",
    "JOB_API_HANDLER",
    "JOB_EXECUTION",
    "JOB_EXECUTION_LOGICAL_ROLE",
    "PIPELINE_NAME",
    "PIPELINE_STORE_STEM",
    "PROCESSING_JOB_PREFIX",
    "REGISTRY_BUCKET",
    "REGISTRY_LOOKUP",
    "STRATEGY_SELECTION",
    "REPO",
    "RESEARCH_BUCKET",
    "SCHEDULE_ROLE",
    "STATE_CHANGE_RULE",
    "STATE_HANDLER",
    "TRAINING_STATE_CHANGE_RULE",
    "bucket_name",
    "deploy_role_name",
    "ecr_repository_name",
    "env_name",
    "exec_role_name",
    "function_name",
    "own_ssm",
    "pipeline_store_bucket_name",
    "processing_job_prefix",
    "role_name",
    "shared_name",
    "stage_role_name",
    "tooling_role_names",
]

REPO = "financemodel"
ENVIRONMENTS = ("beta", "gamma", "prod")

# ----------------------------------------------------------------- logical names (per environment)
RESEARCH_BUCKET = "research-workspace"
REGISTRY_BUCKET = "model-registry"
CONTROL_TABLE = "job-control"
JOB_API = "job-api"
#: Lambda functions (logical name -> handler). The role of each function is ``<logical>-role``.
JOB_API_HANDLER = "job-api-handler"
DISPATCHER = "job-dispatcher"
STATE_HANDLER = "job-state-handler"
REGISTRY_LOOKUP = "job-registry-lookup"
#: The strategy-selection operation (contracts 1.1.0 ``ssm.PRODUCTION_STRATEGY_WRITER``): the only
#: writer of ``/finplan/<env>/financemodel/config/production-strategy``.
STRATEGY_SELECTION = "strategy-selection"
FUNCTIONS: dict[str, str] = {
    JOB_API_HANDLER: "finplan_model.control.handlers.api_handler",
    DISPATCHER: "finplan_model.control.handlers.dispatcher_handler",
    STATE_HANDLER: "finplan_model.control.handlers.state_change_handler",
    REGISTRY_LOOKUP: "finplan_model.registry.handlers.lookup_handler",
    STRATEGY_SELECTION: "finplan_model.control.selection.selection_handler",
}
JOB_EXECUTION = "job-execution"
#: ``logical-role`` tag of the job-execution role (matrix row ``sagemaker-job-definitions``, contracts 1.0.0 D16).
JOB_EXECUTION_LOGICAL_ROLE = "job-execution-role"
APPROVER = "approver"
SCHEDULE_ROLE = "job-dispatcher-schedule"
STATE_CHANGE_RULE = "job-state-change"
#: SageMaker Training job state changes (job types with ``sagemaker_job: training``, e.g. model_selection).
TRAINING_STATE_CHANGE_RULE = "training-job-state-change"
#: SageMaker Processing and Training job names start with ``fm-<env>-`` (control plane, docs/job-execution.md).
PROCESSING_JOB_PREFIX = "fm-{env}-"

# ----------------------------------------------------------------- account-level (shared)
PIPELINE_STORE_STEM = "pipeline-store"
IMAGE_NAME = "financemodel-cpu"
ECR_REPOSITORY = "cpu-images"


def env_name(env: str, logical: str, suffix: str | None = None) -> str:
    """``finplan-<env>-financemodel-<logical>[-<suffix>]``."""
    return contract_boundaries.resource_name(env, REPO, logical, suffix)


def shared_name(logical: str, suffix: str | None = None) -> str:
    """``finplan-shared-financemodel-<logical>[-<suffix>]``."""
    return contract_boundaries.resource_name(contract_ssm.SHARED, REPO, logical, suffix)


def role_name(env: str, logical: str) -> str:
    name = env_name(env, logical, "role")
    if len(name) > 64:  # pragma: no cover - guarded by a unit test over every name
        raise ValueError(f"role name {name!r} exceeds 64 characters")
    return name


def function_name(env: str, logical: str) -> str:
    return env_name(env, logical)


def bucket_name(env: str, logical: str, account: str) -> str:
    """Globally unique bucket name: ``finplan-<env>-financemodel-<logical>-<account>``.

    ``account`` is a CloudFormation token (``Aws.ACCOUNT_ID``) in IaC, the caller's account at run
    time, or a placeholder in tests; it is never a literal in a repository file.
    """
    return f"{env_name(env, logical)}-{account}"


def pipeline_store_bucket_name(account: str) -> str:
    return f"{shared_name(PIPELINE_STORE_STEM)}-{account}"


def ecr_repository_name() -> str:
    """The account-level, digest-addressed image repository (matrix row financemodel-ecr-repositories)."""
    return shared_name(ECR_REPOSITORY)


def processing_job_prefix(env: str) -> str:
    return PROCESSING_JOB_PREFIX.format(env=env)


PIPELINE_NAME = shared_name("pipeline")


def deploy_role_name(env: str) -> str:
    """Deploy action role of ``env`` (CodePipeline assumes it); declared in the tooling stack."""
    return shared_name("deploy-role", env)


def exec_role_name(env: str) -> str:
    """CloudFormation execution role of ``env``."""
    return shared_name("deploy-role", f"{env}-exec")


def stage_role_name(env: str) -> str:
    """Manifest publisher and environment test runner of ``env`` (``finplan-<env>-financemodel-pipeline-stage-role``)."""
    return role_name(env, "pipeline-stage")


def tooling_role_names() -> list[str]:
    """Account-level (environment-independent) tooling roles of the FinanceModel pipeline.

    The bootstrap publishes them at ``/finplan/shared/financemodel/config/budget-enforced-role-names``
    (contracts 1.0.0, D16), so the platform's 100% budget deny covers them; the per-environment lists
    published by the pipeline no longer repeat them.
    """
    return [shared_name("pipeline", "role"), shared_name("pipeline-build-project", "role")]


def own_ssm(env: str, category: str, name: str) -> str:
    return contract_ssm.build(env, REPO, category, name)
