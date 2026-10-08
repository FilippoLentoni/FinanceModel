"""FinanceModel account-level stacks (environment ``shared``; task 10.5), deployed **only** by the
authenticated bootstrap (``scripts/bootstrap.py``, ``docs/bootstrap.md``), never by the pipeline.

Mirrors the FinancialPlanning tooling pattern that is already deployed (contracts D6, D11, D13):

* ``finplan-shared-financemodel-pipeline-store`` (:class:`StoreStack`): the pipeline store bucket
  ``finplan-shared-financemodel-pipeline-store-<account>`` (CodePipeline artifacts, the
  content-addressed CDK file assets under ``assets/``, the release ledger under ``releases/``, the
  staged tooling template under ``bootstrap/``). SSE-S3, TLS only, Block Public Access, versioned;
  ``releases/`` never expires. Retained on deletion. It is small and asset-free, so it uses
  :class:`aws_cdk.LegacyStackSynthesizer`: the CLI deploys it inline with the operator's credentials
  and references **no** ``cdk-hnb659fds`` role and **no** ``cdk-*-assets`` bucket (lesson of the
  platform bootstrap).
* ``finplan-shared-financemodel-tooling`` (:class:`ToolingStack`): the account-level image
  repository ``finplan-shared-financemodel-cpu-images`` (matrix row ``financemodel-ecr-repositories``:
  immutable tags, scan on push, digest-addressed; only release-tagged images are kept, the newest
  :data:`KEEP_RELEASE_IMAGES`) and the pipeline (:mod:`infra.stacks.pipeline` adds it). It uses a
  :class:`aws_cdk.CliCredentialsStackSynthesizer` that stages the template in the store under
  ``bootstrap/`` with the operator's CLI credentials (no ``CDKToolkit``).

FinanceModel creates **no** permission boundary and **no** budget: both belong to the
FinancialPlanning tooling stack (contracts D1, "Shared budget alarms"; the boundaries
``finplan-<env>-permission-boundary``, ``finplan-<env>-research-permission-boundary`` and
``finplan-shared-permission-boundary`` already exist). Every role here carries the shared boundary by
name, the per-environment pipeline roles carry their environment's boundary. FinanceModel only
**publishes** its enforced role names (``/finplan/<env>/financemodel/config/budget-enforced-role-names``,
``scripts/release.py``) for the platform's budget action.

Environment stacks deployed by the pipeline use :func:`deployment_synthesizer`: file assets in the
store under ``assets/`` (published by the build stage), no bootstrap-version rule and no role ARN.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import aws_cdk as cdk
import jsii
from aws_cdk import Aws, Duration, Tags
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct
from finplan_contracts import boundaries as contract_boundaries
from finplan_contracts import ssm as contract_ssm

from . import naming as n
from .common import tag_role

__all__ = [
    "ASSET_PREFIX",
    "BOOTSTRAP_PREFIX",
    "CACHE_PREFIX",
    "KEEP_RELEASE_IMAGES",
    "LOG_RETENTION",
    "RELEASES_PREFIX",
    "STORE_CONSTRUCT_ID",
    "STORE_STACK_NAME",
    "TOOLING_CONSTRUCT_ID",
    "TOOLING_STACK_NAME",
    "StoreStack",
    "ToolingStack",
    "add_log_group",
    "add_to_app",
    "deployment_synthesizer",
    "get_tooling_stack",
    "tooling_synthesizer",
]

TOOLING_CONSTRUCT_ID = "Tooling"
TOOLING_STACK_NAME = n.shared_name("tooling")
STORE_CONSTRUCT_ID = "PipelineStore"
STORE_STACK_NAME = n.shared_name("pipeline-store")
ASSET_PREFIX = "assets/"
BOOTSTRAP_PREFIX = "bootstrap/"
RELEASES_PREFIX = "releases/"
CACHE_PREFIX = "cache/"
STORE_PREFIX_EXPIRY_DAYS = 30
STORE_NONCURRENT_EXPIRY_DAYS = 7
STORE_ABORT_MULTIPART_DAYS = 7
#: Release-tagged images kept in the repository (rollback window; storage is billed per GB-month).
KEEP_RELEASE_IMAGES = 10
QUALIFIER = "finplan"
#: Retention of every explicit log group of the account-level stacks (mirrors FinancialPlanning).
LOG_RETENTION = logs.RetentionDays.ONE_MONTH


def add_log_group(scope: Construct, cid: str, name: str, logical_role: str) -> logs.LogGroup:
    """An explicit log group: 30-day retention, deleted with the stack, tagged with its owner's logical role.

    Same helper as the FinancialPlanning tooling stack's ``add_log_group``. Declaring the group
    (instead of letting the service create it on first write) bounds the retention and lets the
    stack remove it. The ``logical-role`` tag attributes it to the owning matrix row (the group's
    name references no template resource, so it cannot be parent-attributed).
    """
    group = logs.LogGroup(scope, cid, log_group_name=name, retention=LOG_RETENTION, removal_policy=cdk.RemovalPolicy.DESTROY)
    tag_role(group, logical_role)
    return group


def _shared_tags(stack: cdk.Stack) -> None:
    base = contract_ssm.cost_allocation_tags(n.REPO, contract_ssm.SHARED, "placeholder")
    for key in ("project", "owner-repo", "environment"):
        Tags.of(stack).add(key, base[key])


def _env_stack_synthesizer() -> cdk.CliCredentialsStackSynthesizer:
    return cdk.CliCredentialsStackSynthesizer(
        file_assets_bucket_name=n.pipeline_store_bucket_name("${AWS::AccountId}"),
        bucket_prefix=ASSET_PREFIX,
        image_assets_repository_name=n.ecr_repository_name(),
        qualifier=QUALIFIER,
    )


@jsii.implements(cdk.IReusableStackSynthesizer)
class _PerStackSynthesizer:
    """App-level default that binds a fresh synthesizer to every stack (a shared instance would share
    one asset-manifest builder across stacks in this CDK version)."""

    def reusable_bind(self, stack: cdk.Stack) -> cdk.IBoundStackSynthesizer:
        synth = _env_stack_synthesizer()
        synth.bind(stack)
        return synth


def deployment_synthesizer() -> cdk.IReusableStackSynthesizer:
    """Synthesizer of the environment stacks the pipeline deploys (``scripts/synth.py``)."""
    return _PerStackSynthesizer()


def tooling_synthesizer() -> cdk.CliCredentialsStackSynthesizer:
    """The tooling template is staged in the store (``bootstrap/``) with the operator's CLI credentials."""
    return cdk.CliCredentialsStackSynthesizer(
        file_assets_bucket_name=n.pipeline_store_bucket_name("${AWS::AccountId}"),
        bucket_prefix=BOOTSTRAP_PREFIX,
        image_assets_repository_name=n.ecr_repository_name(),
        qualifier=QUALIFIER,
    )


class StoreStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, *, shared: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(
            scope,
            construct_id,
            stack_name=STORE_STACK_NAME,
            env=cdk.Environment(region=str(shared["region"])),
            synthesizer=cdk.LegacyStackSynthesizer(),
            termination_protection=True,
            description="FinanceModel pipeline store (environment shared): pipeline artifacts, content-addressed CDK assets, release ledger. Deployed only by the authenticated bootstrap.",
            **kwargs,
        )
        _shared_tags(self)
        self.bucket = s3.Bucket(
            self,
            "Store",
            bucket_name=n.pipeline_store_bucket_name(Aws.ACCOUNT_ID),
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=cdk.RemovalPolicy.RETAIN,
            lifecycle_rules=[
                # CodePipeline stores artifacts under the first 20 characters of the pipeline name
                s3.LifecycleRule(id="pipeline-artifacts", prefix=n.PIPELINE_NAME[:20] + "/", expiration=Duration.days(STORE_PREFIX_EXPIRY_DAYS)),
                s3.LifecycleRule(id="build-cache", prefix=CACHE_PREFIX, expiration=Duration.days(14)),
                s3.LifecycleRule(id="assets", prefix=ASSET_PREFIX, expiration=Duration.days(STORE_PREFIX_EXPIRY_DAYS)),
                s3.LifecycleRule(id="bootstrap", prefix=BOOTSTRAP_PREFIX, expiration=Duration.days(STORE_PREFIX_EXPIRY_DAYS)),
                s3.LifecycleRule(id="noncurrent", noncurrent_version_expiration=Duration.days(STORE_NONCURRENT_EXPIRY_DAYS), abort_incomplete_multipart_upload_after=Duration.days(STORE_ABORT_MULTIPART_DAYS)),
            ],
        )
        tag_role(self.bucket, "pipeline-artifact-bucket")


class ToolingStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, *, shared: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(
            scope,
            construct_id,
            stack_name=TOOLING_STACK_NAME,
            env=cdk.Environment(region=str(shared["region"])),
            synthesizer=tooling_synthesizer(),
            termination_protection=True,
            description="FinanceModel account-level tooling (environment shared): CPU image repository and the FinanceModel pipeline. Deployed only by the authenticated bootstrap. No budget and no permission boundary (both FinancialPlanning-owned).",
            **kwargs,
        )
        self.shared = shared
        _shared_tags(self)
        #: Roles of this stack the platform's budget action must deny (published by the release step).
        self.enforced_roles: list[iam.Role] = []
        boundary = iam.ManagedPolicy.from_managed_policy_name(self, "SharedBoundary", contract_boundaries.boundary_name(contract_ssm.SHARED))
        iam.PermissionsBoundary.of(self).apply(boundary)
        self.repository = ecr.Repository(
            self,
            "CpuImages",
            repository_name=n.ecr_repository_name(),
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            image_scan_on_push=True,
            encryption=ecr.RepositoryEncryption.AES_256,
            removal_policy=cdk.RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(rule_priority=1, description="untagged layers of interrupted pushes", tag_status=ecr.TagStatus.UNTAGGED, max_image_age=Duration.days(1)),
                ecr.LifecycleRule(rule_priority=2, description="keep the newest release images (rollback window)", tag_status=ecr.TagStatus.TAGGED, tag_prefix_list=["rel_"], max_image_count=KEEP_RELEASE_IMAGES),
            ],
        )
        tag_role(self.repository, "financemodel-image-repository")

    def register_enforced_role(self, role: iam.Role) -> None:
        self.enforced_roles.append(role)

    @staticmethod
    def environment_boundary(scope: Construct, env: str) -> iam.IManagedPolicy:
        return iam.ManagedPolicy.from_managed_policy_name(scope, f"{env.capitalize()}Boundary", contract_boundaries.boundary_name(env))


def get_tooling_stack(app: cdk.App, shared: Mapping[str, Any]) -> ToolingStack:
    existing = app.node.try_find_child(TOOLING_CONSTRUCT_ID)
    if existing is not None:
        assert isinstance(existing, ToolingStack)
        return existing
    store = app.node.try_find_child(STORE_CONSTRUCT_ID) or StoreStack(app, STORE_CONSTRUCT_ID, shared=shared)
    tooling = ToolingStack(app, TOOLING_CONSTRUCT_ID, shared=shared)
    tooling.add_stack_dependency(store)  # the CLI stages the tooling template in the store
    return tooling


def add_to_app(app: cdk.App, shared: Mapping[str, Any], stages: Mapping[str, Any]) -> None:
    """``infra/app.py`` hook: the account-level store and tooling stacks."""
    get_tooling_stack(app, shared)
