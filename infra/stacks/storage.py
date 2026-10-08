"""Per-environment stateful resources: ``finplan-<env>-financemodel-storage`` (tasks 10.1, 10.3; WS-01, WS-02).

* **Research workspace storage** (matrix row ``research-workspace-storage``, logical role
  ``research-workspace-bucket``): ``finplan-<env>-financemodel-research-workspace-<account>``.
  SSE-S3 encryption at rest, Block Public Access, TLS only, bucket-owner-enforced ownership,
  versioning. Bucket policy: principals of other environments are denied everything, and object
  access is limited to FinanceModel principals of this environment (``finplan-<env>-financemodel-*``).
  Lifecycle (WS-02): only the ``scratch/`` prefix (unreferenced intermediate files, for example a
  cancelled run's temporary features) expires, after ``retention.scratch_days``. Run hand-off
  documents (``runs/``), content-addressed artifacts (``artifacts/``, the only thing results and
  registered model versions reference, by checksum) and the dataset catalog never expire.
  Noncurrent versions expire after ``retention.noncurrent_version_days`` and incomplete multipart
  uploads are aborted after ``retention.incomplete_multipart_days``. Its location is published only
  as ``/finplan/<env>/financemodel/config/research-storage-ref`` (by the release step) and never
  appears in a repository file.
* **Model registry** (row ``model-registry``, logical role ``model-registry``):
  ``finplan-<env>-financemodel-model-registry-<account>``, the same protections plus write-once
  records (every ``PutObject`` must carry ``If-None-Match``) and no deletes (REG-02), enforced by
  its bucket policy (matrix row ``model-registry`` lists ``AWS::S3::BucketPolicy`` since contracts
  1.0.0, D16) and again by the writers' identity policies.
* **Job control table** (row ``job-control-plane``, logical role ``job-control-table``):
  ``finplan-<env>-financemodel-job-control``, the control plane's single table
  (:data:`finplan_model.control.store.TABLE_SPEC`): on-demand, point-in-time recovery, TTL, no
  stream, and a resource policy that admits only FinanceModel principals of this environment.

Buckets and the table are retained when the stack is deleted (``docs/operations.md``, teardown);
the table has deletion protection in prod.
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
from aws_cdk import Aws, Duration
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3

from finplan_model.control.store import TABLE_SPEC

from . import naming as n
from .common import ModelStack, StageContext, stack_name, tag_role
from .policies import control_table_policy, registry_bucket_policy, research_bucket_policy

__all__ = ["SCRATCH_PREFIX", "StorageStack", "add_to_stage"]

SCRATCH_PREFIX = "scratch/"


class StorageStack(ModelStack):
    def __init__(self, scope: Any, construct_id: str, *, ctx: StageContext, **kwargs: Any) -> None:
        cfg = ctx.cfg
        super().__init__(scope, construct_id, cfg=cfg, description=f"FinanceModel {cfg.env} research storage, model registry and job control table", stack_name=stack_name(cfg.env, "storage"), **kwargs)
        env = cfg.env
        ret = cfg.retention
        self.research = self._bucket(
            "ResearchWorkspace",
            n.RESEARCH_BUCKET,
            "research-workspace-bucket",
            lifecycle=[
                s3.LifecycleRule(id="expire-unreferenced-scratch", prefix=SCRATCH_PREFIX, expiration=Duration.days(int(ret["scratch_days"]))),
                s3.LifecycleRule(
                    id="noncurrent-and-multipart",
                    noncurrent_version_expiration=Duration.days(int(ret["noncurrent_version_days"])),
                    abort_incomplete_multipart_upload_after=Duration.days(int(ret["incomplete_multipart_days"])),
                ),
            ],
        )
        for st in research_bucket_policy(env, self.research.bucket_arn, partition=Aws.PARTITION, account=Aws.ACCOUNT_ID):
            self.research.add_to_resource_policy(iam.PolicyStatement.from_json(st))

        self.registry = self._bucket(
            "ModelRegistry",
            n.REGISTRY_BUCKET,
            "model-registry",
            lifecycle=[s3.LifecycleRule(id="abort-multipart", abort_incomplete_multipart_upload_after=Duration.days(int(ret["incomplete_multipart_days"])))],
        )
        for st in registry_bucket_policy(env, self.registry.bucket_arn, partition=Aws.PARTITION, account=Aws.ACCOUNT_ID):
            self.registry.add_to_resource_policy(iam.PolicyStatement.from_json(st))

        table_name = n.env_name(env, n.CONTROL_TABLE)
        partition, region, account = Aws.PARTITION, Aws.REGION, Aws.ACCOUNT_ID
        table_arn = f"arn:{partition}:dynamodb:{region}:{account}:table/{table_name}"
        spec = TABLE_SPEC
        self.table = dynamodb.Table(
            self,
            "JobControlTable",
            table_name=table_name,
            partition_key=dynamodb.Attribute(name=spec["partition_key"]["name"], type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name=spec["sort_key"]["name"], type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=bool(spec["point_in_time_recovery"])),
            time_to_live_attribute=spec["ttl_attribute"],
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            deletion_protection=env == "prod",
            removal_policy=cdk.RemovalPolicy.RETAIN,
            resource_policy=iam.PolicyDocument.from_json(control_table_policy(env, table_arn, partition=Aws.PARTITION, account=Aws.ACCOUNT_ID)),
        )
        for gsi in spec["global_secondary_indexes"]:
            self.table.add_global_secondary_index(
                index_name=gsi["name"],
                partition_key=dynamodb.Attribute(name=gsi["partition_key"]["name"], type=dynamodb.AttributeType.STRING),
                sort_key=dynamodb.Attribute(name=gsi["sort_key"]["name"], type=dynamodb.AttributeType.STRING),
                projection_type=dynamodb.ProjectionType.ALL,
            )
        tag_role(self.table, "job-control-table")

        cdk.CfnOutput(self, "ResearchStorageRef", value=self.research.bucket_name, description="Research storage bucket (published to config/research-storage-ref by the release step)")
        cdk.CfnOutput(self, "RegistryStorageRef", value=self.registry.bucket_name, description="Model registry bucket (published to config/registry-storage-ref by the release step)")
        cdk.CfnOutput(self, "ControlTableName", value=self.table.table_name, description="Job control table")

    def _bucket(self, cid: str, logical: str, logical_role: str, *, lifecycle: list[s3.LifecycleRule]) -> s3.Bucket:
        bucket = s3.Bucket(
            self,
            cid,
            bucket_name=n.bucket_name(self.env_name, logical, Aws.ACCOUNT_ID),
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=cdk.RemovalPolicy.RETAIN,
            lifecycle_rules=lifecycle,
        )
        tag_role(bucket, logical_role)
        return bucket


def add_to_stage(stage: cdk.Stage, ctx: StageContext) -> None:
    ctx.stacks["storage"] = StorageStack(stage, "Storage", ctx=ctx)
