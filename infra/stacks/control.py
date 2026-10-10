"""Per-environment job control plane: ``finplan-<env>-financemodel-control`` (tasks 10.2, 10.3, 10.4).

Matrix rows ``job-control-plane``, ``job-interface`` and ``sagemaker-job-definitions`` (contracts D1).

* **Lambdas** (Python 3.12, ``arm64``, one shared code bundle, :mod:`infra.stacks.lambda_code`):
  ``job-api-handler`` (the job API), ``job-dispatcher`` (EventBridge Scheduler schedule deployed
  **DISABLED** and armed only while runs are queued, active or awaiting an approval deadline, design
  D1, :mod:`finplan_model.control.wakeup`; also kicked asynchronously by the API), ``job-state-handler`` (EventBridge rules on
  ``SageMaker Processing Job State Change`` and ``SageMaker Training Job State Change`` for job names
  ``fm-<env>-*``) and
  ``job-registry-lookup`` (the registry lineage route the platform calls) and ``strategy-selection``
  (``GET``/``PUT /v1/production-strategy``; its role is the only writer of
  ``/finplan/<env>/financemodel/config/production-strategy``, contracts 1.1.0). Each has its own role
  (:mod:`infra.stacks.policies`) and an explicit 30-day log group (``/aws/lambda/<function>``; it
  references its function, so the ownership check attributes it to the function's row). None of
  them runs strategy code (spec job-deployment-pipeline).
* **Job API** (matrix row ``job-interface``, logical role ``job-api``): a regional REST API
  defined by an OpenAPI body (``AWS_IAM`` auth on every route, Lambda proxy integrations, contract
  gateway responses, the resource policy of :func:`finplan_model.control.policies.job_api_resource_policy`),
  one deployment and the stage ``api`` (throttled). The human approver role
  ``finplan-<env>-financemodel-approver-role`` (identity policy: read jobs, approve, cancel; no
  ``finplan-*`` service role may assume it; row ``job-control-plane``).
* **Job-execution role** (row ``sagemaker-job-definitions``, logical role ``job-execution-role``,
  contracts 1.0.0 D16): ``finplan-<env>-financemodel-job-execution-role``,
  trusted by SageMaker only, bounded by ``finplan-<env>-research-permission-boundary``.

Every role carries the environment permission boundary (:class:`infra.stacks.common.ModelStack`),
except the job-execution role, which carries the research boundary (ENV-04).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import aws_cdk as cdk
from aws_cdk import Aws, Duration, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_scheduler as scheduler

from finplan_model.control import policies as control_policies
from finplan_model.control.api import ROUTES
from finplan_model.control.wakeup import TICK

from . import naming as n
from .common import ModelStack, StageContext, research_boundary, stack_name, tag_role
from .lambda_code import function_code
from .policies import classical_role_policy, control_role_policy, inference_role_policy, job_api_invoker_patterns, job_execution_policy, registry_lookup_policy, schedule_role_policy, strategy_selection_policy

__all__ = ["API_STAGE", "DISPATCH_SCHEDULE_DESCRIPTION", "LAMBDA_ARCHITECTURE", "LOG_RETENTION_DAYS", "OPENAPI_ROUTES", "ControlStack", "add_to_stage", "openapi_body"]

API_STAGE = "api"
LOG_RETENTION_DAYS = 30
LAMBDA_ARCHITECTURE = lambda_.Architecture.ARM_64
#: (method, OpenAPI path, function logical name). The job routes mirror :data:`finplan_model.control.api.ROUTES`.
OPENAPI_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/v1/recommendations", n.JOB_API_HANDLER),
    ("GET", "/v1/performance-evidence", n.JOB_API_HANDLER),
    ("PUT", "/v1/advisory-policy", n.JOB_API_HANDLER),
    ("POST", "/v1/jobs", n.JOB_API_HANDLER),
    ("GET", "/v1/jobs", n.JOB_API_HANDLER),
    ("GET", "/v1/jobs/{run_id}", n.JOB_API_HANDLER),
    ("GET", "/v1/jobs/{run_id}/result", n.JOB_API_HANDLER),
    ("POST", "/v1/jobs/{run_id}/cancel", n.JOB_API_HANDLER),
    ("POST", "/v1/jobs/{run_id}/approve", n.JOB_API_HANDLER),
    ("GET", "/v1/production-strategy", n.STRATEGY_SELECTION),
    ("PUT", "/v1/production-strategy", n.STRATEGY_SELECTION),
    ("GET", "/v1/registry/model-versions/{model_version}", n.REGISTRY_LOOKUP),
    ("GET", "/v1/registry/lineage/{run_id}", n.REGISTRY_LOOKUP),
)
#: API stage throttling (a cost guard: the API is never a bulk interface).
DISPATCH_SCHEDULE_DESCRIPTION = "Dispatcher tick (armed only while runs are queued, active or awaiting an approval deadline): approval expiry, reconcile, stale-lease reclaim, start queued runs"
THROTTLE_RATE = 5
THROTTLE_BURST = 10
_GATEWAY_TYPES = {"MISSING_AUTHENTICATION_TOKEN": "MISSING_AUTHENTICATION_TOKEN", "INVALID_SIGNATURE": "INVALID_SIGNATURE", "EXPIRED_TOKEN": "EXPIRED_TOKEN", "ACCESS_DENIED": "ACCESS_DENIED", "THROTTLED": "THROTTLED"}


def _metadata_role(resource: cdk.CfnResource, logical_role: str) -> None:
    """Ownership attribution for types that cannot carry tags (the check reads ``Metadata``)."""
    resource.add_metadata("logical-role", logical_role)


def openapi_body(env: str, invoke_arns: dict[str, str], contract_version: str) -> dict[str, Any]:
    """OpenAPI 3.0 body of the job API (IAM auth, Lambda proxy integrations, contract gateway responses)."""
    paths: dict[str, dict[str, Any]] = {}
    for method, path, logical in OPENAPI_ROUTES:
        params = [{"name": p, "in": "path", "required": True, "schema": {"type": "string"}} for p in ("run_id", "model_version") if "{" + p + "}" in path]
        op: dict[str, Any] = {
            "operationId": f"{method.lower()}{''.join(s.strip('{}').title().replace('_', '') for s in path.split('/') if s)}",
            "security": [{"sigv4": []}],
            "responses": {"200": {"description": "contract document"}},
            "x-amazon-apigateway-integration": {"type": "aws_proxy", "httpMethod": "POST", "uri": invoke_arns[logical], "passthroughBehavior": "when_no_match"},
        }
        if params:
            op["parameters"] = params
        paths.setdefault(path, {})[method.lower()] = op
    responses = {}
    for gtype, spec in control_policies.gateway_responses(contract_version).items():
        responses[_GATEWAY_TYPES[gtype]] = {"statusCode": spec["status_code"], "responseTemplates": {"application/json": spec["template"]}}
    return {
        "openapi": "3.0.1",
        "info": {"title": n.env_name(env, n.JOB_API), "version": "1.0", "description": "FinanceModel job interface (submit_job, get_job_status, get_job_result, cancel_job, list_jobs, approve_run), the production-strategy selection and the model registry lineage lookup"},
        "paths": paths,
        "components": {"securitySchemes": {"sigv4": {"type": "apiKey", "name": "Authorization", "in": "header", "x-amazon-apigateway-authtype": "awsSigv4"}}},
        "x-amazon-apigateway-gateway-responses": responses,
    }


class ControlStack(ModelStack):
    def __init__(self, scope: Any, construct_id: str, *, ctx: StageContext, **kwargs: Any) -> None:
        cfg = ctx.cfg
        super().__init__(scope, construct_id, cfg=cfg, description=f"FinanceModel {cfg.env} job control plane: job API, dispatcher, state-change handler, registry lookup and roles", stack_name=stack_name(cfg.env, "control"), **kwargs)
        env = cfg.env
        self.roles: dict[str, iam.Role] = {}
        self.functions: dict[str, lambda_.Function] = {}
        p, r, a = Aws.PARTITION, Aws.REGION, Aws.ACCOUNT_ID
        code = function_code()

        # ---------------------------------------------------------------- job-execution role
        job_role = iam.Role(
            self,
            "JobExecutionRole",
            role_name=n.role_name(env, n.JOB_EXECUTION),
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com", conditions={"StringEquals": {"aws:SourceAccount": a}}),
            description=f"FinanceModel {env} SageMaker job-execution role: approved snapshots read-only, staging write-once, research storage",
            inline_policies={"job-execution": iam.PolicyDocument.from_json(job_execution_policy(env, partition=p, region=r, account=a))},
            max_session_duration=Duration.hours(1),
        )
        tag_role(job_role, n.JOB_EXECUTION_LOGICAL_ROLE)
        iam.PermissionsBoundary.of(job_role).apply(research_boundary(job_role, "ResearchBoundary", env))
        self.roles["job-execution"] = job_role
        cdk.CfnOutput(self, "JobRoleRef", value=job_role.role_arn, description="Published at /finplan/<env>/financemodel/job/job-role-ref")

        # ---------------------------------------------------------------- control-plane functions
        env_vars = {"FINPLAN_ENVIRONMENT": env, "FINPLAN_RUNS_TABLE": n.env_name(env, n.CONTROL_TABLE), "FINPLAN_CONFIG_DIR": "/var/task/config", "FINPLAN_DISPATCH_SCHEDULE": n.env_name(env, n.DISPATCHER)}
        api_fn = self._function("JobApiHandler", n.JOB_API_HANDLER, "job-api-handler", control_role_policy(env, "api", partition=p, region=r, account=a), code, {**env_vars, "FINPLAN_DISPATCHER_FUNCTION": n.function_name(env, n.DISPATCHER)}, timeout=29, memory=512)
        dispatcher = self._function("JobDispatcher", n.DISPATCHER, "job-dispatcher-schedule", control_role_policy(env, "dispatcher", partition=p, region=r, account=a), code, env_vars, timeout=60, memory=512)
        state = self._function("JobStateHandler", n.STATE_HANDLER, "job-state-change-rule", control_role_policy(env, "state", partition=p, region=r, account=a), code, env_vars, timeout=60, memory=512)
        cdk.CfnOutput(self, "JobApiRoleRef", value=self.roles[n.JOB_API_HANDLER].role_arn, description="Published at /finplan/<env>/financemodel/job/job-api-role-ref")

        # SageMaker state changes -> state handler
        rule = events.Rule(
            self,
            "JobStateChangeRule",
            rule_name=n.env_name(env, n.STATE_CHANGE_RULE),
            description="SageMaker Processing job state changes of this environment's FinanceModel jobs",
            event_pattern=events.EventPattern(source=["aws.sagemaker"], detail_type=["SageMaker Processing Job State Change"], detail={"ProcessingJobName": [{"prefix": n.processing_job_prefix(env)}]}),
            targets=[targets.LambdaFunction(state, retry_attempts=4, max_event_age=Duration.hours(2))],
        )
        tag_role(rule, "job-state-change-rule")
        # SageMaker Training job state changes (model_selection runs as a Training job) -> same handler
        training_rule = events.Rule(
            self,
            "TrainingJobStateChangeRule",
            rule_name=n.env_name(env, n.TRAINING_STATE_CHANGE_RULE),
            description="SageMaker Training job state changes of this environment's FinanceModel jobs",
            event_pattern=events.EventPattern(source=["aws.sagemaker"], detail_type=["SageMaker Training Job State Change"], detail={"TrainingJobName": [{"prefix": n.processing_job_prefix(env)}]}),
            targets=[targets.LambdaFunction(state, retry_attempts=4, max_event_age=Duration.hours(2))],
        )
        tag_role(training_rule, "job-state-change-rule")

        # dispatcher schedule (EventBridge Scheduler)
        sched_role = iam.Role(
            self,
            "DispatcherScheduleRole",
            role_name=n.role_name(env, n.SCHEDULE_ROLE),
            assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com", conditions={"StringEquals": {"aws:SourceAccount": a}}),
            description="EventBridge Scheduler: invoke the job dispatcher only",
            inline_policies={"invoke-dispatcher": iam.PolicyDocument.from_json(schedule_role_policy(env, partition=p, region=r, account=a))},
        )
        tag_role(sched_role, "job-dispatcher-schedule")
        schedule = scheduler.CfnSchedule(
            self,
            "DispatcherSchedule",
            name=n.env_name(env, n.DISPATCHER),
            description=DISPATCH_SCHEDULE_DESCRIPTION,
            schedule_expression=TICK,
            schedule_expression_timezone="UTC",
            flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(mode="OFF"),
            # Deployed disarmed: the control plane arms it only while runs need the dispatcher (design
            # D1) and leaves it in exactly this state when idle, so an idle environment never drifts.
            state="DISABLED",
            target=scheduler.CfnSchedule.TargetProperty(arn=dispatcher.function_arn, role_arn=sched_role.role_arn, input="{}", retry_policy=scheduler.CfnSchedule.RetryPolicyProperty(maximum_retry_attempts=0, maximum_event_age_in_seconds=60)),
        )
        _metadata_role(schedule, "job-dispatcher-schedule")

        # ---------------------------------------------------------------- job API
        lookup = self._function("RegistryLookup", n.REGISTRY_LOOKUP, "job-api-handler", registry_lookup_policy(env, partition=p, region=r, account=a), code, {"FINPLAN_ENVIRONMENT": env, "FINPLAN_REGISTRY_BUCKET": n.bucket_name(env, n.REGISTRY_BUCKET, a), "FINPLAN_CONFIG_DIR": "/var/task/config"}, timeout=10, memory=256)
        # strategy selection (contracts 1.1.0): the only writer of config/production-strategy (M3)
        selection = self._function("StrategySelection", n.STRATEGY_SELECTION, "job-api-handler", strategy_selection_policy(env, partition=p, region=r, account=a), code, {"FINPLAN_ENVIRONMENT": env, "FINPLAN_RUNS_TABLE": n.env_name(env, n.CONTROL_TABLE), "FINPLAN_CONFIG_DIR": "/var/task/config"}, timeout=29, memory=512)
        self._api(env, {n.JOB_API_HANDLER: api_fn, n.REGISTRY_LOOKUP: lookup, n.STRATEGY_SELECTION: selection})
        if env == "beta":
            inference = self._function("StrategyInference", n.STRATEGY_INFERENCE, "job-api-handler", inference_role_policy(env, partition=p, region=r, account=a), code,
                                       {"FINPLAN_ENVIRONMENT": env, "FINPLAN_CONFIG_DIR": "/var/task/config"}, timeout=270, memory=512)
            cdk.CfnOutput(self, "StrategyFunctionRef", value=inference.function_arn, description="Published at api/strategy-function-ref for direct MCP adapter invocation")
            classical = self._function("ClassicalInference", n.CLASSICAL_INFERENCE, "job-api-handler", classical_role_policy(env, partition=p, region=r, account=a), code,
                                       {"FINPLAN_ENVIRONMENT": env, "FINPLAN_CONFIG_DIR": "/var/task/config"}, timeout=270, memory=1024)
            cdk.CfnOutput(self, "ClassicalFunctionRef", value=classical.function_arn, description="Published at api/classical-function-ref; independent traditional-optimization MCP backend")
            controller = self._function("ResearchController", n.RESEARCH_CONTROLLER, "job-api-handler", classical_role_policy(env, n.RESEARCH_CONTROLLER, partition=p, region=r, account=a), code,
                                        {"FINPLAN_ENVIRONMENT": env, "FINPLAN_CONFIG_DIR": "/var/task/config"}, timeout=270, memory=1024)
            controller.node.default_child.add_property_override("ReservedConcurrentExecutions", 1)
            weekly_role = iam.Role(self, "WeeklyResearchScheduleRole", role_name=n.role_name(env, n.RESEARCH_SCHEDULE),
                                   assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com", conditions={"StringEquals": {"aws:SourceAccount": a}}),
                                   inline_policies={"invoke-weekly-review": iam.PolicyDocument(statements=[iam.PolicyStatement(actions=["lambda:InvokeFunction"], resources=[controller.function_arn])])})
            tag_role(weekly_role, "job-dispatcher-schedule")
            weekly = scheduler.CfnSchedule(self, "WeeklyResearchSchedule", name=n.env_name(env, n.RESEARCH_SCHEDULE),
                    description="One bounded classical research review per week; max one USD 0.50 sandbox job, no strategy activation",
                    schedule_expression="cron(0 9 ? * MON *)", schedule_expression_timezone="America/New_York", state="ENABLED",
                    flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(mode="OFF"),
                    target=scheduler.CfnSchedule.TargetProperty(arn=controller.function_arn, role_arn=weekly_role.role_arn, input='{"trigger":"weekly_classical_research"}',
                           retry_policy=scheduler.CfnSchedule.RetryPolicyProperty(maximum_retry_attempts=1, maximum_event_age_in_seconds=3600)))
            _metadata_role(weekly, "job-dispatcher-schedule")

    # ------------------------------------------------------------------ helpers
    def _function(self, cid: str, logical: str, logical_role: str, policy: dict[str, Any], code: lambda_.Code, environment: dict[str, str], *, timeout: int, memory: int) -> lambda_.Function:
        env = self.env_name
        role = iam.Role(
            self,
            f"{cid}Role",
            role_name=n.role_name(env, logical),
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description=f"FinanceModel {env} {logical} Lambda role",
            inline_policies={logical: iam.PolicyDocument.from_json(policy)},
        )
        tag_role(role, logical_role)
        fn = lambda_.Function(
            self,
            cid,
            function_name=n.function_name(env, logical),
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=LAMBDA_ARCHITECTURE,
            handler=n.FUNCTIONS[logical],
            code=code,
            role=role,
            memory_size=memory,
            timeout=Duration.seconds(timeout),
            environment=environment,
            description=(f"FinanceModel {env} frozen strategy inference (read only; never trains)" if logical == n.STRATEGY_INFERENCE else f"FinanceModel {env} {logical} (validates, records and starts jobs only; never runs strategy code)"),
        )
        tag_role(fn, logical_role)
        # Explicit log group named after the function (Ref): retention bounded, deleted with the stack,
        # attributed to the function's matrix row as a CDK helper.
        group = logs.CfnLogGroup(self, f"{cid}LogGroup", log_group_name=f"/aws/lambda/{fn.function_name}", retention_in_days=LOG_RETENTION_DAYS)
        group.apply_removal_policy(cdk.RemovalPolicy.DESTROY)
        self.roles[logical] = role
        self.functions[logical] = fn
        return fn

    def _api(self, env: str, fns: dict[str, lambda_.Function]) -> None:
        from finplan_contracts import __version__ as contract_version

        p, r, a = Aws.PARTITION, Aws.REGION, Aws.ACCOUNT_ID
        # API Gateway's Lambda integration URI (its "account" field is the literal service name ``lambda``)
        invoke = {k: ":".join(["arn", p, "apigateway", r, "lambda", f"path/2015-03-31/functions/{f.function_arn}/invocations"]) for k, f in fns.items()}
        body = openapi_body(env, invoke, str(contract_version))
        policy = control_policies.job_api_resource_policy(env, invoker_role_patterns=job_api_invoker_patterns(env), approver_role=n.role_name(env, n.APPROVER), partition=p, account=a)
        api = apigw.CfnRestApi(
            self,
            "JobApi",
            name=n.env_name(env, n.JOB_API),
            description="FinanceModel job interface (IAM/SigV4)",
            body=body,
            policy=policy,
            endpoint_configuration=apigw.CfnRestApi.EndpointConfigurationProperty(types=["REGIONAL"]),
            fail_on_warnings=True,
        )
        tag_role(api, "job-api")
        # The deployment's logical ID follows the body, so every API change creates a new deployment.
        digest = hashlib.sha256(json.dumps(Stack.of(self).resolve(body), sort_keys=True).encode()).hexdigest()[:10]
        deployment = apigw.CfnDeployment(self, f"JobApiDeployment{digest}", rest_api_id=api.ref, description="job API deployment")
        stage = apigw.CfnStage(
            self,
            "JobApiStage",
            rest_api_id=api.ref,
            deployment_id=deployment.ref,
            stage_name=API_STAGE,
            method_settings=[apigw.CfnStage.MethodSettingProperty(http_method="*", resource_path="/*", throttling_rate_limit=THROTTLE_RATE, throttling_burst_limit=THROTTLE_BURST, metrics_enabled=False)],
        )
        tag_role(stage, "job-api")
        source = f"arn:{p}:execute-api:{r}:{a}:{api.ref}/*/*/*"
        for logical, fn in fns.items():
            fn.add_permission(f"Invoke{logical.title().replace('-', '')}FromApi", principal=iam.ServicePrincipal("apigateway.amazonaws.com"), source_arn=source)
        endpoint = f"https://{api.ref}.execute-api.{r}.{Aws.URL_SUFFIX}/{API_STAGE}"
        api_resource = f"arn:{p}:execute-api:{r}:{a}:{api.ref}/{API_STAGE}"
        approver = iam.Role(
            self,
            "ApproverRole",
            role_name=n.role_name(env, n.APPROVER),
            assumed_by=iam.AccountRootPrincipal().with_conditions({"ArnNotLike": {"aws:PrincipalArn": f"arn:{p}:iam::{a}:role/finplan-*"}}),
            description=f"FinanceModel {env} human approver of paid runs (approve_run); never assumable by finplan service roles",
            inline_policies={"approve-runs": iam.PolicyDocument.from_json(control_policies.approver_identity_policy(api_resource))},
            max_session_duration=Duration.hours(1),
        )
        tag_role(approver, "job-approver-role")
        self.roles[n.APPROVER] = approver
        cdk.CfnOutput(self, "JobEndpoint", value=endpoint, description="Published at /finplan/<env>/financemodel/api/job-endpoint")
        cdk.CfnOutput(self, "RegistryRef", value=f"{endpoint}/v1/registry", description="Published at /finplan/<env>/financemodel/model/registry-ref (lineage lookups through the job API)")
        cdk.CfnOutput(self, "ApproverRoleRef", value=approver.role_arn, description="Published at /finplan/<env>/financemodel/config/approver-role-ref")


def add_to_stage(stage: cdk.Stage, ctx: StageContext) -> None:
    control = ControlStack(stage, "Control", ctx=ctx)
    storage = ctx.stacks.get("storage")
    if storage is not None:
        control.add_stack_dependency(storage)  # deploy order only: names are deterministic, no cross-stack exports
    ctx.stacks["control"] = control
    _ = ROUTES  # the OpenAPI routes mirror the handler's routing table (checked by a unit test)
