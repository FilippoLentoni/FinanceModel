"""The FinanceModel pipeline (task 10.5; spec job-deployment-pipeline; contracts D6 pipeline standard).

Added to the account-level tooling stack (:mod:`infra.stacks.tooling`), so only the authenticated
bootstrap creates or changes it; the pipeline never updates itself. CodePipeline **V2** with
CodeBuild, stages in the contract order, reusing the deployed FinancialPlanning pattern:

1. **Source**: CodeConnections source of ``FilippoLentoni/FinanceModel`` on ``main``. The
   connection is the existing, reused GitHub CodeConnection, referenced through
   ``/finplan/shared/financemodel/config/codeconnection-ref`` (written by the bootstrap, resolved by
   CloudFormation; never an ARN in a file). ``#{SourceVariables.CommitId}`` goes to the build.
2. **Build**: ``scripts/build_stage.py``: pre-synth gates (contract pin, configuration, leak scan,
   copied-id, consumer conformance, live-permission scan, provider guard, fixture check, unit and
   contract tests under the SageMaker-blocking harness), the Lambda bundle, ``cdk synth`` **once**
   (release mode, :func:`infra.stacks.tooling.deployment_synthesizer`), post-synth gates (ownership
   with zero problems, boundaries, live permissions, pipeline structure, scoped deploy roles), the
   ``financemodel-cpu`` image built **once** and pushed by digest, a new ``release_id`` and the
   artifact digest (assembly plus image digest). Its single output ``BuildOutput`` is the only input
   of every later stage. ``rollback_to_release_id`` re-emits a recorded release instead.
3. **Beta**: deploy ``storage`` then ``control`` (scoped deploy role, CloudFormation execution role),
   ``PublishRelease`` (references, job definitions pinned to the image digest, registry seeding, the
   release manifest, budget-enforced role names), ``IntegrationBetaTests``.
4. **Gamma**: ``PromotionCheck`` (refuses a 0.x contract pin: 0.x is beta-only), the same deploys,
   ``PublishRelease``, ``GammaTests`` (integration plus isolation).
5. **Approval**: one Manual approval action.
6. **Prod**: ``PromotionCheck``, deploys, ``PublishRelease`` (with approver and time),
   ``SmokeTests`` (``list_jobs`` and a dry-run submission only; no paid job).

Experiments are never pipeline executions: they are submitted at run time through the job API
(DEP-01). Until the bootstrap's source-stage dry run has passed the inbound transition into Build is
disabled (``SourceDryRunPassed=false``).

Roles: account-level (``environment=shared``, ``finplan-shared-permission-boundary``)
``pipeline-role`` and ``pipeline-build-project-role``; per environment (``environment=<env>``,
``finplan-<env>-permission-boundary``; matrix row ``pipeline-environment-roles-financemodel``)
``deploy-role-<env>``, ``deploy-role-<env>-exec`` and ``finplan-<env>-financemodel-pipeline-stage-role``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import aws_cdk as cdk
from aws_cdk import Annotations, Aws, CfnCondition, CfnParameter, Duration, Fn, Tags
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as actions
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_ssm as ssm
from finplan_contracts import ssm as contract_ssm

from . import naming as n
from .common import tag_role
from .policies import build_role_statements, deploy_execution_statements, stage_role_statements
from .tooling import ToolingStack, add_log_group, get_tooling_stack

__all__ = [
    "BUILD_STAGE",
    "CONNECTION_PARAMETER",
    "DEFAULT_GITHUB_REPOSITORY",
    "ENV_SUITES",
    "NO_ROLLBACK",
    "ROLLBACK_VARIABLE",
    "SOURCE_STAGE",
    "STAGE_NAMES",
    "UV_VERSION",
    "PipelineResources",
    "add_pipeline",
    "add_to_app",
    "build_spec",
    "ordered_stacks",
    "stage_spec",
    "template_path",
]

ROLLBACK_VARIABLE = "rollback_to_release_id"
NO_ROLLBACK = "none"
SOURCE_STAGE = "Source"
BUILD_STAGE = "Build"
STAGE_NAMES = (SOURCE_STAGE, BUILD_STAGE, "Beta", "Gamma", "Approval", "Prod")
ENV_SUITES = {"beta": "integration-beta", "gamma": "gamma", "prod": "smoke"}
CONNECTION_PARAMETER = contract_ssm.build(contract_ssm.SHARED, n.REPO, "config", "codeconnection-ref")
DEFAULT_GITHUB_REPOSITORY = "FilippoLentoni/FinanceModel"
UV_VERSION = "0.12.23"


def _stmts(docs: list[dict[str, Any]]) -> list[iam.PolicyStatement]:
    return [iam.PolicyStatement.from_json(d) for d in docs]


# ===================================================================== build specs
def _install() -> dict[str, Any]:
    return {"runtime-versions": {"python": "3.12", "nodejs": "22"}, "commands": [f'python3 -m pip install --quiet "uv=={UV_VERSION}"', "uv --version"]}


def build_spec() -> dict[str, Any]:
    return {
        "version": "0.2",
        "env": {"shell": "bash", "variables": {"SOURCE_DATE_EPOCH": "315532800", "UV_LINK_MODE": "copy", "CDK_DISABLE_VERSION_CHECK": "1", "FINPLAN_RELEASE_BUILD": "1"}},
        "phases": {
            "install": _install(),
            "build": {
                "commands": [
                    "uv sync --locked",
                    'uv run python scripts/build_stage.py --out build-output --source-commit "$SOURCE_COMMIT" --rollback-to "$ROLLBACK_TO_RELEASE_ID" --store "$FINPLAN_PIPELINE_STORE" --image-repository "$FINPLAN_IMAGE_REPOSITORY"',
                ]
            },
        },
        "artifacts": {"base-directory": "build-output", "files": ["**/*"]},
        "cache": {"paths": ["/root/.cache/uv/**/*", "/root/.npm/**/*"]},
    }


def stage_spec() -> dict[str, Any]:
    """Post-deploy actions run from BuildOutput only (never the source checkout, never a synth)."""
    return {
        "version": "0.2",
        "env": {"shell": "bash", "variables": {"UV_LINK_MODE": "copy"}},
        "phases": {
            "install": _install(),
            "build": {
                "commands": [
                    "uv sync --locked",
                    'uv run python scripts/stage_runner.py "$FINPLAN_STAGE_ACTION" --env "$FINPLAN_ENV" --release-info release-info.json --pipeline-execution-id "$PIPELINE_EXECUTION_ID" --store "$FINPLAN_PIPELINE_STORE"',
                ]
            },
        },
        "cache": {"paths": ["/root/.cache/uv/**/*"]},
    }


# ===================================================================== stage stacks
def ordered_stacks(stage: cdk.Stage) -> list[cdk.Stack]:
    """The stage's stacks in dependency order (deploy order)."""
    stacks = [c for c in stage.node.children if isinstance(c, cdk.Stack)]
    ordered: list[cdk.Stack] = []
    pending = list(stacks)
    while pending:
        progressed = False
        for st in list(pending):
            if all(d in ordered for d in st.dependencies if d in stacks):
                ordered.append(st)
                pending.remove(st)
                progressed = True
        if not progressed:  # pragma: no cover - CDK rejects cycles earlier
            raise ValueError("cyclic stack dependencies in stage " + stage.node.id)
    return ordered


def template_path(stage: cdk.Stage, stack: cdk.Stack) -> str:
    """Path of the stack template inside BuildOutput (``cdk.out/assembly-<Stage>/<file>``)."""
    return f"cdk.out/{stage.artifact_id}/{stack.template_file}"


# ===================================================================== construction
class PipelineResources:
    def __init__(self, tooling: ToolingStack) -> None:
        self.tooling = tooling
        self.pipeline: codepipeline.Pipeline | None = None
        self.store: s3.IBucket | None = None
        self.roles: dict[str, iam.Role] = {}
        self.projects: dict[str, codebuild.PipelineProject] = {}
        self.log_groups: dict[str, logs.LogGroup] = {}


def _role(scope: ToolingStack, cid: str, name: str, logical: str, principal: iam.IPrincipal, description: str, statements: list[iam.PolicyStatement] | None = None) -> iam.Role:
    role = iam.Role(scope, cid, role_name=name, assumed_by=principal, description=description)
    for st in statements or []:
        role.add_to_principal_policy(st)
    tag_role(role, logical)
    scope.register_enforced_role(role)
    return role


def _scope_to_environment(scope: ToolingStack, role: iam.Role, env: str) -> None:
    """Per-environment pipeline role: ``environment=<env>`` tag and that environment's boundary (ENV-21)."""
    Tags.of(role).add("environment", env, priority=200)
    iam.PermissionsBoundary.of(role).apply(ToolingStack.environment_boundary(role, env))


def _logging(res: PipelineResources, cid: str, project_name: str) -> codebuild.LoggingOptions:
    """CloudWatch logging of a CodeBuild project into its explicit ``/aws/codebuild/<project>`` group.

    Mirrors the FinancialPlanning ``_project_logging``: :func:`infra.stacks.tooling.add_log_group`
    (30-day retention, deleted with the stack, ``logical-role`` ``pipeline-build-project``, matrix
    row ``pipeline-financemodel``, which lists ``AWS::Logs::LogGroup`` since contracts 1.0.0, D16).
    The name is CodeBuild's default, so the build roles' log grants see the same ARN.
    """
    group = add_log_group(res.tooling, cid, f"/aws/codebuild/{project_name}", "pipeline-build-project")
    res.log_groups[project_name] = group
    return codebuild.LoggingOptions(cloud_watch=codebuild.CloudWatchLoggingOptions(log_group=group))


def add_pipeline(tooling: ToolingStack, stages: Mapping[str, Any], shared: Mapping[str, Any]) -> PipelineResources:
    res = PipelineResources(tooling)
    st = tooling
    source_cfg = shared.get("source") or {}
    github = str(source_cfg.get("repository") or DEFAULT_GITHUB_REPOSITORY)
    branch = str(source_cfg.get("branch") or "main")
    owner, repo_name = github.split("/", 1)
    p, r, a = Aws.PARTITION, Aws.REGION, Aws.ACCOUNT_ID

    dry_run_passed = CfnParameter(st, "SourceDryRunPassed", type="String", default="false", allowed_values=["false", "true"], description="true once the bootstrap's source-stage dry run fetched main; until then the transition into Build is disabled.")
    dry_run_cond = CfnCondition(st, "SourceDryRunPassedCondition", expression=Fn.condition_equals(dry_run_passed.value_as_string, "true"))

    store = s3.Bucket.from_bucket_name(st, "Store", n.pipeline_store_bucket_name(a))
    res.store = store

    pipeline_role = _role(st, "PipelineRole", n.shared_name("pipeline", "role"), "pipeline-role", iam.ServicePrincipal("codepipeline.amazonaws.com"), "CodePipeline service role of the FinanceModel pipeline")
    build_role = _role(st, "BuildRole", n.shared_name("pipeline-build-project", "role"), "pipeline-role", iam.ServicePrincipal("codebuild.amazonaws.com"), "Build stage: gates, synth, image build and push, asset publishing, release packaging", _stmts(build_role_statements(store.bucket_arn, partition=p, region=r, account=a)))
    res.roles.update(pipeline=pipeline_role, build=build_role)

    env_common = {"FINPLAN_PIPELINE_STORE": codebuild.BuildEnvironmentVariable(value=store.bucket_name)}
    build_env = codebuild.BuildEnvironment(build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2023_5, compute_type=codebuild.ComputeType.SMALL, privileged=True)
    stage_env = codebuild.BuildEnvironment(build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2023_5, compute_type=codebuild.ComputeType.SMALL, privileged=False)
    build_project_name = n.shared_name("pipeline-build-project")
    build_project = codebuild.PipelineProject(
        st,
        "BuildProject",
        project_name=build_project_name,
        role=build_role,
        environment=build_env,
        environment_variables={**env_common, "FINPLAN_IMAGE_REPOSITORY": codebuild.BuildEnvironmentVariable(value=tooling.repository.repository_name)},
        build_spec=codebuild.BuildSpec.from_object(build_spec()),
        timeout=Duration.minutes(45),
        cache=codebuild.Cache.local(codebuild.LocalCacheMode.CUSTOM, codebuild.LocalCacheMode.DOCKER_LAYER),
        logging=_logging(res, "BuildProjectLogGroup", build_project_name),
        description="Build stage: gates, cdk synth, financemodel-cpu image (built once, by digest), assets, release_id",
    )
    tag_role(build_project, "pipeline-build-project")
    res.projects["build"] = build_project

    source_output = codepipeline.Artifact("SourceOutput")
    build_output = codepipeline.Artifact("BuildOutput")
    connection_arn = ssm.StringParameter.value_for_string_parameter(st, CONNECTION_PARAMETER)
    source = actions.CodeStarConnectionsSourceAction(action_name="Source", owner=owner, repo=repo_name, branch=branch, connection_arn=connection_arn, output=source_output, trigger_on_push=True, variables_namespace="SourceVariables")
    build = actions.CodeBuildAction(
        action_name="BuildAndTest",
        project=build_project,
        input=source_output,
        outputs=[build_output],
        type=actions.CodeBuildActionType.BUILD,
        environment_variables={
            "SOURCE_COMMIT": codebuild.BuildEnvironmentVariable(value=source.variables.commit_id),
            "ROLLBACK_TO_RELEASE_ID": codebuild.BuildEnvironmentVariable(value=f"#{{variables.{ROLLBACK_VARIABLE}}}"),
        },
    )
    pipeline = codepipeline.Pipeline(
        st,
        "Pipeline",
        pipeline_name=n.PIPELINE_NAME,
        pipeline_type=codepipeline.PipelineType.V2,
        artifact_bucket=store,
        role=pipeline_role,
        cross_account_keys=False,
        restart_execution_on_update=False,
        use_pipeline_role_for_actions=True,
        variables=[codepipeline.Variable(variable_name=ROLLBACK_VARIABLE, default_value=NO_ROLLBACK, description="Set to a recorded release_id to redeploy its stored artifacts without rebuilding (contracts D6).")],
        stages=[codepipeline.StageProps(stage_name=SOURCE_STAGE, actions=[source]), codepipeline.StageProps(stage_name=BUILD_STAGE, actions=[build])],
    )
    tag_role(pipeline, "pipeline")
    res.pipeline = pipeline

    for env in n.ENVIRONMENTS:
        if env == "prod":
            pipeline.add_stage(stage_name="Approval", actions=[actions.ManualApprovalAction(action_name="ApproveProd", additional_information="Approve promotion of this FinanceModel release to prod after the gamma tests passed. The approver and time are recorded in the prod release manifest.")])
        _add_env_stage(res, env, stages[env], build_output, env_common, stage_env)

    cfn = pipeline.node.default_child
    assert isinstance(cfn, codepipeline.CfnPipeline)
    cfn.add_property_override("DisableInboundStageTransitions", Fn.condition_if(dry_run_cond.logical_id, Aws.NO_VALUE, [{"StageName": BUILD_STAGE, "Reason": "finplan bootstrap: the source-stage dry run has not passed yet"}]))
    cdk.CfnOutput(st, "PipelineName", value=n.PIPELINE_NAME)
    cdk.CfnOutput(st, "ImageRepositoryName", value=tooling.repository.repository_name)
    return res


def _add_env_stage(res: PipelineResources, env: str, ctx: Any, build_output: codepipeline.Artifact, env_common: dict[str, codebuild.BuildEnvironmentVariable], stage_env: codebuild.BuildEnvironment) -> None:
    st = res.tooling
    cap = env.capitalize()
    assert res.store is not None and res.pipeline is not None
    p, r, a = Aws.PARTITION, Aws.REGION, Aws.ACCOUNT_ID
    deploy_role = _role(st, f"DeployRole{cap}", n.deploy_role_name(env), "deploy-role", iam.ArnPrincipal(res.roles["pipeline"].role_arn), f"Scoped {env} deploy action role (CodePipeline assumes it)")
    exec_role = _role(st, f"DeployExecRole{cap}", n.exec_role_name(env), "deploy-role", iam.ServicePrincipal("cloudformation.amazonaws.com"), f"CloudFormation execution role for the {env} FinanceModel stacks", _stmts(deploy_execution_statements(env, res.store.bucket_arn, partition=p, region=r, account=a)))
    stage_role = _role(st, f"StageRole{cap}", n.stage_role_name(env), "pipeline-role", iam.ServicePrincipal("codebuild.amazonaws.com"), f"{env} release publisher, registry seeder and test runner", _stmts(stage_role_statements(env, res.store.bucket_arn, partition=p, region=r, account=a)))
    for role in (deploy_role, exec_role, stage_role):
        _scope_to_environment(st, role, env)
    res.roles.update({f"deploy-{env}": deploy_role, f"exec-{env}": exec_role, f"stage-{env}": stage_role})

    project_name = n.shared_name("pipeline-build-project", f"{env}-stage")
    project = codebuild.PipelineProject(
        st,
        f"StageProject{cap}",
        project_name=project_name,
        role=stage_role,
        environment=stage_env,
        environment_variables={**env_common, "FINPLAN_ENV": codebuild.BuildEnvironmentVariable(value=env)},
        build_spec=codebuild.BuildSpec.from_object(stage_spec()),
        timeout=Duration.minutes(30),
        cache=codebuild.Cache.local(codebuild.LocalCacheMode.CUSTOM),
        logging=_logging(res, f"StageProject{cap}LogGroup", project_name),
        description=f"{env}: promotion check, release publishing and the {ENV_SUITES[env]} suite",
    )
    tag_role(project, "pipeline-build-project")
    res.projects[env] = project

    stage = cdk.Stage.of(ctx.stacks["storage"]) if ctx.stacks.get("storage") is not None else None
    if stage is None:  # pragma: no cover - the storage module always runs first
        raise ValueError("environment stacks must live in a cdk.Stage")
    execution_id = codebuild.BuildEnvironmentVariable(value="#{codepipeline.PipelineExecutionId}")
    stage_actions: list[codepipeline.IAction] = []
    order = 0
    if env != "beta":
        order = 1
        stage_actions.append(actions.CodeBuildAction(action_name="PromotionCheck", project=project, input=build_output, type=actions.CodeBuildActionType.TEST, run_order=order, environment_variables={"FINPLAN_STAGE_ACTION": codebuild.BuildEnvironmentVariable(value="precheck"), "PIPELINE_EXECUTION_ID": execution_id}))
    for stack in ordered_stacks(stage):
        order += 1
        stage_actions.append(
            actions.CloudFormationCreateUpdateStackAction(
                action_name=f"Deploy{stack.node.id}",
                stack_name=stack.stack_name,
                template_path=build_output.at_path(template_path(stage, stack)),
                admin_permissions=False,
                role=deploy_role,
                deployment_role=exec_role,
                cfn_capabilities=[cdk.CfnCapabilities.NAMED_IAM, cdk.CfnCapabilities.AUTO_EXPAND],
                replace_on_failure=False,
                run_order=order,
            )
        )
    stage_actions.append(actions.CodeBuildAction(action_name="PublishRelease", project=project, input=build_output, type=actions.CodeBuildActionType.BUILD, run_order=order + 1, environment_variables={"FINPLAN_STAGE_ACTION": codebuild.BuildEnvironmentVariable(value="publish"), "PIPELINE_EXECUTION_ID": execution_id}))
    suite = ENV_SUITES[env]
    stage_actions.append(
        actions.CodeBuildAction(
            action_name="".join(part.capitalize() for part in suite.split("-")) + "Tests",
            project=project,
            input=build_output,
            type=actions.CodeBuildActionType.TEST,
            run_order=order + 2,
            environment_variables={"FINPLAN_STAGE_ACTION": codebuild.BuildEnvironmentVariable(value="tests"), "PIPELINE_EXECUTION_ID": execution_id},
        )
    )
    res.pipeline.add_stage(stage_name=cap, actions=stage_actions)


def add_to_app(app: cdk.App, shared: Mapping[str, Any], stages: Mapping[str, Any]) -> None:
    """``infra/app.py`` hook. The pipeline needs all three environment stages."""
    tooling = get_tooling_stack(app, shared)
    missing = [e for e in n.ENVIRONMENTS if e not in stages]
    if missing:
        Annotations.of(tooling).add_warning_v2("finplan:pipeline-skipped", f"pipeline not synthesized: environments {missing} are not selected (-c envs=...)")
        return
    add_pipeline(tooling, stages, shared)
