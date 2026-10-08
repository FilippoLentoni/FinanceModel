"""Pipeline and account-level tooling (task 10.5; DEP-01, DEP-02, DEP-03, DEP-05 structure; ENV-09, ENV-12)."""

from __future__ import annotations

import json

import pytest
from finplan_contracts.bootstrap import check_deploy_roles
from finplan_contracts.pipeline_check import check_pipeline_template

from infra.stacks import naming as n
from infra.stacks.pipeline import STAGE_NAMES, build_spec, stage_spec

pytestmark = pytest.mark.synth
TOOLING = "finplan-shared-financemodel-tooling"


def _pipeline(assembly):
    (p,) = assembly.resources(TOOLING, "AWS::CodePipeline::Pipeline").values()
    return p["Properties"]


def _tags(res):
    return {t["Key"]: t["Value"] for t in res["Properties"].get("Tags", [])}


def test_env09_pipeline_standard_and_scoped_deploy_roles(assembly):
    for asm in (assembly,):
        template = asm.stack(TOOLING)
        assert check_pipeline_template(template) == []
        assert check_deploy_roles(template) == []


def test_dep03_stage_order_source_and_artifact_only_promotion(assembly):
    props = _pipeline(assembly)
    assert props["PipelineType"] == "V2" and props["Name"] == n.PIPELINE_NAME
    assert [s["Name"] for s in props["Stages"]] == list(STAGE_NAMES)
    (source,) = props["Stages"][0]["Actions"]
    cfg = source["Configuration"]
    assert cfg["FullRepositoryId"] == "FilippoLentoni/FinanceModel" and cfg["BranchName"] == "main"
    assert cfg["ConnectionArn"] == {"Ref": next(k for k in assembly.stack(TOOLING)["Parameters"] if k.startswith("SsmParameterValuefinplansharedfinancemodelconfigcodeconnectionref"))}
    assert assembly.stack(TOOLING)["Parameters"][cfg["ConnectionArn"]["Ref"]]["Default"] == "/finplan/shared/financemodel/config/codeconnection-ref"
    build_outputs = {o["Name"] for a in props["Stages"][1]["Actions"] for o in a.get("OutputArtifacts", [])}
    assert build_outputs == {"BuildOutput"}
    for stage in props["Stages"][2:]:
        for action in stage["Actions"]:
            assert {i["Name"] for i in action.get("InputArtifacts", [])} <= {"BuildOutput"}
    assert "rollback_to_release_id" in [v["Name"] for v in props["Variables"]]
    assert "Fn::If" in json.dumps(props["DisableInboundStageTransitions"])  # Build stays disabled until the dry run passed


def test_dep05_environment_stages(assembly):
    stages = {s["Name"]: s["Actions"] for s in _pipeline(assembly)["Stages"]}
    for env, tests in (("Beta", "IntegrationBetaTests"), ("Gamma", "GammaTests"), ("Prod", "SmokeTests")):
        names = [a["Name"] for a in sorted(stages[env], key=lambda a: a["RunOrder"])]
        deploys = [a for a in stages[env] if a["ActionTypeId"]["Category"] == "Deploy"]
        assert [a["Name"] for a in deploys] == ["DeployStorage", "DeployControl"]
        assert names[-2:] == ["PublishRelease", tests]
        if env != "Beta":
            assert names[0] == "PromotionCheck"  # contract pin (0.x never leaves beta) and image digest
        for d in deploys:
            assert d["Configuration"]["StackName"] == f"finplan-{env.lower()}-financemodel-{d['Name'][6:].lower()}"
            assert d["Configuration"]["TemplatePath"].startswith("BuildOutput::cdk.out/assembly-")
    assert [a["ActionTypeId"]["Category"] for a in stages["Approval"]] == ["Approval"]


def test_dep01_dep02_build_spec_and_stage_spec():
    spec = json.dumps(build_spec())
    assert "scripts/build_stage.py" in spec and "FINPLAN_RELEASE_BUILD" in spec
    stage = json.dumps(stage_spec())
    assert "cdk synth" not in stage and "build_stage" not in stage and "scripts/stage_runner.py" in stage
    # experiments are never pipeline executions: no stage action submits a job or runs a strategy
    assert "submit" not in stage and "run_backtest" not in spec


def test_pipeline_roles_are_scoped_and_bounded(assembly):
    roles = assembly.resources(TOOLING, "AWS::IAM::Role")
    by_name = {r["Properties"]["RoleName"]: r for r in roles.values()}
    for env in n.ENVIRONMENTS:
        for name in (n.deploy_role_name(env), n.exec_role_name(env), n.stage_role_name(env)):
            role = by_name[name]
            assert _tags(role)["environment"] == env
            assert f"finplan-{env}-permission-boundary" in json.dumps(role["Properties"]["PermissionsBoundary"])
    for name in (n.shared_name("pipeline", "role"), n.shared_name("pipeline-build-project", "role")):
        assert _tags(by_name[name])["environment"] == "shared"
        assert "finplan-shared-permission-boundary" in json.dumps(by_name[name]["Properties"]["PermissionsBoundary"])
    projects = assembly.resources(TOOLING, "AWS::CodeBuild::Project")
    build = next(p for p in projects.values() if p["Properties"]["Name"] == n.shared_name("pipeline-build-project"))
    assert build["Properties"]["Environment"]["PrivilegedMode"] is True  # docker build of the CPU image
    assert build["Properties"]["Environment"]["Type"] == "LINUX_CONTAINER"


def test_image_repository_and_store(assembly):
    (repo,) = assembly.resources(TOOLING, "AWS::ECR::Repository").values()
    props = repo["Properties"]
    assert props["RepositoryName"] == n.ecr_repository_name() and props["ImageTagMutability"] == "IMMUTABLE"
    assert props["ImageScanningConfiguration"] == {"ScanOnPush": True}
    assert _tags(repo)["logical-role"] == "financemodel-image-repository" and _tags(repo)["environment"] == "shared"
    store = assembly.stack("finplan-shared-financemodel-pipeline-store")
    (bucket,) = [r for r in store["Resources"].values() if r["Type"] == "AWS::S3::Bucket"]
    rules = {r["Id"]: r for r in bucket["Properties"]["LifecycleConfiguration"]["Rules"]}
    assert "releases" not in json.dumps([r.get("Prefix") for r in rules.values()])  # the ledger never expires
    assert bucket["DeletionPolicy"] == "Retain"


def test_bootstrap_stacks_need_no_cdk_toolkit(assembly):
    from scripts.bootstrap import BOOTSTRAP_STACKS, bootstrap_assembly, cdk_bootstrap_references

    for name in BOOTSTRAP_STACKS:
        text = json.dumps(assembly.stack(name))
        assert "hnb659fds" not in text and "cdk-" not in text.replace("cdk-metadata", "")
    out = bootstrap_assembly(assembly.directory, assembly.directory.parent / "boot")
    manifest = json.loads((out / "manifest.json").read_text())
    stacks = sorted(a["properties"]["stackName"] for a in manifest["artifacts"].values() if a["type"] == "aws:cloudformation:stack")
    assert stacks == sorted(BOOTSTRAP_STACKS)
    assert cdk_bootstrap_references(out) == []
    assets = json.loads((out / "Tooling.assets.json").read_text())
    (dest,) = [d for f in assets["files"].values() for d in f["destinations"].values()]
    assert dest["bucketName"] == "finplan-shared-financemodel-pipeline-store-${AWS::AccountId}" and dest["objectKey"].startswith("bootstrap/")


def test_codebuild_projects_log_to_explicit_30_day_groups(assembly):
    """The four FinanceModel CodeBuild projects (build plus the beta, gamma and prod stage projects)
    log to explicit ``/aws/codebuild/<project>`` groups: 30 days, deleted with the stack, attributed to
    matrix row ``pipeline-financemodel`` (mirrors FinancialPlanning ``add_log_group``)."""
    from finplan_contracts.ownership import check_template

    template = assembly.stack(TOOLING)
    projects = assembly.resources(TOOLING, "AWS::CodeBuild::Project")
    groups = assembly.resources(TOOLING, "AWS::Logs::LogGroup")
    assert len(projects) == 4 and len(groups) == 4
    names = [g["Properties"]["LogGroupName"] for g in groups.values()]
    expected = [n.shared_name("pipeline-build-project")] + [n.shared_name("pipeline-build-project", f"{e}-stage") for e in n.ENVIRONMENTS]
    assert sorted(names) == sorted(f"/aws/codebuild/{p}" for p in expected)
    for g in groups.values():
        assert g["Properties"]["RetentionInDays"] == 30
        assert g["DeletionPolicy"] == "Delete" and g["UpdateReplacePolicy"] == "Delete"
        assert _tags(g)["logical-role"] == "pipeline-build-project" and _tags(g)["environment"] == "shared"
    for p in projects.values():
        cw = p["Properties"]["LogsConfig"]["CloudWatchLogs"]
        assert cw["Status"] == "ENABLED" and cw["GroupName"]["Ref"] in groups
    report = check_template(template, "financemodel", name=TOOLING)
    assert report.ok, [str(x) for x in report.problems]
