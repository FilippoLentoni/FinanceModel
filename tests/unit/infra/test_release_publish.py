"""Release manifest and published references (task 10.4; DEP-04, ENV-06, ENV-07, DEP-03 digest), with
moto SSM/S3 and a fake CloudFormation client; registry seeding (design D9)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import boto3
import pytest
from finplan_contracts import ssm as contract_ssm
from finplan_contracts.validate import validate
from moto import mock_aws

from scripts.release import ManifestError, ReleaseInfo, assembly_digest, enforced_role_names, planned_parameters, publish_release, registry_seeder

ACCT = "<account-id>"
DIGEST = "sha256:" + "a" * 64
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
RELEASE = "rel_01KDVDNAZ83BAMMYCEGWF33DPM"


class FakeCfn:
    def __init__(self, outputs: dict[str, dict[str, str]]) -> None:
        self.outputs = outputs

    def describe_stacks(self, StackName: str) -> dict:  # noqa: N803 - boto3 shape
        if StackName not in self.outputs:
            raise RuntimeError("Stack does not exist")
        return {"Stacks": [{"Outputs": [{"OutputKey": k, "OutputValue": v} for k, v in self.outputs[StackName].items()]}]}


def _info(**kw) -> ReleaseInfo:
    base = dict(release_id=RELEASE, source_commit="0" * 40, artifact_digest="sha256:" + "b" * 64, contract_version="1.0.0", contract_digest="sha256:" + "c" * 64, served_contract_majors=[1], region="us-east-2", built_at="2026-10-08T00:00:00Z", image_repository="finplan-shared-financemodel-cpu-images", image_digest=DIGEST)
    base.update(kw)
    return ReleaseInfo(**base)


def _outputs(env: str, *, gaps: bool = True) -> dict[str, dict[str, str]]:
    """``gaps=False`` drops the job API and job-role outputs (what a 0.2.2-era deploy produced)."""
    storage = {"ResearchStorageRef": f"example-research-{env}", "RegistryStorageRef": f"example-registry-{env}", "ControlTableName": f"finplan-{env}-financemodel-job-control"}
    control = {"JobApiRoleRef": f"arn:aws:iam::{ACCT}:role/finplan-{env}-financemodel-job-api-handler-role"}
    if gaps:
        control |= {"JobRoleRef": f"arn:aws:iam::{ACCT}:role/finplan-{env}-financemodel-job-execution-role", "JobEndpoint": "https://api1.execute-api.us-east-2.amazonaws.com/api", "RegistryRef": "https://api1.execute-api.us-east-2.amazonaws.com/api/v1/registry", "ApproverRoleRef": f"arn:aws:iam::{ACCT}:role/finplan-{env}-financemodel-approver-role"}
    return {f"finplan-{env}-financemodel-storage": storage, f"finplan-{env}-financemodel-control": control}


@pytest.fixture
def aws():
    with mock_aws():
        yield {"ssm": boto3.client("ssm", region_name="us-east-2"), "s3": boto3.client("s3", region_name="us-east-2")}


def _get(ssm, name):
    return ssm.get_parameter(Name=name)["Parameter"]["Value"]


def test_beta_12_release_requires_and_publishes_strategy_function(aws):
    outputs = _outputs('beta')
    arn = f'arn:aws:lambda:us-east-2:{ACCT}:function:finplan-beta-financemodel-job-api-handler-inference'
    with pytest.raises(ManifestError, match='StrategyFunctionRef'):
        publish_release(_info(contract_version='1.2.0'), 'beta', ssm=aws['ssm'], cfn=FakeCfn(outputs), account=ACCT, now=NOW)
    outputs['finplan-beta-financemodel-control']['StrategyFunctionRef'] = arn
    manifest = publish_release(_info(contract_version='1.2.0'), 'beta', ssm=aws['ssm'], cfn=FakeCfn(outputs), account=ACCT, now=NOW)
    name = manifest['outputs']['strategy-function-ref']
    assert name == '/finplan/beta/financemodel/api/strategy-function-ref'
    assert _get(aws['ssm'],name) == arn


def test_dep04_beta_manifest_lists_endpoint_registry_job_types_and_contract(aws):
    manifest = publish_release(_info(), "beta", ssm=aws["ssm"], cfn=FakeCfn(_outputs("beta", gaps=True)), account=ACCT, now=NOW)
    assert validate(manifest, "release-manifest").valid  # ENV-06
    out = manifest["outputs"]
    assert out["job-endpoint"] == "/finplan/beta/financemodel/api/job-endpoint"
    assert out["registry-ref"] == "/finplan/beta/financemodel/model/registry-ref"
    assert {"job-prepare-dataset", "job-run-backtest", "job-run-benchmark", "job-report"} <= set(out)
    assert manifest["contract_version"] == "1.0.0" and manifest["previous_release_id"] is None
    for key, name in out.items():  # ENV-07: every name is a valid own-segment parameter
        assert contract_ssm.validation_errors(name) == [], key
        assert contract_ssm.parse(name).repo == "financemodel" and contract_ssm.parse(name).environment == "beta"
        assert _get(aws["ssm"], name)
    job = json.loads(_get(aws["ssm"], out["job-run-backtest"]))
    assert job["image_uri"].endswith(f"/finplan-shared-financemodel-cpu-images@{DIGEST}") and job["deployed"] is True
    assert _get(aws["ssm"], "/finplan/beta/financemodel/release/current-release-id") == RELEASE
    assert json.loads(_get(aws["ssm"], "/finplan/beta/financemodel/release/manifest"))["release_id"] == RELEASE


def test_a_deploy_without_the_job_api_outputs_fails_the_stage(aws):
    """Since contracts 1.0.0 the job API and job role always deploy: missing outputs are a failed deploy."""
    with pytest.raises(ManifestError, match="JobRoleRef|JobEndpoint"):
        publish_release(_info(), "beta", ssm=aws["ssm"], cfn=FakeCfn(_outputs("beta", gaps=False)), account=ACCT, now=NOW)
    assert not aws["ssm"].describe_parameters()["Parameters"]  # nothing published


def test_budget_enforced_role_names_published_and_no_budget_written(aws):
    publish_release(_info(), "gamma", ssm=aws["ssm"], cfn=FakeCfn(_outputs("gamma", gaps=True)), account=ACCT, now=NOW)
    names = _get(aws["ssm"], "/finplan/gamma/financemodel/config/budget-enforced-role-names").split(",")
    assert "finplan-gamma-financemodel-job-api-handler-role" in names and "finplan-gamma-financemodel-job-dispatcher-role" in names
    assert "finplan-gamma-financemodel-job-execution-role" in names
    # account-level tooling roles are in the shared list the bootstrap writes (contracts D16), not here
    assert not [x for x in names if x.startswith("finplan-shared-financemodel-pipeline")]
    assert "finplan-shared-financemodel-deploy-role-gamma" in names and "finplan-gamma-financemodel-pipeline-stage-role" in names
    assert contract_ssm.validate_value("/finplan/gamma/financemodel/config/budget-enforced-role-names", ",".join(names)) == []
    shared = aws["ssm"].describe_parameters()["Parameters"]
    assert not [p for p in shared if p["Name"].startswith("/finplan/shared/")]  # FinanceModel writes nothing shared
    assert enforced_role_names("gamma", {}) == [n for n in names if "job-execution" not in n]


def test_prod_needs_the_approval_and_rollback_is_recorded(aws):
    with pytest.raises(ManifestError, match="approval"):
        publish_release(_info(), "prod", ssm=aws["ssm"], cfn=FakeCfn(_outputs("prod", gaps=True)), account=ACCT, now=NOW)
    first = publish_release(_info(), "prod", ssm=aws["ssm"], cfn=FakeCfn(_outputs("prod", gaps=True)), account=ACCT, now=NOW, approval={"approved_by": "user/approver", "approved_at": "2026-10-08T11:00:00Z"})
    assert first["approved_by"] == "user/approver"
    later = _info(release_id="rel_01KDVDNBYGX5V5HY2JSK5XWKHC", rollback=True)
    second = publish_release(later, "prod", ssm=aws["ssm"], cfn=FakeCfn(_outputs("prod", gaps=True)), account=ACCT, now=NOW, approval={"approved_by": "user/approver", "approved_at": "2026-10-08T11:00:00Z"})
    assert second["previous_release_id"] == RELEASE and second["rolled_back_from"] == RELEASE


def test_job_definitions_must_reference_a_digest():
    with pytest.raises(ManifestError, match="digest"):
        planned_parameters("beta", _info(image_digest="latest"), _outputs("beta", gaps=True)["finplan-beta-financemodel-storage"], account=ACCT, deployed_job_types=["run_backtest"])


def test_missing_stack_output_fails_the_stage(aws):
    outputs = _outputs("beta", gaps=False)
    outputs["finplan-beta-financemodel-control"] = {}
    with pytest.raises(ManifestError, match="JobApiRoleRef"):
        publish_release(_info(), "beta", ssm=aws["ssm"], cfn=FakeCfn(outputs), account=ACCT, now=NOW)
    with pytest.raises(ManifestError, match="not deployed"):
        publish_release(_info(), "beta", ssm=aws["ssm"], cfn=FakeCfn({}), account=ACCT, now=NOW)


def test_registry_seeded_with_baselines_for_the_release_digest(aws):
    s3 = aws["s3"]
    s3.create_bucket(Bucket="example-registry-beta", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
    seed = registry_seeder(s3, "beta", DIGEST)
    publish_release(_info(), "beta", ssm=aws["ssm"], cfn=FakeCfn(_outputs("beta", gaps=True)), account=ACCT, now=NOW, seed=seed)
    records = seed(_outputs("beta", gaps=True)["finplan-beta-financemodel-storage"])  # second run: idempotent
    assert set(records) == {"buy_and_hold", "cash", "equal_weight", "mean_variance", "min_variance", "scenario_cvar"}
    assert all(r["created"] is False and r["image_digest"] == DIGEST for r in records.values())


def test_dep03_artifact_digest_covers_assembly_and_image(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "a.template.json").write_text('{"Resources": {}}')
    d1 = assembly_digest(tmp_path, DIGEST)
    assert d1 == assembly_digest(tmp_path, DIGEST)
    assert d1 != assembly_digest(tmp_path, "sha256:" + "d" * 64)
    (tmp_path / "a.template.json").write_text('{"Resources": {"X": {}}}')
    assert d1 != assembly_digest(tmp_path, DIGEST)
