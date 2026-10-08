"""OWN-01 (ownership check with zero problems against the pinned contract matrix), ENV-16, ENV-18,
ENV-05 on every synthesized template (task 10.3). Since contracts 1.0.0 (D16) the matrix carries
every FinanceModel row, so nothing is held back."""

from __future__ import annotations

from pathlib import Path

import pytest
from finplan_contracts import live_perms
from finplan_contracts.boundaries import check_role_boundaries, check_shared_resources
from finplan_contracts.ownership import Matrix, check_template

pytestmark = pytest.mark.synth


def test_own01_every_template_passes_the_pinned_matrix(assembly):
    assert len(assembly.templates) == 8
    for name, template in assembly.templates.items():
        report = check_template(template, "financemodel", name=name)
        assert report.ok, [str(p) for p in report.problems]
        assert report.checked == len(template["Resources"])


#: (CloudFormation type, logical role) pairs contracts 1.0.0 added for FinanceModel (D16); they were
#: the four "contract gaps" of 0.2.2 and are now synthesized unconditionally.
CONTRACT_100_PAIRS = (
    ("AWS::ApiGateway::Deployment", "job-api"),
    ("AWS::ApiGateway::Stage", "job-api"),
    ("AWS::IAM::Role", "job-execution-role"),
    ("AWS::S3::BucketPolicy", "model-registry"),
    ("AWS::Logs::LogGroup", "pipeline-build-project"),
)


def test_pinned_matrix_carries_the_financemodel_rows():
    m = Matrix.load()
    for rtype, role in CONTRACT_100_PAIRS:
        rows = [r for r in m.rows_for_role(role) if rtype in (r.get("resource_types") or []) and r.get("owner") == "financemodel"]
        assert rows, (rtype, role)


def _logical_role(res):
    tags = res.get("Properties", {}).get("Tags")
    if isinstance(tags, list):
        found = {t.get("Key"): t.get("Value") for t in tags}.get("logical-role")
        if found:
            return found
    return (res.get("Metadata") or {}).get("logical-role")


def test_former_gap_resources_are_synthesized_and_owned(assembly):
    """No resource waits for a matrix row any more: each formerly gated pair exists and passes OWN-01."""
    seen = set()
    for template in assembly.templates.values():
        for res in template["Resources"].values():
            pair = (res.get("Type"), _logical_role(res))
            if pair in CONTRACT_100_PAIRS:
                seen.add(pair)
    untaggable = {("AWS::ApiGateway::Deployment", "job-api"), ("AWS::S3::BucketPolicy", "model-registry")}  # attributed to their parent
    missing = set(CONTRACT_100_PAIRS) - seen - untaggable
    assert not missing, missing
    for env in ("beta", "gamma", "prod"):
        assert assembly.resources(f"finplan-{env}-financemodel-control", "AWS::ApiGateway::Deployment"), env
        assert len(assembly.resources(f"finplan-{env}-financemodel-storage", "AWS::S3::BucketPolicy")) == 2, env
    assert not list(Path(__file__).resolve().parents[3].joinpath("infra", "stacks").glob("contract_gaps.py"))


def test_env18_env16_env05_on_every_template(assembly):
    for asm in (assembly,):
        for name, template in asm.templates.items():
            assert check_role_boundaries(template) == [], name
            assert check_shared_resources(template, repo="financemodel") == [], name
        n_scanned, findings = live_perms.scan_paths(sorted(asm.directory.rglob("*.template.json")))
        assert n_scanned >= 8 and findings == []


def test_no_budget_and_no_boundary_is_created_by_financemodel(assembly):
    for asm in (assembly,):
        for name, template in asm.templates.items():
            types = {r["Type"] for r in template["Resources"].values()}
            assert not any(t.startswith("AWS::Budgets::") for t in types), name  # FinancialPlanning owns the budget
            assert "AWS::IAM::ManagedPolicy" not in types, name  # boundaries come from the platform tooling stack
            assert "AWS::SSM::Parameter" not in types, name  # references are published by the release step
            assert "AWS::KMS::Key" not in types, name
