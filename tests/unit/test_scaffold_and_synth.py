"""Task 1.1: repository layout, and the CDK app synthesizes offline (empty skeleton and with a module)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk import aws_iam as iam

from infra.app import build_app
from infra.stacks.common import ModelStack, StageContext, stack_name, tag_role

ROOT = Path(__file__).resolve().parents[2]


def test_layout_exists():
    for rel in (
        "src/finplan_model/core",
        "src/finplan_model/sim",
        "src/finplan_model/evaluate",
        "src/finplan_model/datasets",
        "src/finplan_model/strategies",
        "src/finplan_model/jobs",
        "src/finplan_model/control",
        "src/finplan_model/staging",
        "src/finplan_model/registry",
        "src/finplan_model/reporting",
        "infra/app.py",
        "config/beta.json",
        "config/gamma.json",
        "config/prod.json",
        "config/shared.json",
        "tests/unit",
        "tests/contract",
        "tests/integration",
        "tests/smoke",
        "contracts-pin.json",
        "cdk.json",
    ):
        assert (ROOT / rel).exists(), rel


@pytest.mark.synth
def test_empty_skeleton_synthesizes_offline(tmp_path):
    app = build_app(cdk.App(outdir=str(tmp_path)), env_modules=(), app_modules=())
    asm = app.synth()
    assert (Path(asm.directory) / "manifest.json").is_file()
    stacks = [s for st in app.node.children if isinstance(st, cdk.Stage) for s in st.synth().stacks]
    assert sorted(s.stack_name for s in stacks) == [f"finplan-{e}-financemodel-skeleton" for e in ("beta", "gamma", "prod")]
    for s in stacks:
        assert {r["Type"] for r in s.template["Resources"].values()} == {"AWS::CDK::Metadata"}
    bare = build_app(cdk.App(outdir=str(tmp_path / "bare")), env_modules=(), app_modules=(), placeholder=False)
    assert bare.synth().stacks == []


@pytest.mark.synth
def test_module_hook_gets_tags_and_permission_boundary(tmp_path, monkeypatch):
    mod = types.ModuleType("infra.stacks._probe")

    def add_to_stage(stage: cdk.Stage, ctx: StageContext) -> None:
        stack = ModelStack(stage, "Probe", cfg=ctx.cfg, description="synth probe", stack_name=stack_name(ctx.cfg.env, "probe"))
        role = iam.Role(stack, "ProbeRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"))
        tag_role(role, "probe")
        ctx.stacks["probe"] = stack

    mod.add_to_stage = add_to_stage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infra.stacks._probe", mod)
    app = build_app(cdk.App(outdir=str(tmp_path)), envs=["beta", "prod"], env_modules=("infra.stacks._probe",), app_modules=())
    app.synth()
    stacks = [s for st in app.node.children if isinstance(st, cdk.Stage) for s in st.synth().stacks]
    names = sorted(s.stack_name for s in stacks)
    assert names == ["finplan-beta-financemodel-probe", "finplan-prod-financemodel-probe"]
    beta = next(s for s in stacks if s.stack_name.startswith("finplan-beta"))
    role = next(r for r in beta.template["Resources"].values() if r["Type"] == "AWS::IAM::Role")
    tags = {t["Key"]: t["Value"] for t in role["Properties"]["Tags"]}
    assert tags == {"project": "finplan", "owner-repo": "financemodel", "environment": "beta", "logical-role": "probe"}
    assert "finplan-beta-permission-boundary" in json.dumps(role["Properties"]["PermissionsBoundary"])
    prod = next(s for s in stacks if s.stack_name.startswith("finplan-prod"))
    assert prod.termination_protection is True
