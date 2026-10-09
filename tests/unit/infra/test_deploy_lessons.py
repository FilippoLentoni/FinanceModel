"""Regressions for the lessons of FinancialPlanning's real deploys (each one broke a deployment that
offline tests had passed). Offline: synthesized templates and contract policy simulation only.

* L1 bootstrap synthesizers: the bootstrap assembly references no ``cdk-hnb659fds`` role and no
  ``cdk-*-assets`` bucket; the store is inline (legacy), the tooling template is staged in the store.
* L2 DynamoDB: no stream action in a table resource policy, and no stream.
* L3 Lambda: every function with an explicit role may write its own log streams (identity policy
  AND permission boundary); the functions run on arm64 (bundle: test_build_and_bootstrap.py).
* L6 bootstrap: no second AWS Budget, budget action or budget parameter from FinanceModel; the
  shared boundary's ``ProtectBudgetActionRole`` guard does not block any FinanceModel role.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from finplan_contracts import boundaries as cb
from finplan_contracts.iam import Request, evaluate

from infra.stacks import naming as n
from infra.stacks import policies as pol

ACCT = "<account-id>"
C = {"partition": "aws", "region": "us-east-2", "account": ACCT}
ENVS = n.ENVIRONMENTS


# ----------------------------------------------------------------- L1
def test_bootstrap_assembly_needs_no_cdk_toolkit_roles(assembly, tmp_path):
    """The platform store stack once carried the default cdk-hnb659fds role ARNs, so its first
    bootstrap failed in an account without CDKToolkit."""
    from scripts.bootstrap import BOOTSTRAP_STACKS, bootstrap_assembly

    out = bootstrap_assembly(assembly.directory, tmp_path / "b")
    manifest = json.loads((out / "manifest.json").read_text())
    stacks = [a for a in manifest["artifacts"].values() if a.get("type") == "aws:cloudformation:stack"]
    assert sorted(a["properties"]["stackName"] for a in stacks) == sorted(BOOTSTRAP_STACKS)
    for art in stacks:
        props = art.get("properties", {})
        for key in ("assumeRoleArn", "cloudFormationExecutionRoleArn", "lookupRole", "requiresBootstrapStackVersion", "bootstrapStackVersionSsmParameter"):
            assert not props.get(key), (art, key)
    text = "".join(p.read_text() for p in out.glob("*.json"))
    assert "cdk-hnb659fds" not in text and not re.search(r"cdk-[a-z0-9]+-assets-", text)
    for path in out.glob("*.assets.json"):
        doc = json.loads(path.read_text())
        assert not doc.get("dockerImages")
        for asset in doc["files"].values():
            for dest in asset["destinations"].values():
                assert dest["bucketName"].startswith("finplan-shared-financemodel-pipeline-store-"), (path.name, dest)
                assert dest["objectKey"].startswith("bootstrap/") and "assumeRoleArn" not in dest
    store = manifest["artifacts"]["PipelineStore"]["properties"]
    assert not store.get("stackTemplateAssetObjectUrl"), "the store stack deploys inline (LegacyStackSynthesizer)"
    for name in BOOTSTRAP_STACKS:
        tpl = assembly.stack(name)
        assert "BootstrapVersion" not in tpl.get("Parameters", {}) and "CheckBootstrapVersion" not in tpl.get("Rules", {})


def test_environment_stacks_need_no_cdk_toolkit(assembly):
    for name, tpl in assembly.templates.items():
        text = json.dumps(tpl)
        assert "cdk-hnb659fds" not in text and not re.search(r"cdk-[a-z0-9]+-assets-", text), name
        assert "BootstrapVersion" not in tpl.get("Parameters", {}), name


# ----------------------------------------------------------------- L2
STREAM_ACTIONS = {"dynamodb:GetRecords", "dynamodb:GetShardIterator", "dynamodb:DescribeStream", "dynamodb:ListStreams"}


def _actions(stmt: dict[str, Any]) -> set[str]:
    a = stmt.get("Action", [])
    return set(a if isinstance(a, list) else [a])


def test_dynamodb_table_policies_have_no_stream_actions(assembly):
    """DynamoDB rejects stream actions in a table resource policy ("Invalid policy document"),
    which failed the platform's first beta deploy."""
    tables = [(name, r) for name, tpl in assembly.templates.items() for r in tpl["Resources"].values() if r["Type"] == "AWS::DynamoDB::Table"]
    assert len(tables) == len(ENVS)
    for name, r in tables:
        p = r["Properties"]
        assert "StreamSpecification" not in p, name
        stmts = p["ResourcePolicy"]["PolicyDocument"]["Statement"]
        assert stmts
        for s in stmts:
            assert not STREAM_ACTIONS & _actions(s), (name, s.get("Sid"))
            assert not any(a in ("dynamodb:*", "*") for a in _actions(s)), (name, s.get("Sid"))  # a wildcard implies stream actions
    assert not STREAM_ACTIONS & set(pol.DYNAMODB_DATA_ACTIONS)


# ----------------------------------------------------------------- L3
def _lambda_roles(tpl: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    res = tpl["Resources"]
    out = []
    for r in res.values():
        if r["Type"] != "AWS::Lambda::Function":
            continue
        role_ref = r["Properties"]["Role"]
        assert "Fn::GetAtt" in role_ref, "every FinanceModel function has an explicit role"
        out.append((r["Properties"]["FunctionName"], r, res[role_ref["Fn::GetAtt"][0]]))
    return out


def test_every_function_with_an_explicit_role_can_write_its_own_log_streams(assembly):
    seen = 0
    for name, tpl in assembly.templates.items():
        for fn, func, role in _lambda_roles(tpl):
            seen += 1
            assert func["Properties"]["Architectures"] == ["arm64"], fn
            text = json.dumps(role["Properties"].get("Policies", []))
            assert "logs:CreateLogStream" in text and "logs:PutLogEvents" in text, fn
            assert f"log-group:/aws/lambda/{fn}:*" in text, f"{fn} may write only its own log group"
            groups = [r for r in tpl["Resources"].values() if r["Type"] == "AWS::Logs::LogGroup" and "/aws/lambda/" in json.dumps(r["Properties"]["LogGroupName"])]
            assert any(fn in json.dumps(g["Properties"]["LogGroupName"]) or "Ref" in json.dumps(g["Properties"]["LogGroupName"]) for g in groups), fn
    assert seen == (len(n.FUNCTIONS) - 1) * len(ENVS) + 1  # dedicated inference is beta-only


@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("logical,doc", [(n.JOB_API_HANDLER, "api"), (n.DISPATCHER, "dispatcher"), (n.STATE_HANDLER, "state"), (n.REGISTRY_LOOKUP, "lookup")])
def test_log_writes_pass_the_identity_policy_and_the_boundary(env, logical, doc):
    policy = pol.registry_lookup_policy(env, **C) if doc == "lookup" else pol.control_role_policy(env, doc, **C)
    boundary = cb.env_permission_boundary(env, **C)
    stream = f"arn:aws:logs:us-east-2:{ACCT}:log-group:/aws/lambda/{n.function_name(env, logical)}:log-stream:2026/10/08/[$LATEST]abc"
    for action in ("logs:CreateLogStream", "logs:PutLogEvents"):
        assert evaluate(Request(action, stream), [policy], boundary).allowed, (env, logical, action)
    other = stream.replace(n.function_name(env, logical), "finplan-beta-financialplanning-plan-api")
    assert not evaluate(Request("logs:PutLogEvents", other), [policy], boundary).allowed


@pytest.mark.parametrize("env", ENVS)
def test_stage_role_may_invoke_only_its_own_dispatcher(env):
    role = {"identity": [{"Version": "2012-10-17", "Statement": pol.stage_role_statements(env, f"arn:aws:s3:::{n.pipeline_store_bucket_name(ACCT)}", **C)}], "boundary": cb.env_permission_boundary(env, **C)}
    fn = f"arn:aws:lambda:us-east-2:{ACCT}:function:{n.function_name(env, n.DISPATCHER)}"
    assert evaluate(Request("lambda:InvokeFunction", fn), role["identity"], role["boundary"]).allowed
    api = fn.replace(n.DISPATCHER, n.JOB_API_HANDLER)
    assert not evaluate(Request("lambda:InvokeFunction", api), role["identity"], role["boundary"]).allowed
    other = [e for e in ENVS if e != env][0]
    assert not evaluate(Request("lambda:InvokeFunction", fn.replace(f"-{env}-", f"-{other}-")), role["identity"], role["boundary"]).allowed


# ----------------------------------------------------------------- L6
def test_financemodel_creates_no_budget(assembly):
    for name, tpl in assembly.templates.items():
        types = {r["Type"] for r in tpl["Resources"].values()}
        assert not {t for t in types if t.startswith("AWS::Budgets::")}, name
        text = json.dumps(tpl)
        assert "budget-allocation" not in text and "cost-ceiling-usd" not in text, name


def _fm_role_names(asm) -> list[str]:
    names = []
    for tpl in asm.templates.values():
        for r in tpl["Resources"].values():
            if r["Type"] == "AWS::IAM::Role":
                names.append(r["Properties"]["RoleName"])
    return names


def test_protect_budget_action_role_guard_does_not_block_financemodel_roles(assembly):
    """The shared boundary denies creating, re-trusting, re-permissioning or passing a role named
    ``finplan-shared-*-budget-action-role`` (contracts 0.2.2, D15). No FinanceModel role may match
    it, or the bootstrap (tooling roles) and the pipeline (PassRole) would be denied."""
    names = _fm_role_names(assembly)
    assert len(names) >= 11 + 6 * len(ENVS)
    pattern = re.compile("^" + re.escape(cb.BUDGET_ACTION_ROLE_NAME_PATTERN).replace(r"\*", ".*") + "$")
    assert not [x for x in names if pattern.match(x)]
    shared = cb.shared_permission_boundary(**C)
    allow_all = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}
    actions = ("iam:CreateRole", "iam:UpdateAssumeRolePolicy", "iam:PutRolePolicy", "iam:AttachRolePolicy", "iam:PassRole")
    for name in names:
        arn = f"arn:aws:iam::{ACCT}:role/{name}"
        for action in actions:
            res = evaluate(Request(action, arn, {"aws:PrincipalArn": f"arn:aws:iam::{ACCT}:role/{n.shared_name('pipeline', 'role')}", "iam:PermissionsBoundary": f"arn:aws:iam::{ACCT}:policy/finplan-beta-permission-boundary"}), [allow_all], shared)
            assert not any("ProtectBudgetActionRole" in d for d in res.matched_deny), (name, action, res.matched_deny)
    # the guard is live: the budget action role itself is protected
    guarded = evaluate(Request("iam:PassRole", f"arn:aws:iam::{ACCT}:role/finplan-shared-financialplanning-budget-action-role"), [allow_all], shared)
    assert any("ProtectBudgetActionRole" in d for d in guarded.matched_deny)


@pytest.mark.parametrize("env", ENVS)
def test_exec_role_may_create_research_and_env_bounded_roles(env):
    """The CloudFormation execution role (environment boundary) creates the job-execution role with
    the research boundary and every other role with the environment boundary."""
    doc = {"Version": "2012-10-17", "Statement": pol.deploy_execution_statements(env, f"arn:aws:s3:::{n.pipeline_store_bucket_name(ACCT)}", **C)}
    boundary = cb.env_permission_boundary(env, **C)
    for logical, b in ((n.JOB_EXECUTION, cb.research_boundary_name(env)), (n.JOB_API_HANDLER, cb.boundary_name(env))):
        arn = f"arn:aws:iam::{ACCT}:role/{n.role_name(env, logical)}"
        req = Request("iam:CreateRole", arn, {"iam:PermissionsBoundary": f"arn:aws:iam::{ACCT}:policy/{b}"})
        assert evaluate(req, [doc], boundary).allowed, (env, logical)
    bare = Request("iam:CreateRole", f"arn:aws:iam::{ACCT}:role/{n.role_name(env, n.JOB_API_HANDLER)}", {})
    assert not evaluate(bare, [doc], boundary).allowed


def test_bootstrap_reads_the_shared_finplan_config_and_ignores_platform_keys(tmp_path, monkeypatch):
    from scripts import bootstrap as fm_bootstrap

    shared = tmp_path / "home" / "bootstrap.json"
    shared.parent.mkdir()
    shared.write_text(json.dumps({"account_id": "<account-id>", "primary_region": "us-east-2", "repo": "financialplanning", "codeconnection_arn": "connection-placeholder", "github_repository": "FilippoLentoni/FinancialPlanning", "pipeline_name": "finplan-shared-financialplanning-pipeline", "budget_notification_email": "someone@example.invalid", "scope_budget_to_project_tag": False}))
    local = fm_bootstrap.read_local_config(None, environ={"FINPLAN_BOOTSTRAP_CONFIG": str(shared)}, overlay=tmp_path / "absent.json")
    assert local == {"account_id": "<account-id>", "primary_region": "us-east-2", "codeconnection_arn": "connection-placeholder"}
    config = fm_bootstrap.load_bootstrap_config(local, ssm=None)
    assert (config.repo, config.github_repository, config.pipeline_name) == ("financemodel", "FilippoLentoni/FinanceModel", "finplan-shared-financemodel-pipeline")
    assert config.codeconnection_arn == "connection-placeholder"  # the platform's existing connection, reused
    overlay = tmp_path / "fm.json"
    overlay.write_text(json.dumps({"primary_region": "us-east-2", "github_repository": "FilippoLentoni/FinanceModel"}))
    merged = fm_bootstrap.read_local_config(shared, environ={"FINPLAN_ACCOUNT_ID": "<other>"}, overlay=overlay)
    assert merged["account_id"] == "<other>" and merged["github_repository"] == "FilippoLentoni/FinanceModel" and "budget_notification_email" not in merged
    inside = fm_bootstrap.ROOT / "config" / "shared.json"
    with pytest.raises(fm_bootstrap.BootstrapStop, match="inside the repository"):
        fm_bootstrap.read_local_config(inside, environ={}, overlay=None)


def test_bootstrap_needs_no_botocore_crt():
    """The DevDesktop instance role is a plain credential provider: the bootstrap must not need
    ``botocore[crt]`` (only ``aws login`` sessions do)."""
    from scripts import bootstrap as fm_bootstrap

    import ast

    tree = ast.parse((fm_bootstrap.ROOT / "scripts" / "bootstrap.py").read_text())
    imported = {a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "awscrt" not in imported
    runbook = (fm_bootstrap.ROOT / "docs" / "bootstrap.md").read_text()
    steps = runbook[runbook.index("## Steps"):]
    assert 'uv run python scripts/bootstrap.py' in steps and '--with "botocore[crt]" python scripts/bootstrap.py' not in steps
    clients = fm_bootstrap.make_clients("us-east-2")  # fake credentials offline; builds without awscrt
    assert clients.sts.meta.region_name == "us-east-2" and clients.pricing.meta.region_name == "us-east-1"


# ----------------------------------------------------------------- L3: the post-synth bundle gate
def _asset(root, name, *, platform=None, complete=True):
    a = root / f"asset.{name}"
    (a / "finplan_model").mkdir(parents=True)
    if complete:
        for entry in ("finplan_contracts", "jsonschema", "rfc8785", "numpy", "boto3", "config"):
            (a / entry).mkdir()
        (a / "bundle-manifest.json").write_text(json.dumps({"python_platform": platform}))
    return a


def test_lambda_bundle_gate_refuses_source_only_and_x86_code(tmp_path):
    from scripts.build_gates import GateContext, gate_lambda_bundle

    asm = tmp_path / "cdk.out"
    _asset(asm, "ok", platform="aarch64-manylinux_2_28")
    (asm / "manifest.json").write_text("{}")
    assert gate_lambda_bundle(GateContext(root=tmp_path, assembly=asm)) == []
    _asset(asm, "src", complete=False)
    x86 = _asset(asm, "x86", platform="x86_64-manylinux_2_28")
    so = x86 / "numpy" / "core.so"
    so.write_bytes(b"\x7fELF\x02\x01\x01" + b"\0" * 11 + (0x3E).to_bytes(2, "little") + b"\0" * 40)
    problems = gate_lambda_bundle(GateContext(root=tmp_path, assembly=asm))
    text = "\n".join(problems)
    assert "asset.src: finplan_contracts missing" in text and "asset.src: bundle-manifest.json missing" in text
    assert "asset.x86: built for x86_64-manylinux_2_28" in text and "numpy/core.so is not an arm64 binary" in text
