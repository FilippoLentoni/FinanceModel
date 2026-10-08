#!/usr/bin/env python3
"""One-time authenticated bootstrap of the FinanceModel pipeline (task 10.5; contracts D6, D11, D12;
ENV-12, ENV-13). Runbook: ``docs/bootstrap.md``.

DO NOT RUN during implementation work. The user approved the bootstrap in principle on 2026-10-07;
it runs once, by a human, only after this IaC is synthesized, and only after the exact stacks and a
cost estimate have been shown and confirmed.

The sequence is the contract package's :func:`finplan_contracts.bootstrap.run_bootstrap` (never
re-implemented); this entry point supplies the FinanceModel specifics:

1. **Assembly** (:func:`bootstrap_assembly`): refuses without a synthesized assembly, copies only the
   two account-level stacks ``finplan-shared-financemodel-pipeline-store`` and
   ``finplan-shared-financemodel-tooling`` into ``cdk.out.bootstrap/`` and refuses an assembly that
   references the CDK bootstrap (``cdk-hnb659fds`` roles, ``cdk-*-assets`` buckets) or carries
   container-image assets (lesson of the platform bootstrap).
2. **Pre-run plan** (contract): the exact stacks with their resource types and a monthly estimate
   from the AWS Price List API (no price is written in this repository).
3. **Caller, region, connection, scoped-role checks** (contract). The CodeConnection is the
   **same existing connection the platform uses**: when the local configuration names none, it is
   read (read-only) from ``/finplan/shared/financialplanning/config/codeconnection-ref``.

Local configuration (:func:`read_local_config`) is read **the same way as the platform's**
(:func:`finplan_contracts.bootstrap.load_config`): ``--config PATH``, else
``$FINPLAN_BOOTSTRAP_CONFIG``, else the shared ``~/.finplan/bootstrap.json`` the FinancialPlanning
bootstrap already uses (account, primary region, the existing connection), refused inside the
repository; then the optional FinanceModel overlay ``~/.finplan/financemodel-bootstrap.json``;
then ``FINPLAN_ACCOUNT_ID`` / ``FINPLAN_PRIMARY_REGION`` / ``FINPLAN_CODECONNECTION_ARN``. The
platform-only keys of the shared file (its repository, pipeline name, budget notification address)
are ignored and never printed: ``repo``, ``github_repository`` and ``pipeline_name`` are always
FinanceModel's.

Credentials come from the default boto3 chain only: the DevDesktop instance role works as is (no
``botocore[crt]``, which only ``aws login`` sessions need).
4. **Confirmation**: the operator types ``deploy``.
5. **Connection reference** ``/finplan/shared/financemodel/config/codeconnection-ref``.
6. **Deploy** (:class:`ToolingDeployer`): ``npx aws-cdk@2 deploy --all`` of the filtered assembly.
   No budget parameter is written and no budget is created: the project budget, its alerts, its deny
   action and the default allocation belong to the FinancialPlanning bootstrap. The deployer only
   checks (read-only) that the shared allocation exists and reports it. After a successful deploy it
   writes, as the ``bootstrap`` writer:

   * ``/finplan/shared/financemodel/config/budget-enforced-role-names`` - the account-level pipeline
     and build role names (:func:`infra.stacks.naming.tooling_role_names`; contracts 1.0.0, D16), read
     by the FinancialPlanning bootstrap for its 100% budget deny;
   * ``/finplan/<env>/financemodel/config/instance-prices`` for beta, gamma and prod, **only** when
     absent or stale, with on-demand SageMaker Processing prices fetched from the AWS Price List API
     at run time (``scripts/instance_prices.py``; no price is ever in this repository);

   and reports, read-only, whether the operator has set ``config/integration-snapshot-id`` in beta and
   gamma (never written: optional override; without it the suite uses the snapshot the platform's
   own suite publishes, a real SPY snapshot in phase 2, user decision 26).
7. **Source-stage dry run** (contract): an execution must fetch ``main`` of
   ``FilippoLentoni/FinanceModel`` before the stages after Source are enabled; otherwise it stops
   with the extend-the-GitHub-App-installation message.

Every AWS client is injected, so the unit suite runs the whole sequence with mocks and no AWS call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from finplan_contracts import bootstrap as contract_bootstrap  # noqa: E402
from finplan_contracts import budget as contract_budget  # noqa: E402
from finplan_contracts import ssm as contract_ssm  # noqa: E402
from finplan_contracts.bootstrap import BootstrapConfig, BootstrapStop, Clients, Plan  # noqa: E402

from infra.stacks.naming import REPO, tooling_role_names  # noqa: E402
from infra.stacks.tooling import STORE_STACK_NAME, TOOLING_STACK_NAME  # noqa: E402

__all__ = [
    "BOOTSTRAP_STACKS",
    "INTEGRATION_SNAPSHOT_ENVS",
    "PLATFORM_CONNECTION_PARAMETER",
    "SHARED_ENFORCED_ROLES_PARAMETER",
    "ToolingDeployer",
    "bootstrap_assembly",
    "cdk_bootstrap_references",
    "load_bootstrap_config",
    "main",
    "publish_tooling_role_names",
    "read_local_config",
    "report_operator_parameters",
    "run",
]

BOOTSTRAP_STACKS = (STORE_STACK_NAME, TOOLING_STACK_NAME)
DEFAULT_ASSEMBLY = ROOT / "cdk.out"
BOOTSTRAP_ASSEMBLY = ROOT / "cdk.out.bootstrap"
#: Optional FinanceModel overlay on top of the shared configuration (never required).
DEFAULT_CONFIG = Path("~/.finplan/financemodel-bootstrap.json")
#: The shared local configuration the platform bootstrap reads (contract default).
SHARED_CONFIG = contract_bootstrap.DEFAULT_CONFIG_PATH
#: Keys of the shared file that describe the platform's own pipeline, or are platform-only.
_PLATFORM_ONLY_KEYS = ("repo", "github_repository", "pipeline_name", "source_stage", "next_stage", "budget_notification_email", "scope_budget_to_project_tag")
DRY_RUN_RECORD = Path("~/.finplan/financemodel-source-dry-run.json")
PLATFORM_CONNECTION_PARAMETER = contract_ssm.build(contract_ssm.SHARED, "financialplanning", "config", "codeconnection-ref")
_CDK_BOOTSTRAP_RE = re.compile(r"cdk-hnb659fds|cdk-[a-z0-9]+-assets-")
#: Account-level tooling role names for the platform's budget action (contracts 1.0.0, D16).
SHARED_ENFORCED_ROLES_PARAMETER = contract_ssm.build(contract_ssm.SHARED, REPO, "config", contract_ssm.BUDGET_ENFORCED_ROLE_NAMES)
#: Environments whose deployed suite runs a fixture job on an operator-chosen approved snapshot.
INTEGRATION_SNAPSHOT_ENVS = ("beta", "gamma")


# ===================================================================== assembly
def cdk_bootstrap_references(path: Path) -> list[str]:
    """Files of an assembly directory that reference CDK bootstrap roles or asset buckets."""
    found = []
    for p in sorted(path.rglob("*.json")):
        if _CDK_BOOTSTRAP_RE.search(p.read_text(encoding="utf-8", errors="replace")):
            found.append(p.relative_to(path).as_posix())
    return found


def bootstrap_assembly(assembly: str | os.PathLike[str], out: str | os.PathLike[str]) -> Path:
    src = Path(assembly)
    manifest_path = src / "manifest.json"
    if not manifest_path.is_file():
        raise BootstrapStop("prerun", f"the bootstrap IaC is not synthesized: no cloud assembly at {src} (run: uv run python scripts/synth.py). Nothing was deployed.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arts = manifest.get("artifacts") or {}
    keep = {aid: a for aid, a in arts.items() if a.get("type") == "aws:cloudformation:stack" and (a.get("properties") or {}).get("stackName") in BOOTSTRAP_STACKS}
    names = {(a.get("properties") or {}).get("stackName") for a in keep.values()}
    missing = [s for s in BOOTSTRAP_STACKS if s not in names]
    if missing:
        raise BootstrapStop("prerun", f"the synthesized assembly does not contain the tooling stacks {missing}; nothing was deployed")
    for _aid, art in list(keep.items()):
        for dep in art.get("dependencies") or []:
            if dep in arts and arts[dep].get("type") == "cdk:asset-manifest":
                keep[dep] = arts[dep]
    dst = Path(out)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    files: set[str] = set()
    for art in keep.values():
        props = art.get("properties") or {}
        for key in ("templateFile", "file"):
            if props.get(key):
                files.add(props[key])
        if art.get("additionalMetadataFile"):
            files.add(art["additionalMetadataFile"])
        if art.get("type") == "cdk:asset-manifest":
            doc = json.loads((src / props["file"]).read_text(encoding="utf-8"))
            for asset in (doc.get("files") or {}).values():
                files.add(str((asset.get("source") or {}).get("path")))
            if doc.get("dockerImages"):
                raise BootstrapStop("prerun", "the tooling stacks must not contain container-image assets")
    for rel in sorted(files):
        s = src / rel
        if s.is_dir():
            shutil.copytree(s, dst / rel)
        elif s.is_file():
            shutil.copy2(s, dst / rel)
        else:
            raise BootstrapStop("prerun", f"{rel} is missing from the cloud assembly")
    out_manifest = {**{k: v for k, v in manifest.items() if k != "artifacts"}, "artifacts": {}}
    for aid, art in keep.items():
        art = json.loads(json.dumps(art))
        art["dependencies"] = [d for d in art.get("dependencies") or [] if d in keep]
        props = art.get("properties") or {}
        if "additionalDependencies" in props:
            props["additionalDependencies"] = [d for d in props["additionalDependencies"] if d in keep]
        out_manifest["artifacts"][aid] = art
    (dst / "manifest.json").write_text(json.dumps(out_manifest, indent=1) + "\n", encoding="utf-8")
    refs = cdk_bootstrap_references(dst)
    if refs:
        raise BootstrapStop("prerun", f"the bootstrap assembly references the CDK bootstrap stack (cdk-hnb659fds roles or cdk-*-assets buckets) in {refs}; it must deploy without CDKToolkit")
    return dst


# ===================================================================== configuration
def _get(ssm: Any, name: str) -> str | None:
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") == "ParameterNotFound":
            return None
        raise


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def read_local_config(path: str | os.PathLike[str] | None = None, *, environ: Mapping[str, str] | None = None, overlay: Path | None = DEFAULT_CONFIG, repo_root: Path = ROOT) -> dict[str, Any]:
    """The local untracked configuration, resolved like the platform's (module docstring)."""
    env = os.environ if environ is None else environ
    shared = Path(path) if path else Path(env[contract_bootstrap.CONFIG_ENV]) if env.get(contract_bootstrap.CONFIG_ENV) else SHARED_CONFIG
    data: dict[str, Any] = {}
    for candidate, drop in ((shared, _PLATFORM_ONLY_KEYS), (overlay, _PLATFORM_ONLY_KEYS[3:])):
        if candidate is None:
            continue
        candidate = Path(candidate).expanduser()
        if not candidate.is_file():
            continue
        if _inside(candidate, repo_root):
            raise BootstrapStop("configuration", f"bootstrap configuration {candidate} is inside the repository; keep it local and untracked (for example ~/.finplan/bootstrap.json)")
        doc = json.loads(candidate.read_text(encoding="utf-8"))
        data.update({k: v for k, v in doc.items() if k not in drop})
    for env_key, key in (("FINPLAN_ACCOUNT_ID", "account_id"), ("FINPLAN_PRIMARY_REGION", "primary_region"), ("FINPLAN_CODECONNECTION_ARN", "codeconnection_arn")):
        if env.get(env_key):
            data[key] = env[env_key]
    return data


def load_bootstrap_config(local: Mapping[str, Any], ssm: Any) -> BootstrapConfig:
    """The contract configuration for ``financemodel``; the CodeConnection defaults to the platform's (reused)."""
    from finplan_model.core.config import load_shared_config

    source = load_shared_config().get("source") or {}
    data = dict(local)
    data["repo"] = REPO
    data.setdefault("github_repository", str(source.get("repository") or "FilippoLentoni/FinanceModel"))
    data["pipeline_name"] = f"finplan-shared-{REPO}-pipeline"
    for key in _PLATFORM_ONLY_KEYS[5:]:
        data.pop(key, None)
    if not data.get("codeconnection_arn"):
        reused = _get(ssm, PLATFORM_CONNECTION_PARAMETER)
        if not reused:
            raise BootstrapStop("connection", f"no CodeConnection in the local configuration and none published at {PLATFORM_CONNECTION_PARAMETER}; bootstrap FinancialPlanning first or name the existing connection locally")
        data["codeconnection_arn"] = reused
    try:
        return BootstrapConfig.from_mapping(data)
    except ValueError as exc:
        raise BootstrapStop("configuration", str(exc)) from None


# ===================================================================== deploy
class ToolingDeployer:
    """The ``deployer`` callback of ``run_bootstrap``: read-only budget check, then the CDK deploy."""

    def __init__(self, assembly: Path, ssm: Any, *, region: str, dry_run_passed: bool, runner: Callable[..., Any] = subprocess.run, out: Callable[[str], None] = print, pricing: Any | None = None) -> None:
        self.assembly = assembly
        self.ssm = ssm
        self.pricing = pricing
        self.region = region
        self.dry_run_passed = dry_run_passed
        self.runner = runner
        self.out = out
        self.commands: list[list[str]] = []

    def command(self) -> list[str]:
        return ["npx", "--yes", "aws-cdk@2", "deploy", "--app", str(self.assembly), "--all", "--require-approval", "never", "--progress", "events", "--parameters", f"{TOOLING_STACK_NAME}:SourceDryRunPassed={'true' if self.dry_run_passed else 'false'}"]

    def __call__(self, stack_names: list[str]) -> None:
        if sorted(stack_names) != sorted(BOOTSTRAP_STACKS):
            raise BootstrapStop("deploy", f"the bootstrap deploys only {list(BOOTSTRAP_STACKS)}, got {stack_names}")
        allocation = _get(self.ssm, contract_budget.ALLOCATION_PARAMETER)
        if allocation is None:
            self.out(f"[WARN] {contract_budget.ALLOCATION_PARAMETER} is absent: FinanceModel pre-flight checks use the contract defaults until the FinancialPlanning bootstrap writes it (FinanceModel never writes it)")
        else:
            self.out(f"[OK] shared budget allocation (FinancialPlanning-owned, read-only): {allocation}")
        cmd = self.command()
        self.commands.append(cmd)
        self.out("running: " + " ".join(cmd))
        env = {**os.environ, "AWS_REGION": self.region, "AWS_DEFAULT_REGION": self.region, "CDK_DISABLE_VERSION_CHECK": "1"}
        proc = self.runner(cmd, cwd=str(ROOT), env=env)
        if int(getattr(proc, "returncode", 1)) != 0:
            raise BootstrapStop("deploy", "cdk deploy of the FinanceModel tooling stacks failed (CloudFormation rolls back automatically)")
        publish_tooling_role_names(self.ssm, out=self.out)
        if self.pricing is None:
            self.out("[WARN] no pricing client: instance prices not written; run scripts/instance_prices.py --write")
        else:
            from scripts.instance_prices import PriceUnavailable, ensure_instance_prices

            try:
                ensure_instance_prices(self.ssm, self.pricing, out=self.out)
            except PriceUnavailable as exc:
                # not a deploy failure: paid jobs stay refused (PRECONDITION_FAILED) until prices exist
                self.out(f"[WARN] instance prices not written: {exc}; run scripts/instance_prices.py --write later")
        report_operator_parameters(self.ssm, out=self.out)


def publish_tooling_role_names(ssm: Any, *, out: Callable[[str], None] = print) -> str:
    """Write the account-level tooling role names (contracts 1.0.0, D16) as the ``bootstrap`` writer."""
    value = ",".join(tooling_role_names())
    decision = contract_ssm.check_write(SHARED_ENFORCED_ROLES_PARAMETER, contract_ssm.Writer(REPO, "bootstrap"))
    problems = list(decision.reasons) + contract_ssm.validate_value(SHARED_ENFORCED_ROLES_PARAMETER, value)
    if problems:
        raise BootstrapStop("budget-roles", f"{SHARED_ENFORCED_ROLES_PARAMETER}: " + "; ".join(problems))
    if _get(ssm, SHARED_ENFORCED_ROLES_PARAMETER) == value:
        out(f"[OK] {SHARED_ENFORCED_ROLES_PARAMETER} already lists {value}")
        return value
    ssm.put_parameter(Name=SHARED_ENFORCED_ROLES_PARAMETER, Value=value, Type="String", Overwrite=True, Description="FinanceModel account-level tooling roles the platform budget action denies at 100% (written by the FinanceModel bootstrap)")
    out(f"[WROTE] {SHARED_ENFORCED_ROLES_PARAMETER} = {value} (re-run the FinancialPlanning bootstrap so its budget action covers them)")
    return value


def report_operator_parameters(ssm: Any, *, out: Callable[[str], None] = print) -> dict[str, bool]:
    """Read-only: which operator-only parameters are set (``integration-snapshot-id``; docs/operations.md)."""
    status: dict[str, bool] = {}
    for env in INTEGRATION_SNAPSHOT_ENVS:
        name = contract_ssm.build(env, REPO, "config", "integration-snapshot-id")
        present = bool((_get(ssm, name) or "").strip())
        status[name] = present
        if present:
            out(f"[OK] {name} is set")
        else:
            platform_key = contract_ssm.build(env, "financialplanning", "config", "integration-snapshot-id")
            out(f"[ACTION] {name} is not set: the {env} deployed suite uses {platform_key} (published by the platform's suite; real or synthetic) and fails if neither is set (docs/operations.md)")
    return status


# ===================================================================== orchestration
def _interactive_approve(plan: Plan) -> bool:  # pragma: no cover - interactive
    print("\nThe stacks and cost estimate above will be deployed under the user's in-principle approval of 2026-10-07.")
    return input("Type 'deploy' to deploy exactly these stacks, anything else to stop: ").strip() == "deploy"


def run(
    config: BootstrapConfig,
    clients: Clients,
    *,
    session_region: str | None,
    assembly: Path = DEFAULT_ASSEMBLY,
    bootstrap_dir: Path = BOOTSTRAP_ASSEMBLY,
    approve: Callable[[Plan], bool] = _interactive_approve,
    runner: Callable[..., Any] = subprocess.run,
    record_path: Path | None = None,
    sleep: Callable[[float], None] | None = None,
    usage_rules: Mapping[str, Any] | None = None,
    out: Callable[[str], None] = print,
) -> contract_bootstrap.Report:
    filtered = bootstrap_assembly(assembly, bootstrap_dir)
    record = Path(record_path).expanduser() if record_path else None
    passed = bool(record and record.is_file() and json.loads(record.read_text(encoding="utf-8")).get("deploy_stages_enabled") is True)
    deployer = ToolingDeployer(filtered, clients.ssm, region=config.primary_region, dry_run_passed=passed, runner=runner, out=out, pricing=clients.pricing)
    kwargs: dict[str, Any] = {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    return contract_bootstrap.run_bootstrap(config, clients, session_region=session_region, assembly_dir=filtered, approve=approve, deployer=deployer, out=out, record_path=record, usage_rules=usage_rules, **kwargs)


def make_clients(region: str) -> Clients:
    """boto3 clients from the default credential chain (instance role, environment, profile).
    Nothing here needs ``botocore[crt]``."""
    import boto3

    session = boto3.session.Session(region_name=region)
    return Clients(sts=session.client("sts"), codeconnections=session.client("codeconnections"), ssm=session.client("ssm"), codepipeline=session.client("codepipeline"), pricing=session.client("pricing", region_name="us-east-1"))


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - the authenticated run is a human step
    ap = argparse.ArgumentParser(description="One-time authenticated bootstrap of the FinanceModel pipeline. Read docs/bootstrap.md first.")
    ap.add_argument("--config", type=Path, default=None, help="local untracked configuration (default: $FINPLAN_BOOTSTRAP_CONFIG, else ~/.finplan/bootstrap.json, plus the optional ~/.finplan/financemodel-bootstrap.json overlay)")
    ap.add_argument("--assembly", type=Path, default=DEFAULT_ASSEMBLY)
    ap.add_argument("--record", type=Path, default=DRY_RUN_RECORD)
    args = ap.parse_args(argv)
    try:
        local = read_local_config(args.config)
    except (BootstrapStop, ValueError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    region = str(local.get("primary_region") or "us-east-2")
    clients = make_clients(region)
    args.record.expanduser().parent.mkdir(parents=True, exist_ok=True)
    try:
        config = load_bootstrap_config(local, clients.ssm)
        run(config, clients, session_region=region, assembly=args.assembly, record_path=args.record)
    except BootstrapStop as stop:
        print(f"[STOPPED] {stop.step}: {stop.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
