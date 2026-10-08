#!/usr/bin/env python3
"""Build-stage gates of the FinanceModel pipeline (task 1.2; contracts D6 build stage; DEP-03).

The build stage fails on any gate failure. Every contract gate reuses the pinned
``finplan-contracts`` package; nothing is copied or re-implemented.

=====================  =====  ==============================================================
gate                   stage  what
=====================  =====  ==============================================================
contracts-pin          pre    ``scripts/check_contracts_pin.py`` (version + wheel SHA-256 + pyproject
                              + uv.lock); ``--env gamma|prod`` refuses the 0.x pin
config                 pre    every ``config/<env>.json`` and ``config/shared.json`` validates
leak-scan              pre    ``finplan_contracts.leak_scan`` over the repository: account IDs,
                              ARNs with accounts, bucket names, endpoints, secrets (OWN-03, ENV-08)
copied-id              pre    ``finplan_contracts.copied_id``: no contract schema ``$id`` or schema
                              copy anywhere in the repository (CS-01)
conformance            pre    ``finplan-conformance conformance --mode consumer --expect-version <pin>``
live-perm-scan         pre    no live broker/exchange/wallet client in dependencies or source, and
                              ``finplan_contracts.live_perms`` over repository JSON/YAML (ENV-05, SIM-07)
provider-guard         pre    no market-data provider client (``yfinance`` ...) in dependencies, images
                              or source, no provider endpoint reference (DS-12)
fixture-check          pre    fixtures and price-like data files carry ``synthetic: true`` (DS-13)
unit                   pre    ``pytest tests/unit tests/contract`` under the offline harness (DEP-02)
ownership              post   ``finplan_contracts.ownership`` per synthesized template (OWN-01)
boundaries             post   permission boundary on every role (ENV-18), shared-resource rule (ENV-16)
live-perm-scan-synth   post   ``finplan_contracts.live_perms`` over every synthesized template (ENV-05)
pipeline-structure     post   ``finplan_contracts.pipeline_check`` (ENV-09, DEP-03) and
                              ``bootstrap.check_deploy_roles`` (ENV-12) on the tooling template
cost                   post   cost-allocation tags on every taggable resource; no always-on or
                              provisioned compute (endpoints, instances, NAT, load balancers,
                              provisioned tables or concurrency, API caches, GPU types)
lambda-bundle          post   every Lambda code asset carrying ``finplan_model`` is a complete bundle
                              from ``scripts/lambda_bundle.py`` (never a source-only package), built
                              for ``aarch64-manylinux_2_28`` with no non-arm64 shared object
=====================  =====  ==============================================================

Other task groups add gates by appending to :data:`GATES` (name, stage, function); a function
takes a :class:`GateContext` and returns a list of problems.

Usage: ``uv run python scripts/build_gates.py --stage pre|post|all [--assembly cdk.out] [--only GATE ...]``;
exit 1 when any gate fails. Runs offline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = ["GATES", "GateContext", "GateResult", "LIVE_TRADING_PACKAGES", "main", "run_gates", "templates_of"]

REPO = "financemodel"
#: Directories never scanned in addition to the contract defaults (build outputs, local caches).
EXTRA_EXCLUDES = frozenset({".build", "build-output", "cdk.out", "cdk.out.bootstrap", "vendor"})
#: Client libraries for live brokers, exchanges, payment or wallet APIs. FinanceModel is paper only
#: (spec paper-execution-simulator, "Paper only, never live"); none may be a dependency or import.
LIVE_TRADING_PACKAGES = frozenset(
    {
        "ccxt",
        "alpaca-trade-api",
        "alpaca-py",
        "alpaca",
        "coinbase",
        "coinbase-advanced-py",
        "cbpro",
        "ib-insync",
        "ib_insync",
        "ibapi",
        "robin-stocks",
        "robin_stocks",
        "python-binance",
        "binance",
        "krakenex",
        "pykrakenapi",
        "tda-api",
        "schwab-py",
        "oandapyv20",
        "web3",
    }
)


@dataclass
class GateContext:
    root: Path = ROOT
    assembly: Path | None = None
    env: str | None = None
    run_unit: bool = True
    extra_pytest_args: tuple[str, ...] = ()
    notes: dict[str, list[str]] = field(default_factory=dict)

    def note(self, gate: str, text: str) -> None:
        self.notes.setdefault(gate, []).append(text)


@dataclass
class GateResult:
    name: str
    problems: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def templates_of(assembly: Path) -> list[Path]:
    return sorted(assembly.rglob("*.template.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _call_main(fn: Callable[[list[str]], int], argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            rc = fn(argv)
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue()[-4000:]


def _excludes() -> frozenset[str]:
    from finplan_contracts import leak_scan

    return frozenset(leak_scan.DEFAULT_EXCLUDE_DIRS | EXTRA_EXCLUDES)


# ===================================================================== pre gates
def gate_contracts_pin(ctx: GateContext) -> list[str]:
    from scripts.check_contracts_pin import check

    return [f"contracts pin: {p}" for p in check(ctx.root, env=ctx.env)]


def gate_config(ctx: GateContext) -> list[str]:
    from finplan_model.core.config import ConfigError, load_all, load_shared_config

    try:
        load_all(ctx.root / "config")
        load_shared_config(ctx.root / "config")
    except ConfigError as exc:
        return [f"config: {p}" for p in exc.problems]
    except (OSError, ValueError) as exc:
        return [f"config: {exc}"]
    return []


def gate_leak_scan(ctx: GateContext) -> list[str]:
    from finplan_contracts import leak_scan

    n, findings = leak_scan.scan_paths([ctx.root], exclude_dirs=_excludes())
    ctx.note("leak-scan", f"{n} files scanned")
    return [f"leak: {f}" for f in findings]


def gate_copied_id(ctx: GateContext) -> list[str]:
    from finplan_contracts import copied_id

    found = copied_id.scan_tree(ctx.root, exclude_dirs=copied_id.EXCLUDE_DIRS | EXTRA_EXCLUDES | {".claude"})
    return [f"copied contract schema: {c}" for c in found]


def gate_conformance(ctx: GateContext) -> list[str]:
    from finplan_contracts import conformance

    from scripts.check_contracts_pin import load_pin

    version = load_pin(ctx.root)["version"]
    rc, out = _call_main(conformance.main, ["--mode", "consumer", "--expect-version", version])
    if rc == 0:
        return []
    lines = out.strip().splitlines()
    return ["conformance (consumer mode) failed: " + (lines[-1] if lines else "no output")]


_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def live_trading_problems(root: Path) -> list[str]:
    """Live broker/exchange/wallet clients in dependency files, the lock, Dockerfiles or source."""
    forbidden = {_norm(p) for p in LIVE_TRADING_PACKAGES}
    problems: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        deps = list(data.get("project", {}).get("dependencies", []))
        for group in data.get("project", {}).get("optional-dependencies", {}).values():
            deps += list(group)
        for group in data.get("dependency-groups", {}).values():
            deps += [d for d in group if isinstance(d, str)]
        for dep in deps:
            name = _norm(re.split(r"[\s\[<>=!~;]", dep, maxsplit=1)[0])
            if name in forbidden:
                problems.append(f"pyproject.toml declares live-trading client {name}")
    lock = root / "uv.lock"
    if lock.is_file():
        for pkg in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
            if _norm(str(pkg.get("name", ""))) in forbidden:
                problems.append(f"uv.lock resolves live-trading client {pkg['name']}")
    excludes = _excludes()
    for path in sorted(root.rglob("*")):
        if any(part in excludes for part in path.relative_to(root).parts):
            continue
        if path.is_file() and (path.name.startswith("Dockerfile") or path.name.startswith("requirements")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                for tok in re.findall(r"[A-Za-z0-9_.-]+", line):
                    if _norm(tok) in forbidden:
                        problems.append(f"{path.relative_to(root)} installs live-trading client {tok}")
        elif path.is_file() and path.suffix == ".py" and "tests" not in path.relative_to(root).parts:
            for mod in _IMPORT_RE.findall(path.read_text(encoding="utf-8", errors="replace")):
                if _norm(mod) in forbidden:
                    problems.append(f"{path.relative_to(root)} imports live-trading client {mod}")
    return problems


def gate_live_perms(ctx: GateContext) -> list[str]:
    from finplan_contracts import live_perms

    problems = live_trading_problems(ctx.root)
    excludes = _excludes() | {".claude", "openspec"}
    docs = [p for p in sorted(ctx.root.rglob("*")) if p.is_file() and p.suffix in (".json", ".yaml", ".yml") and not any(part in excludes for part in p.relative_to(ctx.root).parts)]
    n, findings = live_perms.scan_paths(docs)
    ctx.note("live-perm-scan", f"{n} documents scanned")
    return problems + [f"live-financial permission: {f}" for f in findings]


def gate_unit(ctx: GateContext) -> list[str]:
    if not ctx.run_unit:
        ctx.note("unit", "skipped by request")
        return []
    # FINPLAN_RELEASE_BUILD=0: the unit suite synthesizes offline from the source tree even inside the
    # release build (which sets FINPLAN_RELEASE_BUILD=1 for its own synth); FINPLAN_TARGET_ENV is
    # cleared so the offline harness always applies.
    env = {k: v for k, v in os.environ.items() if k != "FINPLAN_TARGET_ENV"} | {"FINPLAN_RELEASE_BUILD": "0"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/unit", "tests/contract", *ctx.extra_pytest_args], cwd=ctx.root, capture_output=True, text=True, check=False, env=env)
    if proc.returncode == 0:
        return []
    lines = (proc.stdout + proc.stderr).strip().splitlines() or ["?"]
    return ["unit/contract tests failed: " + lines[-1]]


def gate_provider_guard(ctx: GateContext) -> list[str]:
    """No market-data provider client (``yfinance`` ...) in dependencies, images or source, and no
    provider endpoint reference (research-datasets DS-12; task 3.7)."""
    from finplan_model.datasets.provider_guard import DEFAULT_EXCLUDES, provider_problems

    return [f"provider guard: {p}" for p in provider_problems(ctx.root, excludes=DEFAULT_EXCLUDES | EXTRA_EXCLUDES)]


def gate_fixture_check(ctx: GateContext) -> list[str]:
    """Every fixture or price-like data file carries ``synthetic: true`` (DS-13; task 3.7)."""
    from finplan_model.datasets.provider_guard import DEFAULT_EXCLUDES, fixture_problems

    return [f"fixture check: {p}" for p in fixture_problems(ctx.root, excludes=DEFAULT_EXCLUDES | EXTRA_EXCLUDES)]


# ===================================================================== post gates
def _need_assembly(ctx: GateContext) -> Path:
    if ctx.assembly is None or not (ctx.assembly / "manifest.json").is_file():
        raise FileNotFoundError("post-synth gates need a synthesized cloud assembly (--assembly)")
    return ctx.assembly


def gate_ownership(ctx: GateContext) -> list[str]:
    from finplan_contracts.ownership import check_template

    problems: list[str] = []
    templates = templates_of(_need_assembly(ctx))
    for path in templates:
        report = check_template(_load(path), REPO, name=path.name).to_dict()
        problems += [f"ownership {path.name}: {p['logical_id']} ({p['resource_type']}): {p['message']}" for p in report["problems"]]
    ctx.note("ownership", f"{len(templates)} templates checked")
    return problems


def gate_boundaries(ctx: GateContext) -> list[str]:
    from finplan_contracts.boundaries import check_role_boundaries, check_shared_resources

    problems: list[str] = []
    for path in templates_of(_need_assembly(ctx)):
        t = _load(path)
        problems += [f"boundary {path.name}: {f}" for f in check_role_boundaries(t)]
        problems += [f"shared {path.name}: {f}" for f in check_shared_resources(t, repo=REPO)]
    return problems


def gate_live_perms_synth(ctx: GateContext) -> list[str]:
    from finplan_contracts import live_perms

    n, findings = live_perms.scan_paths(templates_of(_need_assembly(ctx)))
    ctx.note("live-perm-scan-synth", f"{n} templates scanned")
    return [f"live-financial permission: {f}" for f in findings]


def gate_pipeline_structure(ctx: GateContext) -> list[str]:
    from finplan_contracts.bootstrap import check_deploy_roles
    from finplan_contracts.pipeline_check import check_pipeline_template

    problems: list[str] = []
    pipelines = 0
    for path in templates_of(_need_assembly(ctx)):
        t = _load(path)
        if not any(isinstance(r, dict) and r.get("Type") == "AWS::CodePipeline::Pipeline" for r in (t.get("Resources") or {}).values()):
            continue
        pipelines += 1
        problems += [f"pipeline {path.name}: {f}" for f in check_pipeline_template(t)]
        problems += [f"deploy roles {path.name}: {p}" for p in check_deploy_roles(t)]
    if pipelines == 0:
        problems.append("no pipeline template in the assembly (synthesize all three environments)")
    return problems


#: Resource types (and properties) that bill while idle; FinanceModel phase 1 has none (design: no always-on compute).
ALWAYS_ON_TYPES = frozenset(
    {
        "AWS::EC2::Instance",
        "AWS::EC2::NatGateway",
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        "AWS::ElasticLoadBalancing::LoadBalancer",
        "AWS::SageMaker::Endpoint",
        "AWS::SageMaker::EndpointConfig",
        "AWS::SageMaker::NotebookInstance",
        "AWS::RDS::DBInstance",
        "AWS::RDS::DBCluster",
        "AWS::ECS::Service",
        "AWS::EKS::Cluster",
        "AWS::ElastiCache::CacheCluster",
        "AWS::OpenSearchService::Domain",
    }
)
_TAG_KEYS = ("project", "owner-repo", "environment", "logical-role")
#: CDK helpers the ownership matrix attributes to their parent construct (an own ``logical-role`` tag
#: would bypass that attribution); they still need the other cost tags.
_PARENT_ATTRIBUTED = frozenset({"AWS::Logs::LogGroup"})


def cost_problems(template: dict[str, Any], name: str) -> list[str]:
    problems: list[str] = []
    for lid, res in sorted((template.get("Resources") or {}).items()):
        if not isinstance(res, dict):
            continue
        rtype, props = str(res.get("Type")), res.get("Properties") or {}
        if rtype in ALWAYS_ON_TYPES:
            problems.append(f"{name}: {lid} ({rtype}) is always-on compute")
        if rtype == "AWS::EC2::VPCEndpoint" and props.get("VpcEndpointType") == "Interface":
            problems.append(f"{name}: {lid} is an interface endpoint (billed per hour)")
        if rtype == "AWS::DynamoDB::Table" and props.get("BillingMode") != "PAY_PER_REQUEST":
            problems.append(f"{name}: {lid} is a provisioned table")
        if rtype == "AWS::Lambda::Version" and props.get("ProvisionedConcurrencyConfig"):
            problems.append(f"{name}: {lid} has provisioned concurrency")
        if rtype == "AWS::ApiGateway::Stage" and props.get("CacheClusterEnabled"):
            problems.append(f"{name}: {lid} enables an API cache")
        if re.search(r"ml\.(p[0-9]|g[0-9]|inf|trn)", json.dumps(props)):
            problems.append(f"{name}: {lid} names a GPU or accelerator instance type")
        tags = props.get("Tags")
        if isinstance(tags, list):
            keys = {t.get("Key") for t in tags if isinstance(t, dict)}
            missing = [k for k in _TAG_KEYS if k not in keys and not (k == "logical-role" and rtype in _PARENT_ATTRIBUTED and "logical-role" not in keys)]
            if missing:
                problems.append(f"{name}: {lid} ({rtype}) lacks cost tags {missing}")
    return problems


def gate_cost(ctx: GateContext) -> list[str]:
    problems: list[str] = []
    for path in templates_of(_need_assembly(ctx)):
        problems += cost_problems(_load(path), path.name)
    return problems


def gate_lambda_bundle(ctx: GateContext) -> list[str]:
    from infra.stacks.lambda_code import bundle_problems

    problems: list[str] = []
    assembly = _need_assembly(ctx)
    for asset in sorted(p for p in assembly.rglob("asset.*") if p.is_dir()):
        if not (asset / "finplan_model").is_dir():
            continue
        problems += [f"Lambda code {asset.name}: {p}" for p in bundle_problems(asset)]
        manifest = asset / "bundle-manifest.json"
        if manifest.is_file():
            from scripts.lambda_bundle import LAMBDA_PLATFORM, foreign_binaries

            platform = json.loads(manifest.read_text(encoding="utf-8")).get("python_platform")
            if platform != LAMBDA_PLATFORM:
                problems.append(f"Lambda code {asset.name}: built for {platform}, the functions run on {LAMBDA_PLATFORM} (arm64)")
            problems += [f"Lambda code {asset.name}: {b} is not an arm64 binary" for b in foreign_binaries(asset)]
    return problems


Gate = tuple[str, str, Callable[[GateContext], list[str]]]
GATES: list[Gate] = [
    ("contracts-pin", "pre", gate_contracts_pin),
    ("config", "pre", gate_config),
    ("leak-scan", "pre", gate_leak_scan),
    ("copied-id", "pre", gate_copied_id),
    ("conformance", "pre", gate_conformance),
    ("live-perm-scan", "pre", gate_live_perms),
    ("provider-guard", "pre", gate_provider_guard),
    ("fixture-check", "pre", gate_fixture_check),
    ("unit", "pre", gate_unit),
    ("ownership", "post", gate_ownership),
    ("boundaries", "post", gate_boundaries),
    ("live-perm-scan-synth", "post", gate_live_perms_synth),
    ("pipeline-structure", "post", gate_pipeline_structure),
    ("cost", "post", gate_cost),
    ("lambda-bundle", "post", gate_lambda_bundle),
]


def run_gates(ctx: GateContext, *, stage: str = "all", only: Iterable[str] | None = None) -> list[GateResult]:
    selected = set(only or ())
    unknown = selected - {g[0] for g in GATES}
    if unknown:
        raise ValueError(f"unknown gates: {sorted(unknown)}")
    results: list[GateResult] = []
    for name, gstage, fn in GATES:
        if selected and name not in selected:
            continue
        if not selected and stage != "all" and gstage != stage:
            continue
        try:
            problems = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a crashing gate is a failing gate
            problems = [f"{name} could not run: {type(exc).__name__}: {exc}"]
        results.append(GateResult(name, problems, ctx.notes.get(name, [])))
    return results


def report(results: list[GateResult], out: Callable[[str], None] = print) -> bool:
    for r in results:
        out(f"[{'PASS' if r.ok else 'FAIL'}] {r.name}" + (f" ({'; '.join(r.notes)})" if r.notes else ""))
        for p in r.problems[:200]:
            out(f"    {p}")
    return all(r.ok for r in results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the FinanceModel build-stage gates.")
    ap.add_argument("--stage", choices=("pre", "post", "all"), default="all")
    ap.add_argument("--only", nargs="*", help="run only these gates")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--assembly", type=Path, help="synthesized cloud assembly (post gates)")
    ap.add_argument("--env", choices=("beta", "gamma", "prod"), help="deployment target for the contracts-pin gate")
    ap.add_argument("--skip-unit", action="store_true")
    args = ap.parse_args(argv)
    ctx = GateContext(root=args.root.resolve(), assembly=args.assembly.resolve() if args.assembly else None, env=args.env, run_unit=not args.skip_unit)
    return 0 if report(run_gates(ctx, stage=args.stage, only=args.only)) else 1


if __name__ == "__main__":
    sys.exit(main())
