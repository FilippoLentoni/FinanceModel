#!/usr/bin/env python3
"""Actions of each environment stage (tasks 10.4, 10.5; DEP-04, DEP-05). They run in the
per-environment stage project from the BuildOutput artifact only (never the source checkout, never
``cdk synth``):

``precheck`` (gamma and prod, before any deploy)
    Promotion check: the pinned ``finplan-contracts`` must be allowed in the target environment
    (``scripts/check_contracts_pin.py --env``: 0.x is beta-only) and ``release-info.json`` must
    carry a digest-pinned image. A failure stops the stage before anything is deployed.
``publish``
    :func:`scripts.release.publish_release` with the registry seeder; prod first reads the manual
    approval (approver, time) of this pipeline execution.
``tests``
    The environment suite: ``integration-beta`` (beta) and ``gamma`` (gamma) run
    ``tests/integration``; ``smoke`` (prod) runs ``tests/smoke``. ``FINPLAN_TARGET_ENV`` and
    ``FINPLAN_SUITE`` tell the tests where they run (and switch the offline harness off for them).
    Every suite must execute at least one test: zero executed tests is a false pass and fails
    (lesson of the platform's first pipeline run).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.release import ReleaseInfo, approval_record, publish_release, registry_seeder  # noqa: E402

__all__ = ["SUITES", "main", "precheck_action", "publish_action", "suite_counts", "tests_action"]

SUITES: dict[str, tuple[str, list[str]]] = {
    "beta": ("integration-beta", ["tests/integration"]),
    "gamma": ("gamma", ["tests/integration"]),
    "prod": ("smoke", ["tests/smoke"]),
}


def precheck_action(env: str, info: ReleaseInfo, *, root: Path = ROOT) -> list[str]:
    """Problems that stop promotion to ``env`` (empty: promote)."""
    from scripts.check_contracts_pin import check

    problems = [f"contracts pin: {p}" for p in check(root, env=env)]
    if not (info.image_digest or "").startswith("sha256:"):
        problems.append("release-info.json carries no digest-pinned financemodel-cpu image")
    return problems


def publish_action(env: str, info: ReleaseInfo, *, ssm: Any, cfn: Any, s3: Any | None, codepipeline: Any | None, store: str | None, pipeline_name: str, execution_id: str | None, account: str) -> dict[str, Any]:
    approval = None
    if env == "prod":
        if codepipeline is None or not execution_id:
            raise RuntimeError("the prod manifest needs the pipeline execution ID to read the approval")
        approval = approval_record(codepipeline, pipeline_name, execution_id)
    seed = registry_seeder(s3, env, info.image_digest) if (s3 is not None and info.image_digest) else None
    return publish_release(info, env, ssm=ssm, cfn=cfn, account=account, s3=s3, store_bucket=store, approval=approval, seed=seed)


def suite_counts(junit_xml: Path) -> dict[str, int]:
    root = ET.parse(junit_xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = {k: 0 for k in ("tests", "failures", "errors", "skipped")}
    for s in suites:
        for k in total:
            total[k] += int(s.get(k, 0))
    total["executed"] = total["tests"] - total["skipped"]
    return total


def tests_action(env: str, *, root: Path = ROOT, release_id: str | None = None, run: Callable[..., Any] = subprocess.run, environ: Mapping[str, str] | None = None, out: Callable[[str], None] = print) -> int:
    suite, paths = SUITES[env]
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        env_vars = {**(environ if environ is not None else os.environ), "FINPLAN_TARGET_ENV": env, "FINPLAN_SUITE": suite}
        if release_id:
            env_vars["FINPLAN_RELEASE_ID"] = release_id
        proc = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *paths], cwd=root, env=env_vars)
        rc = int(getattr(proc, "returncode", 1))
        counts = suite_counts(junit) if junit.is_file() else {"tests": 0, "executed": 0, "failures": 0, "errors": 0, "skipped": 0}
    out(f"{suite}: {counts}")
    if rc == 5 or counts["executed"] < 1:
        out(f"FAIL: the {suite} suite executed no test (a stage with zero executed tests is a false pass)")
        return 1
    return 0 if rc == 0 else 1


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CodeBuild entry point (needs AWS)
    ap = argparse.ArgumentParser(description="FinanceModel stage actions: promotion check, release publishing or environment tests.")
    ap.add_argument("action", choices=("precheck", "publish", "tests"))
    ap.add_argument("--env", required=True, choices=("beta", "gamma", "prod"))
    ap.add_argument("--release-info", type=Path, default=ROOT / "release-info.json")
    ap.add_argument("--pipeline-execution-id", default=None)
    ap.add_argument("--store", default=os.environ.get("FINPLAN_PIPELINE_STORE"))
    args = ap.parse_args(argv)
    info = ReleaseInfo.load(args.release_info)
    if args.action == "precheck":
        problems = precheck_action(args.env, info)
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1 if problems else 0
    if args.action == "tests":
        return tests_action(args.env, release_id=info.release_id)
    import boto3

    from finplan_model.core.aws_clients import s3_client
    from infra.stacks.naming import PIPELINE_NAME

    session = boto3.session.Session(region_name=info.region)
    account = session.client("sts").get_caller_identity()["Account"]
    manifest = publish_action(
        args.env,
        info,
        ssm=session.client("ssm"),
        cfn=session.client("cloudformation"),
        s3=s3_client(info.region, session=session),
        codepipeline=session.client("codepipeline"),
        store=args.store,
        pipeline_name=PIPELINE_NAME,
        execution_id=args.pipeline_execution_id,
        account=account,
    )
    print(f"published {args.env} manifest for {manifest['release_id']} (previous {manifest['previous_release_id']}; outputs {sorted(manifest['outputs'])})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
