#!/usr/bin/env python3
"""The FinanceModel pipeline's Build stage (task 10.5; DEP-01, DEP-02, DEP-03; contracts D6).

Normal build (``--rollback-to`` empty or ``none``):

1. pre-synth gates (:mod:`scripts.build_gates`): contract pin, configuration, leak scan, copied-id,
   consumer conformance, live-permission scan, provider guard, fixture check, unit and contract
   tests under the offline harness (no network, no AWS, **no SageMaker call**: DEP-02);
2. the Lambda bundle (:mod:`scripts.lambda_bundle`), then ``cdk synth`` once
   (:func:`scripts.synth.synth`) in **release mode**: a function without a complete bundle fails;
3. post-synth gates: ownership (zero problems), boundaries, live-permission scan, pipeline
   structure and scoped deploy roles, cost tags and always-on resources, Lambda bundle;
4. the ``financemodel-cpu`` image, built **once**, pushed and pinned by digest
   (:mod:`scripts.container_image`);
5. BuildOutput: the cloud assembly, ``release-info.json`` (new ``release_id``, artifact digest over
   the assembly and the image digest, pinned contract version and digest, served majors, image) and
   the files the post-deploy actions need (never a rebuild);
6. the file assets go to the pipeline store and BuildOutput to the release ledger.

Any failure raises :class:`BuildFailed` before anything is written to ``--out``: a failing gate
produces **no artifact**. Rollback (``--rollback-to rel_...``): the recorded release's stored
BuildOutput is re-emitted (digest re-verified, ``rollback: true``); nothing is rebuilt.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts import build_gates  # noqa: E402
from scripts.release import ReleaseInfo, assembly_digest, contract_pin, fetch_build_output, mint_release_id, store_build_output  # noqa: E402

__all__ = ["PACKAGE_PATHS", "BuildFailed", "main", "run_build", "run_rollback"]

#: Copied into BuildOutput for the post-deploy actions (they never read the source checkout).
PACKAGE_PATHS = ("pyproject.toml", "uv.lock", "README.md", "contracts-pin.json", "cdk.json", "vendor", "src", "config", "scripts", "tests", "infra")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", "cdk.out")
_SHA = re.compile(r"^[0-9a-f]{40}$")
ImageFn = Callable[[str, str], Any]


class BuildFailed(RuntimeError):
    pass


def _region(root: Path) -> str:
    env = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if env:
        return env
    return str(json.loads((root / "config" / "shared.json").read_text(encoding="utf-8"))["region"])


def _served_majors(root: Path) -> list[int]:
    from finplan_model.core.config import load_config

    return sorted({int(m) for env in ("beta", "gamma", "prod") for m in load_config(env, root / "config").served_contract_majors})


def _default_synth(out: Path) -> Path:
    from scripts.synth import synth

    return synth(out)


def _default_bundle(root: Path, out: Path) -> dict[str, Any]:
    from scripts.lambda_bundle import build_bundle

    return build_bundle(root, out, replace=True)


def run_build(
    root: Path,
    out: Path,
    *,
    source_commit: str,
    s3: Any | None = None,
    store: str | None = None,
    account: str | None = None,
    region: str | None = None,
    run_unit: bool = True,
    synth_fn: Callable[[Path], Path] = _default_synth,
    bundle_fn: Callable[[Path, Path], dict[str, Any]] = _default_bundle,
    image_fn: ImageFn | None = None,
    gates: tuple[str, ...] = ("pre", "post"),
    only: tuple[str, ...] | None = None,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> ReleaseInfo:
    from scripts.synth import release_environment

    if not _SHA.match(source_commit):
        raise BuildFailed("source commit must be the 40-character commit ID from the Source stage")
    if out.exists():
        raise BuildFailed(f"{out} already exists; the build output is produced once per build")
    region = region or _region(root)
    work = root / ".build" / "stage"
    bundle = root / ".build" / "lambda-bundle"
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(bundle, ignore_errors=True)
    work.mkdir(parents=True)
    release_id = mint_release_id(now)
    try:
        ctx = build_gates.GateContext(root=root, run_unit=run_unit)
        if "pre" in gates and not _gates_ok(ctx, "pre", only, log):
            raise BuildFailed("pre-synth gates failed; no artifact produced")
        from infra.stacks.lambda_code import SourceOnlyCodeError
        from scripts.lambda_bundle import BundleError

        try:
            manifest = bundle_fn(root, bundle)
        except BundleError as exc:
            raise BuildFailed(f"Lambda bundle build failed; no artifact produced: {exc}") from exc
        log(f"lambda bundle: {manifest.get('unzipped_bytes', 0) / 2**20:.1f} MiB unzipped, {manifest.get('files', 0)} files, {manifest.get('python_platform')}")
        try:
            with release_environment(bundle):
                assembly = synth_fn(work / "cdk.out")
        except SourceOnlyCodeError as exc:
            raise BuildFailed(f"release synth refused source-only Lambda code; no artifact produced: {exc}") from exc
        ctx.assembly = assembly
        if "post" in gates and not _gates_ok(ctx, "post", only, log):
            raise BuildFailed("post-synth gates failed; no artifact produced")
        image = image_fn(release_id, source_commit) if image_fn is not None else None
        for rel in PACKAGE_PATHS:
            src = root / rel
            if src.is_dir():
                shutil.copytree(src, work / rel, ignore=_IGNORE)
            elif src.is_file():
                shutil.copy2(src, work / rel)
        version, digest = contract_pin(root)
        image_digest = getattr(image, "digest", None)
        info = ReleaseInfo(
            release_id=release_id,
            source_commit=source_commit,
            artifact_digest=assembly_digest(assembly, image_digest),
            contract_version=version,
            contract_digest=digest,
            served_contract_majors=_served_majors(root),
            region=region,
            built_at=(now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            image_repository=getattr(image, "repository", None),
            image_digest=image_digest,
            base_images=dict(getattr(image, "base_images", {}) or {}),
        )
        (work / "release-info.json").write_text(info.to_json(), encoding="utf-8")
        if s3 is not None:
            if not (store and account):
                raise BuildFailed("publishing needs the pipeline store name and the account")
            from scripts.publish_assets import publish

            for p in publish(assembly, s3, account=account, region=region):
                log(f"asset {'uploaded' if p.uploaded else 'present'}: {p.key}")
            store_build_output(s3, store, info, work)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work), str(out))
        log(f"release {info.release_id} digest {info.artifact_digest} image {info.image_digest}")
        return info
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(bundle, ignore_errors=True)
        with contextlib.suppress(OSError):
            work.parent.rmdir()


def _names(stage: str) -> set[str]:
    return {name for name, gstage, _ in build_gates.GATES if gstage == stage}


def _gates_ok(ctx: build_gates.GateContext, stage: str, only: tuple[str, ...] | None, log: Callable[[str], None]) -> bool:
    selected = None if only is None else [g for g in only if g in _names(stage)]
    if selected == []:
        return True
    return build_gates.report(build_gates.run_gates(ctx, stage=stage, only=selected), log)


def run_rollback(release_id: str, out: Path, *, s3: Any, store: str, log: Callable[[str], None] = print) -> ReleaseInfo:
    if out.exists():
        raise BuildFailed(f"{out} already exists")
    info = fetch_build_output(s3, store, release_id, out)
    log(f"rollback: re-emitting stored release {release_id} (digest {info.artifact_digest}); nothing rebuilt")
    return info


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CodeBuild entry point (needs AWS)
    ap = argparse.ArgumentParser(description="FinanceModel pipeline Build stage.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--rollback-to", default="none")
    ap.add_argument("--store", default=os.environ.get("FINPLAN_PIPELINE_STORE"))
    ap.add_argument("--image-repository", default=os.environ.get("FINPLAN_IMAGE_REPOSITORY"))
    ap.add_argument("--no-publish", action="store_true", help="local run: no image push, no asset publishing, no release ledger")
    args = ap.parse_args(argv)
    s3 = account = image_fn = None
    region = _region(ROOT)
    if not args.no_publish:
        import boto3

        from scripts.container_image import build_and_push
        from scripts.publish_assets import account_from_build_arn

        from finplan_model.core.aws_clients import s3_client

        s3 = s3_client(region)
        account = account_from_build_arn(os.environ.get("CODEBUILD_BUILD_ARN"))
        ecr = boto3.client("ecr", region_name=region)
        repo = args.image_repository

        def image_fn(release_id: str, commit: str) -> Any:
            return build_and_push(release_id=release_id, source_commit=commit, account=str(account), region=region, repository=str(repo), ecr=ecr)

    try:
        if args.rollback_to and args.rollback_to not in ("none", ""):
            if s3 is None or not args.store:
                raise BuildFailed("rollback needs the pipeline store")
            run_rollback(args.rollback_to, args.out, s3=s3, store=args.store)
        else:
            run_build(ROOT, args.out, source_commit=args.source_commit, s3=s3, store=args.store, account=account, region=region, image_fn=image_fn)
    except BuildFailed as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
