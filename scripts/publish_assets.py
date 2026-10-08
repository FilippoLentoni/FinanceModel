#!/usr/bin/env python3
"""Publish the cloud assembly's file assets to the FinanceModel pipeline store (task 10.5; build stage).

The environment stacks are synthesized with :func:`infra.stacks.tooling.deployment_synthesizer`, so
the Lambda code asset is addressed as ``assets/<sha256>.zip`` in
``finplan-shared-financemodel-pipeline-store-<account>`` and the pipeline's CloudFormation actions need
nothing else. Every ``*.assets.json`` manifest (nested stage assemblies included) is read and each
**file** asset uploaded once: deterministic zips (sorted entries, fixed timestamps and modes),
write-once (``If-None-Match: *``; the key carries the content hash, an existing key is skipped).
Keys under ``bootstrap/`` (the staged tooling template) are published only by the bootstrap.
Container images are not CDK assets here: the build stage builds and pushes ``financemodel-cpu``
itself and records its digest (``scripts/container_image.py``); a docker-image asset is refused.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from infra.stacks import naming as n  # noqa: E402
from infra.stacks.tooling import BOOTSTRAP_PREFIX  # noqa: E402

__all__ = ["PublishError", "Published", "account_from_build_arn", "deterministic_zip", "iter_asset_manifests", "planned_uploads", "publish"]

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class PublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class Published:
    asset_id: str
    bucket: str
    key: str
    size: int
    uploaded: bool


def deterministic_zip(directory: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(directory).as_posix(), date_time=ZIP_EPOCH)
            mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    return buf.getvalue()


def iter_asset_manifests(assembly: Path) -> Iterator[Path]:
    yield from sorted(assembly.rglob("*.assets.json"))


def _resolve(value: str, account: str, region: str) -> str:
    return value.replace("${AWS::AccountId}", account).replace("${AWS::Region}", region).replace("${AWS::Partition}", "aws")


def planned_uploads(assembly: Path, *, account: str, region: str, skip_prefixes: tuple[str, ...] = (BOOTSTRAP_PREFIX,)) -> list[tuple[str, Path, str, str, str]]:
    store = n.pipeline_store_bucket_name(account)
    out: dict[tuple[str, str], tuple[str, Path, str, str, str]] = {}
    for manifest in iter_asset_manifests(assembly):
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        if doc.get("dockerImages"):
            raise PublishError(f"{manifest.name}: container-image CDK assets are not used; the build stage pushes financemodel-cpu by digest")
        for asset_id, asset in sorted((doc.get("files") or {}).items()):
            src = asset.get("source") or {}
            path = (manifest.parent / str(src.get("path", ""))).resolve()
            packaging = str(src.get("packaging", "file"))
            if packaging not in ("zip", "file"):
                raise PublishError(f"{manifest.name}: asset {asset_id} has unsupported packaging {packaging!r}")
            for dest in (asset.get("destinations") or {}).values():
                bucket = _resolve(str(dest["bucketName"]), account, region)
                key = _resolve(str(dest["objectKey"]), account, region)
                if bucket != store:
                    raise PublishError(f"{manifest.name}: asset {asset_id} targets a bucket outside the FinanceModel pipeline store")
                if key.startswith(skip_prefixes):
                    continue
                out[(bucket, key)] = (asset_id, path, packaging, bucket, key)
    return [out[k] for k in sorted(out)]


def _exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", "")) if hasattr(exc, "response") else ""
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def publish(assembly: str | os.PathLike[str], s3: Any, *, account: str, region: str) -> list[Published]:
    results: list[Published] = []
    for asset_id, path, packaging, bucket, key in planned_uploads(Path(assembly), account=account, region=region):
        if _exists(s3, bucket, key):
            results.append(Published(asset_id, bucket, key, 0, False))
            continue
        if packaging == "zip":
            if not path.is_dir():
                raise PublishError(f"asset {asset_id}: source directory is missing from the assembly")
            body = deterministic_zip(path)
        else:
            if not path.is_file():
                raise PublishError(f"asset {asset_id}: source file is missing from the assembly")
            body = path.read_bytes()
        s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
        results.append(Published(asset_id, bucket, key, len(body), True))
    return results


def account_from_build_arn(arn: str | None) -> str | None:
    """``arn:aws:codebuild:<region>:<account>:build/...`` -> account (the build's own account)."""
    parts = (arn or "").split(":")
    return parts[4] if len(parts) > 5 and parts[2] == "codebuild" else None


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs AWS
    ap = argparse.ArgumentParser(description="Publish file assets of a cloud assembly to the FinanceModel pipeline store.")
    ap.add_argument("assembly")
    ap.add_argument("--account", default=None)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    args = ap.parse_args(argv)
    account = args.account or account_from_build_arn(os.environ.get("CODEBUILD_BUILD_ARN"))
    if not account or not args.region:
        print("account and region are required (--account/--region or the CodeBuild environment)", file=sys.stderr)
        return 2
    from finplan_model.core.aws_clients import s3_client

    for p in publish(args.assembly, s3_client(args.region), account=account, region=args.region):
        print(f"{'uploaded' if p.uploaded else 'present '} {p.key}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
