#!/usr/bin/env python3
"""Build the ``financemodel-cpu`` image ONCE in the build stage, push it and record its digest
(task 10.5; spec job-deployment-pipeline "Standard stages and immutable images"; DEP-03).

1. The base images named by the Dockerfile's ``ARG PYTHON_IMAGE=`` / ``ARG UV_IMAGE=`` defaults are
   pulled and **pinned by digest** (``<name>@sha256:...``); the pinned references are passed as
   build args and recorded in ``release-info.json`` (``base_images``).
2. ``docker build --platform linux/amd64`` (SageMaker ``ml.m5.xlarge`` is x86_64) from the repository
   root with ``container/Dockerfile`` (the ``.dockerignore`` allow-list keeps tests, fixtures and
   market data out of the image).
3. The image is tagged with the ``release_id`` (the repository's tags are immutable) and pushed to
   the account-level repository ``finplan-shared-financemodel-cpu-images``.
   Before the push, the built image must import the job entry point and its runtime
   dependencies (``finplan_contracts``, ``numpy``, ``scipy``, ``boto3``) with its own interpreter
   (:data:`IMAGE_IMPORT_CHECK`; platform lesson L3: code shipped without its dependencies failed at
   start). A failing check pushes nothing.
4. The pushed digest is read back from ECR; every environment's job definitions reference
   ``<repository>@<digest>``, never a tag, so beta, gamma and prod run the same bytes.

Every command goes through an injected runner and the ECR client is injected, so the unit suite runs
the whole sequence with fakes (no docker, no AWS).
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = Path("container/Dockerfile")
BASE_ARGS = ("PYTHON_IMAGE", "UV_IMAGE")
#: Modules the job image must import with only its own environment (dependencies installed).
IMAGE_IMPORTS = ("finplan_model.jobs.entrypoint", "finplan_model.strategies.optimizers", "finplan_contracts", "numpy", "scipy", "boto3", "jsonschema")
IMAGE_IMPORT_CHECK = "import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]; print('image imports ok')"
_ARG_RE = re.compile(r"^ARG\s+(?P<name>[A-Z_]+)=(?P<value>\S+)\s*$", re.MULTILINE)
_DIGEST_REF_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}\Z")

Runner = Callable[..., str]

__all__ = ["IMAGE_IMPORTS", "IMAGE_IMPORT_CHECK", "BuiltImage", "ImageError", "base_image_defaults", "build_and_push"]


class ImageError(RuntimeError):
    pass


@dataclass
class BuiltImage:
    repository: str
    tag: str
    digest: str
    base_images: dict[str, str] = field(default_factory=dict)


def _run(cmd: list[str], *, cwd: Path, input: str | None = None) -> str:  # noqa: A002
    # BuildKit: honours container/Dockerfile.dockerignore (the build-context allow-list)
    env = {**os.environ, "DOCKER_BUILDKIT": "1"}
    proc = subprocess.run(cmd, cwd=cwd, input=input, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise ImageError(f"{' '.join(cmd[:3])} ... failed ({proc.returncode}): {(proc.stdout + proc.stderr)[-2000:]}")
    return proc.stdout


def base_image_defaults(root: Path = ROOT) -> dict[str, str]:
    text = (root / DOCKERFILE).read_text(encoding="utf-8")
    found = {m["name"]: m["value"] for m in _ARG_RE.finditer(text)}
    missing = [a for a in BASE_ARGS if a not in found]
    if missing:
        raise ImageError(f"{DOCKERFILE} has no default for {missing}")
    return {a: found[a] for a in BASE_ARGS}


def pin_base_images(root: Path, run: Runner) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for arg, ref in base_image_defaults(root).items():
        if _DIGEST_REF_RE.match(ref):
            pinned[arg] = ref
            continue
        run(["docker", "pull", "--platform", "linux/amd64", ref], cwd=root)
        digests = run(["docker", "inspect", "--format", "{{range .RepoDigests}}{{println .}}{{end}}", ref], cwd=root).split()
        name = ref.rsplit(":", 1)[0] if ":" in ref.rsplit("/", 1)[-1] else ref
        match = next((d for d in digests if d.startswith(name + "@sha256:")), digests[0] if digests else "")
        if not _DIGEST_REF_RE.match(match):
            raise ImageError(f"could not pin {ref} by digest")
        pinned[arg] = match
    return pinned


def build_and_push(*, release_id: str, source_commit: str, account: str, region: str, repository: str, ecr: Any, root: Path = ROOT, run: Runner = _run) -> BuiltImage:
    if not re.fullmatch(r"rel_[0-9A-HJKMNP-TV-Z]{26}", release_id):
        raise ImageError("the image tag must be the release_id")
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    uri = f"{registry}/{repository}"
    base = pin_base_images(root, run)
    cmd = ["docker", "build", "--platform", "linux/amd64", "-f", str(DOCKERFILE), "-t", f"{uri}:{release_id}", "--label", f"org.opencontainers.image.revision={source_commit}", "--label", f"finplan.release-id={release_id}"]
    for arg, ref in sorted(base.items()):
        cmd += ["--build-arg", f"{arg}={ref}"]
    cmd.append(".")
    run(cmd, cwd=root)
    try:
        run(["docker", "run", "--rm", "--network", "none", "--platform", "linux/amd64", "--entrypoint", "python", f"{uri}:{release_id}", "-c", IMAGE_IMPORT_CHECK, *IMAGE_IMPORTS], cwd=root)
    except ImageError as exc:
        raise ImageError(f"the built image does not import its job entry point and dependencies (nothing pushed): {exc}") from None
    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    run(["docker", "login", "--username", user, "--password-stdin", registry], cwd=root, input=password)
    run(["docker", "push", f"{uri}:{release_id}"], cwd=root)
    details = ecr.describe_images(repositoryName=repository, imageIds=[{"imageTag": release_id}])["imageDetails"]
    if not details or not str(details[0].get("imageDigest", "")).startswith("sha256:"):
        raise ImageError("the pushed image has no digest in ECR")
    return BuiltImage(repository=repository, tag=release_id, digest=str(details[0]["imageDigest"]), base_images=base)


if __name__ == "__main__":  # pragma: no cover
    print(base_image_defaults())
    sys.exit(0)
