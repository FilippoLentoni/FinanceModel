"""Code asset of the FinanceModel control-plane Lambdas (task 10.3; mirrors the platform fix of 2026-10-07).

The build stage builds **one** bundle (``scripts/lambda_bundle.py``): the locked runtime closure from
``uv.lock`` (``uv export --frozen --no-dev``, the vendored digest-pinned contract wheel included),
the ``finplan_model`` package and ``config/``, for CPython 3.12 on ``aarch64-manylinux_2_28``
(the functions run on ``arm64``, as the platform's do). It sets ``FINPLAN_LAMBDA_BUNDLE_DIR``.

**Release mode** (``FINPLAN_RELEASE_BUILD=1``, or any CodeBuild build unless
``FINPLAN_RELEASE_BUILD=0``): a missing or incomplete bundle FAILS the synth (:class:`SourceOnlyCodeError`),
because a source-only package fails at init (``No module named 'finplan_contracts'``; the platform
shipped exactly that once). Outside release mode (local synth, tests) the ``src/`` tree is packaged,
which is fine for templates and not deployable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from aws_cdk import aws_lambda as lambda_

__all__ = ["BUNDLE_DIR_ENV", "BUNDLE_MANIFEST", "BUNDLE_REQUIRED", "RELEASE_ENV", "SourceOnlyCodeError", "bundle_problems", "function_code", "release_mode"]

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR_ENV = "FINPLAN_LAMBDA_BUNDLE_DIR"
RELEASE_ENV = "FINPLAN_RELEASE_BUILD"
BUNDLE_MANIFEST = "bundle-manifest.json"
#: Entries a deployable bundle must contain.
BUNDLE_REQUIRED = ("finplan_model", "finplan_contracts", "jsonschema", "rfc8785", "numpy", "boto3", "config", BUNDLE_MANIFEST)


class SourceOnlyCodeError(RuntimeError):
    """Release-mode synth without a complete dependency bundle."""


def release_mode(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    flag = env.get(RELEASE_ENV, "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return bool(env.get("CODEBUILD_BUILD_ID"))


def bundle_problems(path: Path) -> list[str]:
    if not path.is_dir():
        return [f"{path} does not exist"]
    return [f"{name} missing" for name in BUNDLE_REQUIRED if not ((path / name).exists() or (path / f"{name}.py").is_file())]


def function_code(environ: Mapping[str, str] | None = None) -> lambda_.Code:
    """The (shared) code asset of every FinanceModel function."""
    env = os.environ if environ is None else environ
    release = release_mode(env)
    raw = env.get(BUNDLE_DIR_ENV)
    if raw:
        path = Path(raw)
        problems = bundle_problems(path)
        if not problems:
            return lambda_.Code.from_asset(str(path))
        if release:
            raise SourceOnlyCodeError(f"Lambda bundle at {path} is incomplete: {'; '.join(problems)}")
    if release:
        raise SourceOnlyCodeError(
            "release synth without a dependency bundle: a source-only package fails at init "
            f"(No module named 'finplan_contracts'). Build it (scripts/lambda_bundle.py) and set {BUNDLE_DIR_ENV}."
        )
    return lambda_.Code.from_asset(str(REPO_ROOT / "src"), exclude=["**/__pycache__", "**/*.pyc", "**/.pytest_cache"])
