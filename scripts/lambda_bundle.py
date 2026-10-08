#!/usr/bin/env python3
"""Code bundle of the FinanceModel control-plane Lambdas (task 10.3; mirrors the platform's
2026-10-07 fix for source-only Lambda packages).

One bundle serves every function (``job-api-handler``, ``job-dispatcher``, ``job-state-handler``,
``job-registry-lookup``; handlers in :data:`infra.stacks.naming.FUNCTIONS`). It contains:

* the locked runtime closure from ``uv.lock`` (``uv export --frozen --no-dev --no-emit-project``),
  installed with ``--require-hashes --only-binary :all:`` for CPython 3.12 on **arm64**
  (``aarch64-manylinux_2_28``, the functions' architecture, as in the platform: Lambda
  ``python3.12`` runs Amazon Linux 2023, glibc 2.34; ``manylinux2014`` is not enough since numpy
  2.x ships only ``manylinux_2_28`` aarch64 wheels). The closure includes the vendored, digest-pinned ``finplan-contracts`` wheel
  (its SHA-256 is re-checked against ``contracts-pin.json``), ``jsonschema``, ``rfc8785``, ``numpy``
  and the pinned ``boto3``;
* the ``finplan_model`` package and ``config/`` (``FINPLAN_CONFIG_DIR=/var/task/config``);
* ``bundle-manifest.json`` (handlers, target platform, file count, unzipped size).

The build stage builds it before the release-mode synth (``FINPLAN_LAMBDA_BUNDLE_DIR``); a
release-mode synth without it fails (:mod:`infra.stacks.lambda_code`).

Usage::

    uv run python scripts/lambda_bundle.py --out .build/lambda-bundle                 # release bundle
    uv run python scripts/lambda_bundle.py --out /tmp/b --local --import-check        # host platform + import check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = ["ELF_AARCH64", "LAMBDA_PLATFORM", "LAMBDA_UNZIPPED_LIMIT", "BundleError", "build_bundle", "bundle_size", "foreign_binaries", "import_check", "verify_bundle"]

#: uv ``--python-platform`` of the Lambda runtime (python3.12, arm64, Amazon Linux 2023), as in
#: the platform's ``scripts/lambda_bundle.py``. The functions run on ``arm64``
#: (:data:`infra.stacks.control.LAMBDA_ARCHITECTURE`).
LAMBDA_PLATFORM = "aarch64-manylinux_2_28"
#: ELF ``e_machine`` of arm64 (the Lambda functions' architecture).
ELF_AARCH64 = 0xB7
PYTHON_VERSION = "3.12"
LAMBDA_UNZIPPED_LIMIT = 250 * 1024 * 1024
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")

Runner = Callable[[list[str], Path], str]


class BundleError(RuntimeError):
    pass


def _uv() -> str:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise BundleError("uv is required to build the Lambda bundle (it reads the pinned closure from uv.lock)")
    return uv


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise BundleError(f"{' '.join(cmd[:4])} ... failed ({proc.returncode}): {(proc.stdout + proc.stderr)[-2000:]}")
    return proc.stdout


def _check_contracts_wheel(root: Path) -> None:
    pin = json.loads((root / "contracts-pin.json").read_text(encoding="utf-8"))
    wheel = root / pin["artifact"]
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != pin["sha256"]:
        raise BundleError(f"vendored contract wheel {wheel.name} digest {digest} != contracts-pin.json {pin['sha256']}")


def bundle_size(path: Path) -> tuple[int, int]:
    total = files = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
            files += 1
    return total, files


def install_commands(requirements: Path, dest: Path, *, python_platform: str | None, python: str | None, offline: bool = False) -> list[list[str]]:
    cmd = [_uv(), "pip", "install", "--quiet", "--target", str(dest), "--no-deps", "--require-hashes", "--only-binary", ":all:", "--no-config", "-r", str(requirements)]
    if python_platform:
        cmd += ["--python-platform", python_platform, "--python-version", PYTHON_VERSION]
    cmd += ["--python", python or sys.executable]
    if offline:
        cmd.append("--offline")
    return [cmd]


def build_bundle(
    root: Path,
    dest: Path,
    *,
    python_platform: str | None = LAMBDA_PLATFORM,
    python: str | None = None,
    limit: int = LAMBDA_UNZIPPED_LIMIT,
    replace: bool = False,
    offline: bool = False,
    runner: Runner = _run,
) -> dict[str, Any]:
    """Build the bundle into ``dest``. ``python_platform=None`` builds for the interpreter ``python``."""
    from infra.stacks.lambda_code import BUNDLE_MANIFEST
    from infra.stacks.naming import FUNCTIONS

    if dest.exists():
        if not replace:
            raise BundleError(f"{dest} already exists")
        shutil.rmtree(dest)
    _check_contracts_wheel(root)
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "requirements.txt"
        runner([_uv(), "export", "--frozen", "--no-dev", "--no-emit-project", "--format", "requirements-txt", "--quiet", "-o", str(req)], root)
        for cmd in install_commands(req, dest, python_platform=python_platform, python=python, offline=offline):
            runner(cmd, root)  # cwd=root: the vendored contract wheel is a path relative to the project
    shutil.rmtree(dest / "bin", ignore_errors=True)
    for record in (*dest.glob("*.dist-info/direct_url.json"), *dest.glob("*.dist-info/uv_cache.json")):
        record.unlink()
    (dest / ".lock").unlink(missing_ok=True)
    for cache in list(dest.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    shutil.copytree(root / "src" / "finplan_model", dest / "finplan_model", ignore=_IGNORE)
    shutil.copytree(root / "config", dest / "config", ignore=_IGNORE)
    size, files = bundle_size(dest)
    manifest = {
        "functions": dict(FUNCTIONS),
        "python_version": PYTHON_VERSION,
        "python_platform": python_platform or "host",
        "files": files,
        "unzipped_bytes": size,
        "limit_unzipped_bytes": limit,
    }
    (dest / BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    problems = verify_bundle(dest, limit=limit)
    if python_platform == LAMBDA_PLATFORM:
        problems += [f"{b} is not an arm64 binary" for b in foreign_binaries(dest)]
    if problems:
        raise BundleError("Lambda bundle: " + "; ".join(problems))
    return manifest


def verify_bundle(path: Path, *, limit: int = LAMBDA_UNZIPPED_LIMIT) -> list[str]:
    """Structural problems of a bundle directory (empty: deployable)."""
    from infra.stacks.lambda_code import bundle_problems
    from infra.stacks.naming import FUNCTIONS

    problems = list(bundle_problems(path))
    for handler in FUNCTIONS.values():
        module = handler.rsplit(".", 1)[0]
        if not (path / (module.replace(".", "/") + ".py")).is_file():
            problems.append(f"handler module {module} missing")
    if not (path / "config" / "shared.json").is_file():
        problems.append("config/shared.json missing")
    size, _ = bundle_size(path) if path.is_dir() else (0, 0)
    if size > limit:
        problems.append(f"{size // (1024 * 1024)} MiB unzipped exceeds the Lambda limit of {limit // (1024 * 1024)} MiB")
    return problems


def foreign_binaries(path: Path, machine: int = ELF_AARCH64) -> list[str]:
    """Shared objects in the bundle that are not built for ``machine`` (default arm64)."""
    bad: list[str] = []
    for p in sorted(path.rglob("*.so*")):
        if not p.is_file() or p.is_symlink():
            continue
        with p.open("rb") as fh:
            head = fh.read(20)
        if head[:4] != b"\x7fELF":
            continue
        order = "little" if head[5] == 1 else "big"
        if int.from_bytes(head[18:20], order) != machine:
            bad.append(p.relative_to(path).as_posix())
    return bad


def import_check(path: Path, *, python: str | None = None, env: Mapping[str, str] | None = None) -> None:
    """Import every handler in a fresh interpreter whose only non-stdlib path is the bundle (host-platform bundles only)."""
    from infra.stacks.naming import FUNCTIONS

    handlers = sorted(set(FUNCTIONS.values()))
    code = (
        "import sys, importlib\n"
        f"sys.path.insert(0, {str(path)!r})\n"
        f"for h in {handlers!r}:\n"
        "    mod, fn = h.rsplit('.', 1)\n"
        "    m = importlib.import_module(mod)\n"
        "    assert callable(getattr(m, fn)), h\n"
        f"    assert m.__file__.startswith({str(path)!r}), m.__file__\n"
        "import finplan_contracts, jsonschema, rfc8785\n"
        f"assert finplan_contracts.__file__.startswith({str(path)!r})\n"
        "print('ok')\n"
    )
    run_env = {"PATH": os.environ.get("PATH", ""), "FINPLAN_CONFIG_DIR": str(path / "config"), **(env or {})}
    proc = subprocess.run([python or sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, env=run_env, cwd=path, check=False)
    if proc.returncode != 0:
        raise BundleError(f"the handlers do not import from the bundle alone: {(proc.stdout + proc.stderr)[-2000:]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the FinanceModel Lambda bundle from uv.lock.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--local", action="store_true", help="build for this interpreter's platform instead of Lambda arm64")
    ap.add_argument("--offline", action="store_true", help="install from the local uv cache only")
    ap.add_argument("--import-check", action="store_true", help="import every handler with only the bundle on sys.path (needs --local)")
    args = ap.parse_args(argv)
    if args.import_check and not args.local:
        ap.error("--import-check needs --local (arm64 extension modules do not load on another platform)")
    try:
        manifest = build_bundle(args.root, args.out, python_platform=None if args.local else LAMBDA_PLATFORM, replace=True, offline=args.offline)
        if args.import_check:
            import_check(args.out)
    except BundleError as exc:
        print(f"BUNDLE FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"unzipped_mib": round(manifest["unzipped_bytes"] / 2**20, 1), "files": manifest["files"], "python_platform": manifest["python_platform"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
