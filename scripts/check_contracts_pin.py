#!/usr/bin/env python3
"""Verify the finplan-contracts pin of FinanceModel (task 1.2; contracts CS-04 consumer case).

Fails ("Digest mismatch" or a named problem) unless ALL of these agree:

1. ``contracts-pin.json``: package, exact version and SHA-256 of the pinned wheel;
2. the vendored wheel on disk hashes to that SHA-256 (verified with the contract package's own
   ``finplan_contracts.digests.verify`` when importable);
3. ``pyproject.toml`` depends on ``finplan-contracts==<version>`` exactly (no range), and
   ``[tool.uv.sources]`` points at the pinned artifact;
4. ``uv.lock`` locks that version with the same wheel hash (uv refuses a wheel whose hash differs);
5. the installed distribution has that version (when installed).

``--env gamma|prod`` fails for a 0.x pin: 0.x is beta-only (contracts ``docs/consumer-pinning.md``).
There is no ``--rebuild``: FinanceModel never holds the contract sources (CS-01). The vendored wheel
is the platform's reproducible build of the pinned version, byte for byte: the same artifact the platform build
publishes to CodeArtifact (it refuses to publish different bytes under the same version), so the
pinned digest also identifies the registry artifact. 1.x passes ``--env`` in every environment.

``--repin --from <wheel>`` re-pins to another platform wheel: it copies the wheel into
``vendor/finplan-contracts/``, rewrites ``contracts-pin.json`` (version, artifact, sha256) and the
``pyproject.toml`` dependency and source; run ``uv sync`` afterwards to refresh ``uv.lock``.

Exit codes: 0 ok, 1 mismatch, 2 usage/config error. Runs offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

__all__ = ["check", "load_pin", "main", "repin"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pin(root: Path = ROOT, pin_path: Path | None = None) -> dict:
    return json.loads((pin_path or root / "contracts-pin.json").read_text(encoding="utf-8"))


def check(root: Path = ROOT, *, env: str | None = None, pin_path: Path | None = None, check_installed: bool = True) -> list[str]:
    problems: list[str] = []
    pin = load_pin(root, pin_path)
    package, version = pin["package"], pin["version"]
    digest = str(pin["sha256"]).lower().removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return ["contracts-pin.json: sha256 must be 64 lowercase hex characters"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        return [f"contracts-pin.json: version {version!r} is not an exact semver"]
    wheel = (root / pin["artifact"]).resolve()
    if not wheel.is_file():
        return [f"pinned artifact {pin['artifact']} is missing"]

    # 2. digest of the artifact
    try:
        from finplan_contracts.digests import verify

        ok = verify(wheel, digest)
    except ImportError:  # pragma: no cover - before the package is installed
        ok = _sha256(wheel) == digest
    if not ok:
        problems.append(f"Digest mismatch: {wheel.name} sha256 {_sha256(wheel)} != pinned {digest}")
    if f"-{version}-" not in wheel.name:
        problems.append(f"artifact {wheel.name} does not carry the pinned version {version}")

    # 3. exact pin in pyproject
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject.get("project", {}).get("dependencies", [])
    spec = next((d for d in deps if re.match(rf"^{re.escape(package)}\s*[=<>!~]", d)), None)
    if spec is None:
        problems.append(f"pyproject.toml does not depend on {package}")
    elif re.sub(r"\s", "", spec) != f"{package}=={version}":
        problems.append(f"pyproject.toml must pin {package}=={version} exactly (found {spec!r}); ranges are not allowed")
    src = pyproject.get("tool", {}).get("uv", {}).get("sources", {}).get(package, {})
    if src.get("path") and Path(src["path"]).as_posix() != Path(pin["artifact"]).as_posix():
        problems.append(f"[tool.uv.sources] {package} points at {src['path']}, not the pinned artifact")

    # 4. uv.lock
    lock_path = root / "uv.lock"
    if lock_path.is_file():
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        entry = next((p for p in lock.get("package", []) if p.get("name") == package), None)
        if entry is None:
            problems.append(f"uv.lock has no {package} entry")
        else:
            if entry.get("version") != version:
                problems.append(f"uv.lock locks {package} {entry.get('version')}, pin says {version}")
            hashes = {w.get("hash", "").removeprefix("sha256:") for w in entry.get("wheels", [])}
            if digest not in hashes:
                problems.append(f"Digest mismatch: uv.lock wheel hash {sorted(hashes)} does not include the pinned digest")
    else:
        problems.append("uv.lock is missing")

    # 5. installed distribution
    if check_installed:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as dist_version

        try:
            installed = dist_version(package)
            if installed != version:
                problems.append(f"installed {package} {installed} != pinned {version}")
        except PackageNotFoundError:
            pass

    # 0.x is beta-only
    if env in ("gamma", "prod") and version.startswith("0."):
        problems.append(f"contract version {version} is a 0.x pre-release and may be deployed to beta only; {env} requires 1.0.0 or later")
    return problems


def repin(wheel: Path, root: Path = ROOT, pin_path: Path | None = None) -> dict:
    """Re-pin to ``wheel`` (a platform-built ``finplan_contracts-<ver>-py3-none-any.whl``)."""
    import shutil

    m = re.fullmatch(r"finplan_contracts-([0-9]+\.[0-9]+\.[0-9]+)-py3-none-any\.whl", wheel.name)
    if not m or not wheel.is_file():
        raise ValueError(f"{wheel} is not a finplan_contracts wheel")
    version = m.group(1)
    dest_rel = f"vendor/finplan-contracts/{wheel.name}"
    dest = root / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if wheel.resolve() != dest.resolve():
        shutil.copyfile(wheel, dest)
    pin_file = pin_path or root / "contracts-pin.json"
    pin = load_pin(root, pin_path)
    old_artifact = pin["artifact"]
    old_version = pin["version"]
    pin.update(version=version, artifact=dest_rel, sha256=_sha256(dest))
    pin_file.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    text = text.replace(f'"{pin["package"]}=={old_version}"', f'"{pin["package"]}=={version}"')
    text = text.replace(f'path = "{old_artifact}"', f'path = "{dest_rel}"')
    pyproject.write_text(text, encoding="utf-8")
    return pin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--pin", type=Path, help="pin file (default: <root>/contracts-pin.json)")
    ap.add_argument("--env", choices=["beta", "gamma", "prod"], help="deployment target (0.x pins are beta-only)")
    ap.add_argument("--repin", action="store_true", help="re-pin to the wheel given by --from, then run `uv sync`")
    ap.add_argument("--from", dest="from_wheel", type=Path, help="platform-built wheel to re-pin to (with --repin)")
    args = ap.parse_args(argv)
    if args.repin:
        if not args.from_wheel:
            print("ERROR: --repin requires --from <wheel>", file=sys.stderr)
            return 2
        try:
            pin = repin(args.from_wheel, args.root, args.pin)
        except (OSError, KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"REPINNED: {pin['package']} {pin['version']} sha256 {pin['sha256']}; now run `uv sync`")
        return 0
    try:
        problems = check(args.root, env=args.env, pin_path=args.pin)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print("PASS: finplan-contracts pin verified (version, wheel digest, pyproject, uv.lock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
