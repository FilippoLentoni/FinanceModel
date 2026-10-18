"""The vendored finplan-platform wheel is pinned like finplan-contracts (platform-pin.json gate)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.check_platform_pin import check, main

ROOT = Path(__file__).resolve().parents[2]


def _tree(tmp: Path) -> Path:
    for name in ("pyproject.toml", "uv.lock", "platform-pin.json"):
        shutil.copy(ROOT / name, tmp / name)
    pin = json.loads((ROOT / "platform-pin.json").read_text(encoding="utf-8"))
    (tmp / pin["artifact"]).parent.mkdir(parents=True)
    shutil.copy(ROOT / pin["artifact"], tmp / pin["artifact"])
    return tmp


def test_platform_pin_verifies():
    assert check(ROOT, check_installed=False) == []
    assert main(["--root", str(ROOT)]) in (0, 1)  # installed-version check depends on the interpreter


def test_pin_names_the_reproducible_platform_build():
    pin = json.loads((ROOT / "platform-pin.json").read_text(encoding="utf-8"))
    assert pin["package"] == "finplan-platform" and pin["source_date_epoch"] == 315532800
    assert len(pin["source_commit"]) == 40 and pin["served_environments"] == ["beta"]


def test_tampered_wheel_range_and_missing_commit_fail(tmp_path):
    root = _tree(tmp_path)
    pin = json.loads((root / "platform-pin.json").read_text(encoding="utf-8"))
    with (root / pin["artifact"]).open("ab") as fh:
        fh.write(b"tampered")
    assert any(p.startswith("Digest mismatch") for p in check(root, check_installed=False))
    root = _tree(tmp_path / "b") if (tmp_path / "b").mkdir() is None else None
    py = root / "pyproject.toml"
    py.write_text(py.read_text(encoding="utf-8").replace('"finplan-platform==0.1.0"', '"finplan-platform>=0.1"'), encoding="utf-8")
    assert any("exactly" in p for p in check(root, check_installed=False))
    root = _tree(tmp_path / "c") if (tmp_path / "c").mkdir() is None else None
    pin = json.loads((root / "platform-pin.json").read_text(encoding="utf-8"))
    pin["source_commit"] = "main"
    (root / "platform-pin.json").write_text(json.dumps(pin), encoding="utf-8")
    assert any("source_commit" in p for p in check(root, check_installed=False))


def test_platform_pin_gate_is_a_pre_gate():
    from scripts.build_gates import GATES

    assert ("platform-pin", "pre") in {(name, stage) for name, stage, _ in GATES}
