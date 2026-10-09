"""Task 1.2: contract pin by exact version + digest, and the build gates (copied ``$id``, leak scan,
consumer conformance). Planted violations are built at run time so this file itself stays clean."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.build_gates import GateContext, gate_copied_id, gate_leak_scan, run_gates
from scripts.check_contracts_pin import check, main

ROOT = Path(__file__).resolve().parents[2]


def _copy_pin_tree(tmp: Path) -> Path:
    for name in ("pyproject.toml", "uv.lock", "contracts-pin.json"):
        shutil.copy(ROOT / name, tmp / name)
    (tmp / "vendor" / "finplan-contracts").mkdir(parents=True)
    pin = json.loads((ROOT / "contracts-pin.json").read_text(encoding="utf-8"))
    shutil.copy(ROOT / pin["artifact"], tmp / pin["artifact"])
    return tmp


def test_pin_verifies():
    assert check(ROOT) == []
    assert main(["--root", str(ROOT)]) == 0


def test_pin_is_exact_and_matches_installed_version():
    import finplan_contracts

    pin = json.loads((ROOT / "contracts-pin.json").read_text(encoding="utf-8"))
    assert pin["version"] == finplan_contracts.__version__ == "1.3.0"
    assert pin["served_environments"] == ["beta", "gamma", "prod"]
    assert "registry-ref" in pin["registry"]


def test_tampered_wheel_fails_with_digest_mismatch(tmp_path):
    root = _copy_pin_tree(tmp_path)
    pin = json.loads((root / "contracts-pin.json").read_text(encoding="utf-8"))
    with (root / pin["artifact"]).open("ab") as fh:
        fh.write(b"tampered")
    problems = check(root, check_installed=False)
    assert any(p.startswith("Digest mismatch") for p in problems)


def test_version_range_is_refused(tmp_path):
    root = _copy_pin_tree(tmp_path)
    py = root / "pyproject.toml"
    py.write_text(py.read_text(encoding="utf-8").replace('"finplan-contracts==1.3.0"', '"finplan-contracts>=1.0"'), encoding="utf-8")
    assert any("exactly" in p for p in check(root, check_installed=False))


@pytest.mark.parametrize("env", ["beta", "gamma", "prod"])
def test_one_x_pin_is_promotable_everywhere(env):
    """1.0.0 removed the beta-only stop: the promotion check passes in every environment."""
    assert check(ROOT, env=env) == []


@pytest.mark.parametrize("env", ["gamma", "prod"])
def test_planted_zero_x_pin_is_still_beta_only(tmp_path, env):
    """The contract rule itself stays (contracts D16): a 0.x pin may not leave beta."""
    root = _copy_pin_tree(tmp_path)
    pin_path = root / "contracts-pin.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    old = pin["artifact"]
    pin["version"] = "0.9.0"
    pin["artifact"] = old.replace("1.3.0", "0.9.0")
    (root / old).rename(root / pin["artifact"])
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    assert not any("beta only" in p for p in check(root, env="beta", check_installed=False))
    assert any("beta only" in p for p in check(root, env=env, check_installed=False))


def test_build_fails_on_planted_copied_schema_id(tmp_path):
    assert gate_copied_id(GateContext(root=tmp_path)) == []
    schema_id = "https://contracts.finplan" + ".invalid/core/v1/error.json"
    (tmp_path / "copied.json").write_text(json.dumps({"$id": schema_id, "type": "object"}), encoding="utf-8")
    problems = gate_copied_id(GateContext(root=tmp_path))
    assert problems and "copied" in problems[0]


def test_build_fails_on_planted_account_id(tmp_path):
    assert gate_leak_scan(GateContext(root=tmp_path)) == []
    account = "".join(str(d) for d in (4, 8, 1, 5, 2, 9, 7, 3, 6, 0, 4, 1))
    (tmp_path / "notes.md").write_text(f"deploy to account {account}\n", encoding="utf-8")
    problems = gate_leak_scan(GateContext(root=tmp_path))
    assert problems and "leak" in problems[0]


def test_repository_passes_pre_gates():
    results = run_gates(GateContext(root=ROOT, run_unit=False), stage="pre")
    failing = {r.name: r.problems for r in results if not r.ok}
    assert failing == {}
    assert {r.name for r in results} >= {"contracts-pin", "config", "leak-scan", "copied-id", "conformance", "live-perm-scan"}


def test_repin_rewrites_pin_and_pyproject(tmp_path):
    from scripts.check_contracts_pin import repin

    root = _copy_pin_tree(tmp_path)
    pin = json.loads((root / "contracts-pin.json").read_text(encoding="utf-8"))
    src = tmp_path / "incoming" / "finplan_contracts-1.3.0-py3-none-any.whl"
    src.parent.mkdir()
    src.write_bytes((root / pin["artifact"]).read_bytes())
    out = repin(src, root)
    assert out["version"] == "1.3.0" and out["artifact"].endswith("finplan_contracts-1.3.0-py3-none-any.whl")
    assert '"finplan-contracts==1.3.0"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        repin(tmp_path / "incoming" / "not-a-wheel.whl", root)
