"""DS-12 (no provider dependency or call) and DS-13 (fixtures carry ``synthetic: true``); task 3.7.

Violations are planted in temporary directories at run time, never in the repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finplan_model.datasets.provider_guard import fixture_problems, provider_problems
from scripts.build_gates import GateContext, run_gates

ROOT = Path(__file__).resolve().parents[3]
PROVIDER = "y" + "finance"  # planted names are assembled at run time


def _pyproject(tmp: Path, deps: list[str], dev: list[str] | None = None) -> None:
    groups = f"\n[dependency-groups]\ndev = {json.dumps(dev or [])}\n"
    (tmp / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "0"\ndependencies = {json.dumps(deps)}\n{groups}', encoding="utf-8")


def test_repository_passes_the_provider_guard_and_fixture_check():
    assert provider_problems(ROOT) == []
    assert fixture_problems(ROOT) == []
    results = {r.name: r for r in run_gates(GateContext(root=ROOT, run_unit=False), only=["provider-guard", "fixture-check"])}
    assert results["provider-guard"].ok and results["fixture-check"].ok


@pytest.mark.parametrize("where", ["dependencies", "dev"])
def test_planted_provider_dependency_fails_and_names_the_package(tmp_path, where):
    _pyproject(tmp_path, [f"{PROVIDER}==0.2.66"] if where == "dependencies" else ["numpy"], [f"{PROVIDER}>=0.2"] if where == "dev" else None)
    problems = provider_problems(tmp_path)
    assert problems and PROVIDER in problems[0]


def test_planted_provider_in_an_image_fails(tmp_path):
    (tmp_path / "container").mkdir()
    (tmp_path / "container" / "Dockerfile").write_text(f"FROM python:3.12-slim\nRUN pip install numpy {PROVIDER}==0.2.66\n", encoding="utf-8")
    problems = provider_problems(tmp_path)
    assert any("container/Dockerfile" in p and PROVIDER in p for p in problems)


@pytest.mark.parametrize("line", [f"import {PROVIDER} as yf", f"from {PROVIDER} import Ticker", "import pandas_datareader.data as web", f"mod = __import__('{PROVIDER}')"])
def test_planted_provider_import_fails(tmp_path, line):
    pkg = tmp_path / "src" / "finplan_model"
    pkg.mkdir(parents=True)
    (pkg / "fetch.py").write_text(line + "\n", encoding="utf-8")
    problems = provider_problems(tmp_path)
    assert problems and "fetch.py imports market-data provider client" in problems[0]


def test_planted_provider_endpoint_fails(tmp_path):
    pkg = tmp_path / "src"
    pkg.mkdir()
    host = "query1." + "finance." + "yahoo.com"
    (pkg / "client.py").write_text(f'URL = "https://{host}/v8/finance/chart/SPY"\n', encoding="utf-8")
    problems = provider_problems(tmp_path)
    assert problems and "endpoint" in problems[0]


def test_build_gate_fails_on_planted_provider(tmp_path):
    _pyproject(tmp_path, [PROVIDER])
    (r,) = run_gates(GateContext(root=tmp_path, run_unit=False), only=["provider-guard"])
    assert not r.ok and PROVIDER in r.problems[0]


def _prices_csv() -> str:
    rows = ["Date,Open,High,Low,Close,Adj Close,Volume"] + [f"2024-01-{d:02d},470.1,471.2,469.0,470.5,468.9,61234500" for d in range(2, 12)]
    return "\n".join(rows) + "\n"


def test_planted_non_synthetic_price_file_fails(tmp_path):
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "spy.csv").write_text(_prices_csv(), encoding="utf-8")
    problems = fixture_problems(tmp_path)
    assert problems and "spy.csv" in problems[0]
    (r,) = run_gates(GateContext(root=tmp_path, run_unit=False), only=["fixture-check"])
    assert not r.ok


def test_price_series_anywhere_in_the_repository_fails(tmp_path):
    (tmp_path / "notebooks").mkdir()
    rows = [{"date": f"2024-01-{d:02d}", "close": 470.5 + d} for d in range(2, 12)]
    (tmp_path / "notebooks" / "spy_prices.json").write_text(json.dumps(rows), encoding="utf-8")
    assert fixture_problems(tmp_path)


def test_marked_synthetic_fixtures_pass(tmp_path):
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "spy.csv").write_text("# synthetic: true\n" + _prices_csv(), encoding="utf-8")
    (tmp_path / "tests" / "fixtures" / "series.json").write_text(json.dumps({"synthetic": True, "observations": [{"date": "2024-01-02", "close": 1.0}]}), encoding="utf-8")
    (tmp_path / "tests" / "fixtures" / "list.json").write_text(json.dumps([{"session_date": "2024-01-02", "close": 1.0, "synthetic": True}]), encoding="utf-8")
    assert fixture_problems(tmp_path) == []


def test_unmarked_json_fixture_and_binary_fixture_fail(tmp_path):
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "x.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (tmp_path / "tests" / "fixtures" / "x.parquet").write_bytes(b"PAR1")
    problems = fixture_problems(tmp_path)
    assert len(problems) == 2
