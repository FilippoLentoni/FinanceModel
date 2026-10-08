"""Task 6.1: the ``financemodel-cpu`` Dockerfile is buildable as written and safe (checked statically;
docker is not required in the build stage of unit tests). The container behaviour itself is tested
by running the same entry point locally (``test_job_container.py``)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from finplan_contracts.leak_scan import scan_text

from finplan_model.control.sagemaker import CONTAINER_ENTRYPOINT
from finplan_model.jobs.handlers import HANDLERS

ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "container" / "Dockerfile"
IGNORE = ROOT / "container" / "Dockerfile.dockerignore"


def _instructions() -> list[tuple[str, str]]:
    text = DOCKERFILE.read_text(encoding="utf-8")
    joined = re.sub(r"\\\n\s*", " ", text)
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        op, _, rest = line.partition(" ")
        out.append((op.upper(), rest.strip()))
    return out


def test_entrypoint_matches_the_processing_request():
    ins = _instructions()
    (entry,) = [r for op, r in ins if op == "ENTRYPOINT"]
    assert json.loads(entry) == CONTAINER_ENTRYPOINT
    assert sorted(HANDLERS) == ["daily_recommendation", "prepare_dataset", "report", "run_backtest", "run_benchmark"]


def test_runtime_stage_is_minimal_non_root_and_locked():
    ins = _instructions()
    runs = " ".join(r for op, r in ins if op == "RUN")
    assert "uv sync --frozen --no-dev --no-install-project" in runs and "uv sync --frozen --no-dev --no-editable" in runs
    users = [r for op, r in ins if op == "USER"]
    assert users and not users[-1].startswith("0") and users[-1] != "root"
    copies = " ".join(r for op, r in ins if op == "COPY")
    for forbidden in ("tests", "openspec", ".git", "cdk.out", "infra", "scripts", ".env"):
        assert not re.search(rf"(^|\s){re.escape(forbidden)}(/|\s|$)", copies), forbidden
    assert "vendor/finplan-contracts/" in copies and "config/" in copies
    envs = " ".join(r for op, r in ins if op == "ENV")
    assert "FINPLAN_CONFIG_DIR=/opt/financemodel/config" in envs
    froms = [r for op, r in ins if op == "FROM"]
    assert froms[0].startswith("${UV_IMAGE}") and all(f.startswith("${") for f in froms)


def test_dependencies_are_installed_in_the_image_and_no_docker_hub_frontend():
    """Platform lesson L3: the image runs from its own environment, so the locked runtime closure
    (finplan-contracts, numpy, scipy, boto3) is installed into the venv the runtime stage copies, and
    the entry point is that venv's interpreter."""
    ins = _instructions()
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert not re.search(r"^#\s*syntax=", text, re.M), "the syntax directive pulls a frontend image from Docker Hub"
    envs = " ".join(r for op, r in ins if op == "ENV")
    assert "UV_PROJECT_ENVIRONMENT=/opt/financemodel/venv" in envs and "PATH=/opt/financemodel/venv/bin:" in envs
    copies = [r for op, r in ins if op == "COPY"]
    assert "--from=build /opt/financemodel/venv /opt/financemodel/venv" in copies
    runs = [r for op, r in ins if op == "RUN"]
    assert runs.index("uv sync --frozen --no-dev --no-install-project") < runs.index("uv sync --frozen --no-dev --no-editable")
    (entry,) = [r for op, r in ins if op == "ENTRYPOINT"]
    assert json.loads(entry)[0] == "python"


def test_base_images_are_parameterized_for_digest_pinning():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^ARG PYTHON_IMAGE=public\.ecr\.aws/docker/library/python:3\.12-slim", text, re.M)
    assert re.search(r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+$", text, re.M)


def test_dockerfile_has_no_leaks_or_live_trading_clients():
    for path in (DOCKERFILE, IGNORE):
        assert scan_text(path.read_text(encoding="utf-8"), str(path)) == []
    text = DOCKERFILE.read_text(encoding="utf-8").lower()
    for pkg in ("yfinance", "ccxt", "alpaca", "ib_insync", "coinbase", "pip install"):
        assert pkg not in text


def test_build_context_allow_list_excludes_everything_else():
    lines = [l.strip() for l in IGNORE.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    assert lines[0] == "*"
    allowed = {l[1:] for l in lines if l.startswith("!")}
    assert allowed == {"pyproject.toml", "uv.lock", "README.md", "vendor/finplan-contracts/*.whl", "src/finplan_model/**", "config/*.json"}


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed; the image is built in the pipeline build stage")
def test_docker_build_succeeds():  # pragma: no cover - runs only where docker exists
    out = subprocess.run(["docker", "build", "-f", str(DOCKERFILE), "-t", "financemodel-cpu:test", str(ROOT)], capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
