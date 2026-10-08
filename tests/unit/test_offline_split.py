"""The offline harness applies to offline suites only (platform lesson L5; mirrors FinancialPlanning
``tests/unit/test_offline_env.py``).

The platform's first prod smoke failed with ``UnrecognizedClientException`` because its root
conftest forced fake credentials on every suite, deployed ones included. Here:

* ``tests/unit`` and ``tests/contract`` always run under the offline harness (fake credentials, no
  metadata service, no network, no SageMaker), even if ``FINPLAN_TARGET_ENV`` is set;
* a deployed suite started by ``scripts/stage_runner.py`` (``FINPLAN_TARGET_ENV`` set,
  ``tests/integration`` or ``tests/smoke`` collected on their own) keeps the real credentials.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import boto3

from tests.harness import OFFLINE_MARKER

ROOT = Path(__file__).resolve().parents[2]
#: Credential-shaped placeholders (not real keys) standing in for the stage role's credentials.
STAGE_ENV = {
    "AWS_ACCESS_KEY_ID": "stage-role-access-key-placeholder",
    "AWS_SECRET_ACCESS_KEY": "stage-role-secret-placeholder",
    "AWS_SESSION_TOKEN": "stage-role-session-token-placeholder",
}


def test_this_offline_suite_runs_with_the_fake_credentials() -> None:
    assert os.environ[OFFLINE_MARKER] == "1"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "testing" and os.environ["AWS_EC2_METADATA_DISABLED"] == "true"
    assert os.environ["AWS_CONFIG_FILE"] == os.devnull and os.environ["AWS_SHARED_CREDENTIALS_FILE"] == os.devnull
    assert "AWS_PROFILE" not in os.environ and os.environ["FINPLAN_RELEASE_BUILD"] == "0"
    creds = boto3.session.Session().get_credentials()
    assert creds is not None and creds.access_key == "testing" and creds.method == "env"


def _run_pytest(args: list[str], extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    drop = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", OFFLINE_MARKER, "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE", "AWS_EC2_METADATA_DISABLED", "FINPLAN_RELEASE_BUILD"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(extra_env)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_deployed_suite_mode_keeps_the_stage_credentials() -> None:
    """The stage runner's invocation (FINPLAN_TARGET_ENV + tests/integration) leaves the credentials
    intact. Only the credential probe is selected; it makes no AWS call."""
    proc = _run_pytest(["tests/integration/test_deployed_environment.py", "-k", "real_credentials"], {**STAGE_ENV, "FINPLAN_TARGET_ENV": "beta", "FINPLAN_SUITE": "integration-beta"})
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "1 passed" in proc.stdout


def test_offline_suites_stay_hermetic_even_with_a_target_env() -> None:
    proc = _run_pytest(["tests/unit/test_offline_split.py", "-k", "this_offline_suite"], {**STAGE_ENV, "FINPLAN_TARGET_ENV": "beta"})
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "1 passed" in proc.stdout
