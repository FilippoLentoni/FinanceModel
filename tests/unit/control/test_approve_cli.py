"""The approver CLI (scripts/approve_run.py) signs requests with SigV4 and approves only after showing
the estimate; exercised against the in-process job API (no network)."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.parse import urlparse

from botocore.credentials import Credentials

from scripts.approve_run import main

from .support import APPROVER, SUBMITTER, Harness

ENDPOINT = "https://jobs.example.invalid/beta"


class _Session:
    def __init__(self) -> None:
        self.creds = Credentials("test-access-key", "test-secret-key", "test-session-token")

    def client(self, name: str) -> Any:
        assert name == "ssm"

        class _Ssm:
            def get_parameter(self, Name: str) -> dict:  # noqa: N803
                assert Name == "/finplan/beta/financemodel/api/job-endpoint"
                return {"Parameter": {"Value": ENDPOINT}}

        return _Ssm()

    def get_credentials(self) -> Credentials:
        return self.creds


class _Resp(io.BytesIO):
    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(h: Harness, principal: str, seen: list) -> Any:
    def open_(req, timeout=None):
        headers = {k.lower(): v for k, v in req.header_items()}
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256") and "execute-api" in headers["authorization"]
        path = urlparse(req.full_url).path.removeprefix("/beta")
        seen.append((req.get_method(), path))
        code, body, _ = h.call(req.get_method(), path, principal=principal, body=json.loads(req.data) if req.data else None)
        if code >= 400:
            import urllib.error

            raise urllib.error.HTTPError(req.full_url, code, "err", {}, io.BytesIO(json.dumps(body).encode()))
        return _Resp(code, json.dumps(body).encode())

    return open_


def test_cli_shows_estimate_and_approves_after_confirmation(capsys):
    h = Harness()
    _, sub = h.submit()
    seen: list = []
    rc = main(["--env", "beta", sub["run_id"]], session=_Session(), opener=_opener(h, APPROVER, seen), confirm=lambda _: "y")
    assert rc == 0
    assert seen == [("GET", f"/v1/jobs/{sub['run_id']}"), ("POST", f"/v1/jobs/{sub['run_id']}/approve")]
    out = capsys.readouterr().out
    assert "estimate USD 0.0725" in out and "cpu_research" in out
    assert h.run(sub["run_id"])["state"] == "queued"


def test_cli_without_confirmation_changes_nothing():
    h = Harness()
    _, sub = h.submit()
    rc = main(["--env", "beta", sub["run_id"]], session=_Session(), opener=_opener(h, APPROVER, []), confirm=lambda _: "n")
    assert rc == 1 and h.run(sub["run_id"])["state"] == "awaiting_approval"


def test_cli_with_a_tool_role_is_refused():
    h = Harness()
    _, sub = h.submit()
    rc = main(["--env", "beta", sub["run_id"], "--yes"], session=_Session(), opener=_opener(h, SUBMITTER, []))
    assert rc == 1 and h.run(sub["run_id"])["state"] == "awaiting_approval"
