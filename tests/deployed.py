"""Helpers of the deployed suites (``tests/integration``, ``tests/smoke``), which run only in the
pipeline's stage projects (``scripts/stage_runner.py tests``) with ``FINPLAN_TARGET_ENV`` set and the
stage role's real credentials. Offline (no ``FINPLAN_TARGET_ENV``) every deployed test is skipped.

Nothing here hard-codes an account, endpoint or bucket: every reference is read from SSM under the
environment's own segment, exactly as a consumer would.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pytest

__all__ = ["DeployedEnv", "deployed", "requires_deployed"]

requires_deployed = pytest.mark.skipif(not os.environ.get("FINPLAN_TARGET_ENV"), reason="deployed suite: runs only in the pipeline stage (FINPLAN_TARGET_ENV)")


@dataclass
class DeployedEnv:
    env: str
    session: Any

    @property
    def region(self) -> str:
        return str(self.session.region_name)

    def param(self, name: str) -> str | None:
        try:
            return self.session.client("ssm").get_parameter(Name=name)["Parameter"]["Value"]
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "response", {}).get("Error", {}).get("Code") == "ParameterNotFound":
                return None
            raise

    def own(self, category: str, name: str) -> str:
        from finplan_contracts import ssm as contract_ssm

        return contract_ssm.build(self.env, "financemodel", category, name)

    def manifest(self) -> dict[str, Any]:
        raw = self.param(self.own("release", "manifest"))
        assert raw, "the release manifest is not published"
        return json.loads(raw)

    def job_endpoint(self) -> str | None:
        return self.param(self.own("api", "job-endpoint"))

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        from scripts.approve_run import signed_request

        endpoint = self.job_endpoint()
        # Contracts 1.0.0 (D16) carries the job API's ownership rows: every release deploys and
        # publishes it, so a missing endpoint is a failed deploy, never a skip.
        assert endpoint, f"{self.own('api', 'job-endpoint')} is not published: the job API did not deploy"
        creds = self.session.get_credentials()
        return signed_request(method, endpoint.rstrip("/") + path, region=self.region, credentials=creds, body=body)

    def api_call(self) -> Any:
        """``call(method, path, body=None)`` for :mod:`tests.integration.job_suite` (SigV4, endpoint from SSM)."""
        self.call("GET", "/v1/jobs")  # fails when the job API is not deployed (see :meth:`call`)
        return lambda method, path, body=None: self.call(method, path, body)

    def integration_snapshot_id(self) -> str:
        """The approved synthetic platform snapshot the lifecycle runs on (operator-set, docs/operations.md)."""
        # An operator override wins; otherwise use the approved synthetic snapshot the platform's
        # own beta/gamma suite publishes after it passes (FinancialPlanning pipeline).
        platform_key = f"/finplan/{self.env}/financialplanning/config/integration-snapshot-id"
        sid = self.param(self.own("config", "integration-snapshot-id")) or self.param(platform_key)
        assert sid, f"no integration snapshot: set {self.own('config', 'integration-snapshot-id')} or let the platform pipeline publish {platform_key}"
        return sid.strip()


def deployed() -> DeployedEnv:
    import boto3

    env = os.environ["FINPLAN_TARGET_ENV"]
    from finplan_model.core.config import load_config

    return DeployedEnv(env, boto3.session.Session(region_name=load_config(env).region))
