"""Shared builders for control-plane tests: a fully offline :class:`JobService` (in-memory or
DynamoDB-on-moto store, static settings, the fake SageMaker client, in-memory run IO, frozen clock).

Prices, ARNs and image references here are synthetic placeholders (no account identifiers)."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from typing import Any

from finplan_contracts import __version__ as CONTRACT_VERSION

from finplan_model.control.api import JobApi
from finplan_model.control.auth import Principal
from finplan_model.control.service import JobService, ServiceDeps
from finplan_model.control.settings import StaticSettings
from finplan_model.control.store import InMemoryRunStore
from finplan_model.control.wakeup import InMemoryDispatchSchedule
from finplan_model.core.clock import FrozenClock
from finplan_model.core.config import EnvConfig, load_config
from finplan_model.core.ids import IdMinter
from finplan_model.jobs.runio import InMemoryRunIO
from tests.fakes.sagemaker import FakeSageMaker

SID = "snap_01KDVDNAZ83BAMMYCEGWF33DPM"
IMAGE_URI = "registry.invalid/financemodel-cpu@sha256:" + "ab" * 32
JOB_ROLE_ARN = "arn:aws:iam::<account-id>:role/finplan-beta-financemodel-job-role"
APPROVER_ROLE = "finplan-beta-financemodel-approver-role"
CANDIDATE_ROLE = "finplan-beta-financialplanning-candidate-role"


def arn(role: str, session: str = "s1") -> str:
    return f"arn:aws:sts::<account-id>:assumed-role/{role}/{session}"


SUBMITTER = arn("finplan-beta-financelambdastool-submitter-role")
READER = arn("finplan-beta-financelambdastool-reader-role")
AGENT = arn("finplan-beta-financeagent-runtime-role")
PIPELINE = arn("finplan-beta-financemodel-pipeline-role")
APPROVER = arn(APPROVER_ROLE, "human")
CANDIDATE = arn(CANDIDATE_ROLE)

#: Synthetic test price (not a market or AWS price).
PRICES = {"retrieved_at": "2026-01-01T00:00:00Z", "currency": "USD", "usd_per_hour": {"ml.m5.xlarge": 0.25}}
#: 0.25 USD/h x 900 s + 0.01 storage
BACKTEST_ESTIMATE = 0.0725


def request(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "domain": "finance",
        "domain_schema_version": "1.0",
        "job_type": "run_backtest",
        "purpose": "research",
        "dry_run": False,
        "input_snapshot_id": SID,
        "configuration": {
            "domain": "finance",
            "domain_schema_version": "1.0",
            "payload": {
                "strategy": "equal_weight",
                "objective": "backtest",
                "universe": ["AGG", "SPY"],
                "rebalance_frequency": "monthly",
                "constraints": {"long_only": True, "max_weight": 1.0},
                "fees": {"transaction_cost_bps": 1},
            },
            "synthetic": True,
        },
        "evaluation_window": {"start": "2026-01-05", "end": "2026-03-27"},
        "idempotency_key": "client-key-0001",
        "contract_version": CONTRACT_VERSION,
        "synthetic": True,
    }
    for k, v in over.items():
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    return body


def env_config(env: str = "beta", mutate: Any = None) -> EnvConfig:
    cfg = load_config(env)
    if mutate is None:
        return cfg
    raw = copy.deepcopy(dict(cfg.raw))
    mutate(raw)
    return EnvConfig(env=env, raw=raw)


class Harness:
    def __init__(self, *, cfg: EnvConfig | None = None, store: Any = None, auto_approve: float | None = None, platform: Any = None, **settings: Any) -> None:
        self.cfg = cfg or env_config()
        self.clock = FrozenClock("2026-01-05T09:00:00Z")
        self.ids = IdMinter.seeded(self.clock, 7)
        self.store = store or InMemoryRunStore()
        self.sagemaker = FakeSageMaker()
        self.run_io = InMemoryRunIO()
        defs = {jt: {"job_type": jt, "image_uri": IMAGE_URI, "deployed": True} for jt in self.cfg.raw["job_types"]}
        kwargs: dict[str, Any] = {
            "prices": PRICES,
            "auto_approve": auto_approve,
            "approver_role": APPROVER_ROLE,
            "production_principals": [CANDIDATE_ROLE],
            "job_definitions": defs,
            "job_role": JOB_ROLE_ARN,
            "storage": "example-research-bucket",
        }
        kwargs.update(settings)
        self.settings = StaticSettings(self.cfg, **kwargs)
        self.kicks = 0
        self.kicked: list[str] = []
        self.wakeup = InMemoryDispatchSchedule()
        self.deps = ServiceDeps(cfg=self.cfg, store=self.store, settings=self.settings, sagemaker=self.sagemaker, run_io=self.run_io, clock=self.clock, ids=self.ids, platform=platform, kick=self._kick, wakeup=self.wakeup)
        self.service = JobService(self.deps)
        self.api = JobApi(self.service)

    def _kick(self, run_id: str) -> None:
        self.kicks += 1
        self.kicked.append(run_id)

    # ------------------------------------------------------------------ calls
    def submit(self, principal: str = SUBMITTER, **over: Any) -> tuple[int, dict[str, Any]]:
        return self.service.submit_job(Principal.from_arn(principal), request(**over), correlation_id="corr-test-submit-0001")

    def call(self, method: str, path: str, *, principal: str | None = SUBMITTER, body: Any = None, query: Mapping[str, str] | None = None, headers: Mapping[str, str] | None = None) -> tuple[int, dict[str, Any], dict[str, str]]:
        event: dict[str, Any] = {
            "httpMethod": method,
            "path": path,
            "headers": dict(headers or {}),
            "queryStringParameters": dict(query) if query else None,
            "body": None if body is None else (body if isinstance(body, str) else json.dumps(body)),
            "requestContext": {"identity": {"userArn": principal} if principal else {}},
        }
        resp = self.api.handle(event)
        return resp["statusCode"], json.loads(resp["body"]), resp["headers"]

    def run(self, run_id: str) -> dict[str, Any]:
        r = self.store.get_run(run_id)
        assert r is not None
        return r

    def created_jobs(self) -> list[dict[str, Any]]:
        return [kw for op, kw in self.sagemaker.calls if op == "CreateProcessingJob"]

    def event(self, job_name: str, status: str, **extra: Any) -> dict[str, Any]:
        return self.service.handle_sagemaker_event({"source": "aws.sagemaker", "detail-type": "SageMaker Processing Job State Change", "detail": {"ProcessingJobName": job_name, "ProcessingJobStatus": status, **extra}})

    def start(self, **over: Any) -> str:
        """Submit (auto-approved) and dispatch until the run is starting; returns run_id."""
        code, resp = self.submit(**over)
        assert code == 202, resp
        run_id = resp["run_id"]
        if resp["state"] == "awaiting_approval":
            self.service.approve_run(Principal.from_arn(APPROVER), run_id, {})
        self.service.dispatch()
        assert self.run(run_id)["state"] == "starting", self.run(run_id)["state"]
        return run_id

    def succeed(self, run_id: str, solution_status: str = "optimal") -> dict[str, Any]:
        run = self.run(run_id)
        self.run_io.put_result(run_id, job_result(run, solution_status))
        self.sagemaker.set_status(run["job_name"], "Completed")
        return self.event(run["job_name"], "Completed")


def job_result(run: Mapping[str, Any], solution_status: str = "optimal") -> dict[str, Any]:
    """What the container writes for a succeeded run (contract job-result)."""
    return {
        "run_id": run["run_id"],
        "completion_status": "succeeded",
        "solution_status": solution_status,
        "artifacts": [{"artifact_id": "run_artifact_" + "c" * 40, "owner": "financemodel", "kind": "run_artifact", "checksum": "sha256:" + "c" * 64, "content_type": "application/json", "synthetic": True}],
        "artifacts_complete": True,
        "configuration_id": run["configuration_id"],
        "input_snapshot_id": run["input_snapshot_id"],
        "evaluator_version": "1.0.0",
        "dataset_checksum": "sha256:" + "d" * 64,
        "domain": "finance",
        "domain_schema_version": "1.0",
        "payload": {"performance": {"net_cumulative_return": 0.01, "max_drawdown": -0.02}, "accuracy": {}, "compute_cost": {"estimated_usd": BACKTEST_ESTIMATE}, "synthetic": True},
        "completed_at": "2026-01-05T09:20:00Z",
        "synthetic": True,
    }


def events_logged(caplog: Any, name: str) -> list[dict[str, Any]]:
    out = []
    for rec in caplog.records:
        try:
            d = json.loads(rec.getMessage())
        except (ValueError, TypeError):
            continue
        if isinstance(d, dict) and d.get("event") == name:
            out.append(d)
    return out


def enable_logs(caplog: Any) -> None:
    caplog.set_level(logging.INFO)
