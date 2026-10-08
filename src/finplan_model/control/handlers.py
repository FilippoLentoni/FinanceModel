"""Lambda entry points of the job control plane (design D1). Wired by ``infra`` (task 10.3).

==============================================================  ==========================================
handler                                                         trigger
==============================================================  ==========================================
``finplan_model.control.handlers.api_handler``                  API Gateway REST (IAM auth), job API routes
``finplan_model.control.handlers.dispatcher_handler``           EventBridge Scheduler schedule (armed only
                                                                while runs are pending, design D1) and
                                                                asynchronous kicks ``{"run_id": ...}``
``finplan_model.control.handlers.state_change_handler``         EventBridge rule ``aws.sagemaker`` /
                                                                ``SageMaker Processing Job State Change``
                                                                for names starting ``fm-<env>-``
==============================================================  ==========================================

Environment variables (names only; values are set by the stack, never committed):

* ``FINPLAN_ENVIRONMENT`` - ``beta``, ``gamma`` or ``prod``;
* ``FINPLAN_RUNS_TABLE`` - the run/lease/idempotency table name (:data:`~finplan_model.control.store.TABLE_SPEC`);
* ``FINPLAN_DISPATCHER_FUNCTION`` - optional; the API handler invokes the dispatcher asynchronously
  (payload ``{"run_id": ...}``) after a run becomes ``queued`` so it starts without waiting for a tick;
* ``FINPLAN_DISPATCH_SCHEDULE`` - optional; the dispatcher schedule's name. The API handler, the
  dispatcher and the state handler arm and disarm it (:mod:`finplan_model.control.wakeup`): it runs
  only while runs are queued, active or awaiting an approval deadline;
* ``FINPLAN_CONFIG_DIR`` - optional override of the bundled ``config/`` directory.

Everything else (prices, thresholds, role references, research storage, job definitions, the
platform endpoint) is read from SSM at run time (:class:`~finplan_model.control.settings.SsmSettings`).
None of these handlers runs strategy code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from finplan_model.core.clock import SystemClock
from finplan_model.core.config import load_config
from finplan_model.core.ids import IdMinter

__all__ = ["api_handler", "build_service", "dispatcher_handler", "state_change_handler"]

_SERVICE: Any = None


def build_service(env: str | None = None, *, session: Any = None) -> Any:
    """Build the deployed :class:`~finplan_model.control.service.JobService` (boto3 clients, SSM settings)."""
    import boto3

    from finplan_model.core.platform import HttpPlatformClient
    from finplan_model.jobs.runio import S3RunIO

    from .service import JobService, ServiceDeps
    from .settings import SsmSettings
    from .store import DynamoRunStore

    env = env or os.environ["FINPLAN_ENVIRONMENT"]
    config_dir = os.environ.get("FINPLAN_CONFIG_DIR")
    cfg = load_config(env, Path(config_dir) if config_dir else None)
    session = session or boto3.session.Session(region_name=cfg.region)
    ssm = session.client("ssm")
    settings = SsmSettings(ssm, cfg)
    clock = SystemClock()
    bucket = settings.research_storage()
    if not bucket:
        from finplan_model.core.errors import FinplanError

        raise FinplanError.dependency_unavailable("research storage is not published in this environment", retryable=True)
    platform = None
    try:
        endpoint = ssm.get_parameter(Name=cfg.ssm["plan_endpoint"])["Parameter"]["Value"]
        platform = HttpPlatformClient(endpoint, region=cfg.region, credentials=session.get_credentials())
    except Exception:  # noqa: BLE001 - without the platform endpoint the snapshot check runs in the job
        platform = None
    kick = None
    fn = os.environ.get("FINPLAN_DISPATCHER_FUNCTION")
    if fn:
        lam = session.client("lambda")

        def kick(run_id: str) -> None:
            try:
                lam.invoke(FunctionName=fn, InvocationType="Event", Payload=json.dumps({"run_id": run_id}).encode("utf-8"))
            except Exception:  # noqa: BLE001 - the armed schedule still starts the run
                pass

    wakeup = None
    schedule_name = os.environ.get("FINPLAN_DISPATCH_SCHEDULE")
    if schedule_name:
        from .wakeup import SchedulerDispatchSchedule

        wakeup = SchedulerDispatchSchedule(session.client("scheduler"), schedule_name)

    from finplan_model.core.aws_clients import s3_client

    s3 = s3_client(cfg.region, session=session)
    ids = IdMinter(clock)
    deps = ServiceDeps(
        cfg=cfg,
        store=DynamoRunStore(session.client("dynamodb"), os.environ["FINPLAN_RUNS_TABLE"]),
        settings=settings,
        sagemaker=session.client("sagemaker"),
        run_io=S3RunIO(s3, bucket),
        clock=clock,
        ids=ids,
        platform=platform,
        kick=kick,
        wakeup=wakeup,
    )
    _wire_registry_and_staging(deps, cfg, ssm, s3, platform)
    return JobService(deps)


def _wire_registry_and_staging(deps: Any, cfg: Any, ssm: Any, s3: Any, platform: Any) -> None:
    """Task group 9 hooks: ``model_version`` minting at submission (REG-04), run lineage (REG-03) and
    the platform's staged-output outcome on results (RST-05). Without a published registry reference
    the hooks stay off (the platform's lineage check then answers DEPENDENCY_UNAVAILABLE)."""
    from finplan_model.staging import platform_outcome_hook

    actor = f"finplan-{cfg.env}-financemodel-job-api-handler-role"
    try:
        registry_bucket = ssm.get_parameter(Name=cfg.ssm_name("config", "registry-storage-ref"))["Parameter"]["Value"]
    except Exception:  # noqa: BLE001 - not published yet
        registry_bucket = None
    if registry_bucket:
        from finplan_model.registry import ModelRegistry, S3RegistryStore, lineage_result_hook, model_version_resolver

        registry = ModelRegistry(S3RegistryStore(s3, registry_bucket), clock=deps.clock, ids=deps.ids)
        deps.model_version_resolver = model_version_resolver(registry, actor=actor)
        deps.result_hooks.append(lineage_result_hook(registry, actor=actor))
    if platform is not None:
        deps.result_hooks.append(platform_outcome_hook(platform))


def _service() -> Any:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = build_service()
    return _SERVICE


def api_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    from .api import JobApi

    try:
        svc = _service()
    except Exception as exc:  # noqa: BLE001 - surface as a contract envelope, never a traceback
        from finplan_model.core.errors import as_finplan_error

        err = as_finplan_error(exc)
        from .api import http_status

        return {"statusCode": http_status(err.code), "headers": {"Content-Type": "application/json"}, "body": json.dumps(err.to_envelope(None))}
    return JobApi(svc).handle(event)


def dispatcher_handler(event: dict[str, Any] | None = None, context: Any = None) -> dict[str, Any]:
    hint = (event or {}).get("run_id") if isinstance(event, dict) else None
    return _service().dispatch(hint_run_id=str(hint) if isinstance(hint, str) and hint else None)


def state_change_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    return _service().handle_sagemaker_event(event)
