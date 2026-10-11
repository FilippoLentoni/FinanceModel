"""Direct Lambda interface for frozen strategy inference; no training or selection writes."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from finplan_model.core.clock import SystemClock
from finplan_model.core.config import load_config
from finplan_model.core.errors import FinplanError, as_finplan_error
from finplan_model.core.outcome import require_valid
from finplan_model.rl.advisory import recommendation

log = logging.getLogger("finplan_model.serving")
log.setLevel(logging.INFO)
_SERVICE = None


def build_service():
    import boto3
    from finplan_model.control.production_strategy import SsmStrategyParameter
    from finplan_model.core.artifacts import S3ArtifactStore
    from finplan_model.core.aws_clients import s3_client
    from finplan_model.core.platform import HttpPlatformClient

    env = os.environ["FINPLAN_ENVIRONMENT"]
    cfg = load_config(env, Path(os.environ["FINPLAN_CONFIG_DIR"]))
    session = boto3.session.Session(region_name=cfg.region)
    ssm = session.client("ssm")
    bucket = ssm.get_parameter(Name=cfg.ssm_name("config", "research-storage-ref"))["Parameter"]["Value"]
    endpoint = ssm.get_parameter(Name=cfg.ssm["plan_endpoint"])["Parameter"]["Value"]
    deps = SimpleNamespace(
        advisory_parameter=SsmStrategyParameter(ssm, cfg.ssm_name("config", "advisory-policy")),
        research_plan_parameter=SsmStrategyParameter(ssm, f"/finplan/{env}/financialplanning/config/research-plan-ref"),
        artifacts=S3ArtifactStore(s3_client(cfg.region, session=session), bucket),
        platform=HttpPlatformClient(endpoint, region=cfg.region, credentials=session.get_credentials()),
    )
    return SimpleNamespace(env=env, d=deps, now=SystemClock().now)


def handle(event, service):
    if not isinstance(event, dict) or event.get("environment") != service.env:
        raise FinplanError.validation("serving request belongs to another environment", pointer="/environment")
    if set(event) - {"environment", "request", "headers"}:
        raise FinplanError.validation("unknown serving request field", pointer="")
    return require_valid(recommendation(service, event.get("request")), "tools/recommend-portfolio-response")


def handler(event, context=None):
    global _SERVICE
    headers = event.get("headers", {}) if isinstance(event, dict) else {}
    cid = headers.get("X-Correlation-Id") if isinstance(headers, dict) else None
    try:
        if _SERVICE is None:
            _SERVICE = build_service()
        result = handle(event, _SERVICE)
        status = "OK"
    except Exception as exc:
        err = as_finplan_error(exc)
        result, status = err.to_envelope(cid), err.code
    log.info(json.dumps({"event": "strategy_inference", "environment": getattr(_SERVICE, "env", None), "correlation_id": cid,
                         "caller": headers.get("X-Finplan-Caller") if isinstance(headers, dict) else None,
                         "strategy": (result.get("recommendation") or {}).get("strategy"), "outcome": status}, sort_keys=True))
    return result
