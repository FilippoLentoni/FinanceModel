"""Direct Lambda interface for frozen strategy inference; no training or selection writes.

Serves the MCP tool ``generate_portfolio_recommendation_ppo``: the payload is
``{"environment", "request", "headers", "instruments"?}``; the frozen PPO policy is the one
trained on exactly ``request.instruments`` (see :mod:`finplan_model.rl.policy_selection`).
"""
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
from finplan_model.rl.advisory import PPO_RESPONSE, recommendation

log = logging.getLogger("finplan_model.serving")
log.setLevel(logging.INFO)
_SERVICE = None


def capture_served_input(client,bucket,payload):
    """Content-addressed pre-actor input; concurrent equal retries verify original bytes."""
    from finplan_model.core.artifacts import ArtifactRef,canonical_json_bytes,sha256_checksum,verify_checksum
    data=canonical_json_bytes(payload);checksum=sha256_checksum(data)
    ref=ArtifactRef(artifact_id='served_model_input_'+checksum.split(':')[1],kind='served_model_input',
                    checksum=checksum,content_type='application/json',size_bytes=len(data))
    key='artifacts/served_model_input/'+ref.artifact_id
    try:
        client.put_object(Bucket=bucket,Key=key,Body=data,ContentType=ref.content_type,
                          Metadata={'checksum':checksum,'kind':ref.kind},IfNoneMatch='*')
    except Exception as exc:
        if getattr(exc,'response',{}).get('Error',{}).get('Code') not in ('PreconditionFailed','ConditionalRequestConflict','412','409'):raise
        original=client.get_object(Bucket=bucket,Key=key)['Body'].read()
        verify_checksum(original,checksum,what='served model input')
    return ref.to_dict()


def build_service():
    import boto3
    from finplan_model.control.production_strategy import SsmStrategyParameter
    from finplan_model.core.artifacts import S3ArtifactStore
    from finplan_model.core.aws_clients import s3_client
    from finplan_model.core.platform import HttpPlatformClient
    from finplan_model.rl.policy_selection import list_exports

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
        policy_exports=lambda: list_exports(s3_client(cfg.region, session=session), bucket),
        capture_served_input=lambda payload:capture_served_input(s3_client(cfg.region,session=session),bucket,payload),
        platform=HttpPlatformClient(endpoint, region=cfg.region, credentials=session.get_credentials()),
    )
    return SimpleNamespace(env=env, d=deps, now=SystemClock().now)


def handle(event, service):
    if not isinstance(event, dict) or event.get("environment") != service.env:
        raise FinplanError.validation("serving request belongs to another environment", pointer="/environment")
    if set(event) - {"environment", "request", "headers", "instruments"}:
        raise FinplanError.validation("unknown serving request field", pointer="")
    # ``request.instruments`` is authoritative; a top-level copy must agree with it.
    return require_valid(recommendation(service, event.get("request"), event.get("instruments")), PPO_RESPONSE)


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
