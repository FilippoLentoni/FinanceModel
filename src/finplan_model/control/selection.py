"""The ``strategy-selection`` Lambda: ``GET`` and ``PUT /v1/production-strategy`` on the job API
(spec production-strategy-setting; design M3). Wired by ``infra.stacks.control`` with its own role
``finplan-<env>-financemodel-strategy-selection-role``, the only principal allowed
``ssm:PutParameter`` / ``ssm:DeleteParameter`` on ``/finplan/<env>/financemodel/config/production-strategy``.

=======  ==============================  =====================================================
method   path                            operation
=======  ==============================  =====================================================
GET      ``/v1/production-strategy``     ``get`` (``strategy`` is null when none is selected)
PUT      ``/v1/production-strategy``     body ``{"action": "get"|"set"|"clear", ...}``
=======  ==============================  =====================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from finplan_model.core.errors import ErrorCode, FinplanError

from .api import http_status
from .auth import Principal
from .production_strategy import StrategySelectionService

__all__ = ["SELECTION_ROUTES", "SelectionApi", "build_selection_service", "selection_handler"]

log = logging.getLogger("finplan_model.control.selection")

SELECTION_ROUTES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GET", re.compile(r"^/v1/production-strategy/?\Z")),
    ("PUT", re.compile(r"^/v1/production-strategy/?\Z")),
)
_CID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}\Z")
_MAX_BODY = 16 * 1024
_SERVICE: Any = None


class SelectionApi:
    def __init__(self, service: StrategySelectionService) -> None:
        self.service = service

    def handle(self, event: Mapping[str, Any]) -> dict[str, Any]:
        headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items() if v is not None}
        cid = headers.get("x-correlation-id", "")
        cid = cid if _CID.match(cid) else self.service.ids.correlation_id()
        try:
            status, body = self._dispatch(event)
        except FinplanError as exc:
            status, body = http_status(exc.code), exc.to_envelope(cid)
        except Exception:  # noqa: BLE001 - never leak a traceback
            log.exception("unhandled error (correlation_id=%s)", cid)
            status, body = 500, FinplanError.internal("unexpected failure").to_envelope(cid)
        return {
            "statusCode": status,
            "headers": {"Content-Type": "application/json", "X-Correlation-Id": cid, "Cache-Control": "no-store"},
            "body": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        }

    def _dispatch(self, event: Mapping[str, Any]) -> tuple[int, Any]:
        method = str(event.get("httpMethod") or "").upper()
        path = str(event.get("path") or "")
        if not any(m == method and rx.match(path) for m, rx in SELECTION_ROUTES):
            raise FinplanError(ErrorCode.NOT_FOUND, "no such route", details={"record_type": "route"})
        identity = (event.get("requestContext") or {}).get("identity") or {}
        principal = Principal.from_arn(identity.get("userArn") or identity.get("caller"))
        if method == "GET":
            return 200, self.service.get()
        raw = event.get("body")
        if raw in (None, "") or event.get("isBase64Encoded") or len(str(raw)) > _MAX_BODY:
            raise FinplanError.validation("a JSON request body is required", pointer="")
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise FinplanError.validation("request body is not valid JSON", pointer="") from None
        return self.service.handle(principal, body)


def build_selection_service(env: str | None = None, *, session: Any = None) -> StrategySelectionService:  # pragma: no cover - needs AWS
    import boto3

    from finplan_model.core.clock import SystemClock
    from finplan_model.core.config import load_config
    from finplan_model.core.ids import IdMinter

    from .production_strategy import SsmStrategyParameter
    from .store import DynamoRunStore

    env = env or os.environ["FINPLAN_ENVIRONMENT"]
    config_dir = os.environ.get("FINPLAN_CONFIG_DIR")
    cfg = load_config(env, Path(config_dir) if config_dir else None)
    session = session or boto3.session.Session(region_name=cfg.region)
    clock = SystemClock()

    def audit(record: dict[str, Any]) -> None:
        print(json.dumps(record, sort_keys=True))  # also in the function's log group

    return StrategySelectionService(
        cfg,
        SsmStrategyParameter(session.client("ssm"), cfg.ssm["production_strategy"]),
        DynamoRunStore(session.client("dynamodb"), os.environ["FINPLAN_RUNS_TABLE"]),
        clock,
        IdMinter(clock),
        audit_log=audit,
    )


def selection_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:  # pragma: no cover - thin AWS wrapper
    global _SERVICE
    try:
        if _SERVICE is None:
            _SERVICE = build_selection_service()
    except Exception as exc:  # noqa: BLE001
        from finplan_model.core.errors import as_finplan_error

        err = as_finplan_error(exc)
        return {"statusCode": http_status(err.code), "headers": {"Content-Type": "application/json"}, "body": json.dumps(err.to_envelope(None))}
    return SelectionApi(_SERVICE).handle(event)
