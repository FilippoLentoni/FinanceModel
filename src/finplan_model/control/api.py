"""HTTP routing of the job API (API Gateway REST, IAM/SigV4 auth, Lambda proxy integration).

=======  ======================================  ===============  ============================
method   path                                    operation        success
=======  ======================================  ===============  ============================
POST     ``/v1/jobs``                            ``submit_job``   202 (200 dry run or replay)
GET      ``/v1/jobs``                            ``list_jobs``    200
GET      ``/v1/jobs/{run_id}``                   ``get_job_status``  200
GET      ``/v1/jobs/{run_id}/result``            ``get_job_result``  200
POST     ``/v1/jobs/{run_id}/cancel``            ``cancel_job``   200
POST     ``/v1/jobs/{run_id}/approve``           ``approve_run``  200 (approver role only)
=======  ======================================  ===============  ============================

Errors are contract error envelopes with the HTTP status the platform API uses for the same code.
Every response carries ``X-Correlation-Id`` (a caller-supplied ``x-correlation-id`` is kept when it
is well formed, otherwise one is minted).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from finplan_model.core.errors import ErrorCode, FinplanError

from .auth import Principal
from .service import JobService

__all__ = ["HTTP_STATUS", "JobApi", "ROUTES", "http_status"]

log = logging.getLogger("finplan_model.control.api")

HTTP_STATUS = {
    "VALIDATION_FAILED": 400,
    "INVALID_IDENTIFIER": 400,
    "UNSUPPORTED_CONTRACT_VERSION": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "OPERATION_NOT_PERMITTED": 403,
    "BUDGET_EXCEEDED": 403,
    "NOT_FOUND": 404,
    "CONFLICT": 409,
    "IMMUTABLE_RECORD": 409,
    "IDEMPOTENCY_KEY_REUSED": 422,
    "PRECONDITION_FAILED": 422,
    "RATE_LIMITED": 429,
    "INTERNAL": 500,
    "DEPENDENCY_UNAVAILABLE": 503,
}
_CID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}\Z")
_RUN = r"(?P<run_id>[A-Za-z0-9_]{1,64})"
ROUTES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("GET", re.compile(r"^/v1/recommendations/?\Z"), "recommend_portfolio"),
    ("GET", re.compile(r"^/v1/performance-evidence/?\Z"), "performance_evidence"),
    ("PUT", re.compile(r"^/v1/advisory-policy/?\Z"), "activate_advisory"),
    ("POST", re.compile(r"^/v1/jobs/?\Z"), "submit_job"),
    ("GET", re.compile(r"^/v1/jobs/?\Z"), "list_jobs"),
    ("GET", re.compile(rf"^/v1/jobs/{_RUN}/?\Z"), "get_job_status"),
    ("GET", re.compile(rf"^/v1/jobs/{_RUN}/result/?\Z"), "get_job_result"),
    ("POST", re.compile(rf"^/v1/jobs/{_RUN}/cancel/?\Z"), "cancel_job"),
    ("POST", re.compile(rf"^/v1/jobs/{_RUN}/approve/?\Z"), "approve_run"),
)
_LIST_QUERY = {"state", "page_size", "next_token"}
_MAX_BODY = 256 * 1024


def http_status(code: str) -> int:
    return HTTP_STATUS.get(code, 500)


class JobApi:
    def __init__(self, service: JobService) -> None:
        self.service = service

    def _cid(self, headers: Mapping[str, str]) -> str:
        cid = headers.get("x-correlation-id", "")
        return cid if _CID.match(cid) else self.service.d.ids.correlation_id()

    def handle(self, event: Mapping[str, Any]) -> dict[str, Any]:
        headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items() if v is not None}
        cid = self._cid(headers)
        try:
            status, body = self._dispatch(event, cid)
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

    @staticmethod
    def _body(event: Mapping[str, Any]) -> Any:
        raw = event.get("body")
        if raw in (None, ""):
            return {}
        if event.get("isBase64Encoded"):
            raise FinplanError.validation("binary bodies are not accepted", pointer="")
        if len(str(raw)) > _MAX_BODY:
            raise FinplanError.validation("request body is too large", pointer="")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise FinplanError.validation("request body is not valid JSON", pointer="") from None

    def _dispatch(self, event: Mapping[str, Any], cid: str) -> tuple[int, Any]:
        method = str(event.get("httpMethod") or "").upper()
        path = str(event.get("path") or "")
        route = next(((op, m) for meth, rx, op in ROUTES if meth == method and (m := rx.match(path))), None)
        if route is None:
            raise FinplanError(ErrorCode.NOT_FOUND, "no such route", details={"record_type": "route"})
        op, match = route
        identity = (event.get("requestContext") or {}).get("identity") or {}
        principal = Principal.from_arn(identity.get("userArn") or identity.get("caller"))
        svc = self.service
        run_id = match.groupdict().get("run_id")
        if op in ("recommend_portfolio", "performance_evidence", "activate_advisory"):
            from finplan_model.rl.advisory import activate, recommendation
            from finplan_model.rl.performance import performance_evidence
            if op == "activate_advisory":
                body = self._body(event)
            else:
                raw = (event.get("queryStringParameters") or {}).get("request", "")
                if len(raw) > 16000:
                    raise FinplanError.validation("request exceeds the read bound", pointer="/request")
                try: body = json.loads(raw)
                except (ValueError, TypeError): raise FinplanError.validation("request must contain a JSON object", pointer="/request") from None
            if op == "recommend_portfolio": return 200, recommendation(svc, body)
            if op == "performance_evidence": return 200, performance_evidence(svc, body)
            return 200, activate(svc, principal, body)
        if op == "submit_job":
            return svc.submit_job(principal, self._body(event), correlation_id=cid)
        if op == "list_jobs":
            query = dict(event.get("queryStringParameters") or {})
            unknown = set(query) - _LIST_QUERY
            if unknown:
                raise FinplanError.validation("unknown query parameter", pointer="")
            return 200, svc.list_jobs(principal, query)
        if op == "get_job_status":
            return 200, svc.get_job_status(principal, run_id)
        if op == "get_job_result":
            return 200, svc.get_job_result(principal, run_id)
        if op == "cancel_job":
            return svc.cancel_job(principal, run_id, self._body(event))
        if op == "approve_run":
            return 200, svc.approve_run(principal, run_id, self._body(event))
        raise FinplanError.internal("unrouted operation")  # pragma: no cover
