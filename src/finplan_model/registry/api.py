"""Registry routes of the job API (task 9.3; REG-03), served by the ``job-registry-lookup`` Lambda.

=======  ==================================================  ===================================
method   path                                                answer
=======  ==================================================  ===================================
GET      ``/v1/registry/lineage/{run_id}?model_version=mv_X``  200 ``{run_id, model_version, matches:
                                                             true, strategy, image_digest,
                                                             param_schema_version, status}``;
                                                             404 ``NOT_FOUND`` (``details.record_type``
                                                             ``model_version`` or ``run_lineage``)
GET      ``/v1/registry/model-versions/{model_version}``     200 the record and its lifecycle status
=======  ==================================================  ===================================

The platform resolves ``/finplan/<env>/financemodel/model/registry-ref`` (the value is
``<job-endpoint>/v1/registry``) and calls the lineage route with SigV4 before it commits a staged
bundle. Responses carry identifiers and digests only, never a storage location.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from finplan_model.core.errors import ErrorCode, FinplanError

from .registry import ModelRegistry

__all__ = ["RegistryApi"]

log = logging.getLogger("finplan_model.registry.api")
_LINEAGE = re.compile(r"^/v1/registry/lineage/(?P<run_id>[A-Za-z0-9_]{1,64})/?\Z")
_RECORD = re.compile(r"^/v1/registry/model-versions/(?P<model_version>[A-Za-z0-9_]{1,64})/?\Z")
_CID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}\Z")
_STATUS = {"VALIDATION_FAILED": 400, "INVALID_IDENTIFIER": 400, "NOT_FOUND": 404, "DEPENDENCY_UNAVAILABLE": 503, "INTERNAL": 500}
_PUBLIC_FIELDS = ("model_version", "strategy", "image_digest", "param_schema_version", "artifact_checksum", "registered_at", "status", "synthetic")


class RegistryApi:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def handle(self, event: Mapping[str, Any]) -> dict[str, Any]:
        headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items() if v is not None}
        cid = headers.get("x-correlation-id", "")
        if not _CID.match(cid):
            cid = self.registry.ids.correlation_id()
        try:
            status, body = 200, self._dispatch(event)
        except FinplanError as exc:
            status, body = _STATUS.get(exc.code, 500), exc.to_envelope(cid)
        except Exception:  # noqa: BLE001 - never leak a traceback
            log.exception("unhandled registry error (correlation_id=%s)", cid)
            status, body = 500, FinplanError.internal("unexpected failure").to_envelope(cid)
        return {"statusCode": status, "headers": {"Content-Type": "application/json", "X-Correlation-Id": cid, "Cache-Control": "no-store"}, "body": json.dumps(body, sort_keys=True, separators=(",", ":"))}

    def _dispatch(self, event: Mapping[str, Any]) -> dict[str, Any]:
        method = str(event.get("httpMethod") or "").upper()
        path = str(event.get("path") or "")
        if method != "GET":
            raise FinplanError(ErrorCode.NOT_FOUND, "no such route", details={"record_type": "route"})
        if m := _LINEAGE.match(path):
            query = dict(event.get("queryStringParameters") or {})
            if set(query) - {"model_version"}:
                raise FinplanError.validation("unknown query parameter", pointer="")
            mv = query.get("model_version")
            if not mv:
                raise FinplanError.validation("model_version is required", pointer="/model_version")
            return self.registry.verify_lineage(m.group("run_id"), mv)
        if m := _RECORD.match(path):
            record = self.registry.get(m.group("model_version"))
            return {k: record[k] for k in _PUBLIC_FIELDS if k in record}
        raise FinplanError(ErrorCode.NOT_FOUND, "no such route", details={"record_type": "route"})
