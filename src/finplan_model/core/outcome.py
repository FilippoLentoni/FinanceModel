"""Contract-shaped outcomes shared by jobs and the control plane.

* :func:`require_valid` validates a document against a pinned contract schema and raises the
  contract's own error code (``VALIDATION_FAILED`` with the first failing pointer, or
  ``INVALID_IDENTIFIER`` / ``OPERATION_NOT_PERMITTED`` when the schema says so).
* :func:`failed_job_result` builds the ``core/v1/job-result.json`` document of a failed run: no
  ``solution_status``, an error envelope with the run's ``correlation_id`` and
  ``artifacts_complete`` false (spec job-execution-controls, "Failure semantics"; SIM-09 maps a
  reconciliation break to ``failed`` + ``INTERNAL``).
"""

from __future__ import annotations

from typing import Any

from finplan_contracts.validate import validate as contract_validate

from .context import RunContext
from .errors import ErrorCode, FinplanError, as_finplan_error

__all__ = ["failed_job_result", "require_valid"]


def require_valid(document: Any, schema: str) -> Any:
    result = contract_validate(document, schema)
    if result.valid:
        return document
    first = result.issues[0]
    code = result.code or ErrorCode.VALIDATION_FAILED
    if code == ErrorCode.INVALID_IDENTIFIER:
        raise FinplanError(code, f"document does not validate against {schema}", field=first.field or first.pointer or "unknown")
    if code == ErrorCode.OPERATION_NOT_PERMITTED:
        raise FinplanError(code, f"document does not validate against {schema}", details={"pointer": first.pointer})
    raise FinplanError.validation(f"document does not validate against {schema}", pointer=first.pointer, schema=schema)


def failed_job_result(ctx: RunContext, exc: BaseException, *, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or ctx.run_id
    if rid is None:
        raise ValueError("a failed job result needs a run_id")
    err = as_finplan_error(exc)
    ctx.log("run_failed", code=err.code, reason=err.details.get("reason"))
    doc: dict[str, Any] = {
        "run_id": rid,
        "completion_status": "failed",
        "error": ctx.error_envelope(err),
        "artifacts": [],
        "artifacts_complete": False,
        "evaluator_version": ctx.evaluator_version,
        "completed_at": ctx.now_ts(),
    }
    if ctx.model_version:
        doc["model_version"] = ctx.model_version
    if ctx.synthetic:
        doc["synthetic"] = True
    return require_valid(doc, "job-result")
