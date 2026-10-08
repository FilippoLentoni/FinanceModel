"""Job result documents written by the container (task 6.3; spec experiment-job-interface "Result
retrieval"; JOB-06).

:func:`succeeded_result` builds the contract ``core/v1/job-result.json`` document of a finished job:
``completion_status`` ``succeeded`` with the run-level ``solution_status`` (an infeasible optimizer
run is still ``succeeded``), ``model_version``, ``evaluator_version``, ``dataset_checksum``,
``configuration_id``, ``input_snapshot_id``, trusted artifact references with checksums and the
finance ``run_result`` payload with **separate** portfolio-performance, model-accuracy and
compute-cost sections. Nothing in it is a storage location: the document is checked against the
contract schema and a storage-location scan before it is written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from finplan_model.control.validation import find_storage_location
from finplan_model.core.artifacts import ArtifactRef
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import require_valid

__all__ = ["performance_section", "succeeded_result"]


def performance_section(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Numeric portfolio metrics only (the contract section maps names to numbers)."""
    return {k: float(v) for k, v in sorted(metrics.items()) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}


def succeeded_result(
    ctx: RunContext,
    spec: Mapping[str, Any],
    *,
    solution_status: str,
    artifacts: Sequence[ArtifactRef],
    performance: Mapping[str, Any],
    accuracy: Mapping[str, Any] | None = None,
    dataset_checksum: str | None = None,
    instance_seconds: float | None = None,
    proposed_allocation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "performance": performance_section(performance),
        "accuracy": performance_section(accuracy or {}),
        "compute_cost": {},
    }
    est = (spec.get("cost_estimate") or {}).get("estimated_usd_upper_bound")
    if est is not None:
        payload["compute_cost"]["estimated_usd"] = float(est)
    if instance_seconds is not None:
        payload["compute_cost"]["instance_seconds"] = max(0.0, float(instance_seconds))
    if proposed_allocation:
        payload["proposed_allocation"] = dict(proposed_allocation)
    if ctx.synthetic:
        payload["synthetic"] = True
    doc: dict[str, Any] = {
        "run_id": spec["run_id"],
        "completion_status": "succeeded",
        "solution_status": solution_status,
        "artifacts": [a.to_dict() for a in artifacts],
        "artifacts_complete": True,
        "configuration_id": spec["configuration_id"],
        "input_snapshot_id": spec["input_snapshot_id"],
        "evaluator_version": ctx.evaluator_version,
        "domain": spec.get("domain", "finance"),
        "domain_schema_version": spec.get("domain_schema_version", "1.0"),
        "payload": payload,
        "completed_at": ctx.now_ts(),
    }
    if spec.get("model_version"):
        doc["model_version"] = spec["model_version"]
    if dataset_checksum:
        doc["dataset_checksum"] = dataset_checksum
    if ctx.synthetic:
        doc["synthetic"] = True
    if find_storage_location(doc) is not None:
        raise FinplanError.internal("a job result must not contain storage locations", reason="result_leaks_location")
    return require_valid(doc, "job-result")
