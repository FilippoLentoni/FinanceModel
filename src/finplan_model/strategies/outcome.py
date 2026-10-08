"""Run-level outcome of a baseline evaluation (BASE-05; contracts CS-07).

``completion_status`` and ``solution_status`` are separate: an optimizer run whose decisions were
infeasible still **completed** (``succeeded``) and reports ``solution_status`` ``infeasible`` - it is
never a job failure. Partially infeasible runs (some decisions fell back to holding current weights)
report ``feasible``. Infeasible and unbounded runs produce no production-candidate staging output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["NO_STAGING_STATUSES", "job_outcome"]

#: Run-level solution statuses that never produce a staged production-candidate bundle.
NO_STAGING_STATUSES = ("infeasible", "unbounded")


def job_outcome(result: Any) -> dict[str, Any]:
    """``{completion_status, solution_status, fallback_decisions, stage_output_allowed}`` of an
    :class:`~finplan_model.evaluate.EvaluationResult` (or its stored ``to_dict()`` form)."""
    if isinstance(result, Mapping):
        sim = result["simulation"]
        decisions = sim["decisions"]
        status = str(result.get("solution_status") or sim["solution_status"])
    else:
        decisions = result.simulation.decisions
        status = result.solution_status
    fallbacks = [{"decision_session": d["decision_session"], "solution_status": d["solution_status"]} for d in decisions if d.get("action") in ("fallback_hold_current", "rejected_hold_current") and d["solution_status"] in NO_STAGING_STATUSES]
    return {"completion_status": "succeeded", "solution_status": status, "fallback_decisions": fallbacks, "stage_output_allowed": status not in NO_STAGING_STATUSES}
