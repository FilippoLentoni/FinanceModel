"""Link the platform's acceptance outcome to a staged run's result (spec run-output-staging,
"Platform validation decides commitment"; task 9.4; RST-05).

FinanceModel never triggers acceptance (the platform-side caller invokes
``POST /v1/plans/{plan_id}/staged-outputs/{run_id}/accept``). It only **reads** the recorded outcome
from ``GET /v1/staged-outputs/{run_id}`` (contract ``api/get-staged-output-response``) with the job
API handler role (granted by the platform as ``financemodel-job-api``) and folds it into the
result's ``staging`` block:

==========================================  ======================  ===================  ================
platform outcome                            ``platform_validation``  ``plan_version_id``  run result shows
==========================================  ======================  ===================  ================
none yet (``NOT_FOUND``)                    ``pending``              no                   ``staged``
``accepted`` + ``validated``                ``validated``            yes (from platform)  ``staged``
``accepted`` + ``pending_validation``       ``pending_validation``   yes (from platform)  ``staged``
``accepted`` + ``invalid``                  ``invalid``              **no**               ``staged``
``rejected``                                ``rejected``             no                   ``staged``
``no_version``                              ``no_version``           no                   ``staged``
==========================================  ======================  ===================  ================

The run itself is always reported as ``staged``, never as a plan version; a ``plan_version_id``
appears only after the platform returned one for a version that is not ``invalid``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import require_valid

__all__ = ["link_platform_outcome", "platform_outcome_hook", "staging_view"]

_CLAIMABLE = frozenset({"validated", "pending_validation"})


def staging_view(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """The ``staging`` block fields derived from one platform outcome document."""
    require_valid(dict(outcome), "api/get-staged-output-response")
    kind = outcome["outcome"]
    view: dict[str, Any] = {"platform_outcome": kind, "platform_decided_at": outcome["decided_at"]}
    if kind == "accepted":
        status = str(outcome.get("plan_version_status") or "pending_validation")
        view["platform_validation"] = status
        if status in _CLAIMABLE and outcome.get("plan_version_id"):
            view["plan_version_id"] = outcome["plan_version_id"]
    elif kind == "rejected":
        view["platform_validation"] = "rejected"
        if outcome.get("rejection_kind"):
            view["rejection_kind"] = outcome["rejection_kind"]
        err = outcome.get("error") or {}
        if err.get("code"):
            view["platform_error_code"] = err["code"]
    else:
        view["platform_validation"] = "no_version"
    return view


def link_platform_outcome(result: Mapping[str, Any], client: Any) -> dict[str, Any]:
    """Return ``result`` with its ``staging`` block linked to the platform's recorded outcome."""
    out = dict(result)
    staging = dict(out.get("staging") or {})
    if staging.get("status") != "staged":
        return out
    try:
        outcome = client.get_staged_output(str(out["run_id"]))
    except FinplanError as exc:
        if exc.code == "NOT_FOUND":
            staging.update({"platform_validation": "pending"})
            staging.pop("plan_version_id", None)
            out["staging"] = staging
            return out
        staging["platform_outcome_unavailable"] = True  # transient: keep the last known view
        out["staging"] = staging
        return out
    for key in ("plan_version_id", "platform_outcome", "rejection_kind", "platform_error_code", "platform_outcome_unavailable"):
        staging.pop(key, None)
    staging.update(staging_view(outcome))
    staging["status"] = "staged"
    out["staging"] = staging
    return out


def platform_outcome_hook(client: Any) -> Callable[[Mapping[str, Any], dict[str, Any]], dict[str, Any]]:
    """``ServiceDeps.result_hooks`` entry (``get_job_result``)."""

    def hook(run: Mapping[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        if client is None:
            return result
        return link_platform_outcome(result, client)

    return hook
