"""Deployed job-interface lifecycle (tasks 11.1, 11.2 at zero SageMaker cost; JOB-02..JOB-06, JOB-08,
CTL-05, CTL-07; platform lesson L5).

Runs in the pipeline's beta (``integration-beta``) and gamma (``gamma``) stage actions through the
job API with the stage role's SigV4 credentials and the endpoint from
``/finplan/<env>/financemodel/api/job-endpoint`` (``tests/integration/test_deployed_environment.py``).
The same code runs offline against the in-process API double
(``tests/unit/control/test_deployed_job_suite_double.py``), which proves every request is valid
before it reaches a deployed API.

:func:`run_contract_checks` needs no snapshot and records nothing:

* ``list_jobs`` answers 200 and every listed job validates against ``job-status``;
* a caller-supplied ``run_id`` is ``VALIDATION_FAILED`` (JOB-03) and an unknown run is
  ``NOT_FOUND``; both are contract error envelopes.

:func:`run_job_lifecycle` submits ONE tiny synthetic-labeled CPU ``run_backtest`` on the approved
integration snapshot and cancels it before it can cost anything material. The snapshot may be real
(phase 2: a yfinance SPY snapshot, no ``synthetic`` flag) or synthetic (prod during the phase 2
transition); the suite behaves the same in beta and gamma (data parity, user decision 26):

* dry run (JOB-08): 200, no ``run_id``, a contract cost-estimate block, nothing recorded;
* submission (JOB-02): 202 with a FinanceModel-minted ``run_id``; with the default auto-approve
  threshold of 0 the run waits in ``awaiting_approval`` (CTL-08), so no SageMaker job starts;
* idempotency (JOB-04): the same key and body replays the same ``run_id``; the same key with another
  body is ``IDEMPOTENCY_KEY_REUSED``;
* status (JOB-05) is polled and validates against ``job-status``;
* cancellation (CTL-05): ``cancel_job`` (replayed once with its key) ends in ``cancelled``; a run an
  operator auto-approved may already be ``starting``/``running`` and is stopped, which SageMaker
  bills for the seconds it ran (an upper bound is the job's cost estimate, cents);
* result (JOB-06): the cancelled run's result validates against ``job-result`` with
  ``completion_status`` ``cancelled`` and carries no storage location.

Every idempotency key carries the run key, so pipeline executions never collide.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Call", "IntegrationError", "LifecycleResult", "fixture_request", "run_contract_checks", "run_job_lifecycle"]

#: ``call(method, path, body)`` -> (HTTP status, JSON body)
Call = Callable[..., tuple[int, Any]]
TERMINAL = {"succeeded", "failed", "timed_out", "cancelled"}
NON_TERMINAL = {"awaiting_approval", "queued", "starting", "running", "stopping"}
UNKNOWN_RUN = "run_01KDVDNAZ83BAMMYCEGWF33DPM"


class IntegrationError(AssertionError):
    pass


@dataclass
class LifecycleResult:
    run_id: str = ""
    states: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise IntegrationError(message)


def _valid(doc: Any, schema: str) -> None:
    from finplan_contracts.validate import validate

    res = validate(doc, schema)
    _check(res.valid, f"response does not validate against {schema}: {[str(i) for i in res.issues][:3]} ({doc})")


def _expect(resp: tuple[int, Any], statuses: tuple[int, ...], what: str) -> Any:
    status, body = resp
    _check(status in statuses, f"{what}: HTTP {status}, expected {statuses}: {body}")
    return body


def _error(resp: tuple[int, Any], status: int, code: str, what: str) -> dict[str, Any]:
    body = _expect(resp, (status,), what)
    _valid(body, "error")
    _check(body.get("code") == code, f"{what}: error code {body.get('code')}, expected {code}")
    return body


def fixture_request(snapshot_id: str, idempotency_key: str, *, dry_run: bool = False, strategy: str = "equal_weight") -> dict[str, Any]:
    """A tiny CPU-only ``run_backtest`` submission labeled synthetic (a test record; the snapshot may be real)."""
    from finplan_contracts import __version__ as contract_version

    return {
        "domain": "finance",
        "domain_schema_version": "1.0",
        "job_type": "run_backtest",
        "purpose": "research",
        "dry_run": dry_run,
        "input_snapshot_id": snapshot_id,
        "configuration": {
            "domain": "finance",
            "domain_schema_version": "1.0",
            "payload": {
                "strategy": strategy,
                "objective": "backtest",
                "universe": ["AGG", "SPY"],
                "rebalance_frequency": "monthly",
                "constraints": {"long_only": True, "max_weight": 1.0},
                "fees": {"transaction_cost_bps": 1},
            },
            "synthetic": True,
        },
        "evaluation_window": {"start": "2026-01-05", "end": "2026-03-27"},
        "idempotency_key": idempotency_key,
        "contract_version": str(contract_version),
        "synthetic": True,
    }


#: model_selection (decision 27) must stay under the beta/gamma CPU auto-approve threshold (decision 24).
SELECTION_ESTIMATE_CEILING_USD = 0.25


def model_selection_request(snapshot_id: str, idempotency_key: str, *, dry_run: bool = True, strategy: str = "model_selection") -> dict[str, Any]:
    """A ``model_selection`` submission over the research universe (dry run by default: no run, no job)."""
    body = fixture_request(snapshot_id, idempotency_key, dry_run=dry_run, strategy=strategy)
    body["job_type"] = "model_selection"
    body.pop("evaluation_window", None)
    body["configuration"]["payload"]["universe"] = ["VOO", "GOOGL", "NFLX", "AAPL", "NVDA", "USD_CASH"]
    return body


def run_model_selection_dry_run(call: Call, *, snapshot_id: str, run_key: str) -> list[str]:
    """The deployed ``model_selection`` kind: validated, estimated under USD 0.25 in ``cpu_research``,
    a wrong strategy refused; nothing recorded and no SageMaker job (dry runs only)."""
    steps: list[str] = []
    dry = _expect(call("POST", "/v1/jobs", model_selection_request(snapshot_id, f"it-{run_key}-ms-dry")), (200,), "model_selection dry run")
    _valid(dry, "tools/submit-experiment-response")
    _check(dry.get("dry_run") is True and dry.get("run_id") is None, "a model_selection dry run records nothing")
    est = dry["cost_estimate"]
    _valid(est, "cost-estimate")
    _check(est.get("budget_category") == "cpu_research", "model_selection draws on cpu_research")
    _check(float(est["estimated_usd_upper_bound"]) <= SELECTION_ESTIMATE_CEILING_USD, f"model_selection estimate USD {est['estimated_usd_upper_bound']} is above USD {SELECTION_ESTIMATE_CEILING_USD}")
    steps.append(f"model_selection dry run: estimate USD {est['estimated_usd_upper_bound']} ({est['budget_category']})")
    _error(call("POST", "/v1/jobs", model_selection_request(snapshot_id, f"it-{run_key}-ms-bad", strategy="ppo")), 400, "VALIDATION_FAILED", "a model_selection request naming another strategy")
    steps.append("model_selection with another strategy: VALIDATION_FAILED")
    return steps


def run_contract_checks(call: Call) -> list[str]:
    steps: list[str] = []
    listing = _expect(call("GET", "/v1/jobs"), (200,), "list_jobs")
    _check(isinstance(listing.get("jobs"), list), "list_jobs returns a jobs list")
    for job in listing["jobs"]:
        _valid(job, "job-status")
    steps.append(f"list_jobs: {len(listing['jobs'])} job(s)")
    bad = fixture_request("snap_01KDVDNAZ83BAMMYCEGWF33DPM", "it-contract-run-id-0001", dry_run=True)
    bad["run_id"] = UNKNOWN_RUN
    _error(call("POST", "/v1/jobs", bad), 400, "VALIDATION_FAILED", "a caller-supplied run_id")
    steps.append("caller run_id rejected (VALIDATION_FAILED)")
    _error(call("GET", f"/v1/jobs/{UNKNOWN_RUN}"), 404, "NOT_FOUND", "an unknown run")
    steps.append("unknown run: NOT_FOUND")
    return steps


def _status(call: Call, run_id: str) -> dict[str, Any]:
    doc = _expect(call("GET", f"/v1/jobs/{run_id}"), (200,), "get_job_status")
    _valid(doc, "job-status")
    _check(doc.get("run_id") == run_id, "status of another run")
    return doc


def run_job_lifecycle(call: Call, *, snapshot_id: str, run_key: str, sleep: Callable[[float], None] = time.sleep, poll_interval: float = 10.0, max_wait_seconds: float = 600.0) -> LifecycleResult:
    r = LifecycleResult()
    key = f"it-{run_key}-submit"
    # dry run: valid, estimated, nothing recorded
    dry = _expect(call("POST", "/v1/jobs", fixture_request(snapshot_id, f"it-{run_key}-dry", dry_run=True)), (200,), "dry-run submission")
    _valid(dry, "tools/submit-experiment-response")
    _check(dry.get("dry_run") is True and dry.get("run_id") is None, "a dry run records nothing")
    _valid(dry["cost_estimate"], "cost-estimate")
    r.steps.append(f"dry run: estimate USD {dry['cost_estimate'].get('estimated_usd_upper_bound')}")

    body = fixture_request(snapshot_id, key)
    sub = _expect(call("POST", "/v1/jobs", body), (202,), "submission")
    _valid(sub, "tools/submit-experiment-response")
    run_id = str(sub.get("run_id") or "")
    _check(run_id.startswith("run_"), "FinanceModel mints the run_id")
    _check(sub.get("synthetic") is True, "a test submission labeled synthetic stays synthetic")
    r.run_id = run_id
    r.states.append(str(sub["state"]))
    r.steps.append(f"submitted {run_id} ({sub['state']})")
    try:
        # idempotency: same key + body replays; same key + other body is refused
        replay = _expect(call("POST", "/v1/jobs", body), (200, 202), "replayed submission")
        _check(replay.get("run_id") == run_id, "a replay returns the original run_id")
        changed = fixture_request(snapshot_id, key, strategy="cash")
        _error(call("POST", "/v1/jobs", changed), 422, "IDEMPOTENCY_KEY_REUSED", "the same key with another body")
        r.steps.append("idempotency: replay and IDEMPOTENCY_KEY_REUSED")
        st = _status(call, run_id)
        _check(st["state"] in NON_TERMINAL | TERMINAL, f"unknown state {st['state']}")
        r.states.append(st["state"])
    finally:
        cancel_body = {"idempotency_key": f"it-{run_key}-cancel", "reason": "integration test cleanup"}
        cancelled = _expect(call("POST", f"/v1/jobs/{run_id}/cancel", cancel_body), (200,), "cancel_job")
        again = _expect(call("POST", f"/v1/jobs/{run_id}/cancel", cancel_body), (200,), "replayed cancel_job")
        _check(again.get("run_id") == cancelled.get("run_id") == run_id, "a cancel replay answers for the same run")
    waited = 0.0
    st = _status(call, run_id)
    while st["state"] not in TERMINAL:
        _check(waited < max_wait_seconds, f"run {run_id} did not reach a terminal state in {max_wait_seconds:.0f}s (last {st['state']})")
        sleep(poll_interval)
        waited += poll_interval
        st = _status(call, run_id)
    r.states.append(st["state"])
    _check(st["state"] == "cancelled" or st["state"] in TERMINAL, "terminal after cancel")
    r.steps.append(f"cancelled ({st['state']} after {waited:.0f}s)")
    result = _expect(call("GET", f"/v1/jobs/{run_id}/result"), (200,), "get_job_result")
    _valid(result, "job-result")
    _check(result.get("run_id") == run_id, "result of another run")
    if st["state"] == "cancelled":
        _check(result.get("completion_status") == "cancelled", "a cancelled run reports completion_status cancelled")
    from finplan_model.control.validation import find_storage_location

    _check(find_storage_location(result) is None, "a result never exposes a storage location")
    r.steps.append(f"result: {result.get('completion_status')}")
    return r


def adapt(call_with_kwargs: Callable[..., Any]) -> Call:
    """Wrap a ``call(method, path, body=...)`` that returns (status, body[, headers])."""

    def call(method: str, path: str, body: Mapping[str, Any] | None = None) -> tuple[int, Any]:
        out = call_with_kwargs(method, path, body=body)
        return out[0], out[1]

    return call
