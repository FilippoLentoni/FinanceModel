"""Run lifecycle states and allowed transitions (spec experiment-job-interface, "Job lifecycle states";
design D2).

::

    submit ─► awaiting_approval ─(approve)─► queued ─(lease)─► starting ─► running ─► terminal
                │ (expire/cancel → cancelled)   │ (cancel → cancelled)        │ (cancel) ─► stopping ─► cancelled

Terminal states equal the contract ``completion_status`` values (``succeeded``, ``failed``,
``cancelled``, ``timed_out``) and never change. ``starting`` may fall back to ``queued`` when the
SageMaker start is refused for an account quota or throttling (CTL-04, CTL-06).
"""

from __future__ import annotations

from finplan_contracts.schemas import load_store

from finplan_model.core.errors import FinplanError

__all__ = [
    "ACTIVE_STATES",
    "NON_TERMINAL_STATES",
    "STATES",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "WAITING_STATES",
    "check_transition",
    "is_terminal",
]


def _contract_states() -> tuple[tuple[str, ...], tuple[str, ...]]:
    schema = load_store().get("job-status").schema
    return tuple(schema["$defs"]["state"]["enum"]), tuple(schema["x-finplan-terminal-states"])


#: Every state, from the pinned contract (``core/v1/job-status.json``), never a local copy.
STATES, TERMINAL_STATES = _contract_states()
NON_TERMINAL_STATES: tuple[str, ...] = tuple(s for s in STATES if s not in TERMINAL_STATES)
#: Runs waiting for a human or for a lease (they count against the bounded queue, CTL-03).
WAITING_STATES = ("awaiting_approval", "queued")
#: Runs that hold a lease and may have a SageMaker job.
ACTIVE_STATES = ("starting", "running", "stopping")

_T = set(TERMINAL_STATES)
TRANSITIONS: dict[str, frozenset[str]] = {
    "awaiting_approval": frozenset({"queued", "cancelled"}),
    "queued": frozenset({"starting", "cancelled", "failed"}),
    # A short job can finish before its InProgress event is processed, so starting may go terminal.
    "starting": frozenset({"running", "queued", "stopping"} | _T),
    "running": frozenset({"stopping"} | _T),
    "stopping": frozenset(_T),
}
for _s in TERMINAL_STATES:
    TRANSITIONS[_s] = frozenset()

if set(TRANSITIONS) != set(STATES):  # pragma: no cover - guards a contract change
    raise RuntimeError("lifecycle transition table does not cover the contract states")


def is_terminal(state: str) -> bool:
    return state in _T


def check_transition(current: str, new: str) -> None:
    """Raise ``PRECONDITION_FAILED`` (``invalid_transition``) unless ``current -> new`` is allowed."""
    if new not in TRANSITIONS.get(current, frozenset()):
        raise FinplanError.precondition("the run cannot move to the requested state", reason="invalid_transition", state=current, requested_state=new)
