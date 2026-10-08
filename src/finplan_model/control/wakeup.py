"""Dispatcher wake-up: the dispatcher runs only while runs need it (design D1).

The dispatcher schedule (EventBridge Scheduler ``finplan-<env>-financemodel-job-dispatcher``) is
deployed **DISABLED** with ``rate(1 minute)``. The control plane arms and disarms it at run time:

====================================================  =====================================================
pending work (this environment's run records)         schedule
====================================================  =====================================================
a run ``queued``, ``starting``, ``running`` or         ``rate(1 minute)``, ENABLED (start queued runs, lease
``stopping``                                           heartbeats, reconcile, stale-lease reclaim)
only runs ``awaiting_approval``                        ``at(<earliest approval deadline + 60 s>)``, ENABLED
                                                       (one wake-up to expire the approval)
nothing                                                ``rate(1 minute)``, DISABLED (the deployed state, so
                                                       an idle environment never drifts from its template)
====================================================  =====================================================

Who arms it:

* ``submit_job`` and ``approve_run`` (API handler): a new ``queued`` run arms the tick, a new
  ``awaiting_approval`` run arms at least its expiry wake-up (:meth:`DispatchSchedule.ensure` only
  ever strengthens the current plan). They also kick the dispatcher asynchronously with the run ID.
* the state-change handler: an event for a run that is still active re-arms the tick (lease
  heartbeats), so a deploy that resets the schedule cannot strand a running job.
* the dispatcher, at the end of every tick, sets the plan computed from the records *after* the tick
  (:func:`plan_wakeup`). Before it disarms it re-reads the pending runs once more and re-arms when a
  submission raced with the tick; a kicked run ID is read with a strongly consistent read, so a run
  the index does not show yet still counts.

Idle cost: zero invocations. ``UpdateSchedule`` replaces the whole schedule, so
:class:`SchedulerDispatchSchedule` copies the deployed target from ``GetSchedule`` and changes
only the expression and the state (needs ``scheduler:GetSchedule``, ``scheduler:UpdateSchedule`` and
``iam:PassRole`` of the schedule role to ``scheduler.amazonaws.com``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AT_SLACK",
    "TICK",
    "DispatchSchedule",
    "InMemoryDispatchSchedule",
    "SchedulerDispatchSchedule",
    "WakePlan",
    "at_expression",
    "plan_wakeup",
    "stronger",
]

#: The tick while runs are queued or active.
TICK = "rate(1 minute)"
#: An approval-expiry wake-up fires this long after the deadline (the expiry check is ``>``).
AT_SLACK = timedelta(seconds=60)
#: A one-time wake-up closer than this is replaced by the tick (Scheduler needs a future time).
_AT_MIN_LEAD = timedelta(minutes=2)
_AT_FORMAT = "%Y-%m-%dT%H:%M:%S"


def at_expression(when: datetime) -> str:
    """EventBridge Scheduler one-time expression in UTC (``at(yyyy-mm-ddThh:mm:ss)``)."""
    return f"at({when.astimezone(UTC).strftime(_AT_FORMAT)})"


def _at_time(expression: str) -> datetime | None:
    if not (expression.startswith("at(") and expression.endswith(")")):
        return None
    return datetime.strptime(expression[3:-1], _AT_FORMAT).replace(tzinfo=UTC)


@dataclass(frozen=True)
class WakePlan:
    """``expression`` ``None`` means DISABLED."""

    expression: str | None
    reason: str

    @property
    def enabled(self) -> bool:
        return self.expression is not None

    @property
    def label(self) -> str:
        return self.expression or "disabled"


OFF = WakePlan(None, "no pending runs")


def plan_wakeup(*, queued: int, active: int, approval_deadlines: Iterable[datetime], now: datetime) -> WakePlan:
    """The schedule the pending runs need (module table)."""
    if queued or active:
        return WakePlan(TICK, f"{queued} queued and {active} active run(s)")
    deadlines = sorted(approval_deadlines)
    if deadlines:
        wake = deadlines[0] + AT_SLACK
        if wake <= now + _AT_MIN_LEAD:
            return WakePlan(TICK, "an approval expires now")
        return WakePlan(at_expression(wake), f"{len(deadlines)} run(s) awaiting approval")
    return OFF


def _rank(plan: WakePlan | None) -> tuple[int, float]:
    if plan is None or plan.expression is None:
        return (0, 0.0)
    if plan.expression == TICK:
        return (2, 0.0)
    at = _at_time(plan.expression)
    return (1, -(at.timestamp() if at else 0.0))


def stronger(a: WakePlan | None, b: WakePlan | None) -> WakePlan:
    """The plan that wakes the dispatcher at least as often as both (tick > earlier at > later at > off)."""
    best = max((a, b), key=_rank)
    return best if best is not None else OFF


@runtime_checkable
class DispatchSchedule(Protocol):
    def current(self) -> WakePlan: ...

    def apply(self, plan: WakePlan) -> bool:
        """Set exactly ``plan``; ``True`` when the schedule changed."""
        ...

    def ensure(self, plan: WakePlan) -> bool:
        """Strengthen the current plan to at least ``plan`` (never weakens); ``True`` when changed."""
        ...


class _EnsureMixin:
    def ensure(self, plan: WakePlan) -> bool:
        cur = self.current()  # type: ignore[attr-defined]
        best = stronger(cur, plan)
        if best.expression == cur.expression:
            return False
        return bool(self.apply(best))  # type: ignore[attr-defined]


class SchedulerDispatchSchedule(_EnsureMixin):
    """The deployed schedule, changed through the EventBridge Scheduler API."""

    def __init__(self, client: Any, name: str, *, group: str = "default") -> None:
        self.client = client
        self.name = name
        self.group = group

    def _describe(self) -> dict[str, Any]:
        return dict(self.client.get_schedule(Name=self.name, GroupName=self.group))

    @staticmethod
    def _plan_of(desc: dict[str, Any]) -> WakePlan:
        if desc.get("State") != "ENABLED":
            return OFF
        return WakePlan(str(desc.get("ScheduleExpression")), "deployed")

    def current(self) -> WakePlan:
        return self._plan_of(self._describe())

    def apply(self, plan: WakePlan) -> bool:
        desc = self._describe()
        state = "ENABLED" if plan.enabled else "DISABLED"
        expression = plan.expression or TICK
        if desc.get("State") == state and desc.get("ScheduleExpression") == expression:
            return False
        target = {k: v for k, v in dict(desc["Target"]).items() if k in ("Arn", "RoleArn", "Input", "RetryPolicy", "DeadLetterConfig")}
        request: dict[str, Any] = {
            "Name": self.name,
            "GroupName": self.group,
            "ScheduleExpression": expression,
            "ScheduleExpressionTimezone": "UTC",
            "FlexibleTimeWindow": dict(desc.get("FlexibleTimeWindow") or {"Mode": "OFF"}),
            "Target": target,
            "State": state,
            "ActionAfterCompletion": "NONE",
        }
        if desc.get("Description"):
            request["Description"] = desc["Description"]
        self.client.update_schedule(**request)
        return True


@dataclass
class InMemoryDispatchSchedule(_EnsureMixin):
    """Offline stand-in (tests, local runs): records every change."""

    plan: WakePlan = OFF
    history: list[str] = field(default_factory=list)
    fail: bool = False

    def current(self) -> WakePlan:
        if self.fail:
            raise RuntimeError("scheduler unavailable")
        return self.plan

    def apply(self, plan: WakePlan) -> bool:
        if self.fail:
            raise RuntimeError("scheduler unavailable")
        if plan.expression == self.plan.expression:
            return False
        self.plan = plan
        self.history.append(plan.label)
        return True
