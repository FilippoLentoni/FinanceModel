"""Design D1: the dispatcher runs only while runs are queued, active or awaiting an approval
deadline (finplan_model.control.wakeup). Offline: in-memory schedule and a fake Scheduler client."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from finplan_model.control.auth import Principal
from finplan_model.control.wakeup import AT_SLACK, TICK, InMemoryDispatchSchedule, SchedulerDispatchSchedule, WakePlan, at_expression, plan_wakeup, stronger

from .support import APPROVER, Harness, events_logged

NOW = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


# ---------------------------------------------------------------- the plan
def test_plan_ticks_while_runs_are_queued_or_active():
    assert plan_wakeup(queued=1, active=0, approval_deadlines=[], now=NOW).expression == TICK
    assert plan_wakeup(queued=0, active=1, approval_deadlines=[NOW + timedelta(hours=5)], now=NOW).expression == TICK


def test_plan_wakes_once_at_the_earliest_approval_deadline():
    later, sooner = NOW + timedelta(hours=72), NOW + timedelta(hours=10)
    plan = plan_wakeup(queued=0, active=0, approval_deadlines=[later, sooner], now=NOW)
    assert plan.expression == at_expression(sooner + AT_SLACK) == "at(2026-01-05T19:01:00)"


def test_plan_ticks_when_an_approval_expires_now_and_disarms_when_idle():
    assert plan_wakeup(queued=0, active=0, approval_deadlines=[NOW - timedelta(seconds=1)], now=NOW).expression == TICK
    idle = plan_wakeup(queued=0, active=0, approval_deadlines=[], now=NOW)
    assert idle.expression is None and not idle.enabled and idle.label == "disabled"


def test_stronger_never_weakens():
    tick, early, late, off = WakePlan(TICK, "t"), WakePlan(at_expression(NOW + timedelta(hours=1)), "e"), WakePlan(at_expression(NOW + timedelta(hours=9)), "l"), WakePlan(None, "o")
    assert stronger(off, late) is late and stronger(late, early) is early and stronger(early, tick) is tick
    assert stronger(tick, off) is tick and stronger(None, None).expression is None


# ---------------------------------------------------------------- the control plane arms and disarms
def test_idle_environment_keeps_the_schedule_disarmed():
    h = Harness()
    assert h.service.dispatch()["schedule"] == ["disabled"]
    assert h.wakeup.history == [] and not h.wakeup.plan.enabled


def test_queued_submission_arms_the_tick_and_kicks_with_its_run_id():
    h = Harness(auto_approve=1.0)
    _, resp = h.submit()
    assert h.wakeup.plan.expression == TICK
    assert h.kicked == [resp["run_id"]]


def test_awaiting_approval_arms_only_its_expiry_wakeup():
    h = Harness()
    _, resp = h.submit()
    deadline = h.service._approval_deadline(h.run(resp["run_id"]))
    assert h.wakeup.plan.expression == at_expression(deadline + AT_SLACK)
    assert h.kicks == 0
    # the dispatcher keeps exactly that wake-up while nothing else is pending
    assert h.service.dispatch()["schedule"] == [at_expression(deadline + AT_SLACK)]


def test_approval_expiry_wakeup_cancels_the_run_and_disarms():
    h = Harness()
    _, resp = h.submit()
    h.clock.advance(hours=72, seconds=60)
    summary = h.service.dispatch()
    assert summary["expired"] == [resp["run_id"]] and summary["schedule"] == ["disabled"]
    assert h.run(resp["run_id"])["state"] == "cancelled"


def test_tick_runs_until_the_last_job_ends_then_disarms():
    h = Harness(auto_approve=1.0)
    run_id = h.start()
    assert h.wakeup.plan.expression == TICK
    assert h.service.dispatch()["schedule"] == [TICK]  # starting: lease heartbeats and reconcile
    h.succeed(run_id)
    assert h.run(run_id)["state"] == "succeeded"
    assert h.service.dispatch()["schedule"] == ["disabled"]
    assert h.wakeup.history[-1] == "disabled"


def test_second_queued_run_keeps_the_tick_after_the_first_ends():
    h = Harness(auto_approve=1.0)
    first = h.start()
    _, second = h.submit(idempotency_key="client-key-0002")
    h.succeed(first)
    summary = h.service.dispatch()
    assert second["run_id"] in summary["started"] and summary["schedule"] == [TICK]


def test_approval_arms_the_tick():
    h = Harness()
    _, resp = h.submit()
    h.service.approve_run(Principal.from_arn(APPROVER), resp["run_id"], {})
    assert h.wakeup.plan.expression == TICK and h.kicked == [resp["run_id"]]


def test_state_event_for_an_active_run_rearms_a_reset_schedule():
    """A deploy resets the schedule to DISABLED while a job runs: the next state event re-arms it."""
    h = Harness(auto_approve=1.0)
    run_id = h.start()
    h.wakeup.plan = WakePlan(None, "reset by a deploy")
    h.event(h.run(run_id)["job_name"], "InProgress")
    assert h.run(run_id)["state"] == "running" and h.wakeup.plan.expression == TICK


def test_kicked_run_the_index_does_not_show_yet_keeps_the_dispatcher_armed():
    h = Harness(auto_approve=1.0)
    _, resp = h.submit()
    h.wakeup.plan = WakePlan(None, "raced")
    real = h.store.query_runs
    h.store.query_runs = lambda state=None, **kw: ([], None)  # eventually consistent index lags
    try:
        plan = h.service.rearm(hint_run_id=resp["run_id"])
    finally:
        h.store.query_runs = real
    assert plan.expression == TICK and h.wakeup.plan.expression == TICK


def test_submission_racing_with_a_disarm_is_rearmed():
    h = Harness(auto_approve=1.0)
    real = h.service.pending_plan
    calls = {"n": 0}

    def racing_pending():
        calls["n"] += 1
        if calls["n"] == 1:
            return WakePlan(None, "nothing yet")  # the tick saw no run ...
        h.submit()  # ... a submission lands before the disarm completes
        return real()

    h.service.pending_plan = racing_pending
    assert h.service.rearm().expression == TICK
    assert h.wakeup.plan.expression == TICK


def test_schedule_failures_never_fail_a_submission_or_a_tick(caplog):
    caplog.set_level("INFO")
    h = Harness(auto_approve=1.0)
    h.wakeup.fail = True
    code, resp = h.submit()
    assert code == 202 and h.kicked == [resp["run_id"]]
    assert "schedule" in h.service.dispatch()
    assert events_logged(caplog, "dispatch_schedule_unavailable")


# ---------------------------------------------------------------- the Scheduler adapter
class FakeScheduler:
    def __init__(self) -> None:
        self.desc: dict[str, Any] = {
            "Name": "finplan-beta-financemodel-job-dispatcher",
            "GroupName": "default",
            "ScheduleExpression": TICK,
            "State": "DISABLED",
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "Description": "deployed",
            "Target": {"Arn": "arn:aws:lambda:us-east-2:<account-id>:function:finplan-beta-financemodel-job-dispatcher", "RoleArn": "arn:aws:iam::<account-id>:role/finplan-beta-financemodel-job-dispatcher-schedule-role", "Input": "{}", "RetryPolicy": {"MaximumRetryAttempts": 0, "MaximumEventAgeInSeconds": 60}},
            "Arn": "ignored",
            "CreationDate": NOW,
        }
        self.updates: list[dict[str, Any]] = []

    def get_schedule(self, *, Name: str, GroupName: str) -> dict[str, Any]:
        assert Name == self.desc["Name"] and GroupName == "default"
        return dict(self.desc)

    def update_schedule(self, **kw: Any) -> dict[str, Any]:
        self.updates.append(kw)
        self.desc.update(ScheduleExpression=kw["ScheduleExpression"], State=kw["State"])
        return {"ScheduleArn": "ignored"}


def test_scheduler_adapter_keeps_the_deployed_target_and_changes_only_expression_and_state():
    client = FakeScheduler()
    sched = SchedulerDispatchSchedule(client, client.desc["Name"])
    assert sched.current().expression is None
    assert sched.ensure(WakePlan(TICK, "queued")) is True
    (req,) = client.updates
    assert req["State"] == "ENABLED" and req["ScheduleExpression"] == TICK and req["ScheduleExpressionTimezone"] == "UTC"
    assert req["Target"] == client.desc["Target"] and req["Description"] == "deployed" and req["FlexibleTimeWindow"] == {"Mode": "OFF"}
    assert "Arn" not in req and "CreationDate" not in req
    # ensure never weakens; apply of the same plan is a no-op
    assert sched.ensure(WakePlan(at_expression(NOW + timedelta(hours=3)), "approval")) is False
    assert sched.apply(WakePlan(TICK, "again")) is False and len(client.updates) == 1
    # disarming restores exactly the deployed state (rate(1 minute), DISABLED): no drift when idle
    assert sched.apply(WakePlan(None, "idle")) is True
    assert client.desc["State"] == "DISABLED" and client.desc["ScheduleExpression"] == TICK


@pytest.mark.parametrize("plan", [WakePlan(None, "x"), WakePlan(TICK, "y")])
def test_in_memory_schedule_records_changes(plan):
    s = InMemoryDispatchSchedule()
    changed = s.apply(plan)
    assert changed is plan.enabled and s.history == ([plan.label] if plan.enabled else [])
