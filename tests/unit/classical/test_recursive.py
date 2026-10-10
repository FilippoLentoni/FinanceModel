import copy
from datetime import timedelta

import pytest

from finplan_model.classical.recursive import run_recursive_improvement, scheduled_recursive_review
from finplan_model.classical.research import horizon_evidence_summary
from finplan_model.core.errors import FinplanError


def test_recursive_dry_preview_freezes_portfolio_bounds_and_real_ppo_profile(context):
    s = context.service
    result = run_recursive_improvement(s, {"query": "PPO missing state features", "max_iterations": 2})
    assert result["state"] == "awaiting_experiment_approval" and result["iteration"] == 0
    assert result["proposed_experiment"]["job_type"] == "recursive_evaluate"
    assert result["proposed_experiment"]["candidate_profile"] == "recursive_ppo_features"
    assert all(r["dry_run"] for r in s.d.job_api.calls)
    assert not any(k.startswith("recursive/") for k in s.store.claims)
    with pytest.raises(FinplanError):
        run_recursive_improvement(s, {"cycle_id": result["cycle_id"], "max_iterations": 3})


def test_recursive_resume_feedback_changes_released_profile_and_is_immutable(context):
    s = context.service
    first = run_recursive_improvement(s, {"query": "PPO features"})
    stored = copy.deepcopy(s.store.get(first["analysis_id"]))
    now = s.now()
    s.now = lambda: now + timedelta(seconds=1)
    second = run_recursive_improvement(s, {"cycle_id": first["cycle_id"], "feedback": "trade less: transaction costs and turnover"})
    assert second["proposed_experiment"]["candidate_profile"] == "recursive_ppo_turnover"
    assert second["cycle_id"] == first["cycle_id"]
    assert second["lineage"][-1]["analysis_id"] == first["analysis_id"]
    assert any(e.get("source_type") == "resume_user_feedback" and "turnover" in e["text"] for e in second["evidence"])
    assert s.store.get(first["analysis_id"]) == stored


def test_recursive_paid_intent_is_required_before_compute(context):
    with pytest.raises(FinplanError) as caught:
        run_recursive_improvement(context.service, {"query": "PPO", "dry_run": False})
    assert caught.value.details["reason"] == "confirmation_required"
    assert context.service.d.job_api.calls == []


@pytest.mark.parametrize("query,kind,strategy,compute", [("Qwen agent swarm", "swarm_mode_a", "qwen_swarm", "gpu"), ("TypeSafe Jev", "jev_backtest", "jev", "cpu")])
def test_identified_benchmark_preview_targets_actual_matching_job(context, query, kind, strategy, compute):
    result = run_recursive_improvement(context.service, {"query": query})
    request = result["proposed_experiment"]["tool_request"]
    assert request["name"] == "submit_experiment"
    args = request["arguments"]
    assert args["job_type"] == kind and args["dry_run"]
    assert "compute_class" not in args and "max_runtime_seconds" not in args
    assert args["configuration"]["payload"]["strategy"] == strategy
    assert args["configuration"]["payload"]["rebalance_frequency"] == "monthly"
    assert all(r["dry_run"] for r in context.service.d.job_api.calls)


def test_scheduler_resumes_at_most_one_cycle_without_activating(context):
    first = run_recursive_improvement(context.service, {"query": "PPO features", "dry_run": False, "confirmed_by_user": True, "idempotency_key": "opt-in-weekly"})
    out = scheduled_recursive_review(context.service)
    assert out["cycle_id"] == first["cycle_id"]
    assert sum(not r["dry_run"] for r in context.service.d.job_api.calls) == 1
    assert out["activation"]["status"] == "proposal_only"


def test_external_preview_cannot_starve_or_launch_from_weekly_controller(context):
    run_recursive_improvement(context.service, {"query": "Qwen swarm"})
    assert scheduled_recursive_review(context.service) is None
    assert all(r["dry_run"] for r in context.service.d.job_api.calls)


def test_preview_only_cpu_cycle_never_authorizes_scheduled_compute(context):
    run_recursive_improvement(context.service, {"query": "PPO features"})
    assert scheduled_recursive_review(context.service) is None
    assert all(r["dry_run"] for r in context.service.d.job_api.calls)


def test_paid_resume_replays_existing_job_and_crash_claim(context, monkeypatch):
    s = context.service
    first = run_recursive_improvement(s, {"query": "PPO", "dry_run": False, "confirmed_by_user": True, "idempotency_key": "same-request"})
    before = len(s.d.job_api.calls)
    assert run_recursive_improvement(s, {"cycle_id": first["cycle_id"], "dry_run": False, "confirmed_by_user": True, "idempotency_key": "same-request"}) == first
    assert len(s.d.job_api.calls) == before
    # Simulate a crash after submission and before persisting the iteration. The
    # frozen review and job request must survive even after feedback changes.
    s.store.docs.pop(first["analysis_id"])
    recovered = run_recursive_improvement(s, {"cycle_id": first["cycle_id"], "feedback": "turnover", "dry_run": False, "confirmed_by_user": True, "idempotency_key": "same-request"})
    assert recovered["job"]["run_id"] == first["job"]["run_id"]
    assert recovered["experiment_review_id"] == first["experiment_review_id"]
    paid = [r for r in s.d.job_api.calls if not r["dry_run"]]
    assert paid[-1] == paid[-2]


@pytest.mark.parametrize("gate,reused,reason", [("pass", False, "candidate_requires_human_activation_and_forward_validation"), ("fail", False, "no_improvement_under_declared_return_and_drawdown_criteria"), ("pass", True, "fresh_forward_validation_required")])
def test_measured_result_stops_for_gate_or_reused_holdout(context, gate, reused, reason):
    s = context.service
    first = run_recursive_improvement(s, {"query": "PPO", "dry_run": False, "confirmed_by_user": True, "idempotency_key": "first-run"})
    s.d.job_api.result = lambda rid: {"completion_status": "succeeded", "payload": {"model_selection": {"promotion_check": {"result": gate}, "test_access": {"test_reuse": reused}}, "performance": {"cumulative_return": .01}}}
    second = run_recursive_improvement(s, {"cycle_id": first["cycle_id"]})
    assert second["stopping_reason"] == reason
    assert second["proposed_experiment"] is None and second["activation"]["status"] == "proposal_only"
    assert any(row.get("source_type") == "experiment_result" for row in second["evidence"])


def test_horizon_review_excludes_retrospective_and_overlapping_windows():
    def row(identifier, start, end, eligible=True, config="cfg_same"):
        return {"analysis_id": identifier, "portfolio_id": "paper", "horizon_evaluation": {"primary_horizon_mature": True, "protocol": {"primary_horizon_sessions": 21}, "protocol_checksum": "same-protocol", "source": {"configuration_id": config}, "replay_window": {"start": start}, "windows": [{"horizon_sessions": 21, "status": "mature", "end_date": end}], "evidence_assessment": {"eligible_for_prospective_skill_evidence": eligible, "primary_return_underperformed_unchanged": True}}}
    result = horizon_evidence_summary([row("a", "2026-01-01", "2026-02-01"), row("b", "2026-01-02", "2026-02-02"), row("c", "2026-02-03", "2026-03-03"), row("d", "2026-03-04", "2026-04-04"), row("historical", "2025-01-01", "2025-02-01", False)])
    assert result["status"] == "review_triggered" and result["model_error_proven"] is False
    assert result["groups"][0]["independent_mature_windows"] == 3
    assert result["groups"][0]["overlapping_windows_excluded"] == 1
    assert result["immature_retrospective_or_unidentified_excluded"] == 1
