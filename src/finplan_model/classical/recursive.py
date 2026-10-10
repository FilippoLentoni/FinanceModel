"""Durable, bounded evidence -> hypothesis -> experiment -> review iterations.

Every record is immutable. A conditional iteration claim serializes competing resumes;
the experiment API's own idempotency key makes crash recovery safe. This controller never
executes generated code, approves GPU/vendor work, activates a strategy or changes holdings.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError

from .research import review, run_review

PROFILES = ("recursive_ppo_features", "recursive_ppo_turnover", "recursive_ppo_horizon")
TERMINAL = ("proposal_ready", "stopped")


def _capabilities(service):
    fn = getattr(service.d, "benchmark_capabilities", None)
    if callable(fn):
        return fn()
    return {
        "swarm_mode_a": {"configured": False, "reason": "runtime_configuration_not_available", "gpu_approval_required": True},
        "jev_backtest": {"configured": False, "reason": "runtime_configuration_not_available", "external_vendor_approval_required": True},
    }


def _latest(service, cycle):
    rows = service.store.list(kind="recursive_iteration", portfolio_id=cycle.get("portfolio_id"), limit=100)
    own = [r for r in rows if r.get("cycle_id") == cycle["analysis_id"]]
    return max(own, key=lambda r: (len(r.get("lineage", [])), r["iteration"], r["created_at"], r["analysis_id"])) if own else None


def _summary(result):
    payload = result.get("payload") or {}
    selection = payload.get("model_selection") or {}
    classical = payload.get("weekly_research") or {}
    gate = selection.get("promotion_check") or {}
    metrics = payload.get("performance") or {}
    reused = bool((selection.get("test_access") or {}).get("test_reuse"))
    return {
        "completion_status": result.get("completion_status"),
        "performance": metrics,
        "selection": selection.get("selection") or classical.get("selection"),
        "promotion_check": gate,
        "test_reused": reused,
        "fresh_forward_validation_required": True,
        "objective": "net returns and drawdown on the declared horizon; shaped training reward reported separately",
        "artifacts": result.get("artifacts", [])[:10],
        "activation": "proposal_only",
    }


def run_recursive_improvement(service, body, *, scheduled=False):
    if service.env != "beta":
        raise FinplanError.precondition("recursive improvement is beta only", reason="research_disabled")
    dry = body.get("dry_run", True)
    if not dry and not scheduled and (body.get("confirmed_by_user") is not True or not body.get("idempotency_key")):
        raise FinplanError.precondition("launch requires recorded user intent and an idempotency key", reason="confirmation_required")
    if body.get("cycle_id"):
        cycle = service.store.get(body["cycle_id"])
        if cycle["analysis_kind"] != "recursive_cycle":
            raise FinplanError.validation("cycle_id must identify a recursive cycle", pointer="/cycle_id")
        for key in ("portfolio_id", "max_iterations"):
            if key in body and body[key] != cycle.get(key):
                raise FinplanError.validation("a cycle's portfolio and bounds are immutable", pointer="/" + key)
    else:
        bound = body.get("max_iterations", 3)
        if isinstance(bound, bool) or not isinstance(bound, int) or not 1 <= bound <= 3:
            raise FinplanError.validation("max_iterations must be one to three", pointer="/max_iterations")
        plan = service.recommend({"portfolio_id": body["portfolio_id"]} if body.get("portfolio_id") else {})
        initial = review(service, body, plan_context_id=plan["analysis_id"])
        text = (str(body.get("query", "")) + " " + str(body.get("feedback", ""))).lower()
        family = "qwen" if any(w in text for w in ("qwen", "swarm")) else "jev" if any(w in text for w in ("jev", "typesafe")) else "ppo" if any(w in text for w in ("ppo", "reinforcement", "feature", "horizon", "neural")) else "classical"
        cycle = service.issue("recursive_cycle", {
            "summary": "Bounded recursive research cycle; evidence and literature are hypotheses, activation requires review",
            "max_iterations": bound, "family": family, "initial_review_id": initial["analysis_id"],
            "plan_context_id": plan["analysis_id"], "portfolio_id": plan.get("portfolio_id"),
            "query": str(body.get("query", ""))[:2000], "feedback": str(body.get("feedback", ""))[:4000],
            "bounds": {"weekly_usd": .50, "monthly_usd": 2., "project_usd": 50., "max_parallel_jobs": 1, "max_iterations": bound},
            "activation": {"status": "proposal_only"},
        }, {"week": service.now().strftime("%G-W%V"), "review_id": initial["analysis_id"], "family": family, "max_iterations": bound, "idempotency_key": body.get("idempotency_key")}, portfolio_id=plan.get("portfolio_id"))
        # Public issue responses carry artifact metadata; always operate on the stored record.
        cycle = service.store.get(cycle["analysis_id"])
    previous = _latest(service, cycle)
    iteration = previous["iteration"] if previous else 0
    # Refresh persisted feedback, matured outcomes and literature on every resume.
    # The cycle's book/universe and bounds remain frozen, but its evidence does not.
    initial = service.store.get(cycle["initial_review_id"])
    if previous:
        initial = review(service, {"query": body.get("query", cycle.get("query", "")), "feedback": body.get("feedback", cycle.get("feedback", ""))}, plan_context_id=cycle["plan_context_id"])
    evidence = list(initial.get("evidence", []))
    evidence.append({"source_type": "research_review", "analysis_id": initial["analysis_id"], "sources": initial.get("sources", []), "horizon_evidence": initial.get("horizon_evidence"), "prior_experiment_findings": initial.get("prior_experiment_findings", [])})
    if body.get("feedback") or body.get("query"):
        evidence.append({"source_type": "resume_user_feedback", "text": str(body.get("feedback", ""))[:4000], "query": str(body.get("query", ""))[:2000], "trust": "untrusted_hypothesis_not_executable_instructions"})
    result_summary = None
    stop = None
    if previous and previous["state"] in TERMINAL:
        from .storage import public
        return public(previous)
    if previous and not dry and previous.get("launch_idempotency_key") == body.get("idempotency_key") and previous.get("job"):
        from .storage import public
        return public(previous)
    if previous and previous.get("job"):
        run_id = previous["job"]["run_id"]
        try:
            result = service.d.job_api.result(run_id)
        except FinplanError as exc:
            if exc.code in ("PRECONDITION_FAILED", "CONFLICT", "NOT_FOUND"):
                from .storage import public
                return public(previous)
            raise
        result_summary = _summary(result)
        evidence.append({"source_type": "experiment_result", "run_id": run_id, "result": result_summary})
        if result.get("completion_status") != "succeeded":
            stop = "experiment_failed_requires_review"
        elif result_summary["test_reused"]:
            stop = "fresh_forward_validation_required"
        elif result_summary["promotion_check"].get("result") == "pass":
            stop = "candidate_requires_human_activation_and_forward_validation"
        elif result_summary["promotion_check"].get("result") == "fail":
            stop = "no_improvement_under_declared_return_and_drawdown_criteria"
        elif cycle["family"] in ("qwen", "jev"):
            stop = "external_benchmark_complete_requires_review_and_forward_validation"
    if iteration >= cycle["max_iterations"]:
        stop = stop or "maximum_iterations_reached"
    next_index = min(iteration, 2)
    profile = PROFILES[next_index] if cycle["family"] == "ppo" else "classical_weekly_review"
    feedback_text = (cycle.get("feedback", "") + " " + str(body.get("feedback", "")) + " " + str(body.get("query", ""))).lower()
    if cycle["family"] == "ppo":
        if any(word in feedback_text for word in ("turnover", "fees", "transaction cost", "trade less")):
            profile = "recursive_ppo_turnover"
        elif any(word in feedback_text for word in ("horizon", "long term", "long-term", "episode")):
            profile = "recursive_ppo_horizon"
        if result_summary:
            metrics = result_summary.get("performance", {})
            if float(metrics.get("turnover", 0.) or 0.) > 20:
                profile = "recursive_ppo_turnover"
    proposal = {**initial["proposed_experiment"], "job_type": "recursive_evaluate" if cycle["family"] == "ppo" else "run_benchmark", "objective": profile,
        "candidate_profile": profile, "research_only": True, "automatic_activation": False,
        "required_review": "new source/features/algorithms outside released profiles become a reviewed code/spec change; no generated code is executed",
        "objective_horizon": {"evaluation_windows_sessions": [21, 64, 126], "training_episode_sessions": 126 if profile == "recursive_ppo_horizon" else 64, "gamma": .995 if profile == "recursive_ppo_horizon" else .99},
    }
    if cycle["family"] in ("qwen", "jev"):
        plan = service.plan(cycle["plan_context_id"])
        inputs = plan["solve_inputs"]
        end = date.fromisoformat(inputs["as_of"])
        request = {"domain": "finance", "domain_schema_version": "1.0", "job_type": "swarm_mode_a" if cycle["family"] == "qwen" else "jev_backtest", "purpose": "research", "dry_run": True,
            "input_snapshot_id": inputs["input_snapshot_id"], "contract_version": "1.6.0", "synthetic": True,
            "configuration": {"domain": "finance", "domain_schema_version": "1.0", "synthetic": True, "payload": {
                "strategy": "qwen_swarm" if cycle["family"] == "qwen" else "jev", "objective": "llm_benchmark", "universe": inputs["instruments"], "lookback_days": 60,
                "rebalance_frequency": "monthly", "constraints": {"long_only": True, "max_weight": .6}, "fees": {"transaction_cost_bps": 2}}},
            "evaluation_window": {"start": (end - timedelta(days=365)).isoformat(), "end": end.isoformat()},
            "idempotency_key": "recursive-benchmark-" + cycle["analysis_id"] + "-" + str(iteration + 1)}
        proposal.update(job_type=request["job_type"], objective="llm_benchmark", candidate_profile="exact_qwen_swarm" if cycle["family"] == "qwen" else "typesafe_jev_choice",
            benchmark_family=request["job_type"], budget_category="gpu" if cycle["family"] == "qwen" else "cpu_research",
            tool_request={"name": "submit_experiment", "arguments": request}, manual_compute_approval_required=True,
            external_vendor_cost_cap_usd=.01 if cycle["family"] == "jev" else 0.)
    lineage = (previous.get("lineage", []) if previous else []) + ([{"analysis_id": previous["analysis_id"], "iteration": previous["iteration"], "job_run_id": (previous.get("job") or {}).get("run_id")}] if previous else [])
    payload = {"summary": "Recursive evidence review with bounded, reproducible experiment proposals", "cycle_id": cycle["analysis_id"], "state": "evidence_review", "iteration": iteration,
        "max_iterations": cycle["max_iterations"], "evidence": evidence, "proposed_experiment": None if stop else proposal,
        "lineage": lineage, "activation": {"status": "proposal_only", "required": "explicit reviewed activation with fresh forward evaluation"},
        "benchmark_capabilities": _capabilities(service), "dry_run": dry,
        "unsupported_model_changes": {"status": "reviewable_code_change_required", "proposal": "A released profile may change PPO features/rewards/episode horizon; new model families, arbitrary features or neural architectures require a versioned implementation and evaluation."}}
    if result_summary:
        payload["result_summary"] = result_summary
    if stop:
        payload.update(state="proposal_ready" if stop.startswith("candidate_requires") else "stopped", stopping_reason=stop)
    else:
        # An immutable new review freezes the approved profile and the exact evidence lineage.
        experiment_review = service.issue("research", {**{k: v for k, v in initial.items() if k not in ("analysis_id", "analysis_kind", "created_at", "request_fingerprint", "analysis_ref", "contract_version", "synthetic")},
            "summary": "Recursive candidate experiment frozen for budgeted evaluation", "proposed_experiment": proposal,
        }, {"cycle_id": cycle["analysis_id"], "iteration": iteration + 1, "profile": profile, "parent": previous["analysis_id"] if previous else None, "evidence_review_id": initial["analysis_id"]})
        # Claim only a launch, never a dry preview. Crash recovery uses the same job idempotency.
        claim_key = "recursive/" + cycle["analysis_id"] + "/" + str(iteration + 1)
        claim = {"cycle_id": cycle["analysis_id"], "next_iteration": iteration + 1, "review_id": experiment_review["analysis_id"], "launch_idempotency_key": body.get("idempotency_key")}
        existing_claim = service.store.get_claim(claim_key)
        if not dry and existing_claim:
            if not scheduled and existing_claim.get("launch_idempotency_key") != body.get("idempotency_key"):
                raise FinplanError.precondition("another resume owns this iteration", reason="recursive_iteration_conflict")
            # Recover a crash after the conditional claim using exactly its frozen review.
            claim = existing_claim
            experiment_review = service.store.get(claim["review_id"])
            proposal = experiment_review["proposed_experiment"]
            payload["proposed_experiment"] = proposal
        if not dry and not service.store.claim(claim_key, claim):
            raise FinplanError.precondition("another resume owns this iteration", reason="recursive_iteration_conflict")
        try:
            if cycle["family"] in ("qwen", "jev"):
                request = copy.deepcopy(proposal["tool_request"]["arguments"])
                request.update(max_runtime_seconds=900, compute_class="gpu" if cycle["family"] == "qwen" else "cpu")
                estimate = service.d.job_api.submit(request)
                launched = {"cost_estimate": estimate["cost_estimate"]}
                # Scheduling can observe an approved external/GPU run, but may never create or
                # approve a fresh one. The typed submit_experiment workflow records approval.
                if not dry and not scheduled:
                    request["dry_run"] = False
                    launched["job"] = service.d.job_api.submit(request)
            else:
                launched = run_review(service, {"review_id": experiment_review["analysis_id"], "dry_run": dry, "confirmed_by_user": body.get("confirmed_by_user"), "scheduled": scheduled})
            payload.update(cost_estimate=launched["cost_estimate"], experiment_review_id=experiment_review["analysis_id"])
            if launched.get("job"):
                payload.update(job=launched["job"], state="awaiting_experiment_approval" if launched["job"].get("state") == "awaiting_approval" else "experiment_running", iteration=iteration + 1, launch_idempotency_key=body.get("idempotency_key"))
            else:
                payload["state"] = "awaiting_experiment_approval"
        except FinplanError as exc:
            reason = exc.details.get("reason", exc.code)
            waiting = reason in ("weekly_research_limit", "research_job_overlap")
            payload.update(state="awaiting_experiment_approval" if waiting else "stopped", stopping_reason=reason, error=exc.to_envelope())
    return service.issue("recursive_iteration", payload, {"cycle_id": cycle["analysis_id"], "iteration": payload["iteration"], "parent": previous["analysis_id"] if previous else None, "dry_run": dry,
        "week": service.now().strftime("%G-W%V"), "result_checksum": sha256_checksum(canonical_json_bytes(result_summary)) if result_summary else None,
        "state": payload["state"]}, portfolio_id=cycle.get("portfolio_id"))


def scheduled_recursive_review(service):
    """Resume at most one eligible cycle per weekly invocation, sharing the weekly lease/cap."""
    cycles = service.store.list(kind="recursive_cycle", limit=20)
    for cycle in cycles:
        if cycle.get("family") not in ("ppo", "classical"):
            continue
        latest = _latest(service, cycle)
        authorized = latest and (latest.get("job") or any(row.get("job_run_id") for row in latest.get("lineage", [])))
        if authorized and latest["state"] not in TERMINAL:
            return run_recursive_improvement(service, {"cycle_id": cycle["analysis_id"], "dry_run": False}, scheduled=True)
    return None
