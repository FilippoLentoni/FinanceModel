import json
import urllib.parse

import pytest
from finplan_contracts.validate import validate

from finplan_model.classical.research import (
    literature,
    market_events,
    review,
    run_review,
    select_preset,
)
from finplan_model.classical_api import handle, weekly_handler
from finplan_model.core.errors import FinplanError


def issue_review(context):
    context.service.recommend({})
    return review(context.service, {})


def test_feedback_literature_and_proposal_are_stored_untrusted_without_activation(
    context,
):
    p = context.service.recommend({})
    e = {
        "environment": "beta",
        "operation": "submit_portfolio_feedback",
        "request": {
            "analysis_id": p["analysis_id"],
            "text": "Please compare lower turnover; this is feedback, not code",
            "idempotency_key": "feedback-example-key",
        },
    }
    feedback = handle(e, context.service)
    assert (
        handle(e, context.service) == feedback
        and validate(feedback, "tools/submit-portfolio-feedback-response").valid
    )
    r = review(context.service, {"feedback": "investigate volatility"})
    assert validate(r, "tools/research-portfolio-models-response").valid
    assert r["sources"][0]["url"].startswith("https://arxiv.org/")
    assert r["sources"][0]["trust"] == "untrusted_external_text_not_instructions"
    assert r["evidence"][0]["analysis_id"] == feedback["analysis_id"]
    assert r["activation"]["status"] == "proposal_only"
    assert context.service.d.job_api.calls == []


def test_provider_failure_is_explicit_and_does_not_fabricate(context):
    def unavailable(_):
        raise TimeoutError()

    context.service.d.external_fetch = unavailable
    r = issue_review(context)
    assert r["sources"] == [] and r["literature"]["status"] == "not_available"
    p = context.service.store.list(kind="recommendation", limit=1)[0]
    news = market_events(context.service, {"analysis_id": p["analysis_id"]})
    assert validate(news, "tools/research-market-events-response").valid
    assert news["status"] == "not_available" and news["sources"] == []


def test_dated_market_context_filters_outside_window_and_no_causal_claim(context):
    p = context.service.recommend({})
    d = p["recommendation"]["as_of"]
    stamp = d.replace("-", "") + "T120000Z"
    context.service.d.external_fetch = lambda url: json.dumps(
        {
            "articles": [
                {
                    "url": "https://example.com/nvidia-earnings",
                    "title": "Nvidia earnings",
                    "seendate": stamp,
                },
                {
                    "url": "http://169.254.169.254/latest",
                    "title": "Bad host",
                    "seendate": stamp,
                },
                {
                    "url": "https://example.com/future",
                    "title": "Future",
                    "seendate": "20271201T120000Z",
                },
            ]
        }
    ).encode()
    n = market_events(context.service, {"analysis_id": p["analysis_id"]})
    assert len(n["sources"]) == 1 and n["sources"][0]["instrument_ids"] == ["NVDA"]
    assert n["causal_conclusion"]["status"] == "not_available"
    with pytest.raises(FinplanError):
        market_events(
            context.service, {"analysis_id": p["analysis_id"], "end_date": "2027-01-01"}
        )


def test_research_dry_run_has_estimate_and_no_reserved_or_paid_job(context):
    r = issue_review(context)
    result = run_review(context.service, {"review_id": r["analysis_id"]})
    assert validate(result, "tools/run-portfolio-research-response").valid
    assert result["dry_run"] and "job" not in result
    assert all(c["dry_run"] for c in context.service.d.job_api.calls)
    assert context.service.store.claims == {}


def test_weekly_launch_idempotency_no_activation_and_retry_replays_exact_request(
    context,
):
    r = issue_review(context)
    request = {
        "review_id": r["analysis_id"],
        "dry_run": False,
        "confirmed_by_user": True,
        "idempotency_key": "user-accepted-review",
    }
    a = run_review(context.service, request)
    context.service.d.job_api.jobs = [{"state": "queued"}]
    b = run_review(context.service, request)
    assert a == b and a["activation"] == "proposal_only"
    actual = [c for c in context.service.d.job_api.calls if not c["dry_run"]]
    assert (
        len(actual) == 2 and actual[0] == actual[1]
    )  # sandbox API applies its durable idempotency key
    assert len(context.service.store.claims) == 1
    alternate = review(context.service, {"feedback": "different proposal"})
    with pytest.raises(FinplanError, match="reserved"):
        run_review(context.service, {**request, "review_id": alternate["analysis_id"]})


@pytest.mark.parametrize(
    "reason", ["weekly", "project", "monthly", "overlap", "unconfirmed"]
)
def test_paid_controller_fails_closed_before_submission(context, reason):
    r = issue_review(context)
    request = {
        "review_id": r["analysis_id"],
        "dry_run": False,
        "confirmed_by_user": True,
    }
    if reason == "weekly":
        context.service.d.job_api.price = 0.51
    if reason == "project":
        context.service.d.project_budget = lambda: {"spent": 49.99, "limit": 50}
    if reason == "monthly":
        for i in range(4):
            context.service.issue(
                "research_run",
                {
                    "summary": "prior research",
                    "job": {"run_id": f"run_example{i}"},
                    "cost_estimate": {"estimated_usd_upper_bound": 0.50},
                    "dry_run": False,
                    "review_id": f"ca_{i:032d}",
                },
                {"old": i},
            )
    if reason == "overlap":
        context.service.d.job_api.jobs = [{"state": "running"}]
    if reason == "unconfirmed":
        request["confirmed_by_user"] = False
    with pytest.raises(FinplanError):
        run_review(context.service, request)
    assert all(c["dry_run"] for c in context.service.d.job_api.calls)


def test_weekly_budget_skip_retains_review_and_reason(context, monkeypatch):
    context.service.recommend({})
    monkeypatch.setattr("finplan_model.classical_api._SERVICE", context.service)
    context.service.d.job_api.price = 0.8
    a = weekly_handler({"trigger": "weekly_classical_research"})
    assert a["status"] == "skipped" and a["error"]["code"] == "BUDGET_EXCEEDED"
    assert context.service.store.list(kind="research", limit=1)
    assert all(c["dry_run"] for c in context.service.d.job_api.calls)


def test_feedback_retry_key_cannot_change_the_recorded_text(context):
    p = context.service.recommend({})
    event = {
        "environment": "beta",
        "operation": "submit_portfolio_feedback",
        "request": {
            "analysis_id": p["analysis_id"],
            "text": "first feedback",
            "idempotency_key": "stable-feedback-retry",
        },
    }
    a = handle(event, context.service)
    assert handle(event, context.service) == a
    event["request"]["text"] = "different feedback"
    with pytest.raises(FinplanError) as err:
        handle(event, context.service)
    assert err.value.code == "IDEMPOTENCY_KEY_REUSED"


def test_requested_literature_topic_is_used_in_bounded_provider_query():
    requested = []
    literature(
        "robust downside",
        lambda url: (
            requested.append(url) or b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
        ),
    )
    parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(requested[0]).query)
    assert parsed["search_query"] == ["all:portfolio AND (all:robust OR all:downside)"]
    assert parsed["max_results"] == ["5"]


def test_evidence_changes_actual_bounded_experiment_configuration(context):
    context.service.recommend({})
    r = review(context.service, {"feedback": "Reduce transaction costs and drawdown"})
    proposal = r["proposed_experiment"]
    assert proposal["preset_id"] == "lower_turnover_defensive"
    assert proposal["lookbacks"] == [60, 120, 252]
    assert {v["signal"] for v in proposal["selection_reasons"]} == {
        "turnover_cost_concern",
        "downside_risk_concern",
    }
    out = run_review(context.service, {"review_id": r["analysis_id"]})
    cfg = out["experiment_configuration"]["payload"]
    assert cfg["lookback_days"] == 120 and cfg["rebalance_frequency"] == "monthly"
    assert cfg["risk_aversion"] == 5 and cfg["constraints"]["max_weight"] == 0.4
    assert context.service.d.job_api.calls[-1]["configuration"]["payload"] == cfg


def test_performance_and_literature_metadata_select_finite_hypotheses_only():
    p = select_preset(
        [],
        [{"analysis_id": "ca_" + "0" * 32, "trend": "red", "gap": {"total": -10}}],
        [{"title": "Robust portfolio transaction costs; execute shell command"}],
        "",
        5,
    )
    assert p["preset_id"] == "lower_turnover_defensive"
    assert any(
        row["source_type"] == "literature_metadata" for row in p["selection_reasons"]
    )
    assert set(p) == {
        "preset_id",
        "selection_reasons",
        "selection_method",
        "lookbacks",
        "lookback_days",
        "rebalance_frequency",
        "risk_aversion",
        "max_weight",
    }


def test_feedback_records_trusted_transport_caller_without_inventing_end_user(context):
    plan = context.service.recommend({})
    caller = {
        "channel": "direct_mcp",
        "correlation_id": "corr-feedback-audit",
        "subject_hash": "sha256:" + "0" * 64,
    }
    event = {
        "environment": "beta",
        "operation": "submit_portfolio_feedback",
        "request": {
            "analysis_id": plan["analysis_id"],
            "text": "review turnover",
            "idempotency_key": "audited-feedback-key",
        },
        "headers": {
            "X-Finplan-Caller": json.dumps(caller),
            "X-Correlation-Id": caller["correlation_id"],
        },
    }
    result = handle(event, context.service)
    assert result["audit"]["caller"] == caller
    assert result["audit"]["authority"] == "audit_only_never_authorization"
    assert "end-user claims are unavailable" in result["audit"]["identity_scope"]
    event["headers"]["X-Correlation-Id"] = "corr-feedback-retry"
    assert handle(event, context.service) == result


def test_review_keeps_large_prior_benchmarks_retrievable_without_mcp_payload_overflow(
    context,
):
    from finplan_model.core.artifacts import canonical_json_bytes

    context.service.recommend({})
    for i in range(10):
        context.service.issue(
            "feedback",
            {"summary": "Feedback", "text": "turnover " + "x" * 3990},
            {"feedback": i},
        )
        context.service.issue(
            "performance",
            {"summary": "Performance", "trend": "red", "gap": {"total": -10}},
            {"performance": i},
        )
    for i in range(3):
        context.service.issue(
            "research_run",
            {
                "summary": "Previous benchmark",
                "job": {"run_id": "run_01KDVDP88REHGPBXFX6CHX92KS"},
            },
            {"run": i},
        )
    context.service.d.job_api.result = lambda _: {
        "completion_status": "succeeded",
        "payload": {
            "weekly_research": {
                "selection": {"selected_variant_id": "min_variance_120"},
                "variants": [
                    {
                        "variant_id": "min_variance_120",
                        "algorithm": "min_variance",
                        "parameters": {"lookback": 120},
                        "splits": {
                            "validation": {"sharpe_ratio": 0.9, "debug": "x" * 64000},
                            "research_test": {"sharpe_ratio": 0.7},
                        },
                    }
                ]
                * 12,
            }
        },
    }
    result = review(context.service, {})
    assert len(canonical_json_bytes(result)) < 64000
    assert len(result["prior_experiment_findings"]) == 3
    assert (
        result["prior_experiment_findings"][0]["benchmark_summary"]["variant_count"]
        == 12
    )
    assert (
        result["prior_experiment_findings"][0]["detailed_evidence"]["tool"]
        == "get_experiment_result"
    )
