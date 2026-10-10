import json
import urllib.parse
from datetime import timedelta

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


@pytest.mark.parametrize("scheduled", [False, True])
def test_review_pins_current_saved_plan_despite_newer_historical_scenario(
    context, monkeypatch, scheduled
):
    service = context.service
    current = service.recommend({})
    now = service.now()
    service.now = lambda: now + timedelta(seconds=1)
    historical = service.recommend(
        {
            "input_snapshot_id": context.sid,
            "as_of": context.dates[90].isoformat(),
            "holdings": {
                "weights": [
                    {"instrument_id": i, "weight": 0.2}
                    for i in context.market.instruments
                ],
                "cash_weight": 0.0,
                "portfolio_value": 10000.0,
                "high_watermark": 10000.0,
            },
        }
    )
    assert (
        service.store.list(kind="recommendation", limit=1)[0]["analysis_id"]
        == historical["analysis_id"]
    )
    assert historical["created_at"] > current["created_at"]
    assert historical["recommendation"]["portfolio_state"]["source"] == "supplied"
    assert service.recommend({}) == current
    saved_book_before = json.loads(json.dumps(context.platform.portfolio_states))
    calls = []
    recommend = service.recommend

    def tracked_recommend(body):
        calls.append(body)
        return recommend(body)

    monkeypatch.setattr(service, "recommend", tracked_recommend)
    if scheduled:
        monkeypatch.setattr("finplan_model.classical_api._SERVICE", service)
        result = weekly_handler({"trigger": "weekly_classical_research"})
        issued_review = service.store.get(result["review_id"])
    else:
        issued_review = review(service, {})
        assert review(service, {}) == issued_review
        result = run_review(service, {"review_id": issued_review["analysis_id"]})
    assert issued_review["plan_context_id"] == current["analysis_id"]
    assert calls == ([{}] if scheduled else [{}, {}])
    assert context.platform.portfolio_states == saved_book_before
    assert (
        service.store.get(historical["analysis_id"])["recommendation"]
        == historical["recommendation"]
    )
    submitted = service.d.job_api.calls[-1]
    assert (
        submitted["input_snapshot_id"] == current["recommendation"]["input_snapshot_id"]
    )
    assert submitted["evaluation_window"]["end"] == current["recommendation"]["as_of"]
    assert (
        submitted["evaluation_window"]["end"] != historical["recommendation"]["as_of"]
    )
    assert result["activation"] == "proposal_only"


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


def test_same_week_same_review_request_refreshes_completed_job_evidence_once(context):
    context.service.recommend({})
    request = {"query": "portfolio covariance"}
    initial = review(context.service, request)
    run_review(
        context.service,
        {
            "review_id": initial["analysis_id"],
            "dry_run": False,
            "confirmed_by_user": True,
        },
    )
    completed_result = context.service.d.job_api.result

    def pending_result(_):
        raise FinplanError.precondition("run is pending", reason="run_not_terminal")

    context.service.d.job_api.result = pending_result
    before = review(context.service, request)
    assert before["prior_experiment_findings"][0]["status"] == "not_available"
    assert review(context.service, request) == before
    context.service.d.job_api.result = completed_result
    after = review(context.service, request)
    assert after["analysis_id"] != before["analysis_id"]
    assert after["prior_experiment_findings"][0]["status"] == "succeeded"
    assert review(context.service, request) == after
    assert (
        context.service.store.get(before["analysis_id"])["prior_experiment_findings"][
            0
        ]["status"]
        == "not_available"
    )

    paid_before = sum(not c["dry_run"] for c in context.service.d.job_api.calls)
    with pytest.raises(FinplanError) as exc:
        run_review(
            context.service,
            {
                "review_id": after["analysis_id"],
                "dry_run": False,
                "confirmed_by_user": True,
            },
        )
    assert exc.value.details["reason"] == "weekly_research_limit"
    assert sum(not c["dry_run"] for c in context.service.d.job_api.calls) == paid_before
    assert len(context.service.store.claims) == 1


@pytest.mark.parametrize("change", ["source", "status"])
def test_same_week_same_review_request_refreshes_changed_literature(context, change):
    context.service.recommend({})
    request = {"query": "portfolio covariance"}
    before = review(context.service, request)
    original_fetch = context.service.d.external_fetch
    if change == "source":
        context.service.d.external_fetch = lambda url: original_fetch(url).replace(
            b"2601.00001", b"2601.00002"
        )
    else:

        def unavailable(_):
            raise TimeoutError()

        context.service.d.external_fetch = unavailable
    after = review(context.service, request)
    assert before["analysis_id"] != after["analysis_id"]
    assert review(context.service, request) == after
    context.service.d.external_fetch = original_fetch
    assert review(context.service, request) == before


def test_review_identity_ignores_literature_retrieval_timestamps(context, monkeypatch):
    import copy

    context.service.recommend({})
    metadata = literature("portfolio covariance", context.service.d.external_fetch)
    monkeypatch.setattr(
        "finplan_model.classical.research.literature",
        lambda query, fetch: copy.deepcopy(metadata),
    )
    request = {"query": "portfolio covariance"}
    before = review(context.service, request)
    metadata["retrieved_at"] = "2026-10-10T12:30:00Z"
    metadata["sources"][0]["retrieved_at"] = "2026-10-10T12:30:00Z"
    assert review(context.service, request) == before


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


def test_unapproved_proposal_does_not_block_or_get_changed_by_weekly_research(context):
    r = issue_review(context)
    old_proposal = {
        "run_id": "run_01KDVDP88REHGPBXFX6CHX92KR",
        "state": "awaiting_approval",
        "elapsed_seconds": 0,
        "wait_reason": "awaiting_human_approval",
    }
    context.service.d.job_api.jobs = [dict(old_proposal)]
    result = run_review(
        context.service,
        {
            "review_id": r["analysis_id"],
            "dry_run": False,
            "confirmed_by_user": True,
        },
    )
    assert result["job"]["state"] == "queued"
    assert context.service.d.job_api.jobs == [old_proposal]
    assert sum(not c["dry_run"] for c in context.service.d.job_api.calls) == 1
    assert len(context.service.store.claims) == 1


@pytest.mark.parametrize(
    "state", ["queued", "starting", "running", "stopping", "unknown", None]
)
def test_weekly_overlap_blocks_compute_candidates_and_unknown_states(context, state):
    r = issue_review(context)
    context.service.d.job_api.jobs = [{"state": state}]
    with pytest.raises(FinplanError) as exc:
        run_review(
            context.service,
            {
                "review_id": r["analysis_id"],
                "dry_run": False,
                "confirmed_by_user": True,
            },
        )
    assert exc.value.details["reason"] == "research_job_overlap"
    assert all(c["dry_run"] for c in context.service.d.job_api.calls)
    assert context.service.store.claims == {}


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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Portfolio covariance portfolio shrinkage",
            "all:portfolio AND (all:covariance OR all:shrinkage)",
        ),
        ("portfolio PORTFOLIO", "all:portfolio"),
        ("?!", "all:portfolio"),
        (
            None,
            "all:portfolio AND (all:optimization OR all:reinforcement OR all:learning OR all:covariance OR all:transaction OR all:costs)",
        ),
    ],
)
def test_literature_topics_constrain_portfolio_search_with_safe_empty_fallback(
    query, expected
):
    requested = []
    literature(
        query,
        lambda url: (
            requested.append(url) or b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
        ),
    )
    parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(requested[0]).query)
    assert parsed["search_query"] == [expected]
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
