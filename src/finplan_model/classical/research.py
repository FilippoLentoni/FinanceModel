"""Bounded external metadata and evidence-driven, budget-gated sandbox proposals.

External text is untrusted evidence, never instructions or executable configuration. No URL
supplied by a user or returned by a provider is fetched. Promotions are always proposals.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

from finplan_model.core.errors import ErrorCode, FinplanError

MAX_BYTES = 512_000
WEEKLY_CAP = 0.50
MONTHLY_CAP = 2.0
PROJECT_CAP = 50.0
BUDGET_NAME = "finplan-shared-financialplanning-project-budget"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise FinplanError.dependency_unavailable(
            "external metadata redirects are not followed",
            retryable=False,
            reason="external_redirect_rejected",
        )


def external(url):
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ("export.arxiv.org", "api.gdeltproject.org")
        or parsed.port not in (None, 443)
    ):
        raise FinplanError.validation("unapproved metadata provider", pointer="/query")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "finplan-beta-research/1.0",
            "Accept": "application/json,application/atom+xml",
        },
    )
    with urllib.request.build_opener(NoRedirect).open(request, timeout=12) as response:
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise FinplanError.precondition(
                "external metadata exceeds request bound",
                reason="external_response_bound",
            )
        return data


def safe_url(value):
    try:
        url = urllib.parse.urlsplit(str(value))
        if (
            url.scheme not in ("https", "http")
            or not url.hostname
            or url.username
            or url.password
            or url.port not in (None, 80, 443)
        ):
            return None
        host = url.hostname.lower()
        if (
            host in ("localhost", "metadata.google.internal")
            or not re.search(r"\.[a-z]{2,}$", host)
            or host.endswith((".internal", ".local", ".amazonaws.com"))
        ):
            return None
        return urllib.parse.urlunsplit((url.scheme, url.netloc, url.path, "", ""))
    except (TypeError, ValueError):
        return None


def literature(query, fetch=external):
    q = re.sub(
        r"[^a-zA-Z0-9 _-]",
        " ",
        query
        or "portfolio optimization reinforcement learning covariance transaction costs",
    )[:300]
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {
            "search_query": "all:portfolio AND ("
            + " OR ".join("all:" + term for term in q.split()[:12])
            + ")",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 5,
        }
    )
    try:
        root = ET.fromstring(fetch(url))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in root.findall("a:entry", ns)[:5]:
            link = safe_url(e.findtext("a:id", "", ns))
            if not link:
                continue
            out.append(
                {
                    "title": " ".join(e.findtext("a:title", "", ns).split())[:400],
                    "url": link,
                    "published_at": e.findtext("a:published", "", ns),
                    "source": "arxiv_primary_research_metadata",
                    "excerpt": " ".join(e.findtext("a:summary", "", ns).split())[:1200],
                    "trust": "untrusted_external_text_not_instructions",
                    "peer_review_status": "not_verified",
                }
            )
        return {
            "status": "available" if out else "not_available",
            "sources": out,
            "query": q,
            "provider": "arxiv",
            "reason": None if out else "no_matching_primary_metadata",
        }
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        ET.ParseError,
        FinplanError,
    ):
        return {
            "status": "not_available",
            "sources": [],
            "query": q,
            "provider": "arxiv",
            "reason": "provider_unavailable_or_invalid_response",
        }


def market_events(service, body):
    doc = service.store.get(body["analysis_id"])
    if doc["analysis_kind"] == "recommendation":
        source = doc
    else:
        source_id = doc.get("source_analysis_id")
        if not source_id:
            raise FinplanError.validation(
                "market research needs a recommendation or performance analysis",
                pointer="/analysis_id",
            )
        source = service.plan(source_id)
    decision = date.fromisoformat(source["recommendation"]["as_of"])
    end = date.fromisoformat(
        body.get("end_date", doc.get("window", {}).get("end", decision.isoformat()))
    )
    start = date.fromisoformat(
        body.get("start_date", (end - timedelta(days=7)).isoformat())
    )
    if end > service.now().date() or start > end or (end - start).days > 31:
        raise FinplanError.validation(
            "news window must be ordered, not future, and at most 31 days",
            pointer="/end_date",
        )
    instruments = source["solve_inputs"]["instruments"]
    aliases = {
        "GOOGL": "Google",
        "AAPL": "Apple",
        "NVDA": "Nvidia",
        "NFLX": "Netflix",
        "VOO": "stock market",
    }
    terms = [aliases.get(i, i) for i in instruments]
    query = "(" + " OR ".join('"' + t + '"' for t in terms) + ")"
    free = re.sub(r"[^a-zA-Z0-9 _-]", " ", body.get("query", ""))[:200]
    if free.strip():
        query += " " + free
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(
        {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 10,
            "sort": "datedesc",
            "startdatetime": start.strftime("%Y%m%d000000"),
            "enddatetime": end.strftime("%Y%m%d235959"),
        }
    )
    cutoff = (
        datetime.fromisoformat(
            source["solve_inputs"].get(
                "decision_time", decision.isoformat() + "T21:00:00+00:00"
            )
        )
        if end <= decision
        else None
    )
    fetch = getattr(service.d, "external_fetch", external)
    sources, status, reason = [], "available", None
    try:
        data = json.loads(fetch(url))
        for article in data.get("articles", [])[:10]:
            link = safe_url(article.get("url"))
            stamp = str(article.get("seendate", ""))
            try:
                seen_at = datetime.strptime(stamp, "%Y%m%dT%H%M%S%z")
                seen = seen_at.date()
            except ValueError:
                continue
            if not link or not start <= seen <= end or (cutoff and seen_at > cutoff):
                continue
            title = str(article.get("title", ""))[:500]
            matched = [
                i
                for i in instruments
                if aliases.get(i, i).lower() in title.lower()
                or i.lower() in title.lower()
            ]
            sources.append(
                {
                    "title": title,
                    "url": link,
                    "published_at": stamp,
                    "source": "gdelt_document_metadata",
                    "instrument_ids": matched,
                    "trust": "untrusted_external_text_not_instructions",
                    "timestamp_semantics": "provider first-observed time; original publication time not verified",
                }
            )
        if not sources:
            status, reason = "not_available", "no_matching_dated_metadata"
    except (OSError, ValueError, TypeError, AttributeError, FinplanError):
        status, reason = "not_available", "provider_unavailable_or_invalid_response"
    return service.issue(
        "market_events",
        {
            "summary": "Dated market-event leads for stakeholder investigation; no causal market attribution is established",
            "source_analysis_id": source["analysis_id"],
            "status": status,
            "reason": reason,
            "sources": sources,
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "query": query,
            "analysis_mode": "retrospective_context"
            if end > decision
            else "information_available_by_decision_date",
            "causal_conclusion": {
                "status": "not_available",
                "reason": "headlines and coincident stock moves do not establish causality",
            },
        },
        {
            "source_analysis_id": source["analysis_id"],
            "window": [start.isoformat(), end.isoformat()],
            "query": query,
            "retrieval_day": service.now().date().isoformat(),
        },
        portfolio_id=source.get("portfolio_id"),
    )


def select_preset(feedback, performance, sources, user_feedback, n_instruments):
    """Typed signals select only audited finite presets; source text never becomes code."""
    reasons = []
    lower_turnover = False
    defensive = False
    for row in feedback + [{"analysis_id": None, "text": user_feedback}]:
        text = row.get("text", "").lower()
        if any(
            word in text
            for word in ("turnover", "fees", "transaction cost", "trade less")
        ):
            lower_turnover = True
            reasons.append(
                {
                    "source_type": "user_feedback",
                    "analysis_id": row.get("analysis_id"),
                    "signal": "turnover_cost_concern",
                }
            )
        if any(word in text for word in ("drawdown", "downside", "cvar", "risk")):
            defensive = True
            reasons.append(
                {
                    "source_type": "user_feedback",
                    "analysis_id": row.get("analysis_id"),
                    "signal": "downside_risk_concern",
                }
            )
    for row in performance:
        if (row.get("gap", {}).get("total") or 0) < 0:
            lower_turnover = True
            reasons.append(
                {
                    "source_type": "performance",
                    "analysis_id": row["analysis_id"],
                    "signal": "negative_implementation_gap_hypothesis_not_causal_proof",
                }
            )
        if row.get("trend") == "red":
            defensive = True
            reasons.append(
                {
                    "source_type": "performance",
                    "analysis_id": row["analysis_id"],
                    "signal": "negative_observed_paper_return",
                }
            )
    for i, row in enumerate(sources):
        title = row.get("title", "").lower()
        if any(word in title for word in ("transaction cost", "turnover")):
            lower_turnover = True
            reasons.append(
                {
                    "source_type": "literature_metadata",
                    "source_index": i,
                    "signal": "transaction_cost_topic_only_unverified",
                }
            )
        if any(word in title for word in ("cvar", "downside", "robust", "tail risk")):
            defensive = True
            reasons.append(
                {
                    "source_type": "literature_metadata",
                    "source_index": i,
                    "signal": "downside_risk_topic_only_unverified",
                }
            )
    preset_id = (
        "lower_turnover_defensive"
        if lower_turnover and defensive
        else "lower_turnover"
        if lower_turnover
        else "defensive"
        if defensive
        else "baseline_sensitivity"
    )
    return {
        "preset_id": preset_id,
        "selection_reasons": reasons,
        "selection_method": "deterministic bounded signal rules; metadata topics suggest hypotheses, never paper instructions",
        "lookbacks": [60, 120, 252] if lower_turnover else [20, 60, 120],
        "lookback_days": 120 if lower_turnover else 60,
        "rebalance_frequency": "monthly" if lower_turnover else "weekly",
        "risk_aversion": 5.0 if defensive else 2.0,
        "max_weight": max(1 / max(1, n_instruments), 0.4)
        if defensive
        else max(1 / max(1, n_instruments), 0.6),
    }


def review(service, body):
    sources = literature(
        body.get("query"), getattr(service.d, "external_fetch", external)
    )
    feedback = service.store.list(kind="feedback", limit=10)
    performance = service.store.list(kind="performance", limit=10)
    plans = service.store.list(kind="recommendation", limit=1)
    prior_runs = service.store.list(kind="research_run", limit=20)
    findings = []
    for prior in prior_runs[:3]:
        run_id = prior.get("job", {}).get("run_id")
        if not run_id:
            continue
        try:
            result = service.d.job_api.result(run_id)
            result_payload = result.get("payload", {})
            benchmark = (
                result_payload.get("weekly_research")
                or result_payload.get("benchmark")
                or {}
            )
            selection = benchmark.get("selection") or {}
            selected_id = selection.get("selected_variant_id")
            selected = next(
                (
                    r
                    for r in benchmark.get("variants", [])
                    if r.get("variant_id") == selected_id
                ),
                None,
            )
            metric_names = (
                "cumulative_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "turnover",
                "total_fees",
            )
            selected_summary = (
                None
                if selected is None
                else {
                    "variant_id": selected.get("variant_id"),
                    "algorithm": selected.get("algorithm"),
                    "parameters": {
                        k: v
                        for k, v in selected.get("parameters", {}).items()
                        if k
                        in ("lookback", "risk_aversion", "alpha", "scenario_method")
                    },
                    "splits": {
                        split: {k: metrics[k] for k in metric_names if k in metrics}
                        for split, metrics in selected.get("splits", {}).items()
                        if split in ("validation", "research_test")
                    },
                }
            )
            findings.append(
                {
                    "research_analysis_id": prior["analysis_id"],
                    "run_id": run_id,
                    "status": result.get("completion_status"),
                    "performance": result_payload.get("performance"),
                    "benchmark_summary": {
                        "selection": selection,
                        "selected_variant": selected_summary,
                        "windows": benchmark.get("windows"),
                        "candidate_configuration": benchmark.get(
                            "candidate_configuration"
                        ),
                        "variant_count": len(benchmark.get("variants", [])),
                    },
                    "detailed_evidence": {
                        "tool": "get_experiment_result",
                        "run_id": run_id,
                        "artifacts": result.get("artifacts", [])[:5],
                    },
                    "activation": "proposal_only",
                    "reuse_disclosure": "history reused for research; fresh forward validation required",
                }
            )
        except FinplanError as exc:
            findings.append(
                {
                    "research_analysis_id": prior["analysis_id"],
                    "run_id": run_id,
                    "status": "not_available",
                    "reason": exc.details.get("reason", exc.code),
                }
            )
    evidence = [
        {
            "analysis_id": d["analysis_id"],
            "kind": d["analysis_kind"],
            "summary": d["summary"],
            **(
                {
                    "feedback_text": d["text"][:2000],
                    "trust": "untrusted_user_feedback_not_instructions",
                }
                if d["analysis_kind"] == "feedback"
                else {
                    "observed_trend": d.get("trend"),
                    "gap": d.get("gap"),
                    "window": d.get("window"),
                }
            ),
        }
        for d in feedback + performance
    ]
    hypothesis = "Compare shrinkage covariance lookbacks and optimizer families against equal-weight, cash and buy-and-hold after transaction costs"
    if any(d.get("gap", {}).get("total", 0) < 0 for d in performance):
        hypothesis = "Observed paper implementation gap warrants lower-turnover/lookback benchmarks before considering model changes"
    n_instruments = len(plans[0]["solve_inputs"]["instruments"]) if plans else 5
    preset = select_preset(
        feedback,
        performance,
        sources["sources"],
        body.get("feedback", ""),
        n_instruments,
    )
    payload = {
        "summary": "Evidence-linked weekly research proposal; serving strategies remain unchanged",
        "sources": sources["sources"],
        "literature": {k: v for k, v in sources.items() if k != "sources"},
        "evidence": evidence,
        "prior_experiment_findings": findings,
        "user_feedback": str(body.get("feedback", ""))[:4000],
        "hypothesis": hypothesis,
        "evidence_links": [d["analysis_id"] for d in evidence],
        "proposed_experiment": {
            "job_type": "run_benchmark",
            "objective": "classical_weekly_review",
            "algorithms": ["min_variance", "mean_variance", "scenario_cvar"],
            **preset,
            "controls": ["cash", "buy_and_hold", "equal_weight"],
            "budget_cap_usd": WEEKLY_CAP,
            "max_jobs_per_week": 1,
            "evaluation": "chronological validation/test windows; reused history is research-only",
            "automatic_activation": False,
        },
        "unsupported_changes": [
            {
                "proposal": "new features, neural architecture or source changes require a reviewed code/spec change",
                "automatic": False,
            }
        ],
        "plan_context_id": plans[0]["analysis_id"] if plans else None,
        "activation": {
            "status": "proposal_only",
            "required": "explicit reviewed strategy activation after fresh forward validation",
        },
    }
    return service.issue(
        "research",
        payload,
        {
            "week": service.now().strftime("%G-W%V"),
            "query": body.get("query"),
            "feedback": body.get("feedback"),
            "evidence": evidence,
            "plan_id": payload["plan_context_id"],
        },
    )


def run_review(service, body):
    doc = service.store.get(body["review_id"])
    if doc["analysis_kind"] != "research":
        raise FinplanError.validation(
            "review_id must refer to a research review", pointer="/review_id"
        )
    plan_id = doc.get("plan_context_id")
    if not plan_id:
        raise FinplanError.precondition(
            "issue a classical plan before benchmarking its universe",
            reason="research_plan_missing",
        )
    plan = service.plan(plan_id)
    inputs = plan["solve_inputs"]
    end = date.fromisoformat(inputs["as_of"])
    start = end - timedelta(days=365)
    proposal = doc["proposed_experiment"]
    request = {
        "domain": "finance",
        "domain_schema_version": "1.0",
        "job_type": "run_benchmark",
        "purpose": "research",
        "dry_run": True,
        "input_snapshot_id": inputs["input_snapshot_id"],
        "contract_version": "1.4.0",
        "synthetic": True,
        "configuration": {
            "domain": "finance",
            "domain_schema_version": "1.0",
            "synthetic": True,
            "payload": {
                "strategy": "min_variance",
                "objective": "classical_weekly_review",
                "universe": inputs["instruments"],
                "lookback_days": proposal["lookback_days"],
                "risk_aversion": proposal["risk_aversion"],
                "rebalance_frequency": proposal["rebalance_frequency"],
                "constraints": {
                    "long_only": True,
                    "max_weight": proposal["max_weight"],
                },
                "fees": {"transaction_cost_bps": 2},
            },
        },
        "evaluation_window": {"start": start.isoformat(), "end": end.isoformat()},
        "max_runtime_seconds": 900,
        "idempotency_key": "classical-weekly-" + service.now().strftime("%G-W%V"),
        "compute_class": "cpu",
    }
    client = service.d.job_api
    estimate = client.submit(request)
    amount = float(estimate["cost_estimate"]["estimated_usd_upper_bound"])
    if not math.isfinite(amount) or not 0 <= amount <= WEEKLY_CAP:
        raise FinplanError(
            ErrorCode.BUDGET_EXCEEDED,
            "weekly research estimate exceeds its hard cap",
            details={
                "estimated_usd_upper_bound": amount,
                "remaining_allocation_usd": WEEKLY_CAP,
                "budget_category": "cpu_research",
            },
        )
    dry = body.get("dry_run", True)
    payload = {
        "summary": "Sandbox benchmark dry run; no paid work started"
        if dry
        else "Bounded weekly sandbox benchmark submitted; no strategy activated",
        "review_id": doc["analysis_id"],
        "dry_run": dry,
        "cost_estimate": estimate["cost_estimate"],
        "sources": doc.get("sources", []),
        "experiment_configuration": request["configuration"],
        "activation": "proposal_only",
    }
    if not dry:
        if body.get("confirmed_by_user") is not True and not body.get("scheduled"):
            raise FinplanError.precondition(
                "paid research requires recorded user intent",
                reason="confirmation_required",
            )
        runs = client.list_jobs()
        claim_key = "weekly/" + service.now().strftime("%G-w%V")
        existing_claim = service.store.get_claim(claim_key)
        active = [
            r
            for r in runs
            if r.get("state") not in ("succeeded", "failed", "cancelled", "timed_out")
        ]
        if active and existing_claim is None:
            raise FinplanError.precondition(
                "a sandbox job is already active", reason="research_job_overlap"
            )
        month = service.now().strftime("%Y-%m")
        own = [
            r
            for r in service.store.list(kind="research_run", limit=100)
            if r.get("job", {}).get("run_id")
            and not r.get("dry_run")
            and str(r["created_at"]).startswith(month)
        ]
        spent = sum(
            float((r.get("cost_estimate") or {}).get("estimated_usd_upper_bound", 0))
            for r in own
            if r.get("review_id") != doc["analysis_id"]
        )
        if spent + amount > MONTHLY_CAP:
            raise FinplanError(
                ErrorCode.BUDGET_EXCEEDED,
                "monthly weekly-research cap reached",
                details={
                    "budget_category": "cpu_research",
                    "estimated_usd_upper_bound": amount,
                    "remaining_allocation_usd": max(0.0, MONTHLY_CAP - spent),
                },
            )
        budget = service.d.project_budget()
        if (
            not math.isfinite(budget["spent"])
            or budget["limit"] > PROJECT_CAP
            or budget["spent"] + amount > min(PROJECT_CAP, budget["limit"])
        ):
            raise FinplanError(
                ErrorCode.BUDGET_EXCEEDED,
                "project budget cannot cover weekly research",
                details={
                    "budget_category": "cpu_research",
                    "estimated_usd_upper_bound": amount,
                    "remaining_allocation_usd": max(
                        0.0, min(PROJECT_CAP, budget["limit"]) - budget["spent"]
                    ),
                },
            )
        claim = {
            "review_id": doc["analysis_id"],
            "idempotency_key": request["idempotency_key"],
            "estimated_usd_upper_bound": amount,
        }
        if existing_claim is not None and existing_claim != claim:
            raise FinplanError.precondition(
                "this week already reserved a different research job",
                reason="weekly_research_limit",
            )
        if not service.store.claim(claim_key, claim):
            raise FinplanError.precondition(
                "this week already has a different reserved research job",
                reason="weekly_research_limit",
            )
        request["dry_run"] = False
        payload["job"] = client.submit(request)
        payload["budget_controls"] = {
            "weekly_usd": WEEKLY_CAP,
            "monthly_usd": MONTHLY_CAP,
            "project_usd": PROJECT_CAP,
            "project_reported_spend": budget["spent"],
            "billing_lag": True,
        }
    return service.issue(
        "research_run",
        payload,
        {
            "review_id": doc["analysis_id"],
            "week": service.now().strftime("%G-W%V"),
            "dry_run": dry,
        },
    )
