"""Independent beta Lambda dispatcher for traditional optimization and research evidence."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from finplan_model.classical.research import market_events, review, run_review
from finplan_model.classical.recursive import run_recursive_improvement, scheduled_recursive_review
from finplan_model.classical.service import ClassicalService
from finplan_model.classical.storage import S3Store, public, reference
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.clock import SystemClock
from finplan_model.core.config import load_config
from finplan_model.core.errors import ErrorCode, FinplanError, as_finplan_error
from finplan_model.core.outcome import require_valid
from finplan_model.decision_analysis import compare_decisions, evaluate_decision, explain_decision

log = logging.getLogger("finplan_model.classical")
log.setLevel(logging.INFO)
_SERVICE = None
OPS = (
    "recommend_classical_portfolio",
    "explain_classical_recommendation",
    "compare_classical_plans",
    "evaluate_classical_performance",
    "get_classical_analysis",
    "list_classical_analyses",
    "research_portfolio_models",
    "run_portfolio_research",
    "submit_portfolio_feedback",
    "research_market_events",
    "explain_portfolio_decision",
    "compare_portfolio_decisions",
    "evaluate_portfolio_decision",
    "run_recursive_improvement",
)


class JobApi:
    def __init__(self, endpoint, session, region):
        self.endpoint, self.session, self.region = endpoint.rstrip("/"), session, region

    def call(self, method, path, body=None, query=None):
        import urllib.error
        import urllib.parse
        import urllib.request

        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        url = self.endpoint + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        req = AWSRequest(method=method, url=url, data=data, headers=headers)
        SigV4Auth(self.session.get_credentials(), "execute-api", self.region).add_auth(
            req
        )
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url, data=data, headers=dict(req.headers), method=method
                ),
                timeout=20,
            ) as response:
                out = json.loads(response.read(1_000_000))
        except urllib.error.HTTPError as exc:
            raw = json.loads(exc.read(16_000))
            err = raw.get("error", raw)
            raise FinplanError(
                err.get("code", "DEPENDENCY_UNAVAILABLE"),
                str(err.get("message", "sandbox API request failed"))[:400],
                details=err.get("details") or {},
            ) from None
        return out

    def submit(self, request):
        return self.call("POST", "/v1/jobs", body=request)

    def list_jobs(self):
        query, out = {"page_size": 100}, []
        for _ in range(5):
            page = self.call("GET", "/v1/jobs", query=query)
            out.extend(page["jobs"])
            if not page.get("next_token"):
                return out
            query["next_token"] = page["next_token"]
        raise FinplanError.precondition(
            "sandbox history exceeds the bounded controller audit",
            reason="research_job_history_bound",
        )

    def result(self, run_id):
        return self.call("GET", f"/v1/jobs/{run_id}/result")


def build_service():
    import boto3

    from finplan_model.control.production_strategy import SsmStrategyParameter
    from finplan_model.core.aws_clients import s3_client
    from finplan_model.core.artifacts import S3ArtifactStore
    from finplan_model.core.platform import HttpPlatformClient

    env = os.environ["FINPLAN_ENVIRONMENT"]
    cfg = load_config(env, Path(os.environ["FINPLAN_CONFIG_DIR"]))
    session = boto3.session.Session(region_name=cfg.region)
    ssm = session.client("ssm")
    bucket = ssm.get_parameter(Name=cfg.ssm_name("config", "research-storage-ref"))[
        "Parameter"
    ]["Value"]
    endpoint = ssm.get_parameter(Name=cfg.ssm["plan_endpoint"])["Parameter"]["Value"]
    job_endpoint = ssm.get_parameter(Name=cfg.ssm_name("api", "job-endpoint"))[
        "Parameter"
    ]["Value"]

    def project_budget():
        from finplan_model.classical.research import BUDGET_NAME

        account = session.client("sts").get_caller_identity()["Account"]
        budget = session.client("budgets", region_name="us-east-1").describe_budget(
            AccountId=account, BudgetName=BUDGET_NAME
        )["Budget"]
        return {
            "spent": float(budget["CalculatedSpend"]["ActualSpend"]["Amount"]),
            "limit": float(budget["BudgetLimit"]["Amount"]),
        }

    def benchmark_capabilities():
        from finplan_model.benchmarks.qwen import MODEL_ID, REVISION
        from finplan_model.benchmarks.weights import PREFIX
        from finplan_model.benchmarks.jev import SECRET
        from finplan_model.control.settings import SsmSettings
        settings = SsmSettings(ssm, cfg)
        qwen = {"configured": settings.job_definition("swarm_mode_a") is not None, "model_id": MODEL_ID, "revision": REVISION,
            "compute": "run_scoped_network_isolated_gpu_training", "gpu_approval_required": True, "weights_ready": False}
        try:
            staged = json.loads(s3_client(cfg.region, session=session).get_object(Bucket=bucket, Key=PREFIX + "STAGED.json")["Body"].read())
            qwen["weights_ready"] = staged.get("model_id") == MODEL_ID and staged.get("revision") == REVISION
        except Exception:
            qwen["reason"] = "exact_weights_not_staged_or_status_unavailable"
        jev = {"configured": settings.job_definition("jev_backtest") is not None, "model_alias": "jev-latest", "external_vendor_approval_required": True, "secret_configured": False}
        try:
            session.client("secretsmanager").describe_secret(SecretId=SECRET)
            jev["secret_configured"] = True
        except Exception:
            jev["reason"] = "vendor_secret_not_configured_or_status_unavailable"
        return {"swarm_mode_a": qwen, "jev_backtest": jev, "recursive_evaluate": {"configured": settings.job_definition("recursive_evaluate") is not None, "released_profiles": ["recursive_ppo_features", "recursive_ppo_turnover", "recursive_ppo_horizon"]}}

    deps = SimpleNamespace(
        research_plan_parameter=SsmStrategyParameter(
            ssm, f"/finplan/{env}/financialplanning/config/research-plan-ref"
        ),
        platform=HttpPlatformClient(
            endpoint, region=cfg.region, credentials=session.get_credentials()
        ),
        job_api=JobApi(job_endpoint, session, cfg.region),
        project_budget=project_budget,
        benchmark_capabilities=benchmark_capabilities,
        artifacts=S3ArtifactStore(s3_client(cfg.region, session=session), bucket),
    )
    return ClassicalService(
        env=env,
        deps=deps,
        now=SystemClock().now,
        store=S3Store(s3_client(cfg.region, session=session), bucket),
    )


def feedback_audit(headers):
    """Record bounded transport identity from the IAM-trusted tool hop, never body claims."""
    headers = (
        {str(k).lower(): v for k, v in headers.items()}
        if isinstance(headers, dict)
        else {}
    )
    raw = headers.get("x-finplan-caller")
    if not isinstance(raw, str) or len(raw) > 4096:
        return {
            "caller_status": "not_available",
            "reason": "trusted_caller_header_not_supplied",
        }
    try:
        caller = require_valid(json.loads(raw), "caller")
    except (ValueError, FinplanError):
        raise FinplanError.validation(
            "invalid trusted caller metadata", pointer="/headers/X-Finplan-Caller"
        ) from None
    if len(caller.get("roles", [])) > 16:
        raise FinplanError.validation(
            "caller metadata exceeds role bound", pointer="/headers/X-Finplan-Caller"
        )
    correlation = headers.get("x-correlation-id")
    if correlation is not None and correlation != caller["correlation_id"]:
        raise FinplanError.validation(
            "caller and transport correlation identifiers disagree",
            pointer="/headers/X-Correlation-Id",
        )
    return {
        "caller_status": "available",
        "caller": caller,
        "identity_scope": "trusted tool invocation identity; Gateway environment principal where end-user claims are unavailable",
        "authority": "audit_only_never_authorization",
    }


def handle(event, service):
    if (
        not isinstance(event, dict)
        or event.get("environment") != service.env
        or service.env != "beta"
    ):
        raise FinplanError.validation(
            "classical serving request belongs to another environment",
            pointer="/environment",
        )
    if set(event) - {"environment", "operation", "request", "headers"}:
        raise FinplanError.validation("unknown classical serving field", pointer="")
    op = event.get("operation")
    if op not in OPS:
        raise FinplanError.validation(
            "unknown classical operation", pointer="/operation"
        )
    body = require_valid(
        event.get("request", {}), "tools/" + op.replace("_", "-") + "-request"
    )
    idem_key = (
        body.get("idempotency_key")
        if op in ("recommend_classical_portfolio", "submit_portfolio_feedback", "run_recursive_improvement")
        else None
    )
    idem_claim = (
        "idempotency/"
        + op
        + "/"
        + sha256_checksum(str(idem_key).encode()).split(":")[1]
        if idem_key
        else None
    )
    request_fingerprint = (
        sha256_checksum(canonical_json_bytes(body)) if idem_claim else None
    )
    if idem_claim:
        prior = service.store.get_claim(idem_claim)
        if prior:
            if prior["request_fingerprint"] != request_fingerprint:
                raise FinplanError(
                    ErrorCode.IDEMPOTENCY_KEY_REUSED,
                    "idempotency key already belongs to a different classical request",
                )
            return require_valid(
                public(service.store.get(prior["analysis_id"])),
                "tools/" + op.replace("_", "-") + "-response",
            )
    if op == "recommend_classical_portfolio":
        result = service.recommend(body)
    elif op == "explain_classical_recommendation":
        result = service.explanation(body)
    elif op == "compare_classical_plans":
        result = service.compare(body)
    elif op == "evaluate_classical_performance":
        result = service.performance(body)
    elif op == "get_classical_analysis":
        result = public(service.store.get(body["analysis_id"]))
    elif op == "list_classical_analyses":
        rows = service.store.list(
            portfolio_id=body.get("portfolio_id"),
            kind=body.get("kind"),
            limit=body.get("limit", 20),
        )
        summaries = [
            {
                **{
                    k: v
                    for k, v in public(d).items()
                    if k
                    in (
                        "analysis_id",
                        "analysis_kind",
                        "created_at",
                        "analysis_ref",
                        "summary",
                        "portfolio_id",
                    )
                },
                **(
                    {
                        "as_of": d["recommendation"]["as_of"],
                        "algorithm": d["recommendation"]["strategy"],
                    }
                    if "recommendation" in d
                    else {}
                ),
            }
            for d in rows
        ]
        result = {"analyses": summaries, "contract_version": "1.5.0", "synthetic": True}
    elif op == "research_portfolio_models":
        result = review(service, body)
    elif op == "run_portfolio_research":
        result = run_review(service, body)
    elif op == "research_market_events":
        result = market_events(service, body)
    elif op == "explain_portfolio_decision":
        result = explain_decision(service, body)
    elif op == "compare_portfolio_decisions":
        result = compare_decisions(service, body)
    elif op == "evaluate_portfolio_decision":
        result = evaluate_decision(service, body)
    elif op == "run_recursive_improvement":
        result = run_recursive_improvement(service, body)
    else:
        doc = service.store.get(body["analysis_id"])
        result = service.issue(
            "feedback",
            {
                "summary": "User feedback retained as untrusted research evidence; no strategy changed",
                "source_analysis_id": doc["analysis_id"],
                "source_analysis_ref": reference(doc),
                "text": body["text"],
                "trust": "user_feedback_not_executable_instructions",
                "audit": feedback_audit(event.get("headers")),
            },
            {
                "source_ref": reference(doc),
                "text": body["text"],
                "idempotency_key": body["idempotency_key"],
            },
            portfolio_id=doc.get("portfolio_id"),
        )
    result = require_valid(result, "tools/" + op.replace("_", "-") + "-response")
    if idem_claim:
        claim = {
            "request_fingerprint": request_fingerprint,
            "analysis_id": result["analysis_id"],
        }
        if not service.store.claim(idem_claim, claim):
            prior = service.store.get_claim(idem_claim)
            if prior["request_fingerprint"] != request_fingerprint:
                raise FinplanError(
                    ErrorCode.IDEMPOTENCY_KEY_REUSED,
                    "idempotency key already belongs to a different classical request",
                )
            return require_valid(
                public(service.store.get(prior["analysis_id"])),
                "tools/" + op.replace("_", "-") + "-response",
            )
    return result


def handler(event, context=None):
    global _SERVICE
    headers = event.get("headers", {}) if isinstance(event, dict) else {}
    cid = headers.get("X-Correlation-Id") if isinstance(headers, dict) else None
    try:
        if _SERVICE is None:
            _SERVICE = build_service()
        result = handle(event, _SERVICE)
        status = "OK"
    except Exception as exc:  # noqa: BLE001 - boundary converts every failure to a safe contract error
        err = as_finplan_error(exc)
        result, status = err.to_envelope(cid), err.code
    log.info(
        json.dumps(
            {
                "event": "classical_analysis",
                "operation": event.get("operation")
                if isinstance(event, dict)
                else None,
                "analysis_id": result.get("analysis_id"),
                "outcome": status,
                "correlation_id": cid,
            }
        )
    )
    return result


def weekly_handler(event, context=None):
    """Only the deployed Scheduler role can invoke this entry; event body carries no model code."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = build_service()
    service = _SERVICE
    if service.env != "beta":
        raise FinplanError.precondition(
            "weekly research is beta only", reason="research_disabled"
        )
    resumed = scheduled_recursive_review(service)
    if resumed is not None:
        return resumed
    # Refresh the proposal context from approved completed data; this issues no trade.
    plan = service.recommend({})
    # Reuse an immutable week claim even if Scheduler's event identity changes on a retry.
    result = review(service, {}, plan_context_id=plan["analysis_id"])
    try:
        return run_review(
            service,
            {"review_id": result["analysis_id"], "dry_run": False, "scheduled": True},
        )
    except Exception as exc:  # noqa: BLE001 - boundary converts every failure to a safe contract error
        err = as_finplan_error(exc)
        return service.issue(
            "research_run",
            {
                "summary": "Weekly review retained; sandbox launch skipped by its safety/budget gate",
                "review_id": result["analysis_id"],
                "dry_run": False,
                "status": "skipped",
                "error": err.to_envelope(),
                "activation": "proposal_only",
            },
            {
                "week": service.now().strftime("%G-W%V"),
                "review_id": result["analysis_id"],
                "outcome": err.code,
            },
        )
