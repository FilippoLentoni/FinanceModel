"""Persist proposed paper decisions without applying trades or changing holdings."""

from __future__ import annotations

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import require_valid


def persist_proposal(service, recommendation, saved, provenance, *, source_analysis_id=None, idempotency_key=None):
    if saved is None:
        # An explicitly supplied hypothetical book has no authoritative revision to accept.
        return None
    meta = recommendation["portfolio_state"]
    weights = {r["instrument_id"]: r["weight"] for r in recommendation["target_weights"]}
    weights["USD_CASH"] = recommendation["cash_weight"]
    algorithm = recommendation["strategy"]
    request = {
        "portfolio_id": saved["portfolio_id"],
        "algorithm_family": "reinforcement_learning" if algorithm in ("ppo", "sac") else "optimization",
        "algorithm": algorithm,
        "input_snapshot_id": recommendation["input_snapshot_id"],
        "portfolio_revision": saved["revision"],
        "recommendation": recommendation,
        "provenance": provenance,
        "execution": {
            "reference_date": recommendation["as_of"],
            "reference_prices": {r["instrument_id"]: r["reference_price"] for r in meta["positions"]},
            "target_weights": weights,
            "transaction_cost_bps": 2.0,
        },
    }
    if source_analysis_id:
        request["source_analysis_id"] = source_analysis_id
    identity = {"portfolio_id": saved["portfolio_id"], "proposal": request, "request_key": idempotency_key}
    request["idempotency_key"] = "model-proposal-" + sha256_checksum(canonical_json_bytes(identity)).split(":")[1]
    require_valid(request, "api/create-portfolio-decision-request")
    result = service.d.platform.create_portfolio_decision(saved["portfolio_id"], request)
    decision = result.get("decision") or {}
    if decision.get("portfolio_id") != saved["portfolio_id"] or not decision.get("decision_id"):
        raise FinplanError.dependency_unavailable("Platform returned an invalid persisted decision", retryable=False, reason="portfolio_decision_invalid")
    return decision["decision_id"]
