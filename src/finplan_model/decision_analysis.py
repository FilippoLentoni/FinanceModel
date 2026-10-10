"""Evidence-preserving explanations for issued PPO and optimization paper decisions."""

from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np

from finplan_model.classical.storage import reference
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.jobs.market_loader import load_market
from finplan_model.rl.serving_context import DATASET_ID, completed_date, raw_market


def portfolio_id(service, body):
    if body.get("portfolio_id"):
        return body["portfolio_id"]
    plan_id = service.d.research_plan_parameter.read()
    return service.d.platform.get_plan(plan_id)["plan"]["portfolio_id"]


def decision(service, body, key="decision_id"):
    pid = portfolio_id(service, body)
    doc = service.d.platform.get_portfolio_decision(pid, body[key])["decision"]
    if doc["portfolio_id"] != pid:
        raise FinplanError.precondition("decision belongs to another portfolio", reason="portfolio_decision_mismatch")
    return doc


def decision_ref(doc):
    return {"decision_id": doc["decision_id"], "checksum": doc.get("checksum") or sha256_checksum(canonical_json_bytes(doc)), "portfolio_revision": doc["portfolio_revision"], "input_snapshot_id": doc["input_snapshot_id"], "status": doc["status"]}


def model_identity(doc):
    """Exclude changing market observations and holdings from model-change detection."""
    provenance = doc["provenance"]
    return {
        "algorithm": doc["algorithm"],
        **{key: provenance[key] for key in (
            "implementation", "settings", "policy_artifact", "export_run_id",
            "policy_source_run_id", "configuration_id",
        ) if key in provenance},
    }


def replay(service, doc):
    """Re-evaluate only the checksum-verified frozen actor and recorded point-in-time inputs."""
    from finplan_model.rl.inference import implementation_identity, recommend

    provenance = doc.get("provenance", {})
    inputs, artifact = provenance.get("policy_inputs"), provenance.get("policy_artifact")
    if not inputs or not artifact:
        return {"status": "not_available", "reason": "legacy_decision_has_no_frozen_policy_inputs"}
    if provenance.get("implementation") and provenance["implementation"] != implementation_identity():
        raise FinplanError.precondition("policy inference implementation differs from issued decision", reason="policy_implementation_mismatch")
    bundle = json.loads(service.d.artifacts.get(artifact))
    result = recommend(bundle, np.asarray(inputs["prices"]), inputs["holdings"])
    original = doc["recommendation"]
    expected = [r["weight"] for r in original["target_weights"]] + [original["cash_weight"]]
    actual = [r["weight"] for r in result["target_weights"]] + [result["cash_weight"]]
    if not np.allclose(actual, expected, atol=1e-8, rtol=0):
        raise FinplanError.precondition("frozen actor no longer reproduces issued recommendation", reason="policy_replay_mismatch")
    return {"status": "verified", "weight_tolerance": 1e-8, "maximum_weight_error": float(np.max(np.abs(np.asarray(actual) - expected))), "policy_artifact_checksum": artifact["checksum"], "diagnostics": result["diagnostics"]}


def explain_decision(service, body):
    doc = decision(service, body)
    rec = doc["recommendation"]
    instrument = body.get("instrument_id")
    trades = [r for r in rec["decisions"] if not instrument or r["instrument_id"] == instrument]
    if not trades:
        raise FinplanError.validation("instrument is outside this decision", pointer="/instrument_id")
    payload = {
        "summary": "Issued paper decision explained from immutable model evidence; approval and paper execution shown separately",
        "source_decision_id": doc["decision_id"], "source_decision_ref": decision_ref(doc),
        "recommendation": rec,
        "decision_status": doc["status"], "resolution": doc.get("resolution"),
        "explanation": {
            "trade_logic": [{**r, "reason": "target exposure is below recorded current exposure" if r["action"] == "sell" else "target exposure is above recorded current exposure" if r["action"] == "buy" else "target and current exposure are within the action tolerance"} for r in trades],
            "forecast": rec.get("forecast", {"status": "not_available"}),
            "causality": "Model diagnostics and input counterfactuals do not prove the cause of future market moves",
        },
    }
    if doc["algorithm_family"] == "optimization" and doc.get("source_analysis_id"):
        result = service.explanation({"analysis_id": doc["source_analysis_id"], **({"instrument_id": instrument} if instrument else {})})
        payload["explanation"]["optimizer"] = result["explanation"]
        payload["explanation"]["source_analysis_ref"] = result["analysis_ref"]
    elif doc["algorithm"] in ("ppo", "sac"):
        payload["explanation"]["policy_replay"] = replay(service, doc)
        payload["explanation"]["attribution"] = {"status": "not_available", "reason": "Frozen policy replay and actual actor diagnostics are available; causal or Shapley feature attribution is not implemented for PPO"}
    else:
        payload["explanation"]["diagnostics"] = rec.get("diagnostics", {})
    return service.issue("explanation", payload, {"decision": decision_ref(doc), "instrument_id": instrument}, portfolio_id=doc["portfolio_id"])


def compare_decisions(service, body):
    a, b = [decision(service, body, key) for key in ("previous_decision_id", "current_decision_id")]
    ar, br = a["recommendation"], b["recommendation"]
    if ar["as_of"] > br["as_of"]:
        raise FinplanError.validation("previous decision date follows current decision", pointer="/previous_decision_id")
    weights = [{r["instrument_id"]: r["weight"] for r in rec["target_weights"]} for rec in (ar, br)]
    trades = [{r["instrument_id"]: r for r in rec["decisions"]} for rec in (ar, br)]
    payload = {
        "summary": "Allocation and action changes between two immutable decisions; market, holdings and model changes distinguished",
        "previous_decision_id": a["decision_id"], "current_decision_id": b["decision_id"],
        "previous_decision_ref": decision_ref(a), "current_decision_ref": decision_ref(b),
        "status": "available",
        "alignment": {"previous_algorithm": a["algorithm"], "current_algorithm": b["algorithm"], "previous_as_of": ar["as_of"], "current_as_of": br["as_of"], "snapshot_changed": a["input_snapshot_id"] != b["input_snapshot_id"], "portfolio_revision_changed": a["portfolio_revision"] != b["portfolio_revision"], "model_provenance_changed": model_identity(a) != model_identity(b)},
        "changes": [{"instrument_id": name, "previous_action": trades[0].get(name, {}).get("action", "not_in_universe"), "current_action": trades[1].get(name, {}).get("action", "not_in_universe"), "previous_target_weight": weights[0].get(name, 0.), "current_target_weight": weights[1].get(name, 0.), "target_weight_change": weights[1].get(name, 0.) - weights[0].get(name, 0.), "previous_delta_quantity": trades[0].get(name, {}).get("delta_quantity"), "current_delta_quantity": trades[1].get(name, {}).get("delta_quantity")} for name in sorted(set(weights[0]) | set(weights[1]))],
        "attribution": {"status": "not_available", "reason": "Observed input/output differences are not causal feature attribution"},
    }
    if a.get("source_analysis_id") and b.get("source_analysis_id") and a["algorithm"] == b["algorithm"] and set(weights[0]) == set(weights[1]) and ar.get("settings", {}).get("horizon_sessions") == br.get("settings", {}).get("horizon_sessions"):
        result = service.compare({"previous_analysis_id": a["source_analysis_id"], "current_analysis_id": b["source_analysis_id"]})
        payload["attribution"] = {"status": result["status"], "source_analysis_ref": result["analysis_ref"], "shapley": result.get("shapley"), "changes": result.get("changes"), "reason": result.get("reason")}
    elif a["algorithm"] in ("ppo", "sac") and b["algorithm"] in ("ppo", "sac"):
        payload["policy_replay"] = {"previous": replay(service, a), "current": replay(service, b)}
        payload["input_differences"] = {"previous": a["provenance"].get("policy_inputs"), "current": b["provenance"].get("policy_inputs")}
    return service.issue("comparison", payload, {"previous": decision_ref(a), "current": decision_ref(b)}, portfolio_id=a["portfolio_id"])


def _history(service, pid):
    rows, token = [], None
    for _ in range(20):
        page = service.d.platform.get_portfolio_history(pid, limit=100, next_token=token)
        rows.extend(page["history"])
        token = page.get("next_token")
        if not token:
            return sorted(rows, key=lambda row: row["revision"])
    raise FinplanError.precondition("portfolio history exceeds bounded accounting window", reason="portfolio_history_bound")


def evaluate_decision(service, body):
    doc = decision(service, body)
    rec, pid = doc["recommendation"], doc["portfolio_id"]
    snapshot_id = body.get("observed_snapshot_id") or service.d.platform.get_latest_snapshot(DATASET_ID)["snapshot"]["input_snapshot_id"]
    market, content = load_market(service.d.platform, snapshot_id)
    instruments = [r["instrument_id"] for r in rec["target_weights"]]
    end = completed_date(service, {"as_of": body["end_date"]} if body.get("end_date") else {}, market, instruments)
    start = date.fromisoformat(rec["as_of"])
    if end < start:
        raise FinplanError.validation("observed end precedes issued decision", pointer="/end_date")
    raw = raw_market(content, market)
    dates, prices = raw.view(end, instruments).price_matrix("close")
    if start not in dates or end not in dates:
        raise FinplanError.precondition("approved prices do not cover decision and observation", reason="performance_window_not_covered")
    start_i, end_i = dates.index(start), dates.index(end)
    nav = rec["portfolio_state"]["portfolio_value"]
    target_weights = np.array([r["weight"] for r in rec["target_weights"]])
    returns = prices[end_i] / prices[start_i] - 1
    planned_pnl = float(nav * (target_weights @ returns))
    by_id = {r["instrument_id"]: r for r in rec["portfolio_state"]["positions"]}
    held = np.array([by_id[i]["quantity"] for i in instruments])
    held_pnl = float(held @ (prices[end_i] - prices[start_i]))
    payload = {
        "summary": "Recorded paper holdings versus issued allocation and unchanged-holdings benchmark; no calibrated expected-return claim",
        "source_decision_id": doc["decision_id"], "source_decision_ref": decision_ref(doc),
        "observed_snapshot_id": snapshot_id, "observed_snapshot_checksum": content.snapshot.manifest_checksum,
        "window": {"start": start.isoformat(), "end": end.isoformat(), "completed_forward_sessions": end_i - start_i},
        "status": "not_available", "trend": "not_available",
        "planned_allocation_hold": {"start_value": nav, "end_value": nav + planned_pnl, "pnl": planned_pnl, "return": planned_pnl / nav, "basis": "raw_close_price_return_before_costs; hypothetical hold, not an expected-return forecast"},
        "unchanged_holdings_benchmark": {"pnl": held_pnl, "return": held_pnl / nav},
        "forecast": {"status": "not_available", "reason": "No calibrated return distribution or expectation was issued"},
        "real_execution": {"status": "not_available", "reason": "paper fills only; no broker execution evidence"},
        "decision_status": doc["status"],
    }
    history = _history(service, pid)
    cutoff = market.decision_time(end)
    visible = [row for row in history if row["revision"] == doc["portfolio_revision"] or (row["revision"] > doc["portfolio_revision"] and datetime.fromisoformat(row["recorded_at"].replace("Z", "+00:00")) <= cutoff)]
    expected_revisions = list(range(doc["portfolio_revision"], visible[-1]["revision"] + 1)) if visible else []
    reason = None
    if end_i == start_i:
        reason = "no_forward_observations"
    elif not visible or [row["revision"] for row in visible] != expected_revisions:
        reason = "portfolio_history_incomplete"
    elif datetime.fromisoformat(visible[0]["recorded_at"].replace("Z", "+00:00")) > cutoff:
        reason = "baseline_not_recorded_by_observation_time"
    elif any(not row.get("decision_id") for row in visible[1:]):
        reason = "unaccounted_portfolio_edit_or_external_cash_flow"
    else:
        # A split or dividend requires dated accounting entries; refuse a misleading price-only 'actual'.
        actions = [r for r in content.payload.get("observations", []) if r["instrument_id"] in instruments and start.isoformat() < r["session_date"] <= end.isoformat() and (float(r.get("dividend", 0.) or 0.) != 0. or float(r.get("split_ratio", 1.) or 1.) != 1.)]
        if actions:
            reason = "corporate_action_accounting_not_recorded"
    resolutions, fees = [], 0.
    if reason is None:
        for row in visible[1:]:
            changed = service.d.platform.get_portfolio_decision(pid, row["decision_id"])["decision"]
            resolution = changed.get("resolution", {})
            if changed["status"] != "accepted" or resolution.get("after_revision") != row["revision"] or resolution.get("before_revision") != row["revision"] - 1 or not resolution.get("paper_execution") or resolution.get("recorded_at") != row["recorded_at"] or resolution.get("paper_state") != row["paper_state"]:
                reason = "paper_fill_ledger_incomplete"
                break
            fees += float(resolution.get("transaction_cost", 0.))
            resolutions.append({"decision_id": changed["decision_id"], "resolution": resolution})
    if reason:
        payload["reason"] = reason
        payload["observed_paper"] = {"status": "not_available", "reason": reason}
    else:
        latest = visible[-1]
        book = latest["paper_state"]
        quantities = {r["instrument_id"]: r["quantity"] for r in book["positions"]}
        if set(quantities) - set(instruments):
            raise FinplanError.precondition("observed portfolio contains instruments outside decision universe", reason="performance_universe_changed")
        end_value = float(sum(quantities.get(i, 0.) * prices[end_i, k] for k, i in enumerate(instruments)) + book["cash_balance"])
        observed_pnl = end_value - nav
        gap = observed_pnl - planned_pnl
        payload.update({
            "status": "available", "trend": "green" if observed_pnl >= 0 else "red",
            "gap_trend": "green" if gap >= 0 else "red",
            "observed_paper": {"status": "available", "end_value": end_value, "pnl": observed_pnl, "return": observed_pnl / nav, "revision": latest["revision"], "recorded_paper_costs": fees, "accounting": "complete contiguous recorded paper revisions; no external flows or corporate actions in window"},
            "gap": {"total": gap, "execution_costs": -fees, "allocation_and_execution_timing": gap + fees, "reconciliation_residual": 0., "tolerance": 1e-6},
            "paper_execution_evidence": resolutions,
            "instrument_contributions": [{"instrument_id": name, "raw_price_return": float(returns[k]), "issued_allocation_pnl": float(nav * target_weights[k] * returns[k]), "unchanged_holding_pnl": float(held[k] * (prices[end_i, k] - prices[start_i, k]))} for k, name in enumerate(instruments)],
            "whys": [{"level": 1, "answer": "Compare observed paper value with the recorded target-allocation hold benchmark"}, {"level": 2, "answer": "Recorded costs and allocation/execution timing reconcile the gap; full dated paper revisions and fills are linked"}, {"level": 3, "status": "unresolved", "answer": "A market-cause explanation needs dated external evidence; news is context, not causal proof"}],
        })
    return service.issue("performance", payload, {"decision": decision_ref(doc), "snapshot": snapshot_id, "end": end.isoformat(), "history": [{"revision": r["revision"], "checksum": r.get("checksum"), "paper_state": r["paper_state"]} for r in visible]}, portfolio_id=pid)
