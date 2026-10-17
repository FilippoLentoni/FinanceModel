"""Read-only saved-book resolution and raw-close valuation for frozen inference."""
from __future__ import annotations

import math
from datetime import date

import numpy as np

from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import require_id
from finplan_model.core.outcome import require_valid
from finplan_model.sim.market import Bar, MarketData

from .inference import portfolio_state

DATASET_ID = "finance/equity-etf-daily/research-universe"


def resolve_inputs(service, body):
    """Resolve references only; all data still passes through SnapshotReader verification."""
    saved = None
    if "holdings" not in body:
        portfolio_id = body.get("portfolio_id")
        if portfolio_id is None:
            parameter = getattr(service.d, "research_plan_parameter", None)
            plan_id = parameter.read() if parameter is not None else None
            if not plan_id:
                raise FinplanError.precondition("the default research portfolio is not configured", reason="paper_portfolio_not_configured")
            require_id("plan_id", plan_id)
            plan = service.d.platform.get_plan(plan_id).get("plan") or {}
            if plan.get("plan_id") != plan_id:
                raise FinplanError.dependency_unavailable("platform returned a different plan", retryable=False, reason="paper_state_invalid")
            portfolio_id = plan.get("portfolio_id")
        require_id("portfolio_id", portfolio_id)
        saved = service.d.platform.get_portfolio_state(portfolio_id)
        require_valid(saved, "api/get-portfolio-state-response")
        if saved["portfolio_id"] != portfolio_id:
            raise FinplanError.dependency_unavailable("platform returned a different paper portfolio", retryable=False, reason="paper_state_invalid")
    snapshot_id = body.get("input_snapshot_id")
    if snapshot_id is None:
        record = service.d.platform.get_latest_snapshot(DATASET_ID).get("snapshot") or {}
        if record.get("status") != "approved":
            raise FinplanError.precondition("no approved universe snapshot is available", reason="no_approved_snapshot")
        snapshot_id = record.get("input_snapshot_id")
    require_id("input_snapshot_id", snapshot_id)
    return snapshot_id, saved


def completed_date(service, body, market, instruments):
    if "as_of" in body:
        decision = date.fromisoformat(body["as_of"])
    else:
        candidates = [d for d in market.sessions if market.decision_time(d) <= service.now()
                      and any(market.bar(i, d) is not None for i in instruments)]
        if not candidates:
            raise FinplanError.precondition("no completed market session is available", reason="stale_or_incomplete_market_data")
        decision = candidates[-1]
    if decision > service.now().date():
        raise FinplanError.validation("recommendations cannot use a future decision date", pointer="/as_of")
    if decision not in market.sessions or market.decision_time(decision) > service.now():
        raise FinplanError.precondition("the decision session is unavailable or has not completed", reason="stale_or_incomplete_market_data")
    return decision


def raw_market(content, market):
    """Keep the approved payload's actual closes, not its adjusted actor feature basis."""
    bars = []
    for row in content.payload.get("observations", []):
        if row.get("kind", "completed_daily") != "completed_daily":
            continue
        d = date.fromisoformat(row["session_date"])
        feature_bar = market.bar(row["instrument_id"], d)
        if feature_bar is not None:
            bars.append(Bar.from_observation(row, feature_bar.available_at))
    return MarketData(market.sessions, bars, decision_times={d: market.decision_time(d) for d in market.sessions}, synthetic=market.synthetic)


def value_holdings(body, saved, raw, as_of, instruments):
    dates, prices = raw.view(as_of, instruments).price_matrix("close")
    if not dates or dates[-1] != as_of:
        raise FinplanError.precondition("aligned raw closing prices are unavailable", reason="insufficient_history")
    closes = prices[-1]
    if saved is None:
        holdings = body["holdings"]
        current, cash_weight, value, peak = portfolio_state(instruments, holdings)
        quantities = current * value / closes
        cash = cash_weight * value
        metadata = {"source": "supplied", "as_of": as_of.isoformat()}
    else:
        book = saved["paper_state"]
        require_valid(book, "paper-portfolio-state")
        book_date = date.fromisoformat(book["as_of"])
        if book_date > as_of:
            raise FinplanError.precondition("paper state is newer than the decision date", reason="paper_state_after_decision")
        rows = book["positions"]
        by_id = {r["instrument_id"]: float(r["quantity"]) for r in rows}
        if len(by_id) != len(rows) or set(by_id) - set(instruments):
            raise FinplanError.precondition("paper state contains duplicate or unsupported positions", reason="paper_state_invalid")
        quantities = np.array([by_id.get(i, 0.) for i in instruments])
        cash, stored_peak = float(book["cash_balance"]), float(book["high_watermark"])
        if not np.isfinite(quantities).all() or np.any(quantities < 0) or not math.isfinite(cash) or cash < 0 or not math.isfinite(stored_peak) or stored_peak <= 0:
            raise FinplanError.precondition("paper state contains invalid quantities, cash or high watermark", reason="paper_state_invalid")
        expected = [d for d in raw.sessions if book_date <= d <= as_of]
        visible = [d for d in dates if d >= book_date]
        if book_date not in raw.sessions or not expected or expected != visible:
            raise FinplanError.precondition("the snapshot does not cover every session since the saved paper state", reason="insufficient_history")
        path = prices[[k for k, d in enumerate(dates) if d >= book_date]] @ quantities + cash
        if not np.isfinite(path).all() or np.any(path <= 0):
            raise FinplanError.precondition("paper portfolio must have positive finite value", reason="paper_state_invalid")
        value, peak = float(path[-1]), max(stored_peak, float(path.max()))
        holdings = {"weights": [{"instrument_id": i, "weight": float(quantities[k] * closes[k] / value)} for k, i in enumerate(instruments)],
                    "cash_weight": cash / value, "portfolio_value": value, "high_watermark": peak}
        metadata = {"source": "saved_paper", "portfolio_id": saved["portfolio_id"], "revision": saved["revision"], "as_of": book["as_of"]}
    metadata.update({"valuation_as_of": as_of.isoformat(), "base_currency": "USD", "portfolio_value": value, "high_watermark": peak, "current_cash": cash,
                     "positions": [{"instrument_id": i, "quantity": float(quantities[k]), "reference_price": float(closes[k]), "price_as_of": as_of.isoformat()} for k, i in enumerate(instruments)]})
    return holdings, metadata


def with_quantities(rec, metadata):
    """Deterministic arithmetic only; recommendations never mutate the saved book."""
    positions = {p["instrument_id"]: p for p in metadata["positions"]}
    weights = {p["instrument_id"]: p["weight"] for p in rec["target_weights"]}
    for decision in rec["decisions"]:
        p = positions[decision["instrument_id"]]
        target = weights[decision["instrument_id"]] * metadata["portfolio_value"] / p["reference_price"]
        decision.update({"current_quantity": p["quantity"], "target_quantity": target, "delta_quantity": target - p["quantity"],
                         "reference_price": p["reference_price"], "price_as_of": p["price_as_of"]})
    rec["portfolio_state"] = metadata
    rec["limitations"].extend(["indicative_fractional_shares_before_execution_costs", "paper_book_requires_explicit_fills_and_corporate_action_updates"])
    return rec
