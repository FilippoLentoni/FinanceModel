"""Pinned, point-in-time classical plans and three levels of numerical evidence."""

from __future__ import annotations

import copy
from datetime import date

import numpy as np

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.clock import utc_iso
from finplan_model.core.errors import FinplanError
from finplan_model.jobs.market_loader import load_market
from finplan_model.portfolio_decisions import persist_proposal
from finplan_model.rl.serving_context import (
    completed_date,
    raw_market,
    resolve_inputs,
    value_holdings,
    with_quantities,
)
from finplan_model.strategies.estimators import (
    estimate_covariance,
    estimate_mean,
    pit_returns,
)

from .math import exact_shapley, explain, settings, solve
from .storage import analysis_id, public, reference


class ClassicalService:
    def __init__(self, *, env, deps, now, store):
        self.env, self.d, self.now, self.store = env, deps, now, store

    def issue(self, kind, payload, identity, *, portfolio_id=None, solve_inputs=None):
        fingerprint = sha256_checksum(canonical_json_bytes(identity))
        doc = {
            "analysis_id": analysis_id({"kind": kind, "identity": identity}),
            "analysis_kind": kind,
            "created_at": utc_iso(self.now()),
            "request_fingerprint": fingerprint,
            **payload,
        }
        if portfolio_id:
            doc["portfolio_id"] = portfolio_id
        if solve_inputs is not None:
            doc["solve_inputs"] = solve_inputs
        return public(self.store.put(doc))

    def recommend(self, body):
        algorithm = body.get("algorithm", "min_variance")
        if algorithm not in ("min_variance", "mean_variance", "cvar"):
            raise FinplanError.validation(
                "unsupported classical algorithm", pointer="/algorithm"
            )
        cfg = settings(body.get("settings"))
        snap, saved = resolve_inputs(self, body)
        market, content = load_market(self.d.platform, snap)
        instruments = (
            sorted(r["instrument_id"] for r in saved["paper_state"]["positions"])
            if saved
            else sorted(market.instruments)
        )
        if not 1 <= len(instruments) <= 5:
            raise FinplanError.precondition(
                "classical beta attribution supports one to five instruments",
                reason="classical_universe_bound",
            )
        as_of = completed_date(self, body, market, instruments)
        view = market.view(as_of, instruments)
        returns = pit_returns(view, cfg["lookback_days"])
        if (
            len(returns) < cfg["lookback_days"]
            or returns.shape[1] != len(instruments)
            or not np.isfinite(returns).all()
        ):
            raise FinplanError.precondition(
                "approved snapshot lacks the requested aligned history",
                reason="insufficient_history",
            )
        holdings, metadata = value_holdings(
            body, saved, raw_market(content, market), as_of, instruments
        )
        from pathlib import Path

        import scipy

        from . import math as solver_module

        current_weights = {row["instrument_id"]: row["weight"] for row in holdings["weights"]}
        inputs = {
            "implementation": {
                "version": "finplan-classical/1",
                "solver_source_checksum": sha256_checksum(
                    Path(solver_module.__file__).read_bytes()
                ),
                "scipy_version": scipy.__version__,
            },
            "algorithm": algorithm,
            "instruments": instruments,
            "settings": cfg,
            "current_weights": [current_weights.get(instrument, 0.) for instrument in instruments],
            "expected_returns": estimate_mean(returns, "historical_mean").tolist(),
            "covariance": estimate_covariance(returns, "ledoit_wolf").tolist(),
            "scenarios": returns.tolist(),
            "as_of": as_of.isoformat(),
            "decision_time": market.decision_time(as_of).isoformat(),
            "input_snapshot_id": snap,
            "snapshot_checksum": content.snapshot.manifest_checksum,
            "estimation": {
                "mean": "historical_mean",
                "covariance": "ledoit_wolf",
                "return_basis": "adjusted_completed_close",
                "observations": len(returns),
            },
        }
        solution = solve(inputs)
        if solution["status"] != "optimal":
            raise FinplanError.precondition(
                "classical problem has no certified solution", reason=solution["reason"]
            )
        weights = solution["weights"]
        deltas = np.asarray(weights) - np.asarray(inputs["current_weights"])
        rec = {
            "mode": "advisory_paper",
            "strategy": algorithm,
            "configuration_checksum": sha256_checksum(canonical_json_bytes(cfg)),
            "settings": cfg,
            "target_weights": [
                {"instrument_id": i, "weight": weights[k]}
                for k, i in enumerate(instruments)
            ],
            "cash_weight": solution["cash_weight"],
            "decisions": [
                {
                    "instrument_id": i,
                    "action": "hold"
                    if abs(deltas[k]) < 1e-6
                    else "buy"
                    if deltas[k] > 0
                    else "sell",
                    "delta_weight": float(deltas[k]),
                    "indicative_notional": float(deltas[k])
                    * holdings["portfolio_value"],
                }
                for k, i in enumerate(instruments)
            ],
            "as_of": as_of.isoformat(),
            "input_snapshot_id": snap,
            "snapshot_checksum": content.snapshot.manifest_checksum,
            "solution_status": "optimal",
            "estimated_turnover": solution["metrics"]["turnover"],
            "constraint_outcome": {
                "action": "accepted",
                "violations": [],
                "fixed_cash_weight": cfg["cash_weight"],
                "max_weight": cfg["max_weight"],
            },
            "forecast": {
                "status": "not_available",
                "reason": "historical objective estimates are not a calibrated future return forecast",
            },
            "diagnostics": {
                **solution["metrics"],
                "solver": solution["solver"],
                "settings": cfg,
            },
            "decision_timing": "after_completed_close_for_next_session",
            "limitations": [
                "research_only",
                "no_trade_execution",
                "historical_estimates_not_calibrated_forecasts",
                "horizon_independent_stationary_scaling",
                "hindsight_selected_universe",
            ],
        }
        with_quantities(rec, metadata)
        explanation = explain(inputs, solution)
        payload = {
            "summary": f"{algorithm} paper allocation on {as_of}; exact model attribution against unchanged holdings",
            "recommendation": rec,
            "explanation": explanation,
            "sources": [],
            "bias_disclosures": content.snapshot.record.get("bias_disclosures", []),
        }
        identity = {
            "inputs": inputs,
            "portfolio_state": metadata,
            "idempotency_key": body.get("idempotency_key"),
        }
        aid = analysis_id({"kind": "recommendation", "identity": identity})
        decision_id = persist_proposal(
            self, rec, saved,
            {"implementation": inputs["implementation"], "settings": cfg,
             "decision_time": inputs["decision_time"],
             "snapshot_checksum": content.snapshot.manifest_checksum,
             "paper_execution_cost_bps": 2.0},
            source_analysis_id=aid, idempotency_key=body.get("idempotency_key"),
        )
        if decision_id:
            payload["decision_id"] = decision_id
        return self.issue(
            "recommendation",
            payload,
            identity,
            portfolio_id=saved["portfolio_id"] if saved else None,
            solve_inputs=inputs,
        )

    def plan(self, aid):
        doc = self.store.get(aid)
        if doc["analysis_kind"] != "recommendation":
            raise FinplanError.validation(
                "analysis must identify an issued recommendation",
                pointer="/analysis_id",
            )
        return doc

    def reproduce(self, doc):
        """Certify the frozen implementation and issued endpoint before attribution."""
        from pathlib import Path

        import scipy

        from . import math as solver_module

        implementation = doc["solve_inputs"]["implementation"]
        if implementation != {
            "version": "finplan-classical/1",
            "solver_source_checksum": sha256_checksum(
                Path(solver_module.__file__).read_bytes()
            ),
            "scipy_version": scipy.__version__,
        }:
            raise FinplanError.precondition(
                "issued plan solver implementation differs; original evidence remains retrievable",
                reason="classical_solver_version_mismatch",
            )
        reproduced = solve(doc["solve_inputs"])
        expected = [r["weight"] for r in doc["recommendation"]["target_weights"]]
        if reproduced["status"] != "optimal" or not np.allclose(
            reproduced["weights"], expected, atol=1e-8, rtol=0
        ):
            raise FinplanError.precondition(
                "immutable solve inputs do not reproduce issued allocation",
                reason="classical_plan_reproduction_mismatch",
            )
        return reproduced, {
            "status": "verified",
            "weight_tolerance": 1e-8,
            "maximum_weight_error": float(
                np.max(np.abs(np.array(reproduced["weights"]) - expected))
            ),
            "implementation": implementation,
        }

    def explanation(self, body):
        doc = self.plan(body["analysis_id"])
        reproduced, reproduction = self.reproduce(doc)
        explanation = explain(doc["solve_inputs"], reproduced)
        explanation["reproduction"] = reproduction
        instrument = body.get("instrument_id")
        if instrument:
            if instrument not in doc["solve_inputs"]["instruments"]:
                raise FinplanError.validation(
                    "instrument is outside this plan", pointer="/instrument_id"
                )
            explanation["force_keep"] = [
                r for r in explanation["force_keep"] if r["instrument_id"] == instrument
            ]
            explanation["risk_contributions"] = [
                r
                for r in explanation["risk_contributions"]
                if r["instrument_id"] == instrument
            ]
        return self.issue(
            "explanation",
            {
                "summary": "Reproduced model attribution from immutable issued-plan solve inputs",
                "source_analysis_id": doc["analysis_id"],
                "source_analysis_ref": reference(doc),
                "recommendation": doc["recommendation"],
                "explanation": explanation,
            },
            {"source_ref": reference(doc), "instrument_id": instrument},
            portfolio_id=doc.get("portfolio_id"),
        )

    def compare(self, body):
        a, b = [
            self.plan(body[k]) for k in ("previous_analysis_id", "current_analysis_id")
        ]
        x, y = a["solve_inputs"], b["solve_inputs"]
        if (
            x["instruments"] != y["instruments"]
            or x["algorithm"] != y["algorithm"]
            or x["settings"]["horizon_sessions"] != y["settings"]["horizon_sessions"]
            or a.get("portfolio_id") != b.get("portfolio_id")
        ):
            raise FinplanError.precondition(
                "compare requires the same algorithm, universe, portfolio and objective horizon",
                reason="classical_plans_not_aligned",
            )
        if x["as_of"] > y["as_of"]:
            raise FinplanError.validation(
                "previous plan date follows current plan",
                pointer="/previous_analysis_id",
            )
        previous_result, previous_reproduction = self.reproduce(a)
        current_result, current_reproduction = self.reproduce(b)
        reproduction = {
            "status": "verified",
            "weight_tolerance": 1e-8,
            "previous": previous_reproduction,
            "current": current_reproduction,
        }
        names = ["expected_returns", "risk_inputs", "portfolio_state", "configuration"]
        keys = [
            ["expected_returns"],
            ["covariance", "scenarios"],
            ["current_weights"],
            ["settings"],
        ]
        coalition_status = []

        def game(mask):
            mixed = copy.deepcopy(x)
            for i, group in enumerate(keys):
                if mask & (1 << i):
                    for key in group:
                        mixed[key] = copy.deepcopy(y[key])
            result = (
                previous_result
                if mask == 0
                else current_result
                if mask == 15
                else solve(mixed)
            )
            coalition_status.append({"coalition": mask, "status": result["status"]})
            if result["status"] != "optimal":
                raise FinplanError.precondition(
                    "a Shapley hybrid problem is infeasible; attribution cannot be certified",
                    reason="infeasible_shapley_hybrid",
                )
            return result["weights"]

        coalition_values = {}
        for mask in range(16):
            try:
                coalition_values[mask] = game(mask)
            except FinplanError:
                pass
        if len(coalition_values) != 16:
            return self.issue(
                "comparison",
                {
                    "summary": "Plan-change Shapley unavailable: a required hybrid optimization is infeasible",
                    "status": "not_available",
                    "reason": "infeasible_shapley_hybrid",
                    "previous_analysis_id": a["analysis_id"],
                    "current_analysis_id": b["analysis_id"],
                    "previous_analysis_ref": reference(a),
                    "current_analysis_ref": reference(b),
                    "coalition_status": coalition_status,
                    "reproduction": reproduction,
                    "single_switch_evidence": [
                        {
                            "group": names[i],
                            "weights": coalition_values.get(1 << i),
                            "status": "available"
                            if 1 << i in coalition_values
                            else "not_available",
                        }
                        for i in range(4)
                    ],
                },
                {"previous": reference(a), "current": reference(b)},
                portfolio_id=a.get("portfolio_id"),
            )
        shapley = exact_shapley(names, lambda mask: coalition_values[mask])
        instruments = x["instruments"]
        changes = []
        for k, name in enumerate(instruments):
            da, db = (
                a["recommendation"]["decisions"][k],
                b["recommendation"]["decisions"][k],
            )
            changes.append(
                {
                    "instrument_id": name,
                    "previous_action": da["action"],
                    "current_action": db["action"],
                    "previous_target_weight": a["recommendation"]["target_weights"][k][
                        "weight"
                    ],
                    "current_target_weight": b["recommendation"]["target_weights"][k][
                        "weight"
                    ],
                    "target_weight_change": shapley["final"][k]
                    - shapley["baseline"][k],
                    "current_holding_weight_change": y["current_weights"][k]
                    - x["current_weights"][k],
                    "attribution": [
                        {"group": r["group"], "value": r["value"][k]}
                        for r in shapley["contributions"]
                    ],
                    "trade_delta_weight_change": db["delta_weight"]
                    - da["delta_weight"],
                    "trade_attribution": [
                        {
                            "group": r["group"],
                            "value": r["value"][k]
                            - (
                                y["current_weights"][k] - x["current_weights"][k]
                                if r["group"] == "portfolio_state"
                                else 0.0
                            ),
                        }
                        for r in shapley["contributions"]
                    ],
                }
            )
        shapley.update(
            {
                "game": "resolve the same convex optimizer while swapping four persisted input groups",
                "output_units": "target portfolio weight",
                "coalition_status": coalition_status,
                "risk_group": "covariance and historical scenario distribution; interactions allocated equally by Shapley",
                "causality": "model-input counterfactual attribution, not causal market explanation",
            }
        )
        return self.issue(
            "comparison",
            {
                "summary": "Exact four-group attribution of target allocation changes between immutable issued plans",
                "previous_analysis_id": a["analysis_id"],
                "current_analysis_id": b["analysis_id"],
                "previous_analysis_ref": reference(a),
                "current_analysis_ref": reference(b),
                "alignment": {
                    "algorithm": x["algorithm"],
                    "instruments": instruments,
                    "horizon_sessions": x["settings"]["horizon_sessions"],
                    "previous_as_of": x["as_of"],
                    "current_as_of": y["as_of"],
                    "implementation": x["implementation"],
                },
                "reproduction": reproduction,
                "status": "available",
                "shapley": shapley,
                "changes": changes,
            },
            {"previous": reference(a), "current": reference(b)},
            portfolio_id=a.get("portfolio_id"),
        )

    def performance(self, body):
        doc = self.plan(body["analysis_id"])
        rec, inputs = doc["recommendation"], doc["solve_inputs"]
        snapshot_id = body.get("observed_snapshot_id")
        if not snapshot_id:
            snapshot_id = self.d.platform.get_latest_snapshot(
                "finance/equity-etf-daily/research-universe"
            )["snapshot"]["input_snapshot_id"]
        market, content = load_market(self.d.platform, snapshot_id)
        end = completed_date(
            self,
            {"as_of": body["end_date"]} if "end_date" in body else {},
            market,
            inputs["instruments"],
        )
        start = date.fromisoformat(rec["as_of"])
        if end < start:
            raise FinplanError.validation(
                "observed end precedes issued plan", pointer="/end_date"
            )
        dates, px = market.view(end, inputs["instruments"]).price_matrix("close")
        if start not in dates or end not in dates:
            raise FinplanError.precondition(
                "observations do not cover plan and end dates",
                reason="performance_window_not_covered",
            )
        ai, bi = dates.index(start), dates.index(end)
        if ai == bi:
            return self.issue(
                "performance",
                {
                    "summary": "No completed forward sessions since this plan; trend is not available",
                    "source_analysis_id": doc["analysis_id"],
                    "source_analysis_ref": reference(doc),
                    "status": "not_available",
                    "trend": "not_available",
                    "reason": "no_forward_observations",
                    "actual_source": "saved_paper"
                    if doc.get("portfolio_id")
                    else "hypothetical",
                    "real_execution": {"status": "not_available"},
                },
                {
                    "source": reference(doc),
                    "snapshot": snapshot_id,
                    "end": end.isoformat(),
                },
                portfolio_id=doc.get("portfolio_id"),
            )
        # Strategy return uses adjusted prices to include provider corporate-action adjustments.
        relative = px[bi] / px[ai] - 1
        nav = rec["portfolio_state"]["portfolio_value"]
        weights = np.asarray([w["weight"] for w in rec["target_weights"]])
        old = np.asarray(inputs["current_weights"])
        planned_pnl, held_pnl = (
            nav * float(weights @ relative),
            nav * float(old @ relative),
        )
        contributions = [
            {
                "instrument_id": name,
                "planned_pnl": float(nav * weights[k] * relative[k]),
                "unexecuted_hold_pnl": float(nav * old[k] * relative[k]),
                "exposure_gap": float(nav * (old[k] - weights[k]) * relative[k]),
            }
            for k, name in enumerate(inputs["instruments"])
        ]
        if not doc.get("portfolio_id"):
            return self.issue(
                "performance",
                {
                    "summary": "Issued scenario allocation marked forward; observed account holdings unavailable",
                    "source_analysis_id": doc["analysis_id"],
                    "source_analysis_ref": reference(doc),
                    "status": "partial",
                    "trend": "not_available",
                    "issued_plan_trend": "green" if planned_pnl >= 0 else "red",
                    "trend_reason": "no observed holdings or executions are available for a supplied scenario",
                    "actual_source": "hypothetical",
                    "planned_allocation_hold": {
                        "start_value": nav,
                        "end_value": nav + planned_pnl,
                        "pnl": planned_pnl,
                        "return": planned_pnl / nav,
                    },
                    "observed_paper": {"status": "not_available"},
                    "real_execution": {"status": "not_available"},
                    "forecast": {"status": "not_available"},
                },
                {
                    "source": reference(doc),
                    "snapshot": snapshot_id,
                    "end": end.isoformat(),
                },
            )
        actual_state = self.d.platform.get_portfolio_state(doc["portfolio_id"])
        unchanged = actual_state["revision"] == rec["portfolio_state"]["revision"]
        _holdings, valuation = value_holdings(
            {}, actual_state, raw_market(content, market), end, inputs["instruments"]
        )
        observed_end = valuation["portfolio_value"]
        observed_pnl = observed_end - nav
        raw_gap = observed_pnl - held_pnl
        total_gap = observed_pnl - planned_pnl
        status = "available" if unchanged else "partial"
        payload = {
            "summary": "Paper portfolio versus immutable issued allocation; real execution and calibrated forecast unavailable",
            "source_analysis_id": doc["analysis_id"],
            "source_analysis_ref": reference(doc),
            "observed_snapshot_id": snapshot_id,
            "observed_snapshot_checksum": content.snapshot.manifest_checksum,
            "window": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "completed_forward_sessions": bi - ai,
                "planned_horizon_sessions": inputs["settings"]["horizon_sessions"],
                "partial_horizon": bi - ai < inputs["settings"]["horizon_sessions"],
            },
            "status": status,
            "actual_source": "saved_paper",
            "trend": "green" if observed_pnl >= 0 else "red",
            "observed_trend": "green" if observed_pnl >= 0 else "red",
            "issued_plan_trend": "green" if planned_pnl >= 0 else "red",
            "gap_trend": "green" if total_gap >= 0 else "red",
            "thresholds": {
                "observed_return_green_minimum": 0.0,
                "gap_green_minimum": 0.0,
                "rule": "nonnegative is green; negative is red",
            },
            "trend_definition": "sign of observed saved-paper PnL since issuance; separate issued allocation and implementation-gap trends; not forecast accuracy",
            "planned_allocation_hold": {
                "start_value": nav,
                "end_value": nav + planned_pnl,
                "return": planned_pnl / nav,
                "pnl": planned_pnl,
                "costs": "not_available",
            },
            "observed_paper": {
                "end_value": observed_end,
                "pnl": observed_pnl,
                "revision_unchanged": unchanged,
            },
            "gap": {
                "total": total_gap,
                "exposure": held_pnl - planned_pnl if unchanged else None,
                "raw_vs_adjusted_price_and_corporate_action_basis": raw_gap
                if unchanged
                else None,
                "unattributed": 0.0 if unchanged else total_gap,
                "reconciliation_residual": total_gap
                - (held_pnl - planned_pnl)
                - raw_gap
                if unchanged
                else 0.0,
                "tolerance": 1e-6,
            },
            "instrument_contributions": contributions,
            "forecast": {
                "status": "not_available",
                "reason": "historical in-sample objective estimates are not calibrated expectations",
            },
            "real_execution": {
                "status": "not_available",
                "reason": "no broker fills, fees or dated cash-flow ledger attached to this paper recommendation",
            },
            "whys": [
                {
                    "level": 1,
                    "answer": "Compare adjusted-price return from target allocation with unchanged paper holdings",
                },
                {
                    "level": 2,
                    "answer": "Per-instrument exposure contributions reconcile the gap when saved revision is unchanged",
                },
                {
                    "level": 3,
                    "status": "unresolved",
                    "answer": "News and causal market explanations require dated external evidence; correlation is not proof",
                },
            ],
            "feedback": [
                {
                    "proposal": "benchmark lower turnover and covariance lookback variants on fresh chronological windows",
                    "automatic_activation": False,
                }
            ],
            "limitations": [
                "paper_not_real_execution",
                "no_calibrated_return_forecast",
                "adjusted_return_vs_raw_valuation_basis_disclosed",
                "allocation_hold_not_daily_reoptimization",
            ],
        }
        if abs(payload["gap"]["reconciliation_residual"]) > 1e-6:
            raise FinplanError.precondition(
                "performance accounting does not reconcile",
                reason="performance_accounting_mismatch",
            )
        return self.issue(
            "performance",
            payload,
            {
                "source": reference(doc),
                "snapshot": snapshot_id,
                "end": end.isoformat(),
                "saved_revision": actual_state["revision"],
            },
            portfolio_id=doc.get("portfolio_id"),
        )
