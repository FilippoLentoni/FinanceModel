"""Bounded, objective-aware replay of an issued decision, never today's selected policy.

Every path starts from the issued book, decides after a completed close, and uses
one common next-session execution model. Descriptive counterfactuals are not forecasts.
"""
from __future__ import annotations

import copy
import json
from datetime import date, datetime

import numpy as np

from finplan_model.classical.math import DEFAULTS, cvar, settings
from finplan_model.classical.strategy import ClassicalStrategy
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.rl.inference import implementation_identity, recommend, recommend_baseline
from finplan_model.rl.spec import EnvSpec
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.engine import Simulator
from finplan_model.sim.market import HoldingsView
from finplan_model.sim.metrics import annualized_volatility, max_drawdown, period_returns, sharpe_ratio
from finplan_model.sim.strategy import TargetWeights
from finplan_model.strategies.estimators import estimate_covariance, estimate_mean, pit_returns

VERSION = "finplan-decision-horizon/1"
MAX_SESSIONS = 252


def evaluation_implementation():
    """Preserve replay/execution and estimator transforms, beyond actor weights."""
    from pathlib import Path
    from finplan_model.classical import strategy
    from finplan_model.sim import engine
    from finplan_model.strategies import estimators

    return {"version": VERSION, "source_checksums": {name: sha256_checksum(Path(path).read_bytes()) for name, path in (
        ("horizon_evaluation", __file__), ("execution", engine.__file__),
        ("classical_strategy", strategy.__file__), ("estimators", estimators.__file__),
    )}}


def evaluation_contract(algorithm, *, bundle=None, optimizer_settings=None, cost_bps=2.0):
    """Protocol recorded at issuance, before any forward return is observed."""
    env = EnvSpec.from_dict(bundle["environment"]) if bundle and bundle.get("members") else None
    cfg = settings(optimizer_settings) if algorithm in ("min_variance", "mean_variance", "cvar") else None
    primary = min(MAX_SESSIONS, env.episode_sessions or 63) if env else int(cfg["horizon_sessions"]) if cfg else 63
    gamma = (bundle or {}).get("hyperparameters", {}).get("gamma")
    objective = {
        "algorithm": algorithm,
        "kind": "expected_cumulative_reward" if env else "minimum_variance" if algorithm == "min_variance" else "return_minus_risk" if algorithm == "mean_variance" else "minimum_cvar" if algorithm == "cvar" else "strategy_return",
        "reward": env.document()["reward"] if env else None,
        "reward_formula": env.document()["reward_formula"] if env else None,
        "discount_factor": gamma,
        "discount_status": "recorded" if gamma is not None else "not_preserved_in_export" if env else "not_applicable",
        "training_episode_sessions": env.episode_sessions if env else None,
        "optimizer_settings": cfg,
        "interpretation": "Training objectives and historical estimates do not guarantee future optimality; episode length is not an investment guarantee",
    }
    # Controls are fixed before forward evaluation; settings never selected on this price path.
    controls = [{"name": name, "settings": {**DEFAULTS, "max_weight": cfg["max_weight"] if cfg else 1.0, "horizon_sessions": min(primary, 63)}} for name in ("min_variance", "mean_variance", "cvar")]
    return {
        "version": VERSION, "primary_horizon_sessions": primary,
        "horizons_sessions": sorted(set([1, 5, 21, primary])),
        "primary_horizon_basis": "recorded_training_episode" if env and env.episode_sessions else "declared_optimizer_horizon" if cfg else "fixed_review_window_not_training_horizon",
        "objective": objective, "controls": controls,
        "simulation": {"execution_timing": "next_open", "rebalance_frequency": env.decision_frequency if env else "daily", "cash_rate_annual": 0.0,
                       "fees": {"proportional_bps": float(cost_bps), "fixed_per_trade": 0.0},
                       "spread": {"half_spread_bps": 0.0}, "slippage": {"model": "none", "coefficient_bps": 0.0},
                       "liquidity": {"participation_cap": None, "unfilled": "cancel"}},
        "execution_assumptions": "Counterfactual next-open fills with recorded proportional paper fee; zero spread/slippage and no liquidity cap; same assumptions for every replay path, distinct from accepted reference-price paper fills",
        "evidence_rule": {"minimum_nonoverlapping_mature_windows_for_review": 3, "automatic_model_error_verdict": False, "automatic_activation": False},
    }


class _Control:
    def __init__(self, name):
        self.name = name

    def decide(self, view, holdings):
        if self.name == "unchanged_holdings":
            return TargetWeights(dict(holdings.weights), holdings.cash_weight, "no_effect")
        return TargetWeights({i: 1.0 / len(view.instruments) for i in view.instruments}, 0.0, "feasible")


class _FrozenStrategy:
    def __init__(self, doc, feature_market, *, bundle=None, optimizer=None):
        self.name = "issued_strategy"
        self.doc, self.market, self.bundle, self.optimizer = doc, feature_market, bundle, optimizer
        self.peak = float(doc["recommendation"]["portfolio_state"].get("high_watermark", doc["recommendation"]["portfolio_state"]["portfolio_value"]))
        if bundle and bundle.get("members"):
            self.peak = float(doc["provenance"]["policy_inputs"]["holdings"]["high_watermark"])

    def decide(self, view, holdings):
        self.peak = max(self.peak, holdings.value)
        features = self.market.view(view.decision_session, view.instruments)
        if self.bundle:
            state = {"weights": [{"instrument_id": i, "weight": holdings.weights[i]} for i in view.instruments],
                     "cash_weight": holdings.cash_weight, "portfolio_value": holdings.value, "high_watermark": self.peak}
            if self.bundle.get("members"):
                env = EnvSpec.from_dict(self.bundle["environment"])
                dates, px = features.price_matrix("close", lookback=env.window + 1)
                if len(dates) != env.window + 1 or dates[-1] != view.decision_session:
                    raise FinplanError.precondition("sequential actor lacks completed aligned history", reason="insufficient_policy_replay_history")
                result = recommend(self.bundle, px, state)
            else:
                result = recommend_baseline(self.bundle, self.market, view.decision_session, state)
            target = TargetWeights({r["instrument_id"]: r["weight"] for r in result["target_weights"]}, result["cash_weight"], result["solution_status"], result["diagnostics"])
        else:
            target = self.optimizer.decide(features, holdings)
        if view.decision_session.isoformat() == self.doc["recommendation"]["as_of"]:
            expected = self.doc["recommendation"]
            weights = {r["instrument_id"]: r["weight"] for r in expected["target_weights"]}
            if not np.allclose([target.weights[i] for i in view.instruments] + [target.cash], [weights[i] for i in view.instruments] + [expected["cash_weight"]], rtol=0, atol=1e-8):
                raise FinplanError.precondition("sequential replay does not reproduce original target", reason="sequential_initial_target_mismatch")
        return target


def _classical(name, cfg):
    result = ClassicalStrategy(name)
    result.settings = settings(cfg)
    return result


def _metrics(result, count, objective):
    nav = result.nav[:count + 1]
    values = [r["value"] for r in nav]
    dates = {r["session_date"] for r in nav[1:]}
    fills = [f for f in result.fills if f.session_date in dates]
    returns = period_returns(values)
    traded = sum(f.notional for f in fills)
    cost = sum(f.cost for f in fills)
    out = {"start_value": values[0], "end_value": values[-1], "pnl": values[-1] - values[0], "net_return": values[-1] / values[0] - 1,
           "max_drawdown": max_drawdown(values), "annualized_volatility": annualized_volatility(values), "sharpe": sharpe_ratio(values),
           "costs": cost, "turnover": traded / values[0], "fills": len(fills), "completed_sessions": count,
           "realized_session_cvar_loss": cvar(-np.asarray(returns), (objective.get("optimizer_settings") or {}).get("cvar_alpha", .9)) if returns else None}
    optimizer = objective.get("optimizer_settings")
    if optimizer:
        # Describe this realized path using the objective's terms without treating
        # its sample variance or tail loss as calibrated future risk estimates.
        variance = float(np.var(returns, ddof=1) * count) if len(returns) > 1 else None
        tail = out["realized_session_cvar_loss"]
        penalty = optimizer["turnover_penalty"] * out["turnover"]
        if objective["algorithm"] == "cvar":
            score = -count * tail - penalty if tail is not None else None
        elif variance is None:
            score = None
        elif objective["algorithm"] == "min_variance":
            score = -variance - penalty
        else:
            score = out["net_return"] - optimizer["risk_aversion"] * variance - penalty
        out["objective_diagnostics"] = {"realized_score": score, "realized_horizon_variance": variance,
                                        "basis": "realized path analogue of objective; sample estimates are descriptive, not the ex-ante solve objective or a significance test"}
    reward = objective.get("reward")
    if reward:
        peak = values[0]
        dd_prev = 0.0
        rewards = []
        for k in range(1, len(values)):
            log_r = float(np.log(values[k] / values[k - 1]))
            peak = max(peak, values[k])
            dd = 1 - values[k] / peak
            turnover = sum(f.notional for f in fills if f.session_date == nav[k]["session_date"]) / values[k - 1]
            rewards.append(reward["reward_scale"] * (log_r - reward["risk_penalty"] * log_r ** 2 - reward["drawdown_penalty"] * max(0.0, dd - dd_prev) - reward["turnover_penalty"] * turnover))
            dd_prev = dd
        gamma = objective.get("discount_factor")
        out["reward_diagnostics"] = {"undiscounted_daily_reward_sum": float(sum(rewards)),
                                     "discounted_reward": float(sum((gamma ** k) * r for k, r in enumerate(rewards))) if gamma is not None else None,
                                     "discount_status": objective["discount_status"], "basis": "daily net-return reward components; evaluation-window drawdown reset, diagnostic rather than training-episode value estimate"}
    return out


def evaluate_horizons(service, doc, market, raw, content, end):
    """Return complete evidence or a specific unavailable reason; never change holdings."""
    rec, provenance = doc["recommendation"], doc.get("provenance", {})
    contract = copy.deepcopy(provenance.get("evaluation_contract"))
    predeclared = contract is not None
    bundle = None
    artifact_error = None
    if provenance.get("policy_artifact"):
        try:
            bundle = json.loads(service.d.artifacts.get(provenance["policy_artifact"]))
        except FinplanError as error:
            artifact_error = error.details.get("reason", "frozen_policy_artifact_unavailable")
        except json.JSONDecodeError:
            artifact_error = "frozen_policy_artifact_invalid"
    if contract is None:
        contract = evaluation_contract(doc["algorithm"], bundle=bundle, optimizer_settings=provenance.get("settings"), cost_bps=provenance.get("paper_execution_cost_bps", 2.0))
    protocol_ref = sha256_checksum(canonical_json_bytes(contract))
    start = date.fromisoformat(rec["as_of"])
    sessions = [s for s in raw.sessions if start <= s <= end]
    available = len(sessions) - 1
    primary = contract["primary_horizon_sessions"]
    # Platform's immutable issuance time also preserves proposal idempotency: a
    # dynamically generated timestamp must not enter the proposal fingerprint.
    declared_at = doc.get("created_at")
    next_session = next((s for s in raw.sessions if s > start), None)
    historical = bool(predeclared and declared_at and next_session is not None and
                      datetime.fromisoformat(declared_at.replace("Z", "+00:00")) >= raw.decision_time(next_session))
    protocol_status = "retrospective_legacy_protocol" if not predeclared else "retrospective_historical_request" if historical else "predeclared_at_issuance"
    base = {
        "version": VERSION, "status": "not_available", "full_policy_replay": False,
        "protocol": contract, "protocol_checksum": protocol_ref,
        "protocol_status": protocol_status,
        "protocol_timing": {"declared_at": declared_at, "retrospective": protocol_status != "predeclared_at_issuance",
                            "basis": "Compare actual protocol declaration time with the first forward session close; backdated requests never become prospective evidence"},
        "available_forward_sessions": max(0, available), "primary_horizon_mature": available >= primary,
        "source": {"decision_id": doc["decision_id"], "portfolio_revision": doc["portfolio_revision"], "input_snapshot_id": doc["input_snapshot_id"], "algorithm": doc["algorithm"], "settings": provenance.get("settings"), "policy_artifact": provenance.get("policy_artifact"), "configuration_id": provenance.get("configuration_id"), "implementation": provenance.get("implementation"), "evaluation_implementation": provenance.get("evaluation_implementation"), "observed_snapshot_checksum": content.snapshot.manifest_checksum},
        "limitations": ["one_realized_market_path_not_proof_of_optimality", "overlapping_windows_are_not_independent_trials", "counterfactual_replay_not_observed_execution", "no_calibrated_return_forecast", "provider_historical_vintages_after_issuance_not_fully_verified", "evaluation_cost_model_distinct_from_training_and_reference_price_paper_fills", "no_causal_market_explanation_from_news_alone"],
        "evidence_assessment": {"status": "preliminary" if available < primary else "single_path_review", "model_error_proven": False,
                                "independent_mature_windows": 0, "reason": "Maturity of a window does not establish statistical skill; repeated nonoverlapping out-of-sample evidence is required", "automatic_activation": False},
    }
    try:
        if artifact_error:
            raise FinplanError.precondition("frozen policy evidence unavailable", reason=artifact_error)
        if provenance.get("evaluation_implementation") and provenance["evaluation_implementation"] != evaluation_implementation():
            raise FinplanError.precondition("issued evaluation implementation changed", reason="evaluation_implementation_mismatch")
        if contract.get("version") != VERSION or not contract.get("horizons_sessions") or any(not isinstance(h, int) or not 1 <= h <= MAX_SESSIONS for h in contract["horizons_sessions"]):
            raise FinplanError.precondition("unsupported horizon protocol", reason="evaluation_protocol_invalid")
        if available < 1:
            raise FinplanError.precondition("no forward sessions", reason="no_forward_observations")
        capped = min(available, max(contract["horizons_sessions"]))
        replay_end = sessions[capped]
        instruments = [r["instrument_id"] for r in rec["target_weights"]]
        if instruments != sorted(instruments):
            raise FinplanError.precondition("recorded actor universe order is unsupported", reason="policy_universe_order_mismatch")
        for s in sessions[:capped + 1]:
            if any(raw.bar(i, s) is None or raw.bar(i, s).open is None or raw.bar(i, s).available_at > raw.decision_time(s) for i in instruments):
                raise FinplanError.precondition("replay lacks aligned completed raw bars", reason="replay_market_path_incomplete")
        actions = [r for r in content.payload.get("observations", []) if r["instrument_id"] in instruments and start.isoformat() < r["session_date"] <= replay_end.isoformat() and (float(r.get("dividend", 0.0) or 0.0) != 0 or float(r.get("split_ratio", 1.0) or 1.0) != 1)]
        if actions:
            raise FinplanError.precondition("corporate actions need dated accounting", reason="corporate_action_accounting_not_recorded")
        if bundle:
            if provenance.get("implementation") != implementation_identity():
                raise FinplanError.precondition("inference implementation changed", reason="policy_implementation_mismatch")
            if bundle.get("configuration_id") != provenance.get("configuration_id") or list(bundle.get("instruments", [])) != instruments:
                raise FinplanError.precondition("frozen bundle identity differs", reason="policy_identity_mismatch")
            inputs = provenance.get("policy_inputs")
            if not inputs:
                raise FinplanError.precondition("frozen inputs missing", reason="frozen_policy_inputs_missing")
            if bundle.get("members"):
                env = EnvSpec.from_dict(bundle["environment"])
                _, px = market.view(start, instruments).price_matrix("close", lookback=env.window + 1)
                if px.shape != np.asarray(inputs["prices"]).shape or not np.allclose(px, inputs["prices"], rtol=0, atol=1e-10):
                    raise FinplanError.precondition("issued feature history was restated", reason="issued_policy_history_changed")
            strategy = _FrozenStrategy(doc, market, bundle=bundle)
        elif doc["algorithm"] in ("min_variance", "mean_variance", "cvar"):
            if not doc.get("source_analysis_id"):
                raise FinplanError.precondition("original optimizer inputs missing", reason="frozen_optimizer_inputs_missing")
            original = service.plan(doc["source_analysis_id"])
            service.reproduce(original)
            saved = original["solve_inputs"]
            returns = pit_returns(market.view(start, instruments), saved["settings"]["lookback_days"])
            reconstructed = {"scenarios": returns, "expected_returns": estimate_mean(returns, "historical_mean"), "covariance": estimate_covariance(returns, "ledoit_wolf")}
            if any(np.asarray(saved[key]).shape != np.asarray(value).shape or not np.allclose(saved[key], value, rtol=0, atol=1e-10) for key, value in reconstructed.items()):
                raise FinplanError.precondition("issued optimizer history was restated", reason="issued_optimizer_history_changed")
            strategy = _FrozenStrategy(doc, market, optimizer=_classical(doc["algorithm"], provenance["settings"]))
        else:
            raise FinplanError.precondition("no replay adapter for issued strategy", reason="strategy_replay_not_supported")
        state = rec["portfolio_state"]
        positions = {r["instrument_id"]: float(r["quantity"]) for r in state["positions"]}
        nav = float(state["portfolio_value"])
        cash = float(state["current_cash"])
        initial = HoldingsView(positions, cash, {i: raw.bar(i, start).close for i in instruments}, nav)
        simulation = {**contract["simulation"], "initial_cash": nav}
        # The frozen bundle's exposure constraints apply equally to every comparator.
        simulation["constraints"] = (bundle or {}).get("constraints", {"max_weight": provenance["settings"]["max_weight"]} if provenance.get("settings") else {})
        simulation["constraint_policy"] = (bundle or {}).get("constraint_policy", "project")
        cfg = SimulationConfig.from_dict(simulation)
        paths = {"issued_strategy": strategy, "unchanged_holdings": _Control("unchanged_holdings"), "equal_weight": _Control("equal_weight")}
        paths.update({"control_" + c["name"]: _FrozenStrategy(doc, market, optimizer=_classical(c["name"], c["settings"])) for c in contract["controls"]})
        # Controls must not be required to reproduce the issued strategy's first target.
        for name, path in paths.items():
            if name.startswith("control_"):
                path.doc = {**doc, "recommendation": {**rec, "as_of": "control"}}
        results = {name: Simulator(cfg, raw, universe=instruments).run(path, start=start, end=replay_end, initial_book=initial) for name, path in paths.items()}
        windows = []
        for horizon in contract["horizons_sessions"]:
            count = min(horizon, capped)
            metrics = {name: _metrics(result, count, contract["objective"]) for name, result in results.items()}
            selected = metrics["issued_strategy"]
            windows.append({"horizon_sessions": horizon, "observed_sessions": count, "status": "mature" if available >= horizon else "partial", "end_date": sessions[count].isoformat(),
                            "strategies": metrics, "relative_to_unchanged": {"net_return_difference": selected["net_return"] - metrics["unchanged_holdings"]["net_return"], "pnl_difference": selected["pnl"] - metrics["unchanged_holdings"]["pnl"], "max_drawdown_difference": selected["max_drawdown"] - metrics["unchanged_holdings"]["max_drawdown"]}})
        base.update(status="available", full_policy_replay=True, windows=windows, simulation_configuration_id=cfg.configuration_id, cost_model_id=cfg.cost_model_id,
                    replay_window={"start": start.isoformat(), "end": replay_end.isoformat(), "sessions": capped, "bounded_to_protocol": available > capped},
                    sequential_evidence={name: {"nav": result.nav, "decisions": result.decisions, "fills": [f.to_dict() for f in result.fills], "reconciliation": result.reconciliation} for name, result in results.items()})
        primary_window = next(w for w in windows if w["horizon_sessions"] == primary)
        base["evidence_assessment"]["primary_return_underperformed_unchanged"] = primary_window["relative_to_unchanged"]["net_return_difference"] < 0
        base["evidence_assessment"]["eligible_for_prospective_skill_evidence"] = protocol_status == "predeclared_at_issuance"
        base["evidence_assessment"]["recommendation"] = "Retain as preliminary evidence until the primary window matures; daily losses alone do not diagnose the model" if available < primary else "Review objective, risk and costs across independent chronological windows before proposing retraining or a model change"
    except FinplanError as error:
        base["reason"] = error.details.get("reason", "replay_precondition_failed")
    return base
