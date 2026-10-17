"""One sandbox CPU job: finite classical grid, three controls and chronological selection.

No source generation, model activation or publication. Validation picks a candidate; a later
research-only test window reports it once. This is a repeated research test, not pristine holdout.
"""

from __future__ import annotations

from finplan_model.evaluate import evaluate
from finplan_model.jobs.handlers import _bias, _with_bias
from finplan_model.jobs.results import succeeded_result
from finplan_model.jobs.strategy_resolver import resolve_strategy

from .strategy import ClassicalStrategy


def run_weekly_benchmark(inp):
    sessions, _ = inp.market.view(
        inp.window[1] or inp.market.sessions[-1], inp.universe
    ).price_matrix("close")
    payload = inp.spec["configuration"]["payload"]
    lookbacks = (
        (60, 120, 252) if payload.get("lookback_days", 60) >= 120 else (20, 60, 120)
    )
    warmup = max(lookbacks)
    start, end = inp.window
    sessions = [
        d
        for i, d in enumerate(sessions)
        if i >= warmup and (start is None or d >= start) and (end is None or d <= end)
    ]
    if len(sessions) < 60:
        from finplan_model.core.errors import FinplanError

        raise FinplanError.precondition(
            "weekly review needs common maximum-lookback history and at least 60 completed evaluation sessions",
            reason="insufficient_research_history",
        )
    cut = max(21, int(len(sessions) * 0.7))
    windows = {
        "validation": (sessions[0], sessions[cut - 1]),
        "research_test": (sessions[cut], sessions[-1]),
    }
    risk_aversion = float(payload.get("risk_aversion", 2.0))
    max_weight = inp.sim_config.constraints.max_weight
    variants = [("cash", {}), ("buy_and_hold", {}), ("equal_weight", {})]
    for algorithm in ("min_variance", "mean_variance", "scenario_cvar"):
        for lookback in lookbacks:
            params = {"lookback": lookback}
            if algorithm == "mean_variance":
                params["risk_aversion"] = risk_aversion
            if algorithm == "scenario_cvar":
                params.update({"scenario_method": "historical", "alpha": 0.9})
            variants.append((algorithm, params))
    rows = []
    for algorithm, params in variants:
        variant_id = algorithm + "_" + str(params.get("lookback", 0))
        row = {
            "variant_id": variant_id,
            "algorithm": algorithm,
            "parameters": params,
            "splits": {},
        }
        for split, (a, b) in windows.items():
            strategy = (
                ClassicalStrategy(
                    algorithm,
                    lookback=params["lookback"],
                    max_weight=max_weight,
                    risk_aversion=risk_aversion,
                )
                if "lookback" in params
                else resolve_strategy(
                    algorithm, params, constraints=inp.sim_config.constraints
                )
            )
            result = evaluate(
                strategy,
                inp.market,
                inp.sim_config,
                ctx=inp.ctx,
                start=a,
                end=b,
                universe=inp.universe,
            )
            row["splits"][split] = result.metrics
        rows.append(row)

    def score(row):
        m = row["splits"]["validation"]
        return float(m["sharpe_ratio"]) if m.get("sharpe_ratio") is not None else -1e30

    selected = max(rows, key=score)
    evidence = {
        "protocol": "finplan-classical-weekly/1",
        "variants": rows,
        "selection": {
            "split": "validation",
            "metric": "sharpe_ratio",
            "selected_variant_id": selected["variant_id"],
        },
        "windows": {
            k: {"start": a.isoformat(), "end": b.isoformat()}
            for k, (a, b) in windows.items()
        },
        "hypothesis": "Evidence-selected bounded covariance history and risk trade-off sensitivity after transaction costs with daily decisions",
        "candidate_configuration": {
            "lookbacks": list(lookbacks),
            "risk_aversion": risk_aversion,
            "max_weight": max_weight,
            "rebalance_frequency": inp.sim_config.rebalance_frequency,
        },
        "common_warmup": {
            "required_completed_returns": warmup,
            "evaluation_sessions": len(sessions),
            "first_evaluation_session": sessions[0].isoformat(),
        },
        "activation": "proposal_only",
        "limitations": [
            "hindsight_selected_universe",
            "reused_history_research_test",
            "fresh_forward_paper_validation_required",
            "no_rl_training",
            "costs_use_existing_simulator",
            "optimizer_matches_classical_mcp_objective_with_turnover_proxy",
        ],
        "bias_section": _bias(inp),
    }
    ref = inp.artifacts.put_json(
        evidence, kind="run_artifact", synthetic=True, domain="finance"
    )
    out = succeeded_result(
        inp.ctx,
        inp.spec,
        solution_status="feasible",
        artifacts=[ref],
        performance=selected["splits"]["research_test"],
        dataset_checksum=inp.market.dataset_checksum,
    )
    out["payload"]["weekly_research"] = evidence
    return _with_bias(out, evidence["bias_section"])
