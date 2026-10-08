"""The ``model_selection`` job (decision 27; specs rl-strategies and baseline-strategies; ``docs/model-selection.md``).

One SageMaker **Training** job (CPU, ``ml.m5.xlarge``) on one approved universe snapshot:

1. restrict the market data to the protocol's data range (decision 27: 2025-2026 only) and resolve
   the train, validation and test windows to sessions;
2. **controls** (``cash``, ``buy_and_hold``, ``equal_weight``) on train and validation;
3. **traditional** optimizers: every grid point evaluated on validation, the best (declared metric,
   ties to the first grid point) kept, then evaluated on train;
4. **RL**: for each algorithm, reward configuration and seed, train on the training window with
   validation checkpoint selection (:func:`finplan_model.rl.train.train_policy`), evaluate the stored
   policy (deterministic) on validation with the common evaluator; the configuration with the best
   mean validation metric across seeds and its best seed are chosen; all seeds of the chosen
   configuration are also evaluated on train;
5. the **selection record** (validation ranking, chosen hyperparameters, selected candidate) is
   frozen and checksummed; only then does the evaluator open the test window
   (:class:`SplitEvaluator`: a test evaluation before the freeze is ``OPERATION_NOT_PERMITTED``);
6. **test**, once: every family's chosen model, every seed of the chosen RL configurations and the
   incumbent; the promotion check of criteria v1 (decision 15c) on the selected candidate;
7. the result: ``payload.benchmark`` (test comparison), ``payload.model_selection`` (per-split
   comparisons, tuning tables, chosen hyperparameters, RL seed statistics, a separate training-reward
   section, the promotion check, caveats, the compute record) and the mandatory
   "Hindsight and survivorship bias" section; artifacts: the full evidence and every trained policy
   (``rl_policy``, with checksum, seed and ``configuration_id``).

Every portfolio number comes from :func:`finplan_model.evaluate.evaluate` with the same simulation
configuration (costs, daily rebalance, long-only, weight cap 1, universe plus cash) for every family.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.context import RunContext
from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.evaluate import EvaluationResult, assert_comparable, evaluate
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import MarketData
from finplan_model.sim.strategy import Strategy

from .protocol import (
    CONTROL_NAMES,
    PROTOCOL_VERSION,
    grid_points,
    protocol_id,
    validate_protocol,
)

__all__ = ["SCHEMA", "SelectionOutcome", "SplitEvaluator", "restrict_market", "run_model_selection", "select_models"]

SCHEMA = "finplan.model_selection/1"
JOB_TYPE = "model_selection"
ALLOWED_PURPOSES = ("research", "holdout_evaluation")
_METRIC_KEYS = ("total_return", "cagr", "ann_volatility", "sharpe", "max_drawdown", "turnover", "transaction_cost_fraction")
#: Run-result size guard (the result is stored in the run item; keep it well inside the item limit).
MAX_RESULT_BYTES = 300_000


# ===================================================================== data
def restrict_market(market: MarketData, start: date, end: date | None) -> MarketData:
    """The market data limited to ``[start, end]`` (nothing outside is visible to any strategy)."""
    sessions = [s for s in market.sessions if s >= start and (end is None or s <= end)]
    if len(sessions) < 2:
        raise FinplanError.precondition("the snapshot has no data in the protocol's data range", reason="no_data_in_range", start=start.isoformat(), end=end.isoformat() if end else None)
    keep = set(sessions)
    bars = [b for i in market.instruments for b in market.bars_of(i) if b.session_date in keep]
    return MarketData(sessions, bars, decision_times={s: market.decision_time(s) for s in sessions}, synthetic=market.synthetic, dataset_id=market.dataset_id, dataset_checksum=market.dataset_checksum)


def _window(market: MarketData, split: Mapping[str, Any], name: str) -> tuple[date, date]:
    start = date.fromisoformat(split["start"])
    end = date.fromisoformat(split["end"]) if split.get("end") else market.sessions[-1]
    sessions = [s for s in market.sessions if start <= s <= end]
    if len(sessions) < 2:
        raise FinplanError.precondition(f"the {name} split has fewer than two sessions in the snapshot", reason="insufficient_split_data", split=name)
    return sessions[0], sessions[-1]


# ===================================================================== evaluator with a test lock
class SplitEvaluator:
    """The common evaluator per split; the test window opens only after the selection is frozen."""

    def __init__(self, market: MarketData, config: SimulationConfig, ctx: RunContext | None, universe: Sequence[str], windows: Mapping[str, tuple[date, date]]) -> None:
        self.market, self.config, self.ctx, self.universe, self.windows = market, config, ctx, list(universe), dict(windows)
        self.selection_checksum: str | None = None
        self.calls: dict[str, int] = {k: 0 for k in self.windows}

    def run(self, strategy: Strategy, split: str) -> EvaluationResult:
        if split == "test" and self.selection_checksum is None:
            raise FinplanError(ErrorCode.OPERATION_NOT_PERMITTED, "the test window opens only after the validation selection is frozen", details={"split": split})
        start, end = self.windows[split]
        self.calls[split] += 1
        return evaluate(strategy, self.market, self.config, ctx=self.ctx, start=start, end=end, universe=self.universe)

    def open_test(self, selection_record: Mapping[str, Any]) -> str:
        if self.selection_checksum is not None:
            raise FinplanError(ErrorCode.OPERATION_NOT_PERMITTED, "the selection is already frozen; the test window is evaluated once")
        self.selection_checksum = sha256_checksum(canonical_json_bytes(dict(selection_record)))
        return self.selection_checksum


# ===================================================================== helpers
def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return None
    return round(float(x), 6) + 0.0


def metrics_of(res: EvaluationResult) -> dict[str, float | None]:
    sim, met = res.simulation, res.metrics
    cfg = sim.config
    total = float(met.get("net_cumulative_return") or 0.0)
    periods = max(len(sim.nav) - 1, 1)
    cagr = -1.0 if total <= -1.0 else (1.0 + total) ** (cfg.periods_per_year / periods) - 1.0
    costs = float(sim.summary.get("total_costs") or 0.0)
    init = float(cfg.initial_cash)
    return {
        "total_return": _num(total),
        "cagr": _num(cagr),
        "ann_volatility": _num(met.get("annualized_volatility")),
        "sharpe": _num(met.get("sharpe_ratio")),
        "max_drawdown": _num(met.get("max_drawdown")),
        "turnover": _num(met.get("turnover")),
        "transaction_cost_fraction": _num(costs / init if init else 0.0),
    }


def _score(metrics: Mapping[str, Any], metric: str) -> float:
    key = "total_return" if metric == "net_return" else metric
    v = metrics.get(key)
    return -math.inf if v is None else float(v)


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": len(xs), "mean": _num(statistics.fmean(xs)), "std": _num(statistics.stdev(xs)) if len(xs) > 1 else 0.0, "min": _num(min(xs)), "max": _num(max(xs))}


def seed_statistics(rows: Sequence[Mapping[str, Mapping[str, Any]]], split: str) -> dict[str, Any]:
    return {k: _stats([r[split].get(k) for r in rows if split in r]) for k in ("total_return", "sharpe", "max_drawdown", "ann_volatility", "turnover")}


@dataclass
class Candidate:
    name: str
    family: str  # control | traditional | rl
    params: dict[str, Any]
    factory: Callable[[], Strategy]
    configuration_id: str | None = None
    seed: int | None = None
    results: dict[str, EvaluationResult] = field(default_factory=dict)

    def metrics(self, split: str) -> dict[str, Any]:
        return metrics_of(self.results[split]) if split in self.results else {}


@dataclass
class SelectionOutcome:
    summary: dict[str, Any]
    test_section: dict[str, Any]
    selected: Candidate
    evidence: dict[str, Any]
    policies: list[tuple[dict[str, Any], bytes]]
    performance: dict[str, Any]


# ===================================================================== core
def select_models(
    market: MarketData,
    config: SimulationConfig,
    *,
    protocol: Mapping[str, Any],
    universe: Sequence[str] | None = None,
    ctx: RunContext | None = None,
    deadline: float | None = None,
    incumbent: str | None = None,
    prior_test_accesses: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> SelectionOutcome:
    """Run the protocol end to end (see the module docstring); pure computation, no I/O."""
    from finplan_model.jobs.comparison import benchmark_section
    from finplan_model.jobs.strategy_resolver import resolve_strategy

    t_start = clock()
    phases: dict[str, float] = {}
    proto = validate_protocol(protocol)
    rule = proto["selection"]
    metric = rule["metric"]
    data_start = date.fromisoformat(proto["data_start"])
    test_end = date.fromisoformat(proto["splits"]["test"]["end"]) if proto["splits"]["test"].get("end") else None
    mkt = restrict_market(market, data_start, test_end)
    names = sorted(universe or mkt.instruments)
    missing = [i for i in names if i not in mkt.instruments]
    if missing:
        raise FinplanError.precondition("the snapshot does not cover every universe instrument in the protocol's data range", reason="universe_not_covered", instruments=missing[:10])
    windows = {s: _window(mkt, proto["splits"][s], s) for s in ("train", "validation", "test")}
    ev = SplitEvaluator(mkt, config, ctx, names, windows)
    log = ctx.log if ctx is not None else (lambda *a, **k: {})
    cons = config.constraints

    def mark(phase: str, t0: float) -> None:
        phases[phase] = round(phases.get(phase, 0.0) + clock() - t0, 3)

    # ---------------------------------------------------------------- controls
    t0 = clock()
    candidates: list[Candidate] = []
    for name in CONTROL_NAMES:
        c = Candidate(name, "control", {}, (lambda n=name: resolve_strategy(n, {}, constraints=cons)))
        for split in ("train", "validation"):
            c.results[split] = ev.run(c.factory(), split)
        candidates.append(c)
    mark("controls", t0)

    # ---------------------------------------------------------------- traditional optimizers
    t0 = clock()
    tuning: dict[str, Any] = {}
    for name, block in proto["traditional"].items():
        rows: list[dict[str, Any]] = []
        best: tuple[float, int, dict[str, Any], EvaluationResult] | None = None
        for k, point in enumerate(grid_points(block["grid"])):
            params = {**dict(block.get("fixed") or {}), **point}
            res = ev.run(resolve_strategy(name, params, constraints=cons), "validation")
            met = metrics_of(res)
            rows.append({"params": point, "validation": met})
            score = _score(met, metric)
            if best is None or score > best[0] + 1e-12:
                best = (score, k, params, res)
        assert best is not None
        params = best[2]
        c = Candidate(name, "traditional", params, (lambda n=name, p=dict(params): resolve_strategy(n, p, constraints=cons)))
        c.results["validation"] = best[3]
        c.results["train"] = ev.run(c.factory(), "train")
        candidates.append(c)
        tuning[name] = {"grid": rows, "chosen": params, "chosen_grid_index": best[1], "fixed": dict(block.get("fixed") or {})}
        log("traditional_tuned", strategy=name, chosen=params)
    mark("traditional_tuning", t0)

    # ---------------------------------------------------------------- reinforcement learning
    rl_section: dict[str, Any] = {}
    training_reward: dict[str, Any] = {}
    policies: list[tuple[dict[str, Any], bytes]] = []
    rl_seed_candidates: dict[str, list[Candidate]] = {}
    rl_proto = proto["rl"]
    if rl_proto.get("algorithms"):
        t0 = clock()
        rl_chosen = _run_rl(mkt, config, names, windows, rl_proto, metric, ev, ctx, deadline, clock, rl_section, training_reward, policies, rl_seed_candidates)
        candidates += rl_chosen
        mark("rl_training_and_validation", t0)

    # ---------------------------------------------------------------- freeze the selection
    incumbent_name = incumbent if incumbent in [c.name for c in candidates if c.family != "rl"] else proto["incumbent_fallback"]
    incumbent_source = "production_strategy" if incumbent and incumbent == incumbent_name else "fallback_no_comparable_production_strategy"
    ranking = sorted(
        ({"strategy": c.name, "family": c.family, "seed": c.seed, "validation_" + metric: _num(_score(c.metrics("validation"), metric)) if math.isfinite(_score(c.metrics("validation"), metric)) else None} for c in candidates),
        key=lambda r: (-(r["validation_" + metric] if r["validation_" + metric] is not None else -math.inf)),
    )
    eligible = [r for r in ranking if r["strategy"] != incumbent_name]
    if not eligible:
        raise FinplanError.precondition("no candidate other than the incumbent was evaluated", reason="no_candidates")
    selected_name = eligible[0]["strategy"]
    selected = next(c for c in candidates if c.name == selected_name)
    chosen_hp = {c.name: ({"seed": c.seed, "configuration_id": c.configuration_id, **c.params} if c.family == "rl" else dict(c.params)) for c in candidates}
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": protocol_id(proto),
        "rule": {**rule, "direction": "higher is better", "tie_break": "first in declared grid / family order"},
        "validation_ranking": ranking,
        "selected": {"strategy": selected.name, "family": selected.family, "seed": selected.seed},
        "chosen_hyperparameters": chosen_hp,
        "incumbent": incumbent_name,
    }
    checksum = ev.open_test(record)
    record["selection_checksum"] = checksum
    log("selection_frozen", selected=selected.name, selection_checksum=checksum)

    # ---------------------------------------------------------------- test, once
    t0 = clock()
    for c in candidates:
        c.results["test"] = ev.run(c.factory(), "test")
    for seeds in rl_seed_candidates.values():
        for c in seeds:
            if "test" not in c.results:
                c.results["test"] = ev.run(c.factory(), "test")
    inc = next((c for c in candidates if c.name == incumbent_name), None)
    mark("test_evaluation", t0)
    for split in ("train", "validation", "test"):
        assert_comparable([c.results[split] for c in candidates if split in c.results])

    # ---------------------------------------------------------------- reports
    comparisons: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        section = benchmark_section([c.results[split] for c in candidates], mkt, primary=selected.name)
        for row, c in zip(section["strategies"], candidates):
            row["family"] = c.family
            row["params"] = dict(c.params)
            if c.family == "rl":
                row["configuration_id"] = c.configuration_id
                row["seed"] = c.seed
            row["selected"] = c.name == selected.name
            row["incumbent"] = c.name == incumbent_name
        comparisons[split] = section
    for algo, seeds in rl_seed_candidates.items():
        rows = [{"seed": c.seed, **{s: c.metrics(s) for s in ("train", "validation", "test") if s in c.results}} for c in seeds]
        block = rl_section[algo]
        block["seeds"] = [{**r, "selected": r["seed"] == block["chosen"]["seed"]} for r in rows]
        block["seed_statistics"] = {s: seed_statistics(rows, s) for s in ("train", "validation", "test")}
    gate = promotion_check(selected, inc, incumbent_source)
    if prior_test_accesses:
        gate.update(result="research_only", reason="test_period_reused_for_development")
    n_train = len([s for s in mkt.sessions if windows["train"][0] <= s <= windows["train"][1]])
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": record["protocol_id"],
        "data": {
            "data_start": mkt.sessions[0].isoformat(),
            "data_end": mkt.sessions[-1].isoformat(),
            "splits": {s: {"start": w[0].isoformat(), "end": w[1].isoformat(), "sessions": len([d for d in mkt.sessions if w[0] <= d <= w[1]])} for s, w in windows.items()},
            "universe": list(names),
            "cash": "residual weight, simulation cash rate",
        },
        "evaluation": {
            "evaluator": "finplan_model.evaluate.evaluate (common evaluator; same simulation configuration for every family)",
            "simulation_configuration_id": config.configuration_id,
            "cost_model_id": config.cost_model_id,
            "rebalance_frequency": config.rebalance_frequency,
            "long_only": bool(cons.long_only),
            "max_weight": cons.max_weight,
            "execution_timing": config.execution_timing,
        },
        "selection": record,
        "comparison": comparisons,
        "traditional": tuning,
        "rl": rl_section,
        "training_reward": {"note": "shaped RL training rewards; reported separately and never as portfolio performance", "reward_formula": _reward_formula(), "runs": training_reward},
        "promotion_check": gate,
        "test_access": {"evaluated_once_in_this_run": True, "split_evaluations": dict(ev.calls), "prior_runs_on_this_test_period": int(prior_test_accesses), "test_reuse": int(prior_test_accesses) > 0},
        "caveats": caveats(n_train, windows, mkt, rl_section, reused=bool(prior_test_accesses)),
        "compute": {"phase_seconds": phases, "wall_seconds": round(clock() - t_start, 3)},
    }
    evidence = {
        "schema": SCHEMA + "/evidence",
        "protocol": proto,
        "selection": record,
        "results": {split: {(c.name if c.seed is None else f"{c.name}-seed{c.seed}"): c.results[split].to_dict() for c in candidates if split in c.results} for split in ("train", "validation", "test")},
        "rl_seed_results": {algo: [{"seed": c.seed, "configuration_id": c.configuration_id, **{s: {"metrics": c.metrics(s), "result_checksum": c.results[s].result_checksum} for s in c.results}} for c in seeds] for algo, seeds in rl_seed_candidates.items()},
    }
    performance = dict(selected.results["test"].metrics)
    return SelectionOutcome(summary, comparisons["test"], selected, evidence, policies, performance)


def _reward_formula() -> str:
    from finplan_model.rl.spec import REWARD_FORMULA

    return REWARD_FORMULA


def _run_rl(
    mkt: MarketData,
    config: SimulationConfig,
    names: Sequence[str],
    windows: Mapping[str, tuple[date, date]],
    rl_proto: Mapping[str, Any],
    metric: str,
    ev: SplitEvaluator,
    ctx: RunContext | None,
    deadline: float | None,
    clock: Callable[[], float],
    rl_section: dict[str, Any],
    training_reward: dict[str, Any],
    policies: list[tuple[dict[str, Any], bytes]],
    rl_seed_candidates: dict[str, list[Candidate]],
) -> list[Candidate]:
    from finplan_model.core.ids import configuration_id
    from finplan_model.rl.arrays import PriceArrays
    from finplan_model.rl.policy import EnsemblePolicyStrategy, RLPolicyStrategy
    from finplan_model.rl.spec import EnvSpec, policy_configuration
    from finplan_model.rl.train import load_predictor, train_policy

    log = ctx.log if ctx is not None else (lambda *a, **k: {})
    arrays = PriceArrays.from_market(mkt, names, execution_timing=config.execution_timing)
    t_win = arrays.window_indices(*windows["train"])
    v_win = arrays.window_indices(*windows["validation"])
    base = EnvSpec.from_dict(rl_proto.get("env"))
    if base.decision_frequency != config.rebalance_frequency:
        # every family must decide on the same schedule (user decision 28: daily)
        raise FinplanError.validation(
            f"RL decides {base.decision_frequency} but the common evaluator rebalances {config.rebalance_frequency}; set configuration.rebalance_frequency to match",
            pointer="/configuration/rebalance_frequency",
        )
    grid = grid_points(rl_proto.get("grid") or {}) or [{}]
    training_range = {"start": windows["train"][0].isoformat(), "end": windows["train"][1].isoformat()}
    chosen: list[Candidate] = []
    for algo in rl_proto["algorithms"]:
        hp = dict(rl_proto[algo])
        configs: list[dict[str, Any]] = []
        for point in grid:
            env_spec = base.with_reward(**point) if point else base
            conf = policy_configuration(algo, hp, env_spec, simulation_configuration_id=config.configuration_id, training_range=training_range, instruments=tuple(arrays.instruments))
            cid = configuration_id(conf)
            seeds: list[Candidate] = []
            runs: list[dict[str, Any]] = []
            for seed in rl_proto["seeds"]:
                if deadline is not None and clock() > deadline:
                    runs.append({"seed": seed, "status": "skipped_time_budget"})
                    continue
                log("rl_training_started", algorithm=algo, seed=seed, configuration_id=cid, reward=point)
                tp = train_policy(arrays, env_spec, config, algorithm=algo, seed=seed, hyperparameters=hp, train_window=t_win, validation_window=v_win, configuration_id=cid, metric=metric, deadline=deadline)
                log("rl_training_finished", algorithm=algo, seed=seed, seconds=round(tp.seconds, 2), best_step=tp.best_step, stopped=tp.stopped_reason)
                predict = load_predictor(algo, tp.model_bytes)
                strat_factory = (lambda a=algo, p=predict, s=seed, e=env_spec, c=cid: RLPolicyStrategy(a, p, e, seed=s, configuration_id=c))
                cand = Candidate(algo, "rl", {"reward": dict(point)} if point else {}, strat_factory, configuration_id=cid, seed=int(seed))
                cand.results["validation"] = ev.run(cand.factory(), "validation")
                seeds.append(cand)
                runs.append({"seed": seed, "status": "trained", **tp.summary(), "training_reward": tp.training_reward, "validation_curve": tp.validation_curve, "validation": cand.metrics("validation")})
                policies.append(({"algorithm": algo, "seed": int(seed), "configuration_id": cid, "checksum": tp.checksum, "reward": dict(point)}, tp.model_bytes))
            scores = [_score(c.metrics("validation"), metric) for c in seeds]
            finite = [s for s in scores if math.isfinite(s)]
            mean_score = statistics.fmean(finite) if finite else -math.inf
            configs.append({"point": point, "configuration_id": cid, "configuration": conf, "seeds": seeds, "runs": runs, "mean": mean_score})
            training_reward.setdefault(algo, []).extend({"reward": dict(point), **{k: v for k, v in r.items() if k in ("seed", "status", "training_reward", "timesteps_trained", "best_step", "stopped_reason", "seconds")}} for r in runs)
        # An incomplete seed grid cannot win by omitting its slower or unlucky seeds.
        trained = [c for c in configs if len(c["seeds"]) == len(rl_proto["seeds"])
                   and not any(r.get("stopped_reason") == "time_budget" for r in c["runs"])]
        if not trained:
            rl_section[algo] = {"status": "not_trained_time_budget", "grid": [{"reward": c["point"], "configuration_id": c["configuration_id"], "runs": c["runs"]} for c in configs]}
            continue
        best_cfg = trained[0]
        for c in trained[1:]:
            if c["mean"] > best_cfg["mean"] + 1e-12:
                best_cfg = c
        best_seed = best_cfg["seeds"][0]
        for c in best_cfg["seeds"][1:]:
            if _score(c.metrics("validation"), metric) > _score(best_seed.metrics("validation"), metric) + 1e-12:
                best_seed = c
        for c in best_cfg["seeds"]:
            c.results["train"] = ev.run(c.factory(), "train")
        rl_seed_candidates[algo] = best_cfg["seeds"]
        policy_selection = rl_proto.get("policy_selection", "best_seed")
        candidate = best_seed
        if policy_selection == "ensemble":
            factories = [c.factory for c in best_cfg["seeds"]]
            ensemble_id = configuration_id({"base_configuration_id": best_cfg["configuration_id"], "seeds": list(rl_proto["seeds"]), "aggregation": "mean_target_weights"})
            candidate = Candidate(algo, "rl", {"reward": dict(best_cfg["point"]), "policy_selection": "ensemble", "seeds": list(rl_proto["seeds"])},
                                  lambda a=algo, fs=factories: EnsemblePolicyStrategy(a, [f() for f in fs]), configuration_id=ensemble_id)
            for split in ("train", "validation"):
                candidate.results[split] = ev.run(candidate.factory(), split)
        chosen.append(candidate)
        rl_section[algo] = {
            "status": "trained",
            "hyperparameters": hp,
            "environment": best_cfg["configuration"]["environment"],
            "grid": [
                {
                    "reward": c["point"],
                    "configuration_id": c["configuration_id"],
                    "mean_validation_" + metric: _num(c["mean"]) if math.isfinite(c["mean"]) else None,
                    "trained_seeds": len(c["seeds"]),
                    "skipped_seeds": sum(1 for r in c["runs"] if r.get("status") != "trained"),
                    # validation checkpoint selection per seed (environment replay of the evaluator's rules)
                    "checkpoints": [{"seed": r["seed"], "best_step": r.get("best_step"), "stopped_reason": r.get("stopped_reason"), "curve": r.get("validation_curve")} for r in c["runs"] if r.get("status") == "trained"],
                }
                for c in configs
            ],
            "chosen": {"reward": best_cfg["point"], "configuration_id": candidate.configuration_id, "seed": candidate.seed, "policy_selection": policy_selection,
                       "rule": f"configuration: highest mean validation {metric} across all requested seeds; policy: " + ("mean target weights of every seed" if policy_selection == "ensemble" else f"highest validation {metric} seed within it")},
            "deterministic_evaluation": True,
        }
    return chosen


def promotion_check(selected: Candidate, incumbent: Candidate | None, source: str) -> dict[str, Any]:
    """Decision 15c (criteria v1): strictly higher net-of-costs test return and a test maximum drawdown
    no worse than the incumbent's; a pass still needs the user's approval."""
    out: dict[str, Any] = {"criteria_version": 1, "candidate": selected.name, "candidate_seed": selected.seed, "incumbent": incumbent.name if incumbent else None, "incumbent_source": source, "split": "test", "requires_user_approval": True}
    if incumbent is None or "test" not in incumbent.results:
        return {**out, "result": "not_comparable", "reason": "incumbent_not_evaluated"}
    a, b = selected.metrics("test"), incumbent.metrics("test")
    r1 = a.get("total_return") is not None and b.get("total_return") is not None and a["total_return"] > b["total_return"]
    r2 = a.get("max_drawdown") is not None and b.get("max_drawdown") is not None and a["max_drawdown"] <= b["max_drawdown"]
    return {**out, "candidate_test": {"total_return": a.get("total_return"), "max_drawdown": a.get("max_drawdown")}, "incumbent_test": {"total_return": b.get("total_return"), "max_drawdown": b.get("max_drawdown")}, "r1_net_return_beats_incumbent": bool(r1), "r2_drawdown_not_worse": bool(r2), "result": "pass" if r1 and r2 else "fail"}


def caveats(n_train: int, windows: Mapping[str, tuple[date, date]], mkt: MarketData, rl_section: Mapping[str, Any], *, reused: bool = False) -> list[dict[str, str]]:
    n_test = len([s for s in mkt.sessions if windows["test"][0] <= s <= windows["test"][1]])
    out = [
        {"kind": "thin_rl_training_data", "text": f"The RL policies were trained on {n_train} daily sessions (one calendar year, one decision per trading day). About 250 training days is thin for reinforcement learning: expect large seed-to-seed variance and overfitting to the training year; treat RL results as exploratory, not as evidence of a durable edge."},
        {"kind": "hindsight_and_survivorship", "text": "The research universe was chosen in 2026 knowing that these instruments did well (hindsight selection), and it contains no failed or delisted companies (survivorship). Every family benefits from this choice, so absolute returns overstate what a forward-looking investor could expect; compare families with each other rather than with the market."},
        {"kind": "short_single_test_period", "text": f"The {'reused development' if reused else 'test'} period has {n_test} sessions and is a single market path; a ranking is not statistically significant. Live paper trading afterwards is the forward test."},
        {"kind": "validation_reuse", "text": "Grid points, RL checkpoints and reward configurations were chosen on the same validation period, so validation scores are optimistic. A previously inspected test is also development evidence, not independent confirmation."},
        {"kind": "reward_is_not_performance", "text": "RL training rewards are shaped (risk, drawdown and turnover penalties, scaled) and appear only in the training-reward section; portfolio results come from the common evaluator."},
    ]
    if reused:
        out.append({"kind": "test_reuse", "text": "This market path has already been inspected in prior research runs. It cannot establish an out-of-sample edge or qualify this run for promotion."})
    if any(isinstance(v, Mapping) and v.get("status") != "trained" for v in rl_section.values()):
        out.append({"kind": "rl_time_budget", "text": "At least one RL algorithm or seed was skipped because the job's runtime budget ran out; see the rl section."})
    return out


# ===================================================================== job handler
def run_model_selection(inp: Any) -> dict[str, Any]:
    """``python -m finplan_model.jobs model_selection`` (registered in :mod:`finplan_model.jobs.handlers`)."""
    from finplan_model.jobs.handlers import _bias, _with_bias
    from finplan_model.jobs.results import succeeded_result

    started = time.monotonic()
    spec = inp.spec
    if spec.get("purpose") not in ALLOWED_PURPOSES:
        raise FinplanError.validation("model_selection runs with purpose research or holdout_evaluation", pointer="/purpose")
    section = _bias(inp)
    protocol = validate_protocol(spec.get("selection_protocol"))
    reserve = float(protocol["runtime"]["reserve_seconds"])
    max_runtime = float(spec.get("max_runtime_seconds") or 0)
    deadline = started + max(0.0, max_runtime - reserve) if max_runtime else None
    out = select_models(inp.market, inp.sim_config, protocol=protocol, universe=inp.universe, ctx=inp.ctx, deadline=deadline, incumbent=spec.get("incumbent_strategy"), prior_test_accesses=int(spec.get("test_period_prior_accesses") or 0))
    synthetic = True if inp.ctx.synthetic else None
    refs = []
    policy_rows = []
    for meta, data in out.policies:
        ref = inp.artifacts.put(data, kind="rl_policy", content_type="application/zip", synthetic=synthetic, domain="finance")
        refs.append(ref)
        policy_rows.append({**meta, "artifact_id": ref.artifact_id, "size_bytes": len(data)})
    evidence = {**out.evidence, "policies": policy_rows}
    if section is not None:
        evidence["bias_section"] = section
    refs.insert(0, inp.artifacts.put_json(evidence, kind="run_artifact", synthetic=synthetic, domain="finance"))
    summary = out.summary
    summary["policies"] = policy_rows
    est = (spec.get("cost_estimate") or {}).get("estimated_usd_upper_bound")
    wall = time.monotonic() - started
    compute = summary["compute"]
    compute.update(
        {
            "sagemaker_job": "training",
            "instance_type": (spec.get("compute") or {}).get("instance_type"),
            "instance_count": (spec.get("compute") or {}).get("instance_count", 1),
            "max_runtime_seconds": int(max_runtime) if max_runtime else None,
            "job_wall_seconds": round(wall, 3),
            "estimated_usd_upper_bound": est,
            "estimated_actual_usd": round(float(est) * wall / max_runtime, 6) if est is not None and max_runtime else None,
            "estimate_basis": "upper bound x (wall seconds / max runtime seconds); container start-up and image download are billed too and are not included",
            "budget_category": "cpu_research",
        }
    )
    doc = succeeded_result(inp.ctx, spec, solution_status=out.selected.results["test"].solution_status, artifacts=refs, performance=out.performance, dataset_checksum=inp.market.dataset_checksum, instance_seconds=wall, benchmark=out.test_section)
    doc["payload"]["model_selection"] = summary
    doc = _with_bias(doc, section)
    size = len(canonical_json_bytes(doc))
    if size > MAX_RESULT_BYTES:
        # keep the decision-relevant parts; the full detail is in the evidence artifact
        for block in summary.get("rl", {}).values():
            for g in block.get("grid", []) if isinstance(block, Mapping) else []:
                for ck in g.get("checkpoints", []):
                    ck.pop("curve", None)
        summary["result_trimmed"] = True
    from finplan_model.control.validation import find_storage_location
    from finplan_model.core.outcome import require_valid

    if find_storage_location(doc) is not None:
        raise FinplanError.internal("a job result must not contain storage locations", reason="result_leaks_location")
    return require_valid(doc, "job-result")
