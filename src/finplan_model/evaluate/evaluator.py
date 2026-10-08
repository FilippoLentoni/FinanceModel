"""The common evaluator: one entry point for every strategy family (spec paper-execution-simulator).

* :func:`evaluate` runs a strategy through the :class:`~finplan_model.sim.engine.Simulator` and
  returns an :class:`EvaluationResult` stamped with the :class:`EvaluatorIdentity` (evaluator
  version, image digest, dataset identity, simulation configuration and cost-model identity) and
  a deterministic SHA-256 ``result_checksum`` over trades, decisions, NAV and metrics.
* :func:`assert_comparable` is the benchmark comparison guard (SIM-01): runs with different
  evaluator versions, datasets, constraint sets, fee models, execution timing or liquidity models
  are refused with ``VALIDATION_FAILED`` listing the differing settings.
* :func:`replay` re-runs stored strategy outputs through the same evaluator and requires identical
  trades and metrics (SIM-08).

Every strategy family (controls, classical optimizers and later RL, LLM swarm and Jev strategies)
must be evaluated through :func:`evaluate`; it is the only place results are produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.context import EVALUATOR_VERSION, RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.sim import metrics as m
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.engine import SimulationResult, Simulator
from finplan_model.sim.market import MarketData
from finplan_model.sim.strategy import ReplayStrategy, Strategy

__all__ = ["EvaluationResult", "EvaluatorIdentity", "assert_comparable", "evaluate", "replay"]


@dataclass(frozen=True)
class EvaluatorIdentity:
    evaluator_version: str
    image_digest: str | None
    dataset_id: str | None
    dataset_checksum: str | None
    simulation_configuration_id: str
    cost_model_id: str
    settings: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_version": self.evaluator_version,
            "image_digest": self.image_digest,
            "dataset_id": self.dataset_id,
            "dataset_checksum": self.dataset_checksum,
            "simulation_configuration_id": self.simulation_configuration_id,
            "cost_model_id": self.cost_model_id,
            "settings": dict(self.settings),
        }

    @classmethod
    def of(cls, config: SimulationConfig, market: MarketData, *, evaluator_version: str = EVALUATOR_VERSION, image_digest: str | None = None) -> "EvaluatorIdentity":
        return cls(evaluator_version, image_digest, market.dataset_id, market.dataset_checksum, config.configuration_id, config.cost_model_id, config.comparability_settings())

    def comparison_fields(self) -> dict[str, Any]:
        """Named settings compared by the guard. ``image_digest`` is part of the evaluator identity."""
        return {
            "evaluator_version": self.evaluator_version,
            "image_digest": self.image_digest,
            "dataset_id": self.dataset_id,
            "dataset_checksum": self.dataset_checksum,
            **dict(self.settings),
        }


@dataclass
class EvaluationResult:
    strategy: str
    identity: EvaluatorIdentity
    simulation: SimulationResult
    metrics: dict[str, Any]
    synthetic: bool
    evaluation_window: dict[str, str]
    result_checksum: str = ""
    model_version: str | None = None
    run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def solution_status(self) -> str:
        return self.simulation.solution_status

    def canonical_content(self) -> dict[str, Any]:
        """The numeric evidence covered by ``result_checksum`` (no wall-clock values)."""
        return {
            "strategy": self.strategy,
            "identity": self.identity.to_dict(),
            "evaluation_window": self.evaluation_window,
            "metrics": self.metrics,
            "simulation": self.simulation.to_dict(),
            "synthetic": self.synthetic,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.canonical_content()
        d.update(result_checksum=self.result_checksum, solution_status=self.solution_status)
        if self.model_version:
            d["model_version"] = self.model_version
        if self.run_id:
            d["run_id"] = self.run_id
        return d


def _metrics(sim: SimulationResult, cfg: SimulationConfig, risk_free_annual: float | None) -> dict[str, Any]:
    net = [row["value"] for row in sim.nav]
    gross = [row["gross_value"] for row in sim.nav]
    rf = cfg.cash_rate_annual if risk_free_annual is None else risk_free_annual
    s = sim.summary
    return {
        "net_cumulative_return": m.cumulative_return([cfg.initial_cash, *net]),
        "gross_cumulative_return": m.cumulative_return([cfg.initial_cash, *gross]),
        "annualized_volatility": m.annualized_volatility(net, cfg.periods_per_year),
        "sharpe_ratio": m.sharpe_ratio(net, cfg.periods_per_year, rf),
        "sharpe_risk_free_source": "configured_cash_rate" if risk_free_annual is None else "caller_supplied",
        "max_drawdown": m.max_drawdown([cfg.initial_cash, *net]),
        "turnover": s["turnover"],
        "total_fees": s["total_fees"],
        "total_spread_cost": s["total_spread_cost"],
        "total_slippage_cost": s["total_slippage_cost"],
        "total_transaction_costs": s["total_costs"],
        "n_fills": s["n_fills"],
        "n_decisions": s["n_decisions"],
    }


def evaluate(
    strategy: Strategy,
    market: MarketData,
    config: SimulationConfig,
    *,
    ctx: RunContext | None = None,
    start: date | None = None,
    end: date | None = None,
    universe: Sequence[str] | None = None,
    risk_free_annual: float | None = None,
) -> EvaluationResult:
    """Evaluate one strategy over ``[start, end]`` (inclusive) with the common simulator."""
    identity = EvaluatorIdentity.of(config, market, evaluator_version=ctx.evaluator_version if ctx else EVALUATOR_VERSION, image_digest=ctx.image_digest if ctx else None)
    sim = Simulator(config, market, universe=universe).run(strategy, start=start, end=end)
    res = EvaluationResult(
        strategy=strategy.name,
        identity=identity,
        simulation=sim,
        metrics=_metrics(sim, config, risk_free_annual),
        synthetic=bool(market.synthetic),
        evaluation_window={"start": sim.sessions[0], "end": sim.sessions[-1]},
        model_version=ctx.model_version if ctx else None,
        run_id=ctx.run_id if ctx else None,
    )
    res.result_checksum = sha256_checksum(canonical_json_bytes(res.canonical_content()))
    if ctx is not None:
        ctx.log("evaluation_completed", strategy=strategy.name, result_checksum=res.result_checksum, solution_status=res.solution_status, synthetic=res.synthetic)
    return res


def assert_comparable(results: Sequence[EvaluationResult | EvaluatorIdentity]) -> None:
    """Refuse a comparison of runs whose evaluator settings differ (SIM-01)."""
    identities = [r.identity if isinstance(r, EvaluationResult) else r for r in results]
    if len(identities) < 2:
        return
    base = identities[0].comparison_fields()
    differing: set[str] = set()
    for ident in identities[1:]:
        other = ident.comparison_fields()
        for key in sorted(set(base) | set(other)):
            if canonical_json_bytes(base.get(key)) != canonical_json_bytes(other.get(key)):
                differing.add(key)
    if differing:
        raise FinplanError.validation("benchmark comparison refused: runs used different evaluator settings", pointer="/runs", differing_settings=sorted(differing))


def replay(stored: EvaluationResult | Mapping[str, Any], market: MarketData, config: SimulationConfig, *, ctx: RunContext | None = None) -> EvaluationResult:
    """Re-run stored strategy outputs; raise ``INTERNAL`` if trades or metrics differ (SIM-08)."""
    content = stored.canonical_content() if isinstance(stored, EvaluationResult) else dict(stored)
    sim = content["simulation"]
    outputs = {d["decision_session"]: d["strategy_output"] for d in sim["decisions"]}
    window = content["evaluation_window"]
    universe = sorted({i for o in outputs.values() for i in o["weights"]}) or None
    again = evaluate(ReplayStrategy(content["strategy"], outputs), market, config, ctx=ctx, start=date.fromisoformat(window["start"]), end=date.fromisoformat(window["end"]), universe=universe)
    new = again.canonical_content()
    mismatched = [k for k in ("metrics", "simulation", "identity") if canonical_json_bytes(new[k]) != canonical_json_bytes(content[k])]
    if mismatched:
        raise FinplanError.internal("replay of stored strategy outputs does not reproduce the stored result", reason="replay_mismatch", sections=mismatched)
    return again
