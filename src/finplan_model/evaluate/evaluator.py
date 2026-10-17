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
from finplan_model.sim.market import HoldingsView, MarketData
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
    initial_book: HoldingsView | None = None,
) -> EvaluationResult:
    """Evaluate one strategy over ``[start, end]`` (inclusive) with the common simulator."""
    identity = EvaluatorIdentity.of(config, market, evaluator_version=ctx.evaluator_version if ctx else EVALUATOR_VERSION, image_digest=ctx.image_digest if ctx else None)
    sim = Simulator(config, market, universe=universe).run(strategy, start=start, end=end, initial_book=initial_book)
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


def evaluate_point_rebalance(
    target: Mapping[str, Any], *, positions: Mapping[str, float], cash: float,
    quotes: Mapping[str, Mapping[str, Any]], config: SimulationConfig,
    decision_at: str, opportunity_at: str, recorded_at: str,
    universe: Sequence[str], max_quote_age_seconds: float = 120.0,
    execution_calendar: str = "XNYS", quote_protocol: str = "native_completed_1m_unadjusted_close/1",
) -> dict[str, Any]:
    """One forward paper rebalance on genuine native quotes, sharing the daily fill kernel.

    No minute observations become daily Bar objects. The previous hourly policy can
    execute only when actual quotes were captured at the next declared opportunity.
    Quoted, available and executed times remain distinct, preventing hindsight fills.
    """
    import math
    import os
    from pathlib import Path
    from finplan_model.core.clock import parse_utc
    from finplan_model.sim.engine import ExecutionKernel, _Book
    from finplan_model.sim import engine as execution_engine
    from finplan_model.sim.constraints import apply_constraints
    from finplan_model.sim.strategy import coerce_target, FALLBACK_STATUSES, NO_EFFECT
    protocols = {"XNYS": ("native_completed_1m_unadjusted_close/1", "next_hourly_regular_session_quote", "actual_shares_per_quote_minute"),
                 "CRYPTO_24X7": ("native_completed_1h_coinbase_close/1", "next_hourly_crypto_completed_quote", "actual_base_asset_units_per_completed_hour")}
    if execution_calendar not in protocols or quote_protocol != protocols[execution_calendar][0]:
        raise FinplanError.validation("unsupported calendar/quote protocol pair", pointer="/quote_protocol")
    _, execution_rule, liquidity_units = protocols[execution_calendar]
    if config.liquidity.unfilled!='cancel':raise FinplanError.validation('point evaluator requires explicit cancel remainders; no implicit carry',pointer='/liquidity/unfilled')
    ids=tuple(sorted(universe))
    if len(ids)!=len(set(ids)) or not ids:raise FinplanError.validation('point evaluation needs distinct universe',pointer='/universe')
    decision,opportunity,recorded=map(parse_utc,(decision_at,opportunity_at,recorded_at))
    if not decision<opportunity<=recorded:raise FinplanError.precondition('point execution must follow its declared opportunity and actual decision',reason='execution_time_not_eligible')
    if not math.isfinite(cash) or cash<0 or set(positions)-set(ids) or any(not math.isfinite(q) or q<0 for q in positions.values()):raise FinplanError.validation('invalid point book',pointer='/positions')
    policy=coerce_target(target,ids)
    for i in ids:
        q=quotes.get(i)
        if q is None or q.get('status')!='fresh':raise FinplanError.precondition('fresh actual quotes required for point execution',reason='fresh_execution_quote_missing',instrument=i)
        price=q.get('price');volume=q.get('volume')
        if isinstance(price,bool) or not isinstance(price,(int,float)) or not math.isfinite(price) or price<=0:raise FinplanError.validation('invalid point quote price',pointer='/quotes')
        available,quoted=map(parse_utc,(q['available_at'],q['quoted_at']))
        bar_started=parse_utc(q.get('bar_started_at',q['quoted_at']))
        if (execution_calendar=='XNYS' and bar_started<decision) or not decision<quoted<=available<=recorded or available<opportunity or not 0<=(recorded-quoted).total_seconds()<=max_quote_age_seconds:
            raise FinplanError.precondition('execution quote was not actually captured in its eligible window',reason='missed_eligible_execution_price',instrument=i)
        if execution_calendar == 'XNYS':
            regular=q.get('regular_session',{})
            if not q.get('market_open') or not regular.get('start') or not regular.get('end') or not parse_utc(regular['start'])<=recorded<parse_utc(regular['end']) or quoted<parse_utc(regular['start']):
                raise FinplanError.precondition('off-session point fills forbidden',reason='market_closed',instrument=i)
        elif q.get('venue_calendar') != 'CRYPTO_24X7' or q.get('frequency') != '1h' or q.get('provider') != 'Coinbase Exchange public completed candles' or (quoted-bar_started).total_seconds() != 3600:
            raise FinplanError.precondition('crypto point requires actual complete native hourly candle provenance',reason='incompatible_crypto_quote',instrument=i)
        if (config.liquidity.participation_cap is not None or config.slippage.model!='none') and (isinstance(volume,bool) or not isinstance(volume,(int,float)) or not math.isfinite(volume) or volume<=0):
            raise FinplanError.precondition('point liquidity requires actual native quote volume',reason='execution_volume_missing',instrument=i)
    prices={i:float(quotes[i]['price']) for i in ids}
    initial=cash+sum(positions.get(i,0.)*prices[i] for i in ids)
    if not math.isfinite(initial) or initial<=0:raise FinplanError.validation('point portfolio value must be positive',pointer='/cash')
    weights={i:positions.get(i,0.)*prices[i]/initial for i in ids}
    constrained=apply_constraints(policy.weights,policy.cash,config.constraints,current=weights,current_cash=cash/initial,
                                  policy=config.constraint_policy,tol=config.weight_tolerance)
    book=_Book(float(cash),{i:float(positions.get(i,0.)) for i in ids})
    kernel=ExecutionKernel(config);fills=[];unfilled=[]
    executable=constrained.weights is not None and policy.solution_status not in (*FALLBACK_STATUSES,NO_EFFECT)
    orders={i:constrained.weights.get(i,0.)*initial/prices[i]-book.shares[i] for i in ids} if executable else {}
    quantities={}
    for i,quantity in orders.items():
        capped=kernel._cap(quantity,quotes[i].get('volume'))
        quantities[i]=capped
        if abs(quantity-capped)>1e-12:unfilled.append({'instrument_id':i,'quantity':quantity-capped,'reason':'liquidity_cap','handling':config.liquidity.unfilled})
    for i in ids:
        if quantities.get(i,0)<0:fills.append(kernel._fill(book,recorded.date().isoformat(),decision.date().isoformat(),i,quantities[i],prices[i],quotes[i].get('volume')))
    buys=[(i,quantity) for i,quantity in quantities.items() if quantity>0]
    variable=0.
    for i,quantity in buys:
        _,fee,spread,slippage=kernel._cost_rates(quantity,quotes[i].get('volume'))
        variable+=quantity*prices[i]*(1+fee+spread+slippage)
    fixed=config.fees.fixed_per_trade*len(buys)
    scale=1.
    if variable+fixed>book.cash:scale=max(0.,min(1.,(book.cash-fixed)/variable))*(1.-1e-12) if variable>0 else 0.
    for i,quantity in buys:
        if scale<1:unfilled.append({'instrument_id':i,'quantity':quantity*(1-scale),'reason':'cash_limited','handling':'cancel'})
        if quantity*scale>0:fills.append(kernel._fill(book,recorded.date().isoformat(),decision.date().isoformat(),i,quantity*scale,prices[i],quotes[i].get('volume')))
    total_cost=sum(fill.cost for fill in fills)
    final=book.cash+sum(book.shares[i]*prices[i] for i in ids)
    if book.cash < -config.reconciliation_tolerance or not math.isclose(initial,final+total_cost,rel_tol=1e-10,abs_tol=config.reconciliation_tolerance):
        raise FinplanError.internal('point execution reconciliation break',reason='reconciliation_break')
    result={'schema_version':'finplan-point-evaluation/1','evaluator_version':EVALUATOR_VERSION,'point_evaluator_version':'1.0.0',
            'simulation_configuration_id':sha256_checksum(canonical_json_bytes({**config.to_dict(),'execution_timing':execution_rule})),
            'execution_rule':execution_rule,'configuration':{**config.to_dict(),'execution_timing':execution_rule},'cost_model_id':config.cost_model_id,
            'decision_at':decision_at,'opportunity_at':opportunity_at,'execution_at':recorded_at,
            'positions':book.shares,'cash':book.cash,'before_value':initial,'after_value':final,'transaction_cost':total_cost,
            'fills':[fill.to_dict() for fill in fills],'unfilled':unfilled,'constraint_outcome':constrained.to_dict(),
            'turnover':sum(fill.notional for fill in fills)/initial,'quote_provenance':{i:dict(quotes[i]) for i in ids},
            'reference_prices':prices,'reference_date':recorded.date().isoformat(),'market_session_valid':True,
            'transaction_cost_bps':config.fees.proportional_bps,'reconciliation_status':'reconciled',
            'identity':{'evaluator':'finplan-common-point/1','base_evaluator_version':EVALUATOR_VERSION,
                'model_source_commit':os.environ.get('FINPLAN_MODEL_SOURCE_COMMIT','not_supplied'),
                'source_checksum':sha256_checksum(Path(__file__).read_bytes()),
                'execution_kernel_checksum':sha256_checksum(Path(execution_engine.__file__).read_bytes()),
                'execution_timing':execution_rule,'quote_protocol':quote_protocol,'execution_calendar':execution_calendar,
                'max_quote_age_seconds':max_quote_age_seconds,'liquidity_units':liquidity_units,
                'cost_model_id':config.cost_model_id,'constraint_set':config.constraints.to_dict()}}
    result['result_checksum']=sha256_checksum(canonical_json_bytes(result))
    return result
