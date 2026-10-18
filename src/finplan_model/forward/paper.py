"""Incremental paper step through the existing common evaluator; no hourly invented bars."""
from __future__ import annotations
from datetime import datetime, time
from zoneinfo import ZoneInfo
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.evaluate import evaluate
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import HoldingsView,MarketData
from finplan_model.sim.strategy import ReplayStrategy


def apply_next_session(policy,state,market,*,decision_at,observed_at,config=None,universe=None):
    if not market.synthetic:
        return {'status':'pending_market_data','reason':'daily_replay_not_eligible_forward_execution; use actual forward-captured quote evaluator','fills':[]}
    cutoff=datetime.fromisoformat(observed_at.replace('Z','+00:00'))
    decision=datetime.fromisoformat(decision_at.replace('Z','+00:00'))
    ids=tuple(universe or market.instruments)
    later=[s for s in market.sessions if datetime.combine(s,time(9,30),tzinfo=ZoneInfo('America/New_York'))>decision]
    prior=[s for s in market.sessions if not later or s<min(later)]
    if not prior or not later:return {'status':'pending_market_data','reason':'next completed eligible execution session unavailable','fills':[]}
    first,last=max(prior),min(later)
    bars=[market.bar(i,last) for i in ids]
    if any(b is None or b.available_at>cutoff or b.open is None or b.volume is None for b in bars):
        return {'status':'pending_market_data','reason':'next-session raw open/close/volume not yet actually available','fills':[]}
    if any(market.bar(i,first) is None for i in ids):return {'status':'pending_market_data','reason':'predecision raw marks missing','fills':[]}
    # Retain the actual decision timestamp, even when the market is closed that day.
    selected=MarketData([first,last],[market.bar(i,s) for s in (first,last) for i in ids],
        decision_times={first:decision,last:market.decision_time(last)},synthetic=market.synthetic,
        dataset_id=market.dataset_id,dataset_checksum=market.dataset_checksum)
    quantities={r['instrument_id']:float(r['quantity']) for r in state['positions']}
    marks={i:market.bar(i,first).close for i in ids}
    cash=float(state['cash_balance']);initial=cash+sum(quantities.get(i,0)*marks[i] for i in ids)
    pre_execution=cash+sum(quantities.get(i,0)*market.bar(i,last).open for i in ids)
    cfg=SimulationConfig.from_dict({**(config or {}),'initial_cash':initial,'rebalance_frequency':'daily'})
    output={'weights':{r['instrument_id']:r['weight'] for r in policy['target_weights']},'cash':policy['cash_weight'],
            'solution_status':'no_effect' if policy.get('status')=='hold' else policy.get('solution_status','feasible')}
    result=evaluate(ReplayStrategy('forward-policy',{first.isoformat():output}),selected,cfg,
                    initial_book=HoldingsView(quantities,cash,marks,initial),universe=ids)
    fills=[f.to_dict() for f in result.simulation.fills]
    for fill in fills:quantities[fill['instrument_id']]=quantities.get(fill['instrument_id'],0)+fill['quantity']
    nav=result.simulation.nav[-1]
    final={**state,'positions':[{'instrument_id':i,'quantity':quantities.get(i,0)} for i in ids],
           'cash_balance':nav['cash'],'high_watermark':max(state['high_watermark'],*(n['value'] for n in result.simulation.nav)),
           'as_of':last.isoformat(),'source':'paper_rebalance'}
    return {'status':'applied' if fills else 'no_op','state':final,'fills':fills,'valuation':nav,
            'execution':{'market_session_valid':True,'execution_at':datetime.combine(last,time(9,30),tzinfo=ZoneInfo('America/New_York')).isoformat(),
                         'evaluator_version':result.identity.evaluator_version,'quote_provenance':{'dataset_id':market.dataset_id,'dataset_checksum':market.dataset_checksum},
                         'reference_prices':{i:market.bar(i,last).open for i in ids},'reference_date':last.isoformat(),'transaction_cost_bps':cfg.fees.proportional_bps,
                         'transaction_cost':result.metrics['total_transaction_costs'],'before_value':pre_execution,'after_value':nav['cash']+sum(quantities.get(i,0)*market.bar(i,last).open for i in ids),'closing_value':nav['value']},
            'common_evaluator_result':result.to_dict(),'checksum':sha256_checksum(canonical_json_bytes(result.to_dict()))}


def apply_forward_quotes(policy,state,quote_bundle,*,decision_at,opportunity_at,recorded_at,config,universe,execution_calendar="XNYS",quote_protocol="native_completed_1m_unadjusted_close/1",max_quote_age_seconds=120,book_created_at=None,execution_timing=None):
    """Apply a policy to an actual fresh quote at its declared opportunity.

    Default: a prior-hour policy at the next hourly quote. ``execution_timing=
    "same_run_regular_session_quote"``: the policy at its own run's quote.
    """
    from finplan_model.evaluate.evaluator import evaluate_point_rebalance
    from finplan_model.core.errors import FinplanError
    quotes={q['instrument_id']:q for q in quote_bundle.get('quotes',[])}
    from .corporate_actions import unreconciled_actions
    actions=unreconciled_actions(state,{'current_quotes':quote_bundle},recorded_at,book_created_at=book_created_at)
    if actions:return {'status':'pending_market_data','action_status':'PENDING_ACTION_DATA','reason':'pending_corporate_action_reconciliation','fills':[],'corporate_action_evidence':actions}
    target={'weights':{r['instrument_id']:r['weight'] for r in policy['target_weights']},'cash':policy['cash_weight'],
            'solution_status':'no_effect' if policy.get('status')=='hold' else policy.get('solution_status','feasible')}
    try:
        result=evaluate_point_rebalance(target,positions={r['instrument_id']:r['quantity'] for r in state['positions']},
            cash=state['cash_balance'],quotes=quotes,config=SimulationConfig.from_dict(config),decision_at=decision_at,
            opportunity_at=opportunity_at,recorded_at=recorded_at,universe=universe,execution_calendar=execution_calendar,quote_protocol=quote_protocol,max_quote_age_seconds=max_quote_age_seconds,
            **({'execution_timing':execution_timing} if execution_timing else {}))
    except FinplanError as exc:
        return {'status':'pending_market_data','reason':exc.details.get('reason',exc.message),'fills':[]}
    new_state={**state,'positions':[{'instrument_id':i,'quantity':q} for i,q in result['positions'].items()],
               'cash_balance':result['cash'],'high_watermark':max(state['high_watermark'],result['after_value']),
               'as_of':recorded_at[:10],'source':'paper_rebalance'}
    return {'status':'applied' if result['fills'] else 'no_op','state':new_state,'fills':result['fills'],'execution':result,
            'common_evaluator_result':result,'valuation':{'value':result['after_value'],'cash':result['cash']}}
