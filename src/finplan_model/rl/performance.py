"""Deterministic published-allocation accounting evidence and bounded diagnostic feedback.

An execution intent is not an account statement. Missing ledger data fails explicitly. The
unattributed execution component is never presented as proof of a particular real-world cause.
"""
from __future__ import annotations

from datetime import date
import math

from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import require_valid
from finplan_model.jobs.market_loader import load_market

TOL=1e-6


def decompose(publication, version, executions, market, request):
    if publication['plan_version_id']!=version['plan_version_id'] or publication['plan_version_checksum']!=version['checksum']:
        raise FinplanError.precondition('published version checksum does not match',reason='publication_checksum_mismatch')
    window=request['window']
    ledgers=[e for e in executions if e.get('mode')==request['actual_source'] and e.get('ledger',{}).get('payload',{}).get('window')==window]
    if len(ledgers)!=1:
        raise FinplanError.precondition('exactly one recorded account ledger covering this window is required; execution intent alone cannot explain actual returns',reason='actual_ledger_missing_or_ambiguous')
    execution=ledgers[0]; actual=dict(execution['ledger']['payload']); actual.pop('synthetic',None); actual.pop('window'); actual.pop('positions_note',None)
    residual=actual['end_value']-(actual['start_value']+actual['cash_flows']+actual['realized_pnl']+actual['unrealized_pnl']-actual['fees'])
    if not math.isfinite(residual) or abs(residual)>TOL:
        raise FinplanError.validation('the recorded account ledger does not reconcile',pointer='/executions/ledger',residual=residual)
    content=version.get('content',{})
    content=content.get('payload',content)
    allocation=content['allocation']
    dates,px=market.view(date.fromisoformat(window['end']),[r['instrument_id'] for r in allocation['weights']]).price_matrix('close')
    wanted=[date.fromisoformat(window[k]) for k in ('start','end')]
    if any(d not in dates for d in wanted):
        raise FinplanError.precondition('both window boundaries need completed, aligned market data',reason='window_not_covered')
    a,b=[dates.index(d) for d in wanted]
    if b<=a:
        raise FinplanError.validation('performance window must span at least two completed sessions',pointer='/window')
    n=actual['start_value']
    contributions=[{'instrument_id':r['instrument_id'],'pnl':n*float(r['weight'])*(float(px[b,k]/px[a,k])-1)} for k,r in enumerate(allocation['weights'])]
    gross=sum(r['pnl'] for r in contributions)
    # Publication contains fees per trade, not a forecast of turnover over an arbitrary window.
    # Hold the published allocation: no new modeled rebalances or fees inside the stated window.
    planned={'start_value':n,'cash_flows':actual['cash_flows'],'realized_pnl':0.,'unrealized_pnl':gross,'fees':0.,'end_value':n+actual['cash_flows']+gross}
    execution_gap=actual['realized_pnl']+actual['unrealized_pnl']-gross
    cost_gap=-actual['fees']
    total=actual['end_value']-planned['end_value']
    identifiers={'publication_id':publication['publication_id'],'plan_version_id':version['plan_version_id'],'plan_version_checksum':version['checksum'],'execution_ids':sorted(e['execution_id'] for e in executions if e.get('mode')==request['actual_source']),'realized_snapshot_id':request['realized_snapshot_id']}
    whys=[{'level':1,'question':'Where did the observed account outcome differ from holding the published allocation?','answer':'gross execution/exposure difference and recorded costs','execution_gap':execution_gap,'cost_gap':cost_gap},
          {'level':2,'question':'What explains the gross execution/exposure difference?','answer':'not_available: dated fills, position history and cash-flow timing are needed to distinguish allocation, timing and corporate-action effects','status':'unresolved'},
          {'level':2,'question':'What is known about the cost difference?','answer':'recorded account fees versus a no-new-trades allocation-hold baseline','recorded_fees':actual['fees']}]
    feedback=[{'action':'capture_dated_fills_positions_and_cash_flows','reason':'resolve the gross execution/exposure component before changing policy features','automatic':False},
              {'action':'audit_fee_slippage_and_turnover_assumptions','reason':'compare observed costs with the simulator on the same execution schedule','automatic':False},
              {'action':'evaluate_on_fresh_forward_paper_windows','reason':'validate PPO against minimum variance before any production promotion','automatic':False}]
    return {'evidence_kind':'performance_decomposition','identifiers':identifiers,'window':window,'actual_source':request['actual_source'],
            'reconciliation':{'tolerance':TOL,'plan_path':planned,'executed_path':actual},
            'gap':{'total':total,'components':{'execution':execution_gap,'cost':cost_gap,'market_vs_forecast':None,'residual':residual},'not_available':['market_vs_forecast'],'tolerance':TOL},
            'forecast':{'status':'not_available'},'data_quality':{'flags':[],'revisions':[]},'excluded_days':[],
            'instrument_contributions':contributions,'whys':whys,'feedback':feedback,
            'baseline':'published_allocation_hold_no_new_trades','cash_flow_timing':'not_available' if actual['cash_flows'] else 'no_external_flows',
            'limitations':['execution_gap_unattributed_without_fill_history','no_published_return_forecast','allocation_hold_is_not_daily_policy_replay']}


def performance_evidence(service,body):
    require_valid(body,'tools/get-performance-evidence-request')
    platform=service.d.platform
    pub=platform._json('GET',f"/v1/publications/{body['publication_id']}")
    version_doc=platform._json('GET',f"/v1/plan-versions/{pub['plan_version_id']}")
    version=version_doc.get('plan_version',version_doc)
    records=[]; token=None
    for _ in range(10):
        page=platform._json('GET',f"/v1/publications/{body['publication_id']}/executions",{'page_size':100,'next_token':token})
        records.extend(page['executions']);token=page.get('next_token')
        if not token:break
    if token:raise FinplanError.precondition('execution history exceeds the bounded evidence request',reason='execution_history_too_large')
    market,content=load_market(platform,body['realized_snapshot_id'])
    evidence=decompose(pub,version,records,market,body)
    evidence['data_quality']['flags']=[{'snapshot_id':body['realized_snapshot_id'],'flag':f} for f in content.snapshot.quality_flags]
    evidence['data_quality']['flags'] += [{'snapshot_id':version.get('input_snapshot_id'),'flag':'input_snapshot_not_compared_for_revisions'}]
    ref=service.d.artifacts.put_json(evidence,kind='explanation_evidence',domain='finance',synthetic=True if market.synthetic else None)
    return {'evidence':evidence,'evidence_ref':ref.to_dict(),'synthetic':market.synthetic}
