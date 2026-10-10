"""Serving orchestration for the pinned advisory ensemble. No automatic promotion or publication."""
from __future__ import annotations

import json

from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.outcome import require_valid
from finplan_model.core.ids import require_id
from finplan_model.jobs.market_loader import load_market
from finplan_model.portfolio_decisions import persist_proposal
from .inference import implementation_identity, recommend, recommend_baseline
from .serving_context import completed_date, raw_market, resolve_inputs, value_holdings, with_quantities


def activate(service, principal, body):
    if service.env != 'beta':
        raise FinplanError.precondition('advisory strategy serving is enabled in beta only',reason='advisory_disabled')
    allowed=(f'finplan-{service.env}-financemodel-pipeline-stage-role', f'finplan-{service.env}-financialplanning-operator')
    if not principal.role_name or not any(principal.role_name.startswith(p) for p in allowed):
        raise FinplanError(ErrorCode.FORBIDDEN,'only the environment operator may pin an advisory policy')
    if body.get('confirmed_by_user') is not True:
        raise FinplanError.precondition('pinning an advisory policy requires user intent',reason='confirmation_required')
    run_id=body.get('export_run_id'); require_id('run_id',run_id)
    run=service._get(run_id)
    if run['state']!='succeeded' or run['job_type']!='prepare_policy':
        raise FinplanError.precondition('a succeeded policy export is required',reason='policy_export_missing')
    result=service.d.run_io.get_result(run_id) or {}
    refs=[a for a in result.get('artifacts',[]) if a['kind']=='policy_inference']
    if len(refs)!=1:
        raise FinplanError.precondition('policy export artifact is missing',reason='policy_export_missing')
    bundle=json.loads(service.d.artifacts.get(refs[0]))
    if bundle.get('mode')!='advisory_paper' or bundle.get('format') != 'finplan-strategy-bundle/1' or (bundle.get('strategy_id') in ('ppo','sac') and len(bundle.get('members',[]))!=5):
        raise FinplanError.precondition('only a complete frozen research strategy may be pinned',reason='policy_members_missing')
    doc={'export_run_id':run_id,'artifact':refs[0],'source_run_id':bundle['source_run_id'],'strategy_id':bundle['strategy_id'],'mode':'advisory_paper','set_at':service.ts(),'set_by':principal.role_name}
    service.store.append_audit({'event':'advisory_policy_selected','audit_id':service.d.ids.correlation_id(),'at':service.ts(),'old':service.d.advisory_parameter.read(),'new':doc,'actor':principal.role_name})
    service.d.advisory_parameter.write(json.dumps(doc,sort_keys=True))
    return {'advisory_policy':doc}


def recommendation(service, body):
    require_valid(body,'tools/recommend-portfolio-invocation-request')
    if service.env!='beta' or service.d.advisory_parameter is None:
        raise FinplanError.precondition('advisory strategy serving is unavailable in this environment',reason='advisory_disabled')
    raw=service.d.advisory_parameter.read()
    if not raw:
        raise FinplanError.precondition('no advisory policy is pinned',reason='no_advisory_policy')
    pinned=json.loads(raw)
    bundle=json.loads(service.d.artifacts.get(pinned['artifact']))
    if bundle.get('format') != 'finplan-strategy-bundle/1' or bundle.get('mode') != 'advisory_paper':
        raise FinplanError.precondition('the selected strategy artifact format is unsupported',reason='policy_format_invalid')
    snapshot_id,saved=resolve_inputs(service,body)
    market,content=load_market(service.d.platform,snapshot_id)
    as_of=completed_date(service,body,market,bundle['instruments'])
    if as_of.isoformat()<=bundle['available_after']:
        raise FinplanError.precondition('advisory decisions must follow training and validation selection',reason='decision_before_selection_end')
    missing=set(bundle['instruments'])-set(market.instruments)
    if missing:
        raise FinplanError.precondition('snapshot does not cover the trained universe',reason='universe_not_covered')
    lookback=int(bundle['environment']['window'])+1 if bundle.get('members') else None
    dates,px=market.view(as_of,bundle['instruments']).price_matrix('close',lookback=lookback)
    if not dates or dates[-1]!=as_of:
        raise FinplanError.precondition('as_of must name the latest aligned completed session in the supplied snapshot',reason='stale_or_incomplete_market_data')
    holdings,metadata=value_holdings(body,saved,raw_market(content,market),as_of,bundle['instruments'])
    rec=recommend(bundle,px,holdings) if bundle.get('members') else recommend_baseline(bundle,market,as_of,holdings)
    with_quantities(rec,metadata)
    rec.update({'as_of':as_of.isoformat(),'input_snapshot_id':snapshot_id,'snapshot_checksum':content.snapshot.manifest_checksum,'policy_artifact_checksum':pinned['artifact']['checksum'],'export_run_id':pinned['export_run_id'],'decision_timing':'after_completed_close_for_next_session','bias_disclosures':content.snapshot.record.get('bias_disclosures',[])})
    provenance = {
        'implementation': implementation_identity(),
        'policy_artifact': pinned['artifact'],
        'export_run_id': pinned['export_run_id'],
        'policy_source_run_id': bundle['source_run_id'],
        'configuration_id': bundle['configuration_id'],
        'policy_inputs': {'prices': px.tolist(), 'holdings': holdings, 'environment': bundle.get('environment'), 'instruments': bundle['instruments']},
        'snapshot_checksum': content.snapshot.manifest_checksum,
        'decision_time': market.decision_time(as_of).isoformat(),
        'paper_execution_cost_bps': 2.0,
    }
    decision_id = persist_proposal(service, rec, saved, provenance, idempotency_key=body.get('idempotency_key'))
    result = {'recommendation':rec,'synthetic':bool(market.synthetic or saved is not None)}
    if decision_id:
        result['decision_id'] = decision_id
    return result
