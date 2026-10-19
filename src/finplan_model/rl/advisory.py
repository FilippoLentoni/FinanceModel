"""Serving orchestration for the pinned advisory ensemble. No automatic promotion or publication."""
from __future__ import annotations

import json

from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.outcome import require_valid
from finplan_model.core.ids import require_id
from finplan_model.jobs.market_loader import load_market
from finplan_model.portfolio_decisions import persist_proposal
from finplan_model.horizon_evaluation import evaluation_contract, evaluation_implementation
from .inference import implementation_identity, recommend, recommend_baseline
from .serving_context import completed_date, raw_market, resolve_inputs, value_holdings, with_quantities


def activate(service, principal, body):
    if service.env != 'beta':
        raise FinplanError.precondition('advisory strategy serving is enabled in beta only',reason='advisory_disabled')
    allowed=(f'finplan-{service.env}-financemodel-pipeline-stage-role', f'finplan-{service.env}-financialplanning-operator')
    if principal.role_name not in service.cfg.dev_desktop_role_names and (not principal.role_name or not any(principal.role_name.startswith(p) for p in allowed)):
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
    previous=service.d.advisory_parameter.read()
    if body.get('portfolio_id'):
        portfolio_id=body['portfolio_id'];require_id('portfolio_id',portfolio_id)
        from finplan_model.paper_profile import approved_profile_instruments
        portfolio=service.d.platform.get_portfolio(portfolio_id)
        if not portfolio.get('settings',{}).get('paper_research_profile'):
            raise FinplanError.precondition('scoped activation requires an explicit isolated paper profile',reason='paper_profile_identity')
        names=approved_profile_instruments(portfolio,portfolio_id=portfolio_id,available_instruments=bundle['instruments'])
        if set(names)!=set(bundle['instruments']):raise FinplanError.precondition('scoped policy and paper profile universes differ',reason='paper_profile_policy_incompatible')
        if not previous:raise FinplanError.precondition('scoped activation preserves an existing default policy',reason='no_advisory_policy')
        saved=json.loads(previous);saved.setdefault('portfolio_policies',{})[portfolio_id]=doc
    else:
        saved=doc
        if previous and json.loads(previous).get('portfolio_policies'):
            saved['portfolio_policies']=json.loads(previous)['portfolio_policies']
    service.store.append_audit({'event':'advisory_policy_selected','audit_id':service.d.ids.correlation_id(),'at':service.ts(),'old':previous,'new':saved,'actor':principal.role_name})
    service.d.advisory_parameter.write(json.dumps(saved,sort_keys=True))
    return {'advisory_policy':doc}


PPO_REQUEST = 'tools/generate-portfolio-recommendation-ppo-request'
PPO_RESPONSE = 'tools/generate-portfolio-recommendation-ppo-response'


def _policy_inventory(service):
    """Every frozen export this service may serve: ``[(artifact_ref, bundle, export_run_id|None)]``.

    The bounded research-storage listing (``artifacts/policy_inference/*``, the same inventory
    the forward runner's direct PPO book selects from) plus the operator-pinned advisory
    policies, deduplicated by artifact checksum. Pinned entries contribute their export run.
    """
    entries={}
    lister=getattr(service.d,'policy_exports',None)
    if lister is not None:
        for ref,bundle in lister():
            entries.setdefault(ref['checksum'],[ref,bundle,None])
    parameter=getattr(service.d,'advisory_parameter',None)
    raw=parameter.read() if parameter is not None else None
    if raw:
        pinned=json.loads(raw)
        for doc in [pinned,*pinned.get('portfolio_policies',{}).values()]:
            if not doc.get('artifact'):continue
            checksum=doc['artifact']['checksum']
            if checksum not in entries:
                entries[checksum]=[doc['artifact'],json.loads(service.d.artifacts.get(doc['artifact'])),None]
            entries[checksum][2]=entries[checksum][2] or doc.get('export_run_id')
    if lister is None and not raw:
        raise FinplanError.precondition('no frozen PPO policy is available in this environment',reason='no_advisory_policy')
    return [tuple(e) for e in entries.values()]


def select_policy(service, instruments):
    """The frozen price-only PPO policy trained on exactly ``instruments`` (typed error otherwise)."""
    from .policy_selection import select_for_universe
    inventory=_policy_inventory(service)
    ref,bundle=select_for_universe([(ref,bundle) for ref,bundle,_ in inventory],instruments)
    export_run_id=next(run for r,_,run in inventory if r['checksum']==ref['checksum'])
    return {'artifact':ref,'export_run_id':export_run_id or bundle['source_run_id'],
            'export_run_id_basis':'operator_pinned_export_run' if export_run_id else 'policy_source_run_id; exported evaluated members carry no separate export run'},bundle


def recommendation(service, body, instruments=None):
    """Serve ``generate_portfolio_recommendation_ppo``: the frozen policy for the requested universe."""
    require_valid(body,PPO_REQUEST)
    if instruments is not None and list(instruments)!=list(body['instruments']):
        raise FinplanError.validation('top-level instruments must equal request.instruments',pointer='/instruments')
    if service.env!='beta':
        raise FinplanError.precondition('advisory strategy serving is unavailable in this environment',reason='advisory_disabled')
    pinned,bundle=select_policy(service,body['instruments'])
    snapshot_id,saved=resolve_inputs(service,body)
    if bundle.get('format') != 'finplan-strategy-bundle/1' or bundle.get('mode') != 'advisory_paper':
        raise FinplanError.precondition('the selected strategy artifact format is unsupported',reason='policy_format_invalid')
    if saved is not None:
        from finplan_model.paper_profile import approved_profile_instruments
        metadata=service.d.platform.get_portfolio(saved['portfolio_id'])
        allowed=approved_profile_instruments(metadata,portfolio_id=saved['portfolio_id'],available_instruments=bundle['instruments'])
        if set(allowed)!=set(bundle['instruments']):
            raise FinplanError.precondition('selected frozen policy is incompatible with the isolated paper profile universe',reason='paper_profile_policy_incompatible')
    elif 'holdings' in body:
        outside=sorted({r['instrument_id'] for r in body['holdings']['weights']}-set(body['instruments']))
        if outside:
            raise FinplanError.validation('holdings reference instruments outside the requested instruments',pointer='/holdings/weights',outside_instruments=outside)
    market,content=load_market(service.d.platform,snapshot_id)
    as_of=completed_date(service,body,market,bundle['instruments'])
    if as_of.isoformat()<=bundle['available_after']:
        raise FinplanError.precondition('advisory decisions must follow training and validation selection',reason='decision_before_selection_end')
    missing=sorted(set(bundle['instruments'])-set(market.instruments))
    if missing:
        raise FinplanError.validation('requested instruments are missing from the snapshot',pointer='/instruments',missing_instruments=missing)
    lookback=int(bundle['environment']['window'])+1 if bundle.get('members') else None
    dates,px=market.view(as_of,bundle['instruments']).price_matrix('close',lookback=lookback)
    if not dates or dates[-1]!=as_of:
        raise FinplanError.precondition('as_of must name the latest aligned completed session in the supplied snapshot',reason='stale_or_incomplete_market_data')
    holdings,metadata=value_holdings(body,saved,raw_market(content,market),as_of,bundle['instruments'])
    news_kwargs={}
    decision_cutoff=market.decision_time(as_of).isoformat()
    if 'etf_news_metadata_v1' in bundle.get('environment',{}).get('features',[]):
        from finplan_model.etf_news import dt
        decision_cutoff=market.decision_time(as_of).isoformat() if 'as_of' in body else service.now().isoformat()
        created=content.snapshot.record.get('created_at')
        if not created or dt(created)>dt(decision_cutoff):raise FinplanError.precondition('price snapshot unavailable at news-policy decision',reason='snapshot_vintage_unavailable')
        from .etf_news import read_partitioned_news
        news_kwargs={'news_dataset':read_partitioned_news(service.d.platform,bundle['instruments'],decision_cutoff),'decision_time':decision_cutoff}
    if {'gdelt_organization_tone_v1','gdelt_intraday_organization_tone_v1'} & set(bundle.get('environment',{}).get('features',[])):
        from .gdelt import read_live
        from finplan_model.etf_news import dt
        if 'as_of' in body:
            raise FinplanError.precondition('historical GDELT serving requires an explicit morning cutoff; frozen decision replay remains available',reason='gdelt_historical_cutoff_required')
        decision_cutoff=service.now().isoformat()
        if 'gdelt_intraday_organization_tone_v1' in bundle['environment']['features']:
            from .gdelt_intraday import read_live,formation_cutoff
            decision_cutoff=formation_cutoff(service.now().isoformat())
        created=content.snapshot.record.get('created_at')
        if not created or dt(created)>dt(decision_cutoff):raise FinplanError.precondition('price snapshot unavailable at the decision cutoff',reason='snapshot_vintage_unavailable')
        source,_,_=read_live(service.d.platform,bundle['instruments'],decision_cutoff)
        news_kwargs={'news_dataset':source,'decision_time':decision_cutoff}
    captured_inputs=[]
    def capture_actual(prepared):
        from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
        payload={'schema_version':'finplan-served-actual-input/1','captured_at':service.now().isoformat(),
                 'availability_cutoff':service.now().isoformat(),'native_data_decision_time':decision_cutoff,
                 'input_snapshot_id':snapshot_id,'snapshot_checksum':content.snapshot.manifest_checksum,
                 'snapshot_created_at':content.snapshot.record.get('created_at'),'policy_artifact':pinned['artifact'],
                 'policy_configuration_id':bundle['configuration_id'],'prepared':prepared}
        checksum=sha256_checksum(canonical_json_bytes(payload))
        # This serving principal owns immutable classical/* evidence, and has an
        # explicit deny on DynamoDB. The caller's platform activity receipt
        # indexes the exact pre-actor payload and these verified references.
        writer=getattr(service.d,'capture_served_input',None)
        if writer:
            ref=writer(payload)
        elif hasattr(service,'issue'):
            ref=service.issue('served_model_input',{'input_payload':payload},
                {'input_checksum':checksum,'policy_artifact':pinned['artifact']})['analysis_ref']
        else:
            raise FinplanError.dependency_unavailable('actual model input storage is unavailable',reason='input_capture_unavailable')
        captured_inputs.append({'payload':payload,'artifact':ref,'checksum':checksum})
    try:
        rec=recommend(bundle,px,holdings,**news_kwargs,input_capture=capture_actual) if bundle.get('members') else recommend_baseline(bundle,market,as_of,holdings,input_capture=capture_actual)
    except Exception as exc:
        from finplan_model.core.errors import as_finplan_error
        error=as_finplan_error(exc)
        error.details['actual_model_input_refs']=[entry['artifact'] for entry in captured_inputs]
        error.details['actual_model_inputs']=captured_inputs
        raise error from exc
    rec.setdefault('diagnostics',{})['actual_model_inputs']=captured_inputs
    rec['diagnostics']['policy_selection']={'rule':'price_only_ppo_exact_universe; selected ensemble first (finplan_model.rl.policy_selection)',
        'requested_instruments':list(body['instruments']),'policy_instruments':list(bundle['instruments']),
        'policy_artifact':pinned['artifact'],'export_run_id_basis':pinned['export_run_id_basis']}
    with_quantities(rec,metadata)
    rec.update({'as_of':as_of.isoformat(),'input_snapshot_id':snapshot_id,'snapshot_checksum':content.snapshot.manifest_checksum,'policy_artifact_checksum':pinned['artifact']['checksum'],'export_run_id':pinned['export_run_id'],'decision_timing':'after_completed_close_for_next_session','bias_disclosures':content.snapshot.record.get('bias_disclosures',[])})
    rec['decision_at']=decision_cutoff
    rec['serving_time']=service.now().isoformat()
    if news_kwargs:rec['news_timing']='eligible_news_at_decision_with_completed_prices'
    provenance = {
        'implementation': implementation_identity(bundle),
        'policy_artifact': pinned['artifact'],
        'export_run_id': pinned['export_run_id'],
        'policy_source_run_id': bundle['source_run_id'],
        'configuration_id': bundle['configuration_id'],
        'policy_inputs': {'prices': px.tolist(), 'holdings': holdings, 'environment': bundle.get('environment'), 'instruments': bundle['instruments'],**news_kwargs},
        'snapshot_checksum': content.snapshot.manifest_checksum,
        'decision_time': decision_cutoff,
        'paper_execution_cost_bps': 2.0,
        'evaluation_contract': evaluation_contract(bundle['strategy_id'], bundle=bundle),
        'evaluation_implementation': evaluation_implementation(bundle),
    }
    decision_id = persist_proposal(service, rec, saved, provenance, idempotency_key=body.get('idempotency_key'))
    result = {'recommendation':rec,'synthetic':bool(market.synthetic or saved is not None)}
    if decision_id:
        result['decision_id'] = decision_id
    return result
