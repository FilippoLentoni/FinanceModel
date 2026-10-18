"""Metadata-first pending paper reconciliation with a verified durable frontier.

Immutable model inputs/outputs remain untouched. A frontier explicitly retains at
most one pending policy; every other covered older policy must be terminal. No
forecast query or forecast horizon is changed by this execution bookkeeping.
"""
from __future__ import annotations
import socket,os,json
from datetime import UTC,datetime,timedelta
from .research import stamp

POLICIES=frozenset(('succeeded','hold','reused','pending_execution'))


def _parse(value):return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(UTC)


def _attempt(backend,experiment,aid,slot,run,at,reason,status='execution_pending'):
    from .hourly import identifier
    oid=identifier('mo',experiment,aid,slot,'execution_attempt')
    existing=backend.repo.get('portfolio_observation',oid)
    if existing:
        # Preserve the original attempted quote/cutoff on same-slot retry.
        original=json.loads(backend.read_artifact_bytes(existing.doc)) if 'payload' not in existing.doc else existing.doc['payload']
        if original.get('run_id')!=run['run_id']:raise ValueError('existing execution attempt belongs to a different original policy')
        return
    backend.record_observation(experiment,aid,slot,run['portfolio_id'],{
        'run_id':run['run_id'],'status':status,'reason':reason,'actual_fills':[],
        'observed_at':at,'value':None,'observation_kind':'pending','quote_status':'missing'},at,kind='execution_attempt')


def recover_expired_run(backend,header,at):
    """Recover only the captured attempt; never repeat inference or invent a value."""
    current=backend.read_run_header(header['run_id'])
    if current['status']!='running':return {'status':'already_terminal'}
    if current.get('lease_until',0)>_parse(at).timestamp():
        return {'status':'running_other_owner','next_retry_at':datetime.fromtimestamp(current['lease_until'],UTC).isoformat().replace('+00:00','Z')}
    owner=socket.gethostname()+':'+str(os.getpid())
    if not backend.claim_run(current['run_id'],owner,at,lease_seconds=600):
        return {'status':'running_other_owner'}
    outcome={'status':'failed','reason':'recovered_interrupted_invocation_original_input_preserved',
             'forecast':{'available':False},'recovery_at':at,'execution_status':'not_applicable'}
    captured=backend.read_input(current['input_id'])
    original=captured['payload'];holdings=original.get('holdings') or {}
    from .hourly import identifier
    oid=identifier('mo',current['experiment_id'],current['approach_id'],current['slot'],'valuation')
    if not backend.repo.get('portfolio_observation',oid):
        backend.record_observation(current['experiment_id'],current['approach_id'],current['slot'],current['portfolio_id'],{
            'run_id':current['run_id'],'input_id':current['input_id'],'portfolio_id':current['portfolio_id'],
            'revision':holdings.get('revision',current['portfolio_revision']),'value':None,'cash':None,'holdings':holdings.get('positions',[]),
            'observed_at':at,'value_as_of':original.get('availability_cutoff',current['started_at']),
            'observation_kind':'missing','quote_status':'not_observed','decision_status':'failed','execution_status':'not_applicable',
            'reason':'Interrupted invocation recovered; no original portfolio valuation was observed. Captured holdings remain original evidence.',
            'original_input_checksum':captured['metadata']['checksum'],'net_external_cash_flows_since_decision':0},at)
    backend.finish_run(current['run_id'],outcome,at)
    return {'status':'recovered_failed','run_id':current['run_id']}


def reconcile_executions(backend,experiment,source,at,slot,comparison_policy,*,apply,config,approach_ids=None,max_pages=100,allow_execution=True):
    registry=backend.read_registry(experiment)
    approach_ids=approach_ids or [a['approach_id'] for a in (registry or {}).get('approaches',[])]
    declared=(registry or {}).get('comparison_policy') or {}
    first_slot=declared.get('first_scheduled_slot');cadence=declared.get('cadence_seconds',3600)
    summaries=[];failures=[];incomplete=[]
    for aid in approach_ids:
        try:
            headers=[];cursor=None;prior_frontier=None;inherited_pending=None;complete=False
            for _ in range(max_pages):
                page,cursor=backend.query_run_headers(experiment,aid,slot,page_size=8 if not headers else 100,next_token=cursor)
                for header in page:
                    headers.append(header)
                    if header.get('execution_frontier_ref'):
                        proof=backend.read_execution_frontier(header['run_id'])
                        prior_frontier=header['run_id'];inherited_pending=proof.get('pending_run_id');complete=True;break
                if complete or not cursor:
                    complete=True;break
            if not complete:
                raise RuntimeError('bounded pending execution header query exhausted; frontier not advanced')
            # The newest expected head is strongly checked even if the GSI has
            # not propagated it. The backend verifies every covered older slot.
            prior_slot=_parse(slot)-timedelta(seconds=cadence)
            if first_slot and prior_slot>=_parse(first_slot):
                from .hourly import identifier
                expected=backend.read_run_header(identifier('mr',experiment,aid,prior_slot.isoformat().replace('+00:00','Z')))
                if expected['run_id'] not in {h['run_id'] for h in headers}:headers.insert(0,expected)
            if not headers:continue
            if inherited_pending and inherited_pending not in {h['run_id'] for h in headers}:
                headers.append(backend.read_run_header(inherited_pending))
            # Strong leases can change independently of the GSI. Recover expired
            # attempts before a frontier can certify no orphaned running heads.
            active=[]
            for header in headers:
                if header['status']!='running':continue
                recovered=recover_expired_run(backend,header,at)
                header.update(backend.read_run_header(header['run_id']))
                if recovered['status']=='running_other_owner' and header['status']=='running':active.append({'run_id':header['run_id'],**recovered})
            # Manual validation before the declared horizon and late-created
            # historical policies never gain automatic trading opportunities.
            for header in headers:
                if header['status'] not in POLICIES or header.get('execution_ref') or header.get('execution_disposition_ref'):continue
                if first_slot and (_parse(header['slot'])<_parse(first_slot) or _parse(header['started_at'])>=_parse(header['slot'])+timedelta(seconds=cadence)):
                    backend.mark_execution_ineligible(header['run_id'],stamp())
                    header.update(backend.read_run_header(header['run_id']))
            pending_headers=[h for h in headers if h['status'] in POLICIES and not h.get('execution_ref') and not h.get('execution_disposition_ref')]
            pending_headers.sort(key=lambda h:(_parse(h['slot']),h['run_id']),reverse=True)
            selected=None
            for header in pending_headers:
                current=backend.read_run_header(header['run_id'])
                if current['status'] not in POLICIES or current.get('execution_ref') or current.get('execution_disposition_ref'):continue
                candidate=backend.read_run(current['run_id'])
                if not (candidate.get('outcome') or {}).get('target_weights'):
                    raise RuntimeError('completed pending policy lacks original validated target weights')
                selected=candidate;break
            if active:
                incomplete.extend({'approach_id':aid,**row} for row in active)
                summaries.append({'approach_id':aid,'status':'incomplete_running_lease'});continue
            # Oldest-first is essential: after a crash, no newer disposition may
            # imply an older unresolved policy was already closed.
            if selected:
                older=[h for h in pending_headers if (_parse(h['slot']),h['run_id'])<(_parse(selected['slot']),selected['run_id'])]
                for header in reversed(older):
                    try:backend.mark_execution_superseded(header['run_id'],selected['run_id'],stamp())
                    except Exception:
                        refreshed=backend.read_run_header(header['run_id'])
                        if refreshed['status'] not in ('applied','no_op','superseded','execution_excluded'):raise
                book=backend.get_portfolio(selected['portfolio_id'])
                if not allow_execution:
                    pass  # Late retry repairs bookkeeping only, never fills after inputs froze.
                elif book['revision']!=selected['portfolio_revision']:
                    _attempt(backend,experiment,aid,slot,selected,at,'captured_portfolio_revision_changed; no policy rebase or fill')
                else:
                    opportunity=(_parse(slot)+timedelta(minutes=5)).isoformat().replace('+00:00','Z')
                    result=apply(selected['outcome'],book['paper_state'],source.get('current_quotes',{}),
                        decision_at=selected.get('finished_at',selected['started_at']),opportunity_at=opportunity,recorded_at=at,
                        config=config,universe=source['investable_universe'],execution_calendar=comparison_policy.get('market_calendar','XNYS'),
                        quote_protocol=comparison_policy.get('quote_protocol','native_completed_1m_unadjusted_close/1'),
                        max_quote_age_seconds=comparison_policy.get('execution_price_freshness_seconds',120))
                    if result['status'] in ('applied','no_op'):
                        backend.commit_paper_step(selected['run_id'],book['revision'],result['state'],result['fills'],result['execution'],stamp())
                    else:_attempt(backend,experiment,aid,slot,selected,at,result.get('reason','eligible_execution_price_unavailable'))
                latest=backend.read_run_header(selected['run_id'])
                pending_id=selected['run_id'] if latest['status'] in POLICIES and not latest.get('execution_ref') and not latest.get('execution_disposition_ref') else None
            else:pending_id=None
            frontier=backend.record_execution_frontier(headers[0]['run_id'],pending_id,stamp(),previous_frontier_run_id=prior_frontier)
            summaries.append({'approach_id':aid,'status':'execution_frontier_verified','pending_run_id':frontier.get('pending_run_id'),
                              'frontier_run_id':headers[0]['run_id'],'queried_headers':len(headers)})
        except Exception as exc:
            failures.append({'approach_id':aid,'error_type':type(exc).__name__,'reason':str(exc)})
            incomplete.append({'approach_id':aid,'status':'reconciliation_failed','reason':str(exc)})
    return {'approaches':summaries,'failures':failures,'incomplete':incomplete}
