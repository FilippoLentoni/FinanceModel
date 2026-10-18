"""Hermetic current-book Jev control/worker/reuse proof; no real vendor or AWS."""
import copy
import json
from types import SimpleNamespace
import pytest

from finplan_model.benchmarks.forward_jev import (PROTOCOL, checksum, context_identity,
    index_id, journal_id, make_binding, managed_id, run_forward, source_from_snapshot)
from finplan_model.benchmarks.jev import JevClient
from finplan_model.control.auth import Principal
from finplan_model.control.forward_jev import authenticated_operator, verified_binding, submission
from finplan_model.core.artifacts import InMemoryArtifactStore, canonical_json_bytes
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.platform import FixturePlatformClient, build_synthetic_snapshot
from finplan_model.forward.public_preflight import preflight
from finplan_model.forward.approved_jev import ApprovedForwardReader, adapter, current_context
from finplan_model.jobs.handlers import JobInputs
from finplan_model.jobs.market_loader import load_market
from finplan_model.jobs.spec import build_run_spec
from tests.unit.control.support import Harness, SID, APPROVER, arn
from tests.unit.jobs.support import synthetic_payload

CUTOFF='2026-07-11T00:05:00Z'
SLOT='2026-07-11T00:00:00Z'
EXPERIMENT='controlled-forward-jev'
PID='pf_01KDVDNAZ83BAMMYCEGWF33DPM'
ORIGIN='mr_'+'a'*64
INPUT='mi_'+'b'*64
OPERATOR=Principal.from_arn(arn('finplan-devdesktop'))


def fixture():
    platform=FixturePlatformClient()
    record,blobs=build_synthetic_snapshot(synthetic_payload(n_sessions=65),input_snapshot_id=SID,
        created_at='2026-07-10T23:55:00Z',retrieved_at='2026-07-10T23:54:00Z')
    platform.add_snapshot(record,blobs)
    market,content=load_market(platform,SID)
    universe=list(market.instruments)
    state={'positions':[],'cash_balance':100000.,'high_watermark':100000.,'as_of':'2026-07-10','base_currency':'USD','mode':'paper','source':'paper_initialization'}
    holdings={**state,'portfolio_id':PID,'revision':1,'cash':100000.,'cash_weight':1.,'portfolio_value':100000.,'weights':[{'instrument_id':i,'weight':0.} for i in universe]}
    platform.portfolio_states[PID]={'portfolio_id':PID,'revision':1,'synthetic':True,'paper_state':state}
    platform.portfolios[PID]={'portfolio_id':PID,'synthetic':True,'settings':{'isolated_experiment':True,'experiment_id':EXPERIMENT,'approach_id':'jev_systemone'}}
    seed={'source':{'input_snapshot_id':SID,'checksum':content.snapshot.manifest_checksum,'snapshot_created_at':record['created_at'],'investable_universe':universe,'price_basis':'close'}}
    source=source_from_snapshot(content,market,seed)
    base={'run_id':ORIGIN,'input_id':INPUT,'approach_id':'jev_systemone','slot':SLOT,'availability_cutoff':CUTOFF,
        'holdings':holdings,'constraints':{'long_only':True,'max_weight':1.},'code_versions':{'financemodel':'controlled_replay'},'shared_source_checksum':checksum(source)}
    capture=[]
    preflight('jev',base,source,holdings,capture.append,dependency_reader=lambda _: {'status':'controlled_no_live_metadata'})
    captured={**base,'numeric':{'prepared':capture[0]}}
    binding=make_binding(captured,source,experiment_id=EXPERIMENT,captured_checksum=checksum(captured))
    shared_id='mi_'+checksum([EXPERIMENT,'__shared__',SLOT,'shared_source']).split(':')[1]
    class Evidence:
        def read_run_header(self,rid):
            return {'run_id':ORIGIN,'input_id':INPUT,'experiment_id':EXPERIMENT,'approach_id':'jev_systemone','slot':SLOT,'portfolio_id':PID} if rid==ORIGIN else None
        def read_input(self,iid):
            if iid==INPUT:return {'metadata':{'input_id':INPUT,'shared_source_ref':{'input_id':shared_id,'checksum':checksum(source)},'kind':'model_input','experiment_id':EXPERIMENT,'approach_id':'jev_systemone','slot':SLOT,'checksum':checksum(captured)},'payload':copy.deepcopy(captured)}
            assert iid==shared_id
            return {'metadata':{'kind':'shared_source','experiment_id':EXPERIMENT,'slot':SLOT,'checksum':checksum(source)},'payload':copy.deepcopy(source)}
        def get_portfolio(self,pid):
            assert pid==PID
            return {'portfolio_id':PID,'revision':1,'isolated_experiment':True,'experiment_id':EXPERIMENT,'approach_id':'jev_systemone','paper_state':copy.deepcopy(state)}
    evidence=Evidence()
    h=Harness(platform=platform,prices={'retrieved_at':'2026-07-11T00:00:00Z','currency':'USD','usd_per_hour':{'ml.m5.xlarge':.23}})
    h.clock.set('2026-07-11T00:10:00Z');h.deps.artifacts=InMemoryArtifactStore();h.settings.job_definitions['jev_backtest']['forward_protocol']=PROTOCOL;h.settings.job_definitions['jev_backtest']['forward_source_commit']='a'*40
    return SimpleNamespace(platform=platform,market=market,content=content,source=source,base=base,captured=captured,binding=binding,evidence=evidence,h=h)


def submit(f,*,dry_run=False):
    return f.h.service.submit_forward_jev(OPERATOR,f.binding,dry_run=dry_run,correlation_id='controlled-forward-jev',origin_evidence=f.evidence)


def approved_input(f):
    _,out=submit(f)
    f.h.service.approve_run(Principal.from_arn(APPROVER),out['run_id'],{'approved_estimate_usd':out['cost_estimate']['estimated_usd_upper_bound']})
    run=f.h.run(out['run_id'])
    spec=build_run_spec(run,simulation=run['simulation'],image_digest=run['forward_jev']['worker_image_digest'])
    ctx=RunContext(environment='beta',purpose='research',run_id=spec['run_id'],synthetic=True,clock=f.h.clock,image_digest=spec['image_digest'])
    inp=JobInputs(ctx,spec,f.market,f.content,f.h.deps.artifacts,platform=f.platform)
    inp.forward_context_reader=f.h.store.get_forward_context
    return inp


def client_factory(f,seen):
    def factory(inp):
        def transport(payload):
            # Captured request and permanent claim exist BEFORE any simulated transport.
            assert inp.artifacts.resolve(kind='run_artifact',artifact_id=journal_id(f.binding,'claim'))
            assert payload==f.binding['request']
            seen.append(copy.deepcopy(payload))
            return {'model':'controlled-jev-returned-version','answers':{i:{'type':'choice','choice':'buy','confidence':.8,'probabilities':{'buy':.7,'hold':.2,'sell':.1}} for i in payload['questions']},'usage':{'input_tokens':100,'output_tokens':20}}
        return JevClient('synthetic-key-never-persisted',transport=transport,sleep=lambda _:None)
    return factory


def test_dryrun_creates_no_records_claims_jobs_or_approval():
    f=fixture();status,response=submit(f,dry_run=True)
    assert status==200 and response['dry_run']
    assert not f.h.store.runs and not f.h.deps.artifacts.objects and not f.h.created_jobs()
    assert response['cost_estimate']['estimated_usd_upper_bound']==.0675


def test_submit_awaits_independent_human_and_permanent_retry_after_ttl():
    f=fixture();status,response=submit(f)
    assert status==202 and response['state']=='awaiting_approval'
    rid=response['run_id'];assert rid==managed_id(f.binding)
    assert not f.h.created_jobs() and f.h.kicks==0
    with pytest.raises(FinplanError):f.h.service.approve_run(OPERATOR,rid,{})
    with pytest.raises(FinplanError):build_run_spec(f.h.run(rid),simulation=f.h.run(rid)['simulation'],image_digest=None)
    f.h.store.idempotency.clear();f.h.clock.advance(days=90)
    status,retry=submit(f)
    assert status==200 and retry['run_id']==rid and len(f.h.store.runs)==1 and not f.h.created_jobs()


def test_public_submit_cannot_activate_trusted_forward_mode_or_historical_window():
    f=fixture()
    with pytest.raises(FinplanError):f.h.service.submit_job(OPERATOR,submission(f.binding,dry_run=False),correlation_id='controlled')
    body=submission(f.binding,dry_run=False);body['evaluation_window']={'start':'2026-01-05','end':'2026-02-05'}
    from finplan_model.control.validation import validate_submission
    with pytest.raises(FinplanError):validate_submission(body,f.h.cfg,forward_one_decision=True)
    assert not f.h.store.runs


@pytest.mark.parametrize('field', ['origin_run_id','origin_input_id','origin_input_checksum','shared_source_checksum'])
def test_forged_origin_binding_rejected_before_artifact_or_job(field):
    f=fixture();b=copy.deepcopy(f.binding);b[field]='forged-'+field
    with pytest.raises((ValueError,FinplanError)):f.h.service.submit_forward_jev(OPERATOR,b,dry_run=False,correlation_id='controlled',origin_evidence=f.evidence)
    assert not f.h.store.runs and not f.h.deps.artifacts.objects


def test_real_credential_principal_only_exact_beta_development_role():
    cfg=fixture().h.cfg
    session=SimpleNamespace(client=lambda name:SimpleNamespace(get_caller_identity=lambda:{'Arn':arn('finplan-devdesktop')}))
    assert authenticated_operator(session,cfg)==OPERATOR
    session=SimpleNamespace(client=lambda name:SimpleNamespace(get_caller_identity=lambda:{'Arn':arn('finplan-beta-financeagent-runtime-role')}))
    with pytest.raises(FinplanError):authenticated_operator(session,cfg)
    from finplan_model.core.config import load_config
    session=SimpleNamespace(client=lambda name:SimpleNamespace(get_caller_identity=lambda:{'Arn':arn('finplan-devdesktop')}))
    with pytest.raises(FinplanError):authenticated_operator(session,load_config('gamma'))


@pytest.mark.parametrize('change',['revision','cash','isolated','experiment','future_snapshot','expired'])
def test_submit_preconditions_fail_closed(change):
    f=fixture()
    if change=='revision':f.platform.portfolio_states[PID]['revision']=2
    elif change=='cash':f.platform.portfolio_states[PID]['paper_state']['cash_balance']=999.
    elif change=='isolated':f.platform.portfolios[PID]['settings']['isolated_experiment']=False
    elif change=='experiment':f.platform.portfolios[PID]['settings']['experiment_id']='another'
    elif change=='future_snapshot':f.binding['source']['snapshot_created_at']='2099-01-01T00:00:00Z'
    else:f.h.clock.advance(hours=2)
    with pytest.raises((ValueError,FinplanError)):submit(f)
    assert not f.h.store.runs and not f.h.deps.artifacts.objects


def test_worker_exact_public_payload_response_policy_capture_and_no_ledger():
    f=fixture();inp=approved_input(f);seen=[]
    before=copy.deepcopy(f.platform.portfolio_states)
    result=run_forward(inp,client_factory=client_factory(f,seen))
    assert result['completion_status']=='succeeded' and len(seen)==1
    output=inp.artifacts.get_json(result['payload']['forward_jev']['result_ref'])
    original=inp.artifacts.get_json(output['invocation_input_ref'])
    assert original['public_request']==f.binding['request']
    assert output['policy']['forecast']['available'] is False and output['policy']['cash']==pytest.approx(.88)
    assert output['public_response']['model']=='controlled-jev-returned-version'
    assert output['usage']['calls']==1 and not output['ledger_updated']
    assert f.platform.portfolio_states==before
    assert 'synthetic-key-never-persisted' not in json.dumps([json.loads(x[1]) for x in inp.artifacts.objects.values()])


def test_worker_retry_recovers_captured_response_without_second_call_after_expiry(monkeypatch):
    f=fixture();inp=approved_input(f);seen=[]
    # Simulate crash AFTER response sealed but BEFORE a policy/result could be completed.
    import finplan_model.benchmarks.forward_jev as module
    from finplan_model.benchmarks.jev import JevStrategy
    original=JevStrategy.decide
    calls=0
    def crashing(self,*args):
        nonlocal calls
        result=original(self,*args);calls+=1
        if calls==1:raise KeyboardInterrupt('controlled crash after response journal')
        return result
    monkeypatch.setattr(JevStrategy,'decide',crashing)
    with pytest.raises(KeyboardInterrupt):run_forward(inp,client_factory=client_factory(f,seen))
    monkeypatch.setattr(JevStrategy,'decide',original);f.h.clock.advance(hours=3)
    result=run_forward(inp,client_factory=lambda _:pytest.fail('second paid invocation'))
    assert result['completion_status']=='succeeded' and len(seen)==1
    output=inp.artifacts.get_json(result['payload']['forward_jev']['result_ref'])
    assert output['recovered_without_provider_call'] and output['usage']['calls']==1
    claim=inp.artifacts.get_json(output['invocation_claim_ref'])
    assert output['invocation_input_ref']==claim['invocation_input_ref']
    f.h.store.runs[inp.spec['run_id']]['state']='succeeded'
    f.h.store.runs[inp.spec['run_id']]['result']=result
    from finplan_model.forward.managed_costs import terminal_cost_event
    f.h.store.runs[inp.spec['run_id']]['completed_at']=f.h.clock.now().isoformat()
    f.h.store.runs[inp.spec['run_id']]['forward_cost_event']=terminal_cost_event(f.h.store.runs[inp.spec['run_id']])
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    lookup=ApprovedForwardReader(inp.artifacts,f.h.store).lookup(context,'2026-07-11T04:05:00Z')
    assert lookup['status']=='approved_exact_result'


def test_unknown_vendor_side_effect_never_retried():
    f=fixture();inp=approved_input(f);calls=[]
    def crash(_):calls.append(1);raise KeyboardInterrupt('unknown vendor side effect')
    with pytest.raises(KeyboardInterrupt):run_forward(inp,client_factory=crash)
    result=run_forward(inp,client_factory=lambda _:pytest.fail('blind paid retry'))
    assert result['completion_status']=='failed' and calls==[1]
    failure=inp.artifacts.get_json(result['artifacts'][-1])
    assert failure['reason']=='forward_invocation_indeterminate'


@pytest.mark.parametrize('change',['expired','revision','request_corruption','missing_approval'])
def test_worker_ineligible_before_client_factory(change):
    f=fixture();inp=approved_input(f)
    if change=='expired':f.h.clock.advance(hours=2)
    elif change=='revision':f.platform.portfolio_states[PID]['revision']=2
    elif change=='request_corruption':inp.spec['forward_jev']['binding_checksum']='sha256:'+'0'*64
    else:inp.spec['forward_jev']['approval_verified']=False
    if change in ('request_corruption','missing_approval'):
        with pytest.raises(ValueError):run_forward(inp,client_factory=lambda _:pytest.fail('client created'))
    else:
        result=run_forward(inp,client_factory=lambda _:pytest.fail('client created'))
        assert result['completion_status']=='failed'


def completed(f):
    inp=approved_input(f);result=run_forward(inp,client_factory=client_factory(f,[]))
    run=f.h.store.runs[inp.spec['run_id']];run['state']='succeeded';run['result']=result
    from finplan_model.forward.managed_costs import terminal_cost_event
    run['completed_at']=f.h.clock.now().isoformat();run['forward_cost_event']=terminal_cost_event(run)
    return inp,result


def test_hourly_reuses_only_exact_approved_output_and_captures_original_inputs():
    f=fixture();inp,result=completed(f);captured=[]
    reader=ApprovedForwardReader(inp.artifacts,f.h.store)
    invoke=adapter(EXPERIMENT,reader_factory=lambda:reader,dependency_reader=lambda _:{'controlled':True})
    base={**f.base,'run_id':'current-hour-run','input_id':'current-hour-input','availability_cutoff':'2026-07-11T01:05:00Z'}
    out=invoke(base,f.source,f.base['holdings'],captured.append)
    assert out['status']=='reused' and out['usage']['calls']==0 and not out['model_invocation_this_slot']
    assert captured[0]['invocation_state']=='reused_approved_native_daily_signal'
    assert captured[0]['original_preflight_input_id']==INPUT and captured[0]['public_request']['payload']==f.binding['request']
    assert captured[0]['public_response']['model']=='controlled-jev-returned-version'


@pytest.mark.parametrize('change',['unapproved','future','holdings','model','bad_policy','old'])
def test_hourly_rejects_unapproved_future_drift_and_malformed_outputs(change):
    f=fixture();inp,result=completed(f)
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    reader=ApprovedForwardReader(inp.artifacts,f.h.store);cutoff='2026-07-11T01:05:00Z'
    if change=='unapproved':f.h.store.runs[inp.spec['run_id']]['approval']=None
    elif change=='future':cutoff='2026-07-11T00:09:00Z'
    elif change=='holdings':context['holdings']['revision']=2
    elif change=='model':context['request_checksum']='sha256:'+'0'*64
    elif change=='old':cutoff='2026-07-13T01:05:00Z'
    else:
        ref=result['payload']['forward_jev']['result_ref'];stored=inp.artifacts.objects[ref['artifact_id']]
        inp.artifacts.objects[ref['artifact_id']]=(stored[0],stored[1].replace(b'"cash":0.88',b'"cash":0.99'))
    if change in ('holdings','model','old'):
        assert reader.lookup(context,cutoff)['status']!='approved_exact_result'
    else:
        with pytest.raises((ValueError,FinplanError)):reader.lookup(context,cutoff)


def test_no_proposal_remains_original_not_invoked_preflight_without_submission():
    f=fixture();reader=ApprovedForwardReader(f.h.deps.artifacts,f.h.store);captured=[]
    out=adapter(EXPERIMENT,reader_factory=lambda:reader,dependency_reader=lambda _:{'controlled':True})(f.base,f.source,f.base['holdings'],captured.append)
    assert out['status']=='unavailable' and out['usage']['calls']==0
    assert captured[0]['invocation_state']=='not_invoked' and captured[0]['managed_forward_readback']['status']=='no_exact_context_proposal'
    assert not f.h.store.runs and not f.h.deps.artifacts.objects


def test_real_hourly_runtime_preserves_explicit_reused_status_and_original_content():
    f=fixture();inp,result=completed(f);captured=[]
    from finplan_model.forward.runtime import evaluate_approach
    invoke=adapter(EXPERIMENT,reader_factory=lambda:ApprovedForwardReader(inp.artifacts,f.h.store),dependency_reader=lambda _:{'controlled':True})
    out=evaluate_approach('jev_systemone',f.source,f.base['holdings'],cutoff='2026-07-11T01:05:00Z',run_id='current-run',input_id='current-input',capture=captured.append,config={'constraints':f.base['constraints']},external_adapters={'jev':invoke})
    assert out['status']=='reused' and out['usage']['calls']==0
    assert not captured[0]['news']['consumed']
    assert captured[0]['numeric']['prepared']['public_request']['payload']==f.binding['request']


def test_handler_failure_retains_input_artifacts_and_container_returns_failure(monkeypatch):
    f=fixture();inp=approved_input(f)
    f.h.run_io.put_spec(inp.spec['run_id'],inp.spec)
    monkeypatch.setattr('finplan_model.benchmarks.forward_jev._owned_context_reader',lambda _:f.h.store.get_forward_context)
    monkeypatch.setattr('finplan_model.benchmarks.forward_jev._native_client',lambda _:(_ for _ in ()).throw(FinplanError.precondition('controlled failure',reason='controlled_vendor_failure')))
    from finplan_model.jobs.entrypoint import run_job
    code,result=run_job('jev_backtest',inp.spec['run_id'],run_io=f.h.run_io,platform=f.platform,artifacts=inp.artifacts,environment='beta',clock=f.h.clock)
    assert code==1 and result['completion_status']=='failed'
    assert len(result['artifacts'])==2
    assert f.h.run_io.get_result(inp.spec['run_id'])==result
    assert inp.artifacts.get_json(result['artifacts'][0])['public_request']==f.binding['request']


def test_missing_deployed_worker_capability_blocks_submission_but_dryrun_has_no_writes():
    f=fixture();f.h.settings.job_definitions['jev_backtest'].pop('forward_protocol')
    assert submit(f,dry_run=True)[0]==200
    with pytest.raises(FinplanError,match='forward capability'):submit(f)
    assert not f.h.store.runs and not f.h.deps.artifacts.objects


def test_approved_image_drift_fails_before_any_managed_compute():
    f=fixture();inp=approved_input(f)
    f.h.settings.job_definitions['jev_backtest']['image_uri']='registry.invalid/model@sha256:'+'f'*64
    f.h.service.dispatch(hint_run_id=inp.spec['run_id'])
    run=f.h.run(inp.spec['run_id'])
    assert run['state']=='failed' and not f.h.created_jobs()
    assert 'forward_worker_image_changed' in json.dumps(run['error'])


def fresh_binding(f,cutoff='2026-07-11T02:05:00Z'):
    new=copy.deepcopy(f.binding)
    new.update(origin_run_id='mr_'+'c'*64,origin_input_id='mi_'+'d'*64,availability_cutoff=cutoff,scheduled_slot=cutoff[:13]+':00:00Z')
    return new


def test_expired_never_started_context_allows_fresh_separately_approved_identity():
    from finplan_model.control.forward_jev import predecessor
    f=fixture();_,first=submit(f);f.h.clock.advance(hours=2)
    new=fresh_binding(f)
    assert predecessor(f.h.service,new)==first['run_id']
    assert managed_id(new)!=first['run_id'] and not f.h.created_jobs()


def test_permanent_indeterminate_context_is_not_forgotten_after_many_days():
    from finplan_model.control.forward_jev import predecessor
    f=fixture();inp=approved_input(f)
    def unknown(_):raise KeyboardInterrupt('unknown paid boundary')
    with pytest.raises(KeyboardInterrupt):run_forward(inp,client_factory=unknown)
    old=f.h.store.runs[inp.spec['run_id']];old.update(state='failed',job_started=True,attempt=1)
    f.h.clock.advance(days=90)
    with pytest.raises(FinplanError) as error:predecessor(f.h.service,fresh_binding(f,'2026-10-09T02:05:00Z'))
    assert error.value.details['reason']=='forward_context_indeterminate'
    assert f.h.store.get_forward_context(context_identity(f.binding))['managed_run_id']==old['run_id']


def test_verified_no_call_receipt_allows_successor_but_missing_receipt_does_not():
    from finplan_model.control.forward_jev import predecessor
    f=fixture();inp=approved_input(f)
    result=run_forward(inp,client_factory=lambda _:(_ for _ in ()).throw(ValueError('controlled before vendor client')))
    old=f.h.store.runs[inp.spec['run_id']];old.update(state='failed',job_started=True,attempt=1,result=result)
    f.h.clock.advance(hours=2)
    assert predecessor(f.h.service,fresh_binding(f))==old['run_id']
    proof=result['payload']['forward_jev']['failure_ref']
    f.h.deps.artifacts.objects.pop(proof['artifact_id'])
    with pytest.raises(FinplanError):predecessor(f.h.service,fresh_binding(f))


def test_missing_claim_is_never_used_as_proof_of_no_paid_call():
    from finplan_model.control.forward_jev import predecessor
    f=fixture();inp=approved_input(f)
    old=f.h.store.runs[inp.spec['run_id']];old.update(state='failed',job_started=True,attempt=1)
    f.h.clock.advance(days=2)
    with pytest.raises(FinplanError) as error:predecessor(f.h.service,fresh_binding(f,'2026-07-13T02:05:00Z'))
    assert error.value.details['reason']=='forward_context_indeterminate'


def test_expired_forward_human_approval_is_rejected_without_starting_compute():
    f=fixture();_,response=submit(f);f.h.clock.advance(hours=2)
    with pytest.raises(FinplanError) as error:f.h.service.approve_run(Principal.from_arn(APPROVER),response['run_id'],{})
    assert error.value.details['reason']=='forward_request_expired'
    assert f.h.run(response['run_id'])['state']=='cancelled' and not f.h.created_jobs()
    assert f.h.store.query_forward_cost_events(EXPERIMENT)[0][0]['vendor_observed_usd']==0

@pytest.mark.parametrize('fault', ['original_input', 'claim_read', 'preparation'])
def test_unknown_existing_claim_corruption_never_creates_no_call_proof_or_successor(monkeypatch,fault):
    from finplan_model.control.forward_jev import predecessor
    import finplan_model.benchmarks.forward_jev as worker
    f=fixture();inp=approved_input(f)
    with pytest.raises(KeyboardInterrupt):
        run_forward(inp,client_factory=lambda _:(_ for _ in ()).throw(KeyboardInterrupt('unknown paid boundary')))
    original_get=inp.artifacts.get_json
    original_read=worker.read_named
    if fault=='original_input':
        _,claim=original_read(inp.artifacts,journal_id(f.binding,'claim'))
        def get(ref):
            if (ref.to_dict() if hasattr(ref,'to_dict') else ref)['artifact_id']==claim['invocation_input_ref']['artifact_id']:
                raise ValueError('controlled damaged original input')
            return original_get(ref)
        monkeypatch.setattr(inp.artifacts,'get_json',get)
    elif fault=='claim_read':
        def read(store,aid):
            if aid==journal_id(f.binding,'claim'):raise ValueError('controlled unreadable prior claim')
            return original_read(store,aid)
        monkeypatch.setattr(worker,'read_named',read)
    else:
        monkeypatch.setattr(worker,'prepare_native',lambda *a,**k:(_ for _ in ()).throw(ValueError('controlled preparation integrity failure')))
    seen=[];result=run_forward(inp,client_factory=client_factory(f,seen))
    assert seen==[] and result['payload']['forward_jev']['invocation_state']=='indeterminate'
    assert result['payload']['forward_jev']['usage'] is None
    with pytest.raises(FinplanError) as absent:inp.artifacts.resolve(kind='run_artifact',artifact_id=journal_id(f.binding,'no_call'))
    assert absent.value.code=='NOT_FOUND'
    monkeypatch.setattr(worker,'read_named',original_read)
    old=f.h.store.runs[inp.spec['run_id']];old.update(state='failed',job_started=True,attempt=1,result=result)
    f.h.clock.advance(days=2)
    with pytest.raises(FinplanError) as blocked:predecessor(f.h.service,fresh_binding(f,'2026-07-13T02:05:00Z'))
    assert blocked.value.details['reason']=='forward_context_indeterminate'

@pytest.mark.parametrize('field,value',[('worker_image_digest','sha256:'+'f'*64),('worker_source_commit','f'*40)])
def test_reader_requires_original_invocation_worker_identity_equal_to_approved_descriptor(field,value):
    f=fixture();inp,result=completed(f)
    f.h.store.runs[inp.spec['run_id']]['forward_jev'][field]=value
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    with pytest.raises(ValueError,match='worker identity'):
        ApprovedForwardReader(inp.artifacts,f.h.store).lookup(context,'2026-07-11T01:05:00Z')


def test_owned_claim_transport_without_response_is_indeterminate_and_never_retried():
    from finplan_model.control.forward_jev import predecessor
    f=fixture();inp=approved_input(f);calls=[]
    def factory(_):
        def transport(payload):
            calls.append(payload)
            raise ValueError('controlled unknown vendor transport outcome')
        return JevClient('controlled-unused-key',transport=transport,sleep=lambda _:None)
    result=run_forward(inp,client_factory=factory)
    section=result['payload']['forward_jev']
    assert section['invocation_state']=='indeterminate' and section['usage']['transport_calls']>=1
    assert run_forward(inp,client_factory=lambda _:pytest.fail('paid retry'))['payload']['forward_jev']['invocation_state']=='indeterminate'
    old=f.h.store.runs[inp.spec['run_id']];old.update(state='failed',job_started=True,attempt=1,result=result)
    f.h.clock.advance(days=2)
    with pytest.raises(FinplanError) as blocked:predecessor(f.h.service,fresh_binding(f,'2026-07-13T02:05:00Z'))
    assert blocked.value.details['reason']=='forward_context_indeterminate'
    from finplan_model.forward.managed_costs import terminal_cost_event
    old['completed_at']=f.h.clock.now().isoformat()
    assert terminal_cost_event(old)['vendor_observed_usd'] is None
