"""Controlled fake-backend forward Qwen; no GPU, weights, approval or AWS used."""
import copy
import json
from types import SimpleNamespace
import pytest
from finplan_model.benchmarks.forward_qwen import (PROTOCOL,checksum,make_binding,managed_id,run_forward,journal_id,context_identity)
from finplan_model.control.forward_qwen import submission
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.artifacts import LocalArtifactStore,canonical_json_bytes
from finplan_model.control.auth import Principal
from finplan_model.jobs.spec import build_run_spec
from finplan_model.jobs.handlers import JobInputs
from finplan_model.forward.public_preflight import preflight
from finplan_model.forward.approved_qwen import ApprovedForwardReader,adapter,current_context
from tests.unit.benchmarks.test_forward_jev import fixture as jev_fixture,OPERATOR,EXPERIMENT,PID,SLOT,ORIGIN,INPUT
from tests.unit.control.support import APPROVER


def fixture():
    f=jev_fixture();f.base['approach_id']='qwen_swarm_mode_a';f.platform.portfolios[PID]['settings']['approach_id']='qwen_swarm_mode_a'
    captures=[];preflight('qwen',f.base,f.source,f.base['holdings'],captures.append,dependency_reader=lambda _: {'status':'controlled_no_gpu'})
    f.captured={**f.base,'numeric':{'prepared':captures[0]}}
    f.binding=make_binding(f.captured,f.source,experiment_id=EXPERIMENT,captured_checksum=checksum(f.captured))
    shared_id='mi_'+checksum([EXPERIMENT,'__shared__',SLOT,'shared_source']).split(':')[1]
    class Evidence:
        def read_run_header(self,rid):
            return {'run_id':ORIGIN,'input_id':INPUT,'experiment_id':EXPERIMENT,'approach_id':'qwen_swarm_mode_a','slot':SLOT,'portfolio_id':PID} if rid==ORIGIN else None
        def read_input(self,iid):
            if iid==INPUT:return {'metadata':{'input_id':INPUT,'shared_source_ref':{'input_id':shared_id,'checksum':checksum(f.source)},'kind':'model_input','experiment_id':EXPERIMENT,'approach_id':'qwen_swarm_mode_a','slot':SLOT,'checksum':checksum(f.captured)},'payload':copy.deepcopy(f.captured)}
            assert iid==shared_id
            return {'metadata':{'kind':'shared_source','experiment_id':EXPERIMENT,'slot':SLOT,'checksum':checksum(f.source)},'payload':copy.deepcopy(f.source)}
        def get_portfolio(self,pid):
            return {'portfolio_id':PID,'revision':1,'isolated_experiment':True,'experiment_id':EXPERIMENT,'approach_id':'qwen_swarm_mode_a'}
    f.evidence=Evidence();f.h.settings.job_definitions['swarm_mode_a'].update(forward_protocol=PROTOCOL,forward_source_commit='b'*40,code_checksum='sha256:'+'c'*64)
    f.h.settings.prices['usd_per_hour']['ml.g6.12xlarge']=1.
    return f


def submit(f,dry=False):
    return f.h.service.submit_forward_qwen(OPERATOR,f.binding,dry_run=dry,correlation_id='controlled-qwen',origin_evidence=f.evidence)


def approved(f):
    _,response=submit(f);rid=response['run_id']
    f.h.service.approve_run(Principal.from_arn(APPROVER),rid,{'approved_estimate_usd':response['cost_estimate']['estimated_usd_upper_bound']})
    run=f.h.run(rid);spec=build_run_spec(run,simulation=run['simulation'],image_digest=run['forward_qwen']['worker_image_digest'])
    ctx=RunContext(environment='beta',purpose='research',run_id=rid,synthetic=True,clock=f.h.clock,image_digest=spec['image_digest'])
    inp=JobInputs(ctx,spec,f.market,f.content,f.h.deps.artifacts,platform=None)
    inp.offline_code_checksum=spec['forward_qwen']['worker_code_checksum']
    return inp


class Backend:
    def __init__(self,f,inp,*,revise=False,fail_at=None):self.f,self.inp,self.revise,self.fail_at=f,inp,revise,fail_at;self.seen=[]
    def __call__(self,messages):
        index=len(self.seen);store=self.inp.artifacts
        ref=store.resolve(kind='run_artifact',artifact_id=journal_id(self.f.binding,'role_'+str(index)+'_input'))
        captured=store.get_json(ref);assert captured['messages']==messages and captured['messages_checksum']==checksum(messages)
        if index:
            prev=store.resolve(kind='run_artifact',artifact_id=journal_id(self.f.binding,'role_'+str(index-1)+'_response'));assert store.get_json(prev)['received_at']<=captured['captured_at']
        self.seen.append(copy.deepcopy(messages))
        if self.fail_at==index:raise RuntimeError('controlled_backend_failure')
        role=captured['role'];state=json.loads(messages[1]['content'])['evidence']
        if role=='market_analyst':return '{"summary":"actual controlled market output"}'
        if role=='risk_analyst':return '{"summary":"actual controlled risk output"}'
        assert state['market_summary']=='actual controlled market output' and state['risk_summary']=='actual controlled risk output'
        if role=='allocator':return json.dumps({'weights':{i:.5/len(self.f.market.instruments) for i in self.f.market.instruments},'cash':.5})
        if role=='critic':return json.dumps({'revise':self.revise,'reason':'controlled revision'})
        return json.dumps({'choice':'revised' if self.revise else 'original'})


def finish(f,inp,doc):
    run=f.h.run(inp.spec['run_id'])
    f.h.store.update_run({**run,'state':'succeeded','revision':run['revision']+1,'result':doc,
        'completed_at':f.h.clock.now().isoformat(),'forward_cost_event':{'event_id':'controlled-cost'}},run['revision'],[])


def test_dryrun_no_records_or_compute_and_public_forward_rejected():
    f=fixture();status,response=submit(f,True)
    assert status==200 and response['dry_run'] and not f.h.store.runs and not f.h.deps.artifacts.objects
    with pytest.raises(FinplanError):f.h.service.submit_job(OPERATOR,submission(f.binding,dry_run=False),correlation_id='controlled')
    assert not f.h.store.runs


def test_independent_gpu_approval_permanent_id_and_context_family_isolation():
    f=fixture();status,response=submit(f);rid=response['run_id']
    assert status==202 and response['state']=='awaiting_approval' and f.h.run(rid)['compute_class']=='gpu'
    with pytest.raises(FinplanError):f.h.service.approve_run(OPERATOR,rid,{})
    assert f.h.store.get_forward_context(context_identity(f.binding))['managed_run_id']==rid
    f.h.store.idempotency.clear();f.h.clock.advance(days=90)
    assert submit(f)[1]['run_id']==rid and len(f.h.store.runs)==1
    assert context_identity(f.binding)!=context_identity(jev_fixture().binding)


@pytest.mark.parametrize('field',['origin_input_checksum','origin_run_id','request_checksum','native_sizing','holdings'])
def test_original_binding_tamper_never_creates_artifacts(field):
    f=fixture();b=copy.deepcopy(f.binding);b[field]='forged'
    with pytest.raises((ValueError,TypeError,KeyError,FinplanError)):
        f.h.service.submit_forward_qwen(OPERATOR,b,dry_run=False,correlation_id='controlled',origin_evidence=f.evidence)
    assert not f.h.store.runs and not f.h.deps.artifacts.objects


@pytest.mark.parametrize('revise,expected',[ (False,5),(True,6)])
def test_native_actual_current_book_role_capture_and_recovery_no_gpu(revise,expected):
    f=fixture();inp=approved(f);before=copy.deepcopy(f.platform.portfolio_states)
    backend=Backend(f,inp,revise=revise);doc=run_forward(inp,backend=backend)
    assert doc['completion_status']=='succeeded' and len(backend.seen)==expected
    assert f.platform.portfolio_states==before and 'evaluation_window' not in inp.spec
    output=inp.artifacts.get_json(doc['payload']['forward_qwen']['result_ref'])
    assert output['policy']['cash']==.5 and output['ledger_updated'] is False
    assert len(output['role_receipts'])==expected
    f.h.clock.advance(hours=3)
    assert run_forward(inp,backend=lambda _:pytest.fail('no repeat model'))==doc


def test_partial_role_failure_retains_exact_input_and_blocks_retry_and_successor():
    f=fixture();inp=approved(f);backend=Backend(f,inp,fail_at=2)
    result=run_forward(inp,backend=backend)
    assert result['completion_status']=='failed' and len(backend.seen)==3
    section=result['payload']['forward_qwen'];failure=inp.artifacts.get_json(section['failure_ref'])
    assert failure['invocation_state']=='indeterminate' and len(failure['role_receipts'])==2
    assert any(inp.artifacts.get_json(ref).get('call_index')==2 and inp.artifacts.get_json(ref).get('kind')=='role_input' for ref in result['artifacts'])
    assert run_forward(inp,backend=lambda _:pytest.fail('unknown role never rerun'))==result
    run=f.h.run(inp.spec['run_id']);f.h.store.update_run({**run,'state':'failed','job_started':True,'attempt':1,'revision':run['revision']+1,'result':result},run['revision'],[])
    f.h.clock.advance(days=90)
    from finplan_model.control.forward_qwen import predecessor
    b=copy.deepcopy(f.binding);b['scheduled_slot']='2026-10-09T00:00:00Z';b['availability_cutoff']='2026-10-09T00:05:00Z'
    with pytest.raises(FinplanError,match='indeterminate'):predecessor(f.h.service,b)


def test_claim_without_response_is_indeterminate_never_reexecuted():
    f=fixture();inp=approved(f);backend=Backend(f,inp)
    # Simulate hard loss after initial capture by raising a process-level exception.
    class Crash(BaseException):pass
    def crash(messages):backend(messages);raise Crash()
    with pytest.raises(Crash):run_forward(inp,backend=crash)
    assert len(backend.seen)==1
    doc=run_forward(inp,backend=lambda _:pytest.fail('never reinvoke claimed unknown role'))
    assert doc['completion_status']=='failed' and doc['payload']['forward_qwen']['invocation_state']=='indeterminate'


def test_completed_original_reader_and_hourly_adapter_preserve_all_role_inputs():
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp));finish(f,inp,doc)
    reader=ApprovedForwardReader(inp.artifacts,f.h.store)
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    lookup=reader.lookup(context,'2026-07-11T00:20:00Z')
    assert lookup['status']=='approved_exact_result' and len(lookup['role_evidence'])==5
    captured=[];base={**f.base,'availability_cutoff':'2026-07-11T00:20:00Z'}
    outcome=adapter(EXPERIMENT,reader_factory=lambda:reader,dependency_reader=lambda _:{'status':'controlled'})(base,f.source,base['holdings'],captured.append)
    assert outcome['status']=='reused' and outcome['usage']['calls']==0
    assert len(captured[0]['actual_role_evidence'])==5 and captured[0]['model_consumed_inputs'] is True
    assert captured[0]['original_preflight_input_id']==INPUT


@pytest.mark.parametrize('field',['worker_image_digest','worker_source_commit','approval','role_receipts'])
def test_reader_rejects_wrong_worker_approval_or_missing_actual_role_inputs(field):
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp));finish(f,inp,doc)
    ref=doc['payload']['forward_qwen']['result_ref'];output=inp.artifacts.get_json(ref);output[field]=[] if field=='role_receipts' else 'forged'
    new=inp.artifacts.put_json(output,kind='run_artifact');doc['payload']['forward_qwen']['result_ref']=new.to_dict();doc['artifacts'].append(new.to_dict())
    run=f.h.run(inp.spec['run_id']);f.h.store.update_run({**run,'result':doc,'revision':run['revision']+1},run['revision'],[])
    reader=ApprovedForwardReader(inp.artifacts,f.h.store);context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    with pytest.raises(ValueError):reader.lookup(context,'2026-07-11T00:20:00Z')


def test_local_role_capture_fsynced_before_call(tmp_path,monkeypatch):
    f=fixture();inp=approved(f);old=inp.artifacts;inp.artifacts=LocalArtifactStore(tmp_path/'artifacts')
    ref=inp.spec['forward_qwen']['binding_ref'];inp.artifacts.put(old.get(ref),kind='run_artifact',artifact_id=ref['artifact_id'])
    import finplan_model.benchmarks.forward_qwen as module
    calls=[];original=module.os.fsync
    def fsync(fd):calls.append(fd);return original(fd)
    monkeypatch.setattr(module.os,'fsync',fsync)
    backend=Backend(f,inp);doc=run_forward(inp,backend=backend)
    assert doc['completion_status']=='succeeded' and len(calls)>=24


def test_expiry_checked_between_roles_and_no_future_signal_consumed():
    f=fixture();inp=approved(f);native=Backend(f,inp)
    def backend(messages):
        text=native(messages)
        if len(native.seen)==1:f.h.clock.advance(hours=2)
        return text
    doc=run_forward(inp,backend=backend)
    assert doc['completion_status']=='failed' and len(native.seen)==1
    assert doc['payload']['forward_qwen']['invocation_state']=='indeterminate'


@pytest.mark.parametrize('change',['revision','cash','source','expired'])
def test_trusted_submission_rejects_changed_current_book_source_and_expiry(change):
    f=fixture()
    if change=='revision':f.platform.portfolio_states[PID]['revision']=2
    elif change=='cash':f.platform.portfolio_states[PID]['paper_state']['cash_balance']=999.
    elif change=='source':f.source['checksum']='sha256:'+'d'*64
    else:f.h.clock.advance(hours=2)
    with pytest.raises((FinplanError,ValueError)):submit(f)
    assert not f.h.store.runs and not f.h.deps.artifacts.objects


def test_reuse_changed_book_or_source_cannot_select_old_model_result():
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp));finish(f,inp,doc)
    reader=ApprovedForwardReader(inp.artifacts,f.h.store)
    for field in ('revision','cash'):
        holdings=copy.deepcopy(f.base['holdings']);holdings[field]+=1
        context=current_context(f.base,f.source,holdings,f.captured['numeric']['prepared'],EXPERIMENT)
        assert reader.lookup(context,'2026-07-11T00:20:00Z')['status']=='no_exact_context_proposal'
    source=copy.deepcopy(f.source);source['checksum']='sha256:'+'d'*64
    context=current_context(f.base,source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    assert reader.lookup(context,'2026-07-11T00:20:00Z')['status']=='no_exact_context_proposal'


def test_native_invalid_allocator_hold_is_honest_inspectable_result():
    f=fixture();inp=approved(f);normal=Backend(f,inp)
    def invalid(messages):
        result=normal(messages)
        role=inp.artifacts.get_json(inp.artifacts.resolve(kind='run_artifact',artifact_id=journal_id(f.binding,'role_'+str(len(normal.seen)-1)+'_input')))['role']
        return '{}' if role=='allocator' else result
    doc=run_forward(inp,backend=invalid)
    assert doc['completion_status']=='succeeded' and doc['solution_status']=='no_effect' and len(normal.seen)==4
    finish(f,inp,doc);reader=ApprovedForwardReader(inp.artifacts,f.h.store)
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    assert len(reader.lookup(context,'2026-07-11T00:20:00Z')['role_evidence'])==4


@pytest.mark.parametrize('field',['model_id','revision','worker_code_checksum'])
def test_reader_rejects_model_revision_or_code_identity_mismatch(field):
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp));finish(f,inp,doc)
    ref=doc['payload']['forward_qwen']['result_ref'];output=inp.artifacts.get_json(ref);output[field]='forged'
    new=inp.artifacts.put_json(output,kind='run_artifact');doc['payload']['forward_qwen']['result_ref']=new.to_dict();doc['artifacts'].append(new.to_dict())
    run=f.h.run(inp.spec['run_id']);f.h.store.update_run({**run,'result':doc,'revision':run['revision']+1},run['revision'],[])
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    with pytest.raises(ValueError):ApprovedForwardReader(inp.artifacts,f.h.store).lookup(context,'2026-07-11T00:20:00Z')


def test_offline_worker_rejects_unverified_code_before_model_call():
    f=fixture();inp=approved(f);inp.offline_code_checksum='sha256:'+'d'*64
    with pytest.raises(ValueError,match='code channel'):run_forward(inp,backend=lambda _:pytest.fail('no invocation'))
    assert len(inp.artifacts.objects)==1 # original binding only


def test_no_missing_checkpoint_import_or_gpu_constructor_for_fully_captured_retry(monkeypatch):
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp))
    import finplan_model.benchmarks.forward_qwen as module
    monkeypatch.setattr(module,'VllmBackend',lambda *a,**kw:pytest.fail('never load unavailable checkpoint on retry'))
    assert run_forward(inp)==doc


def test_offline_qwen_terminal_cost_is_unique_compute_estimate_and_known_no_vendor():
    from finplan_model.forward.managed_costs import terminal_cost_event
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp));run=f.h.run(inp.spec['run_id'])
    event=terminal_cost_event({**run,'job_started':True,'attempt':1,'state':'succeeded','completed_at':f.h.clock.now().isoformat().replace('+00:00','Z'),'result':doc})
    assert event['approach_id']=='qwen_swarm_mode_a' and event['compute_observed_usd'] is None and event['compute_estimated_usd']>0
    assert event['vendor_observed_usd']==event['vendor_estimated_usd']==0.


def test_forward_qwen_native_worker_branch_never_calls_historical_scope(monkeypatch):
    import finplan_model.benchmarks.jobs as jobs
    f=fixture();inp=approved(f)
    monkeypatch.setattr(jobs,'_scope',lambda *_:pytest.fail('no historical scope'))
    assert jobs.swarm_mode_a(inp,backend=Backend(f,inp))['completion_status']=='succeeded'


def test_beta_only_qwen_capability_leaves_higher_definition_unchanged():
    from scripts.release import planned_parameters
    from tests.unit.infra.test_release_publish import _info as info_factory
    # The helper takes no AWS client; this is a pure publication-plan check.
    release=info_factory()
    for env in ('gamma','prod'):
        plan=planned_parameters(env,release,{},account='<account-id>',deployed_job_types=['swarm_mode_a'])
        doc=json.loads(plan['job-swarm-mode-a'][1]);assert 'forward_protocol' not in doc and 'forward_source_commit' not in doc
    plan=planned_parameters('beta',release,{},account='<account-id>',deployed_job_types=['swarm_mode_a'])
    doc=json.loads(plan['job-swarm-mode-a'][1]);assert doc['forward_protocol']==PROTOCOL and doc['forward_source_commit']==release.source_commit


@pytest.mark.parametrize('error',[ValueError,TypeError,KeyError])
def test_backend_parser_like_exception_cannot_be_misreported_as_native_hold(error):
    f=fixture();inp=approved(f)
    def fail(_):raise error('controlled private error must not leak')
    doc=run_forward(inp,backend=fail)
    assert doc['completion_status']=='failed' and doc['payload']['forward_qwen']['invocation_state']=='indeterminate'
    assert len(doc['artifacts'])>=2 and 'controlled private error' not in json.dumps(doc)
    assert run_forward(inp,backend=lambda _:pytest.fail('unknown model call never repeated'))==doc


def test_completed_invalid_json_native_hold_retains_one_actual_response():
    f=fixture();inp=approved(f)
    doc=run_forward(inp,backend=lambda _:'not-json controlled public final response')
    assert doc['completion_status']=='succeeded' and doc['solution_status']=='no_effect'
    output=inp.artifacts.get_json(doc['payload']['forward_qwen']['result_ref'])
    assert len(output['role_receipts'])==1
    supplied=inp.artifacts.get_json(output['role_receipts'][0]['input_ref'])
    response=inp.artifacts.get_json(output['role_receipts'][0]['response_ref'])
    assert response['response']=='not-json controlled public final response' and supplied['role']=='market_analyst'


@pytest.mark.parametrize('partial',[False,True])
def test_offline_input_channel_and_verified_partial_output_archive_roundtrip(partial):
    import io
    from finplan_model.benchmarks.offline import OfflineBatchIO
    from finplan_model.benchmarks.weights import PREFIX
    from finplan_model.benchmarks.qwen import MODEL_ID,REVISION
    from finplan_model.jobs.runio import InMemoryRunIO
    from finplan_model.core.artifacts import InMemoryArtifactStore
    from tests.unit.benchmarks.test_offline import archive
    f=fixture();inp=approved(f);run={**f.h.run(inp.spec['run_id']),'job_name':'controlled-offline-qwen'}
    manifest=''.join('0'*64+'  '+name+'\n' for name in ('config.json','tokenizer_config.json','LICENSE','controlled.safetensors')).encode()
    storage={PREFIX+'STAGED.json':canonical_json_bytes({'model_id':MODEL_ID,'revision':REVISION,'license':'Apache-2.0','file_count':4,'manifest_checksum':checksum_bytes(manifest)}),PREFIX+'MANIFEST.sha256':manifest}
    class S3:
        def get_object(self,**kw):return {'Body':io.BytesIO(storage[kw['Key']])}
        def put_object(self,**kw):storage[kw['Key']]=kw['Body']
    gateway=OfflineBatchIO(S3(),'example-research-bucket',f.platform,InMemoryRunIO(),inp.artifacts)
    definition={**f.h.settings.job_definitions['swarm_mode_a'],'code_prefix':'releases/qwen-code/controlled/'}
    frozen=gateway.prepare_forward_weights(run,definition)
    run['forward_qwen']={**run['forward_qwen'],**frozen}
    inp.spec['forward_qwen']={**inp.spec['forward_qwen'],**frozen}
    handoff=gateway.prepare(run,inp.spec,definition)
    bundle=json.loads(storage['scratch/offline/'+run['run_id']+'/input.json'])
    assert bundle['forward_qwen_binding']==f.binding and bundle['spec']['forward_qwen']['binding_checksum']==checksum(f.binding)
    assert set(handoff['channels'])=={'bundle','weights','code'} and 'approval' not in bundle['forward_qwen_binding']
    doc=run_forward(inp,backend=Backend(f,inp,fail_at=2 if partial else None))
    files={'result.json':canonical_json_bytes(doc)}
    for ref in doc['artifacts']:files['artifacts/'+ref['kind']+'/'+ref['artifact_id']]=inp.artifacts.get(ref)
    path='scratch/training-output/'+run['job_name']+'/output/model.tar.gz';storage[path]=archive(files)
    imported_store=InMemoryArtifactStore();gateway.artifacts=imported_store
    readback=gateway.import_result(run,{'ModelArtifacts':{'S3ModelArtifacts':'s3://example-research-bucket/'+path}})
    assert readback==doc and readback['completion_status']==('failed' if partial else 'succeeded')
    for ref in doc['artifacts']:assert imported_store.get(ref)==inp.artifacts.get(ref)
    if partial:assert readback['payload']['forward_qwen']['invocation_state']=='indeterminate'


def checksum_bytes(data):
    from finplan_model.core.artifacts import sha256_checksum
    return sha256_checksum(data)


def test_dispatch_uses_existing_isolated_gpu_channels_and_changed_code_refuses_compute():
    f=fixture();inp=approved(f);f.h.settings.leases={'gpu':1}
    f.h.deps.offline_batch=SimpleNamespace(prepare_forward_weights=lambda run,definition:{'weights_proof_ref':f.h.deps.artifacts.put_json({'controlled_proof':True},kind='run_artifact').to_dict(),'weights_manifest_checksum':'sha256:'+'0'*64},prepare=lambda *args:{'channels':{'bundle':'s3://example-research-bucket/controlled-bundle/','weights':'s3://example-research-bucket/controlled-weights/','code':'s3://example-research-bucket/controlled-code/'},'bundle_checksum':'sha256:'+'e'*64,'code_checksum':inp.offline_code_checksum})
    f.h.service.dispatch()
    jobs=[req for op,req in f.h.sagemaker.calls if op=='CreateTrainingJob']
    assert len(jobs)==1 and jobs[0]['EnableNetworkIsolation'] is True
    assert jobs[0]['StoppingCondition']['MaxRuntimeInSeconds']==900
    assert jobs[0]['Environment']['FINPLAN_OFFLINE_CODE_SHA256']==inp.offline_code_checksum
    assert set(ch['ChannelName'] for ch in jobs[0]['InputDataConfig'])=={'bundle','weights','code'}
    assert 'RetryStrategy' not in jobs[0] and 'AWS_ACCESS_KEY_ID' not in jobs[0]['Environment']
    f2=fixture();approved(f2);f2.h.settings.leases={'gpu':1};f2.h.settings.job_definitions['swarm_mode_a']['code_checksum']='sha256:'+'d'*64
    f2.h.service.dispatch();assert not any(op=='CreateTrainingJob' for op,_ in f2.h.sagemaker.calls)
    assert next(iter(f2.h.store.runs.values()))['state']=='failed'


def test_failed_original_role_input_and_public_outputs_remain_report_inspectable():
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp,fail_at=2));run=f.h.run(inp.spec['run_id'])
    f.h.store.update_run({**run,'state':'failed','completed_at':f.h.clock.now().isoformat(),'revision':run['revision']+1,'result':doc},run['revision'],[])
    reader=ApprovedForwardReader(inp.artifacts,f.h.store);captured=[]
    outcome=adapter(EXPERIMENT,reader_factory=lambda:reader,dependency_reader=lambda _:{'status':'controlled'})({**f.base,'availability_cutoff':'2026-07-11T00:20:00Z'},f.source,f.base['holdings'],captured.append)
    assert outcome['status']=='unavailable' and captured[0]['model_consumed_inputs'] is False
    original=captured[0]['managed_forward_readback']['original_failed_invocation']
    assert original['status']=='original_partial_evidence' and original['current_slot_invoked'] is False
    inputs=[a['payload'] for a in original['artifacts'] if a['payload']['kind']=='role_input']
    responses=[a['payload'] for a in original['artifacts'] if a['payload']['kind']=='role_response']
    assert len(inputs)==3 and len(responses)==2 and inputs[-1]['role']=='allocator'
    assert json.loads(inputs[-1]['messages'][1]['content'])['evidence']['market_summary']=='actual controlled market output'


def test_hard_loss_missing_offline_output_is_visible_unusable_not_reconstructed():
    f=fixture();inp=approved(f);run=f.h.run(inp.spec['run_id'])
    f.h.store.update_run({**run,'state':'failed','completed_at':f.h.clock.now().isoformat(),'revision':run['revision']+1,'result':{'artifacts':[]}},run['revision'],[])
    reader=ApprovedForwardReader(inp.artifacts,f.h.store);context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    status=reader.lookup(context,'2026-07-11T00:20:00Z')
    assert status['status']=='managed_failed' and status['original_failed_invocation']['status']=='evidence_unavailable_or_indeterminate'
    assert 'artifacts' not in status['original_failed_invocation']


def test_lost_conditional_role_claim_never_invokes_even_with_identical_clock(monkeypatch):
    f=fixture();inp=approved(f);original=inp.artifacts.create_once;lost=[]
    def race(data,*,kind,artifact_id):
        if '_role_0_input_' in artifact_id and not lost:
            lost.append(artifact_id);assert original(data,kind=kind,artifact_id=artifact_id)
            return False # competing claimant won after our preceding NOT_FOUND
        return original(data,kind=kind,artifact_id=artifact_id)
    monkeypatch.setattr(inp.artifacts,'create_once',race)
    doc=run_forward(inp,backend=lambda _:pytest.fail('losing claimant must not invoke'))
    assert lost and doc['completion_status']=='failed'
    assert doc['payload']['forward_qwen']['invocation_state']=='indeterminate'


def test_offline_numerical_preparation_has_no_aws_sdk_dependency():
    import os,subprocess,sys
    f=fixture()
    code='''import sys,importlib.abc,json
class NoSdk(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('boto3','botocore'):raise ImportError('offline worker has no AWS SDK')
sys.meta_path.insert(0,NoSdk())
from finplan_model.benchmarks.forward_qwen import prepare_native,context_identity
body=json.load(sys.stdin)
view,holdings=prepare_native(body['binding'],body['source'])
assert holdings.cash_weight==1. and len(view.instruments)==2
print('pure_offline_preparation_pass')
'''
    result=subprocess.run([sys.executable,'-c',code],input=json.dumps({'binding':f.binding,'source':f.source}),text=True,capture_output=True,env=os.environ,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='pure_offline_preparation_pass'


def test_actual_backend_manifest_drift_rejects_before_model_call(monkeypatch):
    f=fixture();inp=approved(f)
    proof={'protocol':PROTOCOL,'kind':'weights_proof','managed_run_id':inp.spec['run_id'],'binding_checksum':checksum(f.binding),
        'model_id':'Qwen/Qwen3.6-27B','revision':'6a9e13bd6fc8f0983b9b99948120bc37f49c13e9','license':'Apache-2.0','file_count':4,'manifest_checksum':'sha256:'+'0'*64,
        'manifest_raw_base64':'Y29udHJvbGxlZA==','staged_raw_base64':'e30='}
    import base64
    metadata={k:proof[k] for k in ('model_id','revision','license','file_count','manifest_checksum')}
    metadata_raw=canonical_json_bytes(metadata);proof.update(staged_raw_base64=base64.b64encode(metadata_raw).decode(),staged_checksum=checksum_bytes(metadata_raw),worker_image_digest=inp.ctx.image_digest,worker_source_commit=inp.spec['forward_qwen']['worker_source_commit'],worker_code_checksum=inp.offline_code_checksum)
    # The proof's actual manifest hash is computed from its original bytes.
    original_manifest=base64.b64decode(proof['manifest_raw_base64']);proof['manifest_checksum']=checksum_bytes(original_manifest);metadata['manifest_checksum']=proof['manifest_checksum'];metadata_raw=canonical_json_bytes(metadata);proof.update(staged_raw_base64=base64.b64encode(metadata_raw).decode(),staged_checksum=checksum_bytes(metadata_raw))
    ref=inp.artifacts.put_json(proof,kind='run_artifact');inp.spec['forward_qwen'].update(weights_proof_ref=ref.to_dict(),weights_manifest_checksum=proof['manifest_checksum'])
    class BadBackend:
        manifest={**proof,'manifest_checksum':'sha256:'+'1'*64}
        def __call__(self,_):pytest.fail('never call model with drifted checkpoint')
    import finplan_model.benchmarks.forward_qwen as module
    monkeypatch.setattr(module,'VllmBackend',lambda *a,**k:BadBackend())
    doc=run_forward(inp)
    assert doc['completion_status']=='failed' and doc['payload']['forward_qwen']['usage']['calls']==0
    failure=inp.artifacts.get_json(doc['payload']['forward_qwen']['failure_ref']);assert failure['reason']=='qwen_weights_changed'


def test_controlled_backend_readiness_is_explicit_without_fabricated_weights_hash():
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp))
    output=inp.artifacts.get_json(doc['payload']['forward_qwen']['result_ref'])
    ready=inp.artifacts.get_json(output['backend_ready_ref'])
    assert ready['status']=='controlled_test_backend_no_checkpoint' and ready['manifest_checksum'] is None
    assert ready['license'] is None and ready['file_count'] is None


def test_resume_original_completed_roles_reuses_backend_ready_and_never_reinvokes_them(monkeypatch):
    f=fixture();inp=approved(f);backend=Backend(f,inp)
    import finplan_model.benchmarks.forward_qwen as module
    put=module._put
    class Crash(BaseException):pass
    def stop_after_response(store,binding,kind,payload):
        ref=put(store,binding,kind,payload)
        if kind=='role_1_response':raise Crash()
        return ref
    monkeypatch.setattr(module,'_put',stop_after_response)
    with pytest.raises(Crash):run_forward(inp,backend=backend)
    assert len(backend.seen)==2
    ready_ref=inp.artifacts.resolve(kind='run_artifact',artifact_id=journal_id(f.binding,'backend_ready'));ready_bytes=inp.artifacts.get(ready_ref)
    monkeypatch.setattr(module,'_put',put);new=[]
    def resume(messages):
        new.append(messages)
        role=messages[0]['content'].split('fixed ',1)[1].split(' ',1)[0]
        if role=='allocator':return json.dumps({'weights':{i:.5/len(f.market.instruments) for i in f.market.instruments},'cash':.5})
        if role=='critic':return '{"revise":false}'
        if role=='arbiter':return '{"choice":"original"}'
        pytest.fail('original completed market/risk role must not be invoked again')
    doc=run_forward(inp,backend=resume)
    assert doc['completion_status']=='succeeded' and len(new)==3
    output=inp.artifacts.get_json(doc['payload']['forward_qwen']['result_ref'])
    assert output['backend_ready_ref']==ready_ref.to_dict() and inp.artifacts.get(ready_ref)==ready_bytes
    assert output['usage']['calls']==5 and output['usage']['new_calls_this_attempt']==3 and output['usage']['recovered_role_indices']==[0,1]


def test_reader_uses_strong_terminal_availability_and_original_signal_age():
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=Backend(f,inp));f.h.clock.advance(minutes=30);finish(f,inp,doc)
    reader=ApprovedForwardReader(inp.artifacts,f.h.store)
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    assert reader.lookup(context,'2026-07-11T00:20:00Z')['status']=='managed_result_not_yet_platform_available'
    found=reader.lookup(context,'2026-07-11T01:00:00Z')
    assert found['status']=='approved_exact_result'
    assert found['signal_formed_at']!=found['platform_available_at']
    assert reader.lookup(context,'2026-07-12T00:20:00Z')['status']=='native_daily_signal_expired'


@pytest.mark.parametrize('recovery_hours',[2,26])
def test_recover_all_sealed_role_responses_after_expiry_without_loading_backend_or_refreshing_signal(monkeypatch,recovery_hours):
    f=fixture();inp=approved(f);backend=Backend(f,inp)
    import finplan_model.benchmarks.forward_qwen as module
    put=module._put
    class Crash(BaseException):pass
    def stop(store,binding,kind,payload):
        ref=put(store,binding,kind,payload)
        if kind=='role_4_response':raise Crash()
        return ref
    monkeypatch.setattr(module,'_put',stop)
    with pytest.raises(Crash):run_forward(inp,backend=backend)
    assert len(backend.seen)==5
    original=inp.artifacts.get_json(inp.artifacts.resolve(kind='run_artifact',artifact_id=journal_id(f.binding,'role_4_response')))['received_at']
    f.h.clock.advance(hours=recovery_hours);monkeypatch.setattr(module,'_put',put)
    monkeypatch.setattr(module,'VllmBackend',lambda *a,**k:pytest.fail('sealed response recovery must not load backend'))
    doc=run_forward(inp)
    assert doc['completion_status']=='succeeded'
    output=inp.artifacts.get_json(doc['payload']['forward_qwen']['result_ref'])
    assert output['signal_formed_at']==original and output['signal_formed_at']!=output['issued_at']
    assert output['usage']['new_calls_this_attempt']==0 and output['usage']['calls']==5
    finish(f,inp,doc)
    reader=ApprovedForwardReader(inp.artifacts,f.h.store)
    context=current_context(f.base,f.source,f.base['holdings'],f.captured['numeric']['prepared'],EXPERIMENT)
    lookup=reader.lookup(context,f.h.clock.now().isoformat())
    assert lookup['status']==('approved_exact_result' if recovery_hours<24 else 'native_daily_signal_expired')
    assert lookup['signal_formed_at']==original
