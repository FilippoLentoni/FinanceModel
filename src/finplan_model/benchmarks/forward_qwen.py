"""One independently approved, network-isolated current-book Qwen decision.

The existing offline importer seals actual role receipts. A hard container loss is
indeterminate and never authorizes another compute start for the same context.
"""
from __future__ import annotations
import copy
import base64
import json
import os
import time
from datetime import timedelta
from finplan_model.core.artifacts import canonical_json_bytes, LocalArtifactStore, ArtifactRef
from finplan_model.core.errors import FinplanError
from .forward_jev import checksum, instant, managed_id, verify_current_book, source_from_snapshot, read_named
from .qwen import MODEL_ID, REVISION, ROLES, QwenSwarm, VllmBackend

PROTOCOL = 'finplan-qwen-forward/1'
OBJECTIVE = 'forward_paper_decision'
NATIVE = {'model_id': MODEL_ID, 'revision': REVISION, 'roles': list(ROLES),
          'max_decisions': 1, 'max_role_calls': 7, 'max_tokens_per_response': 256,
          'max_revision_rounds': 1, 'temperature': 0., 'seed': 0, 'enable_thinking': False}


def context_identity(binding):
    return checksum({'protocol':PROTOCOL, **{key: binding[key] for key in ('source','holdings','constraints',
        'request_checksum','local_transformation','native_sizing','experiment_id','approach_id')}})


def journal_id(binding, kind):
    return 'forward_qwen_' + kind + '_' + checksum(binding).split(':')[1][:40]


def make_binding(captured, source, *, experiment_id, captured_checksum):
    prepared = captured['numeric']['prepared']
    first = prepared.get('prepared_role_messages') or []
    if captured['approach_id'] != 'qwen_swarm_mode_a' or len(first) != 1 or first[0].get('role') != 'market_analyst' or first[0].get('supplied_to_model') is not False:
        raise ValueError('original not-invoked Qwen first-role preparation required')
    if checksum(captured) != captured_checksum:
        raise ValueError('original captured Qwen bytes mismatch')
    binding = {'protocol': PROTOCOL, 'experiment_id': experiment_id,
        'approach_id': captured['approach_id'], 'origin_run_id': captured['run_id'],
        'origin_input_id': captured['input_id'], 'origin_input_checksum': captured_checksum,
        'scheduled_slot': captured['slot'], 'availability_cutoff': captured['availability_cutoff'],
        'source': {k: source[k] for k in ('input_snapshot_id','checksum','snapshot_created_at','investable_universe','price_basis')},
        'shared_source_checksum': captured['shared_source_checksum'], 'holdings': copy.deepcopy(captured['holdings']),
        'constraints': copy.deepcopy(captured['constraints']), 'request': {'first_role_messages': copy.deepcopy(first[0]['messages'])},
        'local_transformation': copy.deepcopy(prepared['local_transformation']),
        'native_sizing': copy.deepcopy(NATIVE), 'code_versions': copy.deepcopy(captured['code_versions'])}
    binding['request_checksum'] = checksum(binding['request'])
    return validate_binding(binding)


def validate_binding(binding):
    from finplan_model.control.validation import find_storage_location
    required = {'protocol','experiment_id','approach_id','origin_run_id','origin_input_id','origin_input_checksum',
        'scheduled_slot','availability_cutoff','source','shared_source_checksum','holdings','constraints','request',
        'request_checksum','local_transformation','native_sizing','code_versions'}
    if set(binding) != required or binding['protocol'] != PROTOCOL or binding['approach_id'] != 'qwen_swarm_mode_a' or binding['native_sizing'] != NATIVE:
        raise ValueError('invalid typed Qwen forward binding/model shape')
    if find_storage_location(binding) is not None or checksum(binding['request']) != binding['request_checksum']:
        raise ValueError('invalid original Qwen request content')
    if not isinstance(binding['holdings'],dict) or not isinstance(binding['source'],dict):
        raise ValueError('typed holdings/source objects required')
    if binding['holdings'].get('mode') != 'paper' or not binding['holdings'].get('portfolio_id') or isinstance(binding['holdings'].get('revision'),bool) or not isinstance(binding['holdings'].get('revision'),int):
        raise ValueError('exact isolated paper book revision required')
    if instant(binding['source']['snapshot_created_at']) > instant(binding['availability_cutoff']) or instant(binding['scheduled_slot']) > instant(binding['availability_cutoff']):
        raise ValueError('future input/invalid decision cutoff')
    if set(binding['request']) != {'first_role_messages'} or not isinstance(binding['request']['first_role_messages'],list):
        raise ValueError('exact native initial messages required')
    return binding


def prepare_native(binding, source):
    from finplan_model.benchmarks.forward_inputs import _view
    validate_binding(binding)
    _, view, holdings, private, _ = _view(source, binding['holdings'], binding['availability_cutoff'])
    if checksum(private) != checksum(binding['local_transformation']):
        raise ValueError('native Qwen preprocessing/current holdings differ from original')
    class Prepared(Exception): pass
    def first(messages):
        if canonical_json_bytes(messages) != canonical_json_bytes(binding['request']['first_role_messages']):
            raise ValueError('native Qwen first-role messages differ from original')
        raise Prepared()
    try: QwenSwarm(first, run_id=binding['origin_run_id'], max_decisions=1, max_calls=7).decide(view,holdings)
    except Prepared: return view,holdings
    raise ValueError('native Qwen did not prepare first-role messages')


def validate_weights_proof(proof, descriptor, binding, run_id):
    from finplan_model.core.artifacts import sha256_checksum
    if proof.get('kind')!='weights_proof' or proof.get('managed_run_id')!=run_id or proof.get('binding_checksum')!=checksum(binding) or proof.get('model_id')!=MODEL_ID or proof.get('revision')!=REVISION or proof.get('license')!='Apache-2.0' or proof.get('manifest_checksum')!=descriptor.get('weights_manifest_checksum'):
        raise ValueError('frozen weights proof identity differs')
    for key in ('worker_image_digest','worker_source_commit','worker_code_checksum'):
        if proof.get(key)!=descriptor[key]:raise ValueError('frozen weights proof worker differs')
    staged_raw=base64.b64decode(proof['staged_raw_base64'],validate=True)
    manifest=base64.b64decode(proof['manifest_raw_base64'],validate=True)
    if len(staged_raw)>262144 or len(manifest)>262144 or sha256_checksum(staged_raw)!=proof['staged_checksum'] or sha256_checksum(manifest)!=proof['manifest_checksum']:
        raise ValueError('frozen original weights metadata bytes differ')
    staged=json.loads(staged_raw)
    if any(staged.get(k)!=proof.get(k) for k in ('model_id','revision','license','file_count','manifest_checksum')):
        raise ValueError('frozen original staged metadata differs')
    return proof


def _sync(store, ref):
    """Durable local capture before GPU call; managed uploader seals on completion.

    No AWS credentials/network access is added to the offline GPU container.
    """
    if isinstance(store, LocalArtifactStore):
        path = store.root / ref.kind / ref.artifact_id
        with path.open('rb') as stream: os.fsync(stream.fileno())
        for directory in (path.parent, store.root):
            fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)


def _put(store, binding, kind, payload):
    aid=journal_id(binding,kind)
    if not store.create_once(canonical_json_bytes(payload),kind='run_artifact',artifact_id=aid):
        ref, old=read_named(store,aid)
        if old != payload: raise ValueError('immutable Qwen journal conflict')
    ref,value=read_named(store,aid);_sync(store,ref)
    return ref


def run_forward(inp, *, backend=None, weights_root='/opt/ml/input/data/weights'):
    from finplan_model.jobs.results import succeeded_result
    from finplan_model.core.outcome import failed_job_result
    from finplan_model.forward.runtime import validate_policy
    started=time.monotonic();descriptor=inp.spec['forward_qwen'];store=inp.artifacts
    binding=validate_binding(store.get_json(descriptor['binding_ref']))
    if checksum(binding)!=descriptor['binding_checksum'] or managed_id(binding)!=inp.spec['run_id'] or descriptor['worker_image_digest']!=inp.ctx.image_digest:
        raise ValueError('approved Qwen worker/binding mismatch')
    if getattr(inp, 'offline_code_checksum', None) != descriptor.get('worker_code_checksum') or not descriptor.get('worker_code_checksum'):
        raise ValueError('verified offline code channel differs from approved worker')
    if descriptor.get('approval_verified') is not True or not descriptor.get('approved_by') or not descriptor.get('approved_at'):
        raise ValueError('independent GPU approval missing')
    # A complete original local result is reused without new model construction.
    try:
        _,finished=read_named(store,journal_id(binding,'job_result'))
        if finished['run_id']!=inp.spec['run_id'] or (finished.get('payload') or {}).get('forward_qwen',{}).get('binding_checksum')!=checksum(binding):raise ValueError('saved Qwen result identity mismatch')
        return finished
    except FinplanError as exc:
        if exc.code!='NOT_FOUND':raise
    refs=[];roles=[];attempted=[];calls=0;engine=backend;backend_ready_ref=None;pending={};recovered=[];uncertain_existing=False;previous_engine_tokens=0
    approval={'approved_by':descriptor['approved_by'],'approved_at':descriptor['approved_at']}
    try:
        backend_ready_ref,saved_ready=read_named(store,journal_id(binding,'backend_ready'))
        if saved_ready.get('managed_run_id')!=inp.spec['run_id'] or saved_ready.get('binding_checksum')!=checksum(binding):raise ValueError('saved backend ready association differs')
        refs.append(backend_ready_ref)
    except FinplanError as exc:
        if exc.code!='NOT_FOUND':raise
    def eligible():
        if instant(inp.ctx.now_ts())>=instant(binding['availability_cutoff'])+timedelta(hours=1):
            raise FinplanError.precondition('Qwen current-book request expired before role invocation',reason='forward_request_expired')
        if time.monotonic()-started>=min(900,int(inp.spec['max_runtime_seconds'])):
            raise FinplanError.precondition('Qwen forward runtime bound exceeded',reason='qwen_usage_cap')
    def capture(role,messages,index):
        nonlocal uncertain_existing
        eligible()
        body={'protocol':PROTOCOL,'kind':'role_input','managed_run_id':inp.spec['run_id'],
            'binding_checksum':checksum(binding),'role':role,'call_index':index,'messages':copy.deepcopy(messages),
            'messages_checksum':checksum(messages),'model_id':MODEL_ID,'revision':REVISION,
            'approval':approval,'worker_image_digest':inp.ctx.image_digest,'worker_source_commit':descriptor['worker_source_commit'],'worker_code_checksum':descriptor['worker_code_checksum']}
        aid=journal_id(binding,'role_'+str(index)+'_input')
        try:
            input_ref,original=read_named(store,aid)
            uncertain_existing=True
            for key,value in body.items():
                if original.get(key)!=value:raise ValueError('original Qwen role claim differs')
            try: response_ref,response=read_named(store,journal_id(binding,'role_'+str(index)+'_response'))
            except FinplanError as exc:
                if exc.code=='NOT_FOUND':raise FinplanError.precondition('claimed offline role has no original response; never rerun',reason='qwen_role_indeterminate') from None
                raise
            if response.get('input_ref')!=input_ref.to_dict() or response.get('messages_checksum')!=checksum(messages) or response.get('managed_run_id')!=inp.spec['run_id'] or response.get('binding_checksum')!=checksum(binding):
                raise ValueError('original Qwen response binding differs')
            pending.update(input_ref=input_ref,response_ref=response_ref,response=response,index=index,role=role)
            recovered.append(index)
        except FinplanError as exc:
            if exc.code!='NOT_FOUND':raise
            import uuid
            body.update(captured_at=inp.ctx.now_ts(),attempt_token=uuid.uuid4().hex)
            if not store.create_once(canonical_json_bytes(body),kind='run_artifact',artifact_id=aid):
                uncertain_existing=True
                raise FinplanError.precondition('another worker owns the permanent Qwen role claim; no reinvocation',reason='qwen_role_indeterminate')
            input_ref,sealed=read_named(store,aid)
            if sealed!=body:raise ValueError('owned Qwen role claim changed')
            _sync(store,input_ref)
            pending.clear();pending.update(input_ref=input_ref,index=index,role=role)
        refs.append(pending['input_ref']);attempted.append(index)
    def invoke(messages):
        nonlocal engine,calls,backend_ready_ref
        if pending.get('response') is not None:return pending['response']['response']
        eligible()
        proof_ref=descriptor.get('weights_proof_ref');proof=None
        if backend is None:
            if not proof_ref:raise FinplanError.precondition('frozen precompute weights proof missing',reason='qwen_weights_not_verified')
            proof=validate_weights_proof(store.get_json(proof_ref),descriptor,binding,inp.spec['run_id'])
        if engine is None:engine=VllmBackend(weights_root,max_tokens=256)
        if backend is None:
                manifest=getattr(engine,'manifest',{})
                if proof.get('kind')!='weights_proof' or proof.get('managed_run_id')!=inp.spec['run_id'] or proof.get('binding_checksum')!=checksum(binding) or proof.get('manifest_checksum')!=descriptor.get('weights_manifest_checksum') or any(manifest.get(k)!=proof.get(k) for k in ('model_id','revision','license','file_count','manifest_checksum')):
                    raise FinplanError.precondition('loaded Qwen checkpoint differs from frozen precompute manifest',reason='qwen_weights_changed')
                if not any(ref.to_dict()==proof_ref for ref in refs):refs.append(ArtifactRef.from_dict(proof_ref))
        if backend_ready_ref is None:
            ready={'protocol':PROTOCOL,'kind':'backend_ready','managed_run_id':inp.spec['run_id'],
                'binding_checksum':checksum(binding),'model_id':MODEL_ID,'revision':REVISION,
                'status':'verified_pinned_checkpoint' if proof else 'controlled_test_backend_no_checkpoint',
                'license':proof.get('license') if proof else None,'file_count':proof.get('file_count') if proof else None,
                'manifest_checksum':proof.get('manifest_checksum') if proof else None,'weights_proof_ref':proof_ref if proof else None,
                'staged_raw_base64':proof.get('staged_raw_base64') if proof else None,
                'manifest_raw_base64':proof.get('manifest_raw_base64') if proof else None,
                'ready_at':inp.ctx.now_ts(),'worker_image_digest':inp.ctx.image_digest,
                'worker_source_commit':descriptor['worker_source_commit'],'worker_code_checksum':descriptor['worker_code_checksum']}
            backend_ready_ref=_put(store,binding,'backend_ready',ready);refs.append(backend_ready_ref)
        eligible();calls+=1
        try:
            return engine(messages)
        except Exception as exc:
            raise FinplanError.dependency_unavailable('Qwen role backend failed after original input capture', retryable=False, reason='qwen_role_backend_failed') from exc
    def response(role,messages,text,index):
        nonlocal previous_engine_tokens
        if not isinstance(text,str) or len(text)>16000:raise ValueError('Qwen public role response bound')
        if pending.get('response') is not None:
            ref=pending['response_ref'];body=pending['response']
        else:
            body={'protocol':PROTOCOL,'kind':'role_response','managed_run_id':inp.spec['run_id'],
                'binding_checksum':checksum(binding),'role':role,'call_index':index,'input_ref':pending['input_ref'].to_dict(),
                'messages_checksum':checksum(messages),'response':text,'received_at':inp.ctx.now_ts()}
            observed=getattr(engine,'tokens',None)
            body['model_output_tokens']=observed-previous_engine_tokens if type(observed) is int and observed>=previous_engine_tokens else None
            if type(observed) is int:previous_engine_tokens=observed
            ref=_put(store,binding,'role_'+str(index)+'_response',body)
        refs.append(ref);roles.append({'role':role,'call_index':index,'input_ref':pending['input_ref'].to_dict(),'response_ref':ref.to_dict()})
    def capture_checked(*args):
        try: return capture(*args)
        except Exception as exc:
            if isinstance(exc,FinplanError):raise
            raise FinplanError.precondition('Qwen role input evidence unavailable; no invocation',reason='qwen_role_capture_failed') from exc
    def response_checked(*args):
        try: return response(*args)
        except Exception as exc:
            if isinstance(exc,FinplanError):raise
            raise FinplanError.precondition('Qwen original role response could not be sealed',reason='qwen_role_response_unavailable') from exc
    try:
        eligible();source=source_from_snapshot(inp.snapshot,inp.market,binding)
        view,holdings=prepare_native(binding,source)
        strategy=QwenSwarm(invoke,run_id=inp.spec['run_id'],max_decisions=1,max_calls=7,input_capture=capture_checked,response_capture=response_checked)
        result=strategy.decide(view,holdings)
        if [r['call_index'] for r in roles] != attempted:
            raise FinplanError.precondition('a claimed Qwen role lacks an original response', reason='qwen_role_indeterminate')
        policy=validate_policy({'weights':dict(result.weights),'cash':result.cash,'solution_status':result.solution_status,
            'forecast':{'available':False,'status':'not_available','reason':'native allocation is not a forecast'}},binding['source']['investable_universe'],binding['holdings'])
        output={'protocol':PROTOCOL,'kind':'forward_result','managed_run_id':inp.spec['run_id'],
            'binding_checksum':checksum(binding),'request_checksum':binding['request_checksum'],'policy':policy,
            'role_receipts':roles,'backend_ready_ref':backend_ready_ref.to_dict() if backend_ready_ref else None,'issued_at':inp.ctx.now_ts(),'ledger_updated':False,'model_id':MODEL_ID,'revision':REVISION,
            'approval':approval,'worker_image_digest':inp.ctx.image_digest,'worker_source_commit':descriptor['worker_source_commit'],'worker_code_checksum':descriptor['worker_code_checksum'],
            'usage':{'calls':len(roles),'new_calls_this_attempt':calls,'recovered_role_indices':recovered,
                'model_output_tokens':sum(store.get_json(r['response_ref'])['model_output_tokens'] for r in roles) if all(type(store.get_json(r['response_ref']).get('model_output_tokens')) is int for r in roles) else None,
                'vendor_cost_usd':0.,'external_vendor_estimated_usd':0.,'vendor_price_basis':'offline local GPU model; no external paid vendor transport'},
            'evidence_transport':'fsynced offline originals; immutable controller import; hard-loss missing output is indeterminate/unusable'}
        ref=_put(store,binding,'result',output);refs.append(ref)
        doc=succeeded_result(inp.ctx,inp.spec,solution_status=result.solution_status,artifacts=refs,performance={},instance_seconds=time.monotonic()-started)
        doc['payload']['forward_qwen']={'binding_checksum':checksum(binding),'result_ref':ref.to_dict(),'invocation_state':'completed','usage':output['usage']}
    except Exception as exc:
        failure={'protocol':PROTOCOL,'kind':'failed_forward_result','managed_run_id':inp.spec['run_id'],
            'binding_checksum':checksum(binding),'backend_ready_ref':backend_ready_ref.to_dict() if backend_ready_ref else None,'role_receipts':roles,'captured_input_refs':[r.to_dict() for r in refs],
            'invocation_state':'indeterminate' if calls or pending or uncertain_existing else 'not_invoked',
            'reason':getattr(exc,'details',{}).get('reason',type(exc).__name__),
            'usage':{'calls':len(recovered)+calls,'new_calls_this_attempt':calls,'vendor_cost_usd':0.,'external_vendor_estimated_usd':0.,'vendor_price_basis':'network-isolated local GPU; compute billing unavailable'},
            'ledger_updated':False,'worker_image_digest':inp.ctx.image_digest,'worker_source_commit':descriptor['worker_source_commit'],'worker_code_checksum':descriptor['worker_code_checksum']}
        ref=_put(store,binding,'failure',failure);refs.append(ref)
        doc=failed_job_result(inp.ctx,FinplanError.precondition('Qwen forward failed; original partial evidence retained, no relaunch',reason=failure['reason']),run_id=inp.spec['run_id'])
        doc.update(configuration_id=inp.spec['configuration_id'],input_snapshot_id=inp.spec['input_snapshot_id'],artifacts=[r.to_dict() for r in refs],artifacts_complete=False)
        doc['payload']={'forward_qwen':{'binding_checksum':checksum(binding),'failure_ref':ref.to_dict(),'invocation_state':failure['invocation_state'],'usage':failure['usage']}}
    finally:
        close=getattr(engine,'close',None)
        if close:close()
    from finplan_model.core.outcome import require_valid
    require_valid(doc,'job-result');_put(store,binding,'job_result',doc)
    return doc
