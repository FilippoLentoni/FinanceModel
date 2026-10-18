"""Read-only consumption of exactly bound independently approved Qwen originals."""
from __future__ import annotations
import copy
from datetime import timedelta
from finplan_model.benchmarks.forward_qwen import (PROTOCOL,NATIVE,checksum,context_identity,
    instant,managed_id,validate_binding,prepare_native,journal_id)
from finplan_model.benchmarks.qwen import QwenSwarm
from .approved_jev import ApprovedForwardReader as JevReader


def current_context(base,source,holdings,prepared,experiment):
    first=prepared['prepared_role_messages'][0]
    return {'experiment_id':experiment,'approach_id':'qwen_swarm_mode_a',
        'source':{k:source[k] for k in ('input_snapshot_id','checksum','snapshot_created_at','investable_universe','price_basis')},
        'holdings':copy.deepcopy(holdings),'constraints':copy.deepcopy(base['constraints']),
        'request_checksum':checksum({'first_role_messages':first['messages']}),
        'local_transformation':copy.deepcopy(prepared['local_transformation']),'native_sizing':copy.deepcopy(NATIVE)}


class ApprovedForwardReader(JevReader):
    def lookup(self,context,cutoff):
        index=self.runs.get_forward_context(context_identity(context))
        if index is None:return {'status':'no_exact_context_proposal'}
        binding=validate_binding(self.artifacts.get_json(index['binding_ref']))
        if index['context_id']!=context_identity(context) or checksum(binding)!=index['binding_checksum'] or context_identity(binding)!=context_identity(context) or index['managed_run_id']!=managed_id(binding):
            raise ValueError('approved Qwen permanent context binding mismatch')
        rid=managed_id(binding);run=self.runs.get_run(rid)
        if run is None:raise ValueError('approved Qwen context lacks permanent managed run')
        descriptor=run.get('forward_qwen') or {}
        if descriptor.get('binding_ref')!=index['binding_ref'] or descriptor.get('binding_checksum')!=checksum(binding) or run.get('input_snapshot_id')!=binding['source']['input_snapshot_id']:
            raise ValueError('approved Qwen run binding mismatch')
        if run['state'] in ('succeeded', 'failed', 'cancelled', 'timed_out'):
            platform_available_at = run.get('completed_at')
            if not platform_available_at:
                return {'status': 'managed_terminal_availability_missing', 'managed_run_id': rid}
            if instant(platform_available_at) > instant(cutoff):
                return {'status': 'managed_result_not_yet_platform_available', 'managed_run_id': rid,
                    'platform_available_at': platform_available_at}
        if run['state']!='succeeded':
            state={'status':'managed_'+run['state'],'managed_run_id':rid,'approval_required':True}
            if run['state'] in ('failed','cancelled','timed_out'):
                result=run.get('result') or {};section=(result.get('payload') or {}).get('forward_qwen') or {}
                ref=section.get('failure_ref')
                if ref is None:
                    state['original_failed_invocation']={'status':'evidence_unavailable_or_indeterminate','reason':'offline hard-loss/output unavailable; no input or response reconstructed','binding_ref':index['binding_ref']}
                else:
                    if ref not in result.get('artifacts',[]) or ref.get('artifact_id')!=journal_id(binding,'failure'):
                        raise ValueError('Qwen failed original receipt missing')
                    failure=self.artifacts.get_json(ref)
                    if failure.get('protocol')!=PROTOCOL or failure.get('managed_run_id')!=rid or failure.get('binding_checksum')!=checksum(binding) or failure.get('worker_image_digest')!=descriptor['worker_image_digest'] or failure.get('worker_source_commit')!=descriptor['worker_source_commit'] or failure.get('worker_code_checksum')!=descriptor['worker_code_checksum']:
                        raise ValueError('Qwen failed original receipt association mismatch')
                    artifacts=[]
                    for source_ref in failure.get('captured_input_refs',[]):
                        if source_ref not in result['artifacts']:raise ValueError('Qwen failed role evidence is not declared')
                        content=self.artifacts.get_json(source_ref)
                        if content.get('protocol')!=PROTOCOL or content.get('managed_run_id')!=rid or content.get('binding_checksum')!=checksum(binding):raise ValueError('Qwen failed role evidence binding mismatch')
                        artifacts.append({'ref':source_ref,'payload':content})
                    state['original_failed_invocation']={'status':'original_partial_evidence','failure_ref':ref,'failure':failure,'artifacts':artifacts,'current_slot_invoked':False}
            return state
        approval=run.get('approval') or {}
        if not run.get('approval_required') or not approval.get('approved_by') or approval['approved_by']==run.get('principal_name') or run.get('purpose')!='research' or run.get('job_type')!='swarm_mode_a':
            raise ValueError('independent Qwen approval absent')
        result=run.get('result') or {};section=(result.get('payload') or {}).get('forward_qwen') or {}
        if result.get('completion_status')!='succeeded' or result.get('run_id')!=rid or result.get('input_snapshot_id')!=run['input_snapshot_id'] or result.get('configuration_id')!=run['configuration_id'] or section.get('binding_checksum')!=checksum(binding) or section.get('result_ref') not in result.get('artifacts',[]):
            raise ValueError('Qwen result identity/complete evidence invalid')
        output=self.artifacts.get_json(section['result_ref'])
        expected={'protocol':PROTOCOL,'kind':'forward_result','managed_run_id':rid,'binding_checksum':checksum(binding),
            'request_checksum':binding['request_checksum'],'ledger_updated':False,'model_id':NATIVE['model_id'],'revision':NATIVE['revision'],
            'worker_image_digest':descriptor['worker_image_digest'],'worker_source_commit':descriptor['worker_source_commit'],
            'worker_code_checksum':descriptor['worker_code_checksum'],'approval':{'approved_by':approval['approved_by'],'approved_at':approval['approved_at']}}
        if any(output.get(k)!=v for k,v in expected.items()):raise ValueError('Qwen output worker/model/approval mismatch')
        if instant(output['issued_at']) > instant(platform_available_at):raise ValueError('managed terminal receipt predates original output')
        if instant(output['issued_at'])>instant(cutoff):raise ValueError('Qwen output was unavailable at cutoff')
        ready_ref=output.get('backend_ready_ref')
        if ready_ref not in result.get('artifacts',[]) or ready_ref.get('artifact_id')!=journal_id(binding,'backend_ready'):
            raise ValueError('original Qwen backend readiness receipt missing')
        ready=self.artifacts.get_json(ready_ref)
        if ready.get('kind')!='backend_ready' or any(ready.get(k)!=v for k,v in expected.items() if k in ('protocol','managed_run_id','binding_checksum','model_id','revision','worker_image_digest','worker_source_commit','worker_code_checksum')):
            raise ValueError('original Qwen backend model/worker readiness differs')
        if descriptor.get('weights_proof_ref'):
            proof=self.artifacts.get_json(descriptor['weights_proof_ref'])
            if ready.get('status')!='verified_pinned_checkpoint' or ready.get('weights_proof_ref')!=descriptor['weights_proof_ref'] or ready.get('manifest_checksum')!=descriptor.get('weights_manifest_checksum') or any(ready.get(k)!=proof.get(k) for k in ('model_id','revision','license','file_count','manifest_checksum','staged_raw_base64','manifest_raw_base64')):
                raise ValueError('original Qwen loaded checkpoint differs from frozen proof')
        elif ready.get('status')!='controlled_test_backend_no_checkpoint':
            raise ValueError('original Qwen precompute weights proof is absent')
        receipts=output['role_receipts']
        if not 1<=len(receipts)<=7 or (len(receipts)<5 and output['policy'].get('solution_status')!='no_effect'):raise ValueError('complete native Qwen role evidence missing')
        decoded=[];last=instant(approval['approved_at'])
        for index,receipt in enumerate(receipts):
            if receipt['call_index']!=index or receipt['input_ref']['artifact_id']!=journal_id(binding,'role_'+str(index)+'_input') or receipt['response_ref']['artifact_id']!=journal_id(binding,'role_'+str(index)+'_response') or receipt['input_ref'] not in result['artifacts'] or receipt['response_ref'] not in result['artifacts']:
                raise ValueError('Qwen original named role receipt identity mismatch')
            supplied=self.artifacts.get_json(receipt['input_ref']);response=self.artifacts.get_json(receipt['response_ref'])
            for doc in (supplied,response):
                if doc.get('protocol')!=PROTOCOL or doc.get('managed_run_id')!=rid or doc.get('binding_checksum')!=checksum(binding) or doc.get('role')!=receipt['role'] or doc.get('call_index')!=index:
                    raise ValueError('Qwen role association mismatch')
            if supplied.get('kind')!='role_input' or response.get('kind')!='role_response' or supplied.get('messages_checksum')!=checksum(supplied['messages']) or response.get('messages_checksum')!=supplied['messages_checksum'] or response.get('input_ref')!=receipt['input_ref'] or any(supplied.get(k)!=v for k,v in expected.items() if k in ('approval','worker_image_digest','worker_source_commit','worker_code_checksum','model_id','revision')):
                raise ValueError('Qwen actual supplied content/checksum/worker mismatch')
            captured=instant(supplied['captured_at']);received=instant(response['received_at'])
            if not(last<=captured<=received<=instant(output['issued_at'])):raise ValueError('Qwen role availability timestamps invalid')
            last=received;decoded.append({'role':receipt['role'],'call_index':index,'input':supplied,'response':response,'input_ref':receipt['input_ref'],'response_ref':receipt['response_ref']})
        signal_formed_at=decoded[-1]['response']['received_at']
        if output.get('signal_formed_at',signal_formed_at)!=signal_formed_at:raise ValueError('Qwen original signal formation timestamp differs')
        if instant(cutoff)>=instant(signal_formed_at)+timedelta(days=1):
            return {'status':'native_daily_signal_expired','managed_run_id':rid,'signal_formed_at':signal_formed_at,'platform_available_at':platform_available_at}
        return {'status':'approved_exact_result','binding':binding,'managed_run_id':rid,'binding_ref':descriptor['binding_ref'],
            'result_ref':section['result_ref'],'output':output,'role_evidence':decoded,'approval':copy.deepcopy(approval),
            'signal_formed_at':signal_formed_at,'platform_available_at':platform_available_at,
            'managed_cost_event':copy.deepcopy(run['forward_cost_event'])}


def adapter(experiment,*,reader_factory=ApprovedForwardReader.from_ec2,dependency_reader=None):
    reader=None
    def invoke(base,source,holdings,before):
        nonlocal reader
        from .public_preflight import preflight
        captured=[];unavailable=preflight('qwen',base,source,holdings,captured.append,dependency_reader=dependency_reader);prepared=captured[0]
        try:
            if not prepared.get('prepared_role_messages'):lookup={'status':'native_preparation_ineligible'}
            else:
                if reader is None:reader=reader_factory()
                lookup=reader.lookup(current_context(base,source,holdings,prepared,experiment),base['availability_cutoff'])
        except Exception as exc:lookup={'status':'managed_forward_readback_invalid','error_type':type(exc).__name__}
        if lookup['status']!='approved_exact_result':
            before({**prepared,'managed_forward_readback':lookup});return {**unavailable,'managed_forward_readback':lookup}
        origin=lookup['binding'];output=lookup['output'];roles=lookup['role_evidence'];cursor=0
        view,hv=prepare_native(origin,source)
        def replay(messages):
            nonlocal cursor
            if cursor>=len(roles) or canonical(messages)!=canonical(roles[cursor]['input']['messages']):raise ValueError('native dependent Qwen messages differ from original supplied bytes')
            text=roles[cursor]['response']['response'];cursor+=1;return text
        actual=QwenSwarm(replay,run_id=lookup['managed_run_id'],max_decisions=1,max_calls=7).decide(view,hv)
        if cursor!=len(roles) or dict(actual.weights)!=output['policy']['weights'] or actual.cash!=output['policy']['cash']:
            raise ValueError('approved Qwen original responses/native policy mismatch')
        before({**prepared,'boundary':'approved_managed_qwen_result_reuse','invocation_state':'reused_approved_native_daily_signal',
            'model_consumed_inputs':True,'actual_role_evidence':copy.deepcopy(roles),
            'prepared_role_messages':[{'role':r['role'],'messages':copy.deepcopy(r['input']['messages']),'supplied_to_model':True,'input_ref':r['input_ref'],'response_ref':r['response_ref']} for r in roles],
            'original_preflight_input_id':origin['origin_input_id'],'original_preflight_run_id':origin['origin_run_id'],
            'original_worker_run_id':lookup['managed_run_id'],'original_model_availability_cutoff':origin['availability_cutoff'],
            'managed_binding_ref':lookup['binding_ref'],'managed_result_ref':lookup['result_ref'],'model_issued_at':output['issued_at'], 'signal_formed_at':lookup['signal_formed_at'], 'platform_available_at':lookup['platform_available_at'],
            'approval':lookup['approval'],'reuse_policy':'same exact approved snapshot/native first-role/private preprocessing/book revision; <=24h native signal; no new model invocation'})
        return {**copy.deepcopy(output['policy']),'status':'reused','managed_run_id':lookup['managed_run_id'],
            'signal_origin_run_id':lookup['managed_run_id'],'signal_issued_at':output['issued_at'], 'signal_formed_at':lookup['signal_formed_at'], 'platform_available_at':lookup['platform_available_at'],'model_invocation_this_slot':False,
            'model_cost':0.,'model_cost_basis':'observed no new GPU invocation; original managed cost event retained separately',
            'managed_cost_event':lookup['managed_cost_event'],'usage':{'calls':0,'vendor_cost_usd':0.,'original_usage':output['usage']}}
    return invoke


def canonical(value):
    from finplan_model.core.artifacts import canonical_json_bytes
    return canonical_json_bytes(value)
