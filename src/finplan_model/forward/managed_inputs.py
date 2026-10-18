"""Original managed-worker input/output bytes archived independently of hourly reuse."""
from __future__ import annotations
import base64
import json
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum

MAX_REFS=32
MAX_ORIGINAL_BYTES=8*2**20
PROTOCOL='finplan-managed-inference-evidence/1'


def captured_event(event,reader):
    """No replacement input generation. Missing originals raise, leaving outbox pending."""
    refs=list(event.get('source_evidence_refs') or [])
    if len(refs)>MAX_REFS:raise ValueError('managed original evidence reference bound exceeded')
    unique={};identities={};artifacts=[];total=0;binding=None;kinds=[]
    offset=0
    while offset<len(refs):
        ref=refs[offset];offset+=1
        if len(unique)>=MAX_REFS and (ref.get("kind"),ref.get("artifact_id"),ref.get("checksum")) not in unique:
            raise ValueError("managed original evidence reference bound exceeded")
        if not isinstance(ref,dict) or ref.get('kind')!='run_artifact' or ref.get('owner')!='financemodel':
            raise ValueError('managed evidence must use owned original Model artifacts')
        identity=(ref['kind'],ref['artifact_id']);key=(*identity,ref['checksum'])
        if identity in identities and identities[identity]!=ref['checksum']:raise ValueError('managed original artifact identity conflict')
        identities[identity]=ref['checksum']
        if key in unique:continue
        declared=ref.get('size_bytes')
        if isinstance(declared,bool) or not isinstance(declared,int) or not 0<=declared<=MAX_ORIGINAL_BYTES-total:
            raise ValueError('managed original evidence requires bounded actual artifact sizes')
        raw=reader.artifacts.get(ref)
        if len(raw)!=declared or sha256_checksum(raw)!=ref['checksum']:raise ValueError('managed original evidence checksum/size differs')
        total+=len(raw)
        if total>MAX_ORIGINAL_BYTES:raise ValueError('managed original evidence byte bound exceeded')
        try:payload=json.loads(raw)
        except (ValueError,UnicodeDecodeError):raise ValueError('managed original evidence must be JSON public boundary artifacts') from None
        if not isinstance(payload,dict):raise ValueError('managed original artifact must be a boundary object')
        if ref==event.get('binding_ref'):
            if ref['checksum']!=event['binding_checksum'] or payload.get('experiment_id')!=event['experiment_id'] or payload.get('approach_id')!=event['approach_id'] or payload.get('holdings',{}).get('portfolio_id')!=event['portfolio_id']:
                raise ValueError('original managed binding belongs to another book/context')
            binding=payload
        else:
            if payload.get('managed_run_id')!=event['managed_run_id'] or payload.get('binding_checksum')!=event['binding_checksum']:
                raise ValueError('original managed boundary belongs to another invocation')
            kinds.append(payload.get('kind'))
            # Follow only typed top-level worker journal links, never arbitrary model text.
            for field in ('invocation_input_ref','response_journal_ref','invocation_claim_ref','failure_ref','backend_ready_ref','weights_proof_ref'):
                nested=payload.get(field)
                if nested is not None:
                    if not isinstance(nested,dict):raise ValueError('managed typed journal reference invalid')
                    if not any(existing==nested for existing in refs):refs.append(nested)
            if payload.get('protocol')=='finplan-qwen-forward/1' and payload.get('kind')=='forward_result':
                if not isinstance(payload.get('backend_ready_ref'),dict):raise ValueError('managed original Qwen backend readiness absent')
                roles=payload.get('role_receipts')
                if not isinstance(roles,list) or not 1<=len(roles)<=7:
                    raise ValueError('managed original Qwen role receipt bound invalid')
                for call_index,role in enumerate(roles):
                    if not isinstance(role,dict) or role.get('call_index')!=call_index or role.get('role') not in ('market_analyst','risk_analyst','allocator','critic','arbiter'):
                        raise ValueError('managed original Qwen role identity/order invalid')
                    for field in ('input_ref','response_ref'):
                        nested=role.get(field)
                        if not isinstance(nested,dict):raise ValueError('managed original Qwen role reference absent')
                        if not any(existing==nested for existing in refs):refs.append(nested)
        unique[key]=True
        artifacts.append({'ref':ref,'checksum':ref['checksum'],'size_bytes':len(raw),'content_type':ref['content_type'],
            'raw_base64':base64.b64encode(raw).decode('ascii')})
    if refs and binding is None:raise ValueError('managed original binding is absent from declared source evidence')
    decoded={artifact['ref']['artifact_id']:json.loads(base64.b64decode(artifact['raw_base64'])) for artifact in artifacts}
    for body in decoded.values():
        if body.get('protocol')=='finplan-qwen-forward/1' and body.get('kind')=='forward_result':
            ready=decoded[body['backend_ready_ref']['artifact_id']]
            if ready.get('kind')!='backend_ready' or ready.get('model_id')!=body.get('model_id') or ready.get('revision')!=body.get('revision'):
                raise ValueError('managed original Qwen backend readiness association differs')
            for role in body['role_receipts']:
                supplied=decoded[role['input_ref']['artifact_id']];response=decoded[role['response_ref']['artifact_id']]
                if supplied.get('kind')!='role_input' or response.get('kind')!='role_response' or any(doc.get('role')!=role['role'] or doc.get('call_index')!=role['call_index'] for doc in (supplied,response)) or response.get('input_ref')!=role['input_ref']:
                    raise ValueError('managed original Qwen role content/association differs')
    has_input=any(k in ('role_input','invocation_input') for k in kinds)
    if event['job_state']=='succeeded' and has_input and 'forward_result' in kinds:
        status='complete_original_artifacts'
    elif has_input:status='original_partial_artifacts'
    else:status='evidence_unavailable_or_indeterminate'
    source=binding or {}
    result={'protocol':PROTOCOL,'event_id':'managed_inputs:'+event['managed_run_id'],
        'cost_event_id':event['event_id'],'cost_event_checksum':sha256_checksum(canonical_json_bytes(event)),
        'experiment_id':event['experiment_id'],'approach_id':event['approach_id'],'portfolio_id':event['portfolio_id'],
        'managed_run_id':event['managed_run_id'],'occurred_at':event['occurred_at'],
        'job_state':event['job_state'],'invocation_state':event.get('invocation_state','unknown_or_indeterminate'),
        'binding_ref':event.get('binding_ref'),'binding_checksum':event.get('binding_checksum'),
        'origin_run_id':source.get('origin_run_id'),'origin_input_id':source.get('origin_input_id'),
        'scheduled_slot':source.get('scheduled_slot'),'availability_cutoff':source.get('availability_cutoff'),
        'worker_image_digest':event.get('worker_image_digest'),'worker_source_commit':event.get('worker_source_commit'),
        'worker_code_checksum':None,'approval':event.get('approval'),'evidence_status':status,'artifacts':artifacts}
    if hasattr(reader.runs,'get_run'):
        run=reader.runs.get_run(event['managed_run_id'])
        if run is None:raise ValueError('managed original cost event lacks its permanent run')
        descriptor=run.get('forward_jev') or run.get('forward_qwen') or {}
        if descriptor.get('binding_ref')!=event.get('binding_ref') or descriptor.get('binding_checksum')!=event.get('binding_checksum'):
            raise ValueError('managed original cost and run binding differ')
        result['worker_code_checksum']=descriptor.get('worker_code_checksum')
    return result
