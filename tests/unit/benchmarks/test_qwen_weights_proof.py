"""Controlled metadata-only freeze/ready proof; no checkpoint download or GPU."""
import base64,copy,io,json
from types import SimpleNamespace
import pytest
from finplan_model.benchmarks.offline import OfflineBatchIO
from finplan_model.benchmarks.qwen import MODEL_ID,REVISION
from finplan_model.benchmarks.weights import PREFIX
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.benchmarks.forward_qwen import run_forward,checksum,journal_id
from tests.unit.benchmarks.test_forward_qwen import fixture,approved,Backend


def gateway(f):
    manifest=''.join('0'*64+'  '+name+'\n' for name in ('config.json','tokenizer_config.json','LICENSE','controlled.safetensors')).encode()
    staged={'model_id':MODEL_ID,'revision':REVISION,'license':'Apache-2.0','file_count':4,'manifest_checksum':sha256_checksum(manifest)}
    objects={PREFIX+'STAGED.json':canonical_json_bytes(staged),PREFIX+'MANIFEST.sha256':manifest};gets=[]
    class Storage:
        def get_object(self,**kw):gets.append(kw['Key']);return {'Body':io.BytesIO(objects[kw['Key']])}
        def put_object(self,**kw):objects[kw['Key']]=kw['Body']
    return OfflineBatchIO(Storage(),'example-research-bucket',f.platform,f.h.run_io,f.h.deps.artifacts),objects,gets


def test_precompute_weights_proof_cas_persists_into_exact_spec_and_bundle():
    f=fixture();inp=approved(f);io_backend,objects,gets=gateway(f)
    f.h.deps.offline_batch=io_backend;f.h.settings.leases={'gpu':1}
    f.h.settings.job_definitions['swarm_mode_a']['code_prefix']='releases/qwen-code/controlled/'
    f.h.service.dispatch();run=f.h.run(inp.spec['run_id']);descriptor=run['forward_qwen']
    proof=f.h.deps.artifacts.get_json(descriptor['weights_proof_ref'])
    spec=f.h.run_io.get_spec(run['run_id'],None)
    bundle=json.loads(objects['scratch/offline/'+run['run_id']+'/input.json'])
    assert spec['forward_qwen']==bundle['spec']['forward_qwen']
    assert spec['forward_qwen']['weights_proof_ref']==descriptor['weights_proof_ref']
    assert proof['manifest_checksum']==descriptor['weights_manifest_checksum']
    assert base64.b64decode(proof['manifest_raw_base64'])==objects[PREFIX+'MANIFEST.sha256']
    assert base64.b64decode(proof['staged_raw_base64'])==objects[PREFIX+'STAGED.json']
    assert bundle['forward_qwen_weights_proof']==proof
    assert all(k.endswith(('STAGED.json','MANIFEST.sha256')) for k in gets)
    assert len([op for op,_ in f.h.sagemaker.calls if op=='CreateTrainingJob'])==1


def test_original_frozen_manifest_and_handoff_reject_drift_no_new_proof():
    f=fixture();inp=approved(f);io_backend,objects,gets=gateway(f);run=f.h.run(inp.spec['run_id'])
    definition={**f.h.settings.job_definitions['swarm_mode_a'],'code_prefix':'releases/qwen-code/controlled/'}
    freeze=io_backend.prepare_forward_weights(run,definition);run['forward_qwen'].update(freeze)
    original=f.h.deps.artifacts.get(freeze['weights_proof_ref'])
    assert io_backend.prepare_forward_weights(run,definition)==freeze
    objects[PREFIX+'MANIFEST.sha256']=objects[PREFIX+'MANIFEST.sha256'].replace(b'0',b'1')
    staged=json.loads(objects[PREFIX+'STAGED.json']);staged['manifest_checksum']=sha256_checksum(objects[PREFIX+'MANIFEST.sha256']);objects[PREFIX+'STAGED.json']=canonical_json_bytes(staged)
    with pytest.raises(FinplanError) as error:io_backend.prepare_forward_weights(run,definition)
    assert error.value.details['reason']=='qwen_weights_changed'
    assert f.h.deps.artifacts.get(freeze['weights_proof_ref'])==original
    assert not f.h.sagemaker.calls


def test_verified_backend_ready_is_sealed_before_first_fake_engine_call(monkeypatch):
    f=fixture();inp=approved(f);io_backend,_,_=gateway(f);run=f.h.run(inp.spec['run_id'])
    freeze=io_backend.prepare_forward_weights(run,f.h.settings.job_definitions['swarm_mode_a']);inp.spec['forward_qwen'].update(freeze)
    proof=inp.artifacts.get_json(freeze['weights_proof_ref']);backend=Backend(f,inp)
    backend.manifest={k:proof[k] for k in ('model_id','revision','license','file_count','manifest_checksum')}
    call=backend.__class__.__call__;seen=[]
    class VerifiedFake:
        manifest=backend.manifest
        def __call__(self,messages):
            ready_ref=inp.artifacts.resolve(kind='run_artifact',artifact_id=journal_id(f.binding,'backend_ready'));ready=inp.artifacts.get_json(ready_ref)
            assert ready['manifest_checksum']==proof['manifest_checksum'] and ready['weights_proof_ref']==freeze['weights_proof_ref']
            assert ready['manifest_raw_base64']==proof['manifest_raw_base64'];seen.append(messages)
            return call(backend,messages)
    import finplan_model.benchmarks.forward_qwen as module
    monkeypatch.setattr(module,'VllmBackend',lambda *a,**kw:VerifiedFake())
    result=run_forward(inp)
    assert result['completion_status']=='succeeded' and len(seen)==5
    output=inp.artifacts.get_json(result['payload']['forward_qwen']['result_ref']);ready=inp.artifacts.get_json(output['backend_ready_ref'])
    assert ready['status']=='verified_pinned_checkpoint' and ready['license']=='Apache-2.0' and ready['file_count']==4
    assert freeze['weights_proof_ref'] in result['artifacts']
