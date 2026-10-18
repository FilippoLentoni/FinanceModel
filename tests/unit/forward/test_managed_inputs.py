"""Independent original worker evidence survives context change, day crossing and retry."""
import base64,copy,json
from types import SimpleNamespace
import pytest
from finplan_model.forward.managed_costs import import_pending,terminal_cost_event
from finplan_model.forward.managed_inputs import captured_event
from finplan_model.core.artifacts import canonical_json_bytes
from tests.unit.forward.test_managed_costs import Backend,Outbox
from tests.unit.benchmarks.test_forward_qwen import fixture,approved,Backend as QwenBackend,run_forward


def setup(*,failed=False,at='2026-07-11T23:59:00Z'):
    f=fixture();inp=approved(f);doc=run_forward(inp,backend=QwenBackend(f,inp,fail_at=2 if failed else None))
    run={**f.h.run(inp.spec['run_id']),'state':'failed' if failed else 'succeeded','job_started':True,'attempt':1,'result':doc,'completed_at':at,'job_finished_at':at}
    event=terminal_cost_event(run);outbox=Outbox(event);outbox.get_run=lambda _:copy.deepcopy(run)
    return f,inp,event,SimpleNamespace(runs=outbox,artifacts=inp.artifacts)


@pytest.mark.parametrize('failed',[False,True])
def test_original_immutable_worker_bytes_independent_of_new_context_and_actual_day(failed,tmp_path):
    f,inp,event,reader=setup(failed=failed)
    # Future context/book cannot replace any original bytes.
    f.base['holdings']['revision']=99;f.source['observations']=[]
    backend=Backend();result=import_pending(backend,reader,event['experiment_id'],at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)
    receipt=next(r for r in backend.writes if r['kind'].startswith('managed_inference_evidence:'))
    payload=receipt['payload']
    assert payload['origin_input_id']==f.binding['origin_input_id'] and payload['scheduled_slot']==f.binding['scheduled_slot']
    assert payload['evidence_status']==('original_partial_artifacts' if failed else 'complete_original_artifacts')
    for artifact in payload['artifacts']:
        raw=base64.b64decode(artifact['raw_base64']);assert raw==inp.artifacts.get(artifact['ref'])
    assert payload['worker_code_checksum']==inp.offline_code_checksum
    tasks=[json.loads(p.read_bytes()) for p in tmp_path.glob('*.json')]
    assert len(tasks)==2 and all(t['reporting_day']=='2026-07-11' for t in tasks)
    assert {t['event_id'] for t in tasks}=={event['event_id'],'managed_inputs:'+event['managed_run_id']}
    assert reader.runs.acks and result['imported'][0]['inference_evidence_ref']['checksum']==receipt['checksum']
    original=copy.deepcopy(backend.writes);import_pending(backend,reader,event['experiment_id'],at='2026-07-14T01:00:00Z',refresh_dir=tmp_path)
    assert backend.writes==original and len(tasks)==2


def test_transient_original_read_failure_never_acknowledges_or_regenerates(tmp_path,monkeypatch):
    f,inp,event,reader=setup(failed=True);backend=Backend();get=reader.artifacts.get
    monkeypatch.setattr(reader.artifacts,'get',lambda _:(_ for _ in ()).throw(RuntimeError('controlled S3 read unavailable')))
    with pytest.raises(RuntimeError):import_pending(backend,reader,event['experiment_id'],at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)
    assert not reader.runs.acks and len(backend.writes)==1
    monkeypatch.setattr(reader.artifacts,'get',get)
    import_pending(backend,reader,event['experiment_id'],at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)
    assert len(backend.writes)==2 and reader.runs.acks


def test_missing_hard_loss_originals_remain_unusable_and_queue_task_before_ack(tmp_path):
    f,inp,event,reader=setup(failed=True)
    event['source_evidence_refs']=[event['binding_ref']];reader.runs.event=event
    payload=captured_event(event,reader)
    assert payload['evidence_status']=='evidence_unavailable_or_indeterminate' and len(payload['artifacts'])==1
    backend=Backend();old_ack=reader.runs.ack_forward_cost_event
    def ack(*args):
        assert len(backend.writes)==2 and len(list(tmp_path.glob('*.json')))==2
        old_ack(*args)
    reader.runs.ack_forward_cost_event=ack
    import_pending(backend,reader,event['experiment_id'],at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)


def test_reference_scope_size_and_raw_checksum_mismatch_fail_closed(tmp_path):
    f,inp,event,reader=setup()
    for change in ('owner','size','checksum'):
        bad=copy.deepcopy(event)
        if change=='owner':bad['source_evidence_refs'][0]['owner']='financialplanning'
        elif change=='size':bad['source_evidence_refs'][0]['size_bytes']=9*2**20
        else:bad['source_evidence_refs'][0]['checksum']='sha256:'+'0'*64
        with pytest.raises(Exception):captured_event(bad,reader)


def test_jev_original_named_response_and_claim_links_are_captured_exactly():
    from tests.unit.benchmarks.test_forward_jev import fixture as jf,approved_input,client_factory
    from finplan_model.benchmarks.forward_jev import run_forward as jev_run
    f=jf();inp=approved_input(f);doc=jev_run(inp,client_factory=client_factory(f,[]))
    run={**f.h.run(inp.spec['run_id']),'state':'succeeded','job_started':True,'attempt':1,'result':doc,'completed_at':'2026-07-11T00:20:00Z'}
    event=terminal_cost_event(run);reader=SimpleNamespace(runs=SimpleNamespace(get_run=lambda _:run),artifacts=inp.artifacts)
    payload=captured_event(event,reader)
    assert payload['evidence_status']=='complete_original_artifacts'
    assert len(payload['artifacts'])>len(event['source_evidence_refs'])
    assert any('response_' in a['ref']['artifact_id'] for a in payload['artifacts'])


def test_omitted_qwen_role_ref_is_retrieved_from_typed_original_result_not_silently_complete():
    f,inp,event,reader=setup()
    doc=reader.runs.get_run(event['managed_run_id'])['result'];output=inp.artifacts.get_json(doc['payload']['forward_qwen']['result_ref'])
    omitted=output['role_receipts'][2]['input_ref']
    event['source_evidence_refs']=[ref for ref in event['source_evidence_refs'] if ref!=omitted]
    payload=captured_event(event,reader)
    assert payload['evidence_status']=='complete_original_artifacts'
    actual=next(a for a in payload['artifacts'] if a['ref']==omitted)
    assert base64.b64decode(actual['raw_base64'])==inp.artifacts.get(omitted)
    inp.artifacts.objects.pop(omitted['artifact_id'])
    with pytest.raises(Exception):captured_event(event,reader)


def test_typed_qwen_role_reference_must_match_role_index_and_original_input():
    f,inp,event,reader=setup();doc=reader.runs.get_run(event['managed_run_id'])['result']
    ref=doc['payload']['forward_qwen']['result_ref'];body=inp.artifacts.get_json(ref)
    body['role_receipts'][2]['input_ref']=body['role_receipts'][0]['input_ref']
    new=inp.artifacts.put_json(body,kind='run_artifact')
    event['source_evidence_refs']=[new.to_dict() if row==ref else row for row in event['source_evidence_refs']]
    with pytest.raises(ValueError,match='role content'):captured_event(event,reader)
