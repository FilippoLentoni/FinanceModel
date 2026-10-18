"""Processing release provenance for controlled missing/failed slots, never historical model inputs."""
import copy
import pytest
from finplan_model.forward import hourly
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from test_hourly_recovery import FakeEvidence,run


def versions(monkeypatch,letter):
    values={'financemodel':letter*40,'financialplanning':chr(ord(letter)+1)*40,'financeagent':chr(ord(letter)+2)*40}
    for key,repo in [('FINPLAN_MODEL_SOURCE_COMMIT','financemodel'),('FINPLAN_PLATFORM_SOURCE_COMMIT','financialplanning'),('FINPLAN_AGENT_SOURCE_COMMIT','financeagent')]:monkeypatch.setenv(key,values[repo])
    return values


def captured(e,slot):
    return [e.read_input(rid)['payload'] for (kind,rid),row in e.records.items() if kind=='research_input' and row['slot']==slot and row['kind']=='model_input']


def test_missed_hour_keeps_all_three_processing_versions_and_no_historical_data(tmp_path,monkeypatch):
    expected=versions(monkeypatch,'a');e=FakeEvidence();result=run(e,tmp_path,account_missed=True)
    shared=e.read_input(result['shared_input_id'])['payload']
    assert shared['code_versions']==expected and shared['processing_code_versions']==expected
    assert shared['observations']==[] and shared['news']==[]
    inputs=captured(e,result['slot'])
    assert len(inputs)==len(result['approaches'])
    assert all(row['code_versions']==expected and row['processing_code_versions']==expected for row in inputs)
    assert all(row['invocation_state']=='not_invoked' and row['code_version_scope']=='workflow_processing_only; model_not_invoked' for row in inputs)
    assert all(row['metadata']['code_versions']==expected and row['metadata']['processing_code_versions']==expected for (kind,_),row in e.records.items() if kind=='research_run')


def test_source_failure_captures_actual_processing_versions_for_every_approach(tmp_path,monkeypatch):
    expected=versions(monkeypatch,'b');e=FakeEvidence();run(e,tmp_path/'first')
    monkeypatch.setattr(hourly,'load_current_inputs',lambda **kwargs:(_ for _ in ()).throw(RuntimeError('controlled source failure')))
    slot='2026-07-11T01:00:00Z'
    result=hourly.run_hourly(slot,'controlled-recovery',tmp_path/'failed',backend=e,research={'news':[]},now='2026-07-11T01:05:00Z')
    shared=e.read_input(result['shared_input_id'])['payload']
    assert shared['code_versions']==expected and shared['processing_code_versions']==expected
    assert shared['observations']==[] and 'controlled source failure' in shared['acquisition_failure']
    assert all(row['code_versions']==expected and row['processing_code_versions']==expected for row in captured(e,slot))


def test_new_process_retry_preserves_original_earlier_code_versions_and_bytes(tmp_path,monkeypatch):
    earlier=versions(monkeypatch,'a');e=FakeEvidence();first=run(e,tmp_path,account_missed=True)
    original=copy.deepcopy(e.artifacts);records=copy.deepcopy(e.records)
    versions(monkeypatch,'d');second=run(e,tmp_path,account_missed=True)
    assert second['status']=='complete' and e.artifacts==original and e.records==records
    assert all(row['code_versions']==earlier for row in captured(e,first['slot']))


def test_interim_failed_source_records_all_processing_releases_without_reviewer(tmp_path,monkeypatch):
    class ReceiptEvidence(FakeEvidence):
        def read_run(self,rid):
            row=super().read_run(rid)
            row['input_checksum']=self.records['research_input',row['input_id']]['checksum']
            if row['status']!='running':row['output_ref']={'checksum':sha256_checksum(canonical_json_bytes(row['outcome']))}
            return row
    expected=versions(monkeypatch,'c');e=ReceiptEvidence();run(e,tmp_path/'primary')
    slot='2026-07-11T00:30:00Z';interim='controlled-recovery-interim'
    e.capture_input(interim,'__research_source__',slot,{'status':'failed','acquisition_failure':'controlled provider outage','news':[],'universe':None,'coverage':[],'workers':[]},None,slot,kind='research_source')
    monkeypatch.setattr(hourly,'review',lambda *a,**k:pytest.fail('reviewer invoked on failed source'))
    result=hourly.run_interim(slot,'controlled-recovery',tmp_path/'interim',backend=e)
    assert result['status']=='failed'
    row=e.read_input(result['input_id'])['payload']
    assert row['code_versions']==expected and row['processing_code_versions']==expected
    assert row['invocation_state']=='not_invoked'
