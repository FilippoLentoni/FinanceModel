"""Durable ancillary costs survive no signal reuse, retries and report failures."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from finplan_model.forward.managed_costs import import_pending,terminal_cost_event
from finplan_model.forward.report_refresh import enqueue
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum


def cost_event():
    return {'protocol':'finplan-managed-model-cost/1','event_id':'managed:controlled-job',
        'experiment_id':'controlled','approach_id':'jev_systemone','portfolio_id':'controlled-book',
        'managed_run_id':'controlled-job','occurred_at':'2026-07-11T23:59:00Z',
        'job_state':'failed','invocation_state':'indeterminate','compute_observed_usd':None,
        'compute_estimated_usd':.0675,'vendor_observed_usd':None,'vendor_estimated_usd':None}


class Backend:
    def __init__(self):self.docs={};self.writes=[];self.repo=SimpleNamespace(get=self.get)
    def get(self,table,key):return SimpleNamespace(doc=self.docs[key]) if key in self.docs else None
    def record_observation(self,exp,aid,slot,pid,payload,observed_at,kind):
        oid='mo_'+sha256_checksum(canonical_json_bytes([exp,aid,slot,kind])).split(':')[1]
        doc={'observation_id':oid,'experiment_id':exp,'approach_id':aid,'slot':slot,'portfolio_id':pid,'kind':kind,'checksum':sha256_checksum(canonical_json_bytes(payload)),'observed_at':observed_at,'payload':copy.deepcopy(payload)}
        self.docs[oid]=doc;self.writes.append(doc);return doc
    def read_artifact_bytes(self,doc):return canonical_json_bytes(doc['payload'])


class Outbox:
    def __init__(self,event,*,fail_ack=False,more=False):self.event=event;self.fail_ack=fail_ack;self.more=more;self.acks=[]
    def query_forward_cost_events(self,exp,limit):
        assert exp==self.event['experiment_id'] and limit<=100
        return [copy.deepcopy(self.event)],'scoped-more' if self.more else None
    def ack_forward_cost_event(self,exp,eid,ref,at):
        if self.fail_ack:raise RuntimeError('controlled interruption before ack')
        self.acks.append((exp,eid,ref,at))


def test_import_original_cost_bytes_actual_day_actual_available_time_and_queue_before_ack(tmp_path):
    event=cost_event();outbox=Outbox(event);backend=Backend()
    result=import_pending(backend,SimpleNamespace(runs=outbox),'controlled',at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)
    doc=backend.writes[0]
    assert doc['slot']==event['occurred_at'] and doc['observed_at']=='2026-07-13T01:00:00Z'
    assert backend.read_artifact_bytes(doc)==canonical_json_bytes(event)
    task=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert task['reporting_day']=='2026-07-11' and task['checksum']==doc['checksum']
    assert result['affected_reporting_days']==['2026-07-11'] and outbox.acks
    assert next(tmp_path.glob('*.json')).stat().st_mode&0o777==0o600


def test_crash_after_receipt_queue_before_ack_reuses_original_bytes_without_new_observation(tmp_path):
    outbox=Outbox(cost_event(),fail_ack=True);backend=Backend()
    with pytest.raises(RuntimeError):import_pending(backend,SimpleNamespace(runs=outbox),'controlled',at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)
    original=copy.deepcopy(backend.writes[0]);outbox.fail_ack=False
    result=import_pending(backend,SimpleNamespace(runs=outbox),'controlled',at='2026-07-14T01:00:00Z',refresh_dir=tmp_path)
    assert backend.writes==[original] and len(list(tmp_path.glob('*.json')))==1 and result['status']=='complete'


def test_same_timestamp_different_jobs_have_isolated_permanent_cost_receipts(tmp_path):
    backend=Backend();a=cost_event();b={**a,'event_id':'managed:second-job','managed_run_id':'second-job'}
    for event in (a,b):import_pending(backend,SimpleNamespace(runs=Outbox(event)),'controlled',at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)
    assert len(backend.docs)==2 and len(list(tmp_path.glob('*.json')))==2


def test_missing_or_full_durable_queue_never_acknowledges_cost(tmp_path,monkeypatch):
    monkeypatch.delenv('FINPLAN_REPORT_REFRESH_DIR',raising=False)
    box=Outbox(cost_event());backend=Backend()
    with pytest.raises(ValueError):import_pending(backend,SimpleNamespace(runs=box),'controlled',at='2026-07-13T01:00:00Z')
    assert not box.acks and len(backend.docs)==1
    task={'experiment_id':'controlled','event_id':'managed:one','reporting_day':'2026-07-11'}
    enqueue(tmp_path,task,max_pending=1)
    with pytest.raises(ValueError):enqueue(tmp_path,{**task,'event_id':'managed:two'},max_pending=1)
    assert len(list(tmp_path.glob('*.json')))==1


def test_unknown_billing_never_becomes_zero_or_elapsed_rate_observed():
    run={'run_id':'controlled-job','state':'failed','completed_at':'2026-07-13T01:00:00Z','job_started':True,'attempt':1,
        'forward_jev':{'experiment_id':'controlled','approach_id':'jev_systemone','portfolio_id':'controlled-book','binding_ref':{},'binding_checksum':'controlled','worker_image_digest':'controlled'},
        'result':{'payload':{'forward_jev':{'invocation_state':'indeterminate','usage':None}}},'cost_estimate':{'estimated_usd_upper_bound':.0675}}
    event=terminal_cost_event(run)
    assert event['compute_observed_usd'] is None and event['vendor_observed_usd'] is None
    assert event['compute_estimated_usd']==.0675
    run.update(job_started=False,attempt=0)
    event=terminal_cost_event(run)
    assert event['compute_observed_usd']==event['vendor_observed_usd']==0


def test_equivalent_offset_event_retry_normalizes_only_metadata_slot_not_original_bytes(tmp_path):
    event=cost_event();event['occurred_at']='2026-07-11T23:59:00+00:00'
    outbox=Outbox(event,fail_ack=True);backend=Backend()
    with pytest.raises(RuntimeError):import_pending(backend,SimpleNamespace(runs=outbox),'controlled',at='2026-07-13T01:00:00Z',refresh_dir=tmp_path)
    outbox.fail_ack=False
    import_pending(backend,SimpleNamespace(runs=outbox),'controlled',at='2026-07-14T01:00:00Z',refresh_dir=tmp_path)
    assert len(backend.writes)==1 and backend.writes[0]['slot']=='2026-07-11T23:59:00Z'
    assert backend.read_artifact_bytes(backend.writes[0])==canonical_json_bytes(event)


def test_concurrent_producers_never_exceed_durable_capacity_or_drop_tasks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def write(i):
        task={'experiment_id':'controlled','event_id':'managed:'+str(i),'reporting_day':'2026-07-11'}
        try:return enqueue(tmp_path,task,max_pending=3)
        except ValueError as error:
            assert 'queue full' in str(error)
            return None
    with ThreadPoolExecutor(max_workers=12) as workers:paths=list(workers.map(write,range(24)))
    winners=[path for path in paths if path]
    assert len(winners)==len(list(tmp_path.glob('*.json')))==3
    assert all(Path(path).stat().st_mode&0o777==0o600 for path in winners)
    for path in winners:
        task=json.loads(Path(path).read_bytes())
        assert enqueue(tmp_path,task,max_pending=3)==path
