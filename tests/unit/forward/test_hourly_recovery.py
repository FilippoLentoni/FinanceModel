"""Orchestration recovery on an in-memory backend; storage atomicity has platform tests."""
import copy
import json
from datetime import date,timedelta
from types import SimpleNamespace
import pytest
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.forward.hourly import identifier,run_hourly


class FakeEvidence:
    def __init__(self):self.records={};self.artifacts={};self.books={};self.registered={};self.calls=[];self.repo=self
    def get(self,kind,rid):return SimpleNamespace(doc=copy.deepcopy(self.records[kind,rid]),id=rid) if (kind,rid) in self.records else None
    def require(self,kind,rid):return self.get(kind,rid)
    def read_registry(self,experiment):return copy.deepcopy(self.registered.get(experiment))
    def register_approaches(self,experiment,approaches,policy,at):self.registered[experiment]={'approaches':copy.deepcopy(approaches),'comparison_policy':copy.deepcopy(policy)}
    def ensure_portfolio(self,experiment,aid,capital,universe,at):
        pid=identifier('book',experiment,aid)
        if pid not in self.books:self.books[pid]={'portfolio_id':pid,'revision':1,'paper_state':{'positions':[],'cash_balance':capital,'high_watermark':capital,'as_of':at[:10],'mode':'paper','source':'paper_initialization','base_currency':'USD'}}
        return self.get_portfolio(pid)
    def get_portfolio(self,pid):return copy.deepcopy(self.books[pid])
    def capture_input(self,experiment,aid,slot,payload,shared,at,kind='model_input'):
        iid=identifier('mi',experiment,aid,slot,kind);raw=canonical_json_bytes(payload)
        if iid in self.artifacts:
            assert self.artifacts[iid]==raw
            return self.records['research_input',iid]
        doc={'experiment_id':experiment,'approach_id':aid,'input_id':iid,'slot':slot,'checksum':sha256_checksum(raw),'captured_at':at,'kind':kind}
        self.records['research_input',iid]=doc;self.artifacts[iid]=raw;return copy.deepcopy(doc)
    def capture_shared_source(self,experiment,slot,payload,at):return self.capture_input(experiment,'__shared__',slot,payload,None,at,kind='shared_source')
    def read_input(self,iid):return {'metadata':copy.deepcopy(self.records['research_input',iid]),'payload':json.loads(self.artifacts[iid])}
    def start_run(self,experiment,aid,slot,iid,pid,at,metadata):
        rid=identifier('mr',experiment,aid,slot)
        self.records.setdefault(('research_run',rid),{'run_id':rid,'experiment_id':experiment,'approach_id':aid,'slot':slot,'input_id':iid,'portfolio_id':pid,'portfolio_revision':self.books[pid]['revision'],'started_at':at,'status':'running','metadata':metadata})
        return self.records['research_run',rid]
    def claim_run(self,rid,owner,at,lease_seconds):self.calls.append(rid);return True
    def finish_run(self,rid,outcome,at):self.records['research_run',rid].update(status=outcome['status'],outcome=copy.deepcopy(outcome),finished_at=at)
    def read_run(self,rid):return copy.deepcopy(self.records['research_run',rid])
    def record_observation(self,experiment,aid,slot,pid,payload,at,kind='valuation',*,recorded_at=None):
        oid=identifier('mo',experiment,aid,slot,kind);doc={'observation_id':oid,'portfolio_id':pid,'experiment_id':experiment,'approach_id':aid,'slot':slot,'observed_at':at,'recorded_at':recorded_at or at,'kind':kind,'run_id':payload.get('run_id'),'input_id':payload.get('input_id'),'payload':copy.deepcopy(payload),'checksum':sha256_checksum(canonical_json_bytes(payload))}
        self.records.setdefault(('portfolio_observation',oid),doc);return doc
    def query_day(self,experiment,day,kind='runs',page_size=100,next_token=None):
        table={'runs':'research_run','observations':'portfolio_observation','inputs':'research_input'}[kind]
        return [copy.deepcopy(doc) for (k,_),doc in self.records.items() if k==table and doc['experiment_id']==experiment and doc['slot'][:10]==day],None
    def read_run_header(self,rid):return {k:copy.deepcopy(v) for k,v in self.records['research_run',rid].items() if k!='outcome'}
    def query_run_headers(self,experiment,aid,before_slot,*,page_size=100,next_token=None):
        rows=[self.read_run_header(rid) for (kind,rid),doc in self.records.items() if kind=='research_run' and doc['experiment_id']==experiment and doc['approach_id']==aid and doc['slot']<before_slot]
        rows.sort(key=lambda r:r['slot'],reverse=True);offset=int(next_token or 0)
        return rows[offset:offset+page_size],str(offset+page_size) if offset+page_size<len(rows) else None
    def read_execution_frontier(self,rid):
        row=self.records['research_run',rid];proof=copy.deepcopy(row.get('_frontier'))
        if proof is not None and self.books[row['portfolio_id']].get('unresolved_action_ref'):proof['unresolved_action_ref']=self.books[row['portfolio_id']]['unresolved_action_ref']
        return proof
    def record_execution_frontier(self,rid,pending_run_id,at,previous_frontier_run_id=None,unresolved_action_ref=None):
        row=self.records['research_run',rid]
        if unresolved_action_ref:self.latch_action_blocker(rid,unresolved_action_ref,at)
        if row.get('_frontier'):return self.read_execution_frontier(rid)
        prior=self.read_execution_frontier(previous_frontier_run_id) if previous_frontier_run_id else None
        sticky=self.books[row['portfolio_id']].get('unresolved_action_ref') or (prior or {}).get('unresolved_action_ref')
        proof={'pending_run_id':pending_run_id,'covered_through_slot':row['slot'],'unresolved_action_ref':sticky}
        row.update(execution_frontier_ref={'checksum':'controlled'},_frontier=proof);return self.read_execution_frontier(rid)
    def mark_execution_superseded(self,rid,replacement,at):
        self.records['research_run',rid].update(status='superseded',execution_disposition_ref={'replacement_run_id':replacement})
        return {'status':'superseded','replacement_run_id':replacement}
    def latch_action_blocker(self,rid,ref,at):
        row=self.records['research_run',rid];book=self.books[row['portfolio_id']]
        book.setdefault('unresolved_action_ref',ref)
        return self.read_action_blocker(book['unresolved_action_ref'],experiment_id=row['experiment_id'],approach_id=row['approach_id'],portfolio_id=row['portfolio_id'],through=at)
    def read_action_blocker(self,ref,*,experiment_id,approach_id,portfolio_id,through):
        doc=copy.deepcopy(self.records['portfolio_observation',ref['observation_id']]);payload=doc['payload']
        assert doc['checksum']==ref['checksum'] and doc['experiment_id']==experiment_id and doc['approach_id']==approach_id and doc['portfolio_id']==portfolio_id
        assert payload['status']=='PENDING_ACTION_DATA' and payload['corporate_action_evidence']
        return {'ref':ref,'metadata':doc,'payload':payload}
    def query_pending_forecasts(self,*args,**kwargs):return [],None


def source():
    start=date(2026,5,1)
    return {'investable_universe':['A','B'],'observations':[{'instrument_id':i,'session_date':(start+timedelta(days=n)).isoformat(),'close':100+n*(1 if i=='A' else .5),'available_at':'2026-07-10T00:00:00Z'} for n in range(65) for i in ('A','B')],'execution_observations':[]}


def adapters():
    def controlled(base,source,holdings,before):
        before({'boundary':'controlled_replay_stub','actual_source':{'news':[],'market_descriptors':{'trend':'flat'}},'messages':[]})
        return {'weights':{'A':.5,'B':.5},'cash':0,'forecast':{'available':False}}
    from finplan_model.forward.public_preflight import preflight
    def local_preflight(kind):
        return lambda base,source,holdings,before:preflight(kind,base,source,holdings,before,dependency_reader=lambda k:{'status':'controlled_metadata_fixture'})
    return {'codex':controlled,'langgraph':controlled,'jev':local_preflight('jev'),'qwen':local_preflight('qwen')}


def run(e,path,slot='2026-07-11T00:00:00Z',**kwargs):
    return run_hourly(slot,'controlled-recovery',path,backend=e,source=source(),policies={},research={},now=slot.replace(':00:00Z',':05:00Z'),external_adapters=adapters(),**kwargs)


def test_duplicate_slot_does_not_invoke_again_or_replace_original_sources(tmp_path):
    e=FakeEvidence();first=run(e,tmp_path);calls=list(e.calls);raw=dict(e.artifacts)
    second=run(e,tmp_path)
    assert e.calls==calls
    assert e.artifacts==raw
    assert len(second['approaches'])==len(first['approaches'])
    assert all(row['retry']=='existing_terminal' for row in second['approaches'])
    assert all(row['portfolio_id'] for row in e.registered['controlled-recovery']['approaches'])


def test_terminal_crash_repairs_missing_observation_using_original_holdings(tmp_path):
    e=FakeEvidence();result=run(e,tmp_path);aid='cash'
    oid=identifier('mo','controlled-recovery',aid,result['slot'],'valuation')
    expected=copy.deepcopy(e.records['portfolio_observation',oid]['payload']);del e.records['portfolio_observation',oid]
    pid=next(a['portfolio_id'] for a in result['approaches'] if a['approach_id']==aid)
    e.books[pid]['paper_state']['cash_balance']=77;e.books[pid]['revision']=10
    run(e,tmp_path)
    assert e.records['portfolio_observation',oid]['payload']==expected


def test_missed_historical_slot_has_all_explicit_statuses_and_no_replacement_data(tmp_path):
    e=FakeEvidence();result=run(e,tmp_path,account_missed=True)
    assert all(row['status']=='skipped' for row in result['approaches'])
    shared=e.read_input(result['shared_input_id'])['payload']
    assert shared['observations']==[] and shared['news']==[]
    assert shared['historical_slot_missing'] is True
    assert len([r for (k,_),r in e.records.items() if k=='portfolio_observation'])==len(result['approaches'])


def test_current_slot_active_600second_lease_stays_incomplete_and_recovers_without_reinference(tmp_path):
    class Leased(FakeEvidence):
        def claim_run(self,rid,owner,at,lease_seconds):
            from datetime import datetime
            row=self.records['research_run',rid]
            now=datetime.fromisoformat(at.replace('Z','+00:00')).timestamp()
            if row.get('lease_until',0)>now:return False
            self.calls.append(rid);row['lease_until']=now+lease_seconds;return True
    e=Leased();result=run(e,tmp_path);rid=identifier('mr','controlled-recovery','cash',result['slot'])
    from datetime import datetime
    row=e.records['research_run',rid];row['status']='running';row['lease_until']=datetime.fromisoformat('2026-07-11T00:15:00+00:00').timestamp()
    row.pop('outcome');row.pop('finished_at');calls=list(e.calls);original=dict(e.artifacts)
    active=run(e,tmp_path)
    assert active['status']=='incomplete' and active['incomplete_approaches'][0]['run_id']==rid
    assert e.calls==calls and e.artifacts==original
    recovered=run_hourly(result['slot'],'controlled-recovery',tmp_path,backend=e,source=source(),policies={},research={},now='2026-07-11T00:15:00Z',external_adapters={'codex':lambda *a:pytest.fail('terminal adapter repeated'),'langgraph':lambda *a:pytest.fail('terminal adapter repeated')})
    assert recovered['status']=='complete'
    assert e.calls==calls+[rid] and e.artifacts==original
    assert e.records['research_run',rid]['outcome']['reason']=='recovered_interrupted_invocation_original_input_preserved'


def test_hourly_cli_incomplete_returns_temporary_failure(monkeypatch,tmp_path):
    from finplan_model.forward import hourly
    import sys
    monkeypatch.setattr(sys,'argv',['hourly','--output',str(tmp_path)])
    monkeypatch.setattr(hourly,'run_hourly',lambda *a,**k:{'status':'incomplete'})
    assert hourly.main()==75


def test_136_terminal_approaches_and_one_active_lease_do_not_reopen_model_artifacts(tmp_path):
    from finplan_model.forward.hourly import _register
    from finplan_model.forward.registry import Approach
    class Active(FakeEvidence):
        def claim_run(self,*a,**k):return False
    e=Active();experiment='controlled-many-terminal';slot='2026-07-11T00:00:00Z';at='2026-07-11T00:05:00Z'
    rows=[Approach('controlled_'+str(n),'Controlled replay '+str(n),'control','baseline').to_dict() for n in range(137)]
    data=source();data['registry_approaches']=rows;data['availability_cutoff']=at
    _register(e,experiment,rows,data,at,slot)
    shared=e.capture_shared_source(experiment,slot,data,at)
    for index,row in enumerate(rows):
        aid=row['approach_id'];book=e.ensure_portfolio(experiment,aid,100000,data['investable_universe'],at)
        item=e.capture_input(experiment,aid,slot,{'holdings':{'positions':[],'cash':100000,'portfolio_value':100000}},shared,at)
        rid=e.start_run(experiment,aid,slot,item['input_id'],book['portfolio_id'],at,row)['run_id']
        if index<136:
            e.finish_run(rid,{'status':'failed','reason':'Controlled terminal fixture'},at)
            e.record_observation(experiment,aid,slot,book['portfolio_id'],{'run_id':rid,'input_id':item['input_id'],'portfolio_id':book['portfolio_id'],'value':None},at)
        else:e.records['research_run',rid]['lease_until']=1783728900
    original_read=e.read_input
    e.read_input=lambda iid: original_read(iid) if iid==shared['input_id'] else pytest.fail('terminal model input reopened')
    e.read_run=lambda rid:pytest.fail('terminal outcome reopened')
    result=run_hourly(slot,experiment,tmp_path,backend=e,now=at)
    assert result['status']=='incomplete' and len(result['approaches'])==137
    assert len([r for r in result['approaches'] if r.get('retry')=='existing_terminal'])==136


def test_wrong_bound_valuation_fails_instead_of_skipping_original_evidence(tmp_path):
    e=FakeEvidence();result=run(e,tmp_path)
    oid=identifier('mo','controlled-recovery','cash',result['slot'],'valuation')
    e.records['portfolio_observation',oid]['run_id']='controlled-wrong-run'
    with pytest.raises(ValueError,match='original run/input/portfolio'):run(e,tmp_path)


def test_terminal_repair_preserves_mark_time_but_records_actual_later_availability(tmp_path,monkeypatch):
    from finplan_model.forward import hourly
    e=FakeEvidence();run(e,tmp_path)
    original_slot='2026-07-11T00:00:00Z';original_at='2026-07-11T00:05:00Z'
    oid=next(rid for (kind,rid),doc in e.records.items() if kind=='portfolio_observation')
    old=copy.deepcopy(e.records.pop(('portfolio_observation',oid)))
    monkeypatch.setattr(hourly,'stamp',lambda:'2026-07-11T00:20:00Z')
    run_hourly(original_slot,'controlled-recovery',tmp_path,backend=e,source=source(),policies={},research={},external_adapters=adapters())
    restored=e.records['portfolio_observation',oid]
    assert restored['observed_at']==original_at and restored['payload']['observed_at']==original_at
    assert restored['recorded_at']=='2026-07-11T00:20:00Z'
    assert all(doc['recorded_at']==original_at for (kind,rid),doc in e.records.items() if kind=='portfolio_observation' and rid!=oid)


NINE=['cash','buy_and_hold','equal_weight','mean_variance','min_variance','classical_cvar','ppo','hosted_langgraph','direct_codex_news']


def test_experiment_runs_and_registers_only_the_selected_approaches(tmp_path):
    e=FakeEvidence();result=run(e,tmp_path,approach_ids=NINE)
    assert [row['approach_id'] for row in result['approaches']]==NINE
    assert [a['approach_id'] for a in e.registered['controlled-recovery']['approaches']]==NINE
    books={r['portfolio_id'] for (k,_),r in e.records.items() if k=='portfolio_observation'}
    assert len(books)==len(NINE)


def test_unknown_selected_approach_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='unknown approach ids: not_a_model'):
        run(FakeEvidence(),tmp_path,approach_ids=['cash','not_a_model'])
