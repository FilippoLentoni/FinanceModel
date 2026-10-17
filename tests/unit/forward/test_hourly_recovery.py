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
    def record_observation(self,experiment,aid,slot,pid,payload,at,kind='valuation'):
        oid=identifier('mo',experiment,aid,slot,kind);doc={'observation_id':oid,'portfolio_id':pid,'experiment_id':experiment,'approach_id':aid,'slot':slot,'observed_at':at,'kind':kind,'payload':copy.deepcopy(payload)}
        self.records.setdefault(('portfolio_observation',oid),doc);return doc
    def query_day(self,experiment,day,kind='runs',page_size=100,next_token=None):
        table={'runs':'research_run','observations':'portfolio_observation','inputs':'research_input'}[kind]
        return [copy.deepcopy(doc) for (k,_),doc in self.records.items() if k==table and doc['experiment_id']==experiment and doc['slot'][:10]==day],None
    def query_pending_forecasts(self,*args,**kwargs):return [],None


def source():
    start=date(2026,5,1)
    return {'investable_universe':['A','B'],'observations':[{'instrument_id':i,'session_date':(start+timedelta(days=n)).isoformat(),'close':100+n*(1 if i=='A' else .5),'available_at':'2026-07-10T00:00:00Z'} for n in range(65) for i in ('A','B')],'execution_observations':[]}


def adapters():
    def controlled(base,source,holdings,before):
        before({'boundary':'controlled_replay_stub','actual_source':{'news':[],'market_descriptors':{'trend':'flat'}},'messages':[]})
        return {'weights':{'A':.5,'B':.5},'cash':0,'forecast':{'available':False}}
    return {'codex':controlled,'langgraph':controlled}


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
