"""Controlled restart clock: no real model invocation, retrieval, or AWS request."""
import copy
from datetime import UTC,datetime
from finplan_model.forward import research,hourly
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from test_hourly_recovery import FakeEvidence


class Clock:
    now=0.
    def __init__(self):self.sleeps=[]
    def time(self):return self.now
    def sleep(self,delay):self.sleeps.append(delay);self.now+=delay


def test_original_600second_interim_lease_recovers_failed_with_original_input_and_no_reviewer(tmp_path,monkeypatch):
    clock=Clock();slot='2026-07-11T00:30:00Z';start=datetime.fromisoformat(slot.replace('Z','+00:00')).timestamp()
    class Leased(FakeEvidence):
        def claim_run(self,rid,owner,at,lease_seconds):
            row=self.records['research_run',rid]
            if row.get('lease_until',0)>start+clock.now:return False
            self.calls.append(rid);return True
        def read_run(self,rid):
            row=super().read_run(rid)
            if row['status']!='running':
                row['input_checksum']=self.records['research_input',row['input_id']]['checksum']
                row['output_ref']={'checksum':sha256_checksum(canonical_json_bytes(row['outcome']))}
            return row
    e=Leased();interim='controlled-interim';aid='direct_codex_news'
    book=e.ensure_portfolio(interim,aid,100000,['A'],slot)
    item=e.capture_input(interim,aid,slot,{'numeric':{'prepared':{'actual_source':{'investable_universe':['A']}}}},None,slot)
    rid=e.start_run(interim,aid,slot,item['input_id'],book['portfolio_id'],slot,{})['run_id']
    e.records['research_run',rid]['lease_until']=start+600
    original=copy.deepcopy(e.artifacts)
    monkeypatch.setattr(hourly,'stamp',lambda:datetime.fromtimestamp(start+clock.now,UTC).isoformat().replace('+00:00','Z'))
    monkeypatch.setattr(hourly,'review',lambda *a,**k:(_ for _ in ()).throw(AssertionError('reviewer reinvoked')))
    monkeypatch.setattr(hourly,'load_current_inputs',lambda *a,**k:(_ for _ in ()).throw(AssertionError('source refetched')))
    result=research.run_cycle_with_recovery(slot,'controlled',tmp_path,runner=lambda *args:hourly.run_interim(*args,backend=e),clock=clock.time,sleep=clock.sleep)
    assert result['status']=='failed' and clock.sleeps==[30.]*20
    assert e.artifacts==original and e.calls==[rid]
    assert e.records['research_run',rid]['outcome']['reason']=='recovered_interrupted_interim_original_inputs_preserved'
    assert (tmp_path/'cycle-receipt.json').is_file()


def test_cycle_incomplete_at_deadline_has_no_receipt_and_cli_temporary_exit(tmp_path,monkeypatch):
    clock=Clock();calls=[]
    def active(*args):calls.append(args);raise hourly.NotClaimed('original owner active')
    result=research.run_cycle_with_recovery('2026-07-11T00:30:00Z','controlled',tmp_path,runner=active,clock=clock.time,sleep=clock.sleep,timeout=65)
    assert result['status']=='incomplete' and clock.sleeps==[30.,30.,5.]
    assert len(set(calls))==1 and not (tmp_path/'cycle-receipt.json').exists()
    import sys
    monkeypatch.setattr(sys,'argv',['research','cycle','--slot','2026-07-11T00:30:00Z','--experiment','controlled','--output',str(tmp_path)])
    monkeypatch.setattr(research,'run_cycle_with_recovery',lambda *a:result)
    assert research.main()==75
