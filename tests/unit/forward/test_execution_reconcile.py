"""Bounded execution cleanup; immutable model artifacts are expanded only on demand."""
import copy
from datetime import UTC,datetime,timedelta
import pytest
from finplan_model.forward.execution_reconcile import reconcile_executions,recover_expired_run
from test_hourly_recovery import FakeEvidence


class Backend(FakeEvidence):
    def __init__(self):
        super().__init__();self.expanded=[];self.superseded=[];self.queries=[];self.frontiers=[];self.fail=None
        self.registered['e']={'approaches':[{'approach_id':'a'}],'comparison_policy':{}}
    def seed(self,n,status='succeeded',aid='a',lease=0):
        slot=(datetime(2026,1,1,tzinfo=UTC)+timedelta(hours=n)).isoformat().replace('+00:00','Z')
        book=self.ensure_portfolio('e',aid,100000,['A'],slot)
        item=self.capture_input('e',aid,slot,{'holdings':{'positions':[],'revision':1},'availability_cutoff':slot},None,slot)
        row=self.start_run('e',aid,slot,item['input_id'],book['portfolio_id'],slot,{})
        row.update(status=status,lease_until=lease)
        if status!='running':row.update(outcome={'status':status,'target_weights':{'A':1}},finished_at=slot)
        return row['run_id']
    def read_run(self,rid):self.expanded.append(rid);return super().read_run(rid)
    def query_run_headers(self,*args,**kwargs):self.queries.append(args);return super().query_run_headers(*args,**kwargs)
    def mark_execution_superseded(self,rid,replacement,at):
        if rid==self.fail:raise RuntimeError('controlled failed disposition')
        self.superseded.append(rid);return super().mark_execution_superseded(rid,replacement,at)
    def record_execution_frontier(self,*args,**kwargs):self.frontiers.append(args);return super().record_execution_frontier(*args,**kwargs)
    def claim_run(self,rid,owner,at,lease_seconds):
        row=self.records['research_run',rid];now=datetime.fromisoformat(at.replace('Z','+00:00')).timestamp()
        if row.get('lease_until',0)>now:return False
        self.calls.append(rid);row['lease_until']=now+lease_seconds;return True


def reconcile(b,**kwargs):
    return reconcile_executions(b,'e',{'investable_universe':['A'],'current_quotes':{}},'2026-01-03T00:05:00Z','2026-01-03T00:00:00Z',{},apply=lambda *a,**k:{'status':'pending','reason':'market_closed'},config={},**kwargs)


def test_oldest_first_cleanup_expands_only_latest_policy_and_frontier_reuses_pointer():
    b=Backend();ids=[b.seed(n) for n in range(24)]
    first=reconcile(b)
    assert not first['incomplete'] and b.superseded==ids[:-1]
    assert b.expanded==[ids[-1]]
    b.expanded.clear();b.queries.clear();b.superseded.clear()
    # Failed hours still carry a verified frontier and retain the older policy.
    b.seed(24,status='unavailable');second=reconcile(b)
    assert not second['incomplete'] and not b.superseded
    assert b.expanded==[ids[-1]] and len(b.queries)==1
    assert second['approaches'][0]['pending_run_id']==ids[-1]


def test_cleanup_failure_does_not_skip_older_or_advance_frontier_and_other_approach_continues():
    b=Backend();ids=[b.seed(n) for n in range(3)];b.fail=ids[0]
    b.registered['e']['approaches'].append({'approach_id':'b'});other=b.seed(2,aid='b')
    result=reconcile(b)
    assert result['failures'][0]['approach_id']=='a'
    assert ids[1] not in b.superseded
    assert [args[0] for args in b.frontiers]==[other]


def test_active_original_600s_lease_blocks_frontier_then_expired_recovery_preserves_input():
    b=Backend();now=datetime(2026,1,3,0,5,tzinfo=UTC).timestamp();rid=b.seed(1,'running',lease=now+600)
    original=copy.deepcopy(b.artifacts);first=reconcile(b)
    assert first['incomplete'] and not b.frontiers and not b.calls
    result=recover_expired_run(b,b.read_run_header(rid),'2026-01-03T00:15:00Z')
    assert result['status']=='recovered_failed' and b.calls==[rid]
    assert b.artifacts==original and not b.expanded
    observations=[v for (k,_),v in b.records.items() if k=='portfolio_observation']
    assert len(observations)==1 and observations[0]['payload']['value'] is None
    assert observations[0]['payload']['quote_status']=='not_observed'
    assert not reconcile(b)['incomplete'] and len(b.calls)==1


def test_metadata_overflow_is_explicit_and_never_advances_proof():
    b=Backend();[b.seed(n) for n in range(10)]
    result=reconcile(b,max_pages=1)
    assert result['incomplete'] and 'exhausted' in result['failures'][0]['reason']
    assert not b.expanded and not b.frontiers


def test_late_retry_only_repairs_dispositions_and_never_executes_original_quote():
    b=Backend();b.seed(1);b.seed(2)
    result=reconcile_executions(b,'e',{'investable_universe':['A']},'2026-01-03T00:20:00Z','2026-01-03T00:00:00Z',{},apply=lambda *a,**k:pytest.fail('late retry must not fill'),config={},allow_execution=False)
    assert not result['incomplete'] and len(b.superseded)==1
    assert not any(k=='portfolio_observation' for k,_ in b.records)


def test_manual_before_horizon_is_nontrading_even_when_only_pending_policy():
    b=Backend();rid=b.seed(0);b.registered['e']['comparison_policy']={'first_scheduled_slot':'2026-01-02T00:00:00Z','cadence_seconds':3600}
    def exclude(rid,at):
        b.records['research_run',rid].update(status='execution_excluded',execution_disposition_ref={'status':'excluded_nontrading'})
    b.mark_execution_ineligible=exclude
    result=reconcile_executions(b,'e',{'investable_universe':['A']},'2026-01-02T00:05:00Z','2026-01-02T00:00:00Z',{},apply=lambda *a,**k:pytest.fail('manual policy must never fill'),config={})
    assert not result['incomplete'] and not b.expanded
    assert b.records['research_run',rid]['status']=='execution_excluded'


def test_missing_expected_head_cannot_be_hidden_by_empty_gsi():
    b=Backend();b.registered['e']['comparison_policy']={'first_scheduled_slot':'2026-01-02T23:00:00Z','cadence_seconds':3600}
    result=reconcile(b)
    assert result['incomplete'] and not b.frontiers


def test_gsi_omitted_latest_head_is_strongly_recovered_before_selection():
    b=Backend();rid=b.seed(47);b.registered['e']['comparison_policy']={'first_scheduled_slot':'2026-01-02T23:00:00Z','cadence_seconds':3600}
    b.query_run_headers=lambda *a,**k:([],None)
    result=reconcile(b)
    assert not result['incomplete'] and b.expanded==[rid]


def test_same_slot_pending_attempt_retry_preserves_original_receipt():
    b=Backend();b.seed(1);reconcile(b)
    before=copy.deepcopy([v for (k,_),v in b.records.items() if k=='portfolio_observation'])
    again=reconcile(b)
    assert not again['incomplete']
    assert [v for (k,_),v in b.records.items() if k=='portfolio_observation']==before


def test_recovery_read_failure_keeps_running_head_for_retry_not_terminal_missing_evidence():
    b=Backend();rid=b.seed(1,'running');original_read=b.read_input
    b.read_input=lambda *a:(_ for _ in ()).throw(RuntimeError('temporary artifact outage'))
    with pytest.raises(RuntimeError,match='artifact outage'):
        recover_expired_run(b,b.read_run_header(rid),'2026-01-03T00:05:00Z')
    assert b.records['research_run',rid]['status']=='running'
    b.read_input=original_read
    assert recover_expired_run(b,b.read_run_header(rid),'2026-01-03T00:15:00Z')['status']=='recovered_failed'
