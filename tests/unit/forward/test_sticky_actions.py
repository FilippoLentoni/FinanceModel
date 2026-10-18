"""A captured unresolved entitlement cannot disappear with a source window/policy."""
import copy
from types import SimpleNamespace
import pytest
from finplan_model.forward.execution_reconcile import reconcile_executions
from finplan_model.forward.hourly import record_slot_observation
from test_execution_reconcile import Backend

FIRST='2026-01-03T00:05:00Z'


def setup():
    backend=Backend();rid=backend.seed(1);row=backend.read_run(rid)
    backend.books[row['portfolio_id']]['paper_state'].update(positions=[{'instrument_id':'A','quantity':10.}],as_of='2026-01-03')
    source={'investable_universe':['A'],'execution_observations':[{'instrument_id':'A','session_date':'2026-01-02','dividend':1.,'available_at':'2026-01-02T21:00:00Z'}],'current_quotes':{'quotes':[]}}
    return backend,rid,source


def reconcile(backend,source,slot='2026-01-03T00:00:00Z',at=FIRST):
    return reconcile_executions(backend,'e',source,at,slot,{},apply=lambda *a,**k:pytest.fail('unresolved action reached kernel'),config={})


def test_latched_original_content_survives_window_roll_and_policy_supersession():
    backend,rid,source=setup();first=reconcile(backend,source)
    origin=first['approaches'][0];reference=origin['unresolved_action_ref'];original_artifacts=copy.deepcopy(backend.artifacts)
    newer=backend.seed(49)
    second=reconcile(backend,{'investable_universe':['A'],'execution_observations':[],'current_quotes':{'quotes':[]}},'2026-01-03T02:00:00Z','2026-01-03T02:05:00Z')
    status=second['approaches'][0]
    assert not second['incomplete'] and status['unresolved_action_ref']==reference
    assert status['corporate_action_evidence']==origin['corporate_action_evidence']
    assert status['original_action_observed_at']==FIRST
    assert backend.read_run_header(rid)['status']=='superseded' and status['pending_run_id']==newer
    assert all(backend.artifacts[key]==raw for key,raw in original_artifacts.items())


def test_action_blocks_even_when_origin_policy_already_applied_and_no_policy_pending():
    backend,rid,source=setup();backend.records['research_run',rid].update(status='applied',execution_ref={'checksum':'fixture-applied'})
    result=reconcile(backend,source)
    assert not result['incomplete'] and result['approaches'][0]['pending_run_id'] is None
    assert result['approaches'][0]['action_status']=='PENDING_ACTION_DATA'


def test_late_action_can_latch_existing_frontier_without_rewriting_original_proof():
    backend,rid,source=setup()
    empty={'investable_universe':['A'],'execution_observations':[],'current_quotes':{'quotes':[]}}
    # With no action the fixture's kernel produces a genuine pending quote status.
    initial=reconcile_executions(backend,'e',empty,FIRST,'2026-01-03T00:00:00Z',{},apply=lambda *a,**k:{'status':'pending','reason':'market_closed'},config={})
    assert not initial['incomplete'];old=copy.deepcopy(backend.records['research_run',rid]['_frontier'])
    late=reconcile(backend,source,at='2026-01-03T00:06:00Z')
    assert not late['incomplete'] and late['approaches'][0]['unresolved_action_ref']
    assert backend.records['research_run',rid]['_frontier']==old


def test_sticky_valuation_copies_original_content_and_time_even_without_current_events():
    backend,rid,source=setup();summary=reconcile(backend,source);book=backend.get_portfolio(backend.read_run(rid)['portfolio_id'])
    current={'execution_observations':[],'current_quotes':{'quotes':[]},'execution_reconciliation':summary}
    holdings={'positions':[],'cash':1000.,'portfolio_value':1000.,'revision':1}
    record_slot_observation(backend,'e','a','2026-01-03T02:00:00Z','current-run','current-input',book,holdings,current,'2026-01-03T02:05:00Z',{'status':'succeeded'})
    payload=next(doc['payload'] for (kind,_),doc in backend.records.items() if kind=='portfolio_observation' and doc['kind']=='valuation')
    assert payload['value'] is None and payload['action_status']=='PENDING_ACTION_DATA'
    assert payload['unresolved_action_ref']==summary['approaches'][0]['unresolved_action_ref']
    assert payload['corporate_action_evidence']==summary['approaches'][0]['corporate_action_evidence']
    assert payload['original_action_observed_at']==FIRST and 'not new model context' in payload['action_evidence_usage']


def test_latch_survives_cleanup_failure_before_frontier_and_source_rollover():
    backend,rid,source=setup();backend.seed(2);backend.fail=rid
    first=reconcile(backend,source)
    assert first['incomplete'] and not backend.frontiers
    book=next(iter(backend.books.values()));reference=book['unresolved_action_ref']
    backend.fail=None
    next_result=reconcile(backend,{'investable_universe':['A'],'execution_observations':[],'current_quotes':{'quotes':[]}},'2026-01-03T01:00:00Z','2026-01-03T01:05:00Z')
    assert not next_result['incomplete'] and next_result['approaches'][0]['unresolved_action_ref']==reference


def test_valuation_uses_permanent_book_latch_even_if_reconciliation_failed_before_summary():
    backend,rid,source=setup();backend.seed(2);backend.fail=rid;result=reconcile(backend,source)
    assert result['incomplete'] and not result['approaches']
    book=backend.get_portfolio(backend.read_run(rid)['portfolio_id'])
    record_slot_observation(backend,'e','a','2026-01-03T02:00:00Z','current-run','current-input',book,{'positions':[],'cash':1000.,'portfolio_value':1000.,'revision':1},{'execution_observations':[],'current_quotes':{'quotes':[]},'execution_reconciliation':result},'2026-01-03T02:05:00Z',{'status':'succeeded'})
    valuation=next(doc['payload'] for (kind,_),doc in backend.records.items() if kind=='portfolio_observation' and doc['kind']=='valuation')
    assert valuation['value'] is None and valuation['action_status']=='PENDING_ACTION_DATA'


def test_active_lease_cannot_drop_action_before_recovery_and_source_rollover():
    from datetime import datetime,UTC
    backend,rid,source=setup();lease=datetime(2026,1,3,0,15,tzinfo=UTC).timestamp()
    backend.seed(2,status='running',lease=lease)
    first=reconcile(backend,source)
    assert first['incomplete'] and not backend.frontiers
    reference=next(iter(backend.books.values()))['unresolved_action_ref']
    assert first['approaches'][0]['unresolved_action_ref']==reference
    later=reconcile(backend,{'investable_universe':['A'],'execution_observations':[],'current_quotes':{'quotes':[]}},'2026-01-03T01:00:00Z','2026-01-03T01:05:00Z')
    assert not later['incomplete'] and later['approaches'][0]['unresolved_action_ref']==reference


def test_model_artifact_recovery_failure_cannot_prevent_action_latch():
    backend,rid,source=setup();original_read=backend.read_run
    backend.read_run=lambda *args:(_ for _ in ()).throw(RuntimeError('controlled missing old model artifact'))
    first=reconcile(backend,source)
    assert first['incomplete'] and next(iter(backend.books.values()))['unresolved_action_ref']
    reference=next(iter(backend.books.values()))['unresolved_action_ref'];backend.read_run=original_read
    rolled=reconcile(backend,{'investable_universe':['A'],'execution_observations':[],'current_quotes':{'quotes':[]}},'2026-01-03T01:00:00Z','2026-01-03T01:05:00Z')
    assert not rolled['incomplete'] and rolled['approaches'][0]['unresolved_action_ref']==reference
