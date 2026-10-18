"""Captured older actions cannot disappear behind a later no-op book as_of."""
import copy
from types import SimpleNamespace
import pytest
from finplan_model.forward.corporate_actions import unreconciled_actions,book_inception
from finplan_model.forward.execution_reconcile import reconcile_executions
from test_execution_reconcile import Backend
from test_valuation import observed,quote

AT='2026-10-21T14:05:00Z'

def held():return {'as_of':'2026-10-20','positions':[{'instrument_id':'A','quantity':10.}],'cash_balance':1000.}
def source(day='2026-10-19',**fields):return {'execution_observations':[{'instrument_id':'A','session_date':day,'available_at':day+'T21:00:00Z',**fields}],'current_quotes':{'quotes':[]}}

@pytest.mark.parametrize('fields',[{'dividend':1.},{'split_ratio':2.}])
def test_later_rebalance_date_never_erases_older_captured_unreconciled_action(fields):
    rows=unreconciled_actions(held(),source(**fields),AT,book_created_at='2026-10-18T00:00:00Z')
    assert len(rows)==1 and rows[0]['original_evidence']==source(**fields)['execution_observations'][0]
    assert rows[0]['holding_since_basis']=='book_creation_conservative'
    assert len(unreconciled_actions(held(),source(**fields),AT))==1


def test_verified_creation_and_explicit_acquisition_exclude_only_demonstrably_older_events():
    assert not unreconciled_actions(held(),source('2026-10-16',dividend=1.),AT,book_created_at='2026-10-18T00:00:00Z')
    acquired=held();acquired['positions'][0]['acquired_at']='2026-10-20T14:00:00Z'
    assert not unreconciled_actions(acquired,source(dividend=1.),AT)
    assert len(unreconciled_actions(acquired,source('2026-10-20',dividend=1.),AT))==1


def test_future_unavailable_unheld_and_cash_only_sources_do_not_invent_current_action():
    assert not unreconciled_actions(held(),source('2026-10-22',split_ratio=2.),AT)
    future=source(dividend=1.);future['execution_observations'][0]['available_at']='2026-10-22T00:00:00Z'
    assert not unreconciled_actions(held(),future,AT)
    other=source(dividend=1.);other['execution_observations'][0]['instrument_id']='B'
    assert not unreconciled_actions(held(),other,AT)
    assert not unreconciled_actions({'positions':[]},source(dividend=1.),AT)


def test_daily_and_current_quote_layers_use_same_guard_and_keep_original_evidence():
    data=source(dividend=1.);event={'date':1792416600,'amount':.5}
    data['current_quotes']['quotes']=[{'instrument_id':'A','available_at':'2026-10-21T14:04:10Z','corporate_actions':{'dividends':{'1792416600':event}}}]
    original=copy.deepcopy(data);rows=unreconciled_actions(held(),data,AT)
    assert {r['layer'] for r in rows}=={'captured_daily_action','current_quote_dividends'}
    assert data==original and rows[1]['original_evidence']==event
    data['current_quotes']['quotes'][0]['corporate_actions']=['unparseable action']
    assert len(unreconciled_actions(held(),data,AT))==2


def test_valuation_cannot_mark_fresh_when_daily_action_is_absent_from_current_quote():
    data=source(dividend=1.);data.pop('current_quotes');data['execution_observations'][0]['close']=100.
    row=observed(quote(quoted_at='2026-10-21T14:04:00Z',available_at='2026-10-21T14:04:10Z'),at=AT,source_updates=data)
    assert row['value'] is None and row['observation_kind']=='pending'
    assert row['action_status']=='PENDING_ACTION_DATA' and row['corporate_action_evidence']
    cash=observed(cash_only=True,at=AT,source_updates=data)
    assert cash['value']==1000. and cash['observation_kind']=='valid'


def test_reconciliation_blocks_before_kernel_or_ledger_for_older_daily_action():
    backend=Backend();rid=backend.seed(1);run=backend.read_run(rid)
    backend.books[run['portfolio_id']]['paper_state'].update(positions=[{'instrument_id':'A','quantity':10.}],as_of='2026-01-03')
    data=source('2026-01-02',dividend=1.);data['investable_universe']=['A']
    result=reconcile_executions(backend,'e',data,'2026-01-03T00:05:00Z','2026-01-03T00:00:00Z',{},apply=lambda *a,**k:pytest.fail('kernel invoked with unreconciled action'),config={})
    assert not result['incomplete'] and result['approaches'][0]['pending_run_id']==rid
    attempt=next(v['payload'] for (kind,_),v in backend.records.items() if kind=='portfolio_observation')
    assert attempt['status']=='PENDING_ACTION_DATA' and not attempt['actual_fills'] and attempt['corporate_action_evidence']
    assert backend.books[run['portfolio_id']]['revision']==1


def test_inception_is_read_from_immutable_portfolio_metadata_not_latest_state():
    backend=SimpleNamespace(repo=SimpleNamespace(get=lambda kind,pid:SimpleNamespace(doc={'created_at':'2026-10-18T12:00:00Z'})))
    assert book_inception(backend,{'portfolio_id':'controlled'},held())=='2026-10-18T12:00:00Z'


def test_direct_paper_guard_respects_known_creation_and_does_not_invoke_kernel_for_held_action(monkeypatch):
    from finplan_model.forward.paper import apply_forward_quotes
    import finplan_model.evaluate.evaluator as evaluator
    called=[]
    def kernel(*args,**kwargs):called.append(True);raise RuntimeError('controlled kernel reached')
    monkeypatch.setattr(evaluator,'evaluate_point_rebalance',kernel)
    quotes={'quotes':[{'instrument_id':'A','corporate_actions':{'dividends':{'1':{'date':1792157400,'amount':1.}}},'available_at':'2026-10-21T14:04:10Z'}]}
    policy={'target_weights':[{'instrument_id':'A','weight':.5}],'cash_weight':.5}
    args=dict(decision_at='2026-10-20T14:05:00Z',opportunity_at=AT,recorded_at=AT,config={},universe=['A'])
    blocked=apply_forward_quotes(policy,held(),quotes,**args)
    assert not called and blocked['action_status']=='PENDING_ACTION_DATA'
    with pytest.raises(RuntimeError,match='controlled kernel reached'):
        apply_forward_quotes(policy,held(),quotes,book_created_at='2026-10-18T12:00:00Z',**args)
    assert called==[True]


@pytest.mark.parametrize('when',[True,False,'2026-10-16invalid',{'malformed':'date'}])
def test_malformed_event_time_cannot_be_dropped_as_preinception(when):
    data={'current_quotes':{'quotes':[{'instrument_id':'A','corporate_actions':{'dividends':{'event':{'date':when,'amount':1.}}}}]}}
    rows=unreconciled_actions(held(),data,AT,book_created_at='2026-10-18T12:00:00Z')
    assert len(rows)==1 and rows[0]['event_day'] is None


def test_future_acquisition_claim_cannot_exclude_captured_held_action():
    state=held();state['positions'][0]['acquired_at']='2026-10-22T14:00:00Z'
    assert unreconciled_actions(state,source(dividend=1.),AT)
