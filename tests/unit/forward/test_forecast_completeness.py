"""Strong chronological proof is required before a permanent forecast match."""
import copy
from types import SimpleNamespace

import pytest

from finplan_model.forward.forecasts import freeze
from finplan_model.forward.hourly import _reconcile_forecasts,identifier

EXP='controlled_forecast_proof'
SLOT='2026-10-18T01:00:00Z'
AT='2026-10-18T03:05:00Z'


def observation(oid='first',value=1150):
    return {'observation_id':oid,'portfolio_id':'book','observed_at':'2026-10-18T02:05:00Z',
        'value_as_of':'2026-10-18T02:05:00Z','observation_kind':'valid','quote_status':'fresh',
        'value':value,'net_external_cash_flows_since_decision':50}


class Backend:
    def __init__(self,proof):
        self.proof=proof;self.docs={};self.writes=[];self.matches=[];self.proof_calls=[]
        self.repo=SimpleNamespace(get=lambda table,key:SimpleNamespace(doc=self.docs[key]) if key in self.docs else None)
        frozen=freeze({'target_at':'2026-10-18T02:00:00Z','expected_value':1100,'baseline_value':1000,'units':'USD'},
            decision_at='2026-10-18T01:05:00Z',run_id='run',portfolio_id='book',baseline_value=1000)
        self.run={'run_id':'run','approach_id':'controlled','portfolio_id':'book','slot':SLOT,'outcome':{'forecast':frozen}}
    def query_pending_forecasts(self,exp,at,**kwargs):return [copy.deepcopy(self.run)],None
    def query_observations_after(self,*args,**kwargs):pytest.fail('eventual GSI rows cannot prove first-valid absence')
    def strong_forecast_observation(self,rid,through,**kwargs):
        self.proof_calls.append((rid,through,kwargs));return copy.deepcopy(self.proof)
    def record_observation(self,exp,aid,slot,pid,payload,at,kind):
        oid=identifier('mo',exp,aid,slot,kind);doc={'observation_id':oid,'payload':copy.deepcopy(payload)}
        self.docs[oid]=doc;self.writes.append(doc);return doc
    def mark_forecast_realized(self,rid,oid,at):self.matches.append((rid,oid,at))


def test_hidden_gsi_earlier_mark_is_selected_from_strong_proof_not_later_visible_mark():
    backend=Backend({'status':'complete','observation':observation(),'proof_ref':{'checksum':'controlled-proof'}})
    original=copy.deepcopy(backend.run['outcome']['forecast'])
    result=_reconcile_forecasts(backend,EXP,AT)
    assert result['status']=='complete' and result['forecasts'][0]['observation_id']=='first'
    assert backend.writes[0]['payload']['signed_error']==0
    assert backend.writes[0]['payload']['absolute_error']==0 and backend.matches
    assert backend.run['outcome']['forecast']==original
    assert backend.proof_calls==[('run',AT,{'max_slots':168})]


@pytest.mark.parametrize('status,reason',[('pending_incomplete','scheduled_run_or_valuation_not_terminal_or_missing'),
    ('pending_incomplete','bounded_forecast_search_continues'),('pending','no_valid_quote_yet')])
def test_missing_original_or_bounded_chunk_retains_pending_without_any_match(status,reason):
    backend=Backend({'status':status,'reason':reason,'next_slot':'2026-10-18T02:00:00Z',
        'proof_ref':{'checksum':'partial-proof'},'observation':None})
    result=_reconcile_forecasts(backend,EXP,AT)
    assert result['status']=='pending' and result['forecasts'][0]['reason']==reason
    assert not backend.writes and not backend.matches
    backend.proof={'status':'complete','observation':observation()}
    assert _reconcile_forecasts(backend,EXP,AT)['status']=='complete' and len(backend.matches)==1


def test_retry_proves_original_match_again_without_duplicate_receipt_or_mutation():
    backend=Backend({'status':'complete','observation':observation()})
    _reconcile_forecasts(backend,EXP,AT);old=copy.deepcopy(backend.writes)
    _reconcile_forecasts(backend,EXP,AT)
    assert backend.writes==old and len(backend.matches)==2 and len(backend.proof_calls)==2


def test_old_wrong_later_immutable_match_is_visible_blocker_and_never_rewritten():
    backend=Backend({'status':'complete','observation':observation()})
    from finplan_model.forward.forecasts import reconcile
    bad=reconcile(backend.run['outcome']['forecast'],[observation('later',1250)])
    oid=identifier('mo',EXP,'controlled',SLOT,'forecast_match');backend.docs[oid]={'observation_id':oid,'payload':bad}
    result=_reconcile_forecasts(backend,EXP,AT)
    assert result['forecasts'][0]['reason']=='immutable_match_receipt_conflicts_with_first_valid_proof'
    assert backend.docs[oid]['payload']==bad and not backend.writes and not backend.matches


def test_backend_without_strong_proof_does_not_fallback_to_eventual_index():
    backend=Backend({});backend.strong_forecast_observation=None
    result=_reconcile_forecasts(backend,EXP,AT)
    assert result['forecasts'][0]['reason']=='strong_forecast_observation_backend_unavailable'
    assert not backend.writes and not backend.matches


def test_invalid_complete_future_or_pretarget_proof_never_becomes_realized():
    row=observation();row['value_as_of']='2026-10-18T01:59:59Z'
    backend=Backend({'status':'complete','observation':row})
    with pytest.raises(ValueError,match='eligible original'):_reconcile_forecasts(backend,EXP,AT)
    assert not backend.writes and not backend.matches


def test_future_to_cutoff_complete_proof_is_rejected_without_a_match():
    row=observation();row.update(observed_at='2026-10-18T04:05:00Z',value_as_of='2026-10-18T04:05:00Z')
    backend=Backend({'status':'complete','observation':row})
    with pytest.raises(ValueError,match='future observation'):_reconcile_forecasts(backend,EXP,AT)
    assert not backend.writes and not backend.matches
