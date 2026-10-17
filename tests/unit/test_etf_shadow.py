import copy
from datetime import datetime,UTC,timedelta
import pytest
from finplan_model.etf_shadow import run_study
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from tests.unit.test_etf_news import data


def study():
    decision=datetime(2026,1,2,14,tzinfo=UTC)
    history=[{'session_at':(decision-timedelta(days=21-i)).isoformat(),'available_at':(decision-timedelta(days=21-i)).isoformat(),
              'prices':{'VOO':100+(i%3)*.01,'GOOGL':100+(i%4)*.01}} for i in range(21)]
    step={'decision_at':decision.isoformat(),'snapshot_available_at':(decision-timedelta(minutes=1)).isoformat(),'snapshot_checksum':sha256_checksum(canonical_json_bytes(history)),
          'history':history,'next_open_at':(decision+timedelta(hours=1)).isoformat(),'close_at':(decision+timedelta(hours=2)).isoformat(),
          'next_open':{'VOO':100,'GOOGL':100},'close':{'VOO':110,'GOOGL':90}}
    return {'symbols':['VOO','GOOGL'],'market_data_synthetic':True,'fee_bps':10,'slippage_bps':5,'steps':[step],'algorithm':'mean_variance'}


def news():
    d=data();d['symbols']=['VOO','GOOGL'];d['deliveries'][0]['symbols']=['VOO','GOOGL'];d['deliveries'][0]['records'][0]['symbols']=['VOO']
    second=copy.deepcopy(d['deliveries'][0]['records'][0]);second.update(title='Synthetic stock loss warning',url='https://example.invalid/googl',symbols=['GOOGL'])
    d['deliveries'][0]['records'].append(second)
    return d


def test_realized_costed_matched_performance_and_timing():
    s=study();n=news();before=copy.deepcopy((s,n));r=run_study(s,n)
    assert r['status']=='completed' and (s,n)==before
    for arm in ('price_only','news_enriched'):
        m=r['performance'][arm];assert m['initial_value']==100 and m['fees']>0 and m['slippage_cost']>0
        assert m['total_cost']==pytest.approx(m['fees']+m['slippage_cost']) and m['outcome_observations']==1
        assert all(fill['at']==s['steps'][0]['next_open_at'] for fill in r['books'][arm]['fills'])
    assert r['decisions'][0]['common_inputs_checksum']!=r['decisions'][0]['news_inputs_checksum']
    assert r['accuracy_improvement']=='not_established' and not r['ppo']['eligible']


def test_missing_historical_rights_and_vintage_emit_no_performance():
    s=study();n=news();s['steps'][0]['snapshot_available_at']='2026-01-03T00:00:00Z'
    r=run_study(s,n);assert r['status']=='ineligible' and r['performance'] is None
    assert any(x['reason']=='price_snapshot_vintage_unavailable_at_decision' for x in r['blockers'])
    s=study();n['deliveries'][0]['available_at']='2026-01-03T00:00:00Z'
    assert run_study(s,n)['performance'] is None
    n=news();n['deliveries'][0]['rights']['status']='unverified'
    assert run_study(study(),n)['performance'] is None


def test_no_backdated_fills_tampered_history_or_unsupported_capital():
    s=study();s['steps'][0]['next_open_at']=s['steps'][0]['decision_at']
    with pytest.raises(ValueError):run_study(s,news())
    s=study();s['steps'][0]['history'][0]['prices']['VOO']=101
    with pytest.raises(ValueError,match='checksum'):run_study(s,news())
    s=study();s['initial_cash']=10000
    with pytest.raises(ValueError):run_study(s,news())


def test_forward_initialization_has_no_invented_outcomes():
    s=study();s['steps']=[];r=run_study(s,news())
    assert r['status']=='ineligible' and r['performance'] is None
    assert r['news_corpus_checksum'].startswith('sha256:')
    assert 'no_realized_forward_outcomes' in r['blockers'][0]['reason']
    assert r['prospective_protocol']['initial_cash']==100
