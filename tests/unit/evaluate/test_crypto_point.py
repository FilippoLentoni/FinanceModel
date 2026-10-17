"""Native hourly-quote point fills, controlled fixtures only."""
import copy
from datetime import UTC,datetime,timedelta

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.evaluate.evaluator import evaluate_point_rebalance
from finplan_model.forward.crypto_hourly import CRYPTO_POLICY,source_from_dataset
from finplan_model.forward.hourly import evaluator_config
from finplan_model.sim.config import SimulationConfig


END=datetime(2026,1,17,12,tzinfo=UTC)  # Saturday; 24/7 crypto, closed XNYS.

def iso(t):return t.isoformat().replace('+00:00','Z')


def dataset(end=END):
    products=['BTC-USD','ETH-USD']
    rows={p:[{'start':iso(end-timedelta(hours=48-i)),'open':100+j*50+i,'close':100.5+j*50+i,
              'low':99+j*50+i,'high':102+j*50+i,'volume':1000000.} for i in range(48)] for j,p in enumerate(products)}
    return {'dataset_id':'coinbase-crypto-hourly-v1','as_of':iso(end),'available_at':iso(end+timedelta(minutes=5)),
            'granularity_seconds':3600,'market_data_synthetic':True,'products':products,'bars':rows,'sources':[],
            'dataset_checksum':'sha256:'+'a'*64,'contract_version':'1.7.0'}


def source(end=END):
    return source_from_dataset(dataset(end),acquired_at=iso(end+timedelta(minutes=5)),slot=iso(end),allow_synthetic=True)


def point(quotes=None,**kwargs):
    arguments={'positions':{},'cash':100000.,'quotes':quotes or {q['instrument_id']:q for q in source()['current_quotes']['quotes']},
               'config':SimulationConfig.from_dict(evaluator_config(CRYPTO_POLICY)),
               'decision_at':iso(END-timedelta(minutes=55)),'opportunity_at':iso(END+timedelta(minutes=5)),
               'recorded_at':iso(END+timedelta(minutes=5)),'universe':['BTC-USD','ETH-USD'],
               'execution_calendar':'CRYPTO_24X7','quote_protocol':CRYPTO_POLICY['quote_protocol'],'max_quote_age_seconds':1200}
    arguments.update(kwargs)
    return evaluate_point_rebalance({'weights':{'BTC-USD':.5,'ETH-USD':.5},'cash':0.,'solution_status':'feasible'},**arguments)


def test_actual_completed_hour_closes_and_volume_are_preserved_without_minute_fabrication():
    data=dataset();s=source();quotes=s['current_quotes']['quotes']
    assert s['observations']==[] and s['execution_observations']==[]
    for q in quotes:
        original=data['bars'][q['instrument_id']][-1]
        assert q['price']==original['close'] and q['volume']==original['volume']
        assert q['frequency']=='1h' and q['bar_started_at']==original['start']
        assert q['quoted_at']==data['as_of'] and q['platform_available_at']==data['available_at']
    result=point()
    assert len(result['fills'])==2
    assert result['cash']>=0
    assert result['after_value']+result['transaction_cost']==pytest.approx(100000.)
    assert result['transaction_cost']==pytest.approx(sum(f['notional'] for f in result['fills'])*.007)
    assert result['identity']['execution_calendar']=='CRYPTO_24X7'
    assert result['identity']['liquidity_units']=='actual_base_asset_units_per_completed_hour'
    assert all(q['frequency']=='1h' for q in result['quote_provenance'].values())


def test_weekend_crypto_enabled_without_changing_closed_equity_behavior():
    assert point()['fills']
    quotes={q['instrument_id']:{**q,'bar_started_at':'2026-01-17T11:59:00Z','market_open':False,'regular_session':{}} for q in source()['current_quotes']['quotes']}
    with pytest.raises(FinplanError) as error:
        point(quotes,execution_calendar='XNYS',quote_protocol='native_completed_1m_unadjusted_close/1')
    assert error.value.details['reason']=='market_closed'


@pytest.mark.parametrize('change,reason',[
    ({'provider':'unverified provider'},'incompatible_crypto_quote'),
    ({'frequency':'1m'},'incompatible_crypto_quote'),
    ({'venue_calendar':'XNYS'},'incompatible_crypto_quote'),
    ({'bar_started_at':'2026-01-17T11:59:00Z'},'incompatible_crypto_quote'),
    ({'available_at':'2026-01-17T12:06:00Z'},'missed_eligible_execution_price'),
    ({'quoted_at':'2026-01-17T11:00:00Z'},'missed_eligible_execution_price'),
    ({'volume':None},'execution_volume_missing'),
    ({'status':'stale'},'fresh_execution_quote_missing'),
])
def test_future_stale_or_wrong_provider_hourly_quotes_cannot_fill(change,reason):
    quotes={q['instrument_id']:q for q in source()['current_quotes']['quotes']}
    quotes['BTC-USD'].update(change)
    with pytest.raises(FinplanError) as error:point(quotes)
    assert error.value.details['reason']==reason


def test_source_must_match_hour_and_cannot_be_backdated_or_stale():
    data=dataset()
    for acquired,slot in [(iso(END),iso(END)),(iso(END+timedelta(minutes=21)),iso(END)),
                          (iso(END+timedelta(minutes=5)),iso(END+timedelta(hours=1)))]:
        with pytest.raises(Exception):source_from_dataset(data,acquired_at=acquired,slot=slot,allow_synthetic=True)
    with pytest.raises(ValueError,match='actual approved'):
        source_from_dataset(data,acquired_at=iso(END+timedelta(minutes=5)),slot=iso(END))


def test_prior_decision_required_and_same_hour_close_cannot_fill_itself():
    with pytest.raises(FinplanError):point(decision_at=iso(END+timedelta(minutes=5)))
