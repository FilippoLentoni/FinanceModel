from datetime import UTC,datetime
from types import SimpleNamespace
import copy
import pytest
from finplan_model.etf_news import extract,compare_optimizer,handle,FEATURE_ORDER
from finplan_model.classical.math import settings
from finplan_model.classical.storage import MemoryStore


def data():
    return {'decision_at':'2026-01-02T14:00:00Z','symbols':['SPY','VOO'],'contract_version':'1.8.0','historical_coverage':'forward_collection_only','deliveries':[{
        'record_id':'sha256:'+'a'*64,'provider':'synthetic','query':'synthetic ETF','symbols':['SPY','VOO'],'retrieved_at':'2026-01-02T13:00:00Z','available_at':'2026-01-02T13:01:00Z','quality_status':'validated','eligible':True,'exclusion_reasons':[],
        'rights':{'status':'research_only','scope':'search_metadata','evidence_url':'https://example.invalid/terms','evidence_checksum':'sha256:'+'b'*64},'payload_checksum':'sha256:'+'c'*64,'receipt_checksum':'sha256:'+'d'*64,
        'records':[{'title':'Synthetic ETF growth rises','url':'https://example.invalid/news','published_at':'2026-01-02T12:00:00Z','symbols':['SPY','VOO']}]}]}


def test_point_in_time_missingness_and_training_rights():
    d=data();features,proof,ctx=extract(d,symbols=['SPY','VOO'],decision_time=d['decision_at'])
    assert features['SPY']['sentiment']==1 and features['SPY']['missing']==0
    assert ctx[0]['trust']=='untrusted_source_metadata'
    train,_,_=extract(d,symbols=['SPY'],decision_time=d['decision_at'],training=True)
    assert train['SPY']['missing']==1
    future,_,_=extract(d,symbols=['SPY'],decision_time='2026-01-02T12:30:00Z')
    assert future['SPY']['missing']==1
    stale,_,_=extract(d,symbols=['SPY'],decision_time='2026-01-04T12:30:00Z')
    assert stale['SPY']['missing']==1


def test_unknown_rights_and_missing_publication_excluded():
    d=data();d['deliveries'][0]['rights']['status']='unverified'
    f,_,_=extract(d,symbols=['SPY'],decision_time=d['decision_at']);assert f['SPY']['missing']==1
    d=data();d['deliveries'][0]['records'][0]['published_at']=None
    f,_,_=extract(d,symbols=['SPY'],decision_time=d['decision_at']);assert f['SPY']['missing']==1


def test_matched_optimizer_sensitivity_consumes_actual_features_without_accuracy_claim():
    d=data();f,_,_=extract(d,symbols=['SPY','VOO'],decision_time=d['decision_at']);f['VOO']['sentiment']=-1
    inputs={'algorithm':'mean_variance','instruments':['SPY','VOO'],'settings':settings({'cash_weight':.1,'horizon_sessions':1,'turnover_penalty':0}),
            'current_weights':[.45,.45],'expected_returns':[0.,0.],'covariance':[[.001,0],[0,.001]],'scenarios':[[.01,-.01],[-.01,.01]],'as_of':'2026-01-02','decision_time':d['decision_at'],'input_snapshot_id':'synthetic'}
    original=copy.deepcopy(inputs);out=compare_optimizer(inputs,f)
    assert inputs==original and out['accuracy_improvement']=='not_measured'
    assert out['input_checksum']!=out['enriched_input_checksum']
    assert out['news_enriched']['weights'][0]>out['price_only']['weights'][0]
    f['VOO']['missing']=1;assert compare_optimizer(inputs,f)['status']=='ineligible'


def test_model_protocol_and_immutable_vector_order():
    d=data();svc=SimpleNamespace(env='beta',now=lambda:datetime(2026,1,2,14,tzinfo=UTC),d=SimpleNamespace(platform=SimpleNamespace(get_research_news=lambda symbols,at:d)),store=MemoryStore())
    out=handle(svc,{'symbols':['SPY','VOO']})
    assert out['models']['ppo']['feature_order']==list(FEATURE_ORDER)
    assert len(out['models']['ppo']['state_extension'])==10
    assert not out['models']['ppo']['eligible'] and not out['models']['ppo']['training_rights_eligible']
    assert out['models']['llm']['research_context_eligible']
    with pytest.raises(Exception):handle(svc,{'symbols':['BTC']})


def test_snapshot_baseline_filters_mixed_universe_and_records_price_provenance(monkeypatch):
    from datetime import timedelta
    from finplan_model.sim.market import MarketData,Bar
    from finplan_model.etf_news import snapshot_baseline
    import finplan_model.jobs.market_loader as loader
    start=datetime(2025,10,1,22,tzinfo=UTC)
    times=[start+timedelta(days=i) for i in range(70)]
    market=MarketData([t.date() for t in times],[Bar(s,t.date(),100+i+(i%3)*.1,t) for s in ('VOO','AAPL') for i,t in enumerate(times)],synthetic=True)
    content=SimpleNamespace(snapshot=SimpleNamespace(manifest_checksum='sha256:'+'a'*64))
    monkeypatch.setattr(loader,'load_market',lambda *args:(market,content))
    svc=SimpleNamespace(d=SimpleNamespace(platform=None));decision=times[-1].isoformat()
    inputs,reason=snapshot_baseline(svc,'synthetic-snapshot',['VOO'],decision,'cvar')
    assert reason is None and inputs['instruments']==['VOO'] and inputs['snapshot_checksum']==content.snapshot.manifest_checksum
    assert inputs['settings']['max_weight']==1 and inputs['estimation']['observations']>=20
    missing,reason=snapshot_baseline(svc,'synthetic-snapshot',['SPY'],decision,'cvar')
    assert missing is None and reason=='snapshot_missing_requested_etfs'
