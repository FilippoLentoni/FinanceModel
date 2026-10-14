from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import copy
import pytest
from finplan_model.crypto_research import advance, new_book, rebalance, recommendations, run_hour, validate_dataset, iso, PRODUCTS
from finplan_model.classical.service import ClassicalService
from finplan_model.classical.storage import MemoryStore


def dataset(end):
    rows={}
    for j,p in enumerate(PRODUCTS):
        rows[p]=[{'start':iso(end-timedelta(hours=48-i)),'open':100+j*50+i,'close':100.5+j*50+i,'low':99+j*50+i,'high':102+j*50+i,'volume':10} for i in range(48)]
    return {'dataset_id':'coinbase-crypto-hourly-v1','as_of':iso(end),'available_at':iso(end+timedelta(minutes=5)),
            'granularity_seconds':3600,'market_data_synthetic':True,'products':PRODUCTS,'bars':rows,'sources':[],
            'dataset_checksum':'sha256:'+'a'*64,'contract_version':'1.7.0'}


def test_crypto_optimizers_and_incompatible_models():
    d=dataset(datetime(2026,1,3,tzinfo=UTC));r=recommendations(d)
    for name in ('min_variance','mean_variance','cvar'):
        assert r[name]['eligible'], r[name]
        assert sum(r[name]['weights'].values())==pytest.approx(.9)
        assert max(r[name]['weights'].values())<=.600001
    assert not r['ppo']['eligible'] and not r['llm']['eligible']


def test_fills_only_after_decision_available_and_charge_costs():
    end=datetime(2026,1,3,tzinfo=UTC);d=dataset(end);plans=recommendations(d)
    books=advance(None,d,plans,end+timedelta(minutes=10))
    assert all(b['value']==100 and b['fills']==[] for b in books.values())
    previous={'as_of':d['as_of'],'books':books}
    d2=dataset(end+timedelta(hours=1));books2=advance(previous,d2,recommendations(d2),end+timedelta(hours=1,minutes=10))
    assert books2['btc_buy_hold']['fills']==[]  # next open at +1h has not completed yet
    d3=dataset(end+timedelta(hours=2));books3=advance({'as_of':d2['as_of'],'books':books2},d3,recommendations(d3),end+timedelta(hours=2,minutes=10))
    fill=books3['btc_buy_hold']['fills'][0]
    assert fill['bar_start']==iso(end+timedelta(hours=1)) and fill['fee']>0
    assert books3['btc_buy_hold']['cash']>=0
    assert books3['cash']['value']==100


def test_costed_rebalance_never_overdraws_cash():
    book=new_book();fills=rebalance(book,{'BTC-USD':1.0},{p:100 for p in PRODUCTS},'2026-01-01T00:00:00Z')
    assert book['cash']>=0 and fills[0]['quantity']<1
    assert book['cumulative_fees']>0 and book['cumulative_slippage']>0


def test_snapshot_gaps_and_future_availability_rejected():
    end=datetime(2026,1,3,tzinfo=UTC);d=dataset(end)
    with pytest.raises(Exception):validate_dataset(d,end)
    bad=copy.deepcopy(d);bad['bars']['BTC-USD'].pop(2)
    with pytest.raises(Exception):validate_dataset(bad,end+timedelta(minutes=10))


def test_hourly_replay_and_out_of_order_are_safe():
    end=datetime(2026,1,3,tzinfo=UTC);d=dataset(end)
    service=ClassicalService(env='beta',now=lambda:end+timedelta(minutes=10),store=MemoryStore(),deps=SimpleNamespace(project_budget=lambda:{'spent':1,'limit':100},platform=SimpleNamespace(get_crypto_dataset=lambda:d)))
    first=run_hour(service);second=run_hour(service)
    assert first['analysis_id']==second['analysis_id']
    assert second['experiment']['observation_count']==1


def test_controller_recomputes_targets_after_pending_fills(monkeypatch):
    import finplan_model.crypto_research as crypto
    end=datetime(2026,1,3,tzinfo=UTC);current={'end':end}
    service=ClassicalService(env='beta',now=lambda:current['end']+timedelta(minutes=10),store=MemoryStore(),deps=SimpleNamespace(project_budget=lambda:{'spent':1,'limit':100},platform=SimpleNamespace(get_crypto_dataset=lambda:dataset(current['end']))))
    seen=[];original=crypto.recommendations
    def capture(data,books=None):
        seen.append(copy.deepcopy(books));return original(data,books)
    monkeypatch.setattr(crypto,'recommendations',capture)
    for i in range(3):
        current['end']=end+timedelta(hours=i);run_hour(service)
    assert seen[0]['min_variance']['quantities']['BTC-USD']==0
    assert seen[2]['min_variance']['quantities']['BTC-USD']>0
    assert seen[2]['min_variance']['cumulative_fees']>0


def test_retry_repairs_the_index_after_a_record_only_commit():
    class IndexFailureStore(MemoryStore):
        def __init__(self):super().__init__();self.indexed=set();self.fail=True
        def put(self,doc):
            result=super().put(doc)
            if self.fail:
                self.fail=False
                raise RuntimeError('injected failure after record before index')
            self.indexed.add(doc['analysis_id']);return result
        def list(self,**kwargs):return [d for d in super().list(**kwargs) if d['analysis_id'] in self.indexed]
    end=datetime(2026,1,3,tzinfo=UTC);current={'end':end};store=IndexFailureStore()
    service=ClassicalService(env='beta',now=lambda:current['end']+timedelta(minutes=10),store=store,deps=SimpleNamespace(project_budget=lambda:{'spent':1,'limit':100},platform=SimpleNamespace(get_crypto_dataset=lambda:dataset(current['end']))))
    with pytest.raises(RuntimeError):run_hour(service)
    recovered=run_hour(service)
    assert recovered['analysis_id'] in store.indexed
    current['end']=end+timedelta(hours=1)
    next_record=run_hour(service)
    assert next_record['experiment']['parent_analysis_id']==recovered['analysis_id']
    assert next_record['experiment']['observation_count']==2
