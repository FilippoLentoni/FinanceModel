import copy
from datetime import datetime,UTC,timedelta
import numpy as np
import pytest
from finplan_model.rl.arrays import PriceArrays
from finplan_model.rl.spec import EnvSpec
from finplan_model.rl.etf_news import FrozenEtfNewsProvider
from finplan_model.rl.train import preflight_news_training,train_policy
from tests.unit.test_etf_news import data
from tests.unit.rl.support import sim_config


def fixture(rights):
    start=datetime(2026,1,2,14,tzinfo=UTC);times=tuple(start+timedelta(days=i) for i in range(8));px=np.ones((8,2))*100
    arrays=PriceArrays(tuple(t.date() for t in times),('SPY','VOO'),px,px,np.ones_like(px)*100,np.ones_like(px,dtype=bool),times)
    corpus=data();template=corpus['deliveries'][0];corpus['deliveries']=[]
    for i,t in enumerate(times):
        row=copy.deepcopy(template);row.update(record_id='sha256:'+format(i,'064x'),retrieved_at=(t-timedelta(hours=1)).isoformat(),available_at=(t-timedelta(minutes=30)).isoformat())
        row['rights']['status']=rights
        row['records'][0].update(url='https://example.invalid/news/'+str(i),published_at=(t-timedelta(hours=2)).isoformat())
        corpus['deliveries'].append(row)
    return arrays,FrozenEtfNewsProvider(corpus)


def test_research_only_rejected_before_learner_creation(monkeypatch):
    arrays,provider=fixture('research_only');spec=EnvSpec.from_dict({'window':2,'features':['current_weights','etf_news_metadata_v1']})
    called=[];monkeypatch.setattr('finplan_model.rl.train._algo',lambda _:called.append(True))
    with pytest.raises(Exception,match='licensed'):
        train_policy(arrays,spec,sim_config(),algorithm='ppo',seed=0,hyperparameters={'total_timesteps':2},train_window=(0,3),validation_window=(4,5),configuration_id='synthetic',news_provider=provider)
    assert not called


def test_full_frozen_licensed_coverage_is_preflighted_and_missing_day_fails():
    arrays,provider=fixture('research_and_training');spec=EnvSpec.from_dict({'window':2,'features':['current_weights','etf_news_metadata_v1']})
    result=preflight_news_training(arrays,spec,train_window=(0,3),validation_window=(4,5),news_provider=provider)
    assert result['decision_count']==2 and result['corpus_checksum']==provider.corpus_checksum
    corrupted=provider(arrays.instruments,arrays.decision_times[2].isoformat());corrupted['deliveries']=[r for r in corrupted['deliveries'] if r['available_at']!=provider(arrays.instruments,arrays.decision_times[2].isoformat())['deliveries'][2]['available_at']]
    with pytest.raises(Exception,match='coverage'):preflight_news_training(arrays,spec,train_window=(0,3),validation_window=(4,5),news_provider=FrozenEtfNewsProvider(corrupted))
