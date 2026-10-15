from dataclasses import replace
from datetime import timedelta
import copy
import numpy as np
import pytest
from finplan_model.rl.spec import EnvSpec,build_observation
from finplan_model.rl.env import PortfolioEnv
from finplan_model.rl.arrays import PriceArrays
from finplan_model.rl.etf_news import FrozenEtfNewsProvider
from tests.unit.test_etf_news import data
from tests.unit.rl.support import market,sim_config


def test_opt_in_dimensions_order_normalization_and_default_unchanged():
    base=EnvSpec.from_dict({'window':3})
    spec=EnvSpec.from_dict({'window':3,'features':[*base.features,'etf_news_metadata_v1']})
    px=np.ones((4,2));weights=np.array([.4,.4]);dataset=data()
    default=build_observation(base,px,weights,.2)
    obs=build_observation(spec,px,weights,.2,news_dataset=dataset,instruments=('VOO','SPY'),decision_time=dataset['decision_at'])
    np.testing.assert_array_equal(default,obs[:-10]);np.testing.assert_allclose(obs[-10:], [1,.01,0,2/24,0]*2)
    assert len(obs)==spec.observation_size(2)==base.observation_size(2)+10
    assert base.configuration_id!=spec.configuration_id
    with pytest.raises(Exception,match='explicit'):build_observation(spec,px,weights,.2)
    with pytest.raises(Exception,match='cutoff'):build_observation(spec,px,weights,.2,news_dataset=dataset,instruments=('VOO','SPY'),decision_time='2026-01-01T00:00:00Z')


def test_training_rights_historical_cutoff_and_frozen_provider():
    spec=EnvSpec.from_dict({'window':3,'features':['etf_news_metadata_v1']});dataset=data();provider=FrozenEtfNewsProvider(dataset)
    dataset['deliveries'][0]['records'][0]['title']='mutation does not change frozen data'
    kwargs=dict(instruments=('SPY','VOO'),decision_time='2026-01-02T14:00:00Z')
    frozen=provider(kwargs['instruments'],kwargs['decision_time'])
    assert frozen['deliveries'][0]['records'][0]['title']!='mutation does not change frozen data'
    with pytest.raises(Exception,match='licensed'):build_observation(spec,np.ones((4,2)),np.ones(2)*.4,.2,news_dataset=frozen,news_training=True,**kwargs)
    licensed=copy.deepcopy(frozen);licensed['deliveries'][0]['rights']['status']='research_and_training'
    trained=build_observation(spec,np.ones((4,2)),np.ones(2)*.4,.2,news_dataset=licensed,news_training=True,**kwargs)
    assert trained[-1]==0
    old='2026-01-02T12:00:00Z';historical=FrozenEtfNewsProvider(licensed)(('SPY','VOO'),old)
    with pytest.raises(Exception,match='coverage'):build_observation(spec,np.ones((4,2)),np.ones(2)*.4,.2,news_dataset=historical,instruments=('SPY','VOO'),decision_time=old,news_training=True)
    missing=build_observation(spec,np.ones((4,2)),np.ones(2)*.4,.2,news_dataset=historical,instruments=('SPY','VOO'),decision_time=old)
    np.testing.assert_array_equal(missing,[0,0,0,0,1]*2)


def test_environment_calls_provider_at_each_actual_decision_and_fails_closed_training():
    arrays=PriceArrays.from_market(market())
    arrays=replace(arrays,instruments=('SPY','VOO','QQQ','VTI','BND'))
    spec=EnvSpec.from_dict({'window':3,'features':['current_weights','etf_news_metadata_v1']})
    with pytest.raises(Exception,match='provider'):PortfolioEnv(arrays,spec,sim_config(),i0=10,i1=14,mode='eval')
    calls=[]
    def provider(symbols,cutoff):
        calls.append((symbols,cutoff));d=data();d['symbols']=list(symbols);d['decision_at']=cutoff
        d['deliveries']=[]
        return d
    env=PortfolioEnv(arrays,spec,sim_config(),i0=10,i1=14,mode='eval',news_provider=provider)
    obs,_=env.reset(seed=1)
    assert calls[-1]==(arrays.instruments,arrays.decision_times[10].isoformat())
    assert env.observation_space.contains(obs)
    obs,_,_,_,_=env.step(np.zeros(6))
    assert calls[-1][1]==arrays.decision_times[11].isoformat() and len(calls)==2
    train=PortfolioEnv(arrays,spec,sim_config(),i0=10,i1=20,mode='train',news_provider=provider)
    with pytest.raises(Exception,match='licensed'):train.reset(seed=1)


def test_news_state_common_evaluator_environment_parity():
    from finplan_model.rl.policy import RLPolicyStrategy
    from finplan_model.evaluate import evaluate
    from finplan_model.sim.market import MarketData
    original=market();names=dict(zip(original.instruments,('SPY','VOO','QQQ','VTI','BND')))
    mkt=MarketData(original.sessions,[replace(bar,instrument_id=names[old]) for old in original.instruments for bar in original.bars_of(old)],synthetic=True,
                   decision_times={d:original.decision_time(d) for d in original.sessions})
    spec=EnvSpec.from_dict({'window':3,'features':['current_weights','etf_news_metadata_v1']})
    def provider(symbols,cutoff):
        d=data();d['symbols']=list(symbols);d['decision_at']=cutoff;d['deliveries']=[];return d
    observations=[]
    def predict(obs):observations.append(obs.copy());return np.zeros(6)
    from tests.unit.rl.test_rl_stability import daily_config
    config=daily_config()
    strategy=RLPolicyStrategy('ppo',predict,spec,seed=1,configuration_id='cfg_synthetic',news_provider=provider)
    common=evaluate(strategy,mkt,config,start=mkt.sessions[10],end=mkt.sessions[14])
    expected=np.array(observations);observations.clear()
    env=PortfolioEnv(PriceArrays.from_market(mkt),spec,config,i0=10,i1=14,mode='eval',news_provider=provider)
    observed=env.evaluate(predict)
    np.testing.assert_allclose(observations,expected,atol=1e-6)
    np.testing.assert_allclose(observed['values'],[r['value'] for r in common.simulation.nav],atol=1e-6)
