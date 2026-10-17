"""Leakage, genuine-source export and wide-universe corpus checks (synthetic fixtures)."""
import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.core.platform import build_synthetic_snapshot
from finplan_model.rl.arrays import PriceArrays
from finplan_model.rl.checkpoint import freeze_corpus
from finplan_model.rl.checkpoint_prices import freeze_prices
from finplan_model.rl.etf_news import FrozenEtfNewsProvider, read_partitioned_news, validate_corpus
from finplan_model.rl.spec import EnvSpec, build_observation
from finplan_model.rl.train import preflight_training
from finplan_model.sim.market import MarketData
from tests.unit.rl.support import market, sim_config
from tests.unit.test_etf_news import data
from tests.unit.test_platform_snapshots import _payload, SID


def test_late_price_bar_cannot_enter_direct_learner_arrays():
    m = market()
    bad = [replace(b, available_at=m.decision_time(b.session_date)+timedelta(seconds=1))
           if b.session_date == m.sessions[20] else b for i in m.instruments for b in m.bars_of(i)]
    delayed = MarketData(m.sessions,bad)
    with pytest.raises(FinplanError) as error:
        PriceArrays.from_market(delayed)
    assert error.value.details['reason'] == 'rl_price_availability_mismatch'


@pytest.mark.parametrize('windows', [((0,80),(80,120)), ((0,80),(70,120)), ((0,4),(85,126)), ((0,80),(85,999))])
def test_training_requires_separate_chronological_windows_with_real_history(windows):
    with pytest.raises(FinplanError) as error:
        preflight_training(PriceArrays.from_market(market()),EnvSpec(window=5),train_window=windows[0],validation_window=windows[1])
    assert error.value.details['reason'] == 'training_windows_invalid'


def test_verified_approved_prices_export_identity_calendar_and_no_future_rows(platform):
    payload=_payload();payload.pop('synthetic')
    for row in payload['observations']:row.pop('synthetic')
    for row in payload['instruments']:row.pop('synthetic')
    record,blobs=build_synthetic_snapshot(payload,input_snapshot_id=SID,synthetic=False,provider='synthetic-test-only')
    platform.add_snapshot(record,blobs)
    out=freeze_prices(platform,SID,['SPY'],sim_config().to_dict(),news_start='2026-10-16T20:15:00Z',checkpoint_at='2026-10-30T20:15:00Z',now=datetime(2026,1,7,23,tzinfo=UTC))
    assert len(out['close'])==3 and out['snapshot_checksum']==record['manifest_checksum']
    assert len(out['price_provenance']['verified_artifacts'])==2
    assert out['price_provenance']['availability_mode']=='retrospective_session_close_assumption'
    assert out['synthetic'] is False and out['instruments']==['SPY']
    platform.corrupt('art_payload_'+SID.split('_')[1],b'corrupt')
    with pytest.raises(FinplanError,match='checksum'):
        freeze_prices(platform,SID,['SPY'],sim_config().to_dict(),news_start='2026-10-16T20:15:00Z',checkpoint_at='2026-10-30T20:15:00Z')


def partition_client():
    from types import SimpleNamespace
    calls=[]
    def read(symbols,cutoff):
        calls.append((symbols,cutoff));doc=data();doc['symbols']=list(symbols);doc['decision_at']=cutoff
        r=doc['deliveries'][0];r['record_id']='sha256:'+('a' if symbols[0]=='AAPL' else 'b')*64
        r['symbols']=list(symbols);r['records'][0]['symbols']=list(symbols)
        return doc
    return SimpleNamespace(get_research_news=read),calls


NAMES=['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA','VOO']


def test_eight_asset_observation_uses_validated_bounded_partitions_and_keeps_rights():
    client,calls=partition_client();cutoff=data()['decision_at']
    corpus=read_partitioned_news(client,NAMES,cutoff)
    assert [len(s) for s,t in calls]==[4,4]
    frozen=FrozenEtfNewsProvider(corpus)
    spec=EnvSpec(window=2,features=('current_weights','etf_news_metadata_v1'))
    obs=build_observation(spec,np.ones((3,8))*100,np.zeros(8),1.,news_dataset=frozen(NAMES,cutoff),instruments=tuple(NAMES),decision_time=cutoff)
    assert obs.shape==(49,) and np.all(obs[-40:].reshape(8,5)[:,-1]==0)
    with pytest.raises(FinplanError,match='licensed'):
        build_observation(spec,np.ones((3,8))*100,np.zeros(8),1.,news_dataset=corpus,instruments=tuple(NAMES),decision_time=cutoff,news_training=True)
    altered=copy.deepcopy(corpus);altered['parts'][1]['decision_at']='2026-01-02T15:00:00Z'
    with pytest.raises(FinplanError):validate_corpus(altered)
    altered=copy.deepcopy(corpus);altered['parts'][1]['deliveries'][0]['record_id']=altered['parts'][0]['deliveries'][0]['record_id']
    with pytest.raises(FinplanError,match='immutable'):validate_corpus(altered)


def test_fourteen_day_freeze_partitions_without_exceeding_api_bound():
    client,calls=partition_client()
    corpus,manifest=freeze_corpus(client,NAMES,'2026-01-15T14:00:00Z')
    assert len(calls)==6 and all(len(s)<=4 for s,t in calls)
    assert len(corpus['parts'])==2 and len(validate_corpus(corpus)['deliveries'])==2
    assert len(manifest['partitions'])==2


def test_daily_history_is_separate_frozen_profile_and_preserves_legacy():
    from finplan_model.selection.protocol import DEFAULT_PROTOCOL,validate_protocol
    from finplan_model.selection.recursive_profiles import profile_protocol,DAILY_HISTORY_PROFILE
    before=copy.deepcopy(DEFAULT_PROTOCOL)
    daily=validate_protocol(profile_protocol(before,DAILY_HISTORY_PROFILE))
    assert daily['splits']['train']=={'start':'2013-01-01','end':'2024-12-31'}
    assert daily['splits']['validation']['start']=='2025-01-01' and daily['splits']['test']['start']=='2026-01-01'
    assert daily['rl']['seeds']==[0,1,2,3,4] and daily['rl']['ppo']['total_timesteps']==30000
    assert DEFAULT_PROTOCOL==before
    legacy=profile_protocol(before,'recursive_ppo_state_baseline')
    assert legacy['splits']['train']['start']=='2025-01-01'


def test_predeclared_development_cannot_promote_even_with_a_new_dataset_identity():
    from finplan_model.selection.job import select_models
    from finplan_model.sim.config import SimulationConfig
    from tests.unit.rl.support import tiny_protocol
    p=tiny_protocol();p['rl']['algorithms']=['ppo'];p['rl']['development_evaluation']=True
    cfg=SimulationConfig.from_dict({**sim_config().to_dict(),'rebalance_frequency':'daily'})
    result=select_models(market(),cfg,protocol=p,prior_test_accesses=0)
    assert result.summary['test_access']['prior_runs_on_this_test_period']==0
    assert result.summary['test_access']['evaluation_role']=='development'
    assert result.summary['promotion_check']['result']=='research_only'
    assert result.summary['promotion_check']['reason']=='predeclared_development_period_not_untouched_holdout'
