"""Synthetic numeric archives: timing, source rights, state/export and evaluator parity."""
import copy
import json
from datetime import UTC,date,datetime,time,timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.rl.gdelt import FEATURE_VERSION,STATE_FEATURE,FEATURE_ORDER,FrozenGdeltProvider,corpus,extract,freeze_from_api,morning_market
from finplan_model.rl.spec import EnvSpec,build_observation


def archive(day,*,count=10,tone=2.):
    day=date.fromisoformat(day) if isinstance(day,str) else day
    available=datetime.combine(day+timedelta(days=1),time(12),tzinfo=UTC).isoformat()
    document={'feature_version':FEATURE_VERSION,'source_day':day.isoformat(),'row_count':100,'rejected_rows':0,'synthetic':True,
              'aliases':{'AAPL':['synthetic apple inc'],'VOO':['synthetic vanguard']},'disclosures':['Synthetic organization tone; not stock sentiment'],
              'features':[{'instrument_id':'AAPL','article_count':0,'cluster_count':0,'tone':None,'missing':True},
                          {'instrument_id':'VOO','article_count':count,'cluster_count':count,'tone':tone,'missing':False}],
              'lineage':{'source':'gdelt-gkg-v1','source_day':day.isoformat(),'source_available_at':available,'source_last_modified':available,
                         'retrieved_at':'2026-10-17T05:00:00Z','platform_available_at':'2026-10-17T05:00:00Z','central_available_at':'2026-10-17T05:01:00Z',
                         'url':'https://example.invalid/archive.zip','archive_checksum':'sha256:'+'a'*64,'archive_bytes':1000,'source_etag':None,
                         'availability_basis':'synthetic historical_source_proxy','attribution':'Synthetic test source',
                         'rights_evidence_url':'https://example.invalid/terms','rights_scope':'synthetic machine metadata',
                         'rights':{'rights_status':'research_and_training','scope':'released_GDELT_machine_metadata_only','evidence_url':'https://example.invalid/terms',
                                   'evidence_checksum':'sha256:'+'b'*64,'retrieved_at':'2026-10-17T04:00:00Z','attribution':'Synthetic test source'}}}
    return {'document':document,'checksum':sha256_checksum(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode())}


def dataset(mode='historical_source_proxy'):
    return corpus([archive(date(2026,1,1)+timedelta(days=k)) for k in range(40)],['AAPL','VOO'],availability_mode=mode)


def test_archive_availability_is_separate_from_actual_central_first_seen():
    frozen=dataset();features,proof=extract(frozen,symbols=['AAPL','VOO'],decision_time='2026-01-02T14:00:00Z')
    assert proof['source_day']=='2026-01-01' and proof['availability_mode']=='historical_source_proxy'
    assert features['VOO']['tone_scaled']==.02 and features['AAPL']['missing_tone']==1
    assert features['AAPL']['article_count_log_scaled']==0 # observed zero, not absent source
    with pytest.raises(FinplanError,match='available'):
        extract(dataset('platform'),symbols=['AAPL','VOO'],decision_time='2026-01-02T14:00:00Z')
    with pytest.raises(FinplanError,match='available'):
        extract(frozen,symbols=['AAPL','VOO'],decision_time='2026-01-02T11:59:00Z')


def test_tampered_source_rights_quality_and_conflicting_vintage_fail_closed():
    altered=archive('2026-01-01');altered['document']['features'][1]['tone']=99
    with pytest.raises(FinplanError,match='checksum'):corpus([altered],['AAPL','VOO'])
    altered=archive('2026-01-01');altered['document']['lineage']['rights']['rights_status']='research_only'
    altered['checksum']=sha256_checksum(json.dumps(altered['document'],sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    with pytest.raises(FinplanError):corpus([altered],['AAPL','VOO'])
    with pytest.raises(FinplanError,match='conflicting'):corpus([archive('2026-01-01'),archive('2026-01-01',tone=3)],['AAPL','VOO'])


def test_source_age_limit_and_missing_archive_are_not_filled():
    frozen=corpus([archive('2026-01-01')],['AAPL','VOO'])
    with pytest.raises(FinplanError,match='48-hour'):
        extract(frozen,symbols=['AAPL','VOO'],decision_time='2026-01-05T14:00:00Z')


def test_provider_returns_independent_immutable_selected_vintage_and_five_numeric_inputs():
    provider=FrozenGdeltProvider(dataset());cutoff='2026-01-02T14:00:00Z';selected=provider(['AAPL','VOO'],cutoff)
    assert len(selected['documents'])==1
    selected['documents'][0]['document']['features'][1]['tone']=999
    selected=provider(['AAPL','VOO'],cutoff)
    assert selected['documents'][0]['document']['features'][1]['tone']==2
    spec=EnvSpec(window=2,features=('current_weights',STATE_FEATURE))
    obs=build_observation(spec,np.ones((3,2)),np.zeros(2),1.,news_dataset=selected,instruments=('AAPL','VOO'),decision_time=cutoff)
    assert obs.shape==(13,) and obs[-5]==pytest.approx(.02)
    assert FEATURE_ORDER[0]=='tone_scaled'
    with pytest.raises(FinplanError):EnvSpec.from_dict({'features':['etf_news_metadata_v1',STATE_FEATURE]})


def test_source_python_float_identity_survives_rfc_model_artifact_roundtrip():
    original=dataset();reloaded=json.loads(canonical_json_bytes(original))
    # RFC8785 serializes source tone 2.0 as 2; retained literal source JSON keeps
    # the producer's checksum verifiable without minting a different source ID.
    provider=FrozenGdeltProvider(reloaded)
    assert provider.corpus_checksum==FrozenGdeltProvider(original).corpus_checksum
    assert '2.0' in reloaded['documents'][0]['canonical_document_json']


def test_morning_timing_uses_next_actual_session_dst_and_no_weekday_guess():
    from finplan_model.sim.market import Bar,MarketData
    days=[date(2026,3,6),date(2026,3,9),date(2026,3,10)]
    base=MarketData(days,[Bar('VOO',d,100,datetime.combine(d,time(21),tzinfo=UTC)) for d in days])
    morning=morning_market(base)
    assert morning.decision_time(days[0]).isoformat()=='2026-03-09T13:00:00+00:00' # after US DST change
    assert morning.decision_time(days[1]).isoformat()=='2026-03-10T13:00:00+00:00'
    assert morning.decision_time(days[-1])==base.decision_time(days[-1]) # unused terminal mark


def test_gdelt_common_evaluator_and_training_environment_share_observations_and_net_nav():
    from finplan_model.evaluate import evaluate
    from finplan_model.rl.arrays import PriceArrays
    from finplan_model.rl.env import PortfolioEnv
    from finplan_model.rl.policy import RLPolicyStrategy
    from finplan_model.sim.config import SimulationConfig
    from finplan_model.sim.market import synthetic_market
    from tests.unit.rl.support import sim_config
    market=morning_market(synthetic_market(('AAPL','VOO'),n_sessions=20,start=date(2026,1,2)))
    provider=FrozenGdeltProvider(dataset());spec=EnvSpec(window=2,features=('current_weights',STATE_FEATURE))
    config=SimulationConfig.from_dict({**sim_config().to_dict(),'rebalance_frequency':'daily'});seen=[]
    def predict(obs):seen.append(obs.copy());return np.array([obs[-10],obs[-5],.1])
    result=evaluate(RLPolicyStrategy('ppo',predict,spec,seed=0,configuration_id='synthetic',news_provider=provider),market,config,start=market.sessions[3],end=market.sessions[15])
    common=seen.copy();seen.clear()
    actual=PortfolioEnv(PriceArrays.from_market(market),spec,config,i0=3,i1=15,mode='eval',news_provider=provider).evaluate(predict)
    np.testing.assert_allclose(common,seen,atol=1e-6)
    np.testing.assert_allclose([r['value'] for r in result.simulation.nav],actual['values'],atol=1e-6)


def test_real_tiny_ppo_training_exports_compatible_numeric_actor_and_actual_source_evidence():
    from finplan_model.rl.arrays import PriceArrays
    from finplan_model.rl.train import train_policy
    from finplan_model.rl.inference import recommend
    from finplan_model.rl.inference_v2 import export_actor
    from finplan_model.sim.config import SimulationConfig
    from finplan_model.sim.market import synthetic_market
    from tests.unit.rl.support import sim_config
    market=morning_market(synthetic_market(('AAPL','VOO'),n_sessions=20,start=date(2026,1,2)))
    arrays=PriceArrays.from_market(market);provider=FrozenGdeltProvider(dataset())
    spec=EnvSpec(window=2,features=('current_weights',STATE_FEATURE))
    config=SimulationConfig.from_dict({**sim_config().to_dict(),'rebalance_frequency':'daily'})
    trained=train_policy(arrays,spec,config,algorithm='ppo',seed=0,hyperparameters={'total_timesteps':32,'n_steps':16,'batch_size':8,'n_epochs':1,'net_arch':[8,8],'eval_every':16},train_window=(0,10),validation_window=(11,15),configuration_id='synthetic-gdelt',news_provider=provider)
    assert trained.timesteps_trained>=16 and trained.extra['news_preflight']['feature_group']==STATE_FEATURE
    env=spec.document();env.pop('reward_formula');actor=export_actor(trained.model_bytes,'ppo')
    bundle={'environment':env,'instruments':['AAPL','VOO'],'source_run_id':'run_synthetic','configuration_id':'cfg_synthetic','members':[{'seed':s,'actor':actor} for s in range(5)],
            'constraints':{},'training_gdelt_corpus_checksum':provider.corpus_checksum,'gdelt_feature_protocol':FEATURE_VERSION}
    k=16;cutoff=arrays.decision_times[k].isoformat()
    rec=recommend(bundle,arrays.close[k-2:k+1],{'weights':[],'cash_weight':1.,'portfolio_value':100.,'high_watermark':100.},news_dataset=provider(arrays.instruments,cutoff),decision_time=cutoff)
    evidence=rec['diagnostics']['news_evidence']
    assert evidence['consumed'] and evidence['feature_protocol']==FEATURE_VERSION and evidence['provenance']['source_day']
    assert 'prior_UTC_day_archive_not_same_morning_breaking_news' in evidence['limitations']
