"""Synthetic intraday capture audits verify current-day causality and coverage masks."""
import copy,json
from datetime import UTC,date,datetime,timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pytest
from finplan_model.core.artifacts import sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.rl.gdelt_intraday import FEATURE_VERSION,STATE_FEATURE,corpus,extract,formation_cutoff,FrozenGdeltProvider
from finplan_model.rl.spec import EnvSpec,build_observation
from finplan_model.rl.gdelt import morning_market

def archive(day='2026-10-10',captured=4):
    local=ZoneInfo('America/New_York');cut=datetime.fromisoformat(day+'T09:00:00').replace(tzinfo=local).astimezone(UTC)
    start=cut-timedelta(minutes=75);end=cut-timedelta(minutes=15);sources=[];attempts=[]
    for k in range(4):
        batch=start+timedelta(minutes=15*k);source={'batch_at':batch.isoformat(),'url':'https://example.invalid/synthetic.zip','archive_checksum':'sha256:'+'a'*64,'archive_bytes':100,'source_last_modified':batch.isoformat(),'source_etag':None,'source_available_at':(batch+timedelta(minutes=15)).isoformat(),'retrieved_at':end.isoformat()}
        attempts.append({'batch_at':batch.isoformat(),'url':source['url'],'http_status':200 if k<captured else 404,'retrieved_at':end.isoformat(),'archive_checksum':source['archive_checksum'] if k<captured else None,'error':None if k<captured else 'synthetic_missing'})
        if k<captured:sources.append(source)
    d={'feature_version':FEATURE_VERSION,'source_day':day,'window_start':start.isoformat(),'window_end':end.isoformat(),'decision_cutoff':cut.isoformat(),'row_count':100 if captured else 0,'unique_documents':100 if captured else 0,'duplicate_documents':0,'rejected_rows':0,'quality_status':'complete' if captured==4 else 'partial' if captured else 'unavailable','expected_batches':4,'captured_batches':captured,'missing_batches':[a['batch_at'] for a in attempts if a['http_status']!=200],'aliases':{'AAPL':['synthetic apple inc'],'VOO':['synthetic vanguard']},'disclosures':['Synthetic numeric metadata; not calibrated stock sentiment'],
       'features':[{'instrument_id':s,'article_count':0 if captured else None,'cluster_count':0 if captured else None,'tone':None,'missing':captured<4} for s in ['AAPL','VOO']],
       'lineage':{'source':'gdelt-gkg-v2','sources':sources,'attempts':attempts,'source_available_at':sources[-1]['source_available_at'] if sources else None,'retrieved_at':end.isoformat(),'platform_available_at':end.isoformat(),'central_available_at':end.isoformat(),'availability_basis':'synthetic historical_source_proxy','attribution':'Synthetic test source','rights':{'rights_status':'research_and_training','scope':'released_GDELT_machine_metadata_only','evidence_url':'https://example.invalid/terms','evidence_checksum':'sha256:'+'b'*64,'retrieved_at':end.isoformat(),'attribution':'Synthetic test source'}}}
    return wrap(d)

def wrap(d):return {'document':d,'checksum':sha256_checksum(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode())}

def test_healthy_zero_events_partial_and_unknown_have_distinct_masks():
    for n in (4,3,0):
        w=archive(captured=n);data=corpus([w],['AAPL','VOO']);f,p=extract(data,symbols=['AAPL'],decision_time=w['document']['decision_cutoff'])
        assert f['AAPL']['tone_scaled']==f['AAPL']['article_count_log_scaled']==0
        assert f['AAPL']['source_coverage_fraction']==n/4
        assert p['quality_status']==('complete' if n==4 else 'partial' if n else 'unavailable')
        if n==0:
            with pytest.raises(FinplanError,match='no current-day'):extract(corpus([w],['AAPL'],availability_mode='platform'),symbols=['AAPL'],decision_time=w['document']['decision_cutoff'])

def test_fixed_cutoff_dst_and_before_formation_guard():
    assert formation_cutoff('2026-10-10T19:00:00Z')=='2026-10-10T13:00:00+00:00'
    assert formation_cutoff('2026-12-10T19:00:00Z')=='2026-12-10T14:00:00+00:00'
    with pytest.raises(FinplanError,match='not due'):formation_cutoff('2026-10-10T12:59:59Z')

def test_live_late_first_seen_and_wrong_source_day_fail():
    w=archive();w['document']['lineage']['central_available_at']='2026-10-10T13:01:00Z';w=wrap(w['document']);data=corpus([w],['AAPL'],availability_mode='platform')
    with pytest.raises(FinplanError,match='unavailable'):extract(data,symbols=['AAPL'],decision_time='2026-10-10T13:00:00Z')
    with pytest.raises(FinplanError,match='current-day'):extract(corpus([archive()],['AAPL']),symbols=['AAPL'],decision_time='2026-10-11T13:00:00Z')

def test_versioned_observation_and_provider_cannot_silently_use_daily_source():
    w=archive();data=corpus([w],['AAPL','VOO']);provider=FrozenGdeltProvider(data);cut=w['document']['decision_cutoff'];spec=EnvSpec(window=2,features=('current_weights',STATE_FEATURE))
    obs=build_observation(spec,np.ones((3,2)),np.zeros(2),1.,news_dataset=provider(['AAPL','VOO'],cut),instruments=('AAPL','VOO'),decision_time=cut)
    assert obs.shape==(13,) and obs[-1]==1
    with pytest.raises(FinplanError):EnvSpec.from_dict({'features':[STATE_FEATURE,'gdelt_organization_tone_v1']})
    from finplan_model.selection.recursive_profiles import profile_protocol,GDELT_INTRADAY_PROFILE
    from finplan_model.selection.protocol import DEFAULT_PROTOCOL,validate_protocol
    p=validate_protocol(profile_protocol(DEFAULT_PROTOCOL,GDELT_INTRADAY_PROFILE))
    assert p['splits']['train']['end']=='2025-05-31' and p['splits']['validation']['start']=='2025-07-02'
    assert p['rl']['development_evaluation'] and p['rl']['seeds']==[0,1,2,3,4]


def test_intraday_common_evaluator_and_training_environment_share_observations_and_net_nav():
    from finplan_model.evaluate import evaluate
    from finplan_model.rl.arrays import PriceArrays
    from finplan_model.rl.env import PortfolioEnv
    from finplan_model.rl.policy import RLPolicyStrategy
    from finplan_model.sim.config import SimulationConfig
    from finplan_model.sim.market import synthetic_market
    from tests.unit.rl.support import sim_config
    market=morning_market(synthetic_market(('AAPL','VOO'),n_sessions=20,start=date(2026,1,2)))
    provider=FrozenGdeltProvider(corpus([archive((date(2026,1,1)+timedelta(days=k)).isoformat()) for k in range(40)],['AAPL','VOO']));spec=EnvSpec(window=2,features=('current_weights',STATE_FEATURE))
    config=SimulationConfig.from_dict({**sim_config().to_dict(),'rebalance_frequency':'daily'});seen=[]
    def predict(obs):seen.append(obs.copy());return np.array([obs[-10],obs[-5],.1])
    result=evaluate(RLPolicyStrategy('ppo',predict,spec,seed=0,configuration_id='synthetic',news_provider=provider),market,config,start=market.sessions[3],end=market.sessions[15])
    common=seen.copy();seen.clear()
    actual=PortfolioEnv(PriceArrays.from_market(market),spec,config,i0=3,i1=15,mode='eval',news_provider=provider).evaluate(predict)
    np.testing.assert_allclose(common,seen,atol=1e-6)
    np.testing.assert_allclose([r['value'] for r in result.simulation.nav],actual['values'],atol=1e-6)


def test_intraday_tiny_ppo_training_exports_compatible_numeric_actor_and_actual_source_evidence():
    from finplan_model.rl.arrays import PriceArrays
    from finplan_model.rl.train import train_policy
    from finplan_model.rl.inference import recommend
    from finplan_model.rl.inference_v2 import export_actor
    from finplan_model.sim.config import SimulationConfig
    from finplan_model.sim.market import synthetic_market
    from tests.unit.rl.support import sim_config
    market=morning_market(synthetic_market(('AAPL','VOO'),n_sessions=20,start=date(2026,1,2)))
    arrays=PriceArrays.from_market(market);provider=FrozenGdeltProvider(corpus([archive((date(2026,1,1)+timedelta(days=k)).isoformat()) for k in range(40)],['AAPL','VOO']))
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
    assert evidence['max_window_age_hours']==1.5 and evidence['provenance']['source_day']=='2026-01-27'

@pytest.mark.parametrize('mutation',['quality','duplicate','success_checksum','early_available','unknown_count'])
def test_cross_field_source_audit_inconsistency_fails_closed(mutation):
    w=archive(captured=0 if mutation=='unknown_count' else 3);d=w['document']
    if mutation=='quality':d['quality_status']='complete'
    elif mutation=='duplicate':d['lineage']['attempts'][1]['batch_at']=d['lineage']['attempts'][0]['batch_at']
    elif mutation=='success_checksum':d['lineage']['attempts'][0]['archive_checksum']='sha256:'+'c'*64
    elif mutation=='early_available':d['lineage']['sources'][0]['source_available_at']=d['window_start']
    else:d['features'][0]['article_count']=0
    with pytest.raises(FinplanError):corpus([wrap(d)],['AAPL','VOO'])
