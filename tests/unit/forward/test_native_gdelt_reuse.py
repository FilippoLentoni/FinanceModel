"""Controlled reuse tests: no providers, actors, training or paid calls."""
import copy
import numpy as np
import pytest
from finplan_model.core.errors import FinplanError
from finplan_model.forward.native_gdelt import latest_completed_formation,load_native_inputs
from finplan_model.rl.gdelt_intraday import FEATURE_VERSION,STATE_FEATURE,formation_cutoff,corpus,extract
from finplan_model.rl.spec import EnvSpec,build_observation
from tests.unit.rl.test_gdelt_intraday import archive,wrap

SYMBOLS=['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA','VOO']

@pytest.mark.parametrize('at,expected',[
 ('2026-10-18T03:59:59Z','2026-10-17T13:00:00+00:00'),
 ('2026-10-18T04:00:00Z','2026-10-17T13:00:00+00:00'),
 ('2026-10-18T12:59:59Z','2026-10-17T13:00:00+00:00'),
 ('2026-10-18T13:00:00Z','2026-10-18T13:00:00+00:00'),
 ('2026-11-01T06:30:00Z','2026-10-31T13:00:00+00:00'),
 ('2026-11-01T13:59:59Z','2026-10-31T13:00:00+00:00'),
 ('2026-11-01T14:00:00Z','2026-11-01T14:00:00+00:00'),
 ('2026-03-08T12:59:59Z','2026-03-07T14:00:00+00:00'),
 ('2026-03-08T13:00:00Z','2026-03-08T13:00:00+00:00'),
])
def test_latest_completed_calendar_formation_midnight_and_DST(at,expected):
 assert latest_completed_formation(at)==expected
 with pytest.raises(FinplanError,match='not due'):
  formation_cutoff('2026-10-18T12:59:59Z')


def source(day='2026-10-17',future=False):
 d=archive(day)['document'];template=d['features'][0]
 d['features']=[{**template,'instrument_id':symbol,'article_count':index,'cluster_count':index,'tone':float(index)}for index,symbol in enumerate(SYMBOLS)]
 if future:d['lineage']['central_available_at']=day+'T13:01:00Z'
 return wrap(d)


class ExistingPlatformCapture:
 def __init__(self,documents):self.documents=documents;self.calls=[]
 def get_gdelt_intraday_metadata(self,symbols,start,end,decision_at,*,availability_mode):
  self.calls.append((symbols,start,end,decision_at,availability_mode))
  return {'contract_version':'1.13.1','feature_version':FEATURE_VERSION,'symbols':symbols,'decision_at':decision_at,'availability_mode':availability_mode,
          'coverage':{'start_date':start,'end_date':end,'expected_days':1,'captured_days':len(self.documents),'complete':bool(self.documents),'missing_days':[]if self.documents else[start]},'documents':copy.deepcopy(self.documents)}


def test_prior_capture_reused_with_explicit_actual_age_and_original_provenance():
 original=source();client=ExistingPlatformCapture([original]);result=load_native_inputs(client,SYMBOLS,'2026-10-18T05:00:00Z')
 assert client.calls==[(SYMBOLS,'2026-10-17','2026-10-17','2026-10-17T13:00:00+00:00','platform')]
 assert result['dataset']['documents'][0]['checksum']==original['checksum']
 assert result['native_source_formation_at']=='2026-10-17T13:00:00+00:00'
 assert result['cadence_status']=='reused_native_source_features'
 assert result['source_age_hours_at_reuse']==pytest.approx(16.25)
 assert result['native_window_age_hours']==.25 and result['native_window_max_age_hours']==1.5
 assert result['central_available_at']==original['document']['lineage']['central_available_at']


@pytest.mark.parametrize('after09',[False,True])
def test_missing_selected_formation_remains_unavailable_without_fallback(after09):
 at='2026-10-18T13:01:00Z'if after09 else'2026-10-18T05:00:00Z'
 client=ExistingPlatformCapture([])
 with pytest.raises(FinplanError,match='current-day'):
  load_native_inputs(client,SYMBOLS,at)
 assert len(client.calls)==1
 assert client.calls[0][1]==('2026-10-18'if after09 else'2026-10-17')


def test_late_central_availability_cannot_be_backdated_into_previous_formation():
 with pytest.raises(FinplanError,match='unavailable'):
  load_native_inputs(ExistingPlatformCapture([source(future=True)]),SYMBOLS,'2026-10-18T05:00:00Z')


def test_reused_native_observation_is_exactly98_and_current_holdings_are_separate():
 original=source();dataset=corpus([original],SYMBOLS,availability_mode='platform');native=original['document']['decision_cutoff']
 spec=EnvSpec(window=60,features=('market_summary','current_weights','portfolio_drawdown',STATE_FEATURE))
 prices=100+np.arange(61*8,dtype=np.float64).reshape(61,8)*.01;weights=np.ones(8)*.0625
 expected=build_observation(spec,prices,weights,.5,drawdown=.05,news_dataset=dataset,instruments=tuple(SYMBOLS),decision_time=native)
 reused=load_native_inputs(ExistingPlatformCapture([original]),SYMBOLS,'2026-10-18T05:00:00Z')
 actual=build_observation(spec,prices,weights,.5,drawdown=.05,news_dataset=reused['dataset'],instruments=tuple(SYMBOLS),decision_time=reused['decision_time'])
 assert actual.shape==(98,);np.testing.assert_array_equal(actual,expected)
 changed=build_observation(spec,prices,np.zeros(8),1.,drawdown=.05,news_dataset=reused['dataset'],instruments=tuple(SYMBOLS),decision_time=reused['decision_time'])
 assert not np.array_equal(actual,changed)
 np.testing.assert_array_equal(actual[-40:],changed[-40:])
 # Native extraction still rejects treating previous-day metadata as today's09:00.
 with pytest.raises(FinplanError,match='current-day'):
  extract(reused['dataset'],symbols=SYMBOLS,decision_time='2026-10-18T13:00:00Z')


@pytest.mark.parametrize('status',['unavailable','failed','skipped'])
def test_available_source_never_claims_failed_actor_inferred(status):
 from finplan_model.forward.native_gdelt import annotate_outcome
 outcome=annotate_outcome({'status':status},{'status':'available','dataset':{},'native_source_formation_at':'2026-10-17T13:00:00Z'})
 assert outcome['native_input_status']=='available_reused_native_inputs' and 'signal_status'not in outcome
 assert 'dataset'not in outcome['native_input_detail']


def test_completed_actor_is_honestly_labeled_reused_not_fresh_news():
 from finplan_model.forward.native_gdelt import annotate_outcome
 outcome=annotate_outcome({'status':'succeeded'},{'status':'available','dataset':{}})
 assert outcome['native_input_status']==outcome['signal_status']=='inferred_from_reused_native_inputs'


def test_exact_adapter_provenance_reaches_durable_boundary_without_added_features(monkeypatch):
 from finplan_model.forward.runtime import evaluate_approach
 from finplan_model.forward.registry import APPROACHES
 from tests.unit.forward.test_runtime import source as market_source,holdings
 from finplan_model.rl import inference
 metadata=next(a.to_dict() for a in APPROACHES if a.approach_id=='ppo_gdelt_intraday')
 original_provenance={'native_source_formation_at':'2026-07-10T13:00:00Z',
  'reuse_as_of':'2026-07-11T00:04:00Z','acquired_at':'2026-07-11T00:04:01Z',
  'source_age_hours_at_reuse':11.25,'adapter_version':'hourly_native_feature_reuse/2'}
 metadata['native_input_provenance']=copy.deepcopy(original_provenance)
 captured=[]
 def recommend(policy,prices,state,**kwargs):
  assert not captured
  kwargs['input_capture']({'boundary':'frozen_actor','observation':list(range(13)),'news_evidence':{}})
  assert captured[0]['model']['adapter_provenance']['native_input']==original_provenance
  return {'weights':{'A':.5,'B':.5},'cash':0,'forecast':{'available':False}}
 monkeypatch.setattr(inference,'recommend',recommend)
 policy={'instruments':['A','B'],'environment':EnvSpec(window=2,features=('current_weights',STATE_FEATURE)).document(),'members':[]}
 policy['environment'].pop('reward_formula',None)
 result=evaluate_approach('ppo_gdelt_intraday',market_source(),holdings(),cutoff='2026-07-11T00:05:00Z',
  native_decision_time='2026-07-10T13:00:00Z',run_id='r',input_id='i',capture=captured.append,policy=policy,approach_metadata=metadata)
 assert result['status']=='recommended'
 assert captured[0]['model']['adapter_provenance']['consumed'] is False
 assert len(captured[0]['numeric']['prepared']['observation'])==13
 metadata['native_input_provenance']['source_age_hours_at_reuse']=999
 assert captured[0]['model']['adapter_provenance']['native_input']['source_age_hours_at_reuse']==11.25


def test_unavailable_actor_keeps_attempted_native_metadata_without_false_news():
 from finplan_model.forward.runtime import evaluate_approach
 from finplan_model.forward.registry import APPROACHES
 from tests.unit.forward.test_runtime import source as market_source,holdings
 metadata=next(a.to_dict()for a in APPROACHES if a.approach_id=='ppo_gdelt_intraday')
 metadata['native_input_provenance']={'status':'unavailable','query':{'formation_cutoff':'2026-07-10T13:00:00Z'}}
 captured=[]
 result=evaluate_approach('ppo_gdelt_intraday',market_source(),holdings(),cutoff='2026-07-11T00:05:00Z',run_id='r',input_id='i',capture=captured.append,approach_metadata=metadata)
 assert result['status']=='unavailable' and captured[0]['invocation_state']=='not_invoked'
 assert captured[0]['news']['consumed'] is False
 assert captured[0]['model']['adapter_provenance']['native_input']['status']=='unavailable'
