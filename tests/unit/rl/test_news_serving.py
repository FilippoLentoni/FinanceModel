import copy
import numpy as np
import pytest
from finplan_model.rl.inference import recommend
from finplan_model.rl.spec import EnvSpec
from tests.unit.test_etf_news import data


def actor_bundle():
    spec=EnvSpec.from_dict({'window':2,'features':['log_return_window','current_weights','etf_news_metadata_v1']})
    w=np.zeros((1,spec.observation_size(2)));w[0,7]=1.
    actor={'format':'finplan-actor/1','layers':[{'weight':w.tolist(),'bias':[0.],'activation':'tanh'},
        {'weight':[[1.]],'bias':[0.],'activation':'tanh'},{'weight':[[1.],[-1.],[0.]],'bias':[0.,0.,0.],'activation':'linear'}],'output_transform':'clip'}
    env={k:v for k,v in spec.document().items() if k!='reward_formula'}
    return {'strategy_id':'ppo','source_run_id':'synthetic-run','configuration_id':'synthetic-config','instruments':['SPY','VOO'],'environment':env,'constraints':{},
            'members':[{'seed':s,'actor':actor} for s in range(5)],'training_news_corpus_checksum':'sha256:'+'a'*64,'news_feature_protocol':'etf-search-metadata/1'}


def test_actual_actor_consumes_news_counterfactual_and_checksums():
    bundle=actor_bundle();d=data();holdings={'weights':[{'instrument_id':s,'weight':0.} for s in ('SPY','VOO')],'cash_weight':1.,'portfolio_value':100.,'high_watermark':100.}
    positive=recommend(bundle,np.ones((3,2)),holdings,news_dataset=d,decision_time=d['decision_at'])
    negative=copy.deepcopy(d);negative['deliveries'][0]['records'][0]['title']='Synthetic ETF falls loss'
    changed=recommend(bundle,np.ones((3,2)),holdings,news_dataset=negative,decision_time=d['decision_at'])
    assert positive['diagnostics']['news_evidence']['consumed']
    assert positive['diagnostics']['observation_checksum']!=changed['diagnostics']['observation_checksum']
    assert positive['target_weights'][0]['weight']>changed['target_weights'][0]['weight']
    assert positive['policy_seeds']==list(range(5))
    # This is a hand-constructed synthetic actor fixture, not evidence of a trained policy.


def test_news_policy_requires_training_provenance_and_live_coverage():
    bundle=actor_bundle();d=data();holdings={'weights':[],'cash_weight':1.,'portfolio_value':100.,'high_watermark':100.}
    bad=copy.deepcopy(bundle);bad.pop('training_news_corpus_checksum')
    with pytest.raises(Exception,match='training corpus'):recommend(bad,np.ones((3,2)),holdings,news_dataset=d,decision_time=d['decision_at'])
    d['deliveries'][0]['rights']['status']='unverified'
    with pytest.raises(Exception,match='coverage'):recommend(bundle,np.ones((3,2)),holdings,news_dataset=d,decision_time=d['decision_at'])
