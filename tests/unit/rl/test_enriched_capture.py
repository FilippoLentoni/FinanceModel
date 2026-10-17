"""Capture must work for frozen enriched price-only actors, before numerical inference."""
import copy
import numpy as np
import pytest
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.rl import inference_v2
from finplan_model.rl.spec import EnvSpec,build_observation


@pytest.mark.parametrize('extra',[['cross_asset_risk'],['market_regime'],['recent_returns'],['cross_asset_risk','market_regime','recent_returns']])
def test_nonnews_enriched_capture_precedes_actor_without_changing_allocation(extra,monkeypatch):
    spec=EnvSpec.from_dict({'window':20,'features':['market_summary','current_weights','portfolio_drawdown']+extra})
    environment=spec.document();environment.pop('reward_formula')
    bundle={'strategy_id':'ppo','environment':environment,'instruments':['A','B'],'source_run_id':'controlled_enriched_capture',
            'configuration_id':'cfg_controlled','members':[{'seed':0,'actor':{'algorithm':'ppo'}}],'constraints':{}}
    holdings={'weights':[{'instrument_id':'A','weight':0.},{'instrument_id':'B','weight':0.}],
              'cash_weight':1.,'portfolio_value':100000.,'high_watermark':100000.}
    prices=np.column_stack((np.linspace(100,110,21),np.linspace(100,99,21)))
    captures=[];require_capture=False
    def actor(_,obs):
        if require_capture:assert len(captures)==1
        np.testing.assert_array_equal(obs,build_observation(spec,prices,np.zeros(2),1.,drawdown=0.))
        return np.zeros(3)
    monkeypatch.setattr(inference_v2,'actor_action',actor)
    expected=inference_v2.recommend(bundle,prices,holdings)
    require_capture=True
    actual=inference_v2.recommend(bundle,prices,holdings,input_capture=captures.append)
    assert actual==expected and len(captures)==1
    captured=captures[0]
    assert captured['policy_checksum']==sha256_checksum(canonical_json_bytes(bundle))
    assert captured['feature_order']==environment['features']
    assert captured['news_dataset'] is None and not captured['news_evidence'].get('consumed',False)
    np.testing.assert_array_equal(captured['observation'],build_observation(spec,prices,np.zeros(2),1.,drawdown=0.))
