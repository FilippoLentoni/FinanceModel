from __future__ import annotations

import io
import numpy as np
import gymnasium as gym
import pytest
from stable_baselines3 import PPO, SAC
from finplan_model.rl.inference import actor_action, export_actor, recommend, recommend_baseline
from finplan_model.rl.spec import EnvSpec
from finplan_model.core.errors import FinplanError


class ActorEnv(gym.Env):
    observation_space=gym.spaces.Box(-5,5,shape=(37,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,shape=(6,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return np.zeros(37,dtype=np.float32),{}
    def step(self,action):
        return np.zeros(37,dtype=np.float32),0.,True,False,{}


@pytest.mark.parametrize('algorithm', ['ppo', 'sac'])
def test_export_matches_sb3_deterministic_action_on_varied_observations(algorithm):
    model = (PPO('MlpPolicy',ActorEnv(),n_steps=16,batch_size=16,seed=7) if algorithm == 'ppo'
             else SAC('MlpPolicy',ActorEnv(),policy_kwargs={'net_arch':[64,64]},seed=7,buffer_size=100))
    # Exercise both unclipped and clipped Gaussian means.
    (model.policy.action_net if algorithm == 'ppo' else model.policy.actor.mu).weight.data.mul_(100)
    buf=io.BytesIO();model.save(buf)
    actor=export_actor(buf.getvalue(), algorithm)
    for obs in np.random.default_rng(5).uniform(-5,5,(100,37)).astype(np.float32):
        # Different float32 BLAS kernels amplify error under the deliberately 100x
        # SAC output weights; the measured maximum here is about 1.1e-5.
        tolerance = 2e-6 if algorithm == 'ppo' else 2e-5
        np.testing.assert_allclose(actor_action(actor,obs),model.predict(obs,deterministic=True)[0],atol=tolerance,rtol=2e-6)


def test_recommendation_requires_current_state_and_applies_frozen_constraints():
    actor={'format':'finplan-ppo-actor/1','layers':[{'weight':np.zeros((64,37)).tolist(),'bias':np.zeros(64).tolist(),'activation':'tanh'}, {'weight':np.zeros((64,64)).tolist(),'bias':np.zeros(64).tolist(),'activation':'tanh'}, {'weight':np.zeros((6,64)).tolist(),'bias':[1,-1,-1,-1,-1,-1],'activation':'linear'}]}
    env=EnvSpec.from_dict({'window':60,'features':['market_summary','current_weights','portfolio_drawdown'],'action_scale':1,'rebalance_fraction':.25}).document()
    env.pop('reward_formula')
    bundle={'environment':env,'instruments':['A','B','C','D','E'],'source_run_id':'run_synthetic','configuration_id':'cfg_synthetic','members':[{'seed':s,'actor':actor} for s in range(5)],'constraints':{'max_weight':.15,'max_turnover':.05}}
    holdings={'weights':[{'instrument_id':i,'weight':.1} for i in bundle['instruments']],'cash_weight':.5,'portfolio_value':900.,'high_watermark':1000.}
    rec=recommend(bundle,np.ones((61,5)),holdings)
    assert sum(r['weight'] for r in rec['target_weights'])+rec['cash_weight']==pytest.approx(1)
    assert max(r['weight'] for r in rec['target_weights'])<=.15+1e-8
    assert rec['estimated_turnover']<=.05+1e-8
    assert rec['policy_seeds']==list(range(5))
    assert rec['forecast']['status']=='not_available'
    with pytest.raises(FinplanError):recommend(bundle,np.ones((60,5)),holdings)
    with pytest.raises(FinplanError):recommend(bundle,np.ones((61,5)),{**holdings,'high_watermark':800})


def test_classical_strategy_uses_current_holdings_and_frozen_parameters():
    from finplan_model.sim.market import synthetic_market
    market = synthetic_market(('A', 'B'), n_sessions=140)
    bundle = {'strategy_id':'min_variance', 'parameters':{'lookback':60}, 'constraints':{'max_weight':.6},
              'instruments':['A','B'], 'source_run_id':'run_synthetic', 'configuration_id':'cfg_synthetic', 'members':[]}
    holdings = {'weights':[{'instrument_id':'A','weight':.9}], 'cash_weight':.1,
                'portfolio_value':1000., 'high_watermark':1000.}
    rec = recommend_baseline(bundle, market, market.sessions[-1], holdings)
    assert rec['strategy'] == 'min_variance'
    assert rec['aggregation'] == 'single_strategy' and rec['policy_seeds'] == []
    assert sum(w['weight'] for w in rec['target_weights']) + rec['cash_weight'] == pytest.approx(1)
    assert all(w['weight'] <= .6 + 1e-8 for w in rec['target_weights'])
    bundle.update(strategy_id='buy_and_hold', parameters={}, constraints={})
    held = recommend_baseline(bundle, market, market.sessions[-1], holdings)
    assert held['estimated_turnover'] == pytest.approx(0)
    assert all(d['action'] == 'hold' for d in held['decisions'])


def test_rejected_trade_holds_current_portfolio_as_in_offline_evaluator():
    from finplan_model.rl.inference import allocation_response
    bundle = {'instruments':['A'], 'source_run_id':'run_synthetic', 'configuration_id':'cfg_synthetic',
              'constraints':{'max_weight':.5}, 'constraint_policy':'reject'}
    holdings = {'weights':[{'instrument_id':'A','weight':.2}], 'cash_weight':.8,
                'portfolio_value':1000.,'high_watermark':1000.}
    rec = allocation_response(bundle, holdings, [1.], 0.)
    assert rec['solution_status'] == 'rejected_hold_current'
    assert rec['constraint_outcome']['action'] == 'rejected'
    assert rec['target_weights'] == holdings['weights'] and rec['cash_weight'] == .8
    assert rec['estimated_turnover'] == 0
