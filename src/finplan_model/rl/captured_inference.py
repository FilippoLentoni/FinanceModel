"""Capture-aware execution of unchanged frozen legacy math; old source remains immutable."""
from __future__ import annotations
import numpy as np
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.errors import FinplanError
from .legacy_v1 import inference as legacy


def recommend(bundle,prices,holdings,*,input_capture):
    spec=legacy.EnvSpec.from_dict(bundle['environment']);instruments=list(bundle['instruments'])
    current,cash,value,peak=legacy.portfolio_state(instruments,holdings)
    if prices.shape!=(spec.window+1,len(instruments)) or not np.isfinite(prices).all() or np.any(prices<=0):raise FinplanError.precondition('approved snapshot lacks aligned completed history',reason='insufficient_history')
    if not bundle.get('members'):raise FinplanError.precondition('exported policy has no members',reason='policy_members_missing')
    obs=legacy.build_observation(spec,prices,current,cash,drawdown=1-value/peak)
    from finplan_model.forward.observation_names import observation_names
    names=observation_names(spec,instruments)
    if len(names)!=len(obs):raise ValueError('legacy actual observation ordering mismatch')
    input_capture({'boundary':'frozen_legacy_actor','implementation':legacy.implementation_identity(),'capture_adapter_version':'finplan-legacy-capture/1',
       'policy_source_run_id':bundle['source_run_id'],'configuration_id':bundle['configuration_id'],'instruments':instruments,
       'prices':prices.tolist(),'holdings':dict(holdings),'environment':dict(bundle['environment']),'observation':obs.tolist(),
       'observation_dtype':str(obs.dtype),'feature_order':list(spec.features),'observation_names':names,
       'named_observation':[{'index':k,'name':name,'value':float(obs[k])} for k,name in enumerate(names)],
       'news_evidence':{'consumed':False,'reason':'Frozen legacy policy consumes no news'},
       'normalization':{'return_scale':spec.return_scale,'obs_clip':spec.obs_clip},'policy_checksum':sha256_checksum(canonical_json_bytes(bundle))})
    targets=[legacy.allocation_action(spec,legacy.actor_action(m['actor'],obs),current,cash) for m in bundle['members']]
    return legacy.allocation_response(bundle,holdings,np.mean([t[0] for t in targets],axis=0),float(np.mean([t[1] for t in targets])),
       diagnostics={'input_features':list(spec.features),'observation_dimension':int(obs.size),'observation_checksum':sha256_checksum(obs.tobytes()),
        'feature_window_sessions':spec.window,'observed_drawdown':1-value/peak,
        'member_target_weights':[{'seed':m['seed'],'weights':target[0].tolist(),'cash_weight':float(target[1])} for m,target in zip(bundle['members'],targets)],
        'interpretation':'Actual frozen actor outputs and constraint transform; not causal attribution or a return forecast'})


def recommend_baseline(bundle,market,as_of,holdings,*,input_capture):
    from finplan_model.sim.market import HoldingsView
    from finplan_model.sim.strategy import coerce_target
    from finplan_model.strategies import build_strategy
    from finplan_model.sim.config import SimulationConfig
    instruments=tuple(bundle['instruments']);current,cash,value,_=legacy.portfolio_state(instruments,holdings)
    view=market.view(as_of,instruments);prices={i:view.latest(i) for i in instruments}
    if any(p is None for p in prices.values()):raise FinplanError.precondition('snapshot lacks current instrument prices',reason='insufficient_history')
    observed=HoldingsView({i:float(current[k])*value/prices[i] for k,i in enumerate(instruments)},cash*value,prices,value)
    strategy=build_strategy(bundle['strategy_id'],bundle['parameters'],constraints=SimulationConfig.from_dict({'constraints':bundle.get('constraints')}).constraints)
    captured=[]
    def before(prepared):
        captured.append(prepared);input_capture({'boundary':'served_baseline_solver','holdings':dict(holdings),'prices':prices,'prepared':prepared})
    strategy.input_capture=before
    if bundle['strategy_id'] in ('cash','equal_weight','buy_and_hold'):before({'boundary':'control','parameters':strategy.params})
    target=coerce_target(strategy.decide(view,observed),instruments)
    if not captured:before({'boundary':'not_invoked_insufficient_history','diagnostics':target.diagnostics,'parameters':strategy.params})
    if target.solution_status in ('infeasible','unbounded') or target.diagnostics.get('reason')=='insufficient_history':raise FinplanError.precondition('selected strategy cannot produce valid target',reason=target.diagnostics.get('reason',target.solution_status))
    return legacy.allocation_response(bundle,holdings,[target.weights[i] for i in instruments],target.cash,solution_status=target.solution_status,diagnostics=dict(target.diagnostics))
