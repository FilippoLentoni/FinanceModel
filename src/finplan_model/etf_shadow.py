"""Frozen matched out-of-sample shadow study; no network, accounts or real orders."""
from __future__ import annotations
import copy
import math
from datetime import timedelta
import numpy as np
from .core.artifacts import canonical_json_bytes,sha256_checksum
from .core.errors import FinplanError
from .classical.math import settings,solve,objective,violations
from .etf_news import dt,extract

VERSION='matched-news-shadow/1'
SYMBOLS=('VOO','GOOGL')


def prospective_protocol(symbols=SYMBOLS):
    return {'version':VERSION,'mode':'synthetic_forward_shadow','symbols':list(symbols),'initial_cash':100.,
            'arms':['price_only','news_enriched'],'decision_frequency':'daily_after_completed_close',
            'execution':'first_observed_open_strictly_after_policy_formation','required_evidence':['frozen_price_vintage_available_before_decision','licensed_news_available_before_decision','next_open_and_close_outcomes'],
            'hypotheses':{'sentiment_mean_shift_per_session':.0005,'uncertainty_risk_scale':.25},
            'promotion':'none','ppo_training':'ineligible_without_historical_training_rights_and_compatible_checkpoint',
            'llm_policy':'ineligible_without_frozen_evaluated_policy'}


def number(value,*,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (positive and value<=0):raise ValueError('invalid finite price/value')
    return float(value)


def validate_step(step,symbols,previous_close=None):
    decision,opened,closed=[dt(step[k]) for k in ('decision_at','next_open_at','close_at')]
    if not decision<opened<=closed or (previous_close and decision<previous_close):raise ValueError('outcome overlaps or precedes decision')
    reasons=[]
    if dt(step['snapshot_available_at'])>decision:reasons.append('price_snapshot_vintage_unavailable_at_decision')
    if step['snapshot_checksum']!=sha256_checksum(canonical_json_bytes(step['history'])):raise ValueError('frozen price checksum mismatch')
    rows=step['history']
    if len(rows)>253:raise ValueError('price history exceeds declared 252-session lookback')
    if len(rows)<21:reasons.append('insufficient_price_history')
    dates=[dt(r['session_at']) for r in rows]
    if dates!=sorted(set(dates)):raise ValueError('history sessions not strictly ordered')
    if any(dt(r['available_at'])>decision or dt(r['session_at'])>decision for r in rows):reasons.append('future_price_history')
    for row in rows:
        if set(row['prices'])!=set(symbols):raise ValueError('history universe mismatch')
        for p in row['prices'].values():number(p,positive=True)
    for name in ('next_open','close'):
        if set(step[name])!=set(symbols):raise ValueError('outcome universe mismatch')
        for p in step[name].values():number(p,positive=True)
    return decision,closed,reasons


def book(symbols,cash=100.,quantities=None):
    return {'cash':cash,'quantities':{s:float((quantities or {}).get(s,0)) for s in symbols},'values':[100.], 'fees':0.,'slippage':0.,'traded_notional':0.,'turnover':0.,'fills':[]}


def execute(state,target,opens,closes,*,fee,slippage,opened):
    nav=state['cash']+sum(state['quantities'][s]*opens[s] for s in opens)
    changes={s:nav*target[s]-state['quantities'][s]*opens[s] for s in opens};traded=0.
    for side in ('sell','buy'):
        buys=sum(max(0,v)*(1+slippage)*(1+fee) for v in changes.values())
        scale=min(1.,max(state['cash'],0.)/buys) if buys else 0.
        for s,change in changes.items():
            if (side=='sell' and change>=0) or (side=='buy' and change<=0):continue
            q=min(state['quantities'][s],-change/opens[s]) if side=='sell' else change/opens[s]*scale
            px=opens[s]*(1-slippage if side=='sell' else 1+slippage);notional=q*px;charge=notional*fee
            state['cash']+=notional-charge if side=='sell' else -notional-charge
            state['quantities'][s]+= -q if side=='sell' else q
            state['fees']+=charge;state['slippage']+=q*opens[s]*slippage;traded+=notional
            state['fills'].append({'symbol':s,'side':side,'quantity':q,'price':px,'fee':charge,'at':opened})
    if state['cash']< -1e-7 or any(q< -1e-9 for q in state['quantities'].values()):raise ValueError('paper accounting violated cash/long-only constraint')
    state['cash']=max(state['cash'],0.)
    state['traded_notional']+=traded;state['turnover']+=traded/nav
    state['values'].append(state['cash']+sum(state['quantities'][s]*closes[s] for s in closes))


def metrics(state):
    values=np.asarray(state['values']);peaks=np.maximum.accumulate(values)
    return {'initial_value':float(values[0]),'final_value':float(values[-1]),'net_return':float(values[-1]/values[0]-1),
            'max_drawdown':float(np.max(1-values/peaks)),'turnover':state['turnover'],'traded_notional':state['traded_notional'],
            'fees':state['fees'],'slippage_cost':state['slippage'],'total_cost':state['fees']+state['slippage'],'outcome_observations':len(values)-1}


def bounded_solve(inputs):
    result=solve(inputs)
    if result['status']=='optimal' or len(inputs['instruments'])!=2:return result
    from scipy.optimize import minimize_scalar
    total=1-inputs['settings']['cash_weight'];cap=inputs['settings']['max_weight']
    lo,hi=max(0.,total-cap),min(cap,total)
    fit=minimize_scalar(lambda x:-objective(inputs,[float(x),total-float(x)]),bounds=(lo,hi),method='bounded')
    if not fit.success:return result
    candidates=[lo,hi,float(fit.x),*inputs['current_weights']]
    candidates=[x for x in candidates if lo<=x<=hi]
    x=max(candidates,key=lambda w:objective(inputs,[w,total-w]));weights=[x,total-x]
    if violations(inputs,weights) or not np.isfinite(objective(inputs,weights)):return result
    return {'status':'feasible','weights':weights,'solver':'bounded_convex_two_asset','cash_weight':1-sum(weights)}


def run_study(manifest,news_dataset):
    """Only realized post-decision outcomes become metrics; coverage failures yield no returns."""
    from .core.outcome import require_valid
    require_valid(news_dataset,'api/research-news-response')
    if manifest.get('algorithm','mean_variance') not in ('min_variance','mean_variance','cvar'):raise ValueError('unsupported optimizer')
    symbols=tuple(manifest.get('symbols',SYMBOLS))
    if not symbols or len(set(symbols))!=len(symbols) or set(symbols)-set((*SYMBOLS,'SPY','QQQ','VTI','BND')):raise ValueError('unsupported paper study universe')
    if manifest.get('initial_cash',100)!=100 or manifest.get('initial_quantities',{}):raise ValueError('matched study starts with separate synthetic USD100 cash')
    fee=number(manifest.get('fee_bps',10))/10000;slippage=number(manifest.get('slippage_bps',5))/10000
    if not 0<=fee<=.01 or not 0<=slippage<=.01:raise ValueError('unsupported cost bounds')
    if not isinstance(manifest.get('market_data_synthetic'),bool):raise ValueError('explicit real versus synthetic market-data provenance required')
    steps=manifest['steps']
    if not isinstance(steps,list):raise ValueError('steps must be a frozen decision list')
    if not 0<=len(steps)<=252:raise ValueError('study supports at most 252 frozen decisions and outcomes')
    reasons=[];features=[];previous=None;source=[]
    if not steps:reasons.append({'reason':'no_realized_forward_outcomes; frozen_point_in_time_collection_must_precede_next_market_open'})
    for step in steps:
        decision,closed,blocked=validate_step(step,symbols,previous);previous=closed
        f,provenance,_=extract(news_dataset,symbols=symbols,decision_time=step['decision_at'])
        missing=[s for s in symbols if f[s]['missing']]
        if missing:blocked.append('licensed_point_in_time_news_missing:'+','.join(missing))
        reasons.extend({'decision_at':step['decision_at'],'reason':r} for r in blocked)
        features.append(f);source.append({'decision_at':step['decision_at'],'price_checksum':step['snapshot_checksum'],'news':provenance})
    identity=sha256_checksum(canonical_json_bytes({'implementation':VERSION,'manifest':manifest,'news':news_dataset}))
    report={'study_id':identity,'implementation':VERSION,'synthetic_holdings':True,'market_data_synthetic':bool(manifest.get('market_data_synthetic',False)),
            'symbols':list(symbols),'source_evidence':source,'news_corpus_checksum':sha256_checksum(canonical_json_bytes(news_dataset)),
            'news_collection_available_range':{'first':min((d['available_at'] for d in news_dataset['deliveries']),default=None),'last':max((d['available_at'] for d in news_dataset['deliveries']),default=None)},'costs':{'fee_bps':fee*10000,'slippage_bps':slippage*10000},'prospective_protocol':prospective_protocol(symbols),
            'return_basis':'price_return_no_cash_distributions_modeled',
            'accuracy_improvement':'not_established','ppo':{'eligible':False,'reason':'compatible_checkpoint_and_historical_training_rights_not_provided'},
            'llm':{'eligible':False,'reason':'frozen_evaluated_news_trading_policy_not_provided'}}
    if reasons:return {**report,'status':'ineligible','blockers':reasons,'performance':None}
    arms={name:book(symbols) for name in ('price_only','news_enriched')};decisions=[]
    for step,f in zip(steps,features):
        px=np.array([[r['prices'][s] for s in symbols] for r in step['history']]);returns=px[1:]/px[:-1]-1
        cfg=settings({'lookback_days':min(252,len(returns)),'cash_weight':.1,'max_weight':1. if len(symbols)==1 else .6,'horizon_sessions':1,'turnover_penalty':.001})
        common=arms['price_only'];value=common['cash']+sum(common['quantities'][s]*px[-1,j] for j,s in enumerate(symbols))
        inputs={'algorithm':manifest.get('algorithm','mean_variance'),'instruments':list(symbols),'settings':cfg,
                'current_weights':[common['quantities'][s]*px[-1,j]/value for j,s in enumerate(symbols)],'expected_returns':returns.mean(axis=0).tolist(),
                'covariance':(np.atleast_2d(np.cov(returns,rowvar=False))+np.eye(len(symbols))*1e-10).tolist(),'scenarios':returns.tolist()}
        enriched=copy.deepcopy(inputs);sentiment=np.array([f[s]['sentiment'] for s in symbols]);scale=1+.25*np.array([f[s]['uncertainty'] for s in symbols])
        enriched['expected_returns']=(np.asarray(inputs['expected_returns'])+.0005*sentiment).tolist()
        enriched['covariance']=(np.asarray(inputs['covariance'])*np.outer(scale,scale)).tolist();enriched['scenarios']=(returns*scale).tolist()
        solutions={name:bounded_solve(i) for name,i in [('price_only',inputs),('news_enriched',enriched)]}
        if any(s['status'] not in ('optimal','feasible') for s in solutions.values()):return {**report,'status':'ineligible','blockers':[{'reason':'matched_solver_not_certified'}],'performance':None}
        for name,solution in solutions.items():execute(arms[name],dict(zip(symbols,solution['weights'])),step['next_open'],step['close'],fee=fee,slippage=slippage,opened=step['next_open_at'])
        decisions.append({'decision_at':step['decision_at'],'common_inputs_checksum':sha256_checksum(canonical_json_bytes(inputs)),'news_inputs_checksum':sha256_checksum(canonical_json_bytes(enriched)),
                          'target_weights':{name:dict(zip(symbols,solution['weights'])) for name,solution in solutions.items()}})
    return {**report,'status':'completed','blockers':[],'performance':{name:metrics(state) for name,state in arms.items()},'decisions':decisions,
            'matched_policy_holdings':'both target policies use the price-only observed book; realized execution books may diverge',
            'limitations':['uncalibrated_news_hypothesis','no_live_orders','small_forward_sample_not_profitability_proof'],'books':arms}
