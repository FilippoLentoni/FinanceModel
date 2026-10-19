"""Prepare, capture and invoke one actual adapter. No training or fabricated fallbacks."""
from __future__ import annotations
import copy
import math
import re
import time
from datetime import date, datetime
from typing import Any, Callable
import numpy as np
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.sim.market import Bar, HoldingsView, MarketData, PointInTimeView
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.strategy import coerce_target
from finplan_model.strategies import build_strategy
from .registry import APPROACHES


def input_market(source, cutoff):
    """Only actual completed observations and actual platform availability are eligible."""
    deadline=datetime.fromisoformat(cutoff.replace('Z','+00:00'))
    rows=source.get('observations',[])
    bars=[]
    for row in rows:
        available=row.get('available_at')
        if not available:
            raise ValueError('price observation missing actual availability timestamp')
        stamp=datetime.fromisoformat(available.replace('Z','+00:00'))
        if stamp>deadline or date.fromisoformat(row['session_date'])>deadline.date():
            continue
        bars.append(Bar.from_observation(row, available))
    if not bars:
        raise FinplanError.precondition('no eligible completed market history', reason='prices_unavailable')
    market=MarketData(sorted({b.session_date for b in bars}),bars,synthetic=False,
                      dataset_id=source.get('dataset_id'),dataset_checksum=source.get('checksum'))
    return market


def book_holdings_view(source, holdings, cutoff, universe):
    """The point-in-time view and holdings marked as the direct classical/baseline books see them.

    Positions and cash of the pre-decision book, marked at the latest eligible snapshot feature
    close of each universe instrument. The agent-tool path sends exactly these weights (see
    :func:`classical_tool_holdings`) so agent and direct classical books solve the same problem.
    """
    market=input_market(source,cutoff)
    decision=max(market.sessions)
    view=PointInTimeView(market,decision,datetime.fromisoformat(cutoff.replace('Z','+00:00')),tuple(universe))
    shares={p['instrument_id']:float(p['quantity']) for p in holdings.get('positions',[])}
    quotes={i:view.latest(i) for i in universe}
    if any(q is None for q in quotes.values()):raise FinplanError.precondition('missing eligible instrument prices',reason='prices_unavailable')
    cash=float(holdings.get('cash',holdings.get('cash_balance',0)))
    value=cash+sum(shares.get(i,0)*quotes[i] for i in universe)
    return view,HoldingsView(shares,cash,quotes,value)


def classical_tool_holdings(source, holdings, cutoff, universe):
    """Tool ``holdings`` equal to the direct classical book's ``HoldingsView`` (weights by universe,
    cash weight and value); the high watermark keeps the book's peak and never falls below value."""
    _,hv=book_holdings_view(source,holdings,cutoff,universe)
    weights=hv.weights
    return {'weights':[{'instrument_id':i,'weight':float(weights.get(i,0.))} for i in universe],'cash_weight':float(hv.cash_weight),
            'portfolio_value':float(hv.value),'high_watermark':float(max(holdings.get('high_watermark') or 0.,hv.value))}


def validate_policy(output, universe, holdings):
    """Strict weights-only policy; intents are derived from the pre-decision book."""
    weights=output.get('target_weights',output.get('weights'))
    if isinstance(weights,list):
        ids=[r['instrument_id'] for r in weights]
        if len(ids)!=len(set(ids)):raise ValueError('duplicate target instrument')
        weights={r['instrument_id']:r['weight'] for r in weights}
    if not isinstance(weights,dict) or set(weights)-set(universe):raise ValueError('unsupported target instrument')
    cash=output.get('cash_weight',output.get('cash'))
    vals=[*weights.values(),cash]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in vals):raise ValueError('invalid allocation')
    if abs(sum(weights.values())+cash-1)>1e-6:raise ValueError('target weights must sum to one')
    old=holdings.get('weights',{})
    if isinstance(old,list):old={r['instrument_id']:r['weight'] for r in old}
    intents=[{'instrument_id':i,'action':'hold' if abs(weights.get(i,0)-old.get(i,0))<1e-6 else 'buy' if weights.get(i,0)>old.get(i,0) else 'sell',
              'delta_weight':weights.get(i,0)-old.get(i,0)} for i in universe]
    if holdings.get('portfolio_value') is None:
        quantities={p['instrument_id']:p['quantity'] for p in holdings.get('positions',[])}
        for intent in intents:
            if quantities.get(intent['instrument_id'],0):intent.update(action='hold',intent_status='unresolved_missing_position_valuation',delta_weight=None)
    # An absent zero position has delta zero; it cannot become a sell intent.
    return {**output,'target_weights':[{'instrument_id':i,'weight':float(weights.get(i,0))} for i in universe], 'cash_weight':float(cash),'intents':intents}


def evaluate_approach(approach_id, source, holdings, *, cutoff, run_id, input_id, capture: Callable,
                      config=None, policy=None, news_dataset=None, external_adapters=None, approach_metadata=None, native_decision_time=None, shared_source_checksum=None):
    """Capture callback must durably persist its immutable payload before returning.

    Failures to capture propagate and prevent execution. A failed precondition still saves
    its supplied bundle with invocation_state=not_invoked, never claims inference occurred.
    """
    from .registry import Approach
    approach=Approach(**{k:v for k,v in approach_metadata.items() if k in Approach.__dataclass_fields__}) if approach_metadata else next(a for a in APPROACHES if a.approach_id==approach_id)
    universe=list(source['investable_universe'])
    # Workflow supplies the checksum of its already persisted/verified shared artifact.
    # Direct callers compute it once; neither path mutates the original source.
    shared_checksum=shared_source_checksum if shared_source_checksum is not None else sha256_checksum(canonical_json_bytes(source))
    if not isinstance(shared_checksum,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',shared_checksum):raise ValueError('invalid verified shared-source checksum')
    base={'schema_version':'finplan-model-input/1','approach_id':approach_id,'run_id':run_id,'input_id':input_id,
          'decision_time':cutoff,'availability_cutoff':cutoff,'model':{**approach.to_dict(),'code_versions':source.get('code_versions',{})},'code_versions':source.get('code_versions',{}),
          'trigger':source.get('trigger','manual_validation'),'holdings':copy.deepcopy(holdings),'constraints':copy.deepcopy((config or {}).get('constraints',{})),
          'shared_source_checksum':shared_checksum,
          'news':{'consumed':False,'layer':'not_consumed','items':[],'reason':'News not consumed'},
          'numeric':{'frequency':'completed_daily','price_history_consumed':False,
              'source_history_reference':{'shared_source_checksum':shared_checksum,
                   'layer':'shared_source_only; exact consumed solver arrays or actor price window are in prepared'}}}
    if approach_metadata and approach_metadata.get('native_input_provenance') is not None:
        # Selection/acquisition may follow the rounded scheduled slot. This is
        # audit context, not an extra actor feature or model availability time;
        # the original news lineage and frozen vector remain in prepared inputs.
        base['model']['adapter_provenance']={'consumed':False,
            'role':'Source selection and artifact acquisition metadata; not policy observation values',
            'native_input':copy.deepcopy(approach_metadata['native_input_provenance'])}
    if approach.adapter in ('codex','claude_offline','langgraph','agent_tool','crypto_native') or approach_id=='cash':
        base['numeric'].pop('price_history',None)
        base['numeric']['price_history_consumed']=False
        base['collected_not_consumed']={'price_history':{'shared_source_checksum':base['shared_source_checksum'],
             'status':'not_supplied_to_this_approach','reason':'DS-14 protected raw series; exact derived/provider request is in prepared' if approach.adapter in ('codex','claude_offline') else 'Actual graph sub-boundaries are captured separately' if approach.adapter=='langgraph' else 'No daily-price history supplied; actual native hourly inputs are in prepared' if approach.adapter=='crypto_native' else 'Constant cash control consumes no market history'}}
    if approach.adapter=='crypto_native':
        base['numeric']['frequency']='hourly_crypto_24x7'
    captured=False
    capture_failed=False
    def before(prepared):
        nonlocal captured, capture_failed
        if captured:raise ValueError('multiple boundary captures in one decision')
        payload=copy.deepcopy(base)
        payload['numeric']['prepared']=copy.deepcopy(prepared)
        if approach.adapter=='crypto_native':payload['numeric']['price_history_consumed']=bool(prepared.get('price_history_consumed',False))
        payload['invocation_state']=prepared.get('invocation_state') or ('not_invoked' if str(prepared.get('boundary','')).startswith('not_invoked') else 'invoking')
        if prepared.get('news_evidence',{}).get('consumed'):
            payload['news']={'consumed':True,'layer':approach.news_layer,'items':copy.deepcopy(prepared.get('news_dataset',news_dataset)),
                             'prepared_context':copy.deepcopy(prepared['news_evidence']),
                             'metadata_only':True,'text_status':'No article text supplied to policy'}
        elif approach.adapter in ('codex','claude_offline'):
            invoked=payload['invocation_state']!='not_invoked'
            payload['news']={'consumed':invoked,'layer':'selected_headlines' if invoked else 'prepared_not_supplied',
                             'items':copy.deepcopy(prepared.get('actual_source',source).get('news',[])) if invoked else [],
                             'prepared_context':copy.deepcopy(prepared.get('messages',prepared)), 'metadata_only':False,
                             'context_status':'supplied_at_invocation' if invoked else 'prepared_not_supplied',
                             'reason':'Exact selected public news supplied' if invoked else 'Reviewer not invoked; prepared news context was not consumed'}
        try:
            capture(payload)
        except Exception:
            capture_failed=True
            raise
        captured=True
    start=time.monotonic();cpu_start=time.process_time()
    def compute_usage():return {'process_cpu_seconds':time.process_time()-cpu_start,'execution_host':'authorized_existing_EC2',
        'compute_cost_usd':None,'cost_basis':'existing EC2 allocation; no additional SageMaker/GPU/vendor job launched'}
    try:
        if approach_id=='cash':
            before({'boundary':'constant_cash_control','parameters':{},'price_history_consumed':False,'holdings_consumed':False})
            output={'weights':{i:0. for i in universe},'cash':1.,'solution_status':'not_applicable','forecast':{'available':False,'status':'not_available'}}
        elif approach.adapter in ('baseline','classical'):
            view,hv=book_holdings_view(source,holdings,cutoff,universe)
            quotes=dict(hv.prices)
            simulation=SimulationConfig.from_dict(config or {})
            if approach.adapter=='baseline':strategy=build_strategy(approach.strategy_name or approach_id,approach.parameters or {},constraints=simulation.constraints)
            else:
                from finplan_model.classical.strategy import ClassicalStrategy
                strategy=ClassicalStrategy(approach.strategy_name or approach_id.removeprefix('classical_'),**(approach.parameters or {}))
            strategy.input_capture=before
            if approach.family=='control':before({'boundary':'control_decide','parameters':dict(strategy.params),'eligible_prices':quotes})
            result=coerce_target(strategy.decide(view,hv),tuple(universe))
            if not captured:before({'boundary':'not_invoked_insufficient_history','parameters':dict(getattr(strategy,'params',{})),'diagnostics':result.diagnostics})
            output={'weights':dict(result.weights),'cash':result.cash,'solution_status':result.solution_status,'diagnostics':result.diagnostics,
                    'forecast':{'available':False,'status':'not_available','reason':'allocation/solver diagnostic is not a portfolio value forecast'}}
        elif approach.adapter=='classical_news':
            from finplan_model.etf_news import extract,FINANCIAL_NEWS_UNIVERSE
            from finplan_model.classical.math import settings,solve
            from finplan_model.strategies.estimators import pit_returns,estimate_covariance
            dataset=source.get('paid_news_dataset')
            if not dataset or set(universe)-FINANCIAL_NEWS_UNIVERSE:
                raise FinplanError.precondition('eligible native ETF search metadata missing for cohort',reason='licensed_news_source_or_universe_unavailable')
            features,provenance,context=extract(dataset,symbols=universe,decision_time=cutoff)
            if any(features[i]['missing'] for i in universe):
                before({'boundary':'not_invoked_news_coverage_missing','news_features':features,'news_provenance':provenance,'news_dataset':dataset})
                raise FinplanError.precondition('actual eligible licensed metadata does not cover all cohort symbols',reason='news_coverage_missing')
            market=input_market(source,cutoff)
            view=PointInTimeView(market,max(market.sessions),datetime.fromisoformat(cutoff.replace('Z','+00:00')),tuple(universe))
            returns=pit_returns(view,60)
            if len(returns)<20:raise FinplanError.precondition('insufficient native optimizer history',reason='insufficient_history')
            cfg=settings({'lookback_days':min(60,len(returns)),'cash_weight':.1,'max_weight':1. if len(universe)==1 else .6,'horizon_sessions':1})
            sentiment=np.array([features[i]['sentiment'] for i in universe]);risk=1+.25*np.array([features[i]['uncertainty'] for i in universe])
            current={r['instrument_id']:r['weight'] for r in holdings['weights']}
            inputs={'algorithm':approach.strategy_name,'instruments':universe,'settings':cfg,'current_weights':[current.get(i,0.) for i in universe],
                    'expected_returns':(returns.mean(axis=0)+.0005*sentiment).tolist(),
                    'covariance':(estimate_covariance(returns,'ledoit_wolf')*np.outer(risk,risk)).tolist(),'scenarios':(returns*risk).tolist()}
            before({**inputs,'boundary':'ETF_news_enriched_classical_solve','news_dataset':dataset,
                    'news_evidence':{'consumed':True,'features':features,'provenance':provenance,'eligible_metadata_context':context,
                     'transformation':'etf-search-metadata/1; mean += .0005*sentiment; covariance/scenarios scaled by 1+.25*uncertainty',
                     'calibration':'hypothetical_sensitivity_not_calibrated_forecast'}})
            result=solve(inputs)
            output={'weights':dict(zip(universe,result['weights'])) if result.get('weights') else current,
                    'cash':result.get('cash_weight',holdings['cash_weight']),'solution_status':'optimal' if result['status']=='optimal' else 'no_effect',
                    'diagnostics':result,'forecast':{'available':False,'status':'not_available'}}
        elif approach.adapter=='actor':
            if policy is None:raise FinplanError.precondition('verified frozen policy bundle not configured',reason='frozen_policy_missing')
            from finplan_model.rl.inference import recommend
            from finplan_model.rl.spec import EnvSpec
            spec=EnvSpec.from_dict(policy['environment'])
            if list(policy['instruments'])!=universe:raise FinplanError.precondition('policy fixed universe differs from comparison cohort',reason='incomparable_universe')
            if approach.news_layer=='not_consumed' and any('news' in f or 'gdelt' in f for f in spec.features):raise ValueError('price-only variant cannot consume news policy')
            if approach.news_layer!='not_consumed' and approach.news_layer not in spec.features:raise ValueError('policy feature variant mismatch')
            if any(m.get('actor',{}).get('algorithm','ppo')!=approach_id.split('_')[0] for m in policy.get('members',[])):raise ValueError('actor algorithm identity mismatch')
            actor_cutoff=native_decision_time or cutoff
            market=input_market(source,actor_cutoff)
            view=PointInTimeView(market,max(market.sessions),datetime.fromisoformat(actor_cutoff.replace('Z','+00:00')),tuple(universe))
            _,prices=view.price_matrix(lookback=spec.window+1)
            output=recommend(policy,prices,holdings,news_dataset=news_dataset,decision_time=actor_cutoff,input_capture=before)
        else:
            adapter=(external_adapters or {}).get(approach.adapter)
            if adapter is None:raise FinplanError.precondition('required invocation adapter unavailable',reason=approach.requires or 'adapter_unavailable')
            output=adapter(base,source,holdings,before)
            if not captured:raise ValueError('adapter did not capture actual model boundary input')
        if output.get('status') in ('failed','unavailable','skipped'):
            return {**output,'approach_id':approach_id,'run_id':run_id,'input_id':input_id,'latency_seconds':time.monotonic()-start,'compute_usage':compute_usage(),'forecast':{'available':False,'status':'not_available'}}
        normalized=validate_policy(output,universe,holdings)
        if normalized.get('status')=='hold':
            normalized['solution_status']='no_effect'
            normalized['intents']=[{**intent,'proposed_delta_weight':intent['delta_weight'],'delta_weight':0.,'action':'hold','reason':'explicit_hold_no_rebalance'} for intent in normalized['intents']]
        return {**normalized,'status':'hold' if normalized.get('solution_status') in ('no_effect','infeasible','unbounded','rejected_hold_current') or normalized.get('status')=='hold' else ('reused' if normalized.get('status')=='reused' else 'recommended'),
                'approach_id':approach_id,'run_id':run_id,'input_id':input_id,'latency_seconds':time.monotonic()-start,'compute_usage':compute_usage()}
    except Exception as exc:
        if capture_failed:
            raise
        if not captured:
            payload=copy.deepcopy(base);payload['invocation_state']='not_invoked';payload['precondition_error']=str(exc)
            capture(payload)
        return {'status':'unavailable' if isinstance(exc,FinplanError) and exc.code=='PRECONDITION_FAILED' else 'failed',
                'approach_id':approach_id,'run_id':run_id,'input_id':input_id,'reason':getattr(exc,'details',{}).get('reason',str(exc)),
                'error_type':type(exc).__name__,'public_final_response':getattr(exc,'public_final_response',None),'reviewer_invocation':getattr(exc,'invocation',None),'latency_seconds':time.monotonic()-start,'compute_usage':compute_usage(),'forecast':{'available':False,'status':'not_available'}}
